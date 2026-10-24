"""Continue the latest upstream competition checkpoint; select by official validation AUROC."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import ImageModel, MODEL_NAMES
from prepare import CACHE_DIR, prepare
from utils import PROJECT_DIR, VARIANT_DIR, MODEL_PATH, RECIPE, calculate_metrics, create_transform, save_model, load_model, get_learning_rate, model_memory_format

class CachedDataset(Dataset):
    def __init__(self, split, training=False, excluded=()):
        self.path = str(CACHE_DIR / f"{split}.npy")
        self.array = None
        info = json.loads((CACHE_DIR / f"{split}.json").read_text())
        self.labels = np.asarray(info["labels"], dtype=np.int64)
        excluded = set(excluded)
        self.indices = np.asarray([i for i in range(len(self.labels)) if i not in excluded])
        self.transform = create_transform(training)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        j = self.indices[i]
        return self.transform(Image.fromarray(self.array[j])), int(self.labels[j])

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

def loader(dataset, batch, workers, shuffle=False, seed=20261024):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)

@torch.inference_mode()
def validate(model, batches, device):
    model.eval()
    labels, probabilities = [], []
    for x, y in batches:
        x = x.to(device, non_blocking=True, memory_format=model_memory_format(model.model_config["model_name"]))
        probabilities.extend(torch.softmax(model(x), 1)[:, 1].cpu().tolist())
        labels.extend(y.tolist())
    return calculate_metrics(labels, probabilities), probabilities

def atomic_save(obj, path):
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)

def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU run requires CUDA; refusing accidental long CPU training")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = prepare()
    train_ds = CachedDataset("train", True, report["train_indices_excluded_exact_validation_duplicates"])
    valid_ds = CachedDataset("valid")
    run_dir = PROJECT_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "config.json"
    config = {**vars(args), "recipe": RECIPE, "torch": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip(),
              "train_count": len(train_ds), "valid_count": len(valid_ds),
              "validation_use": "checkpoint selection only", "test_use": "final inference only",
              "initialization": "upstream main v1.12.1 competition-trained best_model.pth; no external pretrained weights",
              "training_precision": "AMP float16", "inference_precision": "FP32",
              "ensemble": "fixed arithmetic mean, threshold=0.5; no TTA"}
    config["source_files_sha256"] = {str(p.relative_to(PROJECT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((VARIANT_DIR / "code").glob("*.py"))}
    config["initialization_sha256"] = "def7e5c5be79903cd863421156f0557786d03bc567722ffd083360c18418d1a6"
    config["memory_layout"] = {"resnet18": "channels_last", "densenet121": "contiguous NCHW"}
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ("recipe", "epochs", "seed", "train_count", "valid_count", "batch_size"):
            if previous[key] != config[key]:
                raise ValueError(f"Cannot resume changed configuration: {key}")
    manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    valid_loader = loader(valid_ds, 64, args.workers)
    for name in MODEL_NAMES:
        start_epoch, best_auroc, history = 0, -1.0, []
        last_path = run_dir / f"{name}_last.pth"
        if last_path.exists() and not args.resume:
            raise FileExistsError("Use --resume or a new --run-name to preserve prior training")
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Upstream trained checkpoint is required; restore Git LFS weights")
        model = load_model(MODEL_PATH, name, for_training=True)
        model.to(device=device, memory_format=model_memory_format(name))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
        if args.resume and last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best_auroc, history = state["epoch"], state["best_auroc"], state["history"]
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
        else:
            baseline, baseline_probabilities = validate(model, valid_loader, device)
            best_auroc = baseline["auroc"]
            history.append({"epoch": 0, "source": "upstream_checkpoint_revalidated_at_256", **baseline})
            save_model(model, MODEL_PATH, baseline, 0)
            np.save(run_dir / f"{name}_valid_probabilities.npy", baseline_probabilities)
            np.save(run_dir / f"{name}_upstream_baseline_probabilities.npy", baseline_probabilities)
            print("BASELINE_RESULT " + json.dumps({"model": name, **baseline}), flush=True)
        labels = train_ds.labels[train_ds.indices]
        positives = int(labels.sum())
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., math.sqrt((len(labels)-positives)/positives)], device=device))
        print(f"TRAIN {name} epochs={args.epochs} start={start_epoch} samples={len(train_ds)} GPU={config['gpu']}", flush=True)
        for epoch in range(start_epoch, args.epochs):
            lr = get_learning_rate(epoch+1) if args.epochs == 5 else 3e-5
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_loader = loader(train_ds, args.batch_size, args.workers, True, args.seed + epoch)
            model.train()
            total_loss, count, started = 0., 0, time.perf_counter()
            for step, (x, y) in enumerate(train_loader, 1):
                x = x.to(device, non_blocking=True, memory_format=model_memory_format(name))
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(y)
                count += len(y)
                if step == 1 or step % 100 == 0 or step == len(train_loader):
                    elapsed = time.perf_counter()-started
                    print(f"{name} epoch={epoch+1}/{args.epochs} batch={step}/{len(train_loader)} loss={total_loss/count:.5f} elapsed={elapsed:.1f}s ETA={(len(train_loader)-step)*elapsed/step:.1f}s", flush=True)
            del train_loader
            metrics, probabilities = validate(model, valid_loader, device)
            if count != len(train_ds):
                raise RuntimeError("Incomplete training epoch")
            entry = {"epoch": epoch+1, "train_samples": count, "train_loss": total_loss/count, "learning_rate": lr,
                     "epoch_seconds": time.perf_counter()-started, **metrics}
            history.append(entry)
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                save_model(model, MODEL_PATH, metrics, epoch+1)
                np.save(run_dir / f"{name}_valid_probabilities.npy", probabilities)
            pd.DataFrame(history).to_csv(run_dir / f"{name}_history.csv", index=False)
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": epoch+1, "best_auroc": best_auroc,
                         "history": history, "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                         "numpy_rng": np.random.get_state()}, last_path)
            print("EPOCH_RESULT " + json.dumps({"model": name, **entry, "best_auroc": best_auroc}), flush=True)
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    all_probabilities, summary = [], {}
    for name in MODEL_NAMES:
        model = load_model(MODEL_PATH, name).to(device=device, memory_format=model_memory_format(name))
        metrics, probabilities = validate(model, valid_loader, device)
        summary[name] = metrics
        all_probabilities.append(probabilities)
        del model
    probabilities = np.mean(all_probabilities, axis=0)
    summary["fixed_mean_ensemble"] = calculate_metrics(valid_ds.labels, probabilities)
    baseline_probabilities = np.mean([np.load(run_dir / f"{n}_upstream_baseline_probabilities.npy") for n in MODEL_NAMES], axis=0)
    summary["upstream_fixed_mean_baseline"] = calculate_metrics(valid_ds.labels, baseline_probabilities)
    np.save(run_dir / "ensemble_valid_probabilities.npy", probabilities)
    (run_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    checkpoint["training_complete"] = True
    checkpoint["epochs_trained_each"] = args.epochs
    checkpoint["initialization_source_sha256"] = "def7e5c5be79903cd863421156f0557786d03bc567722ffd083360c18418d1a6"
    atomic_save(checkpoint, MODEL_PATH)
    (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps({"epochs_each": args.epochs, "metrics": summary}, indent=2), encoding="utf-8")
    print("TRAINING_COMPLETE " + json.dumps(summary), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--run-name", default="20261024_upstream_dual")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error("Invalid epochs, batch size or workers")
    train(args)
