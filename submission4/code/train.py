"""Full fine-tuning on official train; official valid AUROC alone selects epochs."""

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import FundusDataset, ROOT, read_records
from model import NAME, PRETRAIN_PATH, create_model, sha256

RECIPE = "convnext_tiny_in1k_fov512_fullfinetune_bce_pw5_bf16_v1"


def fingerprint(records):
    digest = hashlib.sha256()
    for name, _, label in records:
        digest.update(f"{name}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def checkpoint_load(path):
    path = Path(path)
    with path.open("rb") as file:
        if file.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("Checkpoint is a Git LFS pointer; run git lfs pull")
    return torch.load(path, map_location="cpu", weights_only=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def learning_rate(epoch, total_epochs, peak):
    # One-epoch warmup, then cosine decay; fixed across resume.
    if epoch == 1:
        return peak * 0.3
    fraction = (epoch - 2) / max(1, total_epochs - 2)
    return 1e-6 + 0.5 * (peak - 1e-6) * (1 + math.cos(math.pi * fraction))


def evaluate(model, records, batch_size, workers):
    loader = DataLoader(FundusDataset(records), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    labels = [label for _, _, label in records]
    scores = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            images = images.cuda(non_blocking=True)
            scores.extend(torch.sigmoid(model(images).flatten()).float().cpu().tolist())
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise RuntimeError("Incomplete or nonfinite official-valid predictions")
    return float(roc_auc_score(labels, scores))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", type=Path, help="Existing run directory; resumes last complete epoch")
    return parser.parse_args()


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training needs an available BF16 CUDA GPU")
    if (args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1 or
            args.lr <= 0 or args.weight_decay < 0 or args.workers < 0 or args.patience < 1):
        raise ValueError("Invalid training configuration")
    train_records, valid_records = read_records("train"), read_records("valid")
    train_names = {name.casefold() for name, _, _ in train_records}
    if train_names.intersection(name.casefold() for name, _, _ in valid_records):
        raise ValueError("Official train and valid overlap by filename")
    config = {"recipe": RECIPE, "epochs": args.epochs, "batch_size": args.batch_size,
              "accumulation": args.accumulation, "lr": args.lr,
              "weight_decay": args.weight_decay, "train_sha256": fingerprint(train_records),
              "valid_sha256": fingerprint(valid_records)}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if args.resume:
        run_dir = args.resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != (ROOT / "runs").resolve():
            raise ValueError("--resume must point to an existing submission4/runs child")
        previous = checkpoint_load(run_dir / "last.pth")
        if previous.get("config") != config:
            raise ValueError("Resume configuration/data changed; create a new run")
        model = create_model(pretrained=False)
        model.load_state_dict(previous["model"], strict=True)
        start_epoch = int(previous["epoch"]) + 1
        best_auc, best_epoch = float(previous["best_auc"]), int(previous["best_epoch"])
        stale = int(previous["stale"])
        origin = previous["pretraining"]
    else:
        run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        model = create_model(pretrained=True)
        start_epoch, best_auc, best_epoch, stale = 1, -1.0, 0, 0
        origin = {"model": NAME, "source": "ImageNet-1K", "path": str(PRETRAIN_PATH),
                  "sha256": sha256(PRETRAIN_PATH)}
        (run_dir / "config.json").write_text(json.dumps({**config, "pretraining": origin},
                                                  indent=2), encoding="utf-8")
    model = model.cuda()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full fine-tuning requires every model parameter trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
    print(f"run={run_dir} epochs={start_epoch}..{args.epochs} train={len(train_records)} "
          f"valid={len(valid_records)} all_parameters_trainable=True", flush=True)
    train_set = FundusDataset(train_records, training=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience and epoch > 3:
            print(f"Early stop: {stale} epochs without valid AUROC improvement", flush=True)
            break
        seed = 42 + epoch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            generator=torch.Generator().manual_seed(seed))
        lr = learning_rate(epoch, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_index, (images, labels) in enumerate(loader, 1):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images).flatten()
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), labels.float(), pos_weight=logits.new_tensor(5.0).float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {batch_index}")
            (loss / args.accumulation).backward()
            running_loss += loss.item()
            if batch_index % args.accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index == 1 or batch_index % 500 == 0 or batch_index == len(loader):
                print(f"epoch={epoch} batch={batch_index}/{len(loader)} loss={loss.item():.5f}",
                      flush=True)
        auc = evaluate(model, valid_records, max(args.batch_size, 4), args.workers)
        improved = auc > best_auc
        if improved:
            best_auc, best_epoch, stale = auc, epoch, 0
            candidate = {"recipe": RECIPE, "model_name": NAME,
                         "model": {key: value.detach().cpu().clone()
                                   for key, value in model.state_dict().items()},
                         "epoch": epoch, "official_valid_auroc": auc,
                         "metric_scope": "official_valid", "training_complete": True,
                         "pretraining": origin, "config": config}
            atomic_save(candidate, run_dir / "candidate_best.pth")
        else:
            stale += 1
        state = {"recipe": RECIPE, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "best_auc": best_auc, "best_epoch": best_epoch,
                 "stale": stale, "pretraining": origin, "config": config}
        atomic_save(state, run_dir / "last.pth")
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": epoch, "train_loss": running_loss / len(loader),
                                   "official_valid_auroc": auc, "best_epoch": best_epoch,
                                   "lr": lr}, ensure_ascii=False) + "\n")
        print(f"epoch={epoch} official_valid_auroc={auc:.9f} best={best_auc:.9f} "
              f"best_epoch={best_epoch}", flush=True)
    print(f"Candidate: {run_dir / 'candidate_best.pth'} "
          f"official_valid_auroc={best_auc:.9f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
