"""Submission2 joint head training. Only untouched official valid selects weights."""
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MODEL_NAMES, create_model
from sampling import split_records
from utils import (BATCH_SIZE, BalancedBatchSampler, FundusDataset, NUM_WORKERS,
                   ROOT, classification_loss, metrics, read_records)

RECIPE = "official_train_joint_heads_rgb384_focal_pairauc_bf16_v3"
DAM_RECIPE = "official_train_joint_heads_rgb384_focal_marginauc_bf16_v2"
DEFAULT_PERCENT = 80
BASELINE = ROOT.parent / "submission" / "model" / "best_model.pth"


def load_checkpoint(path):
    if (Path(path).parent / "INVALID_RUN.txt").is_file():
        raise RuntimeError("该训练 run 已标记为无效，不能预测或续训")
    with Path(path).open("rb") as f:
        if f.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("权重文件是 Git LFS 指针")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        data = torch.load(path, map_location="cpu", weights_only=True)
    if not data.get("training_complete") or set(data.get("states", {})) != set(MODEL_NAMES):
        raise ValueError("初始双骨干权重不完整")
    return data


def make_models(checkpoint, device):
    models, parameters = [], []
    for name in MODEL_NAMES:
        model, _ = create_model(name)
        model.load_state_dict(checkpoint["states"][name], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)
        model.to(device).eval()
        models.append(model)
        parameters.extend(model.get_classifier().parameters())
    return models, parameters


def fused_logits(models, images):
    a, b = (model(images).flatten() for model in models)
    probability = (torch.sigmoid(a.float()) + torch.sigmoid(b.float())) / 2
    return torch.logit(probability.clamp(1e-6, 1 - 1e-6))


def objective(logits, labels, method):
    focal = classification_loss(logits, labels, 3.0, 0.0)
    if method == "pairauc":
        return classification_loss(logits, labels, 3.0, 0.1)
    positive, negative = logits[labels >= 0.5], logits[labels < 0.5]
    margin = torch.relu(0.5 + negative[None, :] - positive[:, None]).square().mean()
    return focal + 0.1 * margin


def data_loader(records, training, seed):
    dataset = FundusDataset(records, training=training)
    options = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
    if training:
        sampler = BalancedBatchSampler([label for _, label in records], BATCH_SIZE, seed)
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **options)


def evaluate(models, records, device, title):
    scores, labels = [], []
    loader = data_loader(records, False, 2026)
    with torch.inference_mode():
        for i, (images, targets) in enumerate(loader, 1):
            logits = fused_logits(models, images.to(device, non_blocking=True))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.tolist())
            if i == 1 or i % 100 == 0 or i == len(loader):
                print(f"{title}: {i}/{len(loader)}", flush=True)
    result = metrics(labels, scores)
    print(title, json.dumps(result, ensure_ascii=False), flush=True)
    return result


def save(path, models, settings, auc, initial_auc, epoch, promoted):
    payload = dict(recipe=DAM_RECIPE if settings.method == "marginauc" else RECIPE,
                   states={name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                           for name, model in zip(MODEL_NAMES, models)},
                   selection=dict(name="ensemble", auroc=auc, metric_scope="official_valid"),
                   source_checkpoint=str(settings.init), method=settings.method,
                   mode=settings.mode, percent=settings.percent, initial_auc=initial_auc,
                   epoch=epoch, promoted=promoted, training_complete=True,
                   trainable="both_classifier_heads", inference="mean_sigmoid_fp32_no_tta")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    return path


def run(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("训练需要 CUDA GPU")
    if settings.epochs < 1 or not 0 < settings.percent < 100:
        raise ValueError("epochs >= 1 且 0 < percent < 100")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = load_checkpoint(settings.init)
    models, parameters = make_models(baseline, device)
    official_train = read_records("train")
    official_valid = read_records("valid")
    if settings.mode == "full":
        selected, diagnostic, split = official_train, [], dict(mode="full", count=len(official_train))
    else:
        selected, diagnostic, split = split_records(official_train, settings.percent,
                                                    settings.mode, settings.seed)
    if {str(path) for path, _ in selected} & {str(path) for path, _ in official_valid}:
        raise RuntimeError("官方 valid 混入训练")
    root = Path(settings.output_root).resolve()
    run_dir = (Path(settings.resume).resolve() if settings.resume else
               root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if not run_dir.is_relative_to((root / "runs").resolve()):
        raise ValueError("--resume 必须在当前 submission 的 runs 目录内")
    config = dict(recipe=DAM_RECIPE if settings.method == "marginauc" else RECIPE,
                  mode=settings.mode, percent=settings.percent, seed=settings.seed,
                  method=settings.method, source=str(Path(settings.init).resolve()),
                  lr=settings.lr, split=split,
                  train_count=len(selected), official_valid_count=len(official_valid))
    if settings.resume:
        saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in config.items() if key in saved):
            raise ValueError("--resume 配方、数据或抽样设置与原 run 不同")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("run:", run_dir, "GPU:", torch.cuda.get_device_name(0), "train:", len(selected), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=settings.lr, weight_decay=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress_path = run_dir / "progress.pth"
    if settings.resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress.get("recipe") != config["recipe"] or progress.get("split") != split:
            raise ValueError("续训进度的配方或划分不一致")
        for name, model in zip(MODEL_NAMES, models):
            model.load_state_dict(progress["states"][name], strict=True)
        optimizer.load_state_dict(progress["optimizer"])
        scaler.load_state_dict(progress["scaler"])
        initial = {"auroc": progress["initial_auc"]}
        best_auc = progress["best_auc"]
        first_epoch = progress["epoch"] + 1
        print("精确轮次边界续训，下一轮:", first_epoch, flush=True)
    elif settings.resume:
        raise FileNotFoundError("该旧 run 没有 progress.pth，无法精确续训")
    else:
        initial = evaluate(models, official_valid, device, "official valid baseline")
        best_auc = initial["auroc"]
        first_epoch = 1
    loader = data_loader(selected, True, settings.seed)
    for epoch in range(first_epoch, settings.epochs + 1):
        loader.batch_sampler.set_epoch(epoch)
        total, started = 0.0, time.time()
        for i, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = objective(fused_logits(models, images), targets, settings.method)
            if not torch.isfinite(loss):
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw = [model(images).flatten() for model in models]
                print("non-finite diagnostics:",
                      {"images_finite": bool(torch.isfinite(images).all()),
                       "outputs_finite": [bool(torch.isfinite(x).all()) for x in raw],
                       "heads_finite": [all(bool(torch.isfinite(p).all())
                                            for p in model.get_classifier().parameters())
                                        for model in models]}, flush=True)
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {i}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            if i == 1 or i % 100 == 0 or i == len(loader):
                print(f"epoch:{epoch} batch:{i}/{len(loader)} loss:{total/i:.5f} elapsed:{time.time()-started:.0f}s", flush=True)
        if diagnostic:
            print("internal diagnostic count:", len(diagnostic),
                  "not evaluated: warm-start model previously saw official train", flush=True)
        valid = evaluate(models, official_valid, device, f"official valid epoch {epoch}")
        improved = valid["auroc"] > best_auc
        candidate = save(run_dir / f"candidate_epoch{epoch}.pth", models, settings,
                         valid["auroc"], initial["auroc"], epoch, improved)
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(epoch=epoch, loss=total/len(loader), official_valid=valid,
                                    improved=improved), ensure_ascii=False) + "\n")
        print("candidate:", candidate, "AUROC improved:", improved, flush=True)
        if improved:
            best_auc = valid["auroc"]
            destination = root / "model" / "best_model.pth"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            print("promoted:", destination, flush=True)
        progress = dict(recipe=config["recipe"], split=split, epoch=epoch,
                        initial_auc=initial["auroc"], best_auc=best_auc,
                        states={name: {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()}
                                for name, model in zip(MODEL_NAMES, models)},
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        temp = progress_path.with_suffix(".tmp")
        torch.save(progress, temp)
        os.replace(temp, progress_path)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("holdout", "independent", "full"), default="holdout")
    parser.add_argument("--percent", type=int, default=DEFAULT_PERCENT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--method", choices=("pairauc", "marginauc"), default="pairauc")
    parser.add_argument("--init", type=Path, default=BASELINE)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--resume", type=Path, help="从同一 runs/<run> 上一完整轮继续")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
