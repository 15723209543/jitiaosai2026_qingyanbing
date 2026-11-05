"""Independent Tiny320 RandAugment N1/M4 experiment; official valid is held out."""

import argparse
import copy
import hashlib
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from model_tiny import NAME, PRETRAIN_PATH, create_model
from utils import (DATA_ROOT, ROOT, FundusDataset, atomic_save,
                   calculate_metrics, fingerprint, load_checkpoint, read_records)

RECIPE = "s23_g03_tiny320_randaugment_n1m4_low_lr_from_epoch7_v1"
SOURCE_PTH = ROOT.parent / "runs" / "submission_tiny_randaugment" / "20261104_205005" / "best.pth"
SOURCE_SHA256 = "6686f5e5ae0042c8e4d7334206307d12a7111ce7c9b7d6d1f0d53d93b8890e30"
RECIPE_SFA = "s23_g03_tiny320_erasing_sfa_official_valid_v1"
SEED = 20261024
FIT_COUNT = 41116
EMA_DECAY = 0.996006


class RandAugmentFundusDataset(FundusDataset):
    """Adds a fixed, mild NeurIPS 2020 RandAugment PIL operation only for training."""
    def __init__(self, records, training=False):
        super().__init__(records, training=training)
        if training:
            self.transform = transforms.Compose(
                [transforms.RandAugment(num_ops=1, magnitude=4)] +
                list(self.transform.transforms))


def split_train(records):
    """【报告 S23-g03 思路，自写实现】只在官方 train 内分层取 41,116 张；valid 不入训练。"""
    if len(records) < FIT_COUNT:
        raise ValueError(f"官方 train 数量 {len(records)} 小于复现配方要求 {FIT_COUNT}")
    indices, _ = train_test_split(np.arange(len(records)), train_size=FIT_COUNT,
                                  stratify=[record[2] for record in records], random_state=SEED)
    return [records[int(index)] for index in sorted(indices)]


def update_ema(ema, model):
    """【参考报告 S23-g03，自写】每次真实优化器更新后用 0.996006 指数滑动平均。"""
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.lerp_(source, 1 - EMA_DECAY)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


def score_batch(images, model, device):
    """【自写】单视图 FP32 推理；验证、预测和耗时脚本共用。"""
    with torch.inference_mode():
        logits, _ = model(images.to(device, non_blocking=True))
        return torch.sigmoid(logits).float().cpu()


def evaluate(model, records, batch_size, workers, device):
    """【复用并修改】submission4/code/train.py:evaluate；官方 valid 全量 AUROC。"""
    loader = DataLoader(FundusDataset(records), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model.eval()
    labels, scores = [], []
    for index, (images, targets) in enumerate(loader, 1):
        probabilities = score_batch(images, model, device)
        labels.extend(targets.int().tolist())
        scores.extend(probabilities.tolist())
        if index == 1 or index % 25 == 0 or index == len(loader):
            print(f"valid, batch: {index}/{len(loader)}", flush=True)
    return calculate_metrics(labels, scores)


def learning_rate(step, steps_per_epoch):
    """Independent low-LR recipe: epoch 1 warmup, 8-epoch cosine to 1e-6."""
    warmup = steps_per_epoch
    total = 8 * steps_per_epoch
    if step < warmup:
        return 1e-5 * (0.1 + 0.9 * (step + 1) / warmup)
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1e-6 + (1e-5 - 1e-6) * (1 + math.cos(math.pi * fraction)) / 2


def train_epoch(model, ema, loader, optimizer, scaler, epoch, steps_per_epoch,
                first_step, pos_weight, batch_accumulation, device, amp_dtype):
    """【报告 S23-g03 思路，自写】BCE+0.2辅助头+渐入0.05双视图一致性，梯度累积。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, steps = 0.0, 0
    for index, (first, second, targets) in enumerate(loader, 1):
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, auxiliary = model(first)
            second_logits, _ = model(second)
            main = F.binary_cross_entropy_with_logits(logits.float(), targets,
                                                       pos_weight=pos_weight)
            aux = F.binary_cross_entropy_with_logits(auxiliary.float(), targets,
                                                      pos_weight=pos_weight)
            consistency_weight = 0.05 * min(1.0, epoch / 5)
            consistency = F.mse_loss(torch.sigmoid(logits.float()),
                                     torch.sigmoid(second_logits.float()))
            loss = main + 0.2 * aux + consistency_weight * consistency
        scaler.scale(loss / batch_accumulation).backward()
        total_loss += float(loss.detach())
        if index % batch_accumulation == 0 or index == len(loader):
            rate = learning_rate(first_step + steps, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model)
            steps += 1
        if index == 1 or index % 25 == 0 or index == len(loader):
            predictions = torch.sigmoid(logits.detach().float()) >= 0.5
            positives = targets >= 0.5
            print(f"train epoch:{epoch}, batch:{index}/{len(loader)}, loss:{float(loss):.5f}, "
                  f"真实阳性:{int(positives.sum())}, 报阳:{int(predictions.sum())}, "
                  f"误报:{int((predictions & ~positives).sum())}", flush=True)
    return total_loss / len(loader), steps


def checkpoint(model, ema, optimizer, scaler, epoch, step, best, config, patience):
    """【自写】保存真实模型/优化器状态、训练指纹；用于中断后按完整轮续训。"""
    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "step": step, "best_auroc": best, "config": config,
            "patience": patience}


def run(args):
    """【报告 S23-g03 配方，自写执行】GPU 完整训练；仅官方 valid AUROC 严格升高时更新单份最佳 PTH。"""
    print(f"设备：{'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}; "
          f"PyTorch:{torch.__version__}; CUDA:{torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("本配方只在 GPU 训练；当前机器不执行完整训练")
    if (args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1 or
            args.workers < 0 or args.feature_noise_std not in {0.0, 0.05}):
        raise ValueError("训练参数必须为合法正整数")
    if args.feature_noise_std != 0.0:
        raise ValueError("This RandAugment recipe forbids feature noise")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    official_train, valid = read_records("train"), read_records("valid")
    if {x[0].casefold() for x in official_train} & {x[0].casefold() for x in valid}:
        raise ValueError("官方训练集和验证集文件名重合")
    fit = split_train(official_train)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"官方训练:{len(official_train)}，本配方实际训练:{len(fit)}，官方验证:{len(valid)}；"
          f"AMP:{amp_dtype}，batch:{args.batch_size}×{args.accumulation}", flush=True)
    recipe = RECIPE_SFA if args.feature_noise_std > 0 else RECIPE
    config = {"recipe": recipe, "feature_noise_std": args.feature_noise_std, "randaugment_ops": 1, "randaugment_magnitude": 4, "seed": SEED, "model_name": NAME, "size": 320,
              "train_sha256": fingerprint(fit), "valid_sha256": fingerprint(valid),
              "batch_size": args.batch_size, "accumulation": args.accumulation,
              "epochs": args.epochs, "workers": args.workers,
              "source_sha256": SOURCE_SHA256, "initial_lr": 1e-5,
              "weight_decay": 5e-4, "schedule_epochs": 8}
    run_root = ROOT.parent / "runs" / "submission_tiny_randaugment_low_lr"
    if args.resume:
        run_dir = args.resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != run_root.resolve():
            raise ValueError(f"--resume 必须是本项目 {run_root} 下同配方原有 run")
        previous = load_checkpoint(run_dir / "last.pth")
        # The fixed 40-epoch schedule is unchanged; --epochs only extends the run target.
        recorded = {k: v for k, v in previous["config"].items()
                    if k not in {"pretrain_sha256", "epochs"}}
        current = {k: v for k, v in config.items() if k != "epochs"}
        if recorded != current:
            raise ValueError("恢复时配方或数据指纹变化，不能续训")
        if args.epochs <= previous["epoch"]:
            raise ValueError("目标轮次必须大于断点已完成轮次")
        model = create_model(pretrained=False, feature_noise_std=args.feature_noise_std).cuda()
        model.load_state_dict(previous["model"], strict=True)
        start_epoch, step, best = previous["epoch"] + 1, previous["step"], previous["best_auroc"]
    else:
        run_dir = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True)
        if hashlib.sha256(SOURCE_PTH.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("Warmstart source SHA256 changed")
        source = load_checkpoint(SOURCE_PTH)
        if (source.get("metric_scope") != "official_valid" or
                source.get("selected_epoch") != 7 or
                source.get("recipe") != "s23_g03_tiny320_randaugment_n1m4_official_valid_v1"):
            raise ValueError("Warmstart source recipe/selection changed")
        model = create_model(pretrained=False, feature_noise_std=args.feature_noise_std).cuda()
        model.load_state_dict(source["model"], strict=True)
        start_epoch, step, best = 1, 0, float(source["official_valid_auroc"])
        shutil.copy2(SOURCE_PTH, run_dir / "best.pth")
        print(f"Independent low-LR warmstart official valid AUROC:{best:.12f}", flush=True)
    digest_hash = hashlib.sha256()
    with PRETRAIN_PATH.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest_hash.update(block)
    digest = digest_hash.hexdigest()
    if args.resume and previous["config"].get("pretrain_sha256") != digest:
        raise ValueError("预训练权重 SHA256 改变，拒绝续训")
    config["pretrain_sha256"] = digest
    print(f"非医学 ImageNet-1K 权重 SHA256:{digest}; run:{run_dir}", flush=True)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        ema.load_state_dict(previous["ema"], strict=True)
        baseline = evaluate(ema, valid, args.batch_size, args.workers, "cuda")
        print(f"续训前官方验证 AUROC:{baseline['auroc']:.6f}; 已保存最佳:{best:.6f}", flush=True)
    loader = DataLoader(RandAugmentFundusDataset(fit, training=True), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True)
    steps_per_epoch = math.ceil(len(loader) / args.accumulation)
    positives = sum(row[2] for row in fit)
    pos_weight = torch.tensor(min(6.0, math.sqrt((len(fit) - positives) / positives)), device="cuda")
    patience = 0
    if args.resume:
        if "patience" in previous:
            patience = previous["patience"]
        else:
            selected = load_checkpoint(run_dir / "best.pth")
            if abs(selected["official_valid_auroc"] - best) > 1e-12:
                raise ValueError("断点与最佳权重的官方验证 AUROC 不一致")
            patience = previous["epoch"] - selected["selected_epoch"]
        if patience < 0:
            raise ValueError("断点早停计数非法")
    for epoch in range(start_epoch, args.epochs + 1):
        loss, updates = train_epoch(model, ema, loader, optimizer, scaler, epoch,
                                    steps_per_epoch, step, pos_weight,
                                    args.accumulation, "cuda", amp_dtype)
        step += updates
        result = evaluate(ema, valid, args.batch_size, args.workers, "cuda")
        print(f"epoch:{epoch}, loss:{loss:.5f}, valid AUROC:{result['auroc']:.6f}, "
              f"AP:{result['ap']:.6f}, 真实阳性:{result['positive']}, "
              f"报对阳性:{result['tp']}, 误报:{result['fp']}, 漏报:{result['fn']}", flush=True)
        if result["auroc"] > best:
            best, patience = result["auroc"], 0
            selected = {"recipe": recipe, "model_name": NAME, "model":
                        {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                        "config": config, "metric_scope": "official_valid",
                        "official_valid_auroc": best, "official_valid_ap": result["ap"],
                        "selected_epoch": epoch, "training_complete": True}
            atomic_save(selected, run_dir / "best.pth")
            # 本路线仅保存独立run候选，不能覆盖正式Small模型。
            print(f"Tiny run官方验证AUROC提高，候选保存在 {run_dir / 'best.pth'}", flush=True)
        else:
            patience += 1
        atomic_save(checkpoint(model, ema, optimizer, scaler, epoch, step, best, config, patience),
                    run_dir / "last.pth")
        if epoch >= 3 and patience >= 3:
            print(f"早停：第{epoch}轮，最佳官方验证 AUROC:{best:.6f}", flush=True)
            break
    print(f"训练结束；最佳官方验证 AUROC:{best:.6f}；未生成测试 CSV", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--feature-noise-std", type=float, default=0.0,
                        help="ICCV 2021 特征高斯扰动实验；0 为原 S23 g03 对照")
    parser.add_argument("--resume", type=Path)
    run(parser.parse_args())
