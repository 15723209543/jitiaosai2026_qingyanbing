"""S23-g03 Tiny320 近似复现实验：官方 train 内部分层留出，官方 valid AUROC 选最佳 EMA。"""

import argparse
import copy
import hashlib
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import NAME, PRETRAIN_PATH, TINY_NAME, TINY_PRETRAIN_PATH, create_model, create_tiny_model
from utils import (DATA_ROOT, ROOT, FundusDataset, atomic_save,
                   calculate_metrics, fingerprint, load_checkpoint, read_records)

SMALL_RECIPE = "small320_erasing_ema_aux_consistency_official_valid_v1"
TINY_RECIPE = "s23_g03_tiny320_erasing_official_valid_v3"
TINY_RECIPE_SFA = "s23_g03_tiny320_erasing_sfa_official_valid_v1"
SEED = 20261024
FIT_COUNT = 41116
EMA_DECAY = 0.996006


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
    """【参考报告 40 轮调度，自写】第 1 轮预热，此后按 40 轮余弦衰减。"""
    warmup = steps_per_epoch
    total = 40 * steps_per_epoch
    if step < warmup:
        return 1e-4 * (0.1 + 0.9 * (step + 1) / warmup)
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * fraction)) / 2


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
    if args.arch == "small":
        if args.feature_noise_std != 0:
            raise ValueError("Small recipe does not use feature noise")
        recipe, model_name, pretrain_path = SMALL_RECIPE, NAME, PRETRAIN_PATH
        model_factory = lambda pretrained: create_model(pretrained=pretrained)
        run_root = ROOT.parent / "runs" / "submission_new"
    else:
        recipe = TINY_RECIPE_SFA if args.feature_noise_std > 0 else TINY_RECIPE
        model_name, pretrain_path = TINY_NAME, TINY_PRETRAIN_PATH
        model_factory = lambda pretrained: create_tiny_model(pretrained=pretrained, feature_noise_std=args.feature_noise_std)
        run_root = ROOT.parent / "runs" / ("submission_tiny_sfa" if args.feature_noise_std > 0 else "submission_tiny")
    config = {"recipe": recipe, "seed": SEED, "model_name": model_name, "size": 320,
              "train_sha256": fingerprint(fit), "valid_sha256": fingerprint(valid),
              "batch_size": args.batch_size, "accumulation": args.accumulation,
              "epochs": args.epochs, "workers": args.workers}
    if args.arch == "tiny":
        config["feature_noise_std"] = args.feature_noise_std
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
        model = model_factory(False).cuda()
        model.load_state_dict(previous["model"], strict=True)
        start_epoch, step, best = previous["epoch"] + 1, previous["step"], previous["best_auroc"]
    else:
        run_dir = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True)
        model = model_factory(True).cuda()
        start_epoch, step, best = 1, 0, -1.0
    digest_hash = hashlib.sha256()
    with pretrain_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest_hash.update(block)
    digest = digest_hash.hexdigest()
    if args.resume and previous["config"].get("pretrain_sha256") != digest:
        raise ValueError("预训练权重 SHA256 改变，拒绝续训")
    config["pretrain_sha256"] = digest
    print(f"非医学 ImageNet-1K 权重 SHA256:{digest}; run:{run_dir}", flush=True)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        ema.load_state_dict(previous["ema"], strict=True)
        baseline = evaluate(ema, valid, args.batch_size, args.workers, "cuda")
        print(f"续训前官方验证 AUROC:{baseline['auroc']:.6f}; 已保存最佳:{best:.6f}", flush=True)
    loader = DataLoader(FundusDataset(fit, training=True), batch_size=args.batch_size,
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
            selected = {"recipe": recipe, "model_name": model_name, "model":
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
        if epoch >= 10 and patience >= 6:
            print(f"早停：第{epoch}轮，最佳官方验证 AUROC:{best:.6f}", flush=True)
            break
    print(f"训练结束；最佳官方验证 AUROC:{best:.6f}；未生成测试 CSV", flush=True)


if __name__ == "__main__" and "--pairauc" not in __import__("sys").argv:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("small", "tiny"), required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--feature-noise-std", type=float, default=0.0,
                        help="ICCV 2021 特征高斯扰动实验；0 为原 S23 g03 对照")
    parser.add_argument("--resume", type=Path)
    run(parser.parse_args())


# November 6 Small pairwise-AUC random-subset recipe; invoke with --pairauc.
"""Small320 warm-start with a new seeded official-train subset each epoch."""
import argparse
import copy
import hashlib
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import NAME, create_model
from utils import ROOT, FundusDataset, atomic_save, fingerprint, load_checkpoint, read_records

PAIR_AUC_RECIPE = "small320_pairauc15_random12000_selection70_v1"
PAIR_AUC_SEED = 20261109
PAIR_AUC_BASE = None  # Supply the verified warm-start with --warmstart.
PAIR_AUC_RUN_ROOT = ROOT / "runs" / "small_pair_auc_subset"
PAIR_AUC_SCHEDULE_EPOCHS = 12  # Fixed across --resume calls.
PAIR_AUC_SUBSET_SIZE = 12000


def pair_auc_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pair_auc_rate(step, updates_per_epoch):
    fraction = min(1.0, step / (PAIR_AUC_SCHEDULE_EPOCHS * updates_per_epoch))
    return 5e-7 + (2e-6 - 5e-7) * (1 + math.cos(math.pi * fraction)) / 2


def pair_auc_train_one(model, ema, records, optimizer, scaler, pos_weight, epoch, step,
              batch_size, accumulation, workers, amp_dtype):
    generator = torch.Generator().manual_seed(PAIR_AUC_SEED + epoch)
    positives = sum(r[2] >= 0.5 for r in records)
    negatives = len(records) - positives
    if not positives or not negatives:
        raise ValueError('Pairwise AUC needs both classes')
    weights = [0.15 / positives if r[2] >= 0.5 else 0.85 / negatives for r in records]
    sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True, generator=generator)
    loader = DataLoader(FundusDataset(records, training=True), batch_size=batch_size,
                        sampler=sampler, num_workers=workers, pin_memory=True)
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, updates = 0.0, 0
    for index, (first, second, target) in enumerate(loader, 1):
        first = first.cuda(non_blocking=True)
        second = second.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, auxiliary = model(first)
            second_logits, _ = model(second)
            main = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pos_weight)
            aux = F.binary_cross_entropy_with_logits(auxiliary.float(), target, pos_weight=pos_weight)
            consistency = F.mse_loss(torch.sigmoid(logits.float()), torch.sigmoid(second_logits.float()))
            positive = logits.float()[target >= 0.5]
            negative = logits.float()[target < 0.5]
            pairwise = (F.relu(0.5 + negative[None, :] - positive[:, None]).square().mean()
                        if positive.numel() and negative.numel() else logits.float().sum() * 0.0)
            loss = main + 0.2 * aux + 0.05 * consistency + 0.05 * pairwise
        scaler.scale(loss / accumulation).backward()
        loss_sum += float(loss.detach())
        if index % accumulation == 0 or index == len(loader):
            for group in optimizer.param_groups:
                group["lr"] = pair_auc_rate(step + updates, updates_per_epoch)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model)
            updates += 1
        if index % 100 == 0 or index == len(loader):
            print(f"epoch:{epoch} batch:{index}/{len(loader)} loss:{float(loss.detach()):.5f}", flush=True)
    return loss_sum / len(loader), updates


def pair_auc_run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if not (1 <= args.epochs <= PAIR_AUC_SCHEDULE_EPOCHS and args.batch_size > 0 and
            args.accumulation > 0 and args.workers >= 0):
        raise ValueError("Invalid training arguments")
    random.seed(PAIR_AUC_SEED)
    np.random.seed(PAIR_AUC_SEED)
    torch.manual_seed(PAIR_AUC_SEED)
    torch.cuda.manual_seed_all(PAIR_AUC_SEED)
    train, valid = read_records("train"), read_records("valid")
    assert len(train) == 51396 and len(valid) == 5849
    if {x[0].casefold() for x in train} & {x[0].casefold() for x in valid}:
        raise ValueError("Train and official valid overlap")
    select_ids, audit_ids = train_test_split(np.arange(len(valid)), test_size=0.30,
                                             stratify=[r[2] for r in valid], random_state=20261106)
    selection = [valid[int(i)] for i in sorted(select_ids)]
    audit = [valid[int(i)] for i in sorted(audit_ids)]
    if len(selection) + len(audit) != len(valid):
        raise ValueError("Invalid validation split")
    warmstart = Path(args.warmstart).resolve()
    base = load_checkpoint(warmstart)
    if base.get("metric_scope") != "official_valid_selection70" or base.get("training_complete") is not True:
        raise ValueError("Unverified warm-start checkpoint")
    config = {"recipe": PAIR_AUC_RECIPE, "seed": PAIR_AUC_SEED, "train_sha256": fingerprint(train),
              "selection_sha256": fingerprint(selection), "audit_sha256": fingerprint(audit),
              "warmstart_sha256": pair_auc_sha(warmstart),
              "batch_size": args.batch_size, "accumulation": args.accumulation,
              "workers": args.workers, "initial_lr": 2e-6, "final_lr": 5e-7,
              "schedule_epochs": PAIR_AUC_SCHEDULE_EPOCHS, "subset_size": PAIR_AUC_SUBSET_SIZE,
              "subset_seed_rule": "PAIR_AUC_SEED+epoch; 12000 unique official-train records then 15pct-positive weighted draws with replacement", "pairwise_margin": 0.5, "pairwise_weight": 0.05, "target_positive_fraction": 0.15}
    if args.resume:
        run_dir = args.resume.resolve()
        if run_dir.parent != PAIR_AUC_RUN_ROOT.resolve():
            raise ValueError("Resume only from this recipe's pair_auc_run directory")
        prior = load_checkpoint(run_dir / "last.pth")
        if prior["config"] != config or prior["epoch"] >= args.epochs:
            raise ValueError("Checkpoint recipe or target epoch mismatch")
        model = create_model(pretrained=False).cuda()
        model.load_state_dict(prior["model"], strict=True)
        ema = copy.deepcopy(model).eval()
        ema.load_state_dict(prior["ema"], strict=True)
        start, step, best, patience = prior["epoch"] + 1, prior["step"], prior["best_auroc"], prior["patience"]
    else:
        run_dir = PAIR_AUC_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        model = create_model(pretrained=False).cuda()
        model.load_state_dict(base["model"], strict=True)
        ema = copy.deepcopy(model).eval()
        start, step, best, patience = 1, 0, -1.0, 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    if args.resume:
        optimizer.load_state_dict(prior["optimizer"])
        scaler.load_state_dict(prior["scaler"])
    print(f"pair_auc_run:{run_dir} train:{len(train)} selection_valid:{len(selection)} "
          f"audit_held_for_final:{len(audit)} batch:{args.batch_size}x{args.accumulation}", flush=True)
    if not args.resume:
        baseline = evaluate(ema, selection, args.batch_size, args.workers, "cuda")
        best = baseline["auroc"]
        print(f"start_selection_AUROC:{best:.12f}", flush=True)
    positive = sum(r[2] for r in train)
    pos_weight = torch.tensor(1.0, device="cuda")
    for epoch in range(start, args.epochs + 1):
        subset = random.Random(PAIR_AUC_SEED + epoch).sample(train, PAIR_AUC_SUBSET_SIZE)
        print(f"epoch:{epoch} subset:{len(subset)} subset_sha256:{fingerprint(subset)}", flush=True)
        loss, updates = pair_auc_train_one(model, ema, subset, optimizer, scaler, pos_weight, epoch, step,
                                  args.batch_size, args.accumulation, args.workers, amp_dtype)
        step += updates
        result = evaluate(ema, selection, args.batch_size, args.workers, "cuda")
        auc = result["auroc"]
        print(f"epoch:{epoch} train_loss:{loss:.6f} selection_valid_AUROC:{auc:.12f} AP:{result['ap']:.9f}", flush=True)
        if auc > best:
            best, patience = auc, 0
            atomic_save({"recipe": PAIR_AUC_RECIPE, "model_name": NAME,
                         "model": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                         "config": config, "metric_scope": "official_valid_selection70",
                         "selection_valid_auroc": auc, "selection_valid_ap": result["ap"],
                         "selected_epoch": epoch, "training_complete": True}, run_dir / "best.pth")
            print(f"improved_candidate:{run_dir / 'best.pth'}", flush=True)
        else:
            patience += 1
        atomic_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "step": step, "best_auroc": best,
                     "patience": patience, "config": config}, run_dir / "last.pth")
        if patience >= 4:
            print("stopped_after_four_epochs_without_AUROC_gain", flush=True)
            break
    print(f"finished_best_selection_valid_AUROC:{best:.12f}; audit_not_evaluated", flush=True)

if __name__ == "__main__" and "--pairauc" in __import__("sys").argv:
    __import__("sys").argv.remove("--pairauc")
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    pair_auc_run(parser.parse_args())
