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
from train import EMA_DECAY, evaluate, update_ema
from utils import ROOT, FundusDataset, atomic_save, fingerprint, load_checkpoint, read_records

RECIPE = "small320_pairauc10_random10000_refine_selection70_v1"
SEED = 20261110
BASE = ROOT.parent / "runs" / "submission_small_pair_auc_subset" / "20261106_141143" / "candidate_epoch6.pth"
RUN_ROOT = ROOT.parent / "runs" / "submission_small_pair_auc_refine"
SCHEDULE_EPOCHS = 6  # Fixed across --resume calls.
SUBSET_SIZE = 10000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rate(step, updates_per_epoch):
    fraction = min(1.0, step / (SCHEDULE_EPOCHS * updates_per_epoch))
    return 3e-7 + (1e-6 - 3e-7) * (1 + math.cos(math.pi * fraction)) / 2


def train_one(model, ema, records, optimizer, scaler, pos_weight, epoch, step,
              batch_size, accumulation, workers, amp_dtype):
    generator = torch.Generator().manual_seed(SEED + epoch)
    positives = sum(r[2] >= 0.5 for r in records)
    negatives = len(records) - positives
    if not positives or not negatives:
        raise ValueError('Pairwise AUC needs both classes')
    weights = [0.10 / positives if r[2] >= 0.5 else 0.90 / negatives for r in records]
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
            loss = main + 0.2 * aux + 0.05 * consistency + 0.02 * pairwise
        scaler.scale(loss / accumulation).backward()
        loss_sum += float(loss.detach())
        if index % accumulation == 0 or index == len(loader):
            for group in optimizer.param_groups:
                group["lr"] = rate(step + updates, updates_per_epoch)
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


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if not (1 <= args.epochs <= SCHEDULE_EPOCHS and args.batch_size > 0 and
            args.accumulation > 0 and args.workers >= 0):
        raise ValueError("Invalid training arguments")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
    base = load_checkpoint(BASE)
    if base.get("metric_scope") != "official_valid_selection70" or base.get("training_complete") is not True:
        raise ValueError("Unverified warm-start checkpoint")
    config = {"recipe": RECIPE, "seed": SEED, "train_sha256": fingerprint(train),
              "selection_sha256": fingerprint(selection), "audit_sha256": fingerprint(audit),
              "warmstart_sha256": sha(BASE),
              "batch_size": args.batch_size, "accumulation": args.accumulation,
              "workers": args.workers, "initial_lr": 1e-6, "final_lr": 3e-7,
              "schedule_epochs": SCHEDULE_EPOCHS, "subset_size": SUBSET_SIZE,
              "subset_seed_rule": "SEED+epoch; 10000 unique official-train records then 10pct-positive weighted draws with replacement", "pairwise_margin": 0.5, "pairwise_weight": 0.02, "target_positive_fraction": 0.10}
    if args.resume:
        run_dir = args.resume.resolve()
        if run_dir.parent != RUN_ROOT.resolve():
            raise ValueError("Resume only from this recipe's run directory")
        prior = load_checkpoint(run_dir / "last.pth")
        if prior["config"] != config or prior["epoch"] >= args.epochs:
            raise ValueError("Checkpoint recipe or target epoch mismatch")
        model = create_model(pretrained=False).cuda()
        model.load_state_dict(prior["model"], strict=True)
        ema = copy.deepcopy(model).eval()
        ema.load_state_dict(prior["ema"], strict=True)
        start, step, best, patience = prior["epoch"] + 1, prior["step"], prior["best_auroc"], prior["patience"]
    else:
        run_dir = RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        model = create_model(pretrained=False).cuda()
        model.load_state_dict(base["model"], strict=True)
        ema = copy.deepcopy(model).eval()
        start, step, best, patience = 1, 0, -1.0, 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    if args.resume:
        optimizer.load_state_dict(prior["optimizer"])
        scaler.load_state_dict(prior["scaler"])
    print(f"run:{run_dir} train:{len(train)} selection_valid:{len(selection)} "
          f"audit_held_for_final:{len(audit)} batch:{args.batch_size}x{args.accumulation}", flush=True)
    if not args.resume:
        baseline = evaluate(ema, selection, args.batch_size, args.workers, "cuda")
        best = baseline["auroc"]
        print(f"start_selection_AUROC:{best:.12f}", flush=True)
    positive = sum(r[2] for r in train)
    pos_weight = torch.tensor(1.0, device="cuda")
    for epoch in range(start, args.epochs + 1):
        subset = random.Random(SEED + epoch).sample(train, SUBSET_SIZE)
        print(f"epoch:{epoch} subset:{len(subset)} subset_sha256:{fingerprint(subset)}", flush=True)
        loss, updates = train_one(model, ema, subset, optimizer, scaler, pos_weight, epoch, step,
                                  args.batch_size, args.accumulation, args.workers, amp_dtype)
        step += updates
        result = evaluate(ema, selection, args.batch_size, args.workers, "cuda")
        auc = result["auroc"]
        print(f"epoch:{epoch} train_loss:{loss:.6f} selection_valid_AUROC:{auc:.12f} AP:{result['ap']:.9f}", flush=True)
        if auc > best:
            best, patience = auc, 0
            atomic_save({"recipe": RECIPE, "model_name": NAME,
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
        if patience >= 3:
            print("stopped_after_three_epochs_without_AUROC_gain", flush=True)
            break
    print(f"finished_best_selection_valid_AUROC:{best:.12f}; audit_not_evaluated", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    run(parser.parse_args())
