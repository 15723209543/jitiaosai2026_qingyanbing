"""从零训练双视野 CNN；仅官方 train 更新参数、valid 选最佳，不碰无标签 test。"""

import csv
import math
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model import FundusNet
from predict import evaluate
from utils import (EPOCHS, LR, MODEL_PATH, NUM_WORKERS, TRAIN_BATCH,
                   VARIANT, WEIGHT_DECAY, FundusDataset, progress)


def print_device(device):
    """沿用原 train.py 习惯：只在开头打印一次实际 CPU/GPU 与显存。"""
    if device.type == "cuda":
        print("GPU训练已启用:", torch.cuda.get_device_name(device),
              "显存: %.2f GB" % (torch.cuda.get_device_properties(device).total_memory / 1024**3))
    else:
        print("当前使用CPU训练")
    print("PyTorch版本:", torch.__version__, "CUDA版本:", torch.version.cuda, flush=True)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    """参考课堂 CNN 的清梯度、前向、反向、更新四步；新增 AMP 与梯度裁剪。"""
    model.train()
    start, total_loss, count = time.time(), 0.0, 0
    total_counts = torch.zeros(3, dtype=torch.long, device=device)
    window_counts = torch.zeros_like(total_counts)
    window_loss = window_count = 0

    # 【来源】本项目自写；只在真正打印的批次把GPU计数搬回CPU，避免每批额外等待显卡。
    def window_detail():
        positive, tp, fp = window_counts.cpu().tolist()
        return (f", loss:{window_loss / window_count:.5f}, 真实阳性:{positive}, "
                f"报对阳性:{tp}, 误报:{fp}, 漏报:{positive - tp}")

    for batch_index, (full, disc, labels) in enumerate(loader, 1):
        full = full.to(device, non_blocking=True)
        disc = disc.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(full, disc)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        scaler.step(optimizer)
        scaler.update()
        batch_loss = loss.item() * len(labels)
        total_loss += batch_loss
        count += len(labels)
        # 【修改】当前日志区间单独统计，轮末再输出全集累计值。
        predicted = torch.sigmoid(logits.detach().float()) >= 0.5
        positive = labels >= 0.5
        batch_counts = torch.stack((positive.sum(), (positive & predicted).sum(),
                                    ((~positive) & predicted).sum()))
        total_counts += batch_counts
        window_counts += batch_counts
        window_loss += batch_loss
        window_count += len(labels)
        if progress(f"train epoch {epoch}", batch_index, len(loader), start, window_detail):
            window_loss = window_count = 0
            window_counts.zero_()
    total_positive, total_tp, total_fp = total_counts.cpu().tolist()
    print(f"epoch:{epoch}, 真实阳性:{total_positive}, 报对阳性:{total_tp}, "
          f"误报:{total_fp}, 漏报:{total_positive - total_tp}", flush=True)
    return total_loss / count


def save_best(model, path, epoch, result):
    """自写：只保留一个模型 state_dict，原子写入唯一 best_model.pth。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"architecture": "dual_view_residual_se_v1", "epoch": epoch,
                "state_dict": state, "validation": result}, temporary)
    temporary.replace(path)


def train(epochs=EPOCHS):
    """自写：先验证初始/旧权重，再每轮完整训练、完整验证，AUROC严格提高才更新。"""
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_device(device)
    train_data = FundusDataset("train", augment=True)
    valid_data = FundusDataset("valid")
    positives = sum(train_data.labels)
    print(f"训练集:{len(train_data)}张, 阳性:{positives}, 验证集:{len(valid_data)}张, "
          f"阳性:{sum(valid_data.labels)}, 每批:{TRAIN_BATCH}", flush=True)
    loader = DataLoader(train_data, batch_size=TRAIN_BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
                        persistent_workers=NUM_WORKERS > 0)
    model = FundusNet()
    best_auc = float("-inf")
    if MODEL_PATH.is_file():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "dual_view_residual_se_v1":
            raise ValueError("已有参数架构不匹配，拒绝覆盖")
        model.load_state_dict(checkpoint["state_dict"])
        best_auc = float(checkpoint["validation"]["auroc"])
        print("已读取本方案已有参数，先复核完整验证集。", flush=True)
    model.to(device)
    print("训练前验证基线:", flush=True)
    baseline = evaluate(model, valid_data, device)
    if MODEL_PATH.is_file():
        best_auc = baseline["auroc"]  # 以本次重新计算的完整验证结果为准。
    pos_weight = torch.tensor([math.sqrt((len(train_data) - positives) / positives)],
                              device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=LR / 100)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    records = VARIANT.parent / "项目文档" / "训练记录" / "submission2"
    records.mkdir(parents=True, exist_ok=True)
    record_file = records / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    with record_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "auroc", "auprc",
                                                     "tp", "fp", "fn", "lr", "saved"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch)
            print(f"epoch:{epoch}/{epochs}, loss:{loss:.5f}, lr:{optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)
            result = evaluate(model, valid_data, device)
            better = result["auroc"] > best_auc
            if better:
                best_auc = result["auroc"]
                save_best(model, MODEL_PATH, epoch, result)
                print(f"已更新最佳模型: AUROC {best_auc:.6f}", flush=True)
            writer.writerow(dict(epoch=epoch, loss=loss, auroc=result["auroc"],
                                 auprc=result["auprc"], tp=result["tp"], fp=result["fp"],
                                 fn=result["fn"], lr=optimizer.param_groups[0]["lr"], saved=better))
            stream.flush()
            scheduler.step()
    print("训练完成。最佳参数:", MODEL_PATH, "验证AUROC:", f"{best_auc:.6f}", flush=True)


if __name__ == "__main__":
    train()
