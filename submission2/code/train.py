"""新实验：两种非医学预训练 CNN 分别全量微调，按官方验证 AUROC 保存最佳参数。"""

import argparse
import os
import random
import time
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MODEL_NAMES, create_model
from utils import (BATCH_SIZE, DATA, FundusDataset, MODEL_PATH, NUM_WORKERS, RECIPE,
                   ROOT, batch_report, metrics, print_metrics, print_progress,
                   read_records, should_log_batch)


# 【自写】每个模型独立配方；增加轮数前先看验证 AUROC 和阳性召回，不靠训练 ACC。
CONFIG = {
    "convnext_tiny.fb_in1k": {"epochs": 20, "backbone_lr": 2e-5, "head_lr": 2e-4,
                               "weight_decay": 0.02, "patience": 4},# 20轮、4轮早停、0.02衰减、2e-5学习率、2e-4头学习率
    "efficientnetv2_rw_s.ra2_in1k": {"epochs": 20, "backbone_lr": 3e-5, "head_lr": 3e-4,
                                      "weight_decay": 0.01, "patience": 5},# 20轮、5轮早停、0.01衰减、3e-5学习率、3e-4头学习率
}


def set_seed(seed=2026):
    """【自写】固定训练随机源；验证和测试始终没有随机变换。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_license():
    """【自写】从 Apache 官方站保存完整许可证正文；缺失时拒绝继续预训练。"""
    license_path = ROOT / "licenses" / "Apache-2.0.txt"
    if not license_path.is_file():
        with urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30) as response:
            content = response.read()
        if b"Apache License" not in content[:150] or b"TERMS AND CONDITIONS" not in content:
            raise RuntimeError("无法核实 Apache-2.0 许可证全文")
        license_path.parent.mkdir(parents=True, exist_ok=True)
        license_path.write_bytes(content)
    return license_path


def loader(records, training, batch_size=BATCH_SIZE):
    """【改自课堂 CNN DataLoader】每轮遍历全部官方训练图片，不做有放回抽样。"""
    return DataLoader(FundusDataset(records, training=training), batch_size=batch_size,
                      shuffle=training, num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=NUM_WORKERS > 0)


def validate(model, valid_loader, device, title):
    """【改自课堂 CNN evaluate】按原验证集逐批计算连续阳性分数和 AUROC。"""
    model.eval()
    labels, scores, started = [], [], time.time()
    with torch.inference_mode():
        for index, (images, targets) in enumerate(valid_loader, 1):
            logits = model(images.to(device, non_blocking=True)).flatten()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.tolist())
            if should_log_batch(index, len(valid_loader)):
                print_progress("valid", index, len(valid_loader), started,
                               batch_report(logits, targets))
    result = metrics(labels, scores)
    print_metrics(title, result)
    return result, np.asarray(scores)


def train_epoch(model, train_loader, optimizer, scaler, pos_weight, device, epoch, name):
    """【改自课堂 CNN train】AMP 全模型微调、阳性加权 focal、梯度裁剪。"""
    model.train()
    total_loss, started = 0.0, time.time()
    for index, (images, targets) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images).flatten()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none",
                                                     pos_weight=pos_weight)
            p = torch.sigmoid(logits)
            pt = targets * p + (1 - targets) * (1 - p)
            loss = ((1 - pt) * bce).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if should_log_batch(index, len(train_loader)):
            print_progress(f"{name} epoch:{epoch} train", index, len(train_loader), started,
                           f", loss:{loss.item():.5f}" + batch_report(logits, targets))
    return total_loss / len(train_loader)


def save_checkpoint(best_states, best_metrics, origins, selected=None):
    """【自写】两个分支统一保存到一个 PTH；仅最终双分支完备才允许预测。"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"recipe": RECIPE, "states": best_states, "best_metrics": best_metrics,
                  "pretraining": origins, "selection": selected,
                  "training_complete": selected is not None,
                  "torch_version": torch.__version__}
    temporary = MODEL_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, MODEL_PATH)
    if MODEL_PATH.stat().st_size > 500_000_000:
        raise RuntimeError("最终模型超过赛项 500MB 限制")


def select_inference(best_states, valid_records, device):
    """【自写】仅用官方验证集比较固定双模型均值与两个单模型，不碰测试图。"""
    valid_loader = loader(valid_records, training=False)
    labels = [label for _, label in valid_records]
    vectors = {}
    for name in MODEL_NAMES:
        model, _ = create_model(name)
        model.load_state_dict(best_states[name], strict=True)
        model.to(device)
        _, vectors[name] = validate(model, valid_loader, device, name + " 最佳参数复核")
        del model
        torch.cuda.empty_cache()
    candidates = {MODEL_NAMES[0]: vectors[MODEL_NAMES[0]],
                  MODEL_NAMES[1]: vectors[MODEL_NAMES[1]],
                  "ensemble": (vectors[MODEL_NAMES[0]] + vectors[MODEL_NAMES[1]]) / 2}
    measured = {name: metrics(labels, scores) for name, scores in candidates.items()}
    for name, result in measured.items():
        print_metrics("最终候选 " + name, result)
    selected = max(candidates, key=lambda name: measured[name]["auroc"])
    print("只按官方验证 AUROC 选择推理方案:", selected, flush=True)
    return {"name": selected, "validation": measured[selected]}


def train():
    """【自写】先验证起点，再逐模型完整训练；只有 AUROC 严格提高才更新该分支。"""
    if not torch.cuda.is_available():
        raise RuntimeError("本方案需 GPU。请在带 CUDA 的训练机安装匹配的 PyTorch，不要在此 CPU 机启动完整训练。")
    if MODEL_PATH.exists():
        raise FileExistsError(f"候选权重已存在，避免覆盖: {MODEL_PATH}。请先备份并明确处理后再训练。")
    set_seed()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    print("GPU训练已启用:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, flush=True)
    print("预训练许可正文:", ensure_license(), flush=True)
    train_records, valid_records = read_records("train"), read_records("valid")
    labels = [label for _, label in train_records]
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("训练集必须包含阳性和阴性")
    pos_weight = torch.tensor(min(10.0, (negatives / positives) ** 0.5), device=device)
    print(f"训练:{len(train_records)}张 阳性:{positives} 阴性:{negatives}; "
          f"官方验证:{len(valid_records)}张; pos_weight:{pos_weight.item():.3f}", flush=True)
    valid_loader = loader(valid_records, training=False)
    best_states, best_metrics, origins = {}, {}, {}
    for name in MODEL_NAMES:
        config = CONFIG[name]
        print("开始模型:", name, "配置:", config, flush=True)
        model, origin = create_model(name, ROOT / "pretrain_model")
        origins[name] = origin
        print("原权重:", origin["url"], "SHA256:", origin["sha256"], flush=True)
        model.to(device)
        validate(model, valid_loader, device, name + " 训练前")
        classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in classifier_ids],
             "lr": config["backbone_lr"]},
            {"params": [p for p in model.parameters() if id(p) in classifier_ids],
             "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        scaler = torch.amp.GradScaler("cuda")
        train_loader = loader(train_records, training=True)
        best_auc, stale = -1.0, 0
        for epoch in range(1, config["epochs"] + 1):
            loss = train_epoch(model, train_loader, optimizer, scaler, pos_weight, device, epoch, name)
            print(f"{name} epoch:{epoch}, train_loss:{loss:.6f}", flush=True)
            result, _ = validate(model, valid_loader, device, f"{name} epoch:{epoch}")
            if result["auroc"] > best_auc:
                best_auc, stale = result["auroc"], 0
                best_states[name] = {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()}
                best_metrics[name] = {"epoch": epoch, **result}
                save_checkpoint(best_states, best_metrics, origins)
                print("验证 AUROC 提高，已保存该分支最佳参数。", flush=True)
            else:
                stale += 1
                print("未提高最佳 AUROC；连续未提高:", stale, flush=True)
            schedule.step()
            if stale >= config["patience"]:
                print("提前停止当前模型，保留最佳验证轮。", flush=True)
                break
        del train_loader, optimizer, schedule, scaler, model
        torch.cuda.empty_cache()
    selected = select_inference(best_states, valid_records, device)
    save_checkpoint(best_states, best_metrics, origins, selected)
    print(f"完整训练结束。最终单文件:{MODEL_PATH}, 大小:{MODEL_PATH.stat().st_size / 1e6:.1f}MB", flush=True)
    print("train.py 不生成提交 CSV。需要提交时单独运行 predict.py。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 全量训练；不生成测试 CSV")
    parser.parse_args()
    train()
