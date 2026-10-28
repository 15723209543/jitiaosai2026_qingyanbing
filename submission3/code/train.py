"""实验3：官方 train+valid 既训练又评估；AUROC 为训练集内指标，不能当成独立验证。"""

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
from utils import (BASELINE_MODEL_PATH, BASE_RECIPE, BATCH_SIZE, DATA, FundusDataset,
                   MODEL_PATH, NUM_WORKERS, RECIPE, ROOT, batch_report, fit_records,
                   metrics, print_metrics, print_progress, load_resume_checkpoint, restore_candidate_model,
                   resume_baseline_auc, resume_start_epoch, should_log_batch)


# 【自写】每个模型独立配方；增加轮数前先看验证 AUROC 和阳性召回，不靠训练 ACC。
CONFIG = {
    "convnext_tiny.fb_in1k": {"epochs": 30, "backbone_lr": 2e-5, "head_lr": 2e-4,
                               "weight_decay": 0.02, "patience": 8},  # 本次最多30轮；重新运行从第1轮计数
    "efficientnetv2_rw_s.ra2_in1k": {"epochs": 38, "backbone_lr": 3e-5, "head_lr": 3e-4,
                                      "weight_decay": 0.01, "patience": 10},  # 本次最多38轮；重新运行从第1轮计数
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
                      shuffle=training, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=NUM_WORKERS > 0)


def select_device():
    """【自写】启动时报告计算设备；有 CUDA 优先 GPU，否则允许 CPU 训练。"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU训练已启用:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("当前使用CPU训练（可继续运行，但完整训练会较慢）", flush=True)
    print("PyTorch版本:", torch.__version__, "CUDA版本:", torch.version.cuda, flush=True)
    return device


def validate(model, valid_loader, device, title):
    """【改自课堂 CNN evaluate】对合并图逐批计算训练集内 AUROC。"""
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
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
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
    metrics_without_epochs = {name: {key: value for key, value in result.items()
                                     if key != "epoch"}
                              for name, result in best_metrics.items()}
    checkpoint = {"recipe": RECIPE, "states": best_states, "best_metrics": metrics_without_epochs,
                  "pretraining": origins, "selection": selected,
                  "training_complete": selected is not None,
                  "torch_version": torch.__version__}
    temporary = MODEL_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, MODEL_PATH)
    if MODEL_PATH.stat().st_size > 500_000_000:
        raise RuntimeError("最终模型超过赛项 500MB 限制")


def select_inference(best_states, valid_records, device):
    """【自写】实验3仅按训练集内 AUROC 比较分支及均值；不是独立验证。"""
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
    print("只按训练集内 AUROC 选择推理方案（非独立验证）:", selected, flush=True)
    return {"name": selected, "insample": measured[selected]}


def starting_auc(checkpoint, name, measured_auc):
    """【自写】正式版验证 AUROC 不与实验3训练集内 AUROC 比；候选版需复核同一口径。"""
    if checkpoint is None:
        return -1.0
    if (checkpoint["recipe"] == BASE_RECIPE or
            checkpoint["best_metrics"][name].get("metric_scope") == "official_valid"):
        return measured_auc
    return resume_baseline_auc(checkpoint, name, measured_auc)


def train():
    """【自写】实验3合并官方 train/valid 训练并用同图评估；该 AUROC 不代表泛化。"""
    device = select_device()
    set_seed()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    checkpoint = (load_resume_checkpoint(MODEL_PATH, MODEL_NAMES) if MODEL_PATH.is_file()
                  else load_resume_checkpoint(BASELINE_MODEL_PATH, MODEL_NAMES, BASE_RECIPE))
    if checkpoint is not None:
        print("检测到已有参数，先在合并数据上计算训练集内基线；本次从第1轮重新计数。", flush=True)
    print("预训练许可正文:", ensure_license(), flush=True)
    train_records = fit_records()
    valid_records = train_records
    labels = [label for _, label in train_records]
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("训练集必须包含阳性和阴性")
    pos_weight = torch.tensor(min(10.0, (negatives / positives) ** 0.5), device=device)
    print(f"实验3 train+valid 全部用于训练和训练集内评估:{len(train_records)}张 "
          f"阳性:{positives} 阴性:{negatives}; pos_weight:{pos_weight.item():.3f}; "
          "该 AUROC 存在数据泄漏，不是独立验证成绩。", flush=True)
    valid_loader = loader(valid_records, training=False)
    best_states = dict(checkpoint["states"]) if checkpoint else {}
    best_metrics = ({name: {key: value for key, value in result.items() if key != "epoch"}
                     for name, result in checkpoint["best_metrics"].items()} if checkpoint else {})
    if checkpoint is not None and checkpoint["recipe"] == BASE_RECIPE:
        for result in best_metrics.values():
            result["metric_scope"] = "official_valid"
    origins = dict(checkpoint["pretraining"]) if checkpoint else {}
    for name in MODEL_NAMES:
        config = CONFIG[name]
        print("开始模型:", name, "配置:", config, "评价口径:训练集内AUROC", flush=True)
        model, origin, resumed = restore_candidate_model(
            name, checkpoint, create_model, ROOT / "pretrain_model")
        origins[name] = origin
        print("原权重:", origin["url"], "SHA256:", origin["sha256"], flush=True)
        model.to(device)
        baseline, _ = validate(model, valid_loader, device, name + " 训练集内续训前复核" if resumed else name + " 训练集内训练前")
        start_epoch = resume_start_epoch(checkpoint, name)
        best_auc = starting_auc(checkpoint, name, baseline["auroc"]) if resumed else -1.0
        if resumed and best_metrics[name].get("metric_scope") != "insample":
            best_metrics[name] = {**baseline, "metric_scope": "insample"}
        if resumed:
            print(f"{name} 已载入上次最佳权重，复核AUROC:{baseline['auroc']:.6f}，"
                  f"本次选模门槛:{best_auc:.6f}；从第1轮开始。", flush=True)
        classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in classifier_ids],
             "lr": config["backbone_lr"]},
            {"params": [p for p in model.parameters() if id(p) in classifier_ids],
             "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"] - start_epoch + 1)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        train_loader = loader(train_records, training=True)
        stale = 0
        for epoch in range(start_epoch, config["epochs"] + 1):
            loss = train_epoch(model, train_loader, optimizer, scaler, pos_weight, device, epoch, name)
            print(f"{name} epoch:{epoch}, train_loss:{loss:.6f}", flush=True)
            result, _ = validate(model, valid_loader, device, f"{name} epoch:{epoch} 训练集内")
            if result["auroc"] > best_auc:
                best_auc, stale = result["auroc"], 0
                best_states[name] = {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()}
                best_metrics[name] = {**result, "metric_scope": "insample"}
                save_checkpoint(best_states, best_metrics, origins)
                print("训练集内 AUROC 提高，已保存该分支参数；不代表独立验证提升。", flush=True)
            else:
                stale += 1
                print("训练集内 AUROC 未提高；连续未提高:", stale, flush=True)
            schedule.step()
            if stale >= config["patience"]:
                print("提前停止当前模型，保留最佳验证轮。", flush=True)
                break
        del train_loader, optimizer, schedule, scaler, model
        torch.cuda.empty_cache()
    selected = select_inference(best_states, valid_records, device)
    save_checkpoint(best_states, best_metrics, origins, selected)
    print(f"完整训练结束。最终单文件:{MODEL_PATH}, 大小:{MODEL_PATH.stat().st_size / 1e6:.1f}MB", flush=True)
    print("train.py 不生成提交 CSV。训练集内 AUROC 不代表泛化；需要预测时单独运行 predict.py。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="优先 GPU、无 CUDA 时用 CPU 全量训练；不生成测试 CSV")
    parser.parse_args()
    train()
