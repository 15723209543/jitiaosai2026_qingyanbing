"""课堂CNN的todo3：五模型分别训练、验证；单模最佳和联合最佳都存入一份pth。"""

import copy
import sys
import time
from datetime import datetime
from itertools import groupby

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("训练程序已启动, 正在加载PyTorch和数据读取模块...", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ImageModel, create_dataset
from predict import evaluate
from utils import (CPU_THREADS, EPOCHS, FocalLoss, MODEL_NAMES, MODEL_PATH, MODEL_TRAINING,
                   RANK_WEIGHTS, RECORD_DIR, RECIPE, get_learning_rate,
                   load_ensemble, print_progress, print_rank_weights, save_ensemble, save_training_record)

SEED, TRAIN_BATCH_SIZE = 0, 64


# 【来源】参考用户提供的CUDA检测代码；有CUDA时优先使用第一块显卡。
def choose_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 【来源】沿用submission3；训练开始前只打印一次实际设备。
def print_device_info(device):
    if device.type == "cuda":
        print("=" * 50)
        print("GPU训练已启用")
        print("GPU型号:", torch.cuda.get_device_name(0))
        print("PyTorch版本:", torch.__version__)
        print("CUDA版本:", torch.version.cuda)
        print("GPU显存: %.2f GB" % (
            torch.cuda.get_device_properties(0).total_memory / 1024**3))
        print("=" * 50, flush=True)
    else:
        print("当前使用CPU训练")
        print("PyTorch版本:", torch.__version__)
        print("CUDA是否可用:", torch.cuda.is_available(), flush=True)


DEVICE = choose_device()


# 【来源】本项目自写；训练启动时逐行打印五套独立参数，便于核对续训配置。
def print_training_profiles():
    for name in MODEL_NAMES:
        config = MODEL_TRAINING[name]
        regularization = (f"dropout:{config['dropout']:.2f}" if name != "convnext_tiny" else
                          f"随机深度:{config['stochastic_depth']:.2f}")
        print(f"{name}: 目标轮次:{EPOCHS[name]}, 初训学习率:{config['learning_rate']:.2e}, "
              f"续训学习率:{config['finetune_lr']:.2e}, 权重衰减:{config['weight_decay']:.2e}, "
              f"阳性损失占比:{config['positive_share']:.2f}, focal γ:{config['focal_gamma']:.1f}, "
              f"梯度截断:{config['max_grad_norm']:.1f}, {regularization}", flush=True)


# 【来源】沿用submission3；只凭验证AUROC严格提高来更新参数。
def is_better(candidate, previous):
    return previous is None or candidate["auroc"] > previous["auroc"]


# 【来源】修改submission3训练顺序；每个模型累计轮数独立，已完成者不再训练。
def training_order(rounds_by_model, completed_epochs=None):
    completed_epochs = completed_epochs or {}
    for epoch in range(1, max(rounds_by_model.values(), default=0) + 1):
        for name in MODEL_NAMES:
            if completed_epochs.get(name, 0) < epoch <= rounds_by_model[name]:
                yield epoch, name


# 【来源】复用课堂CNN的DataLoader洗牌；每轮遍历全部官方训练图片。
def create_train_loader(train_dataset):
    return DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                      pin_memory=DEVICE.type == "cuda")


# 【来源】修改submission3类别加权；每个模型单独设置阳性损失占比，仍遍历训练全集。
def create_class_weights(labels, name):
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2)
    if torch.any(counts == 0):
        raise ValueError("训练集必须同时包含0和1")
    positive_share = MODEL_TRAINING[name]["positive_share"]
    return torch.tensor([(1 - positive_share) / counts[0], positive_share / counts[1]],
                        dtype=torch.float32)


# 【来源】CNN沿用原交叉熵；四个非CNN模型依各自gamma使用Focal Loss，不共用损失对象。
def create_criterion(labels, name):
    weights = create_class_weights(labels, name).to(DEVICE)
    gamma = MODEL_TRAINING[name]["focal_gamma"]
    return nn.CrossEntropyLoss(weight=weights) if gamma == 0 else FocalLoss(weights, gamma)


# 【来源】原余弦学习率；各模型独立设置初训及30轮后续训的起点、目标轮数。
def get_model_learning_rate(name, epoch):
    config = MODEL_TRAINING[name]
    base_epochs = config["base_epochs"]
    if epoch <= base_epochs:
        return get_learning_rate(epoch, base_epochs, config["learning_rate"], config["min_lr"])
    return get_learning_rate(epoch - base_epochs, EPOCHS[name] - base_epochs,
                             config["finetune_lr"], config["min_lr"])


# 【来源】复用课堂CNN交叉熵前向/反向/更新；只更新当前网络。
def train_step(model, x, y, optimizer, criterion, max_grad_norm):
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    predicted_positive = torch.argmax(y_pred, dim=1) == 1
    return (loss.item(), int((y == 1).sum().item()),
            int((predicted_positive & (y == 1)).sum().item()),
            int((predicted_positive & (y == 0)).sum().item()))


# 【来源】修改submission3单模型循环；训练全集一轮，再验证并更新本模型最佳。
def train_one(name, epoch, model, best_model, loader, valid_dataset, optimizer, criterion, best_metrics):
    lr = get_model_learning_rate(name, epoch)
    optimizer.param_groups[0]["lr"] = lr
    total_loss, total_positive, total_tp, total_fp, total_samples, start = 0.0, 0, 0, 0, 0, time.time()
    print(f"开始训练 {name} epoch: {epoch}/{EPOCHS[name]}, lr:{lr:.2e}", flush=True)
    for batch_idx, (x, y) in enumerate(loader, 1):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        loss, positives, tp, fp = train_step(
            model, x, y, optimizer, criterion, MODEL_TRAINING[name]["max_grad_norm"])
        total_loss += loss * len(y)
        total_positive += positives
        total_tp += tp
        total_fp += fp
        total_samples += len(y)
        print_progress(f"{name} epoch: {epoch}/{EPOCHS[name]}", batch_idx, len(loader), start,
                       f", loss:{total_loss / total_samples:.5f}, 真实阳性:{total_positive}, "
                       f"报对阳性:{total_tp}, 误报:{total_fp}")
    print(f"epoch: {epoch}, loss: {total_loss / total_samples:.5f}, "
          f"真实阳性:{total_positive}, 报对阳性:{total_tp}, 误报:{total_fp}, "
          f"time:{time.time() - start:.2f}s", flush=True)
    metrics = evaluate(valid_dataset, {name: model})
    if is_better(metrics, best_metrics.get(name)):
        best_metrics[name] = metrics
        best_model.load_state_dict(model.state_dict())
        print(f"已更新{name}独立最佳, 验证AUROC:{metrics['auroc']:.6f}", flush=True)
    else:
        print(f"未超过{name}独立最佳AUROC:{best_metrics[name]['auroc']:.6f}", flush=True)
    return {"model": name, "epoch": epoch, "train_loss": total_loss / total_samples,
            "train_positives": total_positive, "train_tp": total_tp, "train_fp": total_fp,
            "learning_rate": lr, **metrics}


# todo 3. 五模型训练；第一轮前验证基线；每轮五模型训练后联合验证。
# 【来源】沿用课堂CNN训练入口，改为单模最佳与整组最佳两层选模、一个pth文件。
def train(train_dataset, valid_dataset):
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    run_dir = RECORD_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_five_models")
    run_dir.mkdir(parents=True, exist_ok=False)
    history = []
    if MODEL_PATH.is_file():
        best_models, checkpoint = load_ensemble(MODEL_PATH, "individual")
        ensemble_models, _ = load_ensemble(MODEL_PATH, "ensemble")
        models = copy.deepcopy(best_models)
        best_metrics = dict(checkpoint["best_metrics"])
        completed = dict(checkpoint.get("completed_epochs") or
                         {name: checkpoint.get("epoch", 0) for name in MODEL_NAMES})
        old_weights = tuple(checkpoint.get("rank_weights", ()))
        ensemble_best = checkpoint.get("validation") if old_weights == tuple(RANK_WEIGHTS) else None
        print("从已有单文件五模型参数继续训练。", flush=True)
    else:
        models = {name: ImageModel(name) for name in MODEL_NAMES}
        print("未找到五模型参数，五个分支从随机参数开始；不使用外部预训练权重。", flush=True)
        best_models, ensemble_models = copy.deepcopy(models), copy.deepcopy(models)
        best_metrics, ensemble_best = {}, None
        completed = {name: 0 for name in MODEL_NAMES}
    for model in models.values():
        model.to(DEVICE)
    print("并行预读取完整训练集与验证集图片...", flush=True)
    train_dataset.preload()
    valid_dataset.preload()
    print(f"训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"每轮{len(create_train_loader(train_dataset))}批, CPU线程:{CPU_THREADS}", flush=True)
    print("训练前依次验证五个模型的基线。", flush=True)
    for name in MODEL_NAMES:
        print("训练前单模型验证:", name, flush=True)
        metrics = evaluate(valid_dataset, {name: models[name]})
        if is_better(metrics, best_metrics.get(name)):
            best_metrics[name] = metrics
            best_models[name].load_state_dict(models[name].state_dict())
    if MODEL_PATH.is_file() and ensemble_best is None:
        print("排序权重已改变，重新验证旧整组快照作为基线。", flush=True)
        ensemble_best = evaluate(valid_dataset, ensemble_models)
    print("训练前五模型联合验证。", flush=True)
    baseline = evaluate(valid_dataset, models)
    if is_better(baseline, ensemble_best):
        ensemble_best = baseline
        for name in MODEL_NAMES:
            ensemble_models[name].load_state_dict(models[name].state_dict())
    save_ensemble(best_models, ensemble_models, MODEL_PATH, ensemble_best,
                  best_metrics, completed_epochs=completed)
    loaders = {name: create_train_loader(train_dataset) for name in MODEL_NAMES}
    optimizers = {name: optim.AdamW(models[name].parameters(),
                                   weight_decay=MODEL_TRAINING[name]["weight_decay"])
                  for name in MODEL_NAMES if EPOCHS[name] > 0}
    criteria = {name: create_criterion(train_dataset.labels, name) for name in MODEL_NAMES
                if EPOCHS[name] > 0}
    for epoch, jobs in groupby(training_order(EPOCHS, completed), key=lambda item: item[0]):
        for _, name in jobs:
            history.append(train_one(name, epoch, models[name], best_models[name], loaders[name],
                                     valid_dataset, optimizers[name], criteria[name], best_metrics))
            completed[name] = epoch
        print(f"第{epoch}轮五模型联合验证:", flush=True)
        combined = evaluate(valid_dataset, models)
        if is_better(combined, ensemble_best):
            ensemble_best = combined
            for name in MODEL_NAMES:
                ensemble_models[name].load_state_dict(models[name].state_dict())
            print(f"已更新整组最佳, 验证AUROC:{combined['auroc']:.6f}", flush=True)
        else:
            print(f"未超过整组最佳AUROC:{ensemble_best['auroc']:.6f}", flush=True)
        save_ensemble(best_models, ensemble_models, MODEL_PATH, ensemble_best,
                      best_metrics, epoch=epoch, completed_epochs=completed)
        save_training_record(run_dir, history,
                             {"recipe": RECIPE, "best_metrics": best_metrics,
                              "combined_validation": ensemble_best, "last_epoch": epoch,
                              "completed_epochs": completed, "rank_weights": RANK_WEIGHTS,
                              "model_training": MODEL_TRAINING,
                              "complete": False})
    save_ensemble(best_models, ensemble_models, MODEL_PATH, ensemble_best,
                  best_metrics, epoch=max(completed.values(), default=0),
                  complete=True, completed_epochs=completed)
    size = MODEL_PATH.stat().st_size
    if size > 500_000_000:
        raise ValueError(f"五模型权重{size / 1_000_000:.1f}MB超过500MB限制")
    save_training_record(run_dir, history,
                         {"recipe": RECIPE, "best_metrics": best_metrics,
                          "combined_validation": ensemble_best,
                          "completed_epochs": completed, "rank_weights": RANK_WEIGHTS,
                          "model_training": MODEL_TRAINING,
                          "model_file": str(MODEL_PATH), "complete": True})
    print(f"训练结束, 最佳联合AUROC:{ensemble_best['auroc']:.6f}, "
          f"AUPRC:{ensemble_best['auprc']:.6f}, "
          f"预测阳性:{ensemble_best['tp'] + ensemble_best['fp']}, "
          f"参数大小:{size / 1_000_000:.1f}MB", flush=True)
    print("训练不会写答案；需要提交CSV时单独运行predict.py。", flush=True)
    return ensemble_best


# 5. 测试：直接运行只训练及验证，不读取无标签test，不写答案。
if __name__ == "__main__":
    print_device_info(DEVICE)
    print_rank_weights()
    print_training_profiles()
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train(*create_dataset())
