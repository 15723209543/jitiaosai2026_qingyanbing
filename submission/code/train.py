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
from predict import collect_probabilities, evaluate
from utils import (CPU_THREADS, EPOCHS, FocalLoss, MODEL_NAMES, MODEL_PATH, MODEL_TRAINING,
                   RANK_WEIGHTS, RECORD_DIR, RECIPE, calculate_metrics, get_learning_rate,
                   load_ensemble, print_progress, print_rank_weights, save_ensemble,
                   rank_weighted_probabilities, save_training_record, tune_rank_weights)

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


# 【来源】本项目自写；训练启动时逐行打印五套独立参数，便于核对本次轮数。
def print_training_profiles():
    for name in MODEL_NAMES:
        config = MODEL_TRAINING[name]
        regularization = (f"dropout:{config['dropout']:.2f}" if name != "convnext_tiny" else
                          f"随机深度:{config['stochastic_depth']:.2f}")
        print(f"{name}: 本次轮数:{EPOCHS[name]}, 学习率:{config['learning_rate']:.2e}, "
              f"权重衰减:{config['weight_decay']:.2e}, "
              f"阳性损失占比:{config['positive_share']:.2f}, focal γ:{config['focal_gamma']:.1f}, "
              f"梯度截断:{config['max_grad_norm']:.1f}, {regularization}", flush=True)


# 【来源】沿用submission3；只凭验证AUROC严格提高来更新参数。
def is_better(candidate, previous):
    return previous is None or candidate["auroc"] > previous["auroc"]


# 【来源】修改submission3训练顺序；本次运行从第0轮开始，各模型轮数独立。
def training_order(rounds_by_model, completed_epochs=None):
    completed_epochs = completed_epochs or {}
    for epoch in range(max(rounds_by_model.values(), default=0)):
        for name in MODEL_NAMES:
            if completed_epochs.get(name, 0) <= epoch < rounds_by_model[name]:
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


# 【来源】原余弦学习率；本次每个模型从第0轮开始单独调度。
def get_model_learning_rate(name, epoch):
    config = MODEL_TRAINING[name]
    return get_learning_rate(epoch + 1, EPOCHS[name], config["learning_rate"], config["min_lr"])


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
    window_positive = window_tp = window_fp = 0
    print(f"开始训练 {name} epoch: {epoch}/{EPOCHS[name]}, lr:{lr:.2e}", flush=True)
    for batch_idx, (x, y) in enumerate(loader, 1):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        loss, positives, tp, fp = train_step(
            model, x, y, optimizer, criterion, MODEL_TRAINING[name]["max_grad_norm"])
        total_loss += loss * len(y)
        total_positive += positives
        total_tp += tp
        total_fp += fp
        window_positive += positives
        window_tp += tp
        window_fp += fp
        total_samples += len(y)
        # 【修改】进度行仅统计自上次输出以来的图片；整轮误报/漏报仍在轮末保存。
        printed = print_progress(f"{name} epoch: {epoch}/{EPOCHS[name]}", batch_idx, len(loader), start,
                                 f", loss:{total_loss / total_samples:.5f}, "
                                 f"真实阳性:{window_positive}, 报对阳性:{window_tp}, "
                                 f"误报:{window_fp}, 漏报:{window_positive - window_tp}")
        if printed:
            window_positive = window_tp = window_fp = 0
    print(f"epoch: {epoch}, loss: {total_loss / total_samples:.5f}, "
          f"真实阳性:{total_positive}, 报对阳性:{total_tp}, "
          f"误报:{total_fp}, 漏报:{total_positive - total_tp}, "
          f"time:{time.time() - start:.2f}s", flush=True)
    metrics = evaluate(valid_dataset, {name: model}, show_counts=False)
    if is_better(metrics, best_metrics.get(name)):
        best_metrics[name] = metrics
        best_model.load_state_dict(model.state_dict())
        print(f"已更新{name}独立最佳, 验证AUROC:{metrics['auroc']:.6f}", flush=True)
    else:
        print(f"未超过{name}独立最佳AUROC:{best_metrics[name]['auroc']:.6f}", flush=True)
    return {"model": name, "epoch": epoch, "train_loss": total_loss / total_samples,
            "train_positives": total_positive, "train_tp": total_tp, "train_fp": total_fp,
            "train_fn": total_positive - total_tp,
            "learning_rate": lr, **metrics}


# todo 3. 五模型训练；第一轮前验证基线；每轮五模型训练后联合验证。
# 【来源】沿用课堂CNN训练入口，改为单模最佳与整组最佳两层选模、一个pth文件。
def train(train_dataset, valid_dataset):
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    run_dir = RECORD_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_five_models")
    run_dir.mkdir(parents=True, exist_ok=False)
    history, rank_history = [], []
    if MODEL_PATH.is_file():
        best_models, checkpoint = load_ensemble(MODEL_PATH, "individual")
        ensemble_models, _ = load_ensemble(MODEL_PATH, "ensemble")
        models = copy.deepcopy(best_models)
        best_metrics = dict(checkpoint["best_metrics"])
        best_weights = tuple(checkpoint.get("rank_weights", RANK_WEIGHTS))
        ensemble_best = None  # 旧快照的指标先用当前官方验证集复核。
        print("读取已有五模型参数；本次训练轮次从0重新计数。", flush=True)
    else:
        models = {name: ImageModel(name) for name in MODEL_NAMES}
        print("未找到五模型参数，五个分支从随机参数开始；不使用外部预训练权重。", flush=True)
        best_models, ensemble_models = copy.deepcopy(models), copy.deepcopy(models)
        best_metrics, ensemble_best = {}, None
        best_weights = tuple(RANK_WEIGHTS)
    completed = {name: 0 for name in MODEL_NAMES}
    for model in models.values():
        model.to(DEVICE)
    print("并行预读取完整训练集与验证集图片...", flush=True)
    train_dataset.preload()
    valid_dataset.preload()
    weight_dataset = train_dataset.evaluation_view()
    print(f"训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"每轮{len(create_train_loader(train_dataset))}批, CPU线程:{CPU_THREADS}", flush=True)
    print("训练前依次验证五个模型的基线。", flush=True)
    for name in MODEL_NAMES:
        print("训练前单模型验证:", name, flush=True)
        metrics = evaluate(valid_dataset, {name: models[name]}, show_counts=False)
        if is_better(metrics, best_metrics.get(name)):
            best_metrics[name] = metrics
            best_models[name].load_state_dict(models[name].state_dict())
    if MODEL_PATH.is_file():
        print("旧参数保存的排序占比:", flush=True)
        print_rank_weights(best_weights)
        print("验证旧整组快照及其原有排序占比。", flush=True)
        ensemble_best = evaluate(valid_dataset, ensemble_models, show_counts=False,
                                 weights=best_weights)
    print("训练前五模型联合验证。", flush=True)
    baseline = evaluate(valid_dataset, models, show_counts=False, weights=RANK_WEIGHTS)
    print_rank_weights(RANK_WEIGHTS)
    print(f"初始占比验证AUROC:{baseline['auroc']:.6f}", flush=True)
    if is_better(baseline, ensemble_best):
        ensemble_best, best_weights = baseline, tuple(RANK_WEIGHTS)
        for name in MODEL_NAMES:
            ensemble_models[name].load_state_dict(models[name].state_dict())
    save_ensemble(best_models, ensemble_models, MODEL_PATH, ensemble_best,
                  best_metrics, completed_epochs=completed, rank_weights=best_weights)
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
            completed[name] = epoch + 1
        # 【修改】五模型共同预测训练全集，用真实训练标签只调整排序占比，不再反向更新单模参数。
        print(f"第{epoch}轮五模型训练排序占比: {len(weight_dataset)}张训练图片", flush=True)
        train_labels, _, train_scores = collect_probabilities(
            weight_dataset, models, "weight_train", weights=best_weights,
            return_scores=True, show_progress=False)
        candidate_weights, rank_train_metrics = tune_rank_weights(train_labels, train_scores,
                                                                  best_weights)
        print(f"排序占比训练AUROC:{rank_train_metrics['auroc']:.6f}", flush=True)
        print(f"第{epoch}轮五模型联合验证:", flush=True)
        labels, _, scores = collect_probabilities(valid_dataset, models, "valid",
                                                  weights=candidate_weights, return_scores=True)
        combined = calculate_metrics(labels, rank_weighted_probabilities(
            *(scores[name] for name in MODEL_NAMES), weights=candidate_weights))
        print_rank_weights(candidate_weights)
        print(f"验证AUROC:{combined['auroc']:.6f}, AUPRC:{combined['auprc']:.6f}, "
              f"真实阳性:{combined['tp'] + combined['fn']}, 报对阳性:{combined['tp']}, "
              f"误报:{combined['fp']}, 漏报:{combined['fn']}", flush=True)
        accepted = is_better(combined, ensemble_best)
        rank_history.append({"epoch": epoch, "candidate_weights": candidate_weights,
                             "train_auroc": rank_train_metrics["auroc"],
                             "valid_auroc": combined["auroc"], "accepted": accepted})
        if accepted:
            ensemble_best, best_weights = combined, candidate_weights
            for name in MODEL_NAMES:
                ensemble_models[name].load_state_dict(models[name].state_dict())
            print(f"已更新整组最佳, 验证AUROC:{combined['auroc']:.6f}", flush=True)
        else:
            print(f"未超过整组最佳AUROC:{ensemble_best['auroc']:.6f}", flush=True)
        print("本轮结束后保留的五个排序占比:", flush=True)
        print_rank_weights(best_weights)
        save_ensemble(best_models, ensemble_models, MODEL_PATH, ensemble_best,
                      best_metrics, epoch=epoch, completed_epochs=completed,
                      rank_weights=best_weights)
        save_training_record(run_dir, history,
                             {"recipe": RECIPE, "best_metrics": best_metrics,
                              "combined_validation": ensemble_best, "last_epoch": epoch,
                              "completed_epochs": completed, "rank_weights": best_weights,
                              "rank_history": rank_history,
                              "model_training": MODEL_TRAINING,
                              "complete": False})
    save_ensemble(best_models, ensemble_models, MODEL_PATH, ensemble_best,
                  best_metrics, epoch=max(max(completed.values(), default=0) - 1, 0),
                  complete=True, completed_epochs=completed, rank_weights=best_weights)
    size = MODEL_PATH.stat().st_size
    if size > 500_000_000:
        raise ValueError(f"五模型权重{size / 1_000_000:.1f}MB超过500MB限制")
    save_training_record(run_dir, history,
                         {"recipe": RECIPE, "best_metrics": best_metrics,
                          "combined_validation": ensemble_best,
                          "completed_epochs": completed, "rank_weights": best_weights,
                          "rank_history": rank_history,
                          "model_training": MODEL_TRAINING,
                          "model_file": str(MODEL_PATH), "complete": True})
    print(f"训练结束, 最佳联合AUROC:{ensemble_best['auroc']:.6f}, "
          f"AUPRC:{ensemble_best['auprc']:.6f}, "
          f"真实阳性:{ensemble_best['tp'] + ensemble_best['fn']}, "
          f"报对阳性:{ensemble_best['tp']}, 误报:{ensemble_best['fp']}, "
          f"漏报:{ensemble_best['fn']}, "
          f"参数大小:{size / 1_000_000:.1f}MB", flush=True)
    print_rank_weights(best_weights)
    print("训练不会写答案；需要提交CSV时单独运行predict.py。", flush=True)
    return ensemble_best


# 5. 测试：直接运行只训练及验证，不读取无标签test，不写答案。
if __name__ == "__main__":
    print_device_info(DEVICE)
    print_rank_weights()
    print_training_profiles()
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train(*create_dataset())
