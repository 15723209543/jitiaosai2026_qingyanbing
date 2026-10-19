"""课堂CNN的todo3：三种CNN各自训练、验证与选模，最后封装为一份pth。"""

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("训练程序已启动, 正在加载PyTorch和数据读取模块...", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ImageModel, create_dataset
from predict import evaluate
from utils import (CPU_THREADS, EPOCHS,
                   MAX_GRAD_NORM, MODEL_NAMES, MODEL_PATH, RECORD_DIR,
                   RECIPE, WEIGHT_DECAY,
                   get_learning_rate, load_ensemble, print_progress, save_ensemble,
                   save_training_record)


SEED = 0
TRAIN_BATCH_SIZE = 64


# 【来源】本项目自写；各模型只比较自己的验证AUROC，严格提高才更新该模型参数。
def is_better(candidate, previous):
    return previous is None or candidate["auroc"] > previous["auroc"]


# 【来源】本项目自写；未有最佳验证记录的模型先跑首轮；修改：0轮跳过，各模型按自身轮数结束。
def training_order(best_metrics, rounds_by_model):
    untrained = [name for name in MODEL_NAMES
                 if rounds_by_model[name] > 0 and name not in best_metrics]
    for name in untrained:
        yield 1, name
    for epoch in range(1, max(rounds_by_model.values(), default=0) + 1):
        for name in MODEL_NAMES:
            if epoch <= rounds_by_model[name] and (epoch != 1 or name not in untrained):
                yield epoch, name


# 【来源】复用课堂CNN的DataLoader洗牌；修改：每轮遍历官方训练集全部图片一次。
def create_train_loader(train_dataset):
    return DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


# 【来源】由原25%阳性抽样改成等效类别加权；每张图片仍在每轮恰好使用一次。
def create_class_weights(labels):
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2)
    if torch.any(counts == 0):
        raise ValueError("训练集必须同时包含0和1")
    return torch.tensor([0.75 / counts[0], 0.25 / counts[1]], dtype=torch.float32)


# 【来源】沿用旧03的历史权重读取；仅用赛方数据训练的03参数作为两种骨干初始值。
def load_bootstrap(models, bootstrap_dir):
    for name in ("resnet18", "densenet121"):
        path = Path(bootstrap_dir) / (name + ".pth")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"]
        if config["model_name"] != name or config["image_size"] != 64:
            raise ValueError("03历史参数结构不匹配: %s" % path)
        models[name].load_state_dict(checkpoint["state_dict"])
        print("载入已独立训练的03参数:", name, flush=True)


# 【来源】沿用课堂CNN的交叉熵正向/反向/更新；只更新当前这一个模型。
def train_step(model, x, y, optimizer, criterion):
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), int((torch.argmax(y_pred, dim=1) == y).sum().item())


# 【来源】沿用旧03的单模型训练循环；修改：每次只跑一轮，最佳参数单独更新对应槽位。
def train_one(name, epoch, models, best_models, train_dataset, valid_dataset,
              best_metrics, run_dir, history, optimizer, dataloader, criterion):
    model = models[name]
    if epoch == 1:
        print("训练前单模型验证:", name, flush=True)
        baseline = evaluate(valid_dataset, {name: model})
        if is_better(baseline, best_metrics.get(name)):
            best_metrics[name] = baseline
            best_models[name].load_state_dict(model.state_dict())
            save_ensemble(best_models, MODEL_PATH, {}, best_metrics=best_metrics)
        print(f"模型:{name}, 训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
              f"共{EPOCHS[name]}轮, 每轮{len(dataloader)}批, CPU线程:{CPU_THREADS}", flush=True)

    lr = get_learning_rate(epoch, EPOCHS[name])
    optimizer.param_groups[0]["lr"] = lr
    total_loss, total_correct, total_samples, start = 0.0, 0, 0, time.time()
    print(f"开始训练 {name} epoch: {epoch}/{EPOCHS[name]}, lr:{lr:.2e}", flush=True)
    for batch_idx, (x, y) in enumerate(dataloader, 1):
        loss, correct = train_step(model, x, y, optimizer, criterion)
        total_loss += loss * len(y)
        total_correct += correct
        total_samples += len(y)
        print_progress(f"{name} epoch: {epoch}/{EPOCHS[name]}", batch_idx,
                       len(dataloader), start,
                       f", loss:{total_loss / total_samples:.5f}, "
                       f"acc:{total_correct / total_samples:.4f}")
    print(f"epoch: {epoch}, loss: {total_loss / total_samples:.5f}, "
          f"acc:{total_correct / total_samples:.4f}, time:{time.time() - start:.2f}s", flush=True)
    metrics = evaluate(valid_dataset, {name: model})
    history.append({"model": name, "epoch": epoch,
                    "train_loss": total_loss / total_samples,
                    "train_accuracy": total_correct / total_samples,
                    "learning_rate": lr, **metrics})
    if is_better(metrics, best_metrics.get(name)):
        best_metrics[name] = metrics
        best_models[name].load_state_dict(model.state_dict())
        save_ensemble(best_models, MODEL_PATH, {}, epoch=epoch, best_metrics=best_metrics)
        print(f"已更新{name}独立最佳参数, 验证AUROC:{metrics['auroc']:.6f}", flush=True)
    else:
        print(f"未超过{name}最佳验证AUROC:{best_metrics[name]['auroc']:.6f}", flush=True)
    save_training_record(run_dir, history,
                         {"recipe": RECIPE, "selection_metric": "individual_validation_auroc",
                          "best_metrics": best_metrics, "complete": False})


# todo 3. 模型训练：三个网络各自更新，全部完成后才评估三模型中位数。
# 【来源】沿用课堂CNN的train入口；修改：三组参数统一封装为单个best_model.pth。
def train(train_dataset, valid_dataset, bootstrap_dir=None):
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    run_dir = RECORD_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_three_models")
    run_dir.mkdir(parents=True, exist_ok=False)
    history = []
    if MODEL_PATH.is_file():
        if bootstrap_dir is not None:
            raise ValueError("已有submission3参数时不能再使用bootstrap_dir")
        best_models, checkpoint = load_ensemble(MODEL_PATH)
        models = copy.deepcopy(best_models)
        best_metrics = dict(checkpoint.get("best_metrics", {}))
        print("从已有单文件三模型参数继续分别训练.", flush=True)
    else:
        models = {name: ImageModel(name) for name in MODEL_NAMES}
        best_metrics = {}
        if bootstrap_dir is not None:
            load_bootstrap(models, bootstrap_dir)
        else:
            print("无历史参数, 三种CNN均从随机初始化开始训练.", flush=True)
        best_models = copy.deepcopy(models)
    print("并行预读取完整训练集与验证集图片...", flush=True)
    train_dataset.preload()
    valid_dataset.preload()
    dataloaders = {name: create_train_loader(train_dataset) for name in MODEL_NAMES}
    optimizers = {name: optim.AdamW(models[name].parameters(), weight_decay=WEIGHT_DECAY)
                  for name in MODEL_NAMES}
    criterion = nn.CrossEntropyLoss(weight=create_class_weights(train_dataset.labels))
    for epoch, name in training_order(best_metrics, EPOCHS):
        train_one(name, epoch, models, best_models, train_dataset, valid_dataset,
                  best_metrics, run_dir, history, optimizers[name], dataloaders[name], criterion)

    print("三模型中位数联合验证:", " + ".join(MODEL_NAMES), flush=True)
    combined = evaluate(valid_dataset, best_models)
    save_ensemble(best_models, MODEL_PATH, combined, best_metrics=best_metrics, complete=True)
    if MODEL_PATH.stat().st_size > 500_000_000:
        raise ValueError("最终权重超过500MB限制")
    save_training_record(run_dir, history,
                         {"recipe": RECIPE, "selection_metric": "individual_validation_auroc",
                          "best_metrics": best_metrics, "combined_validation": combined,
                          "model_file": str(MODEL_PATH), "complete": True})
    print(f"训练结束, 三模型验证AUROC:{combined['auroc']:.6f}, "
          f"AUPRC:{combined['auprc']:.6f}, 验证预测阳性:{combined['tp'] + combined['fp']}, "
          f"参数大小:{MODEL_PATH.stat().st_size / 1_000_000:.1f}MB", flush=True)
    print("训练不会写答案；需要提交CSV时单独运行predict.py。", flush=True)
    return combined


# 5. 测试：main只训练及验证。首次可选--bootstrap-dir指向旧03双模型参数目录。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-dir", type=Path, default=None)
    args = parser.parse_args()
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train(*create_dataset(), bootstrap_dir=args.bootstrap_dir)
