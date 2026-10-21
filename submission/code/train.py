"""复用课堂CNN的todo3训练循环；每个单模型独立训练，最后验证等权双模型。"""

# 导包
import sys
import time
import shutil
from datetime import datetime

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("训练程序已启动, 正在加载PyTorch和数据读取模块...", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ImageModel, MODEL_NAMES, create_dataset
from predict import evaluate
from utils import (BATCH_SIZE, CPU_THREADS, EARLY_STOPPING_PATIENCE, EPOCHS,
                   LEARNING_RATE, MAX_GRAD_NORM, MIN_EPOCHS, MODEL_DIR, RECORD_DIR,
                   SAMPLES_PER_EPOCH,
                   WEIGHT_DECAY, create_sampler, get_learning_rate, has_model, load_model,
                   model_path, print_progress, save_model, save_training_record)


# 修改①: 已有参数默认不重复训练；确需重训时改为True，旧参数先备份且只被更高AUROC替换。
RETRAIN_EXISTING = False
SEED = 0


# todo 3. 模型训练.
# 【来源】复用day07 CNN的train循环；修改②: 每次只训练一个网络，AUROC严格提高才保存。
def train_one(model_name, train_dataset, valid_dataset):
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    path = model_path(model_name)
    run_dir = RECORD_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + model_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    # 1. 修改⑤: 每轮均衡抽取8192张，控制CPU训练耗时；每批仍为32张。
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            sampler=create_sampler(train_dataset.labels, SAMPLES_PER_EPOCH))
    print(f"模型:{model_name}, 训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"共{EPOCHS}轮, 每轮{len(dataloader)}批, CPU线程:{CPU_THREADS}", flush=True)

    # 2. 创建模型；如已有参数，只作为成绩基线，不把不同训练过程混合进权重。
    model = ImageModel(model_name)
    best_metrics, history = None, []
    if has_model(model_name):
        shutil.copy2(path, run_dir / "baseline_model.pth")
        print("已备份现有参数, 先验证旧模型...", flush=True)
        best_metrics = evaluate(valid_dataset, [load_model(path, model_name)])
        history.append({"epoch": 0, **best_metrics})

    # 3. 交叉熵训练；均衡采样后不再叠加类别权重。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=get_learning_rate(1),
                            weight_decay=WEIGHT_DECAY)
    candidate_best_auroc, stale_epochs = float("-inf"), 0

    # 4. 循环遍历epoch；每轮显示课堂loss、acc、time，再做独立验证。
    for epoch_idx in range(EPOCHS):
        total_loss, total_correct, total_samples, start = 0.0, 0, 0, time.time()
        learning_rate = get_learning_rate(epoch_idx + 1)
        optimizer.param_groups[0]["lr"] = learning_rate
        print(f"开始训练 {model_name} epoch: {epoch_idx + 1}/{EPOCHS}, lr:{learning_rate:.2e}", flush=True)
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
            optimizer.step()
            total_correct += (torch.argmax(y_pred, dim=1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            print_progress(f"epoch: {epoch_idx + 1}/{EPOCHS}", batch_idx, len(dataloader), start,
                           f", loss:{total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.4f}")
        print(f"epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, "
              f"acc:{total_correct / total_samples:.4f}, time:{time.time() - start:.2f}s", flush=True)

        # 5. 修改③: 只用本模型独立验证AUROC决定权重是否更新。
        metrics = evaluate(valid_dataset, [model])
        if metrics["auroc"] > candidate_best_auroc:
            candidate_best_auroc, stale_epochs = metrics["auroc"], 0
            save_model(model, run_dir / "candidate_model.pth", metrics, epoch_idx + 1)
        else:
            stale_epochs += 1
        history.append({"epoch": epoch_idx + 1, "train_loss": total_loss / total_samples,
                        "train_accuracy": total_correct / total_samples,
                        "learning_rate": learning_rate, **metrics})
        if best_metrics is None or metrics["auroc"] > best_metrics["auroc"]:
            best_metrics = metrics
            save_model(model, path, metrics, epoch_idx + 1)
            print(f"已保存最佳模型: {path}, 验证AUROC:{metrics['auroc']:.6f}", flush=True)
        else:
            print(f"本轮未超过最佳验证AUROC:{best_metrics['auroc']:.6f}, 保留最佳参数", flush=True)
        save_training_record(run_dir, history, model_name)
        if epoch_idx + 1 >= MIN_EPOCHS and stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"{model_name}连续{EARLY_STOPPING_PATIENCE}轮AUROC未提高, 提前结束训练", flush=True)
            break
    print(f"{model_name}训练结束, 保留验证AUROC:{best_metrics['auroc']:.6f}, "
          f"训练记录:{run_dir}", flush=True)
    return path


# 【来源】复用课堂train入口；修改④: 两个网络分开训练或复用，最后只验证组合，不写答案。
def train(train_dataset, valid_dataset):
    torch.set_num_threads(CPU_THREADS)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in MODEL_NAMES:
        path = model_path(model_name)
        if has_model(model_name) and not RETRAIN_EXISTING:
            print(f"已有参数, 跳过重复训练: {path}", flush=True)
        else:
            train_one(model_name, train_dataset, valid_dataset)
        print(f"单模型验证: {model_name}", flush=True)
        evaluate(valid_dataset, [load_model(path, model_name)])
    print("双模型等权平均验证:", " + ".join(MODEL_NAMES), flush=True)
    metrics = evaluate(valid_dataset, [load_model(model_path(name), name) for name in MODEL_NAMES])
    print(f"双模型验证AUROC:{metrics['auroc']:.6f}, AUPRC:{metrics['auprc']:.6f}", flush=True)
    print("训练与验证结束。需要生成提交答案时, 请单独运行predict.py。", flush=True)
    return metrics


# 5. 测试: 点击运行train.py只训练并验证，不运行无标签测试集预测。
if __name__ == "__main__":
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train(*create_dataset())
