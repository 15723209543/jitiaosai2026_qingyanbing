"""复用课堂CNN的todo3训练循环；每个单模型独立训练，最后验证等权双模型。"""

# 导包
import sys
import time
import math
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
from utils import (BATCH_SIZE, CPU_THREADS, EPOCHS, MAX_GRAD_NORM, MODEL_DIR, RECORD_DIR,
                   WEIGHT_DECAY, get_learning_rate, has_model, load_model,
                   model_path, print_progress, save_model, save_training_record)


# 修改①: 已有参数作为256输入的续训起点；每次点击运行都执行五轮全量训练。
SEED = 0


# todo 3. 模型训练.
# 【来源】复用day07 CNN的train循环；修改②: 每次只训练一个网络，AUROC严格提高才保存。
def train_one(model_name, train_dataset, valid_dataset, device):
    torch.manual_seed(SEED)
    path = model_path(model_name)
    run_dir = RECORD_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + model_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    # 1. 修改⑤: 每轮遍历全部训练图片一次，不再只抽取8192张。
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=device.type == "cuda")
    print(f"模型:{model_name}, 训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"共{EPOCHS}轮, 每轮{len(dataloader)}批, 设备:{device}", flush=True)

    # 2. 修改：从已有参数真正续训；输入改为256后，先重测基线，旧64分数不能直接比较。
    existing = has_model(model_name)
    model = (load_model(path, model_name, for_training=True) if existing
             else ImageModel(model_name)).to(device)
    print("训练前验证当前参数(256整图补边)...", flush=True)
    best_metrics = evaluate(valid_dataset, [model])
    history = [{"epoch": 0, **best_metrics}]
    save_model(model, path, best_metrics, 0)
    save_training_record(run_dir, history, model_name)

    # 3. 修改：完整训练集自然比例，交叉熵以类别权重补偿少数阳性。
    positives = sum(int(label) for label in train_dataset.labels)
    negatives = len(train_dataset) - positives
    if not positives or not negatives:
        raise ValueError("训练集必须同时包含阴性和阳性")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        [1.0, math.sqrt(negatives / positives)], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=get_learning_rate(1),
                            weight_decay=WEIGHT_DECAY)
    # 4. 循环遍历epoch；每轮显示课堂loss、acc、time，再做独立验证。
    for epoch_idx in range(EPOCHS):
        total_loss, total_correct, total_samples, start = 0.0, 0, 0, time.time()
        learning_rate = get_learning_rate(epoch_idx + 1)
        optimizer.param_groups[0]["lr"] = learning_rate
        print(f"开始训练 {model_name} epoch: {epoch_idx + 1}/{EPOCHS}, lr:{learning_rate:.2e}", flush=True)
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            model.train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
        history.append({"epoch": epoch_idx + 1, "train_loss": total_loss / total_samples,
                        "train_accuracy": total_correct / total_samples,
                        "learning_rate": learning_rate, **metrics})
        if metrics["auroc"] > best_metrics["auroc"]:
            best_metrics = metrics
            save_model(model, path, metrics, epoch_idx + 1)
            print(f"已保存最佳模型: {path}, 验证AUROC:{metrics['auroc']:.6f}", flush=True)
        else:
            print(f"本轮未超过最佳验证AUROC:{best_metrics['auroc']:.6f}, 保留最佳参数", flush=True)
        save_training_record(run_dir, history, model_name)
    print(f"{model_name}训练结束, 保留验证AUROC:{best_metrics['auroc']:.6f}, "
          f"训练记录:{run_dir}", flush=True)
    return path


# 【来源】复用课堂train入口；修改④: 两个网络分开训练或复用，最后只验证组合，不写答案。
def train(train_dataset, valid_dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"当前使用GPU训练: {torch.cuda.get_device_name(0)}, PyTorch:{torch.__version__}, "
              f"CUDA:{torch.version.cuda}", flush=True)
    else:
        torch.set_num_threads(CPU_THREADS)
        print(f"当前使用CPU训练, PyTorch:{torch.__version__}, 线程:{CPU_THREADS}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in MODEL_NAMES:
        path = model_path(model_name)
        train_one(model_name, train_dataset, valid_dataset, device)
        print(f"单模型验证: {model_name}", flush=True)
        evaluate(valid_dataset, [load_model(path, model_name).to(device)])
    print("双模型等权平均验证:", " + ".join(MODEL_NAMES), flush=True)
    metrics = evaluate(valid_dataset, [load_model(model_path(name), name).to(device) for name in MODEL_NAMES])
    print(f"双模型验证AUROC:{metrics['auroc']:.6f}, AUPRC:{metrics['auprc']:.6f}", flush=True)
    print("训练与验证结束。需要生成提交答案时, 请单独运行predict.py。", flush=True)
    return metrics


# 5. 测试: 点击运行train.py只训练并验证，不运行无标签测试集预测。
if __name__ == "__main__":
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train(*create_dataset())
