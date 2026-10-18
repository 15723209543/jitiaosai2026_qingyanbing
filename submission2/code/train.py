"""课堂CNN的todo3：单次损失函数同时训练两个特征分支和共同分类头。"""

# 导包
import sys
import time
from datetime import datetime

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("训练程序已启动, 正在加载PyTorch和数据读取模块...", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ImageModel, create_dataset
from predict import evaluate
from utils import (BACKBONE_LR_FACTOR, BASELINE_DENSE, BASELINE_RESNET, BATCH_SIZE,
                   CPU_THREADS, EARLY_STOPPING_PATIENCE,
                   EPOCHS, MAX_GRAD_NORM, MIN_EPOCHS, MODEL_PATH, RECORD_DIR,
                   SAMPLES_PER_EPOCH, WEIGHT_DECAY, create_sampler, get_learning_rate,
                   initialize_backbones, load_model, print_progress, save_model,
                   save_training_record)


SEED = 0


# 【来源】复用day07 CNN的正向、交叉熵、反向传播和更新权重步骤；两个骨干共用这一次loss。
def train_step(model, x, y, optimizer, criterion=None):
    model.train()
    # 修改②: 头部预热时骨干保持eval，避免BatchNorm统计量被新头扰动。
    if not any(p.requires_grad for p in model.resnet.parameters()):
        model.resnet.eval()
    if not any(p.requires_grad for p in model.densenet.parameters()):
        model.densenet.eval()
    y_pred = model(x)
    loss = (criterion or nn.CrossEntropyLoss())(y_pred, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), int((torch.argmax(y_pred, dim=1) == y).sum().item())


# todo 3. 模型训练.
# 【来源】复用day07 CNN的train循环；修改③: 首训预热2轮，续训从第1轮更新两个骨干。
def train(train_dataset, valid_dataset):
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    resuming = MODEL_PATH.is_file()
    if resuming:
        model = load_model(MODEL_PATH)
        saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        best_auroc = saved["validation"]["auroc"]
        print("从已有融合参数继续联合训练, 只在验证AUROC提高时覆盖.", flush=True)
    else:
        model = ImageModel()
        if BASELINE_RESNET.is_file() and BASELINE_DENSE.is_file():
            model = initialize_backbones(model)
            print("已载入03的两个单模型骨干, 开始联合训练.", flush=True)
        else:
            # 修改⑥: 发布版仅留一份.pth；若用户主动移除它，仍可从官方数据随机初始化重训。
            print("未发现旧骨干权重, 从随机初始化开始联合训练.", flush=True)
        best_auroc = float("-inf")

    run_dir = RECORD_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_feature_fusion")
    run_dir.mkdir(parents=True, exist_ok=False)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            sampler=create_sampler(train_dataset.labels, SAMPLES_PER_EPOCH))
    print(f"融合模型, 训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"共{EPOCHS}轮, 每轮{len(dataloader)}批, CPU线程:{CPU_THREADS}", flush=True)

    # 修改④: 骨干学习率为融合头的1/10，避免已学到的眼底特征被大步长破坏。
    optimizer = optim.AdamW([
        {"params": list(model.resnet.parameters()) + list(model.densenet.parameters())},
        {"params": model.fusion.parameters()},
    ], weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history, stale_epochs = [], 0
    for epoch_idx in range(EPOCHS):
        epoch = epoch_idx + 1
        joint = resuming or epoch > 2
        for parameter in list(model.resnet.parameters()) + list(model.densenet.parameters()):
            parameter.requires_grad_(joint)
        lr = get_learning_rate(epoch)
        optimizer.param_groups[0]["lr"] = lr * BACKBONE_LR_FACTOR
        optimizer.param_groups[1]["lr"] = lr
        total_loss, total_correct, total_samples, start = 0.0, 0, 0, time.time()
        print(f"开始训练 融合模型 epoch: {epoch}/{EPOCHS}, lr:{lr:.2e}, "
              f"骨干联合更新:{'是' if joint else '否'}", flush=True)
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            loss, correct = train_step(model, x, y, optimizer, criterion)
            total_loss += loss * len(y)
            total_correct += correct
            total_samples += len(y)
            print_progress(f"epoch: {epoch}/{EPOCHS}", batch_idx, len(dataloader), start,
                           f", loss:{total_loss / total_samples:.5f}, "
                           f"acc:{total_correct / total_samples:.4f}")
        print(f"epoch: {epoch}, loss: {total_loss / total_samples:.5f}, "
              f"acc:{total_correct / total_samples:.4f}, time:{time.time() - start:.2f}s", flush=True)

        # 修改⑤: 每轮用同一个融合网络验证，只有验证AUROC严格变大才保存唯一pth。
        metrics = evaluate(valid_dataset, model)
        history.append({"epoch": epoch, "train_loss": total_loss / total_samples,
                        "train_accuracy": total_correct / total_samples, "learning_rate": lr,
                        "joint_backbones": joint, **metrics})
        if metrics["auroc"] > best_auroc:
            best_auroc, stale_epochs = metrics["auroc"], 0
            save_model(model, MODEL_PATH, metrics, epoch)
            print(f"已保存最佳融合模型: {MODEL_PATH}, 验证AUROC:{best_auroc:.6f}", flush=True)
        else:
            stale_epochs += 1
            print(f"本轮未超过最佳验证AUROC:{best_auroc:.6f}, 保留最佳参数", flush=True)
        save_training_record(run_dir, history, "resnet18_densenet121_feature_fusion")
        if epoch >= MIN_EPOCHS and stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"连续{EARLY_STOPPING_PATIENCE}轮AUROC未提高, 提前结束训练", flush=True)
            break
    print(f"融合训练结束, 最佳验证AUROC:{best_auroc:.6f}, 训练记录:{run_dir}", flush=True)
    print("需要生成提交答案时, 请单独运行predict.py。", flush=True)
    return best_auroc


# 5. 测试: 点击运行train.py只训练并验证，不写无标签测试集答案。
if __name__ == "__main__":
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train(*create_dataset())
