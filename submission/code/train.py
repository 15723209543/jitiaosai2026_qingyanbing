"""
案例:
    复用课堂 CNN 的训练逻辑, 训练青光眼二分类模型.
"""

# 导包
import sys
import time
from datetime import datetime

# 修改①: 深度学习库首次导入也需要时间, 点击运行后先立即显示启动信息.
if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("训练程序已启动, 正在加载PyTorch和数据读取模块...", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ImageModel, create_dataset
from predict import evaluate
from utils import (
    BATCH_SIZE, CPU_THREADS, EARLY_STOPPING_PATIENCE, EPOCHS,
    FINETUNE_LEARNING_RATE, LEARNING_RATE, LR_PATIENCE, MAX_GRAD_NORM,
    MIN_LEARNING_RATE, MODEL_PATH, RESUME, TRAINING_RECORD_DIR, WEIGHT_DECAY,
    create_sampler, load_model, print_progress, save_model,
    save_training_record, transfer_model_weights,
)


# todo 3. 模型训练.
# 【来源】复用day07 CNN的train循环；v1.4.3自写分组学习率、梯度截断及严格AUROC选模.
def train(train_dataset, valid_dataset):
    # 优化①: 参考day06 ANN优化版固定随机数种子.
    torch.manual_seed(0)
    torch.set_num_threads(CPU_THREADS)

    # 1. 创建数据加载器.
    # 修改②: 保留每批64张、均衡采样；每轮抽样len(train_dataset)次, 不删减训练图片.
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=create_sampler(train_dataset.labels))
    print(f"训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"共{EPOCHS}轮, 每轮{len(dataloader)}批, CPU线程:{CPU_THREADS}", flush=True)

    # 2. 创建模型对象.
    model = ImageModel()
    print("本轮训练模型: 5层卷积32→64→128→256→512, 全连接4608→512→2", flush=True)
    best_metrics, best_epoch, history = None, 0, []
    best_model_config = model.model_config.copy()
    backbone_learning_rate = LEARNING_RATE
    run_dir = TRAINING_RECORD_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    if MODEL_PATH.is_file():
        # 修改③: 无论是否续训都先验证旧冠军, 避免从头训练时被较差的新模型覆盖.
        previous_model = load_model()
        print("已加载现有参数, 先验证训练前模型...", flush=True)
        best_metrics = evaluate(valid_dataset, previous_model)
        best_model_config = previous_model.model_config.copy()
        history.append({"epoch": 0, **best_metrics})
        if RESUME:
            # 修改⑧: 更换分类头只继承匹配的卷积；相同精简结构则恢复全部权重.
            transfer_model_weights(model, previous_model)
            backbone_learning_rate = FINETUNE_LEARNING_RATE
        del previous_model
        save_training_record(history, best_metrics, best_epoch, run_dir, best_model_config, model.model_config)

    # 3. 创建损失函数对象.
    # 修改④: 已均衡采样, 使用课堂普通交叉熵；AUROC仅用于验证选模, 不当作loss.
    criterion = nn.CrossEntropyLoss()

    # 4. 创建优化器对象.
    # 修改⑨: 已训练卷积用1e-5微调, 新分类头用1e-4学习；从头训练时两组均为1e-4.
    optimizer = optim.Adam([
        {"params": [p for name, p in model.named_parameters() if not name.startswith("classifier.")],
         "lr": backbone_learning_rate},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE,
        threshold=0.0, threshold_mode="abs", min_lr=MIN_LEARNING_RATE,
    )
    # 修改⑩: 调度/早停比较候选自身AUROC；磁盘冠军只决定是否替换正式模型.
    candidate_best_auroc = float("-inf")
    epochs_without_improvement = 0

    # 5. 循环遍历epoch, 开始每轮的训练动作.
    epochs = EPOCHS
    for epoch_idx in range(epochs):
        # 5.1 定义变量, 记录损失、样本数、正确数和开始时间.
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        learning_rate, head_learning_rate = [group["lr"] for group in optimizer.param_groups]
        print(f"开始训练 epoch: {epoch_idx + 1}/{epochs}, "
              f"lr:{learning_rate:.2e}, head_lr:{head_learning_rate:.2e}", flush=True)

        # 5.2 遍历数据加载器, 获取每批次的数据.
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            # 5.2.1 切换训练模式.
            model.train()
            # 5.2.2 模型预测.
            y_pred = model(x)
            # 5.2.3 计算损失.
            loss = criterion(y_pred, y)
            # 5.2.4 梯度清零 + 反向传播 + 参数更新.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 修改⑪: 全部梯度L2范数超过5才缩小；出现NaN/Inf立即报错, 不保存坏参数.
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
            optimizer.step()
            # 5.2.5 统计本轮累计损失和准确率, 首批/每25批/末批显示进度.
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            print_progress(
                f"epoch: {epoch_idx + 1}/{epochs}", batch_idx, len(dataloader), start,
                f", loss:{total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.4f}"
            )

        # 5.2.6 保留课堂CNN的epoch、loss、acc和time输出.
        print(f"epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, "
              f"acc:{total_correct / total_samples:.4f}, time:{time.time() - start:.2f}s", flush=True)

        # 6. 修改⑫: AUROC是唯一选模标准, 相等不替换；ACC和AUPRC都只展示.
        metrics = evaluate(valid_dataset, model)
        if metrics["auroc"] > candidate_best_auroc:
            candidate_best_auroc = metrics["auroc"]
            epochs_without_improvement = 0
            # 本轮自己的最佳也保留, 即使尚未超过旧冠军, 不再只留下CSV指标.
            save_model(model, run_dir / "best_model.pth", metrics, epoch_idx + 1)
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch_idx + 1, "train_loss": total_loss / total_samples,
                        "train_accuracy": total_correct / total_samples,
                        "learning_rate": learning_rate, "head_learning_rate": head_learning_rate, **metrics})
        if best_metrics is None or metrics["auroc"] > best_metrics["auroc"]:
            best_metrics, best_epoch = metrics, epoch_idx + 1
            best_model_config = model.model_config.copy()
            save_model(model, MODEL_PATH, metrics, best_epoch)
            print(f"已保存最佳模型: {MODEL_PATH}, 验证AUROC:{metrics['auroc']:.6f}", flush=True)
        else:
            print(f"本轮未超过最佳验证AUROC:{best_metrics['auroc']:.6f}, 保留最佳参数", flush=True)
        scheduler.step(metrics["auroc"])
        history[-1]["next_learning_rate"] = optimizer.param_groups[0]["lr"]
        history[-1]["next_head_learning_rate"] = optimizer.param_groups[1]["lr"]
        if optimizer.param_groups[0]["lr"] < learning_rate or optimizer.param_groups[1]["lr"] < head_learning_rate:
            print(f"验证AUROC停滞, 下一轮lr:{optimizer.param_groups[0]['lr']:.2e}, "
                  f"head_lr:{optimizer.param_groups[1]['lr']:.2e}", flush=True)
        save_training_record(history, best_metrics, best_epoch, run_dir, best_model_config, model.model_config)
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"本轮训练模型连续{EARLY_STOPPING_PATIENCE}轮AUROC未提高, 提前结束训练", flush=True)
            break

    print(f"训练完成, 保留模型验证Acc:{best_metrics['accuracy']:.6f}, "
          f"AUROC:{best_metrics['auroc']:.6f}, AUPRC:{best_metrics['auprc']:.6f}, "
          f"阳性召回率:{best_metrics['recall']:.2%}", flush=True)
    print(f"训练记录: {run_dir}", flush=True)


# 5. 测试.
if __name__ == "__main__":
    # 1. 获取数据集.
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train_dataset, valid_dataset = create_dataset()
    # 2. 模型训练, 由用户运行.
    train(train_dataset, valid_dataset)
    # 3. 只有单独运行predict.py才生成答案, train不调用预测入口.
    print("训练与验证结束。需要生成提交答案时, 请单独运行predict.py。", flush=True)
