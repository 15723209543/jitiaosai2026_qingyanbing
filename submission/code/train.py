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
    BATCH_SIZE, CPU_THREADS, EARLY_STOPPING_PATIENCE, EPOCHS, IMAGE_SIZE,
    LEARNING_RATE, LR_PATIENCE, MIN_LEARNING_RATE, MODEL_PATH, RESUME,
    TRAINING_RECORD_DIR, WEIGHT_DECAY,
    create_sampler, load_model, print_progress, save_training_record,
)


# todo 3. 模型训练.
# 【来源】复用day07 CNN的train循环；v1.2.0自写改动: 精调、学习率衰减、早停和独立记录.
def train(train_dataset, valid_dataset):
    # 优化①: 参考day06 ANN优化版固定随机数种子.
    torch.manual_seed(0)
    torch.set_num_threads(CPU_THREADS)

    # 1. 创建数据加载器.
    # 修改②: 阳性只有约3.2%, 按类别均衡抽样后不再设置shuffle或额外加权损失.
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=create_sampler(train_dataset.labels))
    print(f"训练集:{len(train_dataset)}张, 验证集:{len(valid_dataset)}张, "
          f"共{EPOCHS}轮, 每轮{len(dataloader)}批, CPU线程:{CPU_THREADS}", flush=True)

    # 2. 创建模型对象.
    model = ImageModel()
    best_metrics, best_epoch, history = None, 0, []
    best_model_config = model.model_config.copy()
    run_dir = TRAINING_RECORD_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    if RESUME and MODEL_PATH.is_file():
        # 修改③: 继续训练已保存的模型权重；Adam状态重新初始化, 不是完全恢复中断点.
        previous_model = load_model()
        print("已加载现有参数, 先验证训练前模型...", flush=True)
        best_metrics = evaluate(valid_dataset, previous_model)
        best_model_config = previous_model.model_config.copy()
        history.append({"epoch": 0, **best_metrics})
        # 修改⑧: 先按旧模型原配置验证, 再把同形状权重交给新激活/失活配置精调.
        model.load_state_dict(previous_model.state_dict())
        del previous_model
        save_training_record(history, best_metrics, best_epoch, run_dir, best_model_config, model.model_config)

    # 3. 创建损失函数对象.
    # 修改④: 已均衡采样, 采用课堂普通交叉熵, 避免对阳性重复加权.
    criterion = nn.CrossEntropyLoss()  # softmax()激活函数 + 损失计算.

    # 4. 创建优化器对象.
    # 优化②: 保留课堂Adam, 学习率减为5e-5；L2权重衰减抑制权重过大.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 优化⑥: 验证AUROC连续2轮没有显著提升时学习率减半, 最低1e-6.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE,
        threshold=1e-4, threshold_mode="abs", min_lr=MIN_LEARNING_RATE,
    )
    if best_metrics is not None:
        scheduler.step(best_metrics["auroc"])
    epochs_without_improvement = 0

    # 5. 循环遍历epoch, 开始每轮的训练动作.
    epochs = EPOCHS
    for epoch_idx in range(epochs):
        # 5.1 定义变量, 记录总损失、总样本数、预测正确数和训练开始时间.
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        learning_rate = optimizer.param_groups[0]["lr"]

        # 5.2 遍历数据加载器, 获取每批次的数据.
        print(f"开始训练 epoch: {epoch_idx + 1}/{epochs}, lr:{learning_rate:.2e}", flush=True)
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            # 5.2.1 切换训练模式.
            model.train()

            # 5.2.2 模型预测.
            y_pred = model(x)

            # 5.2.3 计算损失.
            loss = criterion(y_pred, y)

            # 5.2.4 梯度清零 + 反向传播 + 参数更新.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5.2.5 统计预测正确的样本个数、总损失和总样本个数.
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            # 修改⑤: 首批、每25批及末批显示实时进度, 不再等待整轮结束才输出.
            print_progress(
                f"epoch: {epoch_idx + 1}/{epochs}", batch_idx, len(dataloader), start,
                f", loss:{total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.4f}"
            )

        # 5.2.6 按课堂CNN原格式打印该轮训练信息.
        print(
            f"epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, "
            f"acc:{total_correct / total_samples:.4f}, time:{time.time() - start:.2f}s", flush=True
        )

        # 6. 修改⑥: 每轮调用predict.py验证；先比较AUROC, 相同时比较AUPRC.
        metrics = evaluate(valid_dataset, model)
        history.append({"epoch": epoch_idx + 1, "train_loss": total_loss / total_samples,
                        "train_accuracy": total_correct / total_samples,
                        "learning_rate": learning_rate, **metrics})
        if best_metrics is None or (metrics["auroc"], metrics["auprc"]) > (best_metrics["auroc"], best_metrics["auprc"]):
            best_metrics, best_epoch = metrics, epoch_idx + 1
            best_model_config = model.model_config.copy()
            epochs_without_improvement = 0
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            # 先完整保存临时参数, 再替换最佳参数, 避免写到一半覆盖原模型.
            temporary_path = MODEL_PATH.with_suffix(".tmp")
            # 修改⑨: 权重与实际激活/失活配置放在同一个pth中, 避免配置和权重错配.
            torch.save({"format_version": 2, "state_dict": model.state_dict(),
                        "model_config": best_model_config, "image_size": IMAGE_SIZE}, temporary_path)
            temporary_path.replace(MODEL_PATH)
            print(f"已保存最佳模型: {MODEL_PATH}, AUROC:{metrics['auroc']:.6f}", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"本轮未超过最佳AUROC:{best_metrics['auroc']:.6f}, 保留最佳参数", flush=True)
        scheduler.step(metrics["auroc"])
        history[-1]["next_learning_rate"] = optimizer.param_groups[0]["lr"]
        if optimizer.param_groups[0]["lr"] < learning_rate:
            print(f"验证AUROC停滞, 下一轮lr:{optimizer.param_groups[0]['lr']:.2e}", flush=True)
        save_training_record(history, best_metrics, best_epoch, run_dir, best_model_config, model.model_config)
        # 优化⑦: 不为了凑足12轮持续过拟合；连续5轮没有更佳模型则提前结束.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"连续{EARLY_STOPPING_PATIENCE}轮未改善, 提前结束训练", flush=True)
            break

    print(f"训练完成, 最佳验证AUROC:{best_metrics['auroc']:.6f}, "
          f"阳性召回率:{best_metrics['recall']:.2%}", flush=True)
    print(f"训练记录: {run_dir}", flush=True)


# 5. 测试.
if __name__ == "__main__":
    # 1. 获取数据集.
    print("正在读取glaucoma_first的训练集和官方验证集...", flush=True)
    train_dataset, valid_dataset = create_dataset()

    # 2. 模型训练.
    train(train_dataset, valid_dataset)

    # 3. 修改⑦: 本文件只负责训练和验证；提交答案由用户单独运行predict.py生成.
    print("训练与验证结束。需要生成提交答案时, 请单独运行predict.py。", flush=True)
