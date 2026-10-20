"""课堂CNN的todo4：五模型概率按排名加权，生成提交CSV。"""

import sys
import time

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("预测程序已启动, 正在加载PyTorch...", flush=True)

import torch
from torch.utils.data import DataLoader

from utils import (BATCH_SIZE, CPU_THREADS, ImageDataset, MODEL_NAMES, MODEL_PATH, OUTPUT_CSV,
                   PREDICT_CPU_THREADS, TEST_CSV, TEST_DIR, VALID_CSV, VALID_DIR, calculate_metrics,
                   load_ensemble, load_images, rank_weighted_probabilities, print_progress,
                   print_rank_weights,
                   write_submission)


# 【来源】沿用submission3的DataLoader/无梯度预测；修改：五路概率排序后加权。
def collect_probabilities(dataset, models, stage):
    if len(models) not in (1, 5):
        raise ValueError("验证须传入一个模型，五模型预测须传入全部五个模型")
    if len(models) == 5 and set(models) != set(MODEL_NAMES):
        raise ValueError("五模型名称不完整")
    torch.set_num_threads(CPU_THREADS if stage == "valid" else PREDICT_CPU_THREADS)
    devices = {name: next(model.parameters(), torch.empty(0)).device
               for name, model in models.items()}
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=any(device.type == "cuda" for device in devices.values()))
    labels, scores, start = [], {name: [] for name in models}, time.time()
    window_positives = window_tp = window_fp = 0
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, 1):
            if dataset.labels is None:
                x = batch
            else:
                x, y = batch
                batch_labels = y.tolist()
                labels.extend(batch_labels)
            batch_scores = {}
            for name, model in models.items():
                inputs = x.to(devices[name], non_blocking=True)
                batch_scores[name] = torch.softmax(model(inputs), dim=1)[:, 1].cpu().tolist()
                scores[name].extend(batch_scores[name])
            detail = ""
            if stage == "valid":
                # 【修改】五模型仍按提交时同一排序加权规则计算；这里只累计本次日志区间。
                batch_probabilities = (next(iter(batch_scores.values())) if len(models) == 1 else
                                       rank_weighted_probabilities(*(batch_scores[name] for name in MODEL_NAMES)))
                window_positives += sum(label == 1 for label in batch_labels)
                window_tp += sum(label == 1 and probability >= 0.5
                                 for label, probability in zip(batch_labels, batch_probabilities))
                window_fp += sum(label == 0 and probability >= 0.5
                                 for label, probability in zip(batch_labels, batch_probabilities))
                detail = (f", 真实阳性:{window_positives}, 报对阳性:{window_tp}, "
                          f"误报:{window_fp}")
            if print_progress(stage, batch_idx, len(dataloader), start, detail) and stage == "valid":
                window_positives = window_tp = window_fp = 0
    if len(models) == 1:
        return labels, next(iter(scores.values()))
    return labels, rank_weighted_probabilities(*(scores[name] for name in MODEL_NAMES))


# todo 4. 模型验证；有真实标签才计算AUROC，绝不写答案CSV。
# 【来源】沿用submission3的evaluate；分别支持单模型与五模型联合验证。
def evaluate(valid_dataset, models, show_counts=True):
    if valid_dataset.labels is None:
        raise ValueError("验证集必须有真实标签")
    print(f"开始验证: {len(valid_dataset)}张有标签图片", flush=True)
    labels, probabilities = collect_probabilities(valid_dataset, models, "valid")
    metrics = calculate_metrics(labels, probabilities)
    print(f"AUROC: {metrics['auroc']:.6f}, "
          f"AUPRC(梯形): {metrics['auprc']:.6f}, AP: {metrics['ap']:.6f}", flush=True)
    # 【修改】训练中的各次验证只打印选模所需指标；最终总计由train.py统一输出。
    if show_counts:
        print(f"真实阳性:{metrics['tp'] + metrics['fn']}, 报对阳性:{metrics['tp']}, "
              f"误报:{metrics['fp']}, 漏报:{metrics['fn']}, "
              f"阳性召回率:{metrics['recall']:.2%}, 特异度:{metrics['specificity']:.2%}", flush=True)
    return metrics


# 【来源】沿用submission3的predict入口；只加载单文件整组最佳快照，单独运行才写CSV。
def predict(test_dir=TEST_DIR, output_csv=OUTPUT_CSV):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("缺少五模型参数, 请先运行train.py: %s" % MODEL_PATH)
    models, checkpoint = load_ensemble(MODEL_PATH)
    if not checkpoint["complete"]:
        raise ValueError("五模型尚未完成训练，不能生成提交CSV")
    # 【修改】五模型预测优先使用GPU；无CUDA时自动回退CPU，不更改训练权重。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models.values():
        # 【修改】将模型移动到预测设备，不更改训练权重。
        model.to(device)
    print("预测设备:", device, flush=True)
    # 【修改】先用有标签官方验证集打印训练时相同的指标；仅随后预测无标签测试集并写CSV。
    valid_paths, valid_labels = load_images(VALID_CSV, VALID_DIR)
    evaluate(ImageDataset(valid_paths, valid_labels), models)
    print("开始读取test文件夹；测试集没有真实标签, 不计算准确率.", flush=True)
    test_dataset, image_names = load_images(TEST_CSV, test_dir, labeled=False)
    _, probabilities = collect_probabilities(test_dataset, models, "test")
    output_csv = write_submission(image_names, probabilities, output_csv)
    print("已生成: %s, 共%d张图片, 预测阳性%d张" % (
        output_csv, len(image_names), sum(value >= 0.5 for value in probabilities)), flush=True)
    return output_csv


# 5. 测试：右键运行predict.py才写答案；train.py导入本文件不会预测无标签测试集。
if __name__ == "__main__":
    print_rank_weights()
    predict()
