"""复用课堂CNN的todo4测试循环；双模型平均阳性概率后生成2026提交CSV。"""

# 导包
import sys
import time

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("预测程序已启动, 正在加载PyTorch...", flush=True)

import torch
from torch.utils.data import DataLoader

from model import MODEL_NAMES
from utils import (BATCH_SIZE, CPU_THREADS, OUTPUT_CSV, PREDICT_CPU_THREADS, TEST_CSV,
                   TEST_DIR, calculate_metrics, load_images, load_model, model_path,
                   print_progress, write_submission)


# 【来源】本项目自写；修改①: 两个模型的阳性概率等权平均，不合并类别标签。
def average_probabilities(*vectors):
    if not vectors or len({len(vector) for vector in vectors}) != 1:
        raise ValueError("模型概率向量数量或长度不一致")
    return [sum(values) / len(vectors) for values in zip(*vectors)]


# 【来源】复用day07 CNN测试DataLoader/无梯度循环；单模型验证和双模型预测共用。
def collect_probabilities(dataset, models, stage):
    if not models:
        raise ValueError("至少需要一个模型")
    torch.set_num_threads(CPU_THREADS if stage == "valid" else PREDICT_CPU_THREADS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    labels, scores, start = [], [[] for _ in models], time.time()
    for model in models:
        model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, 1):
            if dataset.labels is None:
                x = batch
            else:
                x, y = batch
                labels.extend(y.tolist())
            for index, model in enumerate(models):
                scores[index].extend(torch.softmax(model(x), dim=1)[:, 1].tolist())
            print_progress(stage, batch_idx, len(dataloader), start)
    return labels, average_probabilities(*scores)


# todo 4. 模型测试.
# 【来源】复用day07 CNN的evaluate；修改②: 接受一或两个模型，验证AUROC不写CSV。
def evaluate(test_dataset, models):
    if test_dataset.labels is None:
        raise ValueError("验证集必须有真实标签")
    print(f"开始验证: {len(test_dataset)}张有标签图片", flush=True)
    labels, probabilities = collect_probabilities(test_dataset, models, "valid")
    metrics = calculate_metrics(labels, probabilities)
    print(f"Acc: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.6f}, "
          f"AUPRC(梯形): {metrics['auprc']:.6f}, AP: {metrics['ap']:.6f}", flush=True)
    print(f"阳性召回率: {metrics['recall']:.2%}, 特异度: {metrics['specificity']:.2%}, "
          f"TP:{metrics['tp']} TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']}", flush=True)
    return metrics


# 【来源】复用课堂CNN预测循环；修改③: 独立加载两个.pth，只有本函数写提交CSV。
def predict(test_dir=TEST_DIR, output_csv=OUTPUT_CSV):
    torch.set_num_threads(PREDICT_CPU_THREADS)
    paths = [model_path(name) for name in MODEL_NAMES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("缺少模型参数, 请先运行本方案train.py: %s" % path)
    models = [load_model(path) for path in paths]
    print("开始读取test文件夹；测试集没有真实标签, 此阶段生成提交CSV, 不计算准确率.", flush=True)
    test_dataset, image_names = load_images(TEST_CSV, test_dir, labeled=False)
    _, probabilities = collect_probabilities(test_dataset, models, "test")
    output_csv = write_submission(image_names, probabilities, output_csv)
    print("已生成: %s, 共%d张图片, 预测阳性%d张" % (
        output_csv, len(image_names), sum(value >= 0.5 for value in probabilities)), flush=True)
    return output_csv


# 5. 测试: train导入本文件不会预测；右键运行predict.py才写答案。
if __name__ == "__main__":
    predict()
