"""
案例:
    复用课堂 CNN 的测试逻辑；main负责本地预测并生成2026提交文件.
"""

# 导包
import sys
import time

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    print("预测程序已启动, 正在加载PyTorch...", flush=True)

import torch
from torch.utils.data import DataLoader

from utils import (
    BATCH_SIZE,
    CPU_THREADS,
    OUTPUT_CSV,
    TEST_DIR,
    calculate_metrics,
    load_test_data,
    load_model,
    print_progress,
    write_submission,
)


# todo 4. 模型测试.
# 【来源】复用day07 CNN的evaluate, 新增验证进度及比赛指标, 返回指标字典.
def evaluate(test_dataset, model=None):
    # 1. 创建测试集数据加载器.
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 修改①: 训练时直接验证当前模型；单独调用时才读取已保存的最佳参数.
    torch.set_num_threads(CPU_THREADS)
    if model is None:
        # 修改⑥: 按参数文件自己的配置加载, 兼容旧ReLU和新LeakyReLU模型.
        model = load_model()

    # 3. 切换测试模式并关闭梯度, 保存真实标签和阳性概率.
    labels, positive_probability, start = [], [], time.time()
    print(f"开始验证: {len(test_dataset)}张有标签图片", flush=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            y_pred = model(x)
            labels.extend(y.tolist())
            positive_probability.extend(torch.softmax(y_pred, dim=1)[:, 1].tolist())
            print_progress("valid", batch_idx, len(dataloader), start)

    # 4. 修改②: 保留课堂Acc输出, 同时显示真正能反映阳性检出和概率排序的指标.
    metrics = calculate_metrics(labels, positive_probability)
    print(f"Acc: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.6f}, "
          f"AUPRC(梯形): {metrics['auprc']:.6f}, AP: {metrics['ap']:.6f}", flush=True)
    print(f"阳性召回率: {metrics['recall']:.2%}, 特异度: {metrics['specificity']:.2%}, "
          f"TP:{metrics['tp']} TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']}", flush=True)
    return metrics


# 【来源】复用课堂CNN测试循环改写, 本地读取无标签测试集并生成2026规定CSV.
def predict(test_dir=TEST_DIR, output_csv=OUTPUT_CSV):
    # 1. 修改②: 2026初赛不再由服务器传参, 默认读取glaucoma_first/test和test.csv.
    torch.set_num_threads(CPU_THREADS)
    print("开始读取test文件夹；测试集没有真实标签, 此阶段生成提交CSV, 不计算准确率.", flush=True)
    test_dataset, image_names = load_test_data(test_dir)

    # 2. 创建测试集数据加载器.
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 创建模型对象并加载训练好的参数.
    model = load_model()

    # 4. 遍历本地无标签测试集.
    positive_probability, start = [], time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, x in enumerate(dataloader, 1):
            output = model(x)

            # 修改③: Probability必须是青光眼阳性类别1的概率, 不能取最大类别置信度.
            probability = torch.softmax(output, dim=1)[:, 1]
            positive_probability.extend(probability.tolist())
            print_progress("test", batch_idx, len(dataloader), start)

    # 5. 修改④: 按官方字段生成UTF-8编码的submission.csv.
    output_csv = write_submission(image_names, positive_probability, output_csv)
    print("已生成: %s, 共%d张图片, 预测阳性%d张" % (
        output_csv, len(image_names), sum(value >= 0.5 for value in positive_probability)
    ), flush=True)
    return output_csv


# 5. 测试.
if __name__ == "__main__":
    # 修改⑤: 只有单独运行本文件才执行测试集预测并生成submission.csv；被train导入时不执行.
    predict()
