"""先复核官方有标签验证集，再单独预测 test 并生成可提交 CSV。"""

import time

import torch
from torch.utils.data import DataLoader

from model import FundusNet
from utils import (EVAL_BATCH, MODEL_PATH, NUM_WORKERS, OUTPUT_CSV,
                   FundusDataset, metrics, progress, write_submission)


@torch.no_grad()
def infer(model, dataset, device, stage="valid"):
    """参考课堂 CNN 测试逻辑：eval/no_grad，按原 CSV 顺序返回连续阳性概率。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    probabilities, labels, start = [], [], time.time()
    for batch_index, (full, disc, batch_labels) in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(full.to(device, non_blocking=True),
                           disc.to(device, non_blocking=True))
        probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
        labels.extend(batch_labels.tolist())
        progress(stage, batch_index, len(loader), start)
    return labels, probabilities


def evaluate(model, valid_data, device):
    """自写：使用完整官方 valid，不读取 test 标签；打印 AUROC/AUPRC/报对/误报/漏报。"""
    print(f"开始验证: {len(valid_data)}张有标签图片", flush=True)
    labels, probabilities = infer(model, valid_data, device)
    result = metrics(labels, probabilities)
    print(f"AUROC:{result['auroc']:.6f}, AUPRC:{result['auprc']:.6f}, "
          f"真实阳性:{result['tp'] + result['fn']}, 报对阳性:{result['tp']}, "
          f"误报:{result['fp']}, 漏报:{result['fn']}", flush=True)
    return result


def predict():
    """自写：只读取一份训练完成的 pth；只有此入口才写 results/submission.csv。"""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"缺少参数，请先训练: {MODEL_PATH}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "dual_view_residual_se_v1":
        raise ValueError("模型参数架构不匹配")
    model = FundusNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print("预测设备:", device, "最佳训练轮:", checkpoint["epoch"], flush=True)
    evaluate(model, FundusDataset("valid"), device)
    dataset = FundusDataset("test", labeled=False)
    _, probabilities = infer(model, dataset, device, stage="test")
    path = write_submission(dataset.names, probabilities, OUTPUT_CSV)
    print("已生成:", path, "总数:", len(probabilities),
          "预测阳性:", sum(p >= 0.5 for p in probabilities), flush=True)
    return path


if __name__ == "__main__":
    predict()
