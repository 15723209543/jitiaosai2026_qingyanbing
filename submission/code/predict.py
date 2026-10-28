"""固定最佳推理方案：只读取已训练 PTH，预测官方 test 后生成三列 CSV。"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import MODEL_NAMES, create_model
from utils import (BATCH_SIZE, FundusDataset, MODEL_PATH, OUTPUT_CSV,
                   inference_model_names, load_resume_checkpoint, metrics, print_metrics, print_progress,
                   read_records, write_submission)


def load_inference_models():
    """【改自旧 predict.load_inference_models】仅加载验证选中的分支，不下载预训练权重。"""
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("尚未生成训练完成的模型: " + str(MODEL_PATH))
    checkpoint = load_resume_checkpoint(MODEL_PATH, MODEL_NAMES)
    chosen = inference_model_names(checkpoint, MODEL_NAMES)
    selected = checkpoint["selection"]["name"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    for name in chosen:
        model, _ = create_model(name)
        model.load_state_dict(checkpoint["states"][name], strict=True)
        models.append(model.to(device).eval())
    print("推理设备:", device, "固定方案:", selected, "FP32, 无TTA", flush=True)
    return models


def score_batch(images, models):
    """【改自旧 predict.score_batch】概率先平均，不能对二值标签投票。"""
    if not models:
        raise ValueError("没有可用模型")
    device = next(models[0].parameters()).device
    with torch.inference_mode():
        x = images.to(device, non_blocking=False)
        scores = [torch.sigmoid(model(x).flatten()) for model in models]
        return torch.stack(scores).mean(dim=0).cpu().tolist()


def collect(records, models, stage):
    """【改自课堂 CNN 测试循环】验证和预测用同一套无随机增强的推理。"""
    dataset = FundusDataset(records, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    labels, scores, started = [], [], time.time()
    for index, batch in enumerate(loader, 1):
        if stage == "valid":
            images, targets = batch
            labels.extend(targets.tolist())
        else:
            images = batch
        scores.extend(score_batch(images, models))
        print_progress(stage, index, len(loader), started)
    return labels, scores


def predict(output_csv=OUTPUT_CSV):
    """【改自旧 predict.predict】先只读验证展示指标，再按官方 test.csv 原顺序写结果。"""
    models = load_inference_models()
    valid_records = read_records("valid")
    labels, validation = collect(valid_records, models, "valid")
    print_metrics("固定模型验证", metrics(labels, validation))
    test_records = read_records("test")
    _, scores = collect(test_records, models, "test")
    result = write_submission(test_records, scores, output_csv)
    print(f"已输出:{result}, 测试图片:{len(test_records)}, 预测阳性:{sum(x >= 0.5 for x in scores)}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证后生成可提交的 submission.csv；不训练")
    parser.add_argument("output_csv", nargs="?", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    predict(args.output_csv)
