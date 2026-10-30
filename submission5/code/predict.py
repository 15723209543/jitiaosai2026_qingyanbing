"""使用 submission5 已完成的检查点预测官方 test，生成三列提交 CSV。"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import MODEL_NAMES, create_model
from utils import (FundusDataset, MODEL_PATH, NUM_WORKERS, OUTPUT_CSV,
                   inference_model_names, load_resume_checkpoint, print_progress,
                   read_records, write_submission)


def load_inference_models(checkpoint_path=MODEL_PATH):
    """只载入已完成选模的本实验权重；不会下载或加载其他实验权重。"""
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("尚未生成 submission5 最终权重: " + str(checkpoint_path))
    checkpoint = load_resume_checkpoint(checkpoint_path, MODEL_NAMES)
    chosen = inference_model_names(checkpoint, MODEL_NAMES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    for name in chosen:
        model, _ = create_model(name)
        model.load_state_dict(checkpoint["states"][name], strict=True)
        models.append(model.to(device).eval())
    sampling = checkpoint.get("sampling") or {}
    print("推理设备:", device, "固定方案:", checkpoint["selection"]["name"],
          "抽样模式:", sampling.get("mode", "未知"), "FP32, 无TTA", flush=True)
    return models


def score_batch(images, models):
    """预测和 cost.py 共用的 FP32 概率计算。"""
    if not models:
        raise ValueError("没有可用模型")
    device = next(models[0].parameters()).device
    with torch.inference_mode():
        images = images.to(device, non_blocking=False)
        probabilities = [torch.sigmoid(model(images).flatten()) for model in models]
        return torch.stack(probabilities).mean(dim=0).cpu().tolist()


def predict(output_csv=OUTPUT_CSV, checkpoint_path=MODEL_PATH):
    """仅读取官方无标签 test；不在测试集调参。"""
    models = load_inference_models(checkpoint_path)
    records = read_records("test")
    loader = DataLoader(FundusDataset(records, training=False), batch_size=1,
                        shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False)
    scores, started = [], time.time()
    for index, images in enumerate(loader, 1):
        scores.extend(score_batch(images, models))
        print_progress("test", index, len(loader), started)
    result = write_submission(records, scores, output_csv)
    print(f"已输出:{result}, 测试图片:{len(records)}, 预测阳性:{sum(x >= 0.5 for x in scores)}",
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从已完成的 submission5 权重生成提交 CSV")
    parser.add_argument("output_csv", nargs="?", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    predict(args.output_csv, args.checkpoint)
