"""仅用已完成的官方验证最佳权重，对官方 test 单视图生成提交 CSV。"""

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import NAME, create_model
from train import RECIPE, score_batch
from utils import MODEL_PATH, ROOT, FundusDataset, load_checkpoint, read_records


def load_model(path=MODEL_PATH):
    """【复用并修改】submission4/code/predict.py:load_model；核对模型配方及官方验证来源。"""
    checkpoint = load_checkpoint(path)
    if (checkpoint.get("recipe") != RECIPE or checkpoint.get("model_name") != NAME or
            checkpoint.get("metric_scope") != "official_valid" or
            checkpoint.get("training_complete") is not True):
        raise ValueError("这不是 submission_new 已通过官方验证的完整权重")
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), checkpoint, device


def predict(output=ROOT / "submission.csv", checkpoint_path=MODEL_PATH, workers=2):
    """【复用并修改】submission4/code/predict.py:predict；按官方 test.csv 顺序写三列。"""
    model, checkpoint, device = load_model(checkpoint_path)
    print(f"预测设备:{device}; 最佳官方验证AUROC:{checkpoint['official_valid_auroc']:.6f}", flush=True)
    records = read_records("test")
    loader = DataLoader(FundusDataset(records), batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=device == "cuda")
    output = Path(output)
    if output.suffix != ".csv":
        raise ValueError("提交文件后缀必须为小写 .csv")
    temporary = output.with_suffix(".csv.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for images in loader:
            probability = float(score_batch(images, model, device)[0])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("预测概率非法")
            probability = round(probability, 10)
            name = str(Path(records[count][0]).with_suffix(".jpg"))
            writer.writerow([name, f"{probability:.10f}", int(probability >= 0.5)])
            count += 1
            if count == 1 or count % 500 == 0 or count == len(records):
                print(f"test, image:{count}/{len(records)}", flush=True)
    if count != len(records):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("测试集未完整预测")
    temporary.replace(output)
    print(f"已生成:{output}, 图片:{count}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    predict(args.output, args.checkpoint, args.workers)
