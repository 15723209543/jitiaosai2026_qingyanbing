"""仅用已完成的官方验证最佳权重，对官方 test 单视图生成提交 CSV。"""

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_tiny import NAME, create_model
from train_tiny import RECIPE, RECIPE_SFA, score_batch
from utils import ROOT, FundusDataset, load_checkpoint, read_records


def load_model(path):
    """【复用并修改】submission4/code/predict.py:load_model；核对模型配方及官方验证来源。"""
    checkpoint = load_checkpoint(path)
    allowed_recipes = {
        RECIPE: 0.0,
        RECIPE_SFA: 0.05,
        "s23_g03_tiny320_randaugment_n1m4_official_valid_v1": 0.0,
        "s23_g03_tiny320_randaugment_n1m4_low_lr_from_epoch7_v1": 0.0,
        "s23_g03_tiny320_rand_low_lr_equal_soup_official_valid_v1": 0.0,
        "s23_g03_tiny320_rand_sfa_soup_75_25_official_valid_v1": 0.0,
        "s23_g03_tiny320_randaugment_n1m4_sam_rho005_from_epoch7_v1": 0.0,
        "s23_g03_tiny320_randaugment_pairauc_balanced_from_soup_v1": 0.0,
    }
    recipe = checkpoint.get("recipe")
    if (recipe not in allowed_recipes or checkpoint.get("model_name") != NAME or
            checkpoint.get("metric_scope") != "official_valid" or
            checkpoint.get("training_complete") is not True):
        raise ValueError("这不是S23 Tiny已通过官方验证的完整权重")
    noise_std = checkpoint.get("config", {}).get("feature_noise_std", 0.0)
    if noise_std != allowed_recipes[recipe]:
        raise ValueError("Tiny 权重配方与特征扰动参数不一致")
    model = create_model(pretrained=False, feature_noise_std=noise_std)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), checkpoint, device


def predict(output, checkpoint_path, workers=2):
    """【复用并修改】submission4/code/predict.py:predict；按官方 test.csv 顺序写三列。"""
    model, checkpoint, device = load_model(checkpoint_path)
    print(f"预测设备:{device}; 最佳官方验证AUROC:{checkpoint['official_valid_auroc']:.6f}", flush=True)
    records = read_records("test")
    loader = DataLoader(FundusDataset(records), batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=device == "cuda")
    output = Path(output)
    if output.suffix != ".csv":
        raise ValueError("提交文件后缀必须为小写 .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    predict(args.output, args.checkpoint, args.workers)
