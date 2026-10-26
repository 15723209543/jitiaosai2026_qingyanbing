"""按规程：同一推理函数，5张预热，100张单张串行计时三轮。"""

import argparse
import csv
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import torch
from predict import load_inference_models, score_batch
from utils import FundusDataset


def list_images(directory, count):
    """【改自旧 cost.list_image_paths】仅从指定计时目录取得确切数量的图片。"""
    paths = sorted((p for p in Path(directory).rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: p.name.casefold())
    if len(paths) != count or len({p.name.casefold() for p in paths}) != count:
        raise ValueError(f"{directory} 应恰好有 {count} 张互不重名的图片")
    return paths


def run_round(paths, models, output_path):
    """【改自旧 cost.benchmark_round】单张从读取图片到落盘 CSV 行完整计时。"""
    dataset = FundusDataset([(p, None) for p in paths])
    durations, scores = [], {}
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for index, path in enumerate(paths):
            start = time.perf_counter()
            score = round(score_batch(dataset[index].unsqueeze(0), models)[0], 6)
            if next(models[0].parameters()).device.type == "cuda":
                torch.cuda.synchronize()
            writer.writerow([path.name, f"{score:.6f}", int(score >= 0.5)])
            file.flush()
            durations.append(time.perf_counter() - start)
            scores[path.name] = score
    return durations, scores


def main(pictures_100, pictures_5):
    """【改自旧 cost.main】逐图三轮取均值；官方指定硬件需另行复测。"""
    measured = list_images(pictures_100, 100)
    warmup = list_images(pictures_5, 5)
    models = load_inference_models()
    with tempfile.TemporaryDirectory() as temp:
        run_round(warmup, models, Path(temp) / "warmup.csv")
        rounds = [run_round(measured, models, Path(temp) / f"round{i}.csv")[0]
                  for i in range(1, 4)]
    per_image = [statistics.mean(values) for values in zip(*rounds)]
    mean = statistics.mean(per_image)
    print("端到端平均: %.6f秒/张, 测试设备:%s, FP32" %
          (mean, next(models[0].parameters()).device), flush=True)
    if mean >= 100:
        raise RuntimeError("超过规程 100 秒/张限制")
    return mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    args = parser.parse_args()
    main(args.pictures_100, args.pictures_5)
