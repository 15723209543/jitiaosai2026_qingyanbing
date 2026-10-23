"""按新规程以batch=1测试端到端耗时；只做固定模型推理，不训练或调参。"""

import argparse
import csv
import statistics
import sys
import tempfile
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_DIR))

import torch

from predict import load_inference_models, score_batch
from utils import ImageDataset, list_image_paths


# 【来源】根据2026新规程第7条自写；每张图片从读取到写出一行结果单独计时。
def benchmark_round(image_paths, models, output_path):
    scores, durations = {}, []
    dataset = ImageDataset(image_paths)
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for index, image_path in enumerate(image_paths):
            start = time.perf_counter()
            image = dataset[index].unsqueeze(0)
            score = score_batch(image, models)[0]
            if next(models[0].parameters()).device.type == "cuda":
                torch.cuda.synchronize()
            writer.writerow([image_path.name, "%.6f" % score, int(score >= 0.5)])
            file.flush()
            durations.append(time.perf_counter() - start)
            scores[image_path.name] = score
    return scores, durations


# 【来源】根据2026新规程第7条自写；核对同一份固定模型的最终提交结果。
def compare_submission(submission_csv, scores):
    with Path(submission_csv).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Image_name", "Probability", "Pred_Label"]:
            raise ValueError("提交CSV字段必须为 Image_name,Probability,Pred_Label")
        expected = {row["Image_name"]: row for row in reader}
    mismatches = 0
    for name, score in scores.items():
        row = expected.get(name)
        if row is None or abs(float(row["Probability"]) - round(score, 6)) > 0.000001 \
                or int(row["Pred_Label"]) != int(score >= 0.5):
            mismatches += 1
    return mismatches


# 【来源】根据2026新规程第7条自写；5张预热，100张连续跑3轮，先逐图平均再全体平均。
def main(pictures_100, pictures_5, submission_csv):
    measured_paths, _ = list_image_paths(pictures_100)
    warmup_paths, _ = list_image_paths(pictures_5)
    if len(measured_paths) != 100 or len(warmup_paths) != 5:
        raise ValueError("须提供100张计时图和5张预热图")
    if not Path(submission_csv).is_file():
        raise FileNotFoundError("先用当前模型生成最终submission.csv，再测一致性: %s" % submission_csv)
    models = load_inference_models()
    device = next(models[0].parameters()).device
    print("推理设备:%s, 精度:fp32, batch=1" % device, flush=True)
    with tempfile.TemporaryDirectory() as directory:
        benchmark_round(warmup_paths, models, Path(directory) / "warmup.csv")
        rounds = []
        for index in range(3):
            scores, durations = benchmark_round(measured_paths, models,
                                                 Path(directory) / ("round%d.csv" % (index + 1)))
            rounds.append(durations)
            print("第%d轮: %.6f秒/张" % (index + 1, statistics.mean(durations)), flush=True)
    per_image = [statistics.mean(round_data[index] for round_data in rounds)
                 for index in range(len(measured_paths))]
    average_seconds = statistics.mean(per_image)
    mismatches = compare_submission(submission_csv, scores)
    print("[CONCLUSION] 本机100张、batch=1串行、3轮；单张端到端平均耗时 %.6f 秒，"
          "%s100秒上限；与submission.csv比对不一致%d张；推理精度fp32。"
          "最终以主办方指定硬件复测为准。" %
          (average_seconds, "超过" if average_seconds > 100 else "未超过", mismatches), flush=True)
    if mismatches:
        raise ValueError("计时预测与最终提交CSV不一致")
    return average_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新规程端到端计时与最终CSV一致性检查")
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--submission-csv", type=Path,
                        default=Path(__file__).resolve().parent / "submission.csv")
    args = parser.parse_args()
    main(args.pictures_100, args.pictures_5, args.submission_csv)
