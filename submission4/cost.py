"""Single-image end-to-end timing using the exact prediction scorer."""

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

from data import FundusDataset
from predict import load_model, score_batch


def image_list(directory, count):
    paths = sorted((p for p in Path(directory).rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: p.name.casefold())
    if len(paths) != count or len({p.name.casefold() for p in paths}) != count:
        raise ValueError(f"{directory}: expected {count} distinct images")
    return paths


def measure(paths, model, output_path):
    dataset = FundusDataset([(path.name, path, None) for path in paths])
    durations = []
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for index, path in enumerate(paths):
            started = time.perf_counter()
            score = round(score_batch(dataset[index].unsqueeze(0), model)[0], 6)
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            writer.writerow([path.name, f"{score:.6f}", int(score >= 0.5)])
            file.flush()
            durations.append(time.perf_counter() - started)
    return durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    measured, warmup = image_list(args.pictures_100, 100), image_list(args.pictures_5, 5)
    model, _ = load_model(args.checkpoint)
    with tempfile.TemporaryDirectory() as temporary:
        measure(warmup, model, Path(temporary) / "warmup.csv")
        rounds = [measure(measured, model, Path(temporary) / f"round{i}.csv")
                  for i in range(3)]
    mean = statistics.mean(statistics.mean(values) for values in zip(*rounds))
    print(f"end_to_end_seconds_per_image={mean:.6f} device={next(model.parameters()).device}")
    if mean >= 0.1:
        raise RuntimeError("Competition 0.1-second limit not met on this machine")
