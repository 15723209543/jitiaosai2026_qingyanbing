"""Measure the same single-view Tiny inference used for submission predictions."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from predict_tiny import load_model
from train_tiny import score_batch
from utils import FundusDataset


def images_in(folder, count):
    paths = sorted((p for p in folder.rglob("*") if p.is_file() and
                    p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: p.name.casefold())
    if len(paths) != count:
        raise ValueError(f"{folder}: expected {count} images, found {len(paths)}")
    return paths


def measure(paths, model, device):
    dataset = FundusDataset([(p.name, p, None) for p in paths])
    timings = []
    for i in range(len(dataset)):
        start = time.perf_counter()
        score_batch(dataset[i].unsqueeze(0), model, device)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pictures100", type=Path, required=True)
    parser.add_argument("--pictures5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, meta, device = load_model(args.checkpoint)
    measure(images_in(args.pictures5, 5), model, device)
    paths = images_in(args.pictures100, 100)
    runs = [measure(paths, model, device) for _ in range(3)]
    seconds = statistics.mean(sum(values) / 3 for values in zip(*runs))
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    record = {
        "method": "single_view_predict_tiny_score_batch",
        "device": device,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "recipe": meta["recipe"],
        "official_valid_auroc": meta["official_valid_auroc"],
        "warmup_images": 5,
        "timed_images": 100,
        "rounds": 3,
        "seconds_per_image": seconds,
        "milliseconds_per_image": seconds * 1000,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
