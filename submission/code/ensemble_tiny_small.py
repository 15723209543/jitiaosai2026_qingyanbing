"""Fixed 50/50 Tiny and Small ensemble; official valid selects it before test use."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from predict_small import load_model as load_small
from predict_tiny import load_model as load_tiny
from utils import MODEL_PATH, FundusDataset, calculate_metrics, read_records


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_pair(small_path, tiny_path):
    small, small_meta, small_device = load_small(small_path)
    tiny, tiny_meta, tiny_device = load_tiny(tiny_path)
    if small_device != tiny_device:
        raise RuntimeError("Ensemble devices differ")
    return (small, tiny), (small_meta, tiny_meta), small_device


def score(images, models, device):
    """Single deterministic view; shared by validation, prediction and timing."""
    with torch.inference_mode():
        batch = images.to(device, non_blocking=True)
        a = torch.sigmoid(models[0](batch)[0]).float()
        b = torch.sigmoid(models[1](batch)[0]).float()
        return (0.5 * a + 0.5 * b).cpu()


def run_valid(models, device, output, sources, batch_size, workers):
    records = read_records("valid")
    loader = DataLoader(FundusDataset(records), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device == "cuda")
    labels, probabilities = [], []
    for images, targets in loader:
        values = score(images, models, device).tolist()
        probabilities.extend(values)
        labels.extend(targets.int().tolist())
    metrics = calculate_metrics(labels, probabilities)
    record = {"metric_scope": "official_valid", "method": "fixed_equal_tiny_small",
              "rows": len(records), "sources": sources, "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run_test(models, device, output, workers):
    records = read_records("test")
    loader = DataLoader(FundusDataset(records), batch_size=1,
                        shuffle=False, num_workers=workers, pin_memory=device == "cuda")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for index, images in enumerate(loader):
            probability = round(float(score(images, models, device)[0]), 10)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid test probability")
            writer.writerow([Path(records[index][0]).with_suffix(".jpg").name,
                             f"{probability:.10f}", int(probability >= 0.5)])
            if (index + 1) % 500 == 0 or index + 1 == len(records):
                print(f"test {index + 1}/{len(records)}", flush=True)
    temporary.replace(output)
    print(f"CSV {output}, rows {len(records)}", flush=True)


def list_images(folder, expected):
    paths = sorted((p for p in folder.rglob("*") if p.is_file() and
                    p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: p.name.casefold())
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} images in {folder}; found {len(paths)}")
    return paths


def measure(paths, models, device):
    dataset = FundusDataset([(p.name, p, None) for p in paths])
    times = []
    for index in range(len(dataset)):
        start = time.perf_counter()
        score(dataset[index].unsqueeze(0), models, device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def run_cost(models, device, output, sources, pictures100, pictures5):
    measure(list_images(pictures5, 5), models, device)
    paths = list_images(pictures100, 100)
    rounds = [measure(paths, models, device) for _ in range(3)]
    mean_seconds = statistics.mean(sum(values) / 3 for values in zip(*rounds))
    record = {"method": "fixed_equal_tiny_small", "device": device,
              "seconds_per_image": mean_seconds, "sources": sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("valid", "test", "cost"))
    parser.add_argument("--tiny-checkpoint", type=Path, required=True)
    parser.add_argument("--small-checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--pictures100", type=Path)
    parser.add_argument("--pictures5", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive; workers cannot be negative")
    if args.mode == "test":
        if args.output.suffix != ".csv":
            parser.error("test output must be .csv")
        if args.selection_record is None:
            parser.error("test mode requires official valid --selection-record")
    if args.mode == "cost" and (not args.pictures100 or not args.pictures5):
        parser.error("cost mode needs --pictures100 and --pictures5")
    models, metadata, device = load_pair(args.small_checkpoint, args.tiny_checkpoint)
    sources = {"small_sha256": digest(args.small_checkpoint),
               "tiny_sha256": digest(args.tiny_checkpoint),
               "small_valid_auroc": metadata[0]["official_valid_auroc"],
               "tiny_valid_auroc": metadata[1]["official_valid_auroc"]}
    if args.mode == "valid":
        run_valid(models, device, args.output, sources, args.batch_size, args.workers)
    elif args.mode == "test":
        selected = json.loads(args.selection_record.read_text(encoding="utf-8"))
        if (selected.get("metric_scope") != "official_valid" or
                selected.get("method") != "fixed_equal_tiny_small" or
                selected.get("sources") != sources or
                selected.get("rows") != 5849 or
                selected.get("metrics", {}).get("auroc", 0) <= 0.9863724439657326):
            raise ValueError("Ensemble has not exceeded the formal official-valid AUROC")
        run_test(models, device, args.output, args.workers)
    else:
        run_cost(models, device, args.output, sources, args.pictures100, args.pictures5)
