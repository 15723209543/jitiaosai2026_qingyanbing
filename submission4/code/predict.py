"""Generate an official test CSV from an explicitly selected valid-best checkpoint."""

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import FundusDataset, ROOT, read_records
from model import NAME, create_model
from train import RECIPE, checkpoint_load


def load_model(checkpoint_path):
    checkpoint = checkpoint_load(checkpoint_path)
    if (checkpoint.get("recipe") != RECIPE or checkpoint.get("model_name") != NAME or
            checkpoint.get("metric_scope") != "official_valid" or
            checkpoint.get("training_complete") is not True or
            not 0 <= checkpoint.get("official_valid_auroc", -1) <= 1):
        raise ValueError("Checkpoint is not a completed official-valid submission4 candidate")
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), checkpoint


def score_batch(images, model):
    """Same FP32, one-view scorer is shared by prediction and cost timing."""
    with torch.inference_mode():
        device = next(model.parameters()).device
        return torch.sigmoid(model(images.to(device)).flatten()).float().cpu().tolist()


def predict(checkpoint_path, output_path, workers=2):
    model, checkpoint = load_model(checkpoint_path)
    records = read_records("test")
    loader = DataLoader(FundusDataset(records), batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=next(model.parameters()).is_cuda)
    scores = []
    for index, images in enumerate(loader, 1):
        scores.extend(score_batch(images, model))
        if index == 1 or index % 500 == 0 or index == len(loader):
            print(f"test {index}/{len(loader)}", flush=True)
    if len(records) != len(scores):
        raise RuntimeError("Incomplete test prediction")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for (name, _, _), probability in zip(records, scores):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Nonfinite/out-of-range test probability")
            probability = round(probability, 6)
            writer.writerow([name, f"{probability:.6f}", int(probability >= 0.5)])
    temporary.replace(output_path)
    print(f"CSV={output_path} rows={len(records)} valid_auroc="
          f"{checkpoint['official_valid_auroc']:.9f}", flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    predict(args.checkpoint, args.output, args.workers)
