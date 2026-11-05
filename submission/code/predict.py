"""Regenerate the selected official-valid Tiny submission."""
import argparse
from pathlib import Path
from predict_tiny import predict
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    predict(args.output, ROOT / "model" / "tiny_best_model.pth", args.workers)