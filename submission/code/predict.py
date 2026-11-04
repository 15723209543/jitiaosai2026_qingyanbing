"""Reproduce the selected official-valid Small+Tiny ensemble submission."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "code" / "ensemble_tiny_small.py"),
                    "--mode", "test", "--small-checkpoint", str(ROOT / "model" / "best_model.pth"),
                    "--tiny-checkpoint", str(ROOT / "model" / "tiny_best_model.pth"),
                    "--selection-record", str(ROOT / "model" / "ensemble_valid.json"),
                    "--output", str(args.output), "--workers", str(args.workers)],
                   cwd=ROOT.parent, check=True)
