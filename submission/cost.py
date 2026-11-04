"""Measure the same selected Small+Tiny ensemble inference implementation."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "ensemble_cost_recheck.json")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "code" / "ensemble_tiny_small.py"),
                    "--mode", "cost", "--small-checkpoint", str(ROOT / "model" / "best_model.pth"),
                    "--tiny-checkpoint", str(ROOT / "model" / "tiny_best_model.pth"),
                    "--pictures100", str(args.pictures_100.resolve()),
                    "--pictures5", str(args.pictures_5.resolve()),
                    "--output", str(args.output)], cwd=ROOT.parent, check=True)
