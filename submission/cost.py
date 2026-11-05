"""Measure the selected Tiny single-view inference used by predict.py."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "tiny_cost_recheck.json")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "code" / "measure_tiny.py"),
                    "--checkpoint", str(ROOT / "model" / "tiny_best_model.pth"),
                    "--pictures100", str(args.pictures_100.resolve()),
                    "--pictures5", str(args.pictures_5.resolve()),
                    "--output", str(args.output)], cwd=ROOT.parent, check=True)