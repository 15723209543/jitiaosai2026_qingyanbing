"""Time the selected ensemble and compare 100 predictions with submission.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from predict import main

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pictures_100", type=Path)
    p.add_argument("pictures_5", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "cost_result.json")
    a = p.parse_args()
    main(["--mode", "cost", "--small-checkpoint", str(ROOT / "model/best_model.pth"),
          "--tiny-checkpoint", str(ROOT / "model/tiny_best_model.pth"),
          "--pictures100", str(a.pictures_100.resolve()),
          "--pictures5", str(a.pictures_5.resolve()),
          "--submission-csv", str(ROOT / "submission.csv"),
          "--output", str(a.output)])
