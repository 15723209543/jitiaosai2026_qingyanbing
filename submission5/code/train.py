"""Compatibility entry: independent sampling via the unified submission4 trainer.

Both modes write runs and the final model under submission4; this directory
does not create a competing checkpoint or CSV.
"""
import sys
from pathlib import Path

shared = Path(__file__).resolve().parents[2] / "submission4" / "code"
sys.path.insert(0, str(shared))
from joint_train import parse_args, run  # noqa: E402

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "independent"])
    run(parse_args())
