"""Compatibility entry: both sampling modes share submission4's final model and CSV."""
import runpy
import sys
from pathlib import Path

shared = Path(__file__).resolve().parents[2] / "submission4" / "code"
sys.path.insert(0, str(shared))

if __name__ == "__main__":
    runpy.run_path(str(shared / "predict.py"), run_name="__main__")
