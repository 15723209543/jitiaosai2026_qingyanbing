"""Compatibility entry: reuse submission4's inference and timing implementation."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    shared = Path(__file__).resolve().parents[1] / "submission4" / "cost.py"
    runpy.run_path(str(shared), run_name="__main__")
