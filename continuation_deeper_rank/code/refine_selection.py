"""The first-improvement rule already locks selection; only verify its evidence."""
import argparse
from pathlib import Path
from select_inference import select
from utils import PROJECT_DIR
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=PROJECT_DIR/'runs/20261025_deeper_rank');select(p.parse_args().run_dir)
