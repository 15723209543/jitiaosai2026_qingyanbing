"""100 张×3轮单张端到端计时；与 predict.py 共用 score_batch。"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import torch

from predict import load_model
from train import score_batch
from utils import FundusDataset


def image_paths(folder, expected):
    """【复用并修改】submission4/cost.py:image_list；验证计时目录图片数量。"""
    paths = sorted((p for p in Path(folder).rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: p.name.casefold())
    if len(paths) != expected:
        raise ValueError(f"{folder} 需要{expected}张图片，实有{len(paths)}张")
    return paths


def measure(paths, model, device):
    """【复用并修改】submission4/cost.py:measure；从单张读图到概率含 GPU 同步。"""
    dataset = FundusDataset([(p.name, p, None) for p in paths])
    durations = []
    for index in range(len(dataset)):
        started = time.perf_counter()
        score_batch(dataset[index].unsqueeze(0), model, device)
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    args = parser.parse_args()
    model, _, device = load_model()
    measure(image_paths(args.pictures_5, 5), model, device)
    rounds = [measure(image_paths(args.pictures_100, 100), model, device) for _ in range(3)]
    print(f"端到端平均秒/张:{statistics.mean(sum(values)/3 for values in zip(*rounds)):.6f}; "
          f"设备:{device}")
