"""官方数据读取、同一预处理、指标与安全检查点；训练和预测共用。"""

import csv
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GLAUCOMA_DATA_ROOT", ROOT.parent / "glaucoma_first"))
MODEL_PATH = ROOT / "model" / "best_model.pth"
SIZE = 320
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_records(split):
    """【复用并修改】submission4/code/data.py:read_records；仅按官方 CSV 顺序读取对应目录。"""
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"未知官方分组: {split}")
    folder, table = DATA_ROOT / split, DATA_ROOT / f"{split}.csv"
    if not folder.is_dir() or not table.is_file():
        raise FileNotFoundError(f"缺少官方数据: {folder} 或 {table}")
    files = {p.name.casefold(): p for p in folder.rglob("*") if p.is_file()}
    records, seen = [], set()
    with table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Image_name"} | ({"True_Label"} if split != "test" else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"官方 CSV 缺少字段: {required}")
        for row in reader:
            name = Path(row["Image_name"]).name
            key = name.casefold()
            if key in seen or key not in files:
                raise ValueError(f"图片缺失或重复: {name}")
            seen.add(key)
            label = None if split == "test" else int(row["True_Label"])
            if label is not None and label not in (0, 1):
                raise ValueError(f"错误标签: {name}")
            records.append((name, files[key], label))
    if not records:
        raise ValueError(f"空数据集: {split}")
    return records


def full_image(image):
    """【修改】整图等比缩放至 320 并补边；不用视盘裁剪，训练/验证/预测一致。"""
    image = image.convert("RGB")
    image.thumbnail((SIZE, SIZE), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE))
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    return canvas


class FundusDataset(Dataset):
    """【复用并修改】submission4/code/data.py:FundusDataset；训练返回独立增强双视图。"""

    def __init__(self, records, training=False):
        self.records = records
        self.training = training
        augmentation = ([transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomRotation(15),
                         transforms.ColorJitter(0.22, 0.22, 0.15)] if training else [])
        tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        if training:
            tail.append(transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)))
        self.transform = transforms.Compose(augmentation + tail)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        _, path, label = self.records[index]
        with Image.open(path) as image:
            image = full_image(image)
            first = self.transform(image)
            if self.training:
                return first, self.transform(image), torch.tensor(label, dtype=torch.float32)
            return first if label is None else (first, torch.tensor(label, dtype=torch.float32))


def calculate_metrics(labels, scores):
    """【自写】官方验证集 AUROC 优先，同时报告 AP、真实阳性、误报、漏报。"""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or len(labels) == 0 or not np.isfinite(scores).all():
        raise ValueError("验证概率缺失或非有限值")
    predictions = scores >= 0.5
    return {"auroc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores)),
            "positive": int(labels.sum()), "tp": int(((labels == 1) & predictions).sum()),
            "fp": int(((labels == 0) & predictions).sum()),
            "fn": int(((labels == 1) & ~predictions).sum())}


def fingerprint(records):
    """【复用并修改】submission4/code/train.py:fingerprint；锁定图片名与标签。"""
    digest = hashlib.sha256()
    for name, _, label in records:
        digest.update(f"{name}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def load_checkpoint(path):
    """【复用并修改】submission4/code/train.py:checkpoint_load；拒绝 LFS 指针。"""
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"仅有 Git LFS 指针，请在 GPU 机执行 git lfs pull: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def atomic_save(value, path):
    """【复用】submission4/code/train.py:atomic_save；写完临时文件才替换检查点。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
