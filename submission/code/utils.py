"""官方图片、双视野预处理、验证指标与提交格式；不使用外部数据。"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


ROOT = Path(__file__).resolve().parents[2]
VARIANT = Path(__file__).resolve().parents[1]
DATA = ROOT / "glaucoma_first"
MODEL_PATH = VARIANT / "model" / "best_model.pth"
OUTPUT_CSV = VARIANT / "results" / "submission.csv"
FULL_SIZE, DISC_SIZE = 320, 192
TRAIN_BATCH, EVAL_BATCH, EPOCHS = 16, 32, 30
LR, WEIGHT_DECAY, NUM_WORKERS = 3e-4, 1e-4, 0


def letterbox(image, size):
    """自写：保留眼底原始宽高比，缩放最长边并以黑色居中填充。"""
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (size, size))
    result.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return result


def find_disc_center(image):
    """自写：用模糊后的绿色通道寻找候选亮区；只是弱定位，不用外部视盘标注。"""
    small = image.getchannel("G").resize((64, 64), Image.Resampling.BOX)
    bright = np.asarray(small.filter(ImageFilter.BoxBlur(3)), dtype=np.float32)
    bright[:8, :] = bright[56:, :] = -1
    bright[:, :8] = bright[:, 56:] = -1
    row, col = np.unravel_index(int(np.argmax(bright)), bright.shape)
    return int((col + 0.5) * image.width / 64), int((row + 0.5) * image.height / 64)


def create_views(image):
    """自写：同时生成全眼底和候选视盘邻域，供同一网络联合训练。"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    x, y = find_disc_center(image)
    side = max(1, int(min(image.size) * 0.42))
    left = min(max(0, x - side // 2), image.width - side)
    top = min(max(0, y - side // 2), image.height - side)
    return letterbox(image, FULL_SIZE), letterbox(
        image.crop((left, top, left + side, top + side)), DISC_SIZE)


class FundusDataset(Dataset):
    """复用课堂 Dataset 接口；每次只读取一张官方图片，避免全量高分辨率缓存。"""

    def __init__(self, split, labeled=True, augment=False):
        table = pd.read_csv(DATA / f"{split}.csv")
        if "Image_name" not in table or (labeled and "True_Label" not in table):
            raise ValueError(f"{split}.csv 缺少必需字段")
        image_dir = DATA / split
        available = {p.name.casefold(): p for p in image_dir.rglob("*") if p.is_file()}
        self.names = table["Image_name"].astype(str).tolist()
        self.paths = [available.get(Path(name).name.casefold()) for name in self.names]
        if any(path is None for path in self.paths):
            raise FileNotFoundError(f"{split} 图片文件不完整")
        self.labels = (table["True_Label"].astype(int).tolist() if labeled else None)
        if self.labels is not None and not set(self.labels) <= {0, 1}:
            raise ValueError("标签必须为0或1")
        self.augment = augment

    def __len__(self):
        """复用 Dataset 接口：返回图片数量。"""
        return len(self.paths)

    def __getitem__(self, index):
        """自写：相同随机翻转同步作用于全图和视盘图，防止两视野错位。"""
        with Image.open(self.paths[index]) as source:
            full, disc = create_views(source)
        if self.augment and torch.rand(()).item() < 0.5:
            full = ImageOps.mirror(full)
            disc = ImageOps.mirror(disc)
        label = -1 if self.labels is None else self.labels[index]
        return (to_tensor(full) - 0.5) / 0.5, (to_tensor(disc) - 0.5) / 0.5, label


def metrics(labels, probabilities, threshold=0.5):
    """自写：AUROC 按连续概率计算；另给出阳性报对、误报和漏报。"""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("验证集必须含两类，且概率均有限")
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= threshold,
                                      labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(labels, probabilities)),
                auprc=float(average_precision_score(labels, probabilities)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def progress(stage, index, total, start, detail=""):
    """复用课堂进度习惯：首批、每25批及末批打印耗时与剩余时间。"""
    if index == 1 or index % 25 == 0 or index == total:
        elapsed = time.time() - start
        detail = detail() if callable(detail) else detail
        print(f"{stage}, batch: {index}/{total} ({index / total:.1%}), "
              f"time:{elapsed:.1f}s, remaining:{elapsed * (total - index) / index / 60:.1f}min{detail}",
              flush=True)
        return True
    return False


def write_submission(names, probabilities, path=OUTPUT_CSV, threshold=0.5):
    """自写：仅 predict.py 调用；生成官方 Image_name/Probability/Pred_Label 三列。"""
    probabilities = np.asarray(probabilities, dtype=float)
    if (len(names) != len(probabilities) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("预测数量或概率无效")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Image_name": names,
                  "Probability": np.round(probabilities, 6),
                  "Pred_Label": (probabilities >= threshold).astype(int)}).to_csv(
                      path, index=False, encoding="utf-8")
    return path
