"""官方数据读取、指标、三模型单文件参数与提交 CSV 工具。"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.data import Dataset
from torchvision.transforms import (ColorJitter, Compose, InterpolationMode, Normalize,
                                    RandomHorizontalFlip, RandomRotation, Resize, ToTensor)


PROJECT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "glaucoma_first"
MODEL_PATH = VARIANT_DIR / "model" / "best_model.pth"
OUTPUT_CSV = VARIANT_DIR / "results" / "submission.csv"
RECORD_DIR = PROJECT_DIR / "项目文档" / "训练记录" / VARIANT_DIR.name
TRAIN_CSV, VALID_CSV, TEST_CSV = (DATA_DIR / f"{name}.csv" for name in ("train", "valid", "test"))
TRAIN_DIR, VALID_DIR, TEST_DIR = (DATA_DIR / name for name in ("train", "valid", "test"))
MODEL_NAMES = ("resnet18", "densenet121", "efficientnet_b0")
IMAGE_SIZE, BATCH_SIZE = 64, 32
EPOCHS = {"resnet18": 12, "densenet121": 12, "efficientnet_b0": 12}
LEARNING_RATE, MIN_LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-6, 1e-4
MAX_GRAD_NORM, CPU_THREADS, PREDICT_CPU_THREADS, LOG_EVERY = 5.0, 8, 4, 25
RECIPE = "official_rgb64_independent_three_median_v1"


# 【来源】复用课堂CNN的Resize/ToTensor/Normalize；修改：只在训练时做轻量增强。
def create_transform(training=False):
    steps = [Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps += [RandomHorizontalFlip(0.5),
                  RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
                  ColorJitter(brightness=0.1, contrast=0.1)]
    return Compose(steps + [ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])


class ImageDataset(Dataset):
    # 【来源】复用课堂CNN的Dataset接口；修改：官方图片缩放后缓存以加快CPU训练。
    def __init__(self, image_paths, labels=None, training=False):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else list(labels)
        self.transform = create_transform(training)
        self.cache = {}

    # 【来源】复用PyTorch Dataset接口；返回图片数。
    def __len__(self):
        return len(self.image_paths)

    # 【来源】沿用原眼底图片缩放缓存；拆出纯读取供多线程预加载复用。
    def _load_image(self, index):
        if index not in self.cache:
            with Image.open(self.image_paths[index]) as image:
                image.draft("RGB", (IMAGE_SIZE, IMAGE_SIZE))
                self.cache[index] = image.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return self.cache[index]

    # 【来源】本项目自写；首次训练前并行解码全部图片，后续轮次仍用各自随机增强。
    def preload(self, workers=8):
        start, total = time.time(), math.ceil(len(self) / 512)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch_idx, offset in enumerate(range(0, len(self), 512), 1):
                for _ in pool.map(self._load_image, range(offset, min(offset + 512, len(self)))):
                    pass
                print_progress("preload", batch_idx, total, start)

    # 【来源】复用课堂CNN取样逻辑；修改：用眼底图片代替CIFAR10。
    def __getitem__(self, index):
        x = self.transform(self._load_image(index))
        if self.labels is None:
            return x
        return x, torch.tensor(int(self.labels[index]), dtype=torch.long)


# 【来源】沿用旧03的官方CSV读取工具；保证验证集、无标签测试集互不混用。
def load_images(csv_path, image_dir, labeled=True):
    if not csv_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError("缺少官方数据: %s 或 %s" % (csv_path, image_dir))
    data = pd.read_csv(csv_path)
    if "Image_name" not in data or (labeled and "True_Label" not in data):
        raise ValueError("官方CSV字段不完整")
    actual = {path.name.casefold(): path for path in image_dir.rglob("*") if path.is_file()}
    names = data["Image_name"].astype(str).tolist()
    paths = []
    for name in names:
        path = actual.get(Path(name).name.casefold())
        if path is None:
            raise FileNotFoundError("找不到图片: %s" % name)
        paths.append(path)
    if not labeled:
        return ImageDataset(paths), names
    labels = data["True_Label"].astype(int).tolist()
    if not set(labels) <= {0, 1}:
        raise ValueError("标签只能为0或1")
    return paths, labels


# 【来源】沿用旧03的create_dataset数据准备；官方train/valid严格分开。
def load_train_data():
    train_paths, train_labels = load_images(TRAIN_CSV, TRAIN_DIR)
    valid_paths, valid_labels = load_images(VALID_CSV, VALID_DIR)
    return (ImageDataset(train_paths, train_labels, training=True),
            ImageDataset(valid_paths, valid_labels))


# 【来源】沿用课堂训练的批次进度输出；显示已耗时和预计剩余时间。
def print_progress(stage, batch_idx, batch_count, start, detail=""):
    if batch_idx == 1 or batch_idx % LOG_EVERY == 0 or batch_idx == batch_count:
        elapsed = time.time() - start
        remaining = elapsed / batch_idx * (batch_count - batch_idx)
        print(f"{stage}, batch: {batch_idx}/{batch_count} ({batch_idx / batch_count:.1%}), "
              f"time:{elapsed:.1f}s, remaining:{remaining / 60:.1f}min{detail}", flush=True)


# 【来源】沿用旧03的验证指标；AUROC使用连续阳性概率，而非0/1标签。
def calculate_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("验证集必须含两类，概率须有限")
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= 0.5, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {"accuracy": float((tp + tn) / len(labels)),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(auc(recall, precision)),
            "ap": float(average_precision_score(labels, probabilities)),
            "recall": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


# 【来源】本项目自写；三个概率取中位数，0.5阈值恰是三模型至少两票阳性。
def median_probabilities(*vectors):
    if len(vectors) != 3 or len({len(vector) for vector in vectors}) != 1:
        raise ValueError("必须提供三个长度相同的概率向量")
    values = np.asarray(vectors, dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("概率必须有限且在0到1之间")
    return np.median(values, axis=0).tolist()


# 【来源】参考旧03的余弦学习率；每个独立模型按自身轮数计算。
def get_learning_rate(epoch, total_epochs):
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch超出范围")
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * (
        1 + math.cos(math.pi * (epoch - 1) / max(total_epochs - 1, 1))) / 2


# 【来源】复用课堂torch.save；修改：三份独立state_dict保存在同一个原子替换的pth中。
def save_ensemble(models, path, validation, epoch=0, best_metrics=None, complete=False):
    if set(models) != set(MODEL_NAMES):
        raise ValueError("必须恰好包含三种指定模型")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"models": {name: models[name].state_dict() for name in MODEL_NAMES},
                "model_names": MODEL_NAMES, "image_size": IMAGE_SIZE, "recipe": RECIPE,
                "best_metrics": best_metrics or {}, "validation": validation,
                "epoch": epoch, "complete": complete}, temporary)
    # 【修改】Windows 上杀毒扫描或文件预览可能短暂占用旧参数；等待释放后再原子替换。
    for attempt in range(30):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(1)


# 【来源】本项目自写；按单个pth内记录的三种结构严格恢复参数。
def load_ensemble(path):
    from model import ImageModel  # 避免循环导入。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["model_names"]) != MODEL_NAMES or
            checkpoint["image_size"] != IMAGE_SIZE or checkpoint["recipe"] != RECIPE):
        raise ValueError("权重的三模型结构或训练方案不匹配")
    models = {name: ImageModel(name) for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        models[name].load_state_dict(checkpoint["models"][name])
    return models, checkpoint


# 【来源】沿用旧03的结构化训练日志；只写项目文档，不写results。
def save_training_record(run_dir, history, summary):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).fillna("").to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# 【来源】沿用旧03的官方提交格式；Pred_Label始终由Probability的0.5阈值生成。
def write_submission(image_names, probabilities, output_csv=OUTPUT_CSV):
    probabilities = np.round(np.asarray(probabilities), 6)
    if len(image_names) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("图片数量或概率无效")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("概率必须在0和1之间")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Image_name": [Path(name).name for name in image_names],
                  "Probability": ["%.6f" % value for value in probabilities],
                  "Pred_Label": (probabilities >= 0.5).astype(int)}).to_csv(
                      output_csv, index=False, encoding="utf-8")
    return output_csv
