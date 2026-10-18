"""数据读取、指标、权重及CSV工具；训练/预测流程仍放在对应todo文件。"""

# 导包
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import (ColorJitter, Compose, InterpolationMode, Normalize,
                                    RandomHorizontalFlip, RandomRotation, Resize, ToTensor)


PROJECT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = VARIANT_DIR / "model"
OUTPUT_CSV = VARIANT_DIR / "results" / "submission.csv"
RECORD_DIR = PROJECT_DIR / "项目文档" / "训练记录" / VARIANT_DIR.name
DATA_DIR = PROJECT_DIR / "glaucoma_first"
TRAIN_CSV, VALID_CSV, TEST_CSV = (DATA_DIR / f"{name}.csv" for name in ("train", "valid", "test"))
TRAIN_DIR, VALID_DIR, TEST_DIR = (DATA_DIR / name for name in ("train", "valid", "test"))
IMAGE_SIZE, BATCH_SIZE = 64, 32
SAMPLES_PER_EPOCH = 8192
EPOCHS, MIN_EPOCHS, EARLY_STOPPING_PATIENCE = 20, 10, 6
WARMUP_EPOCHS, LEARNING_RATE, MIN_LEARNING_RATE = 2, 3e-4, 1e-6
WEIGHT_DECAY, MAX_GRAD_NORM = 1e-4, 5.0
CPU_THREADS, PREDICT_CPU_THREADS, LOG_EVERY = 8, 4, 25
RECIPE = "official_train_rgb64_balanced8192_adamw_v2"


# 【来源】本项目自写；复用课堂Resize/ToTensor/Normalize，随机增强只用于训练。
def create_transform(training=False):
    steps = [Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps += [RandomHorizontalFlip(0.5),
                  RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
                  ColorJitter(brightness=0.1, contrast=0.1)]
    return Compose(steps + [ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])


class ImageDataset(Dataset):
    # 【来源】本项目自写；用赛题图片代替课堂CIFAR10，缓存缩小后的RGB图片。
    def __init__(self, image_paths, labels=None, training=False):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else list(labels)
        self.transform = create_transform(training)
        self.cache = {}

    # 【来源】复用PyTorch Dataset接口；返回样本数。
    def __len__(self):
        return len(self.image_paths)

    # 【来源】复用Dataset取样接口；新增JPEG草图解码及内存缓存。
    def __getitem__(self, index):
        if index not in self.cache:
            with Image.open(self.image_paths[index]) as image:
                image.draft("RGB", (IMAGE_SIZE, IMAGE_SIZE))
                self.cache[index] = image.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        x = self.transform(self.cache[index])
        if self.labels is None:
            return x
        return x, torch.tensor(int(self.labels[index]), dtype=torch.long)


# 【来源】本项目自写；根据官方CSV逐行查找图片，兼容jpg/JPG大小写。
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


# 【来源】参考课堂create_dataset改写；官方train/valid严格分开。
def load_train_data():
    train_paths, train_labels = load_images(TRAIN_CSV, TRAIN_DIR)
    valid_paths, valid_labels = load_images(VALID_CSV, VALID_DIR)
    return (ImageDataset(train_paths, train_labels, training=True),
            ImageDataset(valid_paths, valid_labels))


# 【来源】本项目自写；按类别频数倒数进行有放回均衡采样。
def create_sampler(labels, num_samples=None):
    labels = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError("训练集必须同时包含0和1")
    return WeightedRandomSampler(1.0 / counts[labels],
                                 len(labels) if num_samples is None else num_samples,
                                 replacement=True)


# 【来源】本项目自写；延续课堂首批/中间/末批进度输出。
def print_progress(stage, batch_idx, batch_count, start, detail=""):
    if batch_idx == 1 or batch_idx % LOG_EVERY == 0 or batch_idx == batch_count:
        elapsed = time.time() - start
        remaining = elapsed / batch_idx * (batch_count - batch_idx)
        print(f"{stage}, batch: {batch_idx}/{batch_count} ({batch_idx / batch_count:.1%}), "
              f"time:{elapsed:.1f}s, remaining:{remaining / 60:.1f}min{detail}", flush=True)


# 【来源】本项目自写；验证指标使用阳性概率，AUROC与官方主指标一致。
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


# 【来源】本项目自写；前两轮预热，余弦下降，不用训练准确率调整学习率。
def get_learning_rate(epoch):
    if not 1 <= epoch <= EPOCHS:
        raise ValueError("epoch超出范围")
    if epoch <= WARMUP_EPOCHS:
        return LEARNING_RATE * (0.1 + 0.9 * (epoch - 1) / (WARMUP_EPOCHS - 1))
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * (1 + math.cos(math.pi * progress)) / 2


# 【来源】本项目自写；每个架构在本方案内有独立的.pth。
def model_path(model_name):
    if model_name not in ("resnet18", "densenet121", "mobilenet_v3_large",
                          "efficientnet_b0", "regnet_y_400mf"):
        raise ValueError("不支持的模型名")
    return MODEL_DIR / (model_name + ".pth")


# 【来源】本项目自写；严格按pth内记录的结构恢复，拒绝错误架构权重。
def load_model(path):
    from model import ImageModel  # 避免循环导入。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    model = ImageModel(config["model_name"], config["image_size"])
    if config != model.model_config:
        raise ValueError("权重的模型结构或训练方案不匹配")
    model.load_state_dict(checkpoint["state_dict"])
    return model


# 【来源】复用课堂torch.save；先写临时文件，验证AUROC严格提高才由train调用。
def save_model(model, path, metrics, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "model_config": model.model_config,
                "validation": metrics, "epoch": epoch, "selection_metric": "validation_auroc"}, temporary)
    temporary.replace(path)


# 【来源】本项目自写；复用前先检查其他编号方案的模型配置和完整权重。
def reuse_model(model_name, target):
    from model import ImageModel
    expected = ImageModel(model_name).model_config
    candidate = None
    for number in range(1, 9):
        path = PROJECT_DIR / f"{number:02d}" / "model" / (model_name + ".pth")
        if path == target or not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_config") == expected:
            score = checkpoint.get("validation", {}).get("auroc", float("-inf"))
            if candidate is None or score > candidate[0]:
                candidate = (score, path)
    if candidate is None:
        return None
    load_model(candidate[1])  # 严格核对层名/张量尺寸，再复制到本套模型目录。
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate[1], target)
    return candidate[1]


# 【来源】本项目自写；训练历史只写项目文档，不混入提交results目录。
def save_training_record(run_dir, history, model_name):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).fillna("").to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps({
        "variant": VARIANT_DIR.name, "model_name": model_name, "model_file": str(model_path(model_name)),
        "selection_metric": "validation_auroc", "training_recipe": RECIPE,
        "best_validation": max(history, key=lambda row: row["auroc"]) if history else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# 【来源】本项目自写；按官网字段保存可直接上传的UTF-8三列CSV。
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
