"""官方数据读取、指标、五模型单文件参数与提交 CSV 工具。"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
MODEL_NAMES = ("wide5_slim", "resnet18", "densenet121", "efficientnet_b0", "convnext_tiny") # 模型名称
IMAGE_SIZE, BATCH_SIZE = 64, 32 # 图像大小、批量大小
# 【修改】每次点击train.py都从本次第0轮计数；可读取旧权重，但不继承旧轮次。
# epochs是本次运行要训练的轮数，设0表示不训练但仍验证；各模型独立修改。
# positive_share是整类阳性损失占比，不是阳性采样比例；不混用测试集伪标签。
# focal_gamma=0表示原交叉熵；dropout仅用于模型已有分类头，ConvNeXt用stochastic_depth。
# 下面每一个参数的说明：
# epochs：本次训练轮数，从0开始计数，不叠加旧检查点的轮次。
# learning_rate：本次起始学习率；若加载旧权重，沿用较小的微调学习率。
# min_lr：本次余弦调度的最小学习率。
# weight_decay：权重衰减。
# positive_share：整类阳性损失占比，不是阳性采样比例；不混用测试集伪标签。
# focal_gamma：Focal Loss的gamma参数。
# max_grad_norm：梯度裁剪。
# dropout：部分失活比例。
# stochastic_depth：ConvNeXt的stochastic depth参数。
MODEL_TRAINING = {
    "wide5_slim": dict(epochs=30, learning_rate=1e-4,
                       min_lr=1e-6, weight_decay=1e-4, positive_share=0.25,
                       focal_gamma=0.0, max_grad_norm=5.0, dropout=0.5,
                       stochastic_depth=0.0),
    "resnet18":      dict(epochs=30, learning_rate=3e-5,
                          min_lr=1e-6, weight_decay=2e-4, positive_share=0.35,
                          focal_gamma=1.0, max_grad_norm=3.0, dropout=0.30,
                          stochastic_depth=0.0),
    "densenet121":   dict(epochs=30, learning_rate=2e-5,
                          min_lr=1e-6, weight_decay=3e-4, positive_share=0.40,
                          focal_gamma=1.2, max_grad_norm=2.5, dropout=0.35,
                          stochastic_depth=0.0),
    "efficientnet_b0": dict(epochs=30, learning_rate=5e-5,
                            min_lr=1e-6, weight_decay=1.5e-4, positive_share=0.45,
                            focal_gamma=1.5, max_grad_norm=4.0, dropout=0.25,
                            stochastic_depth=0.0),
    "convnext_tiny": dict(epochs=30, learning_rate=4e-5,
                          min_lr=1e-6, weight_decay=2.5e-4, positive_share=0.50,
                          focal_gamma=2.0, max_grad_norm=1.0, dropout=0.0,
                          stochastic_depth=0.20),
}
EPOCHS = {name: MODEL_TRAINING[name]["epochs"] for name in MODEL_NAMES}
# 【修改】按单张图片的五个阳性概率从高到低排序后依次乘以以下占比。
# 第1项对应最高概率，不固定对应某个模型；五项非负且总和必须为1。
# 全部0.2时是五模型概率平均值；改动后仍以最终概率0.5判阳性。
RANK_WEIGHTS = (0.5, 0.3, 0.2, 0, 0)
RANK_STEP = 0.05 # 每轮用训练集最多移动5个百分点；只用官方有标签验证集验收。
CPU_THREADS, PREDICT_CPU_THREADS, LOG_EVERY = 8, 4, 25 # CPU线程数、预测CPU线程数、日志间隔步
RECIPE = "official_rgb64_independent_five_rank_v1" # 新排序加权方案
LEGACY_RECIPE = "official_rgb64_independent_five_median_v1" # 兼容已经训练好的旧pth


# 【来源】本项目自写；训练和预测开始前打印当前五个排序占比，便于核对配置。
def print_rank_weights(weights=RANK_WEIGHTS):
    print("五模型排序权重（阳性概率从高到低）:", ", ".join(str(value) for value in weights),
          flush=True)


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

    # 【来源】本项目自写；排序占比训练复用已解码图片，但关闭随机增强以稳定估计概率。
    def evaluation_view(self):
        if self.labels is None:
            raise ValueError("无标签图片不能用于训练排序占比")
        view = ImageDataset(self.image_paths, self.labels, training=False)
        view.cache = self.cache
        return view


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
    # 【修改】恢复第1批、每25批、最后一批输出；返回是否打印，供验证窗口计数归零。
    if batch_idx == 1 or batch_idx % LOG_EVERY == 0 or batch_idx == batch_count:
        elapsed = time.time() - start
        remaining = elapsed / batch_idx * (batch_count - batch_idx)
        print(f"{stage}, batch: {batch_idx}/{batch_count} ({batch_idx / batch_count:.1%}), "
              f"time:{elapsed:.1f}s, remaining:{remaining / 60:.1f}min{detail}", flush=True)
        return True
    return False


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


# 【来源】修改原五模型中位数；每张图先排序概率，再按RANK_WEIGHTS加权求和。
def rank_weighted_probabilities(*vectors, weights=None):
    if len(vectors) != 5 or len({len(vector) for vector in vectors}) != 1:
        raise ValueError("必须提供五个长度相同的概率向量")
    values = np.asarray(vectors, dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("概率必须有限且在0到1之间")
    weights = np.asarray(RANK_WEIGHTS if weights is None else weights, dtype=float)
    if (weights.shape != (5,) or not np.isfinite(weights).all() or
            (weights < 0).any() or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-9)):
        raise ValueError("RANK_WEIGHTS必须是五个非负占比，且总和为1")
    return (weights @ np.sort(values, axis=0)[::-1]).tolist()


# 【来源】本项目自写；在官方训练集预测上调整排序占比，验证集只负责验收。
def tune_rank_weights(labels, scores, weights):
    # 【修复】收集函数返回按模型名索引的字典；按固定顺序取五条概率向量。
    # 直接*字典会展开模型名字符串，而不是概率向量。
    if isinstance(scores, dict):
        if set(scores) != set(MODEL_NAMES):
            raise ValueError("排序占比训练必须包含全部五个模型")
        scores = tuple(scores[name] for name in MODEL_NAMES)
    weights = tuple(float(value) for value in weights)
    best_metrics = calculate_metrics(labels, rank_weighted_probabilities(*scores, weights=weights))
    best_weights = weights
    for source in range(5):
        if weights[source] + 1e-9 < RANK_STEP:
            continue
        for target in range(5):
            if source == target:
                continue
            candidate = list(weights)
            candidate[source] -= RANK_STEP
            candidate[target] += RANK_STEP
            if any(candidate[index] + 1e-9 < candidate[index + 1] for index in range(4)):
                continue  # 保持最高概率对应最高占比。
            candidate = tuple(round(value, 10) for value in candidate)
            metrics = calculate_metrics(labels, rank_weighted_probabilities(*scores, weights=candidate))
            if (metrics["auroc"] > best_metrics["auroc"] + 1e-9 or
                    (abs(metrics["auroc"] - best_metrics["auroc"]) <= 1e-9 and
                     metrics["fp"] + metrics["fn"] < best_metrics["fp"] + best_metrics["fn"])):
                best_weights, best_metrics = candidate, metrics
    return best_weights, best_metrics


# 【来源】参考旧03的余弦学习率；每个独立模型按自身轮数计算。
def get_learning_rate(epoch, total_epochs, learning_rate=1e-4, min_learning_rate=1e-6):
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch超出范围")
    return min_learning_rate + (learning_rate - min_learning_rate) * (
        1 + math.cos(math.pi * (epoch - 1) / max(total_epochs - 1, 1))) / 2


# 【来源】参考Lin等人ICCV 2017 Focal Loss；仅用于非CNN分支，强化难例而不增模型参数。
class FocalLoss(nn.Module):
    def __init__(self, weights, gamma):
        super().__init__()
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        weights = self.weights[targets]
        return (weights * (-torch.expm1(-ce)).pow(self.gamma) * ce).sum() / weights.sum()


# 【来源】修改submission3单文件保存；分别保存五个单模最佳和整组最佳快照。
def save_ensemble(individual_models, ensemble_models, path, validation,
                  best_metrics=None, epoch=0, complete=False, completed_epochs=None,
                  rank_weights=None):
    if set(individual_models) != set(MODEL_NAMES) or set(ensemble_models) != set(MODEL_NAMES):
        raise ValueError("必须恰好包含五种指定模型")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"individual_models": {name: individual_models[name].state_dict() for name in MODEL_NAMES},
                "ensemble_models": {name: ensemble_models[name].state_dict() for name in MODEL_NAMES},
                "model_names": MODEL_NAMES, "image_size": IMAGE_SIZE, "recipe": RECIPE,
                "best_metrics": best_metrics or {}, "validation": validation,
                "rank_weights": tuple(RANK_WEIGHTS if rank_weights is None else rank_weights),
                "completed_epochs": completed_epochs or {name: epoch for name in MODEL_NAMES},
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


# 【来源】修改submission3读取；预测使用整组最佳，训练可选单模型最佳。
def load_ensemble(path, snapshot="ensemble"):
    from model import ImageModel  # 避免循环导入。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["model_names"]) != MODEL_NAMES or
            checkpoint["image_size"] != IMAGE_SIZE or
            checkpoint["recipe"] not in (RECIPE, LEGACY_RECIPE)):
        raise ValueError("权重的五模型结构或训练方案不匹配")
    if snapshot not in ("ensemble", "individual"):
        raise ValueError("snapshot只能为ensemble或individual")
    models = {name: ImageModel(name) for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        models[name].load_state_dict(checkpoint[snapshot + "_models"][name])
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
