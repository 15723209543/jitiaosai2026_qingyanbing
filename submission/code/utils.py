"""比赛数据适配工具：只保留课堂 CNN 四步之外必须增加的代码。"""

# 导包
import json
import math
import time
from datetime import datetime
from os.path import relpath
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import (ColorJitter, Compose, InterpolationMode, Normalize,
                                    RandomHorizontalFlip, RandomRotation, Resize, ToTensor)
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


# 修改①: 按2026规程和本次实际下载位置固定数据、模型和结果文件的位置.
SUBMISSION_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SUBMISSION_DIR.parent
DATASET_DIR = PROJECT_DIR / "glaucoma_first"
TRAIN_CSV = DATASET_DIR / "train.csv"
TRAIN_DIR = DATASET_DIR / "train"
VALID_CSV = DATASET_DIR / "valid.csv"
VALID_DIR = DATASET_DIR / "valid"
TEST_CSV = DATASET_DIR / "test.csv"
TEST_DIR = DATASET_DIR / "test"
MODEL_PATH = SUBMISSION_DIR / "model" / "best_model.pth"
OUTPUT_CSV = SUBMISSION_DIR / "results" / "submission.csv"
TRAINING_RECORD_DIR = PROJECT_DIR / "项目文档" / "训练记录"
IMAGE_COLUMN = "Image_name"
LABEL_COLUMN = "True_Label"
# 修改①: 新训练使用128像素；历史模型按权重内记录的尺寸处理.
IMAGE_SIZE = 128
PREPROCESSING = "rgb_bilinear_half_v1"
BATCH_SIZE = 32
EPOCHS = 30
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 8
WARMUP_EPOCHS = 2
LEARNING_RATE = 3e-4
FINETUNE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
MIN_LEARNING_RATE = 1e-6
RESUME = True
CPU_THREADS = 8
# 修改⑤: 单张预测在本机24图抽测时4线程平均/最大耗时更低；训练仍用8线程.
PREDICT_CPU_THREADS = 4
LOG_EVERY = 25


# 【来源】本项目自写: 复用课堂归一化, 方案A仅训练增加温和增强.
def create_transform(image_size, training=False):
    steps = [Resize((image_size, image_size))]
    if training:
        # 修改②: 不做大幅裁剪/遮挡；验证与提交预测固定处理.
        steps += [RandomHorizontalFlip(p=0.5),
                  RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
                  ColorJitter(brightness=0.1, contrast=0.1)]
    return Compose(steps + [ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])


# 【来源】本项目自写: 前两轮线性预热, 随后余弦下降到最低学习率.
def get_learning_rate(epoch, peak_lr=LEARNING_RATE):
    if not 1 <= epoch <= EPOCHS or peak_lr < MIN_LEARNING_RATE:
        raise ValueError("轮数或学习率超出有效范围")
    if epoch <= WARMUP_EPOCHS:
        return peak_lr * (0.1 + 0.9 * (epoch - 1) / (WARMUP_EPOCHS - 1))
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return MIN_LEARNING_RATE + (peak_lr - MIN_LEARNING_RATE) * (1 + math.cos(math.pi * progress)) / 2


class ImageDataset(Dataset):
    # 【来源】本项目自写: 替代课堂CNN自带的CIFAR10数据集.
    def __init__(self, image_paths, labels=None, training=False, image_size=IMAGE_SIZE):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else list(labels)
        self.image_size = image_size
        self.training = training
        self.transform = create_transform(image_size, training)
        # 优化⑤: 只在内存缓存缩小后的RGB图, 避免多轮训练反复解码高分辨率原图.
        # 128像素训练+验证RGB像素约2.62GiB(另有Python开销)；不改动原图.
        self.cache = {}

    # 【来源】复用PyTorch Dataset标准写法: 返回样本数量.
    def __len__(self):
        return len(self.image_paths)

    # 【来源】本项目自写: 读取RGB图片, 有标签时返回x和y, 否则只返回x.
    def __getitem__(self, index):
        # 修改③: 缓存属于当前Dataset自己的尺寸, 不能把旧64缓存冒充128输入.
        if index not in self.cache:
            with Image.open(self.image_paths[index]) as image:
                image.draft("RGB", (self.image_size, self.image_size))
                self.cache[index] = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
        # 缓存不含随机增强；每次取样仍独立执行训练翻转.
        x = self.transform(self.cache[index])
        if self.labels is None:
            return x
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y


# 【来源】本项目自写: 读取一份有标签CSV及其图片, 供训练集和验证集共同使用.
def _load_labeled_images(csv_path, image_dir):
    # 1. 使用pandas读取标签.
    if not csv_path.is_file():
        raise FileNotFoundError("找不到标签CSV: %s" % csv_path)
    data = pd.read_csv(csv_path)
    if IMAGE_COLUMN not in data or LABEL_COLUMN not in data:
        raise ValueError("标注CSV必须包含Image_name和True_Label两列")

    # 2. 修改④: 官方CSV使用.jpg, 实际图片使用.JPG, 因此按文件名忽略大小写匹配.
    image_dir = Path(image_dir)
    actual_images = {
        path.name.casefold(): path
        for path in image_dir.iterdir()
        if path.is_file()
    }
    image_paths = []
    for image_name in data[IMAGE_COLUMN].astype(str):
        image_path = Path(image_name)
        if not image_path.is_absolute():
            original_path = image_dir / image_path
            image_path = actual_images.get(image_path.name.casefold(), original_path)
        if not image_path.is_file():
            raise FileNotFoundError("找不到图片: %s" % image_path)
        image_paths.append(image_path)
    labels = data[LABEL_COLUMN].astype(int).to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("label列只能是0或1")
    return image_paths, labels


# 【来源】参考课堂CNN的create_dataset改写: 优先读取2026规程提供的独立验证集.
def load_train_data():
    # 1. 修改②: CIFAR10改为glaucoma_first中的赛题图片和标签.
    image_paths, labels = _load_labeled_images(TRAIN_CSV, TRAIN_DIR)

    # 2. 修改③: 如果已放入官方验证集, 直接使用, 不再从训练集重复切分.
    if VALID_CSV.is_file():
        valid_image_paths, valid_labels = _load_labeled_images(VALID_CSV, VALID_DIR)
        return (
            ImageDataset(image_paths, labels, training=True),
            ImageDataset(valid_image_paths, valid_labels),
        )

    # 3. 数据尚未提供独立验证集时, 才按类别做8:2分层划分.
    random_state = np.random.RandomState(88)
    train_index, valid_index = [], []
    for label in (0, 1):
        index = np.where(labels == label)[0]
        if len(index) < 2:
            raise ValueError("每个类别至少需要2张图片")
        index = random_state.permutation(index)
        valid_num = min(max(int(round(len(index) * 0.2)), 1), len(index) - 1)
        valid_index.extend(index[:valid_num])
        train_index.extend(index[valid_num:])
    random_state.shuffle(train_index)
    random_state.shuffle(valid_index)

    # 4. 返回训练集和有标签验证集.
    train_dataset = ImageDataset(
        [image_paths[index] for index in train_index], labels[train_index], training=True
    )
    valid_dataset = ImageDataset(
        [image_paths[index] for index in valid_index], labels[valid_index]
    )
    return train_dataset, valid_dataset


# 【来源】本项目自写: 使用PyTorch均衡采样接口, 修正小批量训练时多数批次没有阳性的问题.
def create_sampler(labels):
    labels = torch.tensor(labels, dtype=torch.long)
    class_count = torch.bincount(labels, minlength=2).float()
    if torch.any(class_count == 0):
        raise ValueError("训练集必须同时包含标签0和标签1")
    # 修改⑤: 每类抽取总权重相同, 每轮抽取N次(有放回), 不是每张原图恰好训练一次.
    return WeightedRandomSampler(1.0 / class_count[labels], len(labels), replacement=True)


# 【来源】本项目自写: 共用训练、验证和预测的批次进度输出, 首批/每25批/末批立即打印.
def print_progress(stage, batch_idx, batch_count, start, detail=""):
    if batch_idx == 1 or batch_idx % LOG_EVERY == 0 or batch_idx == batch_count:
        elapsed = time.time() - start
        remaining = elapsed / batch_idx * (batch_count - batch_idx)
        print(
            f"{stage}, batch: {batch_idx}/{batch_count} ({batch_idx / batch_count:.1%}), "
            f"time:{elapsed:.1f}s, remaining:{remaining / 60:.1f}min{detail}", flush=True
        )


# 【来源】本项目自写: 按规程补充概率排序指标, 避免只看不平衡数据上的Acc.
def calculate_metrics(labels, positive_probability):
    probability = np.asarray(positive_probability)
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(probability).all():
        raise ValueError("验证集必须包含两类且预测概率必须有限")
    tn, fp, fn, tp = confusion_matrix(labels, probability >= 0.5, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, probability)
    return {
        "accuracy": float((tp + tn) / len(labels)),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(auc(recall, precision)),  # PR曲线梯形积分, 与AP分开记录.
        "ap": float(average_precision_score(labels, probability)),
        "recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# 【来源】迁移本项目v1.4.3的课堂CNN, 仅用于读取历史模型, 不参与新建网络.
class LegacyImageModel(torch.nn.Module):
    # 【来源】复用原CNN的init；旧三层/窄五层/宽五层/精简分类头统一放工具中.
    def __init__(self, model_config):
        super().__init__()
        self.model_config = dict(model_config)
        self.image_size = model_config["image_size"]
        architecture = model_config.get("architecture")
        slope = model_config.get("negative_slope", 0.0)
        dropout = model_config.get("dropout", 0.5)
        self.activation = torch.nn.LeakyReLU(slope) if slope else torch.nn.ReLU()
        self.activation4 = self.activation5 = self.activation
        self.conv1 = torch.nn.Conv2d(3, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.pool1 = self.pool2 = self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.conv_dropout = torch.nn.Dropout2d(model_config.get("conv_dropout", 0.0))
        if architecture is None:
            self.pool4 = torch.nn.Identity()
            if model_config.get("extra_convs", 0) == 0:
                self.conv4 = self.conv5 = torch.nn.Identity()
                self.activation4 = self.activation5 = torch.nn.Identity()
            else:
                self.conv4 = torch.nn.Conv2d(128, 128, 3, padding=1)
                self.conv5 = torch.nn.Conv2d(128, 128, 3, padding=1)
        else:
            self.conv4 = torch.nn.Conv2d(128, 256, 3, padding=1)
            self.conv5 = torch.nn.Conv2d(256, 512, 3, padding=1)
            self.pool4 = torch.nn.AdaptiveAvgPool2d((3, 3))
        if architecture == "wide5_slim":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(4608, 512), self.activation, torch.nn.Dropout(dropout),
                torch.nn.Linear(512, 2))
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(4608, 2048), self.activation, torch.nn.Dropout(dropout),
                torch.nn.Linear(2048, 2048), self.activation, torch.nn.Dropout(dropout),
                torch.nn.Linear(2048, 2))

    # 【来源】原样保留课堂CNN旧前向顺序；不改变已有92分权重的计算.
    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.activation4(self.conv4(x))
        x = self.activation5(self.conv5(x))
        x = self.pool4(self.conv_dropout(self.pool3(x)))
        return self.classifier(x.reshape(x.size(0), -1))


# 【来源】本项目自写: 按pth自身结构/输入配置加载, 不使用当前训练默认值替代历史配置.
def load_model(model_path=MODEL_PATH):
    from model import ImageModel  # 局部导入避免循环引用.

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        model_config = dict(checkpoint["model_config"])
        state_dict = checkpoint["state_dict"]
        image_size = checkpoint["image_size"]
        if model_config.get("image_size", image_size) != image_size:
            raise ValueError("权重内两处输入尺寸不一致")
    else:
        model_config = {"negative_slope": 0.0, "conv_dropout": 0.0, "dropout": 0.5}
        state_dict, image_size = checkpoint, 64
    architecture = model_config.get("architecture")
    if type(image_size) is not int or image_size < 32:
        raise ValueError("模型输入尺寸必须是至少32的整数")
    if model_config.get("preprocessing", PREPROCESSING) != PREPROCESSING:
        raise ValueError("不支持的模型预处理配置")
    model_config.update(image_size=image_size, preprocessing=PREPROCESSING)
    if architecture == "resnet18_a":
        model = ImageModel(dropout=model_config["dropout"], image_size=image_size)
    elif architecture in (None, "wide5", "wide5_slim"):
        if image_size != 64 or model_config.get("extra_convs", 0) not in (0, 2):
            raise ValueError("旧CNN只支持其原始64像素及已知层数")
        model = LegacyImageModel(model_config)
        # 修改④: 只映射内存中的旧层名, 不修改磁盘原权重.
        renamed = {"linear1": "classifier.0", "linear2": "classifier.3", "output": "classifier.6"}
        state_dict = {renamed.get(name.split(".")[0], name.split(".")[0]) + "." + name.split(".", 1)[1]: value
                      for name, value in state_dict.items()}
    else:
        raise ValueError("不支持的模型结构配置")
    model.load_state_dict(state_dict)
    return model


# 【来源】本项目自写: 仅同结构/同输入配置恢复；旧CNN权重不得迁移到ResNet.
def transfer_model_weights(model, previous_model):
    if model.model_config != previous_model.model_config:
        return False
    model.load_state_dict(previous_model.state_dict())
    return True


# 【来源】复用课堂torch.save；本项目自写临时文件替换, 供正式模型和本轮最佳共用.
def save_model(model, model_path, metrics, epoch):
    # 1. 权重、结构、选模规则、指标和轮数一起保存, 防止记录与参数无法对应.
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".tmp")
    torch.save({"format_version": 5, "state_dict": model.state_dict(),
                "model_config": model.model_config, "image_size": model.image_size,
                "selection_metric": "validation_auroc", "tie_break_metric": None,
                "validation": metrics, "epoch": epoch}, temporary_path)
    # 2. 完整写入后替换；不保存优化器状态, 不是完整断点续训.
    temporary_path.replace(model_path)


# 【来源】本项目自写: v1.4.3分开记录正式冠军与本次候选最佳, 路径相对项目根目录.
def save_training_record(history, best_metrics, best_epoch, run_dir, best_model_config,
                         training_model_config=None, selection_metric="validation_auroc",
                         peak_learning_rate=LEARNING_RATE):
    # 修改⑦: 只在项目文档/训练记录下写训练历史, 不再写入results.
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).fillna("").to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    # 同分时max保留最先出现的一轮, 与AUROC严格提高的保存条件一致；排除旧冠军epoch=0.
    candidate = max((row for row in history if row["epoch"] > 0),
                    key=lambda row: row["auroc"], default=None)
    config = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "best_validated_model", "model_file": Path(relpath(MODEL_PATH, PROJECT_DIR)).as_posix(),
        "path_base": "project_root",
        # 修改⑧: 只记录AUROC选模, 相等不更新；AUPRC仍打印, 但不参与本地决胜.
        "selection_metric": selection_metric,
        "tie_break_metric": None,
        "image_size": best_model_config["image_size"], "batch_size": BATCH_SIZE, "epochs_this_run": EPOCHS,
        "training_image_size": (training_model_config or best_model_config)["image_size"],
        "best_epoch_this_run": best_epoch, "resume": RESUME, "sampling": "balanced_with_replacement",
        "completed_epochs": history[-1]["epoch"],
        "initial_learning_rate": get_learning_rate(1, peak_learning_rate),
        "peak_learning_rate": peak_learning_rate, "weight_decay": WEIGHT_DECAY,
        "finetune_learning_rate": FINETUNE_LEARNING_RATE, "max_grad_norm": MAX_GRAD_NORM,
        "schedule": "two_epoch_warmup_then_cosine", "warmup_epochs": WARMUP_EPOCHS,
        "min_epochs": MIN_EPOCHS, "min_learning_rate": MIN_LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "model_config": best_model_config,
        "training_model_config": training_model_config or best_model_config,
        "baseline_model_file": (Path(relpath(run_dir / "baseline_model.pth", PROJECT_DIR)).as_posix()
                                if (run_dir / "baseline_model.pth").is_file() else None),
        "candidate_best": None if candidate is None else {
            "model_file": Path(relpath(run_dir / "best_model.pth", PROJECT_DIR)).as_posix(),
            "epoch": candidate["epoch"],
            "validation": {key: candidate[key] for key in best_metrics},
            "model_config": training_model_config or best_model_config,
            "image_size": (training_model_config or best_model_config)["image_size"],
        },
        "threshold": 0.5, "positive_label": 1, "validation": best_metrics,
        "submission_columns": ["Image_name", "Probability", "Pred_Label"],
        "note": "best_epoch_this_run=0表示保留训练前模型；初赛只上传results/submission.csv。",
    }
    content = json.dumps(config, ensure_ascii=False, indent=2)
    (run_dir / "run_config.json").write_text(content, encoding="utf-8")
    # 只有本次确实保存了新权重, 才同步更新model/config.json；预测以pth内配置为准.
    if best_epoch > 0:
        MODEL_PATH.with_name("config.json").write_text(content, encoding="utf-8")


# 【来源】本项目自写: 按官方test.csv顺序读取测试图片并构建无标签Dataset.
def load_test_data(test_dir=TEST_DIR, test_csv=TEST_CSV, image_size=IMAGE_SIZE):
    # 修改④: 读取test.csv, 保留官方图片名的大小写和排列顺序.
    suffixes = (".jpg", ".jpeg", ".png")
    test_dir = Path(test_dir)
    test_csv = Path(test_csv)
    if not test_csv.is_file():
        raise FileNotFoundError("找不到测试CSV: %s" % test_csv)
    data = pd.read_csv(test_csv)
    if IMAGE_COLUMN not in data:
        raise ValueError("测试CSV必须包含Image_name列")
    image_names = data[IMAGE_COLUMN].astype(str).tolist()

    # 实际文件为.JPG而CSV为.jpg, 通过小写文件名匹配以兼容大小写敏感系统.
    actual_paths = {
        path.name.casefold(): path
        for path in test_dir.rglob("*") if path.suffix.lower() in suffixes
    }
    image_paths = []
    for image_name in image_names:
        image_path = actual_paths.get(Path(image_name).name.casefold())
        if image_path is None:
            raise FileNotFoundError("找不到测试图片: %s" % image_name)
        image_paths.append(image_path)
    return ImageDataset(image_paths, image_size=image_size), image_names


# 【来源】本项目自写: 按2026规程保存可直接提交的三列CSV.
def write_submission(image_names, positive_probability, output_csv=OUTPUT_CSV):
    # 修改⑤: Probability固定表示青光眼阳性类别1的概率, 0.5为默认阈值.
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # 先统一输出精度再判定标签, 防止0.4999996写成0.500000却仍标0.
    positive_probability = np.round(np.asarray(positive_probability), 6)
    if len(image_names) != len(positive_probability) or not np.isfinite(positive_probability).all():
        raise ValueError("图片数量与概率数量不一致, 或预测概率包含无效值")
    if not ((positive_probability >= 0) & (positive_probability <= 1)).all():
        raise ValueError("预测概率必须位于[0,1]")
    pred_label = (positive_probability >= 0.5).astype(int)
    result = pd.DataFrame({
        "Image_name": [Path(image_name).name for image_name in image_names],
        "Probability": ["%.6f" % value for value in positive_probability],
        "Pred_Label": pred_label,
    })
    result.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv
