"""比赛数据适配工具：只保留课堂 CNN 四步之外必须增加的代码。"""

# 导包
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
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
IMAGE_SIZE = 64
# 优化④: 本机实测64张一批比8张一批的每张计算耗时更低, 同时让批内两类比例更稳定.
BATCH_SIZE = 64
# 修改②: 最多续训12轮, 本轮训练模型自身的所选验证指标连续10轮未提高就提前结束.
EPOCHS = 12
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 1  # ReduceLROnPlateau允许1轮停滞, 第2轮仍无改善时学习率减半.
MIN_LEARNING_RATE = 1e-6
RESUME = True
CPU_THREADS = 8 # 本机8核, 与PyTorch默认一致.
LOG_EVERY = 25

# 优化①: 参考day06 ANN优化版的数据标准化方法，对图像张量进行标准化.
transform = Compose([
    Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_transform = Compose([RandomHorizontalFlip(p=0.5), transform])


class ImageDataset(Dataset):
    # 【来源】本项目自写: 替代课堂CNN自带的CIFAR10数据集.
    def __init__(self, image_paths, labels=None, training=False):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else list(labels)
        # 优化③: 均衡采样会重复阳性图片, 仅训练集增加水平翻转, 验证/测试保持固定变换.
        self.transform = train_transform if training else transform
        # 优化⑤: 只在内存缓存缩小后的RGB图, 避免多轮训练反复解码高分辨率原图.
        # 训练+验证共57245张, RGB像素约671MiB；不在磁盘额外生成图片或改动原数据.
        self.cache = {}

    # 【来源】复用PyTorch Dataset标准写法: 返回样本数量.
    def __len__(self):
        return len(self.image_paths)

    # 【来源】本项目自写: 读取RGB图片, 有标签时返回x和y, 否则只返回x.
    def __getitem__(self, index):
        # 优化②: 原图分辨率很大, 先让JPEG解码器按64*64附近尺寸读取, 再执行课堂Resize.
        if index not in self.cache:
            with Image.open(self.image_paths[index]) as image:
                image.draft("RGB", (IMAGE_SIZE, IMAGE_SIZE))
                self.cache[index] = image.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
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


# 【来源】本项目自写: 兼容旧纯权重及新配置+权重文件, 供训练验证和提交预测共同调用.
def load_model(model_path=MODEL_PATH):
    # 1. 修改⑥: 局部导入避免model.py引用utils时出现循环导入.
    from model import ImageModel

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # 2. 新参数按文件内配置重建；不能用当前训练默认配置解释历史权重.
    if "state_dict" in checkpoint:
        if checkpoint.get("image_size") != IMAGE_SIZE:
            raise ValueError("模型输入尺寸与IMAGE_SIZE不一致, 请使用对应的训练配置")
        model_config = dict(checkpoint["model_config"])
        state_dict = checkpoint["state_dict"]
    else:
        # v1.1.1及更早的参数: ReLU、无卷积失活、全连接Dropout=0.5.
        model_config = {"negative_slope": 0.0, "conv_dropout": 0.0, "dropout": 0.5}
        state_dict = checkpoint
    architecture = model_config.pop("architecture", None)
    extra_convs = model_config.pop("extra_convs", 0)
    if architecture not in (None, "wide5") or extra_convs not in (0, 2):
        raise ValueError("不支持的模型结构配置")
    model = ImageModel(**model_config)
    # 修改⑧: 旧三层/窄五层仅在工具中兼容, 不让model.py承担历史结构分支.
    if architecture is None:
        model.pool4 = torch.nn.Identity()
        if extra_convs == 0:
            for name in ("conv4", "conv5", "activation4", "activation5"):
                setattr(model, name, torch.nn.Identity())
        else:
            model.conv4 = torch.nn.Conv2d(128, 128, 3, 1, 1)
            model.conv5 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        model.model_config = {**model_config, "extra_convs": extra_convs}
    model.load_state_dict(state_dict)
    return model


# 【来源】本项目自写: 加宽模型迁移前三个卷积层；相同结构继续训练则恢复全部权重.
def transfer_model_weights(model, previous_model):
    # 1. 加宽/池化后特征组织已变化, 即使拉平长度相同也不沿用旧分类头.
    old_state = previous_model.state_dict()
    state_dict = old_state
    if model.model_config.get("architecture") == "wide5" and previous_model.model_config.get("architecture") != "wide5":
        state_dict = model.state_dict()
        for layer in ("conv1", "conv2", "conv3"):
            for parameter in ("weight", "bias"):
                name = f"{layer}.{parameter}"
                state_dict[name] = old_state[name]
    # 2. 不改变磁盘文件；共享层缺失/尺寸错误仍直接报错, 其余新层保持默认初始化.
    model.load_state_dict(state_dict)


# 【来源】本项目自写: 每次运行独立保存记录；v1.4.1由训练分支传入实际选模指标.
def save_training_record(history, best_metrics, best_epoch, run_dir, best_model_config,
                         training_model_config=None, selection_metric=None):
    # 修改⑦: 只在项目文档/训练记录下写训练历史, 不再写入results.
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).fillna("").to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    config = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "best_validated_model", "model_file": MODEL_PATH.name,
        # 修改⑧: epoch=0尚未进入选模分支, 指标暂记null；每轮按实际启用方案更新.
        "selection_metric": selection_metric,
        "tie_break_metric": "validation_auprc" if selection_metric == "validation_auroc" else None,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "epochs_this_run": EPOCHS,
        "best_epoch_this_run": best_epoch, "resume": RESUME, "sampling": "balanced_with_replacement",
        "completed_epochs": history[-1]["epoch"],
        "initial_learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "lr_patience": LR_PATIENCE, "min_learning_rate": MIN_LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "model_config": best_model_config,
        "training_model_config": training_model_config or best_model_config,
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
def load_test_data(test_dir=TEST_DIR, test_csv=TEST_CSV):
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
    return ImageDataset(image_paths), image_names


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
