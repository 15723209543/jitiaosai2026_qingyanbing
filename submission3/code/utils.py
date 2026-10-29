"""四文件方案共用的数据、指标和 CSV 工具；不得从测试集拟合任何参数。"""

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "glaucoma_first"
MODEL_PATH = ROOT / "model" / "best_model.pth"
BASELINE_MODEL_PATH = ROOT.parent / "submission" / "model" / "best_model.pth"
OUTPUT_CSV = ROOT / "submission.csv"
IMAGE_SIZE = 384
BATCH_SIZE = 16
# 【本机调整】原图 2592×1944～5184×3456，缩到 384 的 LANCZOS 约 67ms/张，加载是瓶颈。
# 实测 8 进程 81.7 张/秒、12 进程 100.7 张/秒；但每个 worker 提交内存约 1 GB 以上，
# 12 个常驻 worker 叠加第二个 loader 会耗尽本机 25.7 GB 提交上限（WinError 1455 页面文件太小），
# 故取 6 并配合 train.py 中 persistent_workers=False，任何时刻只有一组 worker 存活。
NUM_WORKERS = 6
LOG_EVERY = 25
# 【本机修复】正式 submission/model/best_model.pth 内的实际配方是 v4 official_only（2026-10-28 晋升），
# 原写的 ..._fullframe_v1 会让暖启动直接判定配方不符并退出；这里只改“识别正式权重”的字符串，不改任何训练数学。
BASE_RECIPE = "imagenet1k_convnext_efficientv2_rgb384_fullframe_balanced_pairauc_v4_official_only_20261028"
RECIPE = "imagenet1k_convnext_efficientv2_rgb384_train_valid_insample_v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def letterbox(image):
    """【改自旧 utils.letterbox】完整图像等比例缩到 384 方框，不裁剪、不拉伸。"""
    image = image.convert("RGB")
    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    result.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
    return result


class FundusDataset(Dataset):
    """【改自课堂 CNN Dataset】只在训练分支施加轻量增强；验证/预测完全固定。"""

    def __init__(self, records, training=False):
        self.records = records
        steps = []
        if training:
            steps += [transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(8),
                      transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06)]
        steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        """【复用 PyTorch Dataset 接口】返回全部样本数，不抽样。"""
        return len(self.records)

    def __getitem__(self, index):
        """【改自课堂 CNN Dataset】读取一张完整眼底图及其官方标签。"""
        path, label = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(letterbox(image))
        return tensor if label is None else (tensor, torch.tensor(label, dtype=torch.float32))


def read_records(split):
    """【改自旧 utils.load_images】严格依官方 CSV 顺序读取单个分组。"""
    if split not in {"train", "valid", "test"}:
        raise ValueError("未知数据分组")
    folder, csv_path = DATA / split, DATA / (split + ".csv")
    if not folder.is_dir() or not csv_path.is_file():
        raise FileNotFoundError(f"缺少官方数据: {folder} 或 {csv_path}")
    paths = {p.name.casefold(): p for p in folder.rglob("*") if p.is_file()}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "Image_name" not in reader.fieldnames or (split != "test" and "True_Label" not in reader.fieldnames):
            raise ValueError("官方 CSV 字段不完整")
        records, names = [], set()
        for row in reader:
            name = Path(row["Image_name"]).name
            key = name.casefold()
            if key in names or key not in paths:
                raise ValueError("图片重复或不存在: " + name)
            names.add(key)
            label = None if split == "test" else int(row["True_Label"])
            if label is not None and label not in (0, 1):
                raise ValueError("标签必须为 0 或 1")
            records.append((paths[key], label))
    if not records:
        raise ValueError("空数据集: " + split)
    return records


def fit_records():
    """【自写】实验3按用户要求合并官方 train/valid；同一批图也用于训练集内 AUROC。"""
    records = read_records("train") + read_records("valid")
    names = [path.name.casefold() for path, _ in records]
    if len(names) != len(set(names)):
        raise ValueError("官方 train/valid 合并后存在同名图片，无法安全计算训练集内指标")
    return records


def load_resume_checkpoint(path, model_names, recipe=RECIPE):
    """【自写】只读已有最佳参数；校验配方和分支，拒绝把 Git LFS 指针当模型。"""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as file:
        if file.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("当前文件只是 Git LFS 指针，请在训练机下载真实模型后再续训")
    # 【兼容旧 PTH】旧训练脚本把 torch.__version__（TorchVersion 对象）写入文件；
    # 只临时放行这一种 PyTorch 内置类型，仍保持 weights_only=True，绝不执行任意 pickle。
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("recipe") != recipe:
        raise ValueError("已有模型配方与当前代码不一致，不能直接续训")
    states = checkpoint.get("states")
    best_metrics = checkpoint.get("best_metrics")
    origins = checkpoint.get("pretraining")
    if not all(isinstance(value, dict) for value in (states, best_metrics, origins)):
        raise ValueError("已有模型缺少分支参数、验证指标或预训练来源")
    if set(states) != set(best_metrics) or not set(states).issubset(model_names):
        raise ValueError("已有模型分支和验证记录不一致")
    if checkpoint.get("training_complete") and set(states) != set(model_names):
        raise ValueError("标记为完成的模型缺少分支")
    for name, state in states.items():
        record = best_metrics[name]
        if name not in origins or not state or not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError("已有模型参数或来源不完整: " + name)
        # 兼容旧权重中的 epoch；新权重只保存最佳参数和指标，不再记录轮次。
        if not isinstance(record, dict) or ("epoch" in record and
                                             (type(record["epoch"]) is not int or record["epoch"] < 1)):
            raise ValueError("已有模型最佳轮次无效: " + name)
        auc = record.get("auroc")
        if not isinstance(auc, (int, float)) or not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("已有模型 AUROC 无效: " + name)
    return checkpoint


def inference_model_names(checkpoint, model_names):
    """【自写】只接受已完成且选模有效的检查点；阶段性权重不能用于提交预测。"""
    selection = checkpoint.get("selection") or {}
    selected = selection.get("name") if isinstance(selection, dict) else None
    if not checkpoint.get("training_complete") or selected not in (*model_names, "ensemble"):
        raise RuntimeError("模型未完成最终验证选模；请先续训 train.py，不能用阶段性参数生成提交 CSV")
    return tuple(model_names) if selected == "ensemble" else (selected,)


def resume_start_epoch(checkpoint, name):
    """【自写】每次启动的轮次均从 1 计数；旧权重只提供参数和 AUROC 门槛。"""
    return 1


def resume_baseline_auc(checkpoint, name, measured_auc):
    """【自写】整套官方验证集复核后建立续训门槛；差异过大则保护旧权重。"""
    saved_auc = checkpoint["best_metrics"][name]["auroc"]
    if not math.isfinite(measured_auc) or abs(measured_auc - saved_auc) > 0.002:
        raise ValueError(f"{name} 复核 AUROC 与旧权重记录不符: 当前{measured_auc:.6f}, 已存{saved_auc:.6f}")
    return max(measured_auc, saved_auc)


def restore_candidate_model(name, checkpoint, create_model, pretrain_dir):
    """【自写】已有分支严格载入最佳参数；未训练分支才初始化非医学预训练权重。"""
    if checkpoint is not None and name in checkpoint["states"]:
        model, _ = create_model(name)
        model.load_state_dict(checkpoint["states"][name], strict=True)
        return model, checkpoint["pretraining"][name], True
    model, origin = create_model(name, pretrain_dir)
    return model, origin, False


def print_progress(stage, index, total, started, detail=""):
    """【复用课堂 CNN 终端习惯】首批、每 25 批和末批打印本批进度。"""
    if should_log_batch(index, total):
        elapsed = time.time() - started
        remaining = elapsed * (total - index) / max(index, 1) / 60
        print(f"{stage}, batch: {index}/{total} ({index / total:.1%}), "
              f"time:{elapsed:.1f}s, remaining:{remaining:.1f}min{detail}", flush=True)


def should_log_batch(index, total):
    """【自写】只在首批、每 25 批及末批统计，避免每批同步 GPU 拖慢训练。"""
    return index == 1 or index % LOG_EVERY == 0 or index == total


def batch_report(logits, targets):
    """【自写】按当前批次的前向输出统计阳性、报阳、报对、误报和漏报；不累计。"""
    predicted = logits.detach().flatten() >= 0  # sigmoid(logit) >= 0.5
    positive = targets.detach().flatten().to(logits.device) >= 0.5
    true_positive = predicted & positive
    false_positive = predicted & ~positive
    false_negative = ~predicted & positive
    actual, reported, correct, wrong, missed = torch.stack((
        positive.sum(), predicted.sum(), true_positive.sum(),
        false_positive.sum(), false_negative.sum())).cpu().tolist()
    return (f", 真实阳性:{actual}, 报阳:{reported}, 报对阳性:{correct}, "
            f"误报:{wrong}, 漏报:{missed}")


def metrics(labels, probabilities):
    """【改自旧 utils.calculate_metrics】AUROC 只用官方验证真标签与连续阳性分数。"""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(labels.tolist()) != {0, 1} or len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("验证标签/预测分数无效")
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= 0.5, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            "ap": float(average_precision_score(labels, probabilities)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "recall": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp))}


def print_metrics(title, result):
    """【自写】只显示主指标和阳性检出/误报，避免总体准确率掩盖漏诊。"""
    print(f"{title}: AUROC:{result['auroc']:.6f}, AP:{result['ap']:.6f}, "
          f"真实阳性:{result['tp'] + result['fn']}, 报对阳性:{result['tp']}, "
          f"误报:{result['fp']}, 漏报:{result['fn']}, 召回率:{result['recall']:.2%}", flush=True)


def write_submission(records, scores, output_path=OUTPUT_CSV):
    """【改自旧 utils.write_submission】仅预测脚本写三列、原顺序 CSV。"""
    if len(records) != len(scores) or not scores:
        raise ValueError("测试图片数与预测结果不一致")
    output_path = Path(output_path).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_name", "Probability", "Pred_Label"])
        for (path, _), score in zip(records, scores):
            if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise ValueError("概率必须位于 0 到 1")
            score = round(float(score), 6)
            image_path = Path(path)
            # 官方 test.csv 使用 .jpg；磁盘上的原图可能是 .JPG，提交时仅规范后缀大小写。
            image_name = image_path.with_suffix(image_path.suffix.lower()).name
            writer.writerow([image_name, f"{score:.6f}", int(score >= 0.5)])
    return output_path
