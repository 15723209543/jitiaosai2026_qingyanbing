"""固定最佳推理方案：只读取已训练 PTH，预测官方 test 后生成三列 CSV。"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import MODEL_NAMES, create_model
from utils import (BATCH_SIZE, FundusDataset, MODEL_PATH, OUTPUT_CSV,
                   inference_model_names, load_resume_checkpoint, metrics, print_metrics, print_progress,
                   read_records, write_submission)


def prediction_model_names(checkpoint, model_names, recover_paused=False):
    """暂停恢复仅使用已保存的两分支固定均值，不伪称训练和选模已完成。"""
    if recover_paused and not checkpoint.get("training_complete"):
        states = checkpoint.get("states") or {}
        if set(states) != set(model_names):
            raise RuntimeError("暂停检查点缺少模型分支，无法预测")
        return tuple(model_names)
    return inference_model_names(checkpoint, model_names)


def load_inference_models(checkpoint_path=MODEL_PATH, recover_paused=False):
    """【改自旧 predict.load_inference_models】仅加载验证选中的分支，不下载预训练权重。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("尚未生成模型: " + str(checkpoint_path))
    checkpoint = load_resume_checkpoint(checkpoint_path, MODEL_NAMES)
    chosen = prediction_model_names(checkpoint, MODEL_NAMES, recover_paused)
    selected = (checkpoint.get("selection") or {}).get("name") or "paused_ensemble_preview"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    for name in chosen:
        model, _ = create_model(name)
        model.load_state_dict(checkpoint["states"][name], strict=True)
        models.append(model.to(device).eval())
    print("推理设备:", device, "固定方案:", selected, "FP32, 无TTA", flush=True)
    return models


def score_batch(images, models):
    """【改自旧 predict.score_batch】概率先平均，不能对二值标签投票。"""
    if not models:
        raise ValueError("没有可用模型")
    device = next(models[0].parameters()).device
    with torch.inference_mode():
        x = images.to(device, non_blocking=False)
        scores = [torch.sigmoid(model(x).flatten()) for model in models]
        return torch.stack(scores).mean(dim=0).cpu().tolist()


def collect(records, models, stage):
    """【改自课堂 CNN 测试循环】验证和预测用同一套无随机增强的推理。"""
    dataset = FundusDataset(records, training=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    labels, scores, started = [], [], time.time()
    for index, batch in enumerate(loader, 1):
        if stage == "valid":
            images, targets = batch
            labels.extend(targets.tolist())
        else:
            images = batch
        scores.extend(score_batch(images, models))
        print_progress(stage, index, len(loader), started)
    return labels, scores


def predict(output_csv=OUTPUT_CSV, checkpoint_path=MODEL_PATH, recover_paused=False):
    """【改自旧 predict.predict】先报训练集内指标，再按官方 test.csv 原顺序写结果。"""
    models = load_inference_models(checkpoint_path, recover_paused)
    # 验证集内复核
    """
    labels, validation = collect(valid_records, models, "valid")
    print_metrics("训练集内复核（非独立验证，存在数据泄漏）", metrics(labels, validation))
    """
    # 测试集预测
    test_records = read_records("test")
    _, scores = collect(test_records, models, "test")
    result = write_submission(test_records, scores, output_csv)
    print(f"已输出:{result}, 测试图片:{len(test_records)}, 预测阳性:{sum(x >= 0.5 for x in scores)}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练集内复核后生成 submission.csv；不训练、无独立验证") # 仅用于复核，不训练
    parser.add_argument("output_csv", nargs="?", type=Path, default=OUTPUT_CSV) # 输出文件路径
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH,
                        help="使用指定的冻结检查点文件")
    parser.add_argument("--recover-paused", action="store_true",
                        help="仅供暂停恢复：两分支固定等权预测，不代表已完成最终选模")
    args = parser.parse_args() # 解析命令行参数
    predict(args.output_csv, args.checkpoint, args.recover_paused) # 进行预测
