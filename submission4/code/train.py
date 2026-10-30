"""实验 4：合并官方 train/valid 后固定随机 80/20 划分并训练双骨干。"""

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MODEL_NAMES, create_model
from sampling import split_records
from utils import (BATCH_SIZE, BalancedBatchSampler, FundusDataset, MODEL_PATH,
                   NUM_WORKERS, RECIPE, ROOT, batch_report, classification_loss,
                   fit_records, metrics, print_metrics, print_progress, should_log_batch)


# 直接改这里的 80，或运行时传 --percent 80；同一次 run 的抽样始终固定。
DEFAULT_SAMPLE_PERCENT = 80
DEFAULT_MODE = "holdout"  # 本目录固定为 holdout；另一种抽样见 submission5。
DEFAULT_SEED = 2026
RUNS_DIR = ROOT / "runs"
SOURCE_COMMIT = "d8b8f41"  # 当前 submission 代码；实际训练器见 training/20261028。

# 沿用当前 submission 底层 balanced+pairAUC 配方的参数；抽样比例单独可调。
CONFIG = {
    "convnext_tiny.fb_in1k": {"epochs": 30, "backbone_lr": 1.5e-5, "head_lr": 1.5e-4,
                              "weight_decay": 0.02, "patience": 10, "auc_weight": 0.1},
    "efficientnetv2_rw_s.ra2_in1k": {"epochs": 30, "backbone_lr": 2e-5, "head_lr": 2e-4,
                                     "weight_decay": 0.01, "patience": 10, "auc_weight": 0.1},
}


def set_seed(seed):
    """固定模型初始化以及每轮 DataLoader、增强和 Dropout 的随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_license():
    """保留原配方的 Apache-2.0 许可核查。"""
    license_path = ROOT / "licenses" / "Apache-2.0.txt"
    if not license_path.is_file():
        with urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30) as response:
            content = response.read()
        if b"Apache License" not in content[:150] or b"TERMS AND CONDITIONS" not in content:
            raise RuntimeError("无法核实 Apache-2.0 许可证全文")
        license_path.parent.mkdir(parents=True, exist_ok=True)
        license_path.write_bytes(content)
    return license_path


def loader(records, training, seed=DEFAULT_SEED):
    """沿用均衡批次训练；验证仅用无随机增强的图像。"""
    dataset = FundusDataset(records, training=training)
    options = {"num_workers": NUM_WORKERS, "pin_memory": True,
               "persistent_workers": False}
    if training:
        sampler = BalancedBatchSampler([label for _, label in records],
                                       batch_size=BATCH_SIZE, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **options)


def select_device():
    """项目约定只在 GPU 上执行正式训练。"""
    if not torch.cuda.is_available():
        raise RuntimeError("实验 4 完整训练需要 CUDA GPU；请在训练机运行")
    device = torch.device("cuda:0")
    print("GPU训练已启用:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch版本:", torch.__version__, "CUDA版本:", torch.version.cuda, flush=True)
    return device


def validate(model, valid_loader, device, title):
    """在本 run 固定的评估样本上计算连续阳性分数和 AUROC。"""
    model.eval()
    labels, scores, started = [], [], time.time()
    with torch.inference_mode():
        for index, (images, targets) in enumerate(valid_loader, 1):
            logits = model(images.to(device, non_blocking=True)).flatten()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.tolist())
            if should_log_batch(index, len(valid_loader)):
                print_progress("valid", index, len(valid_loader), started,
                               batch_report(logits, targets))
    result = metrics(labels, scores)
    print_metrics(title, result)
    return result, np.asarray(scores)


def train_epoch(model, train_loader, optimizer, scaler, pos_weight, device, epoch, name,
                auc_weight):
    """沿用当前 submission 的 AMP、均衡批次、focal+pairAUC 和梯度裁剪。"""
    model.train()
    total_loss, started = 0.0, time.time()
    for index, (images, targets) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images).flatten()
            loss = classification_loss(logits, targets, pos_weight, auc_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if should_log_batch(index, len(train_loader)):
            print_progress(f"{name} epoch:{epoch} train", index, len(train_loader), started,
                           f", loss:{loss.item():.5f}" + batch_report(logits, targets))
    return total_loss / len(train_loader)


def model_state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_settings(args):
    """续训沿用 run_config.json 的抽样参数，显式传入冲突参数则拒绝。"""
    if args.resume is None:
        return None, {
            "recipe": RECIPE,
            "source_commit": SOURCE_COMMIT,
            "initialization": "imagenet1k_only",
            "percent": DEFAULT_SAMPLE_PERCENT if args.percent is None else args.percent,
            "mode": DEFAULT_MODE,
            "seed": DEFAULT_SEED if args.seed is None else args.seed,
            "model_config": CONFIG,
        }
    run_dir = Path(args.resume).resolve()
    if not run_dir.is_relative_to(RUNS_DIR.resolve()):
        raise ValueError("--resume 必须指向 submission4/runs 下已有的 run")
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("续训缺少 run_config.json: " + str(config_path))
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if settings.get("recipe") != RECIPE or settings.get("model_config") != CONFIG:
        raise ValueError("续训代码配方或模型配置与原 run 不一致")
    if settings.get("mode") != DEFAULT_MODE:
        raise ValueError("submission4 只支持 holdout 抽样")
    for key in ("percent", "seed"):
        supplied = getattr(args, key)
        if supplied is not None and supplied != settings.get(key):
            raise ValueError(f"续训的 --{key} 与原 run 不一致")
    if settings.get("initialization") != "imagenet1k_only":
        raise ValueError("原 run 的初始化来源不符合本实验要求")
    return run_dir, settings


def prepare_run(run_dir, settings, sampling):
    """新建 run 或校验既有 run 的完整划分指纹。"""
    if run_dir is None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=False, exist_ok=False)
        settings["sampling"] = sampling
        (run_dir / "run_config.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        progress = {
            "recipe": RECIPE, "sampling": sampling, "model_index": 0, "next_epoch": 1,
            "active_state": None, "optimizer_state": None, "schedule_state": None,
            "scaler_state": None, "stale": 0, "best_states": {}, "best_metrics": {},
            "origins": {}, "complete": False,
        }
        atomic_torch_save(run_dir / "progress.pth", progress)
        return run_dir, progress
    if settings.get("sampling") != sampling:
        raise ValueError("官方数据或抽样划分与原 run 不一致，不能续训")
    progress_path = run_dir / "progress.pth"
    if not progress_path.is_file():
        raise FileNotFoundError("续训缺少 progress.pth: " + str(progress_path))
    progress = torch.load(progress_path, map_location="cpu", weights_only=True)
    if not isinstance(progress, dict) or progress.get("recipe") != RECIPE or progress.get("sampling") != sampling:
        raise ValueError("续训进度与原 run 配方或抽样划分不一致")
    if progress.get("complete"):
        raise RuntimeError("该 run 已完成；请新建 run 试验其他参数")
    return run_dir, progress


def save_progress(run_dir, progress):
    atomic_torch_save(run_dir / "progress.pth", progress)


def metric_scope(mode):
    return "random_holdout" if mode == "holdout" else "overlapping_sample"


def make_optimizer(model, config):
    classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in classifier_ids],
         "lr": config["backbone_lr"]},
        {"params": [p for p in model.parameters() if id(p) in classifier_ids],
         "lr": config["head_lr"]}], weight_decay=config["weight_decay"])


def select_inference(best_states, valid_records, device, scope):
    """仅按本 run 固定评估集的 AUROC 比较两个单模型及等权均值。"""
    valid_loader = loader(valid_records, training=False)
    labels = [label for _, label in valid_records]
    vectors = {}
    for name in MODEL_NAMES:
        model, _ = create_model(name)
        model.load_state_dict(best_states[name], strict=True)
        model.to(device)
        _, vectors[name] = validate(model, valid_loader, device, name + " 最佳参数复核")
        del model
        torch.cuda.empty_cache()
    candidates = {MODEL_NAMES[0]: vectors[MODEL_NAMES[0]],
                  MODEL_NAMES[1]: vectors[MODEL_NAMES[1]],
                  "ensemble": (vectors[MODEL_NAMES[0]] + vectors[MODEL_NAMES[1]]) / 2}
    measured = {name: metrics(labels, scores) for name, scores in candidates.items()}
    for name, result in measured.items():
        print_metrics("最终候选 " + name, result)
    selected = max(candidates, key=lambda name: measured[name]["auroc"])
    print("依据", scope, "AUROC 选择推理方案:", selected, flush=True)
    return {"name": selected, "validation": measured[selected], "metric_scope": scope}


def save_final(run_dir, progress, selected, sampling):
    """只写完成了两分支训练和选模的检查点，保留每个 run 的结果。"""
    if set(progress["best_states"]) != set(MODEL_NAMES):
        raise RuntimeError("模型分支不完整，不能输出最终权重")
    checkpoint = {
        "recipe": RECIPE, "states": progress["best_states"],
        "best_metrics": progress["best_metrics"], "pretraining": progress["origins"],
        "selection": selected, "sampling": sampling, "training_complete": True,
        "source_commit": SOURCE_COMMIT, "initialization": "imagenet1k_only",
        "torch_version": str(torch.__version__),
    }
    final_path = run_dir / "final_model.pth"
    atomic_torch_save(final_path, checkpoint)
    if final_path.stat().st_size > 500_000_000:
        raise RuntimeError("最终模型超过赛项 500MB 限制")
    if MODEL_PATH.exists():
        print("已有默认模型，保留其内容。本 run 新模型位于:", final_path, flush=True)
    else:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_path, MODEL_PATH)
        print("首次完成的模型已复制到默认预测位置:", MODEL_PATH, flush=True)
    return final_path


def train(args):
    device = select_device()
    run_dir, settings = resolve_settings(args)
    set_seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    records = fit_records()
    train_records, valid_records, sampling = split_records(
        records, percent=settings["percent"], mode=settings["mode"], seed=settings["seed"])
    run_dir, progress = prepare_run(run_dir, settings, sampling)
    print("run:", run_dir, "代码来源:", SOURCE_COMMIT, "初始权重:仅 ImageNet-1K",
          "RECIPE:", RECIPE, flush=True)
    print("抽样:", json.dumps(sampling, ensure_ascii=False), flush=True)
    if sampling["overlap_count"] != 0:
        raise RuntimeError("holdout 划分发生重叠，拒绝训练")
    print("本 run 固定互斥划分；评估子集仅用于选模。", flush=True)
    print("预训练许可正文:", ensure_license(), flush=True)
    labels = [label for _, label in train_records]
    positives, negatives = sum(labels), len(labels) - sum(labels)
    pos_weight = torch.tensor(min(10.0, (BATCH_SIZE - 1) ** 0.5), device=device)
    print(f"训练:{len(train_records)}张 阳性:{positives} 阴性:{negatives}; "
          f"评估:{len(valid_records)}张; pos_weight:{pos_weight.item():.3f}", flush=True)
    valid_loader = loader(valid_records, training=False)
    scope = metric_scope(settings["mode"])

    for model_index in range(progress["model_index"], len(MODEL_NAMES)):
        name = MODEL_NAMES[model_index]
        config = CONFIG[name]
        print("开始模型:", name, "配置:", config, "评价口径:", scope, flush=True)
        if progress["active_state"] is None:
            set_seed(settings["seed"] + model_index * 100000)
            model, origin = create_model(name, ROOT / "pretrain_model")
            if origin is None or origin.get("training_data") != "ImageNet-1K":
                raise RuntimeError("必须使用可核实的非医学 ImageNet-1K 初始权重")
            progress["origins"][name] = origin
            model.to(device)
            print("原权重:", origin["url"], "SHA256:", origin["sha256"], flush=True)
            baseline, _ = validate(model, valid_loader, device, name + " 训练前")
            print("ImageNet 初始基线 AUROC:", baseline["auroc"],
                  "；仅供参考，第一轮训练完成后才建立最佳权重。", flush=True)
            progress["next_epoch"], progress["stale"] = 1, 0
            optimizer = make_optimizer(model, config)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"], eta_min=1e-6)
            scaler = torch.amp.GradScaler("cuda")
            progress["active_state"] = model_state_cpu(model)
            progress["optimizer_state"] = optimizer.state_dict()
            progress["schedule_state"] = schedule.state_dict()
            progress["scaler_state"] = scaler.state_dict()
            save_progress(run_dir, progress)
        else:
            model, _ = create_model(name)
            model.load_state_dict(progress["active_state"], strict=True)
            model.to(device)
            optimizer = make_optimizer(model, config)
            optimizer.load_state_dict(progress["optimizer_state"])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"], eta_min=1e-6)
            schedule.load_state_dict(progress["schedule_state"])
            scaler = torch.amp.GradScaler("cuda")
            scaler.load_state_dict(progress["scaler_state"])
            print("从已完成轮次继续；下一轮:", progress["next_epoch"], flush=True)
        train_loader = loader(train_records, training=True, seed=settings["seed"])
        best_auc = progress["best_metrics"].get(name, {}).get("auroc", -1.0)
        for epoch in range(progress["next_epoch"], config["epochs"] + 1):
            set_seed(settings["seed"] + model_index * 100000 + epoch)
            train_loader.batch_sampler.set_epoch(epoch)
            loss = train_epoch(model, train_loader, optimizer, scaler, pos_weight, device,
                               epoch, name, config["auc_weight"])
            print(f"{name} epoch:{epoch}, train_loss:{loss:.6f}", flush=True)
            result, _ = validate(model, valid_loader, device, f"{name} epoch:{epoch}")
            if result["auroc"] > best_auc:
                best_auc, progress["stale"] = result["auroc"], 0
                progress["best_states"][name] = model_state_cpu(model)
                progress["best_metrics"][name] = {**result, "metric_scope": scope}
                print("评估 AUROC 提高，已保存该分支最佳参数。", flush=True)
            else:
                progress["stale"] += 1
                print("评估 AUROC 未提高；连续未提高:", progress["stale"], flush=True)
            schedule.step()
            done = epoch == config["epochs"] or progress["stale"] >= config["patience"]
            if done:
                progress.update(model_index=model_index + 1, next_epoch=1,
                                active_state=None, optimizer_state=None, schedule_state=None,
                                scaler_state=None, stale=0)
            else:
                progress.update(model_index=model_index, next_epoch=epoch + 1,
                                active_state=model_state_cpu(model),
                                optimizer_state=optimizer.state_dict(),
                                schedule_state=schedule.state_dict(),
                                scaler_state=scaler.state_dict())
            save_progress(run_dir, progress)
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"model": name, "epoch": epoch, "train_loss": loss,
                                      "metrics": result}, ensure_ascii=False) + "\n")
            if done:
                print("当前模型训练结束，最佳 AUROC:", best_auc, flush=True)
                break
        del train_loader, optimizer, schedule, scaler, model
        torch.cuda.empty_cache()
    selected = select_inference(progress["best_states"], valid_records, device, scope)
    final_path = save_final(run_dir, progress, selected, sampling)
    progress["complete"] = True
    save_progress(run_dir, progress)
    print("完整训练结束。最终模型:", final_path, flush=True)
    print("train.py 不读取 test，也不生成提交 CSV；预测另运行 predict.py。", flush=True)
    return final_path


def parse_args():
    parser = argparse.ArgumentParser(description="GPU 上按可设比例抽样官方 train+valid，训练实验 4")
    parser.add_argument("--percent", type=int, default=None,
                        help="训练抽样百分比，例如 80；默认取脚本顶部 DEFAULT_SAMPLE_PERCENT")
    parser.add_argument("--seed", type=int, default=None, help="固定划分和训练随机数；默认 2026")
    parser.add_argument("--resume", type=Path, help="续训已有 submission4/runs/<run>，从上一完整轮继续")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
