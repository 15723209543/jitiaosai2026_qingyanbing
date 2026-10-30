"""Reproducible class-stratified draws from the official train + valid pool."""

import hashlib
import math
from pathlib import Path
import random


def _path_key(path):
    # Official train/valid image names are unique across the pooled CSVs.
    # A basename makes the split fingerprint portable to another computer.
    return Path(path).name.casefold()


def _pool_sha256(records):
    digest = hashlib.sha256()
    for path, label in records:
        encoded = _path_key(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes((label,)))
    return digest.hexdigest()


def _selection_sha256(pool_hash, indices):
    digest = hashlib.sha256(bytes.fromhex(pool_hash))
    digest.update(len(indices).to_bytes(8, "big"))
    for index in sorted(indices):
        digest.update(index.to_bytes(8, "big"))
    return digest.hexdigest()


def _draw(by_label, positive_count, total_count, seed):
    rng = random.Random(seed)
    negative_count = total_count - positive_count
    return (set(rng.sample(by_label[1], positive_count)) |
            set(rng.sample(by_label[0], negative_count)))


def split_records(records, percent=80, mode="holdout", seed=2026):
    """Return training records, evaluation records, and auditable split metadata.

    ``holdout`` draws ``percent`` for training and uses its complement for
    evaluation. ``independent`` draws ``percent`` twice with separate random
    streams; the resulting evaluation set can overlap training and its metrics
    are not an independent validation estimate. Sampling is once per run.
    """
    if (isinstance(percent, bool) or not isinstance(percent, (int, float)) or
            not math.isfinite(percent) or not 0 < percent < 100):
        raise ValueError("percent 必须是 0 到 100 之间的数字，例如 80")
    if mode not in ("holdout", "independent"):
        raise ValueError("mode 必须是 holdout 或 independent")
    if type(seed) is not int:
        raise ValueError("seed 必须是整数")

    records = list(records)
    by_label = {0: [], 1: []}
    paths = set()
    for index, record in enumerate(records):
        try:
            path, label = record
            key = _path_key(path)
        except (TypeError, ValueError) as error:
            raise ValueError("每条记录必须是 (图片路径, 0或1标签)") from error
        if type(label) is not int or label not in (0, 1):
            raise ValueError("标签必须是整数 0 或 1")
        if key in paths:
            raise ValueError("train+valid 中有重复图片名")
        paths.add(key)
        by_label[label].append(index)

    if min(map(len, by_label.values())) < 2:
        raise ValueError("每类至少需要两张图片，才能分出训练和验证")
    total = len(records)
    count = round(total * percent / 100)
    if count < 2 or count > total - 2:
        raise ValueError("当前比例会让训练集或验证集少于两张图片")

    # Preserve the exact requested total while keeping both classes in both
    # sides of a holdout. The class count may differ by at most the rounding
    # required to satisfy the rare-class constraint.
    lower_positive = max(1, count - (len(by_label[0]) - 1))
    upper_positive = min(len(by_label[1]) - 1, count - 1)
    if lower_positive > upper_positive:
        raise ValueError("当前比例无法在训练和验证中同时保留两类")
    positive_count = min(max(round(count * len(by_label[1]) / total),
                             lower_positive), upper_positive)

    train_indices = _draw(by_label, positive_count, count, seed)
    valid_indices = (set(range(total)) - train_indices if mode == "holdout" else
                     _draw(by_label, positive_count, count, seed + 1))
    train_records = [record for index, record in enumerate(records)
                     if index in train_indices]
    valid_records = [record for index, record in enumerate(records)
                     if index in valid_indices]
    pool_hash = _pool_sha256(records)
    metadata = {
        "percent": percent,
        "mode": mode,
        "seed": seed,
        "pool_count": total,
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "train_positive": sum(records[index][1] for index in train_indices),
        "valid_positive": sum(records[index][1] for index in valid_indices),
        "overlap_count": len(train_indices & valid_indices),
        "train_fraction": len(train_records) / total,
        "valid_fraction": len(valid_records) / total,
        "pool_sha256": pool_hash,
        "train_sha256": _selection_sha256(pool_hash, train_indices),
        "valid_sha256": _selection_sha256(pool_hash, valid_indices),
    }
    return train_records, valid_records, metadata
