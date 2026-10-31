"""Official glaucoma splits and one deterministic 512-pixel inference transform."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "glaucoma_first"
SIZE = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_records(split):
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown official split: {split}")
    image_dir = DATA_ROOT / split
    csv_path = DATA_ROOT / f"{split}.csv"
    if not image_dir.is_dir() or not csv_path.is_file():
        raise FileNotFoundError(f"Missing official split: {split}")
    image_index = {}
    for path in image_dir.rglob("*"):
        if path.is_file():
            key = path.name.casefold()
            if key in image_index:
                raise ValueError(f"Duplicate image filename: {key}")
            image_index[key] = path
    records, seen = [], set()
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Image_name"} | ({"True_Label"} if split != "test" else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Bad official CSV columns: {csv_path}")
        for row in reader:
            name = Path(row["Image_name"]).name
            key = name.casefold()
            if key in seen or key not in image_index:
                raise ValueError(f"Missing/duplicate image: {name}")
            seen.add(key)
            label = None if split == "test" else int(row["True_Label"])
            if label is not None and label not in (0, 1):
                raise ValueError(f"Bad official label: {name}")
            records.append((name, image_index[key], label))
    if not records:
        raise ValueError(f"Empty split: {split}")
    return records


def fov_letterbox(image):
    """Remove only large near-black outer borders, then preserve aspect ratio."""
    image = image.convert("RGB")
    array = np.asarray(image)
    foreground = np.max(array, axis=2) > 8
    rows = np.flatnonzero(np.mean(foreground, axis=1) >= 0.01)
    cols = np.flatnonzero(np.mean(foreground, axis=0) >= 0.01)
    if len(rows) and len(cols):
        left, right = int(cols[0]), int(cols[-1]) + 1
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        margin_x = int(math.ceil(0.02 * (right - left)))
        margin_y = int(math.ceil(0.02 * (bottom - top)))
        left, right = max(0, left - margin_x), min(image.width, right + margin_x)
        top, bottom = max(0, top - margin_y), min(image.height, bottom + margin_y)
        # Do not crop unusually small/disconnected bright regions.
        if (right - left) * (bottom - top) >= 0.5 * image.width * image.height:
            image = image.crop((left, top, right, bottom))
    image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE))
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    return canvas


class FundusDataset(Dataset):
    def __init__(self, records, training=False):
        self.records = records
        augment = ([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1)] if training else [])
        self.transform = transforms.Compose(augment + [transforms.ToTensor(),
                                                        transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        _, path, label = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(fov_letterbox(image))
        return tensor if label is None else (tensor, torch.tensor(label, dtype=torch.float32))
