"""Continue the latest upstream competition checkpoint; select by official validation AUROC."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import ImageModel, MODEL_NAMES
from prepare import CACHE_DIR, prepare
from utils import PROJECT_DIR, VARIANT_DIR, MODEL_PATH, RECIPE, calculate_metrics, create_transform, save_model, load_model, get_learning_rate, model_memory_format

class CachedDataset(Dataset):
    def __init__(self, split, training=False, excluded=()):
        self.path = str(CACHE_DIR / f"{split}.npy")
        self.array = None
        info = json.loads((CACHE_DIR / f"{split}.json").read_text())
        self.labels = np.asarray(info["labels"], dtype=np.int64)
        excluded = set(excluded)
        self.indices = np.asarray([i for i in range(len(self.labels)) if i not in excluded])
        self.transform = create_transform(training)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        j = self.indices[i]
        return self.transform(Image.fromarray(self.array[j])), int(self.labels[j])

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

def loader(dataset, batch, workers, shuffle=False, seed=20261024):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)

@torch.inference_mode()
def validate(model, batches, device):
    model.eval()
    labels, probabilities = [], []
    for x, y in batches:
        x = x.to(device, non_blocking=False, memory_format=model_memory_format(model.model_config["model_name"]))
        probabilities.extend(torch.softmax(model(x), 1)[:, 1].cpu().tolist())
        labels.extend(y.tolist())
    return calculate_metrics(labels, probabilities), probabilities

def atomic_save(obj, path):
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)

