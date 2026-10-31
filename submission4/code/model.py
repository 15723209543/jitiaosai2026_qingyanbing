"""ImageNet-1K ConvNeXt-Tiny; task weights never initialize a fresh run."""

import hashlib
from pathlib import Path

import timm
from safetensors.torch import load_file

from data import ROOT

NAME = "convnext_tiny.fb_in1k"
PRETRAIN_PATH = ROOT.parent / "submission" / "pretrain_model" / NAME / "model.safetensors"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_model(pretrained=False):
    if pretrained:
        if not PRETRAIN_PATH.is_file():
            raise FileNotFoundError(f"Missing ImageNet-1K weight: {PRETRAIN_PATH}")
        model = timm.create_model(NAME, pretrained=False, num_classes=1000,
                                  drop_rate=0.0, drop_path_rate=0.0)
        model.load_state_dict(load_file(str(PRETRAIN_PATH)), strict=True)
        model.reset_classifier(num_classes=1)
        return model
    return timm.create_model(NAME, pretrained=False, num_classes=1,
                             drop_rate=0.0, drop_path_rate=0.0)
