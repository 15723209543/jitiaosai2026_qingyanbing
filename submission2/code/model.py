"""ImageNet-1K 非医学预训练双骨干；仅训练脚本下载原始权重。"""

import hashlib
from pathlib import Path

import timm
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file


MODEL_NAMES = ("convnext_tiny.fb_in1k", "efficientnetv2_rw_s.ra2_in1k")
MODEL_REPOS = {name: "timm/" + name for name in MODEL_NAMES}


def sha256_file(path):
    """【自写】分块计算原始预训练权重 SHA-256，供赛后溯源。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_model(name, pretrain_dir=None):
    """【改自 timm 官方模型卡】先严格载入 ImageNet 原权重，再换成单输出青光眼头。"""
    if name not in MODEL_REPOS:
        raise ValueError("未知模型: " + name)
    if pretrain_dir is None:
        return timm.create_model(name, pretrained=False, num_classes=1, drop_rate=0.1), None
    repo = MODEL_REPOS[name]
    revision = HfApi().model_info(repo).sha
    path = hf_hub_download(repo_id=repo, filename="model.safetensors", revision=revision,
                           local_dir=str(Path(pretrain_dir) / name))
    model = timm.create_model(name, pretrained=False, num_classes=1000, drop_rate=0.1)
    model.load_state_dict(load_file(path), strict=True)
    model.reset_classifier(num_classes=1)
    origin = {"name": name, "repo": repo, "revision": revision,
              "url": f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors",
              "sha256": sha256_file(path), "license": "Apache-2.0",
              "training_data": "ImageNet-1K",
              "original_file": f"pretrain_model/{name}/model.safetensors"}
    return model, origin
