"""ImageNet-1K ConvNeXt-Small + 通道注意力 + 主/辅助二分类头。"""

from pathlib import Path

import timm
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from utils import ROOT

NAME = "convnext_small.fb_in1k"
PRETRAIN_PATH = ROOT.parent / "项目文档" / "预训练权重" / NAME / "model.safetensors"


class GlaucomaNet(nn.Module):
    """【参考报告 S23-g07，结构为自写近似】空间特征注意力及两路分类；非原作者源码。"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        channels = backbone.num_features
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(channels, max(1, channels // 16), 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(max(1, channels // 16), channels, 1),
                                       nn.Sigmoid())
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(channels, 1)
        self.auxiliary = nn.Linear(channels, 1)

    def forward(self, images):
        features = self.backbone.forward_features(images)
        if features.shape[1] != self.backbone.num_features:
            features = features.permute(0, 3, 1, 2)
        pooled = (features * self.attention(features)).mean(dim=(2, 3))
        return self.classifier(self.dropout(pooled)).flatten(), self.auxiliary(pooled).flatten()


def create_model(pretrained=False):
    """【复用并修改】submission4/code/model.py:create_model；仅非医学 ImageNet-1K 权重。"""
    if pretrained and not PRETRAIN_PATH.is_file():
        # 只在首次训练时下载公开的 ImageNet-1K 参数，并留存完整来源文件。
        PRETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        source = timm.create_model(NAME, pretrained=True, num_classes=1000)
        save_file({k: v.contiguous().cpu() for k, v in source.state_dict().items()},
                  str(PRETRAIN_PATH))
        del source
    backbone = timm.create_model(NAME, pretrained=False, num_classes=1000,
                                 drop_rate=0.0, drop_path_rate=0.0)
    if pretrained:
        with PRETRAIN_PATH.open("rb") as handle:
            if handle.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError("预训练权重仅是 Git LFS 指针；请先 git lfs pull")
        backbone.load_state_dict(load_file(str(PRETRAIN_PATH)), strict=True)
    backbone.reset_classifier(num_classes=0)
    return GlaucomaNet(backbone)
