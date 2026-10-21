"""自写双视野残差注意力 CNN；全图和视盘候选区域共享骨干并联合反向传播。"""

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    """参考 SE-Net 的通道注意力思想独立实现，不复制外部参赛代码。"""

    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 8)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(channels, hidden), nn.SiLU(),
                                  nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        """自写：根据全局特征对各通道重新加权。"""
        return x * self.gate(x).unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    """参考残差连接和 SE 思想独立实现，两层卷积保持可训练梯度通路。"""

    def __init__(self, before, after, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(before, after, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(after), nn.SiLU(),
                                  nn.Conv2d(after, after, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(after), ChannelGate(after))
        self.skip = (nn.Identity() if before == after and stride == 1 else
                     nn.Sequential(nn.Conv2d(before, after, 1, stride, bias=False),
                                   nn.BatchNorm2d(after)))
        self.activation = nn.SiLU()

    def forward(self, x):
        """自写：卷积输出与捷径相加，减轻深层退化。"""
        return self.activation(self.body(x) + self.skip(x))


class FundusNet(nn.Module):
    """自写：两个视野共享卷积网络，在特征层融合后只输出一个阳性 logit。"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            ResidualBlock(32, 48, 2), ResidualBlock(48, 48),
            ResidualBlock(48, 96, 2), ResidualBlock(96, 96),
            ResidualBlock(96, 192, 2), ResidualBlock(192, 192),
            ResidualBlock(192, 256, 2), ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(256 * 3, 256), nn.SiLU(),
                                        nn.Dropout(0.25), nn.Linear(256, 1))

    def forward(self, full, disc):
        """自写：全局、局部及逐通道交互特征共同决定一次预测。"""
        a, b = self.backbone(full), self.backbone(disc)
        return self.classifier(torch.cat((a, b, a * b), dim=1)).flatten()
