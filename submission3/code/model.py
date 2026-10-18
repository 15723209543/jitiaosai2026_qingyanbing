"""课堂CNN的todo1/todo2：定义三个分别训练的卷积神经网络。"""

import torch.nn as nn
from torchvision import models

from utils import IMAGE_SIZE, MODEL_NAMES, RECIPE, load_train_data


# todo 1. 准备数据集。
# 【来源】沿用旧03的create_dataset；只读取官方train/valid，工具逻辑放utils。
def create_dataset():
    return load_train_data()


# todo 2. 搭建(卷积)神经网络。
class ImageModel(nn.Module):
    # 【来源】沿用旧03的网络定义；修改：加入EfficientNet-B0作为第三种独立CNN。
    def __init__(self, model_name, image_size=IMAGE_SIZE):
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError("不支持的模型: %s" % model_name)
        self.model_name = model_name
        self.model_config = {"model_name": model_name, "image_size": image_size, "recipe": RECIPE}
        # 只使用结构，禁止下载外部预训练权重。
        builders = {"resnet18": models.resnet18, "densenet121": models.densenet121,
                    "efficientnet_b0": models.efficientnet_b0}
        self.network = builders[model_name](weights=None)
        if model_name == "resnet18":
            last = self.network.fc
            self.network.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(last.in_features, 2))
        elif model_name == "densenet121":
            last = self.network.classifier
            self.network.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last.in_features, 2))
        else:
            last = self.network.classifier[-1]
            self.network.classifier[-1] = nn.Linear(last.in_features, 2)

    # 【来源】沿用课堂CNN的forward；每个模型独立输出两类logits。
    def forward(self, x):
        return self.network(x)


# 5. 测试：直接运行只打印各模型结构和参数量，不训练。
if __name__ == "__main__":
    for name in MODEL_NAMES:
        model = ImageModel(name)
        print("模型:", name)
        print(model)
        print("Total params:", sum(p.numel() for p in model.parameters()))
