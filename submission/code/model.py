"""课堂CNN的todo1/todo2：定义五个分别训练的卷积神经网络。"""

import torch.nn as nn
from torchvision import models

from utils import IMAGE_SIZE, MODEL_NAMES, MODEL_TRAINING, RECIPE, load_train_data


# todo 1. 准备数据集。
# 【来源】沿用旧03的create_dataset；只读取官方train/valid，工具逻辑放utils。
def create_dataset():
    return load_train_data()


# todo 2. 搭建(卷积)神经网络。
class ImageModel(nn.Module):
    # 【来源】复用submission3三模型；修改：加入课堂wide5_slim和ConvNeXt-Tiny。
    def __init__(self, model_name, image_size=IMAGE_SIZE):
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError("不支持的模型: %s" % model_name)
        self.model_name = model_name
        self.model_config = {"model_name": model_name, "image_size": image_size, "recipe": RECIPE}
        # 只使用结构，禁止下载外部预训练权重。
        builders = {"resnet18": models.resnet18, "densenet121": models.densenet121,
                    "efficientnet_b0": models.efficientnet_b0,
                    "convnext_tiny": models.convnext_tiny}
        if model_name == "wide5_slim":
            self.network = LegacyCNN()  # 课堂CNN结构和失活参数完全不变。
        elif model_name == "efficientnet_b0":
            self.network = builders[model_name](weights=None, dropout=MODEL_TRAINING[model_name]["dropout"])
        elif model_name == "convnext_tiny":
            self.network = builders[model_name](
                weights=None, stochastic_depth_prob=MODEL_TRAINING[model_name]["stochastic_depth"])
        else:
            self.network = builders[model_name](weights=None)
        if model_name == "wide5_slim":
            pass  # 课堂模型自身已经输出两类logits。
        elif model_name == "resnet18":
            last = self.network.fc
            self.network.fc = nn.Sequential(nn.Dropout(MODEL_TRAINING[model_name]["dropout"]),
                                            nn.Linear(last.in_features, 2))
        elif model_name == "densenet121":
            last = self.network.classifier
            self.network.classifier = nn.Sequential(nn.Dropout(MODEL_TRAINING[model_name]["dropout"]),
                                                    nn.Linear(last.in_features, 2))
        elif model_name == "efficientnet_b0":
            last = self.network.classifier[-1]
            self.network.classifier[-1] = nn.Linear(last.in_features, 2)
        else:
            last = self.network.classifier[-1]
            self.network.classifier[-1] = nn.Linear(last.in_features, 2)

    # 【来源】沿用课堂CNN的forward；每个模型独立输出两类logits。
    def forward(self, x):
        return self.network(x)


# 【来源】逐层复用旧submission v1.5.1 的 wide5_slim 五层CNN；保留原参数键名以载入旧权重。
class LegacyCNN(nn.Module):
    def __init__(self, negative_slope=0.01, conv_dropout=0.1, dropout=0.5):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope) if negative_slope else nn.ReLU()
        self.activation4 = self.activation
        self.activation5 = self.activation
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)
        self.pool3 = nn.MaxPool2d(2, 2, 0)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.pool4 = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512), self.activation,
            nn.Dropout(p=dropout), nn.Linear(512, 2))
        self.conv_dropout = nn.Dropout2d(p=conv_dropout)

    # 【来源】复用旧submission v1.5.1 ImageModel.forward，输入输出形状不变。
    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.activation4(self.conv4(x))
        x = self.activation5(self.conv5(x))
        x = self.pool4(self.conv_dropout(self.pool3(x)))
        return self.classifier(x.reshape(x.size(0), -1))


# 5. 测试：直接运行只打印各模型结构和参数量，不训练。
if __name__ == "__main__":
    for name in MODEL_NAMES:
        model = ImageModel(name)
        print("模型:", name)
        print(model)
        print("Total params:", sum(p.numel() for p in model.parameters()))
