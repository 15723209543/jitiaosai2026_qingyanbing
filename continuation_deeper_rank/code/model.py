"""
案例: 复用课堂CNN的todo1、todo2接口；本方案同时使用两种独立CNN。
本文件夹固定恢复历史得分最高的ResNet-18 + DenseNet-121双模型组合。
"""

# 导包
import torch.nn as nn
from torchvision import models

from utils import IMAGE_SIZE, RECIPE, load_train_data


# 修改①: 固定历史03方案；两个网络独立训练，预测时平均阳性概率。
MODEL_NAMES = ("resnet18", "densenet121", "resnet34_graft")
BUILDERS = {
    "resnet18": models.resnet18,
    "densenet121": models.densenet121,
    "resnet34_graft": models.resnet34,
}


# todo 1. 准备数据集.
# 【来源】复用day07 CNN的create_dataset接口；读取官方train/valid。
def create_dataset():
    return load_train_data()


# todo 2. 搭建(卷积)神经网络.
class ImageModel(nn.Module):
    # 【来源】复用day07 CNN的__init__位置；修改②: 选用CCF A论文对应的torchvision网络结构。
    def __init__(self, model_name, image_size=IMAGE_SIZE):
        super().__init__()
        if model_name not in BUILDERS:
            raise ValueError("不支持的模型: %s" % model_name)
        self.image_size = image_size
        self.model_config = {"model_name": model_name, "image_size": image_size, "recipe": RECIPE}
        # 修改③: 只借用网络结构，weights=None，不下载或使用外部训练权重。
        self.network = BUILDERS[model_name](weights=None)
        if model_name in ("resnet18", "resnet34_graft"):
            last = self.network.fc
            self.network.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(last.in_features, 2))
        elif model_name == "densenet121":
            last = self.network.classifier
            self.network.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last.in_features, 2))

    # 【来源】复用day07 CNN的forward接口；输出两类logits，交叉熵内计算softmax。
    def forward(self, x):
        return self.network(x)


# 5. 测试: main只打印两种模型的结构和参数，不训练。
if __name__ == "__main__":
    for name in MODEL_NAMES:
        print("模型:", name)
        # 修改④: torchsummary不兼容DenseNet内部传入的list，直接打印结构与参数量。
        model = ImageModel(name)
        print(model)
        print("Total params:", sum(p.numel() for p in model.parameters()))
