"""课堂CNN的todo1/todo2：ResNet18和DenseNet121提取特征，融合后共同分类。"""

# 导包
import torch
import torch.nn as nn
from torchvision import models

from utils import IMAGE_SIZE, RECIPE, load_train_data


# todo 1. 准备数据集.
# 【来源】复用day07 CNN的create_dataset接口；读取官方train/valid，数据处理放utils。
def create_dataset():
    return load_train_data()


# todo 2. 搭建(卷积)神经网络.
class ImageModel(nn.Module):
    # 【来源】复用day07 CNN的__init__结构；修改①: 两个骨干共享一个融合分类头。
    def __init__(self, image_size=IMAGE_SIZE):
        super().__init__()
        self.model_config = {"architecture": "resnet18_densenet121_feature_fusion",
                             "image_size": image_size, "recipe": RECIPE,
                             "fusion_hidden": [4096, 1024]}
        # 修改②: 只用网络结构，不下载外部权重；旧03参数在train.py启动时载入。
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()               # 每张图提取512维特征。
        self.densenet = models.densenet121(weights=None)
        self.densenet.classifier = nn.Identity()      # 每张图提取1024维特征。
        # 修改③: 特征先拼接，再经加权求和、激活和部分失活，最后只输出一次二分类结果。
        self.fusion = nn.Sequential(
            nn.Linear(512 + 1024, 4096), nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1), nn.Dropout(0.35),
            nn.Linear(4096, 1024), nn.LeakyReLU(0.1), nn.Dropout(0.25),
            nn.Linear(1024, 2),
        )

    # 【来源】复用day07 CNN的forward接口；修改④: 在特征层融合，绝不平均两个分类概率。
    def forward(self, x):
        features = torch.cat((self.resnet(x), self.densenet(x)), dim=1)
        return self.fusion(features)


# 5. 测试: 点击运行model.py只打印结构和参数，不训练。
if __name__ == "__main__":
    model = ImageModel()
    print(model)
    print("Total params:", sum(p.numel() for p in model.parameters()))
