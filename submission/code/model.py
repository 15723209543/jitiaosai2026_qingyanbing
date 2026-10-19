"""
案例:
    保留课堂CNN四步接口, 方案A将网络替换为ResNet-18.
"""

# 导包
import torch.nn as nn
from torchvision.models import resnet18
from torchsummary import summary

from utils import IMAGE_SIZE, PREPROCESSING, load_train_data


# todo 1. 准备数据集.
# 【来源】复用day07 CNN的create_dataset接口, 图片改为赛题train/valid.
def create_dataset():
    # 1. 返回训练集和独立验证集, 不把训练图片合入验证集.
    return load_train_data()


# todo 2. 搭建(卷积)神经网络.
class ImageModel(nn.Module):
    # 【来源】课堂CNN的init接口；修改①: 复用torchvision.models.resnet18官方结构.
    def __init__(self, dropout=0.2, image_size=IMAGE_SIZE):
        # 1. 初始化父类成员.
        super().__init__()
        self.image_size = image_size
        self.model_config = {"architecture": "resnet18_a", "dropout": dropout,
                             "image_size": image_size, "preprocessing": PREPROCESSING}
        # 2. 修改②: 只复用结构, weights=None随机初始化, 不下载外部预训练参数.
        # 残差块将输入与卷积结果相加；保留原有BatchNorm、ReLU和全局平均池化.
        self.network = resnet18(weights=None)
        # 3. 修改③: 原1000类输出改为Dropout+二分类；交叉熵接收logits, 不提前softmax.
        self.network.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512, 2))

    # 【来源】复用课堂forward接口；修改④: 调用官方残差网络的前向传播.
    def forward(self, x):
        return self.network(x)


# 5. 测试.
if __name__ == "__main__":
    # 与课堂summary一致: 只打印结构和参数, 不训练、不保存权重.
    model = ImageModel()
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size=1, device="cpu")
