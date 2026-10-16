"""
案例:
    基于课堂 CNN 案例, 准备青光眼数据集并搭建卷积神经网络.
"""

# 导包
import torch
import torch.nn as nn
from torchsummary import summary

from utils import IMAGE_SIZE, load_train_data


# todo 1. 准备数据集.
# 【来源】复用day07 CNN的create_dataset, 只把CIFAR10改为2026赛题训练/验证图片.
def create_dataset():
    # 修改①: 优先返回主办方提供的训练集和独立验证集.
    train_dataset, valid_dataset = load_train_data()
    # 3. 返回数据集.
    return train_dataset, valid_dataset


# todo 2. 搭建(卷积)神经网络.
class ImageModel(nn.Module):
    # 1. 初始化父类成员, 搭建神经网络.
    # 【来源】复用day07 CNN的init；v1.4.3自写改动: 保留五层卷积, 精简全连接层.
    def __init__(self, negative_slope=0.01, conv_dropout=0.1, dropout=0.5):
        # 1.1 初始化父类成员.
        super().__init__()
        # 修改③: 配置随权重保存；旧模型使用ReLU, 不能直接套用新激活函数预测.
        self.model_config = {
            "negative_slope": negative_slope, "conv_dropout": conv_dropout, "dropout": dropout,
            "architecture": "wide5_slim",
        }
        self.activation = nn.LeakyReLU(negative_slope) if negative_slope else nn.ReLU()
        self.activation4 = self.activation
        self.activation5 = self.activation

        # 1.2 搭建神经网络.
        # 优化①: 参考CNN课件增加卷积核数量, 输入3通道, 输出32通道.
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        # 第2个卷积层, 输入32通道, 输出64通道, 卷积计算方式不变.
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        # 优化②: 参考day06 ANN优化版增加网络深度, 原样增加1个卷积+池化块.
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)
        self.pool3 = nn.MaxPool2d(2, 2, 0)

        # 优化⑥: 固定第4、5层, 通道128→256→512；padding=1保持12*12.
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        # 优化⑦: 第3次最大池化后为512*6*6, 再平均汇总为512*3*3, 控制全连接参数量.
        self.pool4 = nn.AdaptiveAvgPool2d((3, 3))

        # 修改⑧: 方案B1, 将4608→2048→2048→2改为4608→512→2.
        # Sequential只是按顺序执行课堂的全连接、激活、失活、输出, 不增加新算法.
        # 分类头变小以减少参数；保留Dropout(0.5), 不把其他变量一次全部修改.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            self.activation,
            nn.Dropout(p=dropout),
            nn.Linear(512, 2),
        )
        # 优化⑤: 第3组卷积/池化后随机关闭10%的特征通道；eval时自动关闭失活.
        self.conv_dropout = nn.Dropout2d(p=conv_dropout)

    # 2. 定义前向传播.
    # 【来源】复用day07 CNN的forward；V1.3.1固定经过五层卷积, 无层数选择分支.
    def forward(self, x):
        # 第1层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维).
        # 修改④: 新训练使用LeakyReLU, 负数保留0.01倍信号；旧权重加载时仍使用ReLU.
        x = self.pool1(self.activation(self.conv1(x)))

        # 第2层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维).
        x = self.pool2(self.activation(self.conv2(x)))

        # 优化②: 第3层后增加第4、5层, 再执行原来的池化和通道失活.
        x = self.activation(self.conv3(x))
        x = self.activation4(self.conv4(x))
        x = self.activation5(self.conv5(x))
        x = self.conv_dropout(self.pool3(x))
        x = self.pool4(x)

        # 细节: 全连接层只能处理二维数据, 所以要将数据进行拉平.
        x = x.reshape(x.size(0), -1)

        # 修改⑨: 一组全连接(加权求和)+激活+Dropout, 再输出两类logits.
        # 输出层不做softmax, CrossEntropyLoss中已包含该过程.
        return self.classifier(x)


# 5. 测试.
if __name__ == "__main__":
    # main函数只负责打印模型结构和参数, 与课堂summary用法一致.
    model = ImageModel()
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size=1)
