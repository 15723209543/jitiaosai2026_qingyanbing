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
    # 【来源】复用day07 CNN的init；v1.3.0自写改动: 再增加两层卷积, 保留旧结构选项.
    def __init__(self, negative_slope=0.01, conv_dropout=0.1, dropout=0.5, extra_convs=2):
        # 1.1 初始化父类成员.
        super().__init__()
        if extra_convs not in (0, 2):
            raise ValueError("extra_convs只能为0(三层卷积)或2(五层卷积)")
        # 修改③: 配置随权重保存；旧模型使用ReLU, 不能直接套用新激活函数预测.
        self.model_config = {
            "negative_slope": negative_slope, "conv_dropout": conv_dropout, "dropout": dropout,
            "extra_convs": extra_convs,
        }
        self.activation = nn.LeakyReLU(negative_slope) if negative_slope else nn.ReLU()

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

        # 优化⑥: 在第3次池化前增加两层3*3卷积；padding=1保持12*12, 不再缩小图片.
        if extra_convs == 2:
            self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
            self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
            # 修改⑤: 新卷积初始化为原样传递特征, 避免随机卷积立即打乱旧特征.
            # 后面仍有LeakyReLU, 所以整个新网络不保证与旧网络输出完全相等.
            for layer in (self.conv4, self.conv5):
                nn.init.dirac_(layer.weight)
                nn.init.zeros_(layer.bias)

        # 优化③: 参考CNN课件增加全连接层的参数量.
        # 64*64图片经过5次卷积和3次池化后, 形状仍为128*6*6.
        self.linear1 = nn.Linear(128 * 6 * 6, 2048)
        self.linear2 = nn.Linear(2048, 2048)

        # 修改②: 青光眼是二分类, 输出分类数由10改为2.
        self.output = nn.Linear(2048, 2)

        # 优化④: 参考CNN课件增加Dropout, 缓解训练集准确率高、测试集准确率低.
        self.dropout = nn.Dropout(p=dropout)
        # 优化⑤: 第3组卷积/池化后随机关闭10%的特征通道；eval时自动关闭失活.
        self.conv_dropout = nn.Dropout2d(p=conv_dropout)

    # 2. 定义前向传播.
    # 【来源】复用day07 CNN的forward；v1.3.0增加两次卷积+激活, 不改变全连接部分.
    def forward(self, x):
        # 第1层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维).
        # 修改④: 新训练使用LeakyReLU, 负数保留0.01倍信号；旧权重加载时仍使用ReLU.
        x = self.pool1(self.activation(self.conv1(x)))

        # 第2层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维).
        x = self.pool2(self.activation(self.conv2(x)))

        # 优化②: 第3层后增加第4、5层, 再执行原来的池化和通道失活.
        x = self.activation(self.conv3(x))
        if self.model_config["extra_convs"] == 2:
            x = self.activation(self.conv4(x))
            x = self.activation(self.conv5(x))
        x = self.conv_dropout(self.pool3(x))

        # 细节: 全连接层只能处理二维数据, 所以要将数据进行拉平.
        x = x.reshape(x.size(0), -1)

        # 全连接层(加权求和) + 激励层(激活函数) + Dropout.
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.dropout(self.activation(self.linear2(x)))

        # 输出层只做加权求和, CrossEntropyLoss中包含softmax过程.
        return self.output(x)


# 5. 测试.
if __name__ == "__main__":
    # main函数只负责打印模型结构和参数, 与课堂summary用法一致.
    model = ImageModel()
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size=1)
