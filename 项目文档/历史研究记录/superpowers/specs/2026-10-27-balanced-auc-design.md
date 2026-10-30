# submission2 均衡采样与 AUC 损失设计

`submission2/` 是现有 `submission/` 的独立实验副本；原提交、权重与 CSV 不改。保留 ConvNeXt-Tiny、EfficientNetV2-S、384 整图补边、官方 train/valid/test 划分和四文件代码布局。训练只用 train；选模只用 valid AUROC；predict 才写 CSV。

考虑两条路径：从零训练（简单但浪费已训练参数）与从旧最佳参数暖启动（快速、需严格区分新旧配方）。采用后者：若实验目录已有新配方 PTH，先加载它；否则从同级 `submission/model/best_model.pth` 读取旧配方参数，完整验证后再训练；旧文件不写入。旧模型不在本机或仅是 LFS 指针时，给出明确错误，不伪称已验证。

每个训练批次至少一张阳性和一张阴性，训练集中每张原图每轮至少出现一次，少数类允许少量重复；验证与预测顺序、分布不变。原加权 focal 损失改为较小的阳性权重并加上成对 logistic 排序损失；损失混合系数固定在训练配置，任何变化须更新 RECIPE 并重新训练。继续使用官方验证 AUROC 严格提高的保存门槛。不得使用外部医学数据/权重、测试调参或 TTA。

只用合成数据测试采样、损失、暖启动、目录路径和脚本语法，不启动完整训练。新目录训练前不含候选 PTH/CSV；完成训练后单份 PTH 写 `submission2/model/best_model.pth`，单独运行 predict 才写 `submission2/submission.csv`。不保证 AUROC 或官网得分提升。
