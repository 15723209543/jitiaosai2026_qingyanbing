# Submission4 配方（待明日 GPU 训练）

配方 ID：`convnext_tiny_in1k_fov512_fullfinetune_bce_pw5_bf16_v1`。

- 单模型：`convnext_tiny.fb_in1k`，只从仓库现有的非医学 ImageNet-1K 原始权重初始化；**全部参数训练**，不读取 submission、submission2、submission3 的青光眼任务权重。
- 仅官方 `glaucoma_first/train` 及 `train.csv` 参与梯度更新。官方 `valid` 的 5849 张只计算连续概率 AUROC 选 best；绝不合并到 train。官方 test 只在训练及选择完成后生成 CSV。
- 输入：RGB；按亮度阈值 8 检测大面积黑边，只在有效区域至少覆盖原图 50% 时去黑边；外扩 2% 后等比缩至 512，黑色补边。训练时水平翻转、±10° 旋转、亮度与对比度 ±10%；验证、测试及计时无随机增强，无 TTA。ImageNet 均值与方差归一化。
- 损失：`BCEWithLogitsLoss(pos_weight=5)`；AdamW，权重衰减 0.01，梯度裁剪 5。默认 BF16、微批量 2、梯度累积 16、峰值 LR 1e-4、首轮 LR 3e-5、随后按 epoch 余弦降至 1e-6；最多 12 轮，第 3 轮后 4 轮不提升则早停。
- 8GB GPU 可修改 `--batch-size`、`--accumulation` 等参数，但参数变化必须新建 run，不能跨配方续训。OOM 时先降低微批量，适当增大累积次数；训练吞吐与每张推理耗时必须实测。
- 每轮完整验证后把最佳候选保存在忽略的 `runs/<时间>/candidate_best.pth`，完整轮状态存在 `last.pth`。中断后同参数 `--resume runs/<时间>`。训练代码不自动覆盖现有正式模型或 CSV。
- 明日三路线分别按官方独立 valid AUROC 比较；只有确定的两个候选生成并核验提交 CSV。与当前正式模型 0.9859689935392411 的比较也须如实记录。官网分数不参与权重、轮数、阈值或融合选择。

研究来源：[同赛题公开配置](https://github.com/liqinghaoa/Eye_Qing_Guang/blob/main/Eye_Qing_Guang_code/configs/timm_convnext_tiny_fb_in1k_512_pw5_official_single3090_cache_v1.yaml)。本实现独立编写；原配置使用 RTX 3090 与 48×16 有效批量，本配方按 8GB 显存另设微批量。公开成绩不能保证在本机重现。
