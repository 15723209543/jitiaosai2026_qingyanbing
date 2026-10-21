# submission：高分辨率双视野 CNN 实验

这是当前活动实验，尚未达到 98% AUROC。仅使用主办方 `glaucoma_first` 数据，从随机参数训练；目前没有运行正式训练，也没有生成新的 `.pth` 或 CSV。旧五模型已归档在项目根目录的 `五模型归档.zip`，不作为本实验的初始化。

## 思路与依据

原 64×64 输入可能丢失视盘边缘、神经纤维层等细节。本方案同时读取整张眼底（保比例填充至 320×320）和亮区启发式定位的候选视盘邻域（192×192），用**共享的自写残差卷积骨干**提取特征，在分类头联合训练；保存的是一份参数，而非两个单模型预测后平均。使用通道注意力、残差连接、SiLU、Dropout 和温和的阳性加权损失，主要目标是改善图像表征与类别不平衡。全图分支保留视盘外信息；亮区定位只是弱启发式，可能找错位置，必须通过官方验证集检验。

研究依据：[AG-CNN（CVPR 2019，视盘相关区域注意力）](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Attention_Based_Glaucoma_Detection_A_Large-Scale_Database_and_CNN_Model_CVPR_2019_paper.pdf)、[眼底视盘外区域也含诊断信息（Scientific Reports 2021）](https://www.nature.com/articles/s41598-021-99605-1)、[视盘定位结合 CNN（Scientific Reports 2023）](https://www.nature.com/articles/s41598-023-44473-0)。搜索到的[同届公开仓库](https://github.com/InkyUmbrella/2026Ncccu_BiDataCompetition)尚无可核实的正式训练成绩，不能据此认定某结构可达 98%。比赛规则禁止抄袭其他参赛者的代码/方案，所以这里没有复制该仓库代码，也没有使用外部数据或预训练权重。

## 目录与运行

| 文件 | 用途 |
| --- | --- |
| `code/model.py` | 自写双视野残差注意力 CNN；一次输出一个阳性 logit |
| `code/utils.py` | 官方 CSV 读取、图像保比例处理、视盘候选定位、指标和提交格式 |
| `code/train.py` | 先验证基线，再完整训练/验证；仅当完整验证集 AUROC 严格提高，更新 `model/best_model.pth`；逐轮记录写项目文档 |
| `code/predict.py` | 读取最佳参数，先复核官方验证集，再预测无标签测试集，写 `results/submission.csv` |

使用 `deepStudy` 解释器，在项目根目录运行：

```powershell
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/train.py
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/predict.py
```

官方文件保持原位：`glaucoma_first/train.csv` 和 `train/`、`valid.csv` 和 `valid/`、`test.csv` 和 `test/`。只用 `train` 更新梯度，`valid` 计算 AUROC 并选模；**绝不把无标签 test 当验证集**。在 `code/utils.py` 修改轮数、批量、学习率等常量；训练可以继续读取此方案已存在的 `.pth`。预测输出 `Image_name,Probability,Pred_Label`，行顺序与官方 `test.csv` 一致，判阳阈值暂固定 0.5。

运行 `train.py` 时先完成一遍官方验证集基线，然后才进入第 1 轮训练。训练进度在第 1 批、每 25 批和最后一批打印本段的 loss、真实阳性、报对阳性、误报、漏报；每轮结束再打印累计统计。预验证期间显示的是 `valid` 进度，不是训练卡住。

## 验收标准与风险

正式训练后应比较现有验证 AUROC 约 0.94、AUPRC、真阳性、误报/漏报以及隐藏榜单成绩；不能用训练准确率代替。当前**没有实测**新方案是否超过 0.98，也未在指定 T4 上证明单张端到端推理不超过 100 ms。保存文件应小于 500 MB，但最终体积和时延都要训练后实际检查。若候选视盘定位频繁偏离、AUROC 不升，优先改定位/输入尺寸并做消融，不要仅增加轮数或宣称达标。
