# 冻结配方与差异

配方 ID：`small320_erasing_ema_aux_consistency_official_valid_v1`。依据用户提供的 2026-11-01 报告 S23 g07；原作者没有随报告提供完整源码、权重、原始分割索引，所以本实现为**注明差异的近似复现**，不是原训练轨迹恢复。

- 单模型 `convnext_small.fb_in1k`，仅非医学 ImageNet-1K 预训练，320×320 整图等比补边；不裁视盘、不用测试 TTA。
- 官方 train 内分层随机取 41,116 张；随机种子 20261024。官方 valid 全部 5,849 张始终只用于 AUROC 选模；其余官方 train 仅留作内部数据，不用于模型选取。
- AdamW 学习率 1e-4，weight decay 1e-4，40轮余弦调度、最多25轮、最少10轮且连续6轮无 AUROC 提升早停。BF16 优先，否则 FP16；默认 micro-batch 2×累积32，显存允许可改为报告的8×8；梯度范数截断2。
- 训练增强：水平翻转0.5，旋转±15°，ColorJitter(0.22,0.22,0.15)，随机擦除0.1、面积0.01–0.05。验证与测试不增强。
- 主 BCE + 0.2 辅助 BCE + 至第5轮线性升至0.05的一致性 MSE。阳性 BCE 权重取 `min(6, sqrt(阴性/阳性))`；报告只写“截断平方根”，未给截断值，本实现固定6，属明确近似。
- EMA 衰减0.996006，选 EMA 的官方 valid AUROC 严格提升时保存单份权重；AP、召回、误报只报告，不用于选模。预测固定阈值0.5。
- 原报告按内部 selection 的 `sqrt(AUROC×梯形PR)` 选第7轮；此处为遵守项目规则改用独立官方 valid AUROC，实际最佳轮次不预设，不能套用原报告指标或官网98.55。
- 原报告的空间注意力与辅助头确切结构、归一化与随机数状态未知；本目录用可运行的通道注意力与辅助头实现。若取得作者原源码/权重，应重新建立配方版本并重新验证。

## 2026-11-02 本机时间盒复现（执行前记录）

用户要求北京时间20:30前交付，8GB GPU与胸片题共享。先按报告的 micro-batch 8×累积8、workers2、CPU线程4，从非医学ImageNet-1K原始权重新开run；本次最多7轮（报告高分选择第7轮），保留代码内40轮余弦调度，官方valid每完整轮选AUROC。因原报告训练上限25轮且按内部joint选模，本次是受时限约束的近似复现，不得宣称精确复刻98.55。显存探针batch8三步峰值分配约3922MiB，稳定单步约0.163秒，完整数据加载耗时仍待实测。任何参数更改记录后新建run，原run只用同参数resume。


## 2026-11-02 submission_new 本机近似复现结果

依据远端 `submission_new` 与用户提供的高分报告，在本机 RTX 5060 Laptop 8GB 使用非医学 ImageNet-1K `convnext_small.fb_in1k` 初始化。官方 train 51,396 张中分层抽取 41,116 张训练；官方 valid 5,849 张只用于 AUROC 选模；test 5,855 张只用于最终预测。实际命令：`D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/code/train.py --epochs 7 --batch-size 8 --accumulation 8 --workers 2`。运行目录 `runs/submission_new/20261102_133834`，原始日志 `runs/submission_new/launch_20261102_full.out.log`。7 轮已完整结束，官方 valid AUROC 依次为 0.957677、0.977851、0.982447、0.983859、**0.9863724439657326**、0.984559、0.984456；第 5 轮最佳，模型 `submission_new/model/best_model.pth` SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`，`training_complete=True`、`metric_scope=official_valid`。该 AUROC 高于既有正式验证基线 0.9859689935392411，但官网分数未知，不能等同报告作者的 98.55。

已用该最佳权重显式预测 `submission_new/submission.csv`，2026-11-02 15:58:32 北京时间生成，SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`。独立核验 5,855 行、官方 test 顺序、三列表头 `Image_name,Probability,Pred_Label`、概率范围与 0.5 标签，错误 0。无 TTA、外部医学数据、医学预训练或测试调参。该实现与报告作者仍有结构、划分、轮次和选模差异，属可运行近似复现。比赛平台分数及指定设备每张 100ms 计时尚未实测。共享 GPU 已于 16:01 确认空闲并交还胸片任务；本日不再重复启动训练。预训练文件 SHA256 `cb43de67007c925cddcddea01655191aeb805dfd9ec76eb1455e8b8de398e8a6`。

### 2026-11-02 19:36 本机速度补测

胸片任务明确释放 GPU 后，在同一台 RTX 5060 Laptop 8GB 上执行 `D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5`。计时图片为官方 valid 的只读副本，5 张预热、100 张×3 轮；单张从读图、预处理至模型输出含 CUDA 同步，与预测共用 `score_batch`。实测平均 **0.057311 秒/张（57.311 ms/张）**，低于 100 ms 参考门槛。此结果仅代表本机与该图像集，不冒称比赛指定设备正式验收。19:36 确认无 Python 进程、GPU 显存 26 MiB，并明确归还胸片任务。此前文档“速度尚未实测”已由本段更新。

## 2026-11-02 晚间目录归档与续训修复

用户反馈本候选官网 **97.65**（ID 29668，20:57:24；[截图记录](../项目文档/提交记录/20261102_submission官网97.65.md)），低于报告作者 98.55。报告 g07 实际训练13轮、joint选第7轮；本机第5轮官方 valid AUROC 最高，第6/7轮下降，不能据此认定轮次不足。高分报告原文已放入 `项目文档/历史研究记录/`。本目录由 `submission_new/` 重命名为 `submission/`；四个旧方案另存本地 ZIP。运行历史断点仍位于 `runs/submission_new/20261102_133834`，不移动，以保留完整优化器/EMA状态。代码仅修复续训目标轮数及早停状态恢复，不改变模型、数据、预处理、学习率或推理配方；此前训练的权重仍有效。明日同配方执行 `python submission/code/train.py --epochs 25 --batch-size 8 --accumulation 8 --workers 2 --resume runs/submission_new/20261102_133834`，只按官方 valid AUROC 严格提升更新最佳。当前断点第7轮，最佳第5轮，连续未提升2轮。旧 CSV 与权重在今晚不覆盖；详细执行门槛见 `项目文档/当前状态/20261103_submission复现续训方案.md`。
