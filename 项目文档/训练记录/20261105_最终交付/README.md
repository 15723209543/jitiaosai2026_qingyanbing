# 2026-11-05 单Tiny最终交付（v1.26.2）

时间均为北京时间。选模仅使用**完整官方valid 5,849张的AUROC**；AP只记录，不使用test预测、历史平台分数或测试标签调参。正式CSV与模型来自同一冻结第3轮EMA权重，官网成绩尚未获得。

## 正式文件

- `submission/model/tiny_best_model.pth`：配方`s23_g03_tiny320_randaugment_pairauc_balanced_from_soup_v1`，原run`runs/submission_tiny_pair_auc/20261105_095121`第3轮EMA。官方valid AUROC **0.9897295956797492**；SHA256 `0feb9619de1c5316bb2d0947bd5ea122fe9957f77d36a7bef174468f2da516c8`；111,644,807字节。不可变备份`runs/submission_tiny_pair_auc/candidate_epoch3.pth`，与正式PTH哈希相同。
- `submission/submission.csv`：2026-11-05 18:27:12生成，5,855行，官方test顺序、表头`Image_name,Probability,Pred_Label`、概率范围、0.5标签阈值独立核验；SHA256 `e6d1a5ef4dbdc35fe9f755d386e8b61af3b6f10ff598d7ba50fd8c02aeee17ea`。使用正式`submission/code/predict.py`显式加载正式Tiny权重重新生成，与第3轮备份CSV哈希一致；与上一正式CSV不同。旧v1.26.1 CSV备份为本目录`previous_v1.26.1_submission.csv`，原SHA256 `e6013be93b43e37dd6af9654a72f2bd4752f7f23213d7a703d860f02bc833236`。
- `submission/model/tiny_cost.json`：正式单Tiny与预测相同的`score_batch`，在本机RTX 5060 Laptop用5张预热、100张官方valid副本×3测得平均**43.439668ms/张**。比赛指定硬件尚未验收。正式`model/`还保留旧Small权重供回退；两PTH合计约310MB，低于500MB，但当前正式预测只加载Tiny。

## 实验序列和选择

此前v1.26.1正式Small+Tiny融合官方valid AUROC0.9875753924850881，用户反馈官网97.97。11月4至5日的Tiny RandAugment、低LR、参数平均、成对AUC逐步提升，过程证据见`../20261104_通宵实验/README.md`与`submission/RECIPE.md`。最高第3轮之后：同配方第4/5/6轮分别约0.989682/0.989656/0.989608，连续三轮未提高而正常早停。温和成对AUC独立run第1/2/3轮约0.989632/0.989522/0.989427；全量官方train（51,396张）独立run第1/2轮约0.989619/0.989592，均未超过第3轮。固定50/50第3轮EMA与全量第2轮EMA参数平均官方valid AUROC0.9896916565341388，亦未超过第3轮。所有新增run均从当时最高兼容已训练权重开始，未把官方valid并入训练；最高候选权重在每次提高后保存为不可变副本。

## 复现与限制

先`git lfs pull`并确认官方数据只读映射。仓库根目录运行：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/predict.py --output submission/submission.csv --workers 1
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5 --output submission/model/tiny_cost_recheck.json
```

该方法是用户高分报告思路与ICCV2021 AUC优化思想的本机近似实验，**没有复现作者98.55官网分数**。本机官方valid AUROC未达到用户目标0.995；平台是否达到98.5只能在上传本次CSV后获知。原始数据与junction目标只读，无外部医学数据或医学预训练权重、无TTA、无test调参。