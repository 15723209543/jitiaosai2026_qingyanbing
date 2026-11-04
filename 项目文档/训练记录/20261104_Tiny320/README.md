# 2026-11-04 Tiny320 与 Small320 候选实验

仅使用非医学ImageNet-1K预训练；官方train 51,396张中按种子20261024分层抽取41,116张训练，官方valid 5,849张仅用于AUROC选模。test 5,855张只在候选选择后预测，无TTA、外部医学数据或测试调参。作者报告的原始源码、权重和精确划分索引本机不存在，因此本实验为注明差异的近似复现。

## Tiny S23 g03 对照

GPU命令：`D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/train_tiny.py --epochs 13 --batch-size 8 --accumulation 8 --workers 2 --feature-noise-std 0`。run `runs/submission_tiny/20261104_095229`，第1–10轮官方valid AUROC：0.956466、0.978117、0.981997、0.984912、0.986096、0.986198、**0.9864871867475788**、0.986446、0.982382、0.982655。第7轮最佳，之后连续3轮未提升；核对完整epoch10断点后只停止本训练进程。最佳候选 `candidate_best_epoch7.pth` SHA256 `dd4253f1ddc6a103478a43b32b21a06d9ff087567be8b95f4936e3930141a647`，官方valid选择标记完整。该PTH大小111,644,359字节，本机同实现单张54.816ms。`candidate_submission.csv`已独立核验5855行、官方test顺序、三列、概率范围与0.5标签，SHA256 `0ff4c6c00cb211f0e5b7987608376d4e2c0218467111b8d8c08f4d4cced6775f`。它的官网分数未知。

## 固定等权融合候选

用户报告曾记录Tiny+Small等权融合；本机在生成test预测前固定概率各0.5。Tiny为上述第7轮，Small为正式 `submission/model/best_model.pth` 第5轮，SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`。完整官方valid AUROC **0.9875753924850881**、AP0.841464854509888、TP146/FP49/FN45；原始机器可读证据 `ensemble_valid.json`，脚本test模式验证该记录及两权重SHA。双模型文件合计309,878,282字节（压缩前），本机同实现单张端到端84.677ms，原始记录 `ensemble_cost.json`；比赛指定硬件仍需验收。

`ensemble_submission.csv`已独立核验5855行、官方test顺序、三列、概率/标签一致，SHA256 `e6013be93b43e37dd6af9654a72f2bd4752f7f23213d7a703d860f02bc833236`，不同于旧版CSV。该CSV的官网分数未知；不根据它调整模型。旧正式CSV备份 `previous_submission_v1.25.1.csv`，SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`。

## 仍在进行

特征扰动SFA独立run `runs/submission_tiny_sfa/20261104_130138`，共享GPU用batch4×累积16、workers1、线程2，优化器有效批量64。第1–5轮官方valid AUROC：0.956072、0.978962、0.982521、0.985726、0.987248；第5轮仍低于固定融合0.987575。第6轮结果待完成后据实补充。若最终不超过融合，以已验证融合为正式交付；若超过，须重新测速度与生成CSV。以上所有选择只按独立官方valid AUROC。

## 最终选择
SFA第6轮官方valid AUROC0.987096，低于第5轮0.987248，也低于预先固定的Small+Tiny各0.5融合0.9875753924850881。SFA在第6轮完整last.pth写入后提前停止，仅保存实验候选。正式选融合，不用测试预测或官网成绩选模。正式推理和计时共用code/ensemble_tiny_small.py；两PTH合计309878282字节、本机84.677ms/张。正式CSV重新由正式入口产生并独立核验，SHA以最终校验结果为准；平台成绩尚未知。

正式CSV已于2026-11-04 16:44:10北京时间从正式部署入口重新生成，独立验证5855行官方顺序及三列/阈值，SHA256 e6013be93b43e37dd6af9654a72f2bd4752f7f23213d7a703d860f02bc833236，与上文融合候选逐字节一致。官网分数未知。
