# 第十轮焦点分类头训练来源

起点previous_round_best_0.963044.pth，SHA256 f7d35a6bb77819750d4b59fdeac0f273b775578ed8ec1bf39e6bafa525a7f6f5。

新建分类头AdamW与随机状态，冻结卷积与BN，FP32训练，仅在GPU训练分类头，学习率1e-4，未改变特征网络，未使用外部预训练。上游与早期来源文件仍供追溯。最终推理仅加载model/best_model.pth。
