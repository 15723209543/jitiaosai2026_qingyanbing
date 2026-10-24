# 第二轮续训来源

本轮从第一轮最佳比赛模型继续，未使用外部预训练。

起点：previous_round_best_0.957992.pth，SHA-256 `134650d23ce1f2c48bdf2e9abf612faff61b3ccf6cd0b39530cba8c1963b8eaf`。

底层GitHub源模型仍为v1.12.1，原始文件与来源见UPSTREAM_CHECKPOINT.md。第一轮代码与训练记录随材料保留。

继续训练恢复了本机 runs/20261024_upstream_dual/*_last.pth 中与源参数匹配的优化器、AMP和随机状态。优化器恢复文件保留在本机，不纳入最终推理材料。完整训练复现需要先按第一轮代码完成5轮，得到匹配状态，再运行本轮代码。
