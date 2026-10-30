# submission3 暂停检查点恢复预测（2026-10-29）

版本：V1.18.1。用户暂停了本机 `submission3` 训练，明确要求使用当前已保存参数生成测试集预测。此操作不继续训练，也不修改原 `best_model.pth`。用户报告“训练成绩96.78下降”；目前没有官网上传文件哈希或评分凭据，不能把该分数归因于本次刚生成的恢复预测 CSV。

## 固定输入与结果

- 原暂停检查点：`D:/青眼病题/submission3/model/best_model.pth`。已冻结复制到 `submission3/model/paused_20261029.pth`，SHA-256：`cb023aff771674485ce42922308eab309c576a1c2a6519ee123f3376f1c1ba16`。
- 检查点有 ConvNeXt-Tiny 与 EfficientNetV2-S 两个分支的参数，但 `selection=None`、`training_complete=False`。本次恢复预测明确固定为两分支**阳性概率等权平均**，不宣称训练完成或最终选模已完成。
- 两个分支记录的 AUROC 约为 0.999998 和 0.999708，评价数据也参与了训练，是**集内指标**，不能与独立验证成绩或官网成绩比较。
- 输出：`submission3/submission.csv`，SHA-256：`624162a328c1661e43807a952c8e25967a2ae920cb891653922fbb1727c05f2d`。5,855 行，与官方 `test.csv` 名称及顺序逐行一致；三列格式、概率范围和 0.5 标签均通过核验。本次报阳 182 张。这是预测文件，未在官网提交或取得官网成绩。

## 复现命令

在仓库根目录、具备真实权重和官方测试图的机器上执行：

```powershell
python submission3/code/predict.py submission3/submission.csv --checkpoint submission3/model/paused_20261029.pth --recover-paused
```

默认入口仍拒绝未完成的检查点。`--recover-paused` 仅在检查点恰有两个完整分支时允许固定均值预测，防止把单分支阶段性权重误作最终模型。原训练进程若恢复并改写 `best_model.pth`，不会改变这里的冻结快照。
