# 发表可行性判断与修回策略

## 总体判断

当前结果可以作为修回论文的主结果继续推进，但必须改变论文主张。

建议主张：

- 本文提出并审计了一个防泄漏的 subject-specific sEMG 膝关节角度预测协议。
- 在 `purged_temporal`、train-only scaler、validation-only selection 下，四通道增强 Transformer 的主结果为 `MAE=6.2966 deg`、`RMSE=9.0339 deg`。
- 相比同一严格协议下的两通道 Transformer，四通道增强模型明显改善：`MAE 9.0575 -> 6.2966`，`RMSE 13.9788 -> 9.0339`。
- 传统基线已补充：RF `MAE=7.9140`，SVR `MAE=8.9029`，XGBoost `MAE=7.9779`，均弱于四通道 Transformer。
- 泄漏审计 `110/110` safe，可以支撑严格协议下的论文结论。

不能再这样写：

- 不能声称当前防泄漏结果超过原 Table I 的 `MAE=3.707 deg`、`RMSE=4.691 deg`。
- 不能把旧的 random-overlap/full-scaler 结果作为主结果。
- 不能把 subject-specific 模型包装成跨受试者通用模型。LOSO ridge 的结果为 `MAE=18.0899 deg`、`RMSE=23.5735 deg`，说明跨受试者泛化仍然较弱。

## 审稿意见覆盖情况

已经解决：

- 研究对象统一表述为 22 名男性受试者，11 healthy + 11 pathological。
- 论文主张从“追求原 Table I 数值”改为“防泄漏协议下的可信评估”。
- 加入 BF/ST，主实验从 RF/VM 两通道扩展为 RF/BF/VM/ST 四通道。
- 补充 dataset summary、每个 subject 窗口数、malformed rows、实验时长。
- 补充 feature correlation matrix。
- 补充 MLP、Transformer、CausalGaitNet、RF、SVR、XGBoost、LOSO ridge 对比。
- 补充 healthy/unhealthy bootstrap CI 和 permutation test。
- 补充 example prediction figures，明确 healthy/pathological。
- 补充 protocol leakage diagram 和 leakage audit。
- 补充多尺度消融：`scales=1` 得到 `MAE=6.6415`、`RMSE=9.7923`；多尺度 `scales=1,2,4,8` 的 matched seed-42 结果为 `MAE=6.3191`、`RMSE=9.0362`。
- 补充鲁棒性测试：Gaussian noise 基本不影响结果；channel dropout 明显劣化；amplitude scaling 中等影响。

## 投稿前剩余工作

必须完成：

- 把 `revised_manuscript_insertions.md` 中的文本合并到论文 Word/LaTeX 源文件。
- 按 IEEE 要求使用 Track Changes、高亮或下划线标记改动。
- 将 `response_to_reviewers.md` 作为逐条回复上传。
- 将 `cover_letter_to_editor.md` 放入 cover letter 或 File Upload 说明。
- 检查新增参考文献格式和 DOI。

## 推荐投稿定位

建议将论文定位为：

> A leakage-aware subject-specific evaluation of Transformer-based knee-angle prediction from lower-limb sEMG.

核心结论建议写成：

> Four-channel sEMG and extended features improve strict subject-specific prediction under a leakage-safe protocol, but the revised leakage-safe performance is lower than the originally optimistic overlapping-window result. This highlights the importance of protocol auditing for sEMG time-series prediction.
