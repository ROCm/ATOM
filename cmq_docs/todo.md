# ATOM Wiki TODO

本文件记录已确认、暂不实施的后续事项。条目列入 TODO 不代表扩大当前工作范围；实施时需要结合当时的目标重新评估。

## Metrics：历史逐请求指标治理

- **状态：暂缓，不属于当前 metrics 精简修改。**
- **范围：** `atom:prefill_request_context_tokens`、`atom:decode_request_context_tokens`。
- **来源：** [Metrics review 第 5 节](metrics_review_3ff61024_c2c68165.md)。它们在 `3ff61024b^` 已存在，不是本轮提交区间引入。
- **问题：** 使用 `request_id`、`sequence_id`、`started_at` 标签，已完成请求保留到 metrics storage 清理；历史时序持续积累，增加存储与 scrape 序列化成本。
- **待评估方案：** 常规运行保留请求长度分布 histogram；逐请求散点数据考虑独立诊断开关、有数量上限的采样或离线 trace。同步评估报表散点图、表格和 CSV 的用途。
- **范围决议（2026-09-17）：** 本轮只优化 `3ff61024b` 至 `c2c68165d`（包含起始提交）新引入的修改，不改变这两个历史指标。

实际性能 A/B 已取消，不是待办。关闭 trace 的包装由 `3ff61024b` 引入，属于本轮范围，处理记录见 review 第 8 节。
