# ATOM Wiki

这里汇总工作过程中积累的 ATOM 代码分析。文档按主题持续补充，每个章节标明核对过的代码版本、适用路径和源码入口，便于后续维护。

## 主题文档

| 文档 | 已有内容 |
| --- | --- |
| [API Server 分析](api_server_analysis.md) | Streaming callback 的注册与调用、线程边界、token 交付、SSE 输出和可选的合并 output delivery 计时 |

## 专项评审

| 文档 | 范围 |
| --- | --- |
| [Metrics review：3ff61024b 至 c2c68165d](metrics_review_3ff61024_c2c68165.md) | 指标价值、采集成本、统计口径、讨论意见与代码落地记录 |

## 实验资料与后续事项

| 入口 | 内容 |
| --- | --- |
| [Metrics review 实验归档](experiments/metrics_review_3ff61024_c2c68165/README.md) | 精简前微基准原始脚本、JSON、源码快照、复现说明与校验清单 |
| [TODO](todo.md) | 历史逐请求 context 指标治理，暂缓实施；当前范围仅处理评审提交区间新增的修改 |

后续 API Server 相关分析直接追加到对应主题文档；其他组件的分析新建主题文档并登记到此索引。当前行为应附源码依据，尚未实施的优化建议应明确标注；代码变化时更新受影响章节及其版本说明。
