# Agentic 请求、调度与 KV 指标

这次增加的四类指标沿用现有 `/metrics → Prometheus → 离线 HTML` 链路。
Agentic PD CI 自动采集，不需要额外开关；服务需要使用更新后的代码重新启动。
TTFT、ITL 的名称和采集边界保持不变。

| 类别 | 指标 | 类型与单位 | 统计口径 |
| --- | --- | --- | --- |
| 请求排队 | `atom:request_queue_time_seconds` | Histogram，秒 | 引擎收到请求到首次真实 forward 下发前的全部等待，包含 KV 加载等待。一个执行过的请求观察一次。 |
| 队列长度 | `atom:scheduler_requests{state="running\|waiting\|waiting_kv"}` | Gauge，请求数 | Running 是调度器持有的运行请求；Waiting 不含外部 KV 等待；Waiting KV 是外部 KV 加载或共享缓存 prefill 等待。 |
| 实际 decode batch | `atom:decode_batch_size` | Histogram，请求数 | 每次有 decode 请求的真实 forward 观察一次，以 `ScheduledBatch.total_seqs_num_decode` 为值。 |
| PD 传输等待 | `atom:pd_kv_transfer_seconds` | Histogram，秒 | D 侧请求进入远端 KV 等待，到 scheduler 收到全部 worker 完成报告。只统计成功的 PD KV 加载。 |
| KV block 状态 | `atom:scheduler_kv_cache_blocks{state="used\|evictable\|vacant\|total"}` | Gauge，block 数 | 每个调度器管理的 KV 块池，按占用、可回收缓存、空闲拆分。 |
| 快照时间 | `atom:scheduler_snapshot_timestamp_seconds` | Gauge，Unix 秒 | engine 生成快照的时间，用来检查忙碌时采集是否滞后。 |

新指标保留 `dp_rank` 和 `engine_role` 标签。普通 connector 服务的
`engine_role="default"`，进程内共享缓存 P/D 分别为 `prefill`、`decode`。
CI 通过 scrape target 添加的 `role="prefill"/"decode"` 才是报告区分服务角色的标签；
`instance` 区分服务器地址。没有请求 ID 标签。

## 排队与传输的边界

例如一个 D 侧请求经历：分配前等待 2 ms、外部 KV 等待 8 ms、完成后等待调度
3 ms，则排队直方图记录 **13 ms**，PD KV 传输等待直方图记录 **8 ms**。
PD KV 等待属于总排队时间的一部分，两者不能相加。

排队时间从 EngineCore 接收线程的 socket 收到请求开始打点，包含反序列化、
引擎内部输入队列、调度准备、KV 加载以及加载完成后的等待；Scheduler 接收请求时
保留原始时间戳。终点在首次 CPU forward 下发前，不是 GPU kernel 开始执行时刻。
直接调用 Scheduler 的场景没有 socket 接收步骤，则以 Scheduler 入队时间为起点。
引擎收到请求前的 Mesh 排队、HTTP 输入处理和 tokenizer 耗时不在该区间内。
Chunked prefill 的后续 chunk、后续 decode 和发生在首次 forward 之后的
抢占重入不会重复观察初始排队时间。首次执行前被拒绝或取消的请求没有排队样本。

PD 指标是**消费者观察到的 KV 加载等待**，其中包括下发工作、握手、发送端等待、
实际传输、完成通知和 scheduler 轮询延迟；它不是纯 RDMA 调用耗时或网络单向时延。
起止均使用 D 侧 `perf_counter()`，无需跨机器时钟同步，也不通过 P/D TTFT 相减计算。
TP 完成聚合，以及传输后端已有的 PP 完成协议，先完成后再记录一个请求样本。
失败、取消后的清理、LMCache offload 和共享缓存 prefill 不产生成功 PD 传输样本。
纯 RDMA 耗时仍需在各传输后端单独增加计时。

## 实际 decode batch

在 engine 下发真实 forward 时采集，不在 `schedule()` 提议批次时采集。
普通 engine、PP head 和共享缓存 disaggregation 的 forward 路径均接入；
PP downstream、DP dummy execution、空 connector batch 和纯 prefill batch 不计入。

例如一次 MTP forward 有 4 个请求、每个请求验证 4 个 token，batch 样本仍为 4。
图捕获的 padding 也不计入。混合批次只取 decode 行数。
Mean/P50/P90/P95/P99 是按 forward 次数统计的分布，不是按 token 或执行时间加权。
没有 decode forward 的窗口保持无数据，不用上一批的 batch 值填充。

## KV block 利用率

复用 `BlockPool.num_used / num_reusable_free / num_free / num_blocks`：

```text
used       = num_used
evictable  = num_reusable_free
vacant     = num_free - num_reusable_free
total      = num_blocks

used + evictable + vacant = total
used / total = KV 块池当前占用率
```

`evictable` 是空闲但仍保留缓存内容的块，可以被后续请求复用，也可重新分配；
缓存保留不等于请求占用了这些块。`indexed` 可能同时包含使用中和空闲的块，
不能直接拿它当作占用量。

保留原有 `atom:kv_cache_usage_ratio`、`atom:kv_cache_blocks_*` 和请求数指标，
补充 `atom:kv_cache_blocks_evictable`、`atom:kv_cache_blocks_vacant`。
报告按块池容量汇总 Used/Cached/Vacant 百分比。
同一个 KV 占比面板内同时显示该角色的 **Used 总数 / Total 总数（blocks）**。
默认显示所选时间范围内最新采样点；鼠标悬停时跟随时间显示对应数量。
Used 和 Total 使用相同时间点的原始采样，不从百分比反推；缺失数据显示 `—`。
表格和 CSV 同时保留数量，单位为 `blocks`，不增加额外面板。
无独立块池的共享缓存 P 进程不会伪造一个空池；这是块池使用率，不是显卡 HBM
使用率，也不代表 block 内 token 填充率。TP/PP 的复制或分片不会再次增加逻辑块数。
如果启用了共享池中的状态缓存，占用原始 PAGE 单元的状态对象也属于 used。

## 采集与报告

Scheduler 在每个事件发生时更新累计直方图。Engine 和 API 每秒更新一次快照，
CI Prometheus 每秒 scrape；PP head 也会推送快照。
抓取或刷新不会再次 observe、清空计数，也不会触发同步 engine RPC 或 GPU 同步。
长 prefill 或 engine 忙碌可能推迟快照；瞬时队列尖峰仍可能落在两次快照之间。
直方图事件不会因快照间隔而丢失。

报告保持两列布局。直方图展示最近 60 秒的 Mean/P50/P90/P95/P99，
队列和 KV 展示采样时刻的状态；默认图表步长为 5 秒。
所有曲线可单独开关，表格、提示框和 CSV 标注各自单位。
百分位来自直方图桶估计，不能相加；容量汇总也可能掩盖某个 DP rank 的高占用，
可按 `instance,dp_rank,engine_role` 查询单个池。

PromQL 示例（与 CI scrape 标签一致）：

```promql
# D 初始排队 P99，ms
1000 * histogram_quantile(0.99,
  sum by (le) (rate(atom:request_queue_time_seconds_bucket{job="atom",role="decode"}[60s])))

# D 实际 batch 均值，请求/forward
sum(rate(atom:decode_batch_size_sum{job="atom",role="decode"}[60s]))
/ sum(rate(atom:decode_batch_size_count{job="atom",role="decode"}[60s]))

# 成功 PD KV 加载等待 P99，ms
1000 * histogram_quantile(0.99,
  sum by (le) (rate(atom:pd_kv_transfer_seconds_bucket{job="atom",role="decode"}[60s])))

# D 占用率，按总容量加权，%
100 * sum(atom:scheduler_kv_cache_blocks{job="atom",role="decode",state="used"})
/ sum(atom:scheduler_kv_cache_blocks{job="atom",role="decode",state="total"})

# 各队列当前请求数
sum by (role,state) (atom:scheduler_requests{job="atom"})

# 各 engine 快照年龄，秒
time() - atom:scheduler_snapshot_timestamp_seconds{job="atom"}
```

## 源码对照

参考本地 SGLang `db3da62333`：

- `python/sglang/srt/observability/req_time_stats.py` 的
  `set_wait_queue_entry_time / set_forward_entry_time / get_queueing_time`：
  首次 forward 记录排队时间。ATOM 按当前指定口径，从引擎收到请求开始累计，
  包含输入队列、分配前等待和外部 KV 等待，因此范围比其“进入 ready queue 后”更广。
- `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` 的
  `_build_scheduled_request_metrics`：区分实际 prefill/decode 请求行。
  参考实现通过 ForwardPassMetrics 发布逐次执行数据；ATOM 接入现有 Prometheus。
- `python/sglang/srt/observability/req_time_stats.py` 的
  `compute_and_observe_kv_transfer_metrics`：优先使用后端传输时间，同时区分
  bootstrap/alloc；无后端时间时回退到传输队列时间，源码注明只覆盖最后一个 chunk。
  ATOM 当前提供上文定义的消费者加载等待，不声称与该后端传输指标等价。
- `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`：
  将可用、可回收、已使用 KV token slots 分开；ATOM 对应到本身的 block 池状态。

实现入口为 `atom/model_engine/scheduler_metrics.py`、各 engine 的 forward 下发处、
`engine_utility.py → llm_engine.py → entrypoints/openai/metrics.py`。
HTML 和 CI 导出位于 `.github/scripts/atomesh/observability/`。
