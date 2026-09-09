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

## 缓存有效性与实际工作量

新增三个调度直方图，在真实 forward 下发处采集，dummy 和空批次不计入：

| 指标 | 单位 | 口径 |
| --- | --- | --- |
| `atom:prefill_request_tokens` | tokens | 请求首次执行本地 prefill 时，`num_prompt_tokens - batch.num_cached_tokens`，最小为 0；每个请求一次。后续 chunk 或抢占重入不重复计数。 |
| `atom:prefill_batch_tokens` | tokens | 每次真实 forward 的 `total_tokens_num_prefill`；每个 chunk 分别计数，排除已缓存前缀、decode tokens 和图 padding。 |
| `atom:decode_context_tokens` | tokens | 每次真实 forward 中 decode 行的 `batch.context_lens` 之和；不乘 TP 数，也不将同一请求按 MTP token 数重复计算。 |

例如输入 100K tokens、首次 prefill 时已有 95K KV：请求直方图记录 5K；
若分成 2K、2K、1K 三个 chunk，下发直方图记录三个样本。后者统计实际调度的
计算，包括抢占后的重新计算；前者描述首次本地 prefill 的未缓存输入量。
Decode 上下文长度是逻辑长度，稀疏 attention 下不等于实际读取 KV 的字节数。

缓存面板默认同时展示 **Total reuse / LMCache / GPU 三条曲线**，使用同一批缓存记账中的三个计数：

- `atom:prefix_cache_cached_tokens_total`：GPU/HBM 已接受的前缀复用 token。
- `atom:prefix_cache_offload_tokens_total`：LMCache 在 GPU 前缀之外补充的、实际接受用于复用的 token。
- `atom:prefix_cache_full_tokens_total`：参与缓存统计的完整输入 token。

总复用率为最近 60 秒 `(GPU 增量 + LMCache 增量) / Input 增量`，按 token 加权。
GPU 与 LMCache 的分子互不重叠，沿用调度器的同一次记账；不使用传输完成时统计的
`atom:lmcache_loaded_tokens_total` 替代 LMCache 复用量。PD 远端加载也不直接归入 LMCache。
分母包含必须计算的尾部输入，和 `/cache_stats` 使用的 reusable-token 分母不同。

本地 prefill 在准入时更新缓存计数；纯 PD 的 D 端在 KV 接收完成、首次 decode
准入时补记一次，使用 D 原先已有的本地前缀，且在注入 P 的首个输出 token 之前
记录输入量。刚收到的远端 KV 和 API 中继承自 P 的 `prefix_cache_hit_tokens`
不计入 D 的本地命中；后续 decode 步骤和重复完成通知不重复计数。中止且未执行
的请求不计；传输失败回退时由实际发生的本地 prefill 路径记账。

**D 端缓存命中率不等于 PD 传输节省比例。** 当前 Mooncake 只有在两侧
`hash_block_size` 相同、请求不带需完整传输的独立状态等条件满足时，才按 D
已有前缀跳过对应块。CPP4→DCP4 默认两侧为 16/64，会回退到全量传输。
本次只补统计，不改变上述传输行为。

三条曲线都直接画在原面板中，通过图例独立开关；同时显示 **Reused / Input** 和
**LMCache / GPU tokens**。例如 Input=10K、GPU=6K、LMCache=3K，则总复用率为 90%，GPU=60%、
LMCache=30%。两种贡献的分母都是 Input，不是两者占已复用 token 的内部份额，
也不是 LMCache 后端查询成功率。表格和 CSV 同步包含贡献比例和 token 数量。

未启用 LMCache 时，上报的补充复用量为 0；旧快照缺失该计数时保留未知。
PromQL 检查各 tier 与输入计数的服务样本覆盖是否一致，避免混合版本部署中
缺少部分 LMCache 数据却显示偏低的总复用率。旧 JSON 的 `hit/cached` 仍保持 GPU-only 含义。
这些数量来自 Prometheus `increase()`，是按采样外推的窗口估计值，展示时四舍五入；
不是请求级精确审计记录。分母为 0 或没有样本时留空，不伪造 0% 命中率。
本地前缀复用和 PD 远端加载是不同来源，不将 PD 加载量直接记为前缀命中。

## GPU forward 计时

`atom:gpu_forward_seconds` 是设备事件直方图，标签包括
`dp_rank, pp_rank, tp_rank, engine_role, phase`；`phase` 分为
`prefill/decode/mixed`。每个 model runner 在初始化、profiling 和 warmup 完成后
启用采集，用当前 CUDA/HIP stream 上的两个计时事件包围真实的 `run_model`。
支持 eager、graph replay 和共享 GPU P/D 所选择的 stream。

该区间涵盖目标模型、logits 计算，以及区间内的设备 stream 通信和等待；
不包含 `prepare_model`、采样、MTP drafting 或 CPU 输入队列。它是 GPU stream
上的经过时间，不是所有 kernel 执行时间之和，也不等于计算单元利用率。
**PP 下一个样本代表一个 worker 的本地 stage，不是整个 PP 请求耗时。**
报告默认合并所选服务内 worker 的直方图，形成 worker forward 时延分布；
不会把 TP/PP 耗时相加为请求延迟。需要定位某张卡时，可在 Prometheus 中按上述
rank 标签查询；报告的实例选择器定位服务端点，不定位服务内部的某张卡。

事件结果通过 `query()` 就绪检查读取，没有增加 `synchronize()`。每个 worker
最多保留 256 对未完成事件，事件仅完成后复用。若积压达到上限，跳过后续计时
并增加 `atom:gpu_forward_dropped_total`，不阻塞推理；失败 forward 不记成功样本。
`atom:gpu_forward_pending` 显示未完成计时数，
`atom:gpu_forward_snapshot_timestamp_seconds` 标识 worker 快照时间。

Engine 每秒异步请求一次设备计时快照；结果通过现有 worker 输出通道中的独立
消息类型传回，不进入 forward 结果队列或 KV quorum 聚合。PP 每个 stage 都推送
设备快照，只有 head 推送调度器计数，避免重复累计请求、缓存和 KV 池。
快照与 API 刷新存在延迟，设备直方图记录的是已完成且已上报的事件。

## 多实例看板

报告默认进入 **Overview**，展示 P/D 各四项：TTFT、排队时间、缓存命中、GPU
forward。可切换 **Latency / Workload / Cache & KV / All metrics**，当前共 19 个
指标面板。保留 All / Prefill / Decode 角色筛选；共有指标在桌面上 P 左 D 右，
移动端按相同顺序纵向排列。全局 Statistics 仍只有 Mean/P50/P90/P95/P99，
队列状态、KV 状态和缓存命中开关留在相应面板。

P、D 各有一个独立实例选择器，默认 **All instances**；服务使用 Prometheus 的
`instance`（主机:端口）标识。同一机器上的两个端口可以单独选择，例如同时比较
`P-A:8010` 和 `D-B:8020`。一个跨机器部署的 TP/PP 服务仍是一个 API 实例，不能
把其 API 地址误解为所有 GPU 的物理地址。配置过但无数据的端点仍保留在选项中。

导出时同时保存角色汇总和 `sum by (instance, le)` 等查询产生的实例数据。
**汇总分位数由合并的直方图计算，汇总命中率和 KV 占比按 token/容量加权**，
不对各实例的 P99 或百分比取算术平均。浏览器选择实例只切换已有数据，不需要
网络请求。某实例缺失的数据保留为空，绝不回退到其他实例或汇总曲线。
表格、悬停数值、窗口数量和 CSV 同步切换，CSV 带 `instance` 列。
Mesh TTFT 保持全局统计，不声称筛到了某个 P/D 路由子集。

JSON 的每个 panel 可增加 `instances: {"host:port": {"series": ..., ...}}`；
新缓存面板带 `cache_breakdown: true`、`series.reuse/gpu/lmcache`，以及
`cache_counts.reused/prompt/gpu/lmcache`；旧缓存面板的 `cache_counts.cached/prompt`
仍可读取。KV 面板继续使用 `block_counts.used/total`。
旧 JSON 缺少实例数据时，汇总图仍然可用，实例选择器显示没有分实例数据。

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
