**Metrics review：`3ff61024b` 至 `c2c68165d`（包含起始提交）**

> 实现状态（2026-09-17）：已根据讨论完成工作区代码修改；实际保留项、开关和验证结果见第 7 节，最新范围决议、trace 补充修改和资料归档见第 8 节。实际性能 A/B 已取消，历史逐请求指标仅列入暂缓 TODO。前六节保留评审时的代码事实、原始建议与用户意见，不能作为修改后的指标清单或当前待办。

本轮只做代码评审、开销分析和精简建议，不修改运行代码。评审以 `c2c68165d02c0db0be46dcdb780570237359415c` 为最终状态，比较基线为 `3ff61024bca742c8864daf7a91d806a352d9a5b0^`，避免漏掉起始提交新增的埋点。重点是 metrics 的生产、传递、导出和报表消费；区间内 PD token ID 复用等功能仅检查其对指标口径的影响。

建议优先让高频细分诊断按需开启，再删除用途有限的指标。当前证据能说明哪些埋点有额外成本，尚不能说明 metrics 是实际 host gap 的主要原因，也不能给出端到端吞吐提升比例。

**1. 新增指标及建议**

区间内新增 15 个 metric family：API 9 个、scheduler 1 个、GPU 2 个、CPU 3 个。`mtp_average_tokens_per_forward` 在此区间新增的是报表展示，生产指标已经存在。

以下名称均省略 `atom:` 前缀。

| 指标                                | 采集频率 / 当前开关                                           | 价值与精简建议                                                                                   | 优先级   |                                                                                                      |
| --------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------- |
| `api_detokenize_chunk_seconds`    | 每次实际 detokenize；默认开启，collector 合并后的 chunk 才执行解码       | 最优先从默认路径移除。已有 ITL 反映交付延迟，解码归因需要时再启用或抽样；ITL 不能替代其归因作用，但不必一直同时全量采集                          | 高     | detokenize 耗时都很小，不建议继续观测，不会形成优化指导                                                                    |
| `gpu_sample_seconds`              | 每个执行 sampling 的真实 worker forward；受 device timer 控制    | 增加独立的细分诊断开关，普通 GPU forward 计时不自动附带它                                                       | CI 下高 | 可以增加独立的细分诊断开关，普通 GPU forward 计时不自动附带它                                                                |
| `gpu_propose_seconds`             | 每个执行 MTP propose 的真实 worker forward；同一开关              | MTP 调优时有价值，保留按需采集；不建议永久删除                                                                 | CI 下高 | 可以增加独立的细分诊断开关，普通 GPU forward 计时不自动附带它                                                                |
| `ttft_output_to_callback_seconds` | 单序列 streaming 请求首个非空 callback 观测一次，但后续 chunk 仍调用计时和去重 | 默认关闭细分或与下一项合并成首输出交付阶段；保留时把首个 callback 的判断放到读时钟之前                                          | 中     | 可以增加独立的细分诊断开关，可以和ttft_callback_to_sse_seconds 合并成一项，这两个指标可以共同指示decode生成token之后一直到送给api server sse的时间 |
| `ttft_callback_to_sse_seconds`    | 单序列 streaming 首个有效 SSE 内容一次                           | 与上一项共同依赖跨线程 stamp map；细分 IPC 与事件循环交付主要用于诊断，可按需开启                                          | 中     | 可以增加独立的细分诊断开关，可以和ttft_callback_to_sse_seconds 合并成一项，这两个指标可以共同指示decode生成token之后一直到送给api server sse的时间 |
| `api_body_parse_seconds`          | chat handler 入口一次，包括部分最终失败的请求                         | 适合排查大请求 JSON/Pydantic 成本，转为诊断项；不能视为纯 JSON 解码时间                                            | 中     | 这个意料之外地占比比较大，所以建议保留                                                                                  |
| `api_chat_template_seconds`       | 文本 chat 每请求一次；PD token ID 复用时记 0                      | 保留诊断能力，默认关闭细分；PD 复用成功时持续写零的收益较低，但不能因此删除 prefill/fallback 的诊断能力                            | 中     | 这个理论上在decode节点现在是不存在的，可以只在 prefill 节点观测                                                              |
| `api_tokenize_seconds`            | 多个共享 preprocess 路径每请求一次；预分词输入记 0                      | 同上；其实际覆盖范围不只 chat，文档需修正                                                                   | 中     | 这个理论上在decode节点现在是不存在的，可以只在 prefill 节点观测                                                              |
| `api_preprocess_wait_seconds`     | 多个共享 preprocess 路径每请求一次                               | 是 executor wall 减 tokenize，混合了 executor 排队、Sequence 构建和恢复 event loop 的等待，不能单独定位线程池拥塞；转诊断项 | 中     | 这个没有什么性能优化的指导意义，删除                                                                                   |
| `ttft_api_enqueue_seconds`        | 单序列 streaming 每请求一次                                   | 单独展示收益较低，建议默认不采集，或合入入口至入队的总耗时；合并应使用明确的新口径，不能静默改变旧指标边界                                     | 低     | 这个和旧指标的 enqueue 的区别是什么？                                                                              |
| `process_threads`                 | 每次 scrape、每个存活进程                                      | 建议删除。仓库内未找到实际查询消费者；总线程数不等于 tokenizer 并发容量，在当前 Linux psutil 实现中还多读一次 `/proc/<pid>/status`  | 低     | 删除                                                                                                   |
| `process_cpus`                    | 每次 scrape                                             | 当前只是 API 进程 affinity 的 CPU 数；建议改成明确的 affinity 信息并低频缓存，或删除“CPU 容量上限”用途                     | 低     | 一直不变，删除，不具有参考价值                                                                                      |
| `ttft_api_preprocess_seconds`     | 单序列 streaming 每请求一次                                   | 建议保留粗粒度 API 准备阶段，用于区分 API 与 engine 延迟；注明其请求覆盖范围                                           | 保留    | 保留粗粒度 API 准备阶段，用于区分 API 与 engine 延迟；注明其请求覆盖范围                                                        |
| `ttft_forward_to_output_seconds`  | 每个产生首 token 的 sequence 一次                             | 建议保留。它覆盖调度 dispatch 至首输出的 wall time，GPU forward 不能替代                                      | 保留    |                                                                                                      |
| `process_cpu_seconds_total`       | 每次 scrape、每个存活进程                                      | 建议保留，CPU 用量有直接排障价值；先修复同名进程退出时的 counter 语义                                                 | 保留并修正 | 这个感觉对性能提升也没什么参考价值，建议删除                                                                               |

请求级指标通常每请求只写一次。删除它们的 CPU 收益小于同等成本的逐 chunk / 逐 forward 埋点，不能把“指标少了几项”直接换算成性能收益。

**2. 需要修正的评审发现**

**[P2] TTFT 阶段均值不能普遍相加，分析工具还混合了 streaming 与 non-streaming。**

- `atom/entrypoints/openai/api_server.py:1295`、`:1323` 的 preprocess/enqueue 仅在单序列 streaming setup 中观测；fanout 的 `:1537` 只记录 tokenize/wait，没有这两个粗阶段。
- `atom/metrics/scheduler.py:251` 的 forward-to-output 是每个 sequence 一次，包含 non-streaming 和 fanout；API 总 TTFT 是每个 HTTP 请求一次。
- `tools/analyze_ttft_breakdown.py:33` 聚合时丢弃所有标签，`:121` 因而把 `streaming=true/false` 混为一个总 TTFT，再与各阶段均值比较。复现：一个 streaming 请求 TTFT=0.1 s、一个 non-streaming 请求 TTFT=10 s，工具输出 5.05 s 作为比较基准。
- `.github/scripts/atomesh/observability/export_report.py:180` 声称阶段均值相加得到 streaming TTFT，但阶段查询只有 role，没有匹配相同请求集合的能力。

建议把“均值可加”限定在相同请求集合、相同统计窗口和覆盖完整的边界上。分析工具至少过滤 streaming、检查缺失指标和 count；count 相同也只是必要条件。若暂不统一埋点，不应输出带有精确归因含义的 residual。

**[P2] 所谓 TTFT 分段没有覆盖 API 发送返回至 engine 收到请求之间的全部区间。**

`api_server.py:1324` 的 enqueue 到 `add_request()` 返回为止；`engine_core.py:613` 在 ZMQ `recv()` 后才记录 `received_at`，scheduler queue time 从此开始。两端没有同一个公共边界，不能认为 ZMQ 传输时间自动计入 queue time。`docs/ttft_breakdown_guide.md:227` 的相关解释不成立。建议承认它是未归因区间；为精简指标，不必仅为填平 residual 新增一个常驻 histogram。

**[P2] CPU counter 按当前存活同名进程求和，会在进程退出时倒退。**

`atom/entrypoints/openai/cpu_metrics.py:128` 每次 scrape 重新求和，`:141` 跳过已退出进程，没有保留其历史贡献。复现：同名 worker 的 CPU 时间分别为 10 s 和 20 s，总数 30 s；第二个退出，第一个增加至 11 s，导出值变成 11 s。Prometheus 会将下降识别为 counter reset，产生错误的 `rate()`；若整组退出，则序列消失，而非注释所说的保持水平。

长期固定 worker 不一定触发此问题，但 collector 扫描的是所有子孙进程，并且显式支持同名分组，不能用 worker 通常稳定来保证 counter 单调。建议按进程身份维护采样增量并累计到稳定分组，同时正确处理退出和 PID 复用；也可交给已有进程 exporter，并权衡额外标签基数。

**[P2] `process_cpus` 不是 cgroup CPU quota，也未必是整个进程树的 CPU 上限。**

`cpu_metrics.py:158` 仅调用 `sched_getaffinity(0)`，没有读取 cgroup 的 quota/period。容器允许在 64 个逻辑 CPU 上运行但 quota 为 2 核时，这里可能返回 64。不同 NUMA 绑定下，API affinity 也不能代表全部 worker 的 CPU 集合。用它作利用率分母会误判饱和程度。

同时，API 进程 CPU 总量包含 tokenizer/native/executor 等线程；约 1 core 的进程用量不能单独证明 event loop 已饱和。建议保留 CPU cores consumed 的原始读数，修正文案和分母口径。

**[P2] 显式记零不保证分位数图显示零。**

`ttft_breakdown.py:34` 的 TTFT buckets 从 1 ms 开始，没有 0 边界。PD 复用时 `api_server.py:1778` 记录的全零样本落入第一个桶；线性插值会得到 P50≈0.5 ms、P99≈0.99 ms。使用真实 histogram 和分析工具复现，100 次零耗时 tokenization 的 P50 估计为 0.5 ms。只有 mean 是精确的零。

这不是 Prometheus 算法错误，而是该 bucket 配置与“跳过阶段应显示零”的目标不符。保留这类诊断 histogram 时可增加零边界，并为亚毫秒阶段选用合适桶；不应为了绘制零线而在默认路径持续观测零值。

**[P3] 首 token 指标在后续 chunk 上仍有无效工作。**

`api_server.py:720` 对每个非空输出先执行 `time.time()`、`perf_counter()` 和函数调用，之后才由 `ttft_breakdown.py:201` 去重；`scheduler.py:3026` 也在每个 `RequestOutput` 中重复携带同一个首输出 timestamp。收益比 detokenize histogram 小，但属于可以移出稳定 decode 路径的成本。建议只在首输出初始化，或随详细诊断开关整体跳过。

**3. 开销核实与边界**

`ATOM_ENABLE_METRICS_DEVICE_TIMER` 的代码默认值为 0（`atom/utils/envs.py:372`），但 `.github/scripts/atomesh/pd_server_atom.sh:294` 在 `aiperf_agentic` 中默认设为 1。代码默认关闭不代表实际 CI 没有采集成本。

`atom/metrics/gpu.py:169` 的每个被测阶段都会调用 poll、记录一对设备 events，完成后读取 elapsed time 并 observe histogram。对同时执行 sampling 和 MTP 的 worker，本次新增项在原 forward 计时之外再增加两对 events，即每步再增加四次 `record()`。共用最多 256 对 pending events，FIFO poll 不会执行 GPU synchronize；有池化和边界控制，但仍有 CPU/driver 成本。此轮没有可用 GPU 实测该成本。

CPU collector 的进程发现每 30 s 更新，但进程名和 CPU/线程字段仍按 scrape 读取。CI 默认 scrape 为 1 s（`collect_metrics.py:27`）。`AtomMetricsExporter.render_async()` 已通过 `asyncio.to_thread()` 渲染（`atom/metrics/exporter.py:563`），不是直接在 event loop 中读取 procfs；后台渲染仍消耗 CPU，Python 序列化仍可能争用 GIL。`cpu_metrics.py` 中有关 inline render 的注释过时。

在本机 Python 3.10.12 上做了不运行 engine 的微基准，使用真实 Prometheus instruments；每项取 7 次测量的中位数，计时器为当前线程 CPU time。detokenizer 使用固定返回值以隔离观测包装的增量成本，测试不是实际 tokenizer benchmark。multiprocess 组在 import Prometheus 前设置独立临时目录。

| 操作 | 普通 `MutexValue` | multiprocess `MmapedValue` |
| --- | ---: | ---: |
| detokenize 不带观测包装路径 | 0.083 µs/次 | 0.083 µs/次 |
| detokenize 带观测包装路径 | 0.643 µs/次 | 1.418 µs/次 |
| 两者差值 | **0.560 µs/次** | **1.335 µs/次** |
| 请求级 histogram `observe(0)` | 0.461 µs/次 | 1.185 µs/次 |
| 已关闭的 `ttft_trace_span` 空 span | 1.354 µs/次 | 1.351 µs/次 |
| 已记录首 callback 后重复计时和去重 | 0.140 µs/次 | 0.139 µs/次 |
| 仅 9 个 API stage histogram 的导出 | 0.561 ms/次 | 0.999 ms/次 |

9 个 API histograms 在该测试中产生 210 条普通时序或 201 条 multiprocess 时序，差异来自 `_created`。这是测试 registry 的额外时序数，不是整个服务的时序总量。

例如 20,000 次实际 detokenize/s，1.335 µs/次对应约 26.7 ms CPU/s，即约 0.027 core；这是按假设频率计算的量级示例，不是现有业务的测量结果。3 个关闭状态的 trace span 每 worker forward 约 4.05 µs，同样不能直接解释毫秒级 host gap。

微基准脚本和原始输出已从会话临时目录保存至 [实验归档](experiments/metrics_review_3ff61024_c2c68165/README.md)，包含原件、源码快照、便携复现入口和校验清单。不含 GPU driver、实际 tokenization、并发锁竞争或完整服务压测成本。

**4. 推荐落地顺序**

1. API 详细阶段、逐 chunk detokenize 增加一个启动时确定的诊断开关，默认关闭。关闭时跳过计时、stamp、observe 和可选 family 注册；无需在每个 chunk 中反复读取环境变量。
2. GPU forward 与 sample/propose 的细分开关分离。agentic CI 的常规性能基线使用基本计时，完整细分在单独诊断运行中启用；MTP 调优时保留 propose 计时。
3. 删除 `process_threads`，修正 CPU counter 和 affinity 口径。入口、调度、首输出三个粗阶段保留所需诊断能力；enqueue 和 callback 细分默认隐藏且停止采集。
4. 同步修改报表查询、metric 帮助文本和分析工具。关闭的指标应该显示“未采集”，不能以零替代。仅删除 HTML 面板、Prometheus 查询或 exporter 字段，不会消除生产端计时与 observe 成本。
5. 使用相同模型、并发、输入/输出长度、MTP 配置、scrape 周期，比较 detail off/on 的 API/engine CPU、TTFT/ITL 和吞吐；GPU forward 与 sample/propose 分组对照，避免把多项变更混成一个无法归因的结果。

建议保持常开：整体 TTFT、ITL、请求/生成 token 计数、scheduler 队列及 queue time、PD KV wait、缓存容量/命中率和低频 CPU counters。`pd_kv_transfer` 虽是 queue time 的子集，但能区分传输瓶颈与排队，不应因“包含关系”删除。MTP tokens-per-forward 可解释接受效率，且此区间只新增报表展示，移除面板不会节省 engine 埋点成本。

**5. 区间之外、与长期 CPU 成本相关的补充**

`atom/metrics/scheduler.py:110` 和 `:124` 的 `prefill_request_context_tokens` / `decode_request_context_tokens` 使用 `request_id`、`sequence_id`、`started_at` 标签；`:169` 保留已完成请求，直到 metrics storage 清理。它们在 `3ff61024b^` 已存在，不是本轮新增问题。

这些时序随历史请求数增长，每次 scrape 重新序列化，长期服务中比固定数量的请求级 histogram 更值得检查。建议作为下一轮精简对象：常规运行使用长度分布 histogram；逐请求 scatter 数据改成有上限的采样或离线 trace。此项不能归因于本次提交。

其他文档口径也应收紧：本机 `perf_counter` 使用 `CLOCK_MONOTONIC`，同一时钟域下不同进程可以比较，不能笼统写成“跨进程无意义”；若实际涉及不同主机则必须处理时钟偏差，不能直接用 monotonic 替换。GPU 各项是 batch/worker 样本，ITL 是 token 加权样本，MTP 均值也未必与其使用相同权重和窗口；三者只能在明确假设下估算，不能把差值直接命名为 CPU 时间。

**6. 验证记录**

运行了现有测试，包括工作区原有的未跟踪 CPU 测试，未修改测试文件：

```text
python -m pytest -q \
  tests/entrypoints/test_ttft_breakdown.py \
  tests/entrypoints/test_streaming_dispatch.py \
  tests/test_gpu_metrics.py \
  tests/test_scheduler_metrics.py \
  tests/test_observability_report.py \
  tests/test_cpu_metrics.py

182 passed, 3 skipped in 2.52s
```

另外用最小输入复现了同名进程 CPU counter 倒退、分析工具混合 streaming/non-streaming 均值、零耗时 histogram 得到非零 P50。现有测试通过不覆盖这些语义问题。本轮未运行真实 GPU 推理或完整性能 A/B，也未修改生产代码。


**7. 讨论决议与实际落地（2026-09-17）**

| 决议 | 实现 |
| --- | --- |
| 删除低收益常驻观测 | 删除 detokenize、preprocess wait、独立 API enqueue 三个 histogram 及对应计时调用 |
| 移除内置 CPU 观测 | 删除 process-tree collector 及三个 CPU family；性能测试使用外部 pidstat / process exporter，affinity 和 quota 作为运行元信息保存 |
| 合并 callback 细分 | 新增 `atom:ttft_output_delivery_seconds`，直接测 scheduler 首输出至 API 首个有效 SSE 内容；替代两个旧 callback histogram |
| 输出交付诊断默认关闭 | `ATOM_ENABLE_METRICS_OUTPUT_DELIVERY=1` 在 API 和 engine 启动前开启；关闭时不注册该 histogram，不记录诊断 wall timestamp |
| 清理逐 chunk 诊断开销 | 删除 callback 中的两次读时钟与全局去重表；启用诊断时用请求自身的 RequestTiming 接收首个 scheduler stamp，scheduler 只传递一次 |
| GPU 细分独立开关 | `ATOM_ENABLE_METRICS_DEVICE_STAGES=1` 控制 sampling/propose；仍要求基本开关 `ATOM_ENABLE_METRICS_DEVICE_TIMER=1`，基本 timer 不再自动启用这两项 |
| template/tokenize 按实际工作采集 | 无 token IDs 时实际执行的分支记录；复用 token IDs 时不记零。保留 prefill、standalone 和 decode fallback 的观测 |
| 保留主要阶段 | 保留 body reception/handling、API preprocess、scheduler forward-to-output、原有 TTFT/ITL、queue time/PD wait 等 |
| 调整报表与分析工具 | 删除对应面板，添加可选 output delivery；工具只使用 streaming 总 TTFT，不再相加阶段均值或推算 residual |

新增 API 诊断通过当前请求的对象传递时间戳，避免额外 map 生命周期管理；不扩大到 fanout。
输出交付终点是服务端生成器的首个有效内容，不是客户端接收；跨主机仍依赖 wall-clock 同步。
诊断未启用或操作没有执行时，报表显示无样本，不替换成“0 ms”。

主要回归：347 passed、4 skipped，包含 API streaming、PD token IDs、request timing、scheduler/GPU metrics、multiprocess 导出与 CI 报表。补充 scheduler 连续输出和工具统计口径用例后，scheduler / scheduler metrics / TTFT 测试组为 187 passed（与主要回归有重叠，不相加计数）。
新增了跨线程首输出交付、预分词不观测、关闭诊断不读 stamp、GPU 细分关闭不增加 events 和 CPU 导出不读取进程的验证。
真实 GPU 性能 A/B 未运行，最新讨论已明确取消，不再作为本轮验收待办；不声明吞吐提升比例。原始未跟踪 CPU 测试在更新前保存的备份已收入 [实验归档](experiments/metrics_review_3ff61024_c2c68165/README.md)。

**8. 后续范围确认、trace 补充修改与资料归档（2026-09-17）**

本轮目标限定为优化 `3ff61024bca742c8864daf7a91d806a352d9a5b0` 至 `c2c68165d02c0db0be46dcdb780570237359415c`（包含起始提交）新引入的观测修改。

| 事项 | 最新决议 / 状态 |
| --- | --- |
| 实际模型/GPU 性能 A/B | 取消。没有必要为比较已决定精简的众多指标开展完整对照；代码与功能验证继续保留，不宣称测得 CPU/吞吐改善比例 |
| 外部 CPU 采集及 affinity/quota 自动归档 | 原本作为 A/B 配套建议，本轮不实施 CI 集成；第 7 节是外部工具使用建议，不代表已接入自动采集 |
| 两个逐请求 context 指标 | 非本轮新增，保持现状，仅列入 [暂缓 TODO](todo.md) |
| trace 关闭后的空包装 | 确认为起始提交 `3ff61024b` 新增，纳入本轮处理 |
| Wiki 实验资料 | 已归档到 [experiments/metrics_review_3ff61024_c2c68165](experiments/metrics_review_3ff61024_c2c68165/README.md)，不再依赖 `/tmp` 原件 |

trace 来源核对：`git log -S 'def ttft_trace_span' 3ff61024b^..c2c68165d -- atom/model_engine/run_labels.py` 定位到 `3ff61024b`；该提交同时新增 API preprocess 与 ModelRunner 三个阶段的调用点。

`ATOM_TTFT_TRACE` 现在在进程启动加载模块时读取一次；请在启动 API/worker 前设置。关闭时，各调用点直接执行阶段函数，跳过 trace context manager 的创建、进入和退出，也不再逐请求/forward 读取环境变量。开启时保留 `ttft[api_preprocess]`、`ttft[prepare_model]`、`ttft[gpu_forward]`、`ttft[postprocess]` 标记。API 计时仍发生在执行 preprocess 的 executor 线程内，不包含 executor 排队时间。

资料归档保留历史脚本和结果原件、被测模块的 `c2c68165d` 源码快照及原 CPU 测试备份；提供显式选择历史源码目录的复现入口、来源清单和 SHA-256。原实验未采集的依赖版本/机器配置不补写推测值，后续重跑也不得覆盖历史结果。

本次 trace 补充回归为 **102 passed**：`tests/test_ttft_trace.py`、`tests/test_run_label.py`、`tests/test_model_runner_staging_fence.py`、`tests/entrypoints/test_ttft_breakdown.py`、`tests/entrypoints/test_pd_prompt_token_ids.py`、`tests/entrypoints/test_api_server_helpers.py`。覆盖关闭时跳过 trace、开启时的阶段进入/退出、异常原样传播、原有 staging 顺序、跨线程 SSE 和 PD token ID 复用；与第 7 节测试有重叠，不累加计数。没有运行实际推理 A/B，也没有重跑历史微基准。
