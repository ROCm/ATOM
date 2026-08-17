# DSV4 PR1894：ATOM 与 SGLang Agentic C32 性能差距排查

记录时间：2026-08-15 UTC

## 目的

记录 ATOM PR1894 在 MI355X 8 卡、Agentic Coding、并发 32 场景下与
InferenceX SGLang CI 的性能差距、日志证据和后续 A/B 顺序，供后续复现和继续排查。

本文结论基于正式 1-hour profile、checkpoint interval/chunk A/B、最终 clean 900-second
端到端对照，以及 2026-08-16 完成的同 token、同 3,072 physical token、8-rank GPU
trace。最终定位不依赖猜测 SGLang 如何恢复 C4/SWA state：ATOM 正式配置保持
`ATOM_STATE_REPLAY_TAIL_TOKENS=0`，C128 可以 replay，C4 state 和 SWA 必须 hit/restore。
早期 tail-replay 实验只保留为 ATOM 内部“少算 physical token”的性能上界，不作为
正确实现，也不用于判断 SGLang 正确性。

## 数据来源

### ATOM C32

本地目录：

```text
/home/hyi/project/008-lmcache-dsv4-pro/.agentic-runtime/
  atom-pr1894-agentic-mem90-c32-c16-c8-16k-state32k-full-20260815T100232Z/c32
```

关键文件：

```text
run-manifest.txt
atom_server.log
aiperf_artifacts/profile_export_aiperf.json
cache-stats-after.json
mtp-stats-after.json
```

启动参数：

```ini
concurrency=32
max-num-seqs=64
gpu-memory-utilization=0.9
max-num-batched-tokens=16384
attn-prefill-chunk-size=16384
state-checkpoint-interval-tokens=32768
level=3
cudagraph-mode=FULL
```

运行身份：

```text
ATOM base SHA: c7261bee0f39be40fa5cfb475f88b4f54fae44a1
ATOM local diff SHA256: 23965f7a62f05a038fcf1fb87b1ac33de049e4369c3b97ae3ee819b2632f271f
InferenceX SHA: 5d1e20a05eb474b0c16d666ab661ef12f2e67728
AIPerf SHA: bcfc235c552a17de6d1a7a5d2345cf012401588d
```

本次 `submission_valid=true`。成功请求 2827 个，另有 1 个
`ClientOSError`；服务端没有 OOM、崩溃或重启，因此该错误不足以解释吞吐差距。

### SGLang C32

InferenceX CI：

```text
https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31762202350/attempts/1
```

重新下载时使用的 artifact ID：

```text
9207600841  server logs
9207600474  raw agentic artifact
9207598535  aggregate JSON
```

聚合结果文件：

```text
dsv4_tp8_conc32_kvdram-hicache_spec-mtp_fp4_sglang_
tp8-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_
conc32_mi355x-amds_05.json
```

SGLang 日志中的关键参数：

```text
model_path=deepseek-ai/DeepSeek-V4-Pro
tp_size=8
ep_size=1
enable_dp_attention=False
kv_cache_dtype=fp8_e4m3
mem_fraction_static=0.85
max_running_requests=64
chunked_prefill_size=8192
max_prefill_tokens=16384
disable_overlap_schedule=False
attention_backend=dsv4
page_size=256
swa_full_tokens_ratio=0.1
cuda graph FULL for decode
enable_hierarchical_cache=True
hicache_ratio=1.5
hicache_write_policy=write_through
```

## 对齐条件

本次对比均为：

- MI355X × 8；
- DeepSeek-V4-Pro；
- TP8、EP1、DPA disabled；
- Agentic Coding 数据集和 C32；
- MTP speculative decoding；
- FP4 权重、FP8 KV；
- 非 disaggregated serving。

如果以后更换镜像或模型挂载，仍需额外核对模型 revision、权重来源和 kernel 版本；
只看模型名不能保证二进制路径完全相同。

## 端到端结果

### 最终 clean 900-second 对照

双方重新 clean 启动、使用相同 Agentic C32 workload 后：

| 指标 | SGLang | ATOM（tail=0） | ATOM 相对 SGLang |
|---|---:|---:|---:|
| 总 token throughput | 109,357.12 tok/s | 96,642.46 tok/s | -11.63% |
| request throughput | 1.10957 req/s | 1.03511 req/s | -6.71% |
| TTFT p50 | 736.142 ms | 860.310 ms | +16.87% |
| ITL p50 | 23.869 ms | 29.893 ms | +25.24% |
| physical new token/request | 4,941.315 | 7,721.818 | +56.27% |

ATOM 每完成一个请求多算：

```text
7,721.818 - 4,941.315 = 2,780.503 physical prefill token/request
2,780.503 × 1.03511 req/s = 2,878.13 extra physical token/s
```

同配置下 3,072-token prefill 的无 profiler wall time 实测约 148–153 ms。把额外
physical work 换成 GPU 时间：

```text
2,878.13 / 3,072 × 148 ms = 138.66 ms/s
2,878.13 / 3,072 × 153 ms = 143.34 ms/s
```

也就是约 13.87%–14.33% 的 GPU 时间。它已经覆盖 ATOM 相对 SGLang 的 11.63%
总吞吐缺口，因此端到端第一主因是 **ATOM 做了更多 physical prefill work**。这不是
用逻辑 input token rate 倒推，而是双方 scheduler log 的实际 new-token 计数。

### 历史 1-hour CI 对照

| 指标 | ATOM C32 | SGLang C32 | SGLang 相对优势 |
|---|---:|---:|---:|
| 总 token 吞吐 | 105,580.33 tok/s | 119,950.43 tok/s | +13.61% |
| 每 GPU 总吞吐 | 13,197.54 tok/s | 14,993.80 tok/s | +13.61% |
| 请求吞吐 | 0.77665 req/s | 0.84407 req/s | +8.68% |
| 输出吞吐 | 736.05 tok/s | 810.19 tok/s | +10.07% |
| TTFT p50 | 766.17 ms | 708.55 ms | 低 7.52% |
| ITL p50 | 24.02 ms | 20.03 ms | 低 16.62% |
| ITL p90 | 38.98 ms | 31.66 ms | 低 18.76% |
| P90 interactivity | 25.66 tok/s/user | 31.58 tok/s/user | +23.09% |

其中：

```text
P90 interactivity = 1000 / ITL_p90_ms
```

## 结论

复查实现、完成 clean 端到端复跑和严格 trace 后，结论为：

1. **第一主因是 physical work 数量，不是凭 active-prefill 指标猜 kernel 快慢。**
   clean 900-second 中 ATOM 每请求算 7,721.818 个 physical prefill token，SGLang 为
   4,941.315，ATOM 多 2,780.503（+56.27%）。按 ATOM request rate 和实测 prefill
   wall time折算，这部分多算约占 13.87%–14.33% GPU 时间，足以覆盖 11.63% 的总吞吐
   缺口。
2. **同 work 下 ATOM kernel/critical path 也有次级差距，但不是 56% physical-work
   差距。**同一组 token、同样 `[2,816, 256]` new token 的主 prefill cluster，8-rank
   平均 span 为 ATOM 180.011 ms、SGLang 167.591 ms；ATOM 长 12.420 ms（+7.41%）。
   kernel busy 只多 5.906 ms（+3.67%），另有 6.514 ms 来自 cluster 内 idle/launch gap。
3. **同-work 的最大 kernel-family 机会是 collective。**ATOM collective 平均
   40.568 ms，SGLang 28.169 ms，差 12.398 ms。具体包括 ATOM 独有的 6 次
   `cross_device_reduce_1stage`（7.382 ms）、2-stage reduce 慢 3.750 ms，以及
   all-gather 多 1.152 ms。但 7.382 ms 是 collective kernel active/synchronization
   time，不能等同于“删掉 6 次调用就回收 7.382 ms”：这些 `[2, 7168]` 小张量调用中
   rank1 仅约 7 us/call，而其余 rank 大量时间在等待。共同的 2-stage reduce 的 p50
   也几乎相同，ATOM 主要慢在 p90/p99 长尾。MoE category 总量只差 0.258 ms；不能把
   stage2 单项差距误写成整个 MoE 落后。attention 的粗分类也会把两端 MQA 放进不同
   category，必须按 kernel family 比较。
4. ATOM 8 rank span 为 179.916–180.141 ms，SGLang 为 167.482–167.653 ms，没有
   TP straggler 能解释总体差距。ATOM rank1 collective active time 较短但 wall span 与
   其他 rank 一致，这是 collective 角色/等待差异，不是 rank1 更快完成请求。
5. C128/C4/SWA 的语义边界保持不变：C128 对齐块可 replay，C4 state 和 SWA 必须
   hit/restore。本文不再根据未完整追踪的 SGLang 内部 restore 细节推断正确或错误，
   性能结论完全由实际 physical-token 计数和同-work trace 支撑。ATOM production
   路径继续使用 `ATOM_STATE_REPLAY_TAIL_TOKENS=0`。
6. 8K scheduler cap、phase timing、nominal KV capacity、HiCache CPU hit 和 MTP
   acceptance 已排除为第一主因。hipBLASLt fallback 仍是独立优化项，但现有证据不支持
   它单独解释端到端差距。

## 严格同 token、同 physical work 的 8-rank trace

### 对齐方法

两端都使用：

```text
token(i, seed) = 1000 + ((i * 48271 + seed) % 120000)
capture seeds  = prefix 109, extension 113, short 127
Q              = 32,768 tokens
Y              = 2,816 tokens
C              = 256 tokens
target         = [Q + Y, C]
```

两个 manifest 的 length、seed 和 u32le SHA256 逐项一致。正式调度日志分别确认：

```text
ATOM:   Scheduled prefill batch: 2 reqs, 3072 new tokens
        cached: [32768, 0], new: [2816, 256]
SGLang: #new-seq: 2, #new-token: 3072, #cached-token: 32768
```

ATOM 8 个 rank 还逐一记录
`prefill_tokens=3072 decode_tokens=0 reqs=2`。因此这不是“逻辑 prompt 看起来相同”，
而是 token 内容、cache 边界、new-token shape 和 request 数都相同。

原始数据：

```text
/home/hyi/project/009-unify-kv/.agentic-runtime/c32-trace-comparison/
  atom-oneprefill-same-token-allranks/
  sglang-checkpoint-same-token/
  strict-same-token-main-cluster-comparison/
```

SGLang profiler 在主 prefill 后还捕获了 completion/sampling 小 cluster。每个 rank 都在
约 167.5 ms 后出现第一个大于 5 ms 的 idle gap；最终比较只保留 gap 前的主 cluster，
统一使用 kernel-only 事件。过滤后 SGLang 每 rank 均为 4,102 个 kernel event；ATOM
每 rank 均为 3,541 个。

### 8-rank wall span 与 kernel busy

| 指标 | ATOM | SGLang | ATOM - SGLang |
|---|---:|---:|---:|
| 主 cluster span，8-rank mean | 180.011 ms | 167.591 ms | +12.420 ms |
| 主 cluster span，min–max | 179.916–180.141 ms | 167.482–167.653 ms | -- |
| kernel busy，8-rank mean | 166.911 ms | 161.005 ms | +5.906 ms |
| cluster 内 idle，8-rank mean | 13.100 ms | 6.586 ms | +6.514 ms |

ATOM 的主 cluster 比 SGLang 长 7.41%，但 kernel busy 只长 3.67%。这说明同-work 次级
差距由 kernel 本身和 launch/idle 两部分组成；不能把 12.420 ms 全部记到某个 kernel。
两端使用不同 profiler（ATOM rocprof、SGLang Kineto），绝对 wall span 可能包含不同的
profiler overhead，因此 kernel-family 排序比跨 profiler 的单次绝对值更适合作优化依据。

### 主要 kernel family（8-rank mean）

| family | ATOM | SGLang | ATOM - SGLang |
|---|---:|---:|---:|
| collective 2-stage reduce | 31.901 ms | 28.151 ms | +3.750 ms |
| collective 1-stage reduce | 7.382 ms | 0 | +7.382 ms |
| collective all-gather | 1.171 ms | 0.018 ms | +1.152 ms |
| MoE stage1 | 14.626 ms | 16.878 ms | -2.252 ms |
| MoE stage2 | 14.750 ms | 9.217 ms | +5.533 ms |
| MoE route reduce | 1.920 ms | 1.900 ms | +0.021 ms |
| DSA top-k | 2.455 ms | 5.508 ms | -3.053 ms |
| PA prefill | 13.673 ms | 11.387 ms | +2.286 ms |
| MQA logits | 9.670 ms | 11.165 ms | -1.494 ms |
| blockscale GEMM | 18.260 ms | 20.057 ms | -1.797 ms |
| rocBLAS/Tensile GEMM | 11.601 ms | 12.270 ms | -0.669 ms |
| C128 compress | 0.316 ms | 0.336 ms | -0.020 ms |
| C4 compress | 0.415 ms | 0.682 ms | -0.267 ms |
| SWA write/scatter | 0.563 ms | 0.290 ms | +0.273 ms |

可见同-work 差距不是“SGLang 所有 kernel 都更快”：ATOM 的 MoE stage1、top-k、MQA
和多类 GEMM 反而更快；ATOM 的主要净损失来自 collective、MoE stage2、PA prefill 和
额外 idle。按 analyzer 粗 category 汇总时，MoE 是 36.075 对 35.816 ms，只差
0.258 ms；collective 才是 40.568 对 28.169 ms，差 12.398 ms。

### Collective 差距是长尾等待和额外 MTP 调用，不是基础带宽慢 44%

两端共同的 126 次 `cross_device_reduce_2stage` launch shape 一致：block size 512、
80 blocks（ATOM rocprof 显示的 `grid_x=40960` 是 work-item 口径，等价于 SGLang
Kineto 的 `grid=[80], block=[512]`）。把 8 rank 的 1,008 次调用放在一起看：

| 2-stage reduce latency | ATOM | SGLang |
|---|---:|---:|
| mean | 253.18 us | 223.42 us |
| min | 184.88 us | 185.24 us |
| p50 | 203.34 us | 202.48 us |
| p90 | 382.72 us | 265.20 us |
| p99 | 1,004.52 us | 666.20 us |

min/p50 几乎相同，说明同 shape kernel 的基础传输/计算速度没有明显差距；ATOM 多出的
3.750 ms 主要来自 p90/p99 长尾。custom all-reduce kernel 会在 GPU 上轮询其他 rank
的 signal/data，因此 profiler 记下的 kernel duration 同时包含通信和等待 peer 到达的
时间。现有 rocprof 的各 GPU device clock 不能当作全局同步时钟，不能从跨 GPU 绝对
timestamp 反推出精确 arrival skew；但分布足以排除“ATOM 每次 collective 都固有更慢”。

ATOM 另外 6 次 one-stage reduce 都在主 target prefill 后的 MTP/draft 尾部，shape 为
`[2, 7168]`（28 KiB BF16/rank）。代码路径与调用数吻合：`EagleProposer` 循环 3 个
draft step，每个 V4 MTP block 分别在 attention row-parallel output 和 MoE output
combine 做一次 TP all-reduce，即 `3 × 2 = 6`。本 trace 每个请求只生成 1 token，ATOM
仍在返回 output 前执行完整 proposal；SGLang 的 prefill 后小 cluster 只有 3 次额外
2-stage reduce、合计约 0.086 ms。ATOM one-stage 在 rank1 约 7 us/call，在其他 rank
通常约 1.4 ms/call，远大于 28 KiB 传输本身，说明 7.382 ms 的跨-rank平均值主要暴露
同步/到达时序或算法 rank-role 差异。它是需要继续定位的症状，不是可以直接承诺的
7.382 ms critical-path 收益。

### c4/c8/c16：publish-before-draft 与 terminal proposal cancellation

为验证上面的“请求结束后是否仍做无用 proposal”，实现了只覆盖 plain MTP 的两阶段
实验路径：target/verify 完成后先把结果交给 scheduler，再由同一个 worker FIFO 完成下一轮
proposal；如果 scheduler 已确认该 batch 的所有请求都结束，则取消这轮永远不会被消费的
proposal。Eagle3、DSpark、EPLB 和非 deferred-output 路径均不进入这个实验分支。

最初打开 split target/proposal 后，completion/acceptance 偶发变化。根因不是 MTP 接受
算法，而是异步 D2H 的源 tensor 生命周期：原队列只保存 CPU 目标 tensor 和 CUDA event，
没有保存 `sampled_tokens`、draft ids 和 MTP status 的 GPU 源 tensor。split target 提前
返回后 Python 引用可被释放，caching allocator 会在 D2H event 完成前复用这块显存，CPU
随后可能读到已覆盖的 token/status。现在三个队列都把源 GPU tensor 保活到对应 event
`synchronize()` 之后。

固定 acceptance=0 的 1,500 次受控测试结果：

| 路径 | 错误出现多 token |
|---|---:|
| 原同步 baseline | 0 / 1,500 |
| 未保活源 tensor，defer + finish | 82 / 1,500 |
| 未保活源 tensor，defer + cancel | 30 / 1,500 |
| 保活源 tensor，defer + cancel | 0 / 1,500 |

最终真实 acceptance `325 / 666 = 48.80%` 的 A/B 只跑 `c4/c8/c16`，每档 8 次无
profiler timing 和 1 次独立 trace；两边 completion 数逐响应一致，accepted/draft 和完整
distribution 也完全一致：`0:56, 1:62, 2:49, 3:55`。

| shape | TTFT 变化 | 端到端 latency 变化 |
|---|---:|---:|
| c4 | -13.1 ms / +6.53% | -8.8 ms / +3.49% |
| c8 | -12.1 ms / +5.86% | -10.6 ms / +3.58% |
| c16 | -15.8 ms / +7.66% | +2.8 ms / -0.74% |

三档 trace 都少 1 个 proposal round 和 9 次 `aiter::all_reduce_` 调用；trace envelope
分别少 9.693、0.847 和 15.336 ms。`c4/c8` 的端到端中位数约快 3.5%，`c16` 在噪声
内持平。因此这是 short-decode 的次级优化，不是 clean C32 中 +56.27% physical work
主差距的替代修复，也不能从单次 profiler envelope 推导稳定生产收益。

默认配置中，`ATOM_DEFER_MTP_PROPOSAL=0`、
`ATOM_CANCEL_TERMINAL_MTP_PROPOSAL=0`，等待完整模型/后端 eval 后再决定是否默认开启。
只覆盖纯 final-prefill、所有请求 `max_tokens=1` 的窄 fast path 仍默认开启。相关测试共
`106 passed`；本轮只完成 DSV4 `c4/c8/c16` 的 workload-equivalence 验证，没有覆盖所有
模型与 spec backend。

原始结果：

```text
/mnt/m2m_nobackup/hyi/atom-real-decode-ab-20260816/
  short-baseline/
  short-fixed-cancel-rate496/
  short-fixed-cancel-rate496-report.md
```

### SGLang 关闭 HiCache 的本地 A/B

为排除 prefix cache checkpoint/HiCache 改变 physical work 的可能，使用同一 SGLang
ROCm 镜像（digest `c57b8ce888b...ed3c24`）重新启动服务，只删除
`--enable-hierarchical-cache` 和所有 `--hicache-*` 参数。普通 GPU radix cache 保持
开启：`enable_hierarchical_cache=False`、`disable_radix_cache=False`、
`page_size=256`。token manifest 与 HiCache-on trace 逐字节相同，服务日志在 prewarm
和正式 capture 中都再次确认：

```text
#new-seq: 2, #new-token: 3072, #cached-token: 32768
new tokens    = [2816, 256]
cached tokens = [32768, 0]
```

因此这次 32,768-token checkpoint hit 来自普通 GPU radix cache，不依赖 HiCache。
修正 rank4 中紧跟主请求、没有形成 5 ms gap 的 MTP draft cluster 后，HiCache on/off
主 prefill 都是每 rank 4,102 个 kernel event、126 次 2-stage reduce 和 2 次 all-gather；
8 个 rank 的 kernel name/call count 逐项完全一致。
响应内容也一致：HiCache on/off 的 prewarm 两条 completion 都分别为 `"s"`、`"้า"`，
正式 capture 都为 `"_"`、`" "`；usage 均为 35,840 prompt token 和 2 completion
token。响应只有 request ID 和时间戳不同。

| 指标（8-rank mean） | HiCache on | HiCache off | on - off |
|---|---:|---:|---:|
| 主 cluster span | 167.591 ms | 165.340 ms | +2.251 ms |
| kernel busy | 161.005 ms | 160.013 ms | +0.992 ms |
| cluster 内 idle | 6.586 ms | 5.327 ms | +1.259 ms |
| 2-stage reduce 总量 | 28.151 ms | 26.826 ms | +1.325 ms |
| all-gather 总量 | 0.018 ms | 0.017 ms | +0.001 ms |

两次运行的非 collective kernel family 基本都在 `±0.2 ms`；差异集中在 collective
长尾，而不是调用或 shape 变化：

| 2-stage reduce latency（8 rank × 126） | HiCache on | HiCache off |
|---|---:|---:|
| mean | 223.422 us | 212.905 us |
| p50 | 202.481 us | 201.561 us |
| p90 | 265.202 us | 252.146 us |
| p95 | 297.924 us | 271.818 us |
| p99 | 666.203 us | 377.318 us |
| max | 914.850 us | 614.326 us |
| `>300 us` 调用数 | 50 | 24 |
| `>500 us` 调用数 | 20 | 4 |

p50 只差 0.920 us，而 p99 和超 300 us 的次数明显波动。单次 on/off 尚不足以把这
2.251 ms 归因于 HiCache 的 host-side 干扰；但它已经能排除“HiCache checkpoint 让
SGLang 少算了 GPU physical work”。关闭 HiCache 后 SGLang 的同-work span 没有回归
到 ATOM 的 180.011 ms，反而略降到 165.340 ms，所以 prefix checkpoint/HiCache 不是
ATOM–SGLang 12.420 ms 同-work 差距的解释。A/B 原始 trace 位于：

```text
/mnt/m2m_nobackup/hyi/sglang-atom-c32-trace-20260815T182000Z/
  sglang-nohicache-checkpoint/attempt2/
```

## 指标口径纠正：AIPerf active prefill 不是物理 GPU token rate

AIPerf active phase：

| 指标 | ATOM | SGLang | SGLang 相对优势 |
|---|---:|---:|---:|
| active prefill throughput | 215,666.86 tok/s | 251,285.22 tok/s | +16.52% |
| effective prefill concurrency | 0.94360 | 0.94427 | 基本相同 |

Prefill concurrency 基本完全相同，但 `active_prefill_throughput` 的 token 是请求的
逻辑 input token，不是 server 实际 forward 的 cache-miss token。当两端 cache resume
策略不同，用这个指标直接推断 prefill kernel 快慢是不成立的。

从 server schedule log 统计正式窗口内的物理新 token：

| 指标 | ATOM | SGLang |
|---|---:|---:|
| 正式窗口请求数 | 2,827 | 3,054 |
| 逻辑 prompt token | 381,633,267 | 431,758,574 |
| 物理 prefill 新 token | 19,755,993 | 13,296,640 |
| 物理新 token/请求 | 6,988 | 4,354 |
| profile-window 物理 hit | 94.823% | 96.921% |

若仅用 physical-miss ratio 缩放 AIPerf active prefill，ATOM 约为 11.16K physical
tok/s，SGLang 约为 7.72K physical tok/s。这个换算仍不是逐 kernel profiler，但至少
说明原来的“16.52% active prefill 差距等于 SGLang kernel 更快”方向相反：ATOM 在
正式窗口做了约 49% 更多物理 prefill token，主要问题先是为什么需要重算这么多。

## 次要待定项：ATOM hipBLASLt fallback

### ATOM hipBLASLt fallback

ATOM `atom_server.log` 中有 1512 行：

```text
HIPBLAS_STATUS_INTERNAL_ERROR ...
Will attempt to recover by calling cublas instead.
```

这些告警由 8 个 TP rank 对同一个 shape 分别输出。按 8 rank 去重后是 189 个唯一
shape；warning 去重语义和同 shape 的真实调用次数尚未通过 trace 证明，因此不能把
它们写成 189 个或 1512 个高频逻辑事件：

| GEMM shape | rank 日志行数 | 估算逻辑事件数 |
|---|---:|---:|
| `m=2048, k=7168, n≈2116–2560` | 1440 | 180 |
| `m=384, k=7168, n≈12942–13312` | 72 | 9 |

告警位置：

```text
/app/aiter-test/aiter/tuned_gemm.py:514
bgemm_internal_cublaslt
```

这些不是典型的小 batch decode shape，更像 prefill 中的 BF16 projection/dense
GEMM。SGLang server log 中没有对应的 `HIPBLAS_STATUS_INTERNAL_ERROR`。

这些 shape 的 tuned config 为 `libtype=torch`，走 `F.linear` 原生路径；失败后
PyTorch 会从 hipBLASLt 回退到 rocBLAS。它值得在 GPU 空闲后用两组代表 shape 做
hipBLASLt/rocBLAS microbenchmark，但不能再把它与逻辑 active prefill 的 16.52%
简单对应。

快速复查命令：

```bash
rg -c 'HIPBLAS_STATUS_INTERNAL_ERROR' atom_server.log

rg 'HIPBLAS_STATUS_INTERNAL_ERROR' atom_server.log \
  | sed -E 's/.* m ([0-9]+) n ([0-9]+) k ([0-9]+) .*/m=\1 n=\2 k=\3/' \
  | sort | uniq -c | sort -nr
```

## 原候选 Gap 2：同步 engine loop 与 overlap scheduling

AIPerf active decode：

| 指标 | ATOM | SGLang | 说明 |
|---|---:|---:|---|
| active decode throughput | 737.76 tok/s | 809.82 tok/s | SGLang +9.77% |
| effective decode concurrency | 19.40 | 17.54 | ATOM 反而高 10.6% |
| 粗略 tok/s/active request | 38.02 | 46.16 | SGLang 约 +21.4% |

`active decode throughput / effective decode concurrency` 只是用于定位的粗略指标，
不能替代逐 batch、逐 kernel profiling。但它说明 SGLang 并不是靠更高的在途 decode
并发赢得吞吐；ATOM 需要维持更多 active decode request，单请求进展仍然更慢。

对应的延迟指标：

| 指标 | ATOM | SGLang |
|---|---:|---:|
| time-to-second-token p50 | 49.11 ms | 27.71 ms |
| time-to-second-token mean | 255.75 ms | 106.27 ms |
| decode duration p50 | 9.26 s | 7.71 s |
| decode duration mean | 24.87 s | 20.91 s |

SGLang 使用：

```text
chunked_prefill_size=8192
max_prefill_tokens=16384
disable_overlap_schedule=False
```

ATOM 使用：

```text
attn_prefill_chunk_size=16384
max_num_batched_tokens=16384
```

代码路径显示 ATOM 每一步同步执行：

```text
scheduler.schedule()
runner_mgr.call_func("forward", ..., wait_out=True)
scheduler.postprocess(...)
output queue
```

SGLang 的 overlap loop 则把当前 `run_batch()` 的结果放入 `result_queue`，在 GPU
执行下一批期间 `process_batch_result()` 上一批。它并非对所有相邻 prefill batch 都
开启 overlap，但 decode-heavy 的 Agentic C32 正是这类流水更容易产生收益的场景。

phase instrumentation 已按 prefill/decode、checkpoint/plain 拆出 schedule、prepare、
forward wait、KV poll、postprocess 和 output wall time。对两个 900-second run 都取
profile start 之后的第一个累计快照和 profile end 之前的最后一个快照；两边恰好都是
888 秒的内部窗口：

| 900s run | engine total | forward wait | 非-forward | 非-forward 占比 |
|---|---:|---:|---:|---:|
| 32K checkpoint baseline | 887.371 s | 876.949 s | 10.421 s | 1.174% |
| 128-token tail replay | 887.736 s | 876.591 s | 11.145 s | 1.255% |

baseline 在该窗口完成 11,424 个 plain decode batch、1,462 个 checkpoint prefill batch
和 97 个 plain prefill batch；tail replay 完成 15,944 个 decode batch和 1,056 个 plain
prefill batch。两边 engine wall time/forward wait 几乎相同，但 replay 用更少的 prefill
重算换来了更多 decode 进展。

这组数据把同步 CPU/scheduler bubble 的可隐藏上限压到约 1.3%，不能解释原始 13.61%
gap。SGLang overlap 仍可能改善提交节奏，但已不是第一或第二主因；tail replay 后若还
有稳定差距，应先做同节点长跑和逐 GPU phase/kernel timing。

## 已确认 Gap 1：完整 state checkpoint、gate 重算与普通 KV 共池

### ATOM

`gpu_memory_utilization=0.9` 时：

```text
num_kvcache_blocks ≈ 86,012
block size = 256 tokens
GPU KV capacity ≈ 22.02M tokens
```

缓存统计：

| 指标 | 数值 |
|---|---:|
| PAGE hit | 93.163% |
| compressed hit | 95.089% |
| lost to checkpoint | 1.926% |
| lost unrecoverable | 0% |

### SGLang

| 指标 | 数值 |
|---|---:|
| GPU KV pool | 9.636M tokens |
| Host HiCache pool | 14.454M tokens |
| GPU cache hit | 96.838% |
| CPU cache hit | 0.154% |
| Overall cache hit | 96.991% |
| 报告的 GPU KV usage 最大值 | 62% |

SGLang 的 host pool 因 write-through 写满，并不等于大量请求从 CPU 回读；实际 CPU
hit 只有 0.154%。所以 HiCache 回读无法解释 13.61% 的总体差距。

两边理论 prefix hit 也很接近：

```text
ATOM theoretical prefix hit   = 97.284%
SGLang theoretical prefix hit = 97.457%
差值                         = 0.173 个百分点
```

原先只把 `lost_to_checkpoint=1.926%` 理解为 32K 粒度带来的“小量重算”，漏掉了
checkpoint 自身对普通 KV 容量的侵占。DSV4 当前几何为：

```text
PAGE unit bytes          = 1,521,920
完整 Active Slot bytes   = 27,142,400（25.89 MiB）
units per checkpoint     = 18
PAGE pool                ≈ 86,012 units / GPU
```

关键代码行为：

- `PageUnitCheckpointStore.begin_store()` 为每个边界申请 18 units；
- `BlockPool.reserve_units()` 直接从普通 PAGE/KV free pool 取这些 block；
- free pool 里的 content-indexed KV 也算可用，`allocate()` 会先 unindex/淘汰它；
- 普通 KV 真正没有 PAGE 可拿时，`ensure_free_units()` 才按 LRU 淘汰 checkpoint；
- KV hash 被逐出后，对应 checkpoint 才会 orphan 并释放。

所以 checkpoint 的默认优先级实际高于可复用的普通 KV。正式 32K run 全程记录
`checkpoints_kept=3284`、`checkpoints_evicted=0`；即使部分后来因 KV hash eviction
而 orphan，在线存量也足以占掉几十个百分点的 PAGE pool。16K interval 运行约 11
分钟时曾达到 1,609 个 kept checkpoint，相当于 gross 28,962 PAGE units，即池子的
33.7%。

### 32K 对 16K interval 的同序号证据

相同累计 1,000 请求：

| interval | compressed hit | actual hit | gate loss |
|---|---:|---:|---:|
| 32K | 89.73% | 87.02% | 2.70pp |
| 16K | 89.14% | 87.18% | 1.96pp |

interval 加密使 gate loss 改善 0.74pp，却令基础 compressed hit 下降 0.59pp，净 actual
hit 只改善 0.16pp。这直接证明“多存 checkpoint”与“保留普通 KV”存在容量跷跷板，
不能靠继续缩 interval 根治。

### 状态语义边界：C128 可重放，C4/SWA 必须 hit

这一节保留早期源码阅读的背景，但**不再据此判断 SGLang 实现正确或错误**。最终性能
定位已经由 clean physical-token 计数和严格同-work trace 闭环，不需要对未完整追踪的
内部 restore 路径作推断。本文只采用两条与 ATOM 设计直接相关的约束：C128 对齐块可以
replay；C4 state 和 SWA 必须从 prefix hit 恢复。

SGLang v0.5.17 实际有两条不同路径，不能混为一谈。

普通 DSV4 KV + HiCache 路径正是正确的混合恢复方式：

- 注册并传输 SWA host pool；
- C4 attention state 和 C4 indexer state 作为依附于 SWA 索引的 HiCache sidecar 一起
  backup/load-back；
- 不注册 C128 state pool。源码注释明确说明 `page_size=256 % 128 == 0`，load-back 从
  C128 对齐边界继续时不消费旧 C128 state。

但目标 CI 不是这条普通路径。它设置了：

```text
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
enable_hierarchical_cache=True
page_size=256
```

`unified_kv_triton` 特殊路径在 HiCache assembler 中显式做了相反选择：

- unified SWA 是 `req_pool_idx` 寻址的 request-local ring，`swa_layer_mapping={}`，不创建
  SWA host pool；
- `if not is_unified_kv` 才创建 C4 state/C4 indexer state host pool，所以该路径也不
  backup/load-back C4 state；
- C128 compressed KV 仍作为普通 compressed sidecar backup/load-back，但 C128 state
  不传输。

为了补 request-local SWA ring，该特殊路径在 prefix match 时执行：

```text
key_limit = input_len - swa_reprefill_tail_tokens()  # 128
```

由于 radix page size 为 256，实际命中边界会 page-align，通常至少重算一个 256-token
block。这段代码足以说明 C128 replay 的 physical work，但单独阅读这一段不能证明完整
C4/SWA restore 是否存在于其他路径，因此不再从这里外推正确性结论。

沿数据路径复核后的边界是：

1. C128 是 ratio=128、non-overlap 的独立压缩块。在 128 对齐边界，partial-state 长度为
   0；给出该块的 128 个输入即可重新生成 C128 compressed KV，所以 C128 tail replay
   本身没有问题。
2. unified KV 的 SWA ring 地址是
   `req_pool_idx * ring_stride + position % ring_stride`。新请求拿到新的 `req_pool_idx`；
   对应 prefix 的每层 SWA KV 必须 hit/copy 到新请求可读的位置。只重算尾部会让最终
   ring 的槽位被覆盖，但 replay 起点最早的 token 在计算时仍缺少更早的 SWA KV，不能
   作为 SWA state 的恢复方法。
3. C4 是 overlap compressor，源码的 state 长度为
   `seq_len % 4 + 4`，所以即使 replay 起点 4-token 对齐，仍会读取起点前 4 个 raw state。
   这些 C4 state 必须由 prefix cache hit 保留/恢复；fresh request slot 或只重放 C128
   tail 都不能替代它。这是 ATOM 正式实现必须满足的约束。

现有测试组合没有直接提供“目标 backend + prefix hit + HiCache”的 token/logit 等价
证据；这属于独立正确性审计，不纳入本轮性能归因，也不据此评价 SGLang 实现。

因此本轮只记录可直接观察到的性能事实：目标 SGLang 配置会在 prefix match 时保留
trailing tail 的 physical work；是否以及如何在其他路径恢复 C4/SWA，不是本性能结论的
前提。若要单独审计正确性，应另做 cache-hit/cache-miss token/logit A/B。ATOM 的 hash
block 为 256，固定退一个 block 的反事实物理成本约为：

```text
2,827 requests × 256 token ≈ 0.72M token
```

对比正式 run 约 8.65M token 被 checkpoint gate 拒绝，即使暂不计移除 checkpoint 后
释放的普通 KV 容量，tail replay 的量级也显著更小。这个数字只是 ATOM 内部反事实
性能上界，不代表 SGLang 的实现方式。调查原型实现了
`ATOM_STATE_REPLAY_TAIL_TOKENS=128`：禁用 persistent state checkpoint、命中固定退一个
256-token block、分配 fresh state slot。这个 fresh slot 可以承接重放后的 C128，但它
同时丢弃了 prefix hit 应恢复的 C4/SWA state，因此不是正确实现。该原型已完成
900-second C32 对照，checkpoint records/units 全程为 0。

作为 ATOM 内部反事实，一个正式窗口性能上界可以闭合物理 prefill 差额。全程固定
replay 的比例为：

```text
3,185 requests × 256 / 449,326,052 full tokens = 0.182pp
checkpoint gate loss                             = 1.926pp
保守净改善（不计 compressed KV 容量回升）        = 1.744pp
```

把 1.744pp 施加到正式窗口的 381,633,267 logical prompt token，约少算 6.66M token；
ATOM 的 19.76M physical new token 会降到约 13.10M，已经接近 SGLang 实测的 13.30M。
移除 checkpoint 还会释放 PAGE pool，因此这是偏保守而不是乐观的预测。

### 900-second tail-replay A/B：性能上界，不是正确性闭环

两个 run 同时启动，软件、模型、C32、16K scheduler、0.9 memory utilization 和 MTP
配置相同；baseline 位于 `crsuse2-m2m-108`，replay 位于 `crsuse2-m2m-107`。除
`ATOM_STATE_REPLAY_TAIL_TOKENS=128` 和对应的 block-manager 实验代码外，engine timing
与 model-runner 代码逐字节相同。

| 指标 | 32K checkpoint baseline | 128-token tail replay | 变化 |
|---|---:|---:|---:|
| 总 token throughput | 96,642.46 tok/s | 112,966.17 tok/s | +16.89% |
| request throughput | 1.03511 req/s | 1.13085 req/s | +9.25% |
| input throughput | 96,008.44 tok/s | 112,182.80 tok/s | +16.85% |
| output throughput | 634.03 tok/s | 783.38 tok/s | +23.56% |
| 完成请求 | 973 | 1,063 | +90 |
| physical prefill batches | 1,601 | 1,091 | -31.86% |
| physical prefill new token | 7,513,329 | 5,289,905 | -29.59% |
| physical new token/request | 7,721.82 | 4,976.39 | -35.55% |
| profile-window physical hit | 91.675% | 94.984% | +3.309pp |
| TTFT p50 | 860.31 ms | 439.91 ms | -48.87% |
| ITL p50 | 29.89 ms | 23.59 ms | -21.08% |
| ITL p90 | 44.39 ms | 35.29 ms | -20.50% |

总窗口完成请求数不同会改变后半段 trace mix，所以又按
`source_trace_id/source_outer_idx/turn_index` 对齐了两边共同完成的 971 个 request：

| 同请求集合指标 | baseline | replay | 变化 |
|---|---:|---:|---:|
| logical input token | 90,183,983 | 90,183,727 | -0.00028% |
| output token | 595,631 | 595,645 | +0.00235% |
| usage cache-read hit | 91.813% | 94.712% | +2.899pp |
| logical cache-read miss | 7,383,599 | 4,768,815 | -35.41% |
| TTFT median | 855.60 ms | 431.83 ms | -49.53% |
| request latency median | 8,258.21 ms | 6,160.63 ms | -25.40% |
| decode duration median | 6,823.43 ms | 5,566.54 ms | -18.42% |

因此吞吐改善不是“replay 恰好跑到了更短的请求”：共同请求的 input/output token
几乎完全一致，而 cache miss 和延迟同时显著下降。

whole-run cache 计数还能拆出两个独立收益：

| cache 指标 | baseline | replay | 变化 |
|---|---:|---:|---:|
| compressed hit | 90.975% | 91.534% | +0.559pp |
| checkpoint/replay loss | 2.677pp | 0.197pp | -2.480pp |
| actual hit | 88.298% | 91.337% | +3.039pp |
| profile 末 checkpoint records | 1,408 | 0 | -1,408 |
| profile 末 checkpoint PAGE units | 25,344 | 0 | -25,344 |
| profile 末 checkpoint GiB（per GPU） | 35.923 GiB | 0 | -35.923 GiB |

baseline 的 checkpoint 在 profile 末占 `25,344 / 86,938 = 29.15%` PAGE units；replay
不但移除了 gate loss，也让 compressed hit 回升 0.559pp，验证了“重算 + 共池挤压”
这两个机制。将短跑测得的 per-request physical prefill 降幅归一到正式 baseline：

```text
6,988 × (1 - 35.55%) ≈ 4,504 physical token/request
SGLang 实测            = 4,354 physical token/request
```

归一后只差约 3.4%，与前面的保守 13.10M 对 13.30M 预测同量级。短跑吞吐提升
16.89% 已超过原始 13.61% gap，但两边位于不同节点，不能把这个数字直接当作最终
production uplift。更重要的是，这些 cache hit、物理重算和同请求延迟只证明“少算会
更快”，不能证明少算后的模型状态正确；这个 ATOM-only 实验本身也不能确认跨框架
第一主因。第一主因由后续 clean 双方 physical-token 计数闭环，而不是由 replay 原型
得出。

正确性/稳定性检查：两边均为 `submission_valid=true`、`was_cancelled=false`、
`error_summary=[]`，没有 server exception/OOM；MTP acceptance 分别为 49.648% 和
49.670%，average tokens/forward 分别为 2.48944 和 2.49009。OSL mismatch 为
11/973 对 9/1,063，平均 mismatch 0.701% 对 0.720%，没有回归。两边在 profile drain
时都出现同一条 AIPerf cancelled-credit timeout，属于共同的收尾行为，不是 replay
新增异常。当前验证只覆盖输出长度和服务稳定性，没有覆盖输出 token 内容或 logits；
在 C4/SWA hit state 未恢复的情况下，该原型应视为未通过正确性验证，不能 production
合入。必须先补固定 prompt 的 cache-hit/cache-miss token/logit 逐项等价测试。

### 0.8 到 0.9 的 A/B 证据

ATOM C32 从 `gpu_memory_utilization=0.8` 提高到 0.9 后：

```text
KV blocks:       约 65.7K -> 86.0K，增加约 31%
总吞吐:          约 105.09K -> 105.58K tok/s，仅增加约 0.47%
compressed hit:  94.86% -> 95.09%，增加约 0.23 个百分点
```

这说明单纯提高 nominal PAGE budget 不能解决问题；新增空间同时可被更多 full-state
checkpoint 消耗，因此这个 A/B 不能再用来排除“checkpoint 造成的有效 KV 容量压力”。

## 已排除或基本排除的方向

### MTP 接受率

```text
ATOM acceptance rate          = 49.63%
ATOM average tokens/forward   = 2.489
SGLang accept rate            ≈ 49.5%
SGLang accept length          ≈ 2.48
```

两边几乎一致，不能用 speculative decoding 接受率解释主要 gap。

### Nominal GPU KV capacity

ATOM 0.9 的 GPU KV 容量比 SGLang 更大，而且从 0.8 到 0.9 基本没有带来吞吐提升。
所以不能把差距归因于 SGLang “nominally 分配了更多 GPU KV”。但 ATOM 的同一 PAGE
pool 还承载 full-state checkpoint，不能把 `num_kvcache_blocks` 全部视作可留给普通 KV
的有效容量；这正是本轮新增的主因。

### HiCache CPU hit

SGLang C32 的 CPU hit 仅 0.154%，不具备解释 13.61% 总吞吐优势的量级。

### 请求失败

ATOM 只有 1 个孤立的 client connection error，提交仍有效，服务端没有 crash/OOM。

## A/B 状态和后续优先级

### 已完成 A/B 1：ATOM prefill chunk 16K 对 8K

必须同时修改 scheduler cap 和 attention 内部切片：

```ini
max-num-batched-tokens=8192
attn-prefill-chunk-size=8192
```

保持以下参数不变：

```ini
concurrency=32
state-checkpoint-interval-tokens=32768
gpu-memory-utilization=0.9
level=3
cudagraph-mode=FULL
```

原先只改 `attn-prefill-chunk-size` 的 run 不能回答 scheduler 8K A/B，已停止并保留
为无效实验。corrected run 的 schedule log 已确认单请求 prefill chunk 最大值正好为
8192。

同为累计 800 请求时，corrected 8K/32K 和原 16K/32K 的 cache stats 几乎完全一致：

| scheduler/checkpoint | actual hit | compressed hit | gate loss |
|---|---:|---:|---:|
| 16K / 32K | 86.05% | 88.74% | 2.69pp |
| 8K / 32K | 85.99% | 88.69% | 2.71pp |

这证明 8K scheduler cap 不修复 state resume/cache 问题。完整 1-hour profile 结果也
没有收益：

| 指标 | 16K baseline | 8K scheduler | 变化 |
|---|---:|---:|---:|
| 总 token throughput | 105,580.33 | 105,364.35 | -0.20% |
| physical prefill batches | 4,953 | 5,625 | +13.57% |
| physical new token/request | 6,988 | 6,902 | -1.24% |
| TTFT p50 | 766.17 ms | 777.01 ms | +1.41%（变差） |
| ITL p50 | 24.02 ms | 24.49 ms | +1.93%（变差） |
| ITL p90 | 38.98 ms | 38.90 ms | 基本持平 |

在 ATOM 当前同步 loop 下，切成更多 batch 没有 overlap 去隐藏新增调度/launch 开销。

重点看：

- physical new tokens/request 和逻辑/物理 hit；
- 总 token/request/output throughput；
- active decode throughput；
- time-to-second-token p50/p90；
- ITL p50/p90；
- effective decode concurrency。

这个 A/B 用于量化 chunk size 和 overlap/interleaving 带来的影响。

### 已完成 A/B 2：固定 tail replay，移除 persistent full-state checkpoint（性能实验）

调查开关：

```ini
ATOM_STATE_REPLAY_TAIL_TOKENS=128
```

实测 checkpoint units/records 为 0；compressed hit 与 actual hit 的差只剩每请求固定
一个 block 的 intentional replay。physical new tokens/request -35.55%，总吞吐
+16.89%，TTFT p50 -48.87%，ITL p50/p90 -21.08%/-20.50%；错误率、OSL 和 MTP
acceptance 无回归，但这些指标不能检测 state/logit 漂移。该实现使用 fresh state，只重算
尾部一个 block，当前应视为正确性未通过的性能上界实验。完整数据见“900-second
tail-replay A/B”小节。

### 已完成的 interval A/B：32K 对 16K checkpoint

| 指标 | 32K baseline | 16K checkpoint | 变化 |
|---|---:|---:|---:|
| 总 token throughput | 105,580.33 | 106,844.67 | +1.20% |
| physical prefill new token | 19,755,993 | 18,574,704 | -5.98% |
| physical new token/request | 6,988 | 6,490 | -7.13% |
| TTFT p50 | 766.17 ms | 747.07 ms | -2.49% |
| ITL p50 | 24.02 ms | 23.60 ms | -1.78% |
| ITL p90 | 38.98 ms | 37.77 ms | -3.11% |

这确认 checkpoint 加密有小幅收益，但远小于 tail replay 的理论空间；它同时增加
checkpoint 存量、压缩普通 KV，无法继续线性扩展。

### 已完成 A/B 3：phase timing / overlap 上限

按 batch kind 累计 schedule、prepare、forward wait、KV poll、postprocess、output 和
total wall time。两个 888-second 内部窗口的非-forward 占比分别为 1.174% 和 1.255%；
同步 loop 的 CPU bubble 不具备解释 13.61% gap 的量级，暂不优先实现 SGLang 式
result queue/双 batch overlap。

### A/B 4：修复或绕开 hipBLASLt fallback

对日志中两组 shape 做 microbenchmark，并比较：

- 正常 hipBLASLt 路径；
- 当前 fallback 路径；
- SGLang 对应 BF16 GEMM backend。

修复后以原始 16K 参数重跑 C32，避免和 chunk A/B 混在一起。

### 已完成 A/B 5：同 token、同 physical work 的 prefill kernel timing

已完成 8-rank 严格对齐，结果见“严格同 token、同 physical work 的 8-rank trace”。
同-work 主 cluster 中 ATOM 比 SGLang 长 12.420 ms；最大 family gap 是 collective
12.398 ms，其中 7.382 ms 是 ATOM 独有的 6 次 MTP one-stage reduce 的
active/synchronization time；由于明显的 rank-role/等待特征，不能把它直接当作删除调用
即可回收的 critical-path 时间。MoE 总量基本持平，PA prefill 仅差 2.286 ms，不能再把
整个 gap 泛化成 attention/MoE kernel 慢。

本次只覆盖 prefill；若 physical-work 优化后仍有稳定 decode gap，再单独抓相同 batch
size、相同 MTP accept path 的 decode/verify trace，不能与本次 prefill 数字混用。

### A/B 6：SGLang no-HiCache

如果 CI 参数允许，跑同一 C32 且关闭 HiCache。根据当前仅 0.154% 的 CPU hit，预计
结果变化较小；这个实验主要用于彻底排除 offload 路径的影响。

## c4/c8/c16 short-decode 严格 physical-prefill 对齐（2026-08-16）

这轮不跑 c128，只跑 `max_tokens=4/8/16`。每个 target 都是两个异构请求：

```text
long:  32768 cached + 2816 new
short:     0 cached +  256 new
batch: 32768 cached + 3072 new
```

SGLang DSV4 state-compatible resume 会从最长 radix match 回退一个 256-token page。
因此不能用 `prefix + 任意 256 token` prime；任意页只与 target 匹配到 32768，回退后
实际成为 32512 hit / 3328 new。本轮改为每个 trial 在计时和 profiler 之外单独提交：

```python
prefix + extension[:256]
```

这样最长 match 是 33024，回退一页后正好是 32768。server log 硬门禁为：24 个 timing
target 加 3 个 profile target，27/27 都是 `cached=32768, new=3072`；三档 warm target
仍为预期的 32512/3328，不计入结果。无效的 throwaway-prime 数据已单独归档。

有效 SGLang 数据：

```text
/mnt/m2m_nobackup/hyi/sglang-real-decode-c4-c8-c16-20260816/client
/mnt/m2m_nobackup/hyi/sglang-real-decode-c4-c8-c16-20260816/torch_trace
/mnt/m2m_nobackup/hyi/sglang-real-decode-c4-c8-c16-20260816/client-cpu-trace
/mnt/m2m_nobackup/hyi/sglang-real-decode-c4-c8-c16-20260816/torch_trace_cpu
/mnt/m2m_nobackup/hyi/sglang-real-decode-c4-c8-c16-20260816/spec-work-audit
```

### 先过 workload gate，不能直接拿旧 ATOM synthetic run 对比

旧 ATOM short run 带 `--spec-decode-acceptance-rate=0.4966666667`，是强制 synthetic
acceptance；SGLang 使用真实 draft/target agreement。profile target 的 proposal/verify
rounds 分别为：

| shape | ATOM synthetic | SGLang real（GPU trace） |
|---|---:|---:|
| c4 | 3 | 4 |
| c8 | 5 | 7 |
| c16 | 7 | 12 |

SGLang native `/generate` 的逐请求审计还显示 acceptance 对 seed 很敏感，同一 batch 的
两条请求可能分别需要 `5/12` 次 verify；重复运行也会改变 profile target 的总轮数。
所以短请求 wall time 的十几到几十毫秒波动，首先要用 proposal/verify work 解释，不能
直接归到某个 kernel 或 collective。

关闭 ATOM synthetic acceptance 后，whole-run acceptance 为 34.05%，并确认每个
measured target 仍是 32768/3072。但同时暴露出独立的 `max_tokens` 语义问题：ATOM
在最后一个 MTP verify block 接受多个 token 时，只设置 `finish_reason=max_tokens`，没有
裁掉超过预算的 accepted tail；SGLang 始终严格截断。

8 个 timing target 的 completion token 总量：

| shape | 理论/SGLang | ATOM real（修复前） | 多生成 |
|---|---:|---:|---:|
| c4 | 64 | 82 | +28.1% |
| c8 | 128 | 140 | +9.4% |
| c16 | 256 | 280 | +9.4% |

因此修复前的 ATOM/SGLang wall 不是同输出 work。修复在 scheduler 的已有 EOS/stop trim
路径上增加 `max_tokens` overflow trim：同时裁 internal committed length 和 client-visible
`new_tokens`，不触碰 state checkpoint pool/lifecycle/eviction，也不触碰 SWA hit/COW。
修复后 8/8 timing target 都严格输出 `2 * max_tokens`：64/128/256。

ATOM real acceptance 修复前后 A/B：

| shape | 修复前 wall median | cap-trim wall median | draft work（前/后，含 profile） | 结论 |
|---|---:|---:|---:|---|
| c4 | 263.27 ms | 254.27 ms | 117 / 117 | -9.01 ms；work 对齐，但单次重启噪声仍在 |
| c8 | 314.16 ms | 315.44 ms | 249 / 246 | +1.28 ms；基本持平 |
| c16 | 381.64 ms | 405.78 ms | 378 / 405 | acceptance/work 不同，不能做性能 A/B |

cap-trim 本身发生在最后一次 verify 已完成之后，不会减少已经执行的 GPU forward；它的
主要价值是修正 API 语义和后续 workload 对齐。c4 的 9 ms 不应全部解释成这个 trim 的
GPU 性能收益。

严格输出后的 cross-backend wall median 是：

| shape | SGLang real | ATOM real cap-trim | ATOM - SGLang |
|---|---:|---:|---:|
| c4 | 241.27 ms | 254.27 ms | +12.99 ms / +5.4% |
| c8 | 296.13 ms | 315.44 ms | +19.31 ms / +6.5% |
| c16 | 345.30 ms | 405.78 ms | +60.48 ms / +17.5% |

这张表只完成了 physical prefill 和输出长度对齐；真实 acceptance/verify work 仍未逐 trial
锁定，尤其 c16 的 ATOM 本轮 draft work 明显更高，所以不能把最后一列当作 kernel gap。

### short-decode trace 对 collective 的限制

SGLang 的 CPU+GPU trace 完整；ATOM 当前 torch trace 虽请求了 CPU+CUDA activity，实际
只含 `cpu_op/user_annotation`，没有 `kernel/gpu_memcpy` 事件。故这两批 short-decode
trace 不能直接比较“8 卡 GPU collective 时间”。SGLang GPU trace 的 8-rank mean 仅用于
描述自身：

| shape | GPU span | union busy | collective calls | collective summed GPU |
|---|---:|---:|---:|---:|
| c4 | 278.39 ms | 243.88 ms | 172 | 38.09 ms |
| c8 | 349.84 ms | 332.20 ms | 205 | 39.32 ms |
| c16 | 460.84 ms | 340.14 ms | 264 | 37.73 ms |

CPU-only 同口径 trace 显示总 envelope 主要随 verify rounds 变化：ATOM cap-trim profile
为 3/6/10 rounds、245.0/304.8/422.9 ms；SGLang CPU-only profile 为 4/7/13 rounds、
264.8/351.8/472.8 ms。它不支持“ATOM short decode 是 collective 固定慢一截”的结论。
前文同-work prefill 的 12.398 ms collective family gap 仍然有效，但不能外推为这轮
short-decode wall gap。

回归：`tests/test_scheduler.py` 和 `tests/test_mtp_deferred_publish.py` 共 106 passed；新增
用例覆盖 MTP 最后一轮接受 3 token、`max_tokens=4` 时 internal completion 和 streamed
output 都严格裁成 4。

## c4/c8/c16 target-only：graph-on wall 与 eager GPU trace（2026-08-16）

为把真实 MTP acceptance 的轮数变化完全拿掉，另做了一轮 target-only 对照。两端都关闭
MTP proposal，只执行相同的 target forward；每个 target 仍严格是：

```text
32768 cached + 3072 new
completion = 2 * max_tokens
max_tokens = 4 / 8 / 16
```

每档 8 次无 profiler timing 的中位数为：

| shape | ATOM graph-on | SGLang graph-on | ATOM - SGLang |
|---|---:|---:|---:|
| c4 | 238.792 ms | 228.545 ms | +10.247 ms |
| c8 | 310.073 ms | 296.290 ms | +13.783 ms |
| c16 | 452.313 ms | 430.585 ms | +21.728 ms |

对三点分别做线性拟合：

```text
ATOM   = 167.672 ms + 17.7915 ms * round
SGLang = 161.398 ms + 16.8296 ms * round
gap    =   6.275 ms +  0.9619 ms * round
```

所以 short decode gap 不是一个固定 checkpoint resume 开销：约 6.3 ms 是
prefill/resume/API 固定项，另有约 0.962 ms 随每轮 target forward 线性增长。原始结果在：

```text
/mnt/m2m_nobackup/hyi/atom-sglang-target-only-c4-c8-c16-20260816
```

### graph-off wall 只做行为确认，不能和 graph-on gap 混算

为了让 profiler 看见 graph 内部 kernel，两端另跑 eager/graph-off。无 profiler wall
中位数为：

| shape | ATOM eager | SGLang eager |
|---|---:|---:|
| c4 | 809.739 ms | 730.523 ms |
| c8 | 1433.010 ms | 1364.627 ms |
| c16 | 2715.443 ms | 2600.509 ms |

eager 的 Python launch 开销远大于正式 graph-on，不能用这张表减出 production kernel
gap；它只证明 c4/c8/c16 的 batch、输出 token 和每轮执行结构都稳定。原始数据：

```text
/mnt/m2m_nobackup/hyi/atom-sglang-eager-target-only-c4-c8-c16-20260816
```

### c4 eager 同窗口 kernel：state/SWA 和 collective 基础算子不是主因

ATOM Kineto 在该 ROCm 组合下没有 GPU event，因此 ATOM 用 rocprof；SGLang 用
CPU+GPU Kineto。窗口都只保留四轮 target decode。ATOM 的推进 rank 是 rank5，SGLang
是 TP0；其他 rank 的 custom all-reduce duration 大多是在 GPU 上轮询等待推进 rank，
不能当作通信计算时间。

| family（每轮） | ATOM rank5 | SGLang TP0 | 说明 |
|---|---:|---:|---|
| kernel event 数 | 2628 | 3271.25 | SGLang eager 有更多 copy/metadata kernel |
| non-collective 总量 | 14.850 ms | 16.673 ms | SGLang eager 反而多 1.823 ms |
| GEMM | 5.696 ms | 5.507 ms | 接近 |
| MoE | 2.996 ms | 3.543 ms | SGLang 多 0.547 ms |
| attention | 1.704 ms | 1.085 ms | SGLang 少 0.619 ms |
| elementwise | 1.872 ms | 1.744 ms | 接近 |
| memory | 0.516 ms | 2.070 ms | SGLang 多 1.554 ms |
| other | 1.876 ms | 2.347 ms | SGLang 多 0.471 ms |
| state/SWA update | 0.191 ms | 0.375 ms | SGLang 四轮均值含 terminal 轮额外更新；前三轮约 0.268 ms |
| collective | 0.928 ms | 4.532 ms | SGLang 均值被 rank 等待污染，不能横向相减 |

SGLang 的 memory 增量主要是每轮 185 次 BF16 copy（0.859 ms）和 274 次 FP32 copy
（0.813 ms）。同名 kernel 也不支持“SGLang 基础 kernel 全面更快”：

- 91 次主 blockscale GEMM：ATOM 1.024 ms，SGLang 0.970 ms；
- MoE stage1：ATOM 0.739 ms，SGLang 0.819 ms；
- MoE sorting：ATOM 0.701 ms，SGLang 0.821 ms；
- MoE stage2：ATOM 0.457 ms，SGLang 0.435 ms，但 SGLang 另有 0.214 ms MoE reduce；
- ATOM MLA decode 主 kernel 0.965 ms；SGLang paged split+reduce 合计约 0.712 ms；
- ATOM 91 次 compressor state update 为 0.191 ms；SGLang C4/C128 write 合计约
  0.272 ms；双方都在 restore 后更新正确 state，不是 replay C4/SWA；
- ATOM 推进 rank 的 123 次 one-stage reduce 为 0.911 ms；SGLang 最干净 c4 round
  仍约 1.44 ms，不能说明 SGLang 的 collective 基础带宽更快。

c8/c16 进一步验证：SGLang TP0 的 non-collective 每轮分别为 15.541/15.810 ms，和 c4
同量级；collective 汇总却升到 24.697/60.463 ms。这种“轮数越多 collective 越慢、
non-collective 不变”的形状正是 profiler 下 peer-arrival 等待累积，不是模型每轮通信量
增长。因此正式的 0.962 ms/轮不能从 eager collective summed duration 直接读取。

### graph replay 的 rank 到达偏斜

正式 graph-on CPU trace 虽看不到 ATOM graph 内部 GPU kernel，但能比较每轮 8-rank
replay 提交的最早/最晚时间。ATOM 用 `decode[bs=2 tok=2 d=2]` annotation 起点，SGLang
用 `hipGraphLaunch` 起点：

| shape | ATOM 平均 8-rank skew | SGLang 平均 8-rank skew | ATOM - SGLang |
|---|---:|---:|---:|
| c4 | 1.551 ms | 0.888 ms | +0.663 ms |
| c8 | 1.299 ms | 0.621 ms | +0.678 ms |
| c16 | 1.483 ms | 1.073 ms | +0.410 ms |

这与 custom all-reduce 的工作方式吻合：早到 rank 在 kernel 内轮询，最晚 rank 决定该轮
何时通过。它能解释每轮 gap 的一部分，但 profiler 会明显放大前几轮的 CPU 调度抖动，
不能把上表逐毫秒等价成无 profiler wall 收益。

尝试用 rocprof selected-region 直接抓 graph-on 内部 kernel 时，8 个 worker 在第一轮
profiled prefill 全部卡在 ROCProfiler HSA async-signal interposition，并持续报告
`Async signal handler still waiting on signal`；请求两分钟没有推进。该运行已终止并保留
日志，不能作为性能数据：

```text
/mnt/m2m_nobackup/hyi/atom-graph-on-rocprof-c4-20260816
```

### `ATOM_NUMA_BIND=1` A/B：实测回收约 0.399 ms/轮

SGLang 正式容器设置了 `SGLANG_SET_CPU_AFFINITY=1`，ATOM 正式 baseline 没有启用现成
的 NUMA binding。只增加 `ATOM_NUMA_BIND=1`，其余 server 参数、`--mark-trace`、token、
checkpoint 和 SWA 路径完全不变。日志确认 GPU0-3 绑定 node0、GPU4-7 绑定 node1，
每个 node 允许 118 个 CPU。

| shape | ATOM baseline | ATOM NUMA bind | 收益 | SGLang | NUMA 后 gap |
|---|---:|---:|---:|---:|---:|
| c4 | 238.792 ms | 237.185 ms | 1.607 ms | 228.545 ms | +8.640 ms |
| c8 | 310.073 ms | 307.673 ms | 2.400 ms | 296.290 ms | +11.383 ms |
| c16 | 452.313 ms | 446.078 ms | 6.235 ms | 430.585 ms | +15.493 ms |

NUMA A/B 拟合为：

```text
ATOM NUMA = 167.982 ms + 17.3924 ms * round
NUMA gain =  -0.310 ms +  0.3990 ms * round
remaining gap vs SGLang = 6.584 ms + 0.5629 ms * round
```

也就是仅做 NUMA-local CPU/memory binding，回收了原 0.9619 ms/轮差距的约 41%。30/30
个异构 target 仍全部是 `cached=[32768,0], new=[2816,256]`，completion 严格为
8/16/32，`lost_to_checkpoint=0`。这项 A/B 没有改 checkpoint pool/lifecycle/refcount，
也没有改 SWA hit/COW。

NUMA 后 profile trace 的 launch skew 并非三档单调下降（profile 对前几轮 CPU 调度的
扰动很大），所以不能把全部 0.399 ms/轮都写成“skew 下降”；更准确的结论是 CPU/内存
NUMA locality 确实进入正式 critical path，而 rank 到达偏斜是其中一个可见症状。当前
binding 仍把同一 node 上四个 worker 都放进同一个 118-core mask，尚未做到 SGLang 式
per-rank CPU shard。下一步应在同一 NUMA node 内给每个 TP worker 分配互不重叠的 CPU
子集做独立 A/B；不要在 graph 前直接加 barrier，因为 barrier 本身可能把同样的等待从
GPU kernel 搬到 CPU，并不能保证减少 critical path。

原始数据：

```text
/mnt/m2m_nobackup/hyi/atom-numa-bind-aligned-target-only-c4-c8-c16-20260816
```

## 下次排查检查表

1. 确认硬件、模型 revision、镜像、TP/EP/DPA、精度和数据集完全一致。
2. active prefill 是逻辑 input-token rate；cache 策略不同时必须从 server log 另算
   physical new tokens 和 new tokens/request，不能拿它直接代表 kernel tok/s。
3. 同时比较 effective prefill/decode concurrency，避免把更高并发误认为更快 kernel。
4. 检查 MTP acceptance rate 和 average accepted length。
5. 将 theoretical、compressed、state-gated actual、GPU、CPU 和 overall hit 分开，
   同时记录 checkpoint 对 PAGE pool 的真实占用，不混用口径。
6. 区分 host pool “写满”与真正的 CPU cache hit。
7. 搜索所有 kernel fallback、OOM、retraction、preemption、graph miss 和 eager fallback。
8. 对 TP 多 rank 重复告警按时间、shape 和 rank 去重，不直接把日志行数当事件数。
9. 比较 prefill chunk、batch token cap、overlap schedule、CUDA graph capture sizes。
10. 每次只修改一个主变量，保留 run manifest、server log、AIPerf JSON 和 cache/MTP
    统计。
11. 逐 kernel 对比必须同时固定 token 内容、cached/new token shape、request 数和 TP
    rank；SGLang trace 需要剔除主 prefill 后的 completion/sampling cluster。

## 当前排查优先级总结

```text
1. 在不改变语义的前提下降低 ATOM physical new token/request：
   C128 对齐块可 replay；C4 state 和每层 SWA 必须通过 checkpoint/COW/copy hit/restore。
   clean 900s 已确认 +56.27% physical work 是第一主因
2. 优化 ATOM collective：先判断 max_tokens=1/请求已完成时是否可跳过完整 3-step MTP
   proposal，并定位 6 次 one-stage reduce 的 rank 等待来源；7.382 ms 是
   active/synchronization time，不预设为可完整回收。再看共同 2-stage reduce 的长尾
   3.750 ms 和 all-gather 的 1.152 ms
3. DSV4 单节点先启用 `ATOM_NUMA_BIND=1` 做 production A/B；target-only 已实测回收
   0.399 ms/decode round。然后实现同一 NUMA node 内 per-rank CPU shard 的独立开关，
   对齐 SGLang CPU-affinity 方式；不要用 graph 前 barrier 代替 affinity
4. 收敛主 prefill cluster 内额外 6.514 ms idle/launch gap；分别测 profiler-off wall time，
   避免把跨 profiler overhead 当真实收益
5. physical-work、NUMA 和 collective 修复后，用同节点 clean 900s/1-hour sequential A/B 验证
   production uplift，并重算 physical new token/request
6. target-only NUMA 后仍剩约 `6.584 ms + 0.5629 ms/round`；优先比较 graph-on
   attention path 和 graph 外 metadata/input-copy，再做 hipBLASLt fallback microbenchmark
7. 如仍有 decode gap，需换用不会死锁 graph replay 的 GPU tracer；当前 rocprof
   selected-region 在 graph-on HSA signal interposition 中卡死，不能继续拿它做数据
8. 8K scheduler chunk、同步 CPU loop、nominal KV capacity、HiCache CPU hit、MTP
   acceptance 已排除为第一主因
```
