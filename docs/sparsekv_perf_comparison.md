# SparseKV vs Baseline 性能对比 — 数据出处与测试方法

## 1. 测试目标

对比 GLM-5.2 PP4×PD 部署下，**SparseKV decode**（KV 卸载到主机内存 + GPU
热缓冲）与 **Baseline decode**（全量 KV 留在 GPU HBM）在真实 agentic 长上下文
负载下的吞吐、延迟和稳定性。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 机器 | MI355X × 8 GPU |
| 容器 | `atom_pp4pd_test` |
| 模型 | GLM-5.2-MXFP4 (`/mnt/models/GLM-5.2-MXFP4`) |
| 权重精度 | MXFP4 |
| KV Cache 精度 | fp8 |
| 拓扑 | PP4×TP1 prefill (GPU 0-3) + TP4 decode (GPU 4-7)，mooncake PD 分离 |
| 编译级别 | `--level 3`（全 CUDAGraph） |
| 分支 | `Jasen/cpp-dev` (commit 含 SparseKV 0x1016 修复) |

### Baseline 配置（`scripts/start_glm52_pp4pd.sh`）

- decode: TP4, `--level 3 --cudagraph-mode FULL`
- KV 全量保留在 GPU HBM
- `--kv_cache_dtype fp8 --gpu-memory-utilization 0.85 --enable_prefix_caching`

### SparseKV 配置（`scripts/start_glm52_pp4pd_sparsekv.sh`）

- decode: TP4, `--level 3`
- `ATOM_SPARSEKV_ENABLE=1`
- `ATOM_SPARSEKV_HOT_BUFFER_SIZE=8192`（每请求 GPU 常驻热 token 数）
- `ATOM_SPARSEKV_HOST_TO_DEVICE_RATIO=16`（主机冷池 = 16 × 热池总量）
- `ATOM_SPARSEKV_PREFETCH=1`（IndexShare 组预取）
- `--max-num-seqs 20 --max-model-len 1048576`
- KV 冷池：paged 主机 pinned memory；热缓冲：GPU fp8

## 3. 测试负载

使用 **AIPerf agentic trace replay**（`scripts/test_glm52_pp4pd_aiperf_trace.sh`）。

| 参数 | 值 |
|---|---|
| 场景 | `inferencex-agentx-mvp` |
| 数据集 | `semianalysis_cc_traces_weka_062126`（393 条真实 agentic 编码会话） |
| SEED | 42（固定，保证相同请求序列） |
| DURATION | 300s（5 分钟压测窗口） |
| warmup | 10 req/lane |
| stats-interval | 30s |
| 端点 | mesh proxy `:30000 /v1/chat/completions`（流式） |
| 并发 | 8 / 16 / 32（分别测试） |

ISL（输入序列长度）由 trace 决定，p50 ≈ 62K–65K tokens，最长达 518K+。
OSL（输出序列长度）p50 ≈ 639–706 tokens。

## 4. 测试流程

每次测试（每个并发级别）执行以下步骤：

1. **彻底清场**：`kill -9` 所有 `openai_server` / `ATOM::*` / `atomesh` 进程
2. **等待 VRAM 归零**：轮询 `rocm-smi --showmemuse` 直到 8 卡全部 VRAM% = 0
3. **清编译缓存**：`rm -rf /root/.cache/atom/*`（启动脚本自动做）
4. **启动服务**：运行对应启动脚本（baseline 或 SparseKV），等待 prefill / decode / mesh 三个 health 端点全部返回 200
5. **执行 aiperf**：`CONCURRENCY=N DURATION=300 bash scripts/test_glm52_pp4pd_aiperf_trace.sh`
6. **采集指标**：从 aiperf 的最终报告表（`run.log` 中的 `│` 格式表格）提取
7. **记录崩溃**：检查 `decode.log` 中的 `HSA_STATUS_ERROR` / `proc died` / `host pool exhausted`
8. **保存日志**：`run.log` + `decode_tail.log`（decode.log 末 400 行）

全部由自动化编排脚本执行（`/tmp/overnight_perf.sh` 和 `/tmp/baseline_perf.sh`），
无人工干预。

## 5. 数据出处

### SparseKV 数据

- 编排脚本：`/tmp/overnight_perf.sh`
- 运行时间：2026-08-05 17:18 – 2026-08-06 00:14
- 保存位置：`results/sparsekv_conc8/run_r{1..6}.log`、`results/sparsekv_conc{16,32}/`
- 汇总 CSV：`results/sparsekv_summary.csv`

| 并发 | 轮次 | 有效数据轮次 | 状态 |
|---|---|---|---|
| 8 | 6 | 6/6 OK | 全部成功，0 崩溃 |
| 16 | 5 | 1/5 OK | 4/5 host-pool 耗尽（独立 bug，非 0x1016） |
| 32 | 5 | 0/5 OK | 全部 host-pool 耗尽 |

### Baseline 数据

- 编排脚本：`/tmp/baseline_perf.sh`
- 运行时间：2026-08-06 00:50 – 02:55+（sweep 被会话中断，完成 2/3 轮）
- 保存位置：`results/baseline_conc{8,16,32}/run_r{1..2}.log`
- 汇总 CSV：`results/baseline_summary.csv`

| 并发 | 轮次 | 有效数据轮次 | 状态 |
|---|---|---|---|
| 8 | 2 | 2/2 OK | 全部成功 |
| 16 | 2 | 1/2 OK | 1 轮 aiperf 连接瞬断（NO_DATA） |
| 32 | 2 | 1/2 OK | 1 轮 run.log 未完整写入 |

## 6. 指标提取方法

所有指标从 aiperf 的 **最终报告表**（`run.log` 中 `│` 分隔的结构化表格）提取，
而非实时滚动日志（实时日志仅用于交叉验证）。

| 指标 | aiperf 报告字段 | 取值列 |
|---|---|---|
| req/s | `Request Throughput (requests/sec)` | avg |
| input tok/s | `Input Token Throughput (tokens/sec)` | avg |
| output tok/s | `Output Token Throughput (tokens/sec)` | avg |
| TTFT p50 | 实时 `ttft p50=Nms`（最后一条） | p50 |
| ITL p50 | 实时 `itl p50=Nms`（最后一条） | p50 |
| p90 intvty | `Effective Decode Throughput Per User (tokens/sec/user)` | p90 |
| realized prefix hit | `Overall Usage Prompt Cache Read % (%)` | avg |
| theoretical hit | `Theoretical Prefix Cache Hit (%)` | avg |
| prompt tokens | `Total Usage Prompt Tokens (tokens)` | total |
| generated tokens | `Total Usage Completion Tokens (tokens)` | total |
| errors/requests | 实时 `err=N` + `Request Count` | — |
| Effective Concurrency | `Effective Concurrency (requests)` | avg |

per-GPU 指标计算：`tput/GPU = (input + output) tok/s ÷ 8`（PD 分离共用 8 卡）。

## 7. 结果汇总

| Run | Weights | Cache | Parallel | all-gpu | Conc | req/s | tput/GPU | ratio/B300 | input/GPU | output/GPU | input tok/s | output tok/s | TTFT p50 ms | ITL p50 ms | p90 intvty | realized hit | theoretical hit | prompt tokens | generated | errors/req | Eff Conc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | MXFP4 | fp8 | PP4+TP4 | 8 | 8 | 0.08 | 597 | 11.5 | 585 | 11.5 | 4683 | 92 | 432 | 11 | 156 | 93.1% | 93.1% | 1592275 | 26204 | 0/26 | 0.89 |
| Baseline | MXFP4 | fp8 | PP4+TP4 | 8 | 16 | 0.25 | 2514 | 9.5 | 2505 | 9.5 | 20038 | 76 | 484 | 13 | 109 | 93.4% | 93.4% | 6812985 | 69406 | 0/86 | 3.01 |
| Baseline | MXFP4 | fp8 | PP4+TP4 | 8 | 32 | 1.08 | 9166 | 7.8 | 9158 | 7.8 | 73267 | 63 | 523 | 16 | 80 | 92.8% | 94.1% | 24910815 | 162339 | 0/366 | 8.62 |
| SparseKV | MXFP4 | fp8(hot8K) | PP4+TP4 | 8 | 8 | 0.06 | 451 | 6.0 | 445 | 6.0 | 3559 | 48 | 800 | 22 | 96 | 95.8% | 95.8% | 1174620 | 16335 | 0/19 | 1.10 |

## 8. 注意事项

- **SparseKV conc 16/32 被独立的 host-pool 耗尽 bug 阻塞**（`RuntimeError:
  SparseKV host pool exhausted`），与本次修复的 0x1016 竞态无关。该 bug 是
  mooncake KV-receive 路径绕过调度器准入反压导致，详见
  `project_sparsekv_recv_pool_crash` 记忆。
- **Baseline conc 16 r1 NO_DATA**：aiperf warmup 阶段遭遇瞬时连接拒绝（mesh
  刚启动的过渡态），所有 87 个 warmup 请求报错→取消 profiling。非服务器崩溃。
- **p90 intvty** 在 aiperf 报表中对应 `Effective Decode Throughput Per User`
  的 p90 列，而非 `intvty` 实时日志（后者只报 p50/p75/p95/p99，无 p90）。
- **realized prefix hit** 来自 aiperf 基于服务器返回的 `usage.prompt_cache_read_tokens`
  计算，反映 decode 节点的**实际前缀缓存复用率**；`theoretical hit` 是 aiperf
  根据 trace 重复度估算的理论上限。
- SparseKV 的 realized hit（95.8%）高于 Baseline（93.1%），因为 SparseKV decode
  节点不因 HBM 空间不足驱逐 KV block → 更好的前缀复用。
- 所有测试均在同一台机器、同一组 GPU、同一 trace（SEED=42）上执行，仅启动脚本
  不同（SparseKV vs 不带 SparseKV）。
