# AF_PIECEWISE segmented zero-copy (DeepSeek-V4 / DSpark)

分支基点：`c5f0c4bf`（`lirzhang/dspark-attn-cudagraph` 线，即已成型的 **copy 版**
AF_PIECEWISE）。本分支在其上加 **segmented（真·zero-copy）** 路径 + 省内存收尾。

> 注意：本 feature 及其前置（`attn_ffn_piecewise.py` 的 `piecewise_core`、
> `cuda_graph.py` 的 `CudagraphCaptureRunner/StableOutputs` 等）**不在 jiaoliang/main
> 上**（main 已比本线新 ~16 万行）。rebase 到最新 main 是独立的移植工作。

---

## 1. 两个版本的区别（copy vs segmented）

每层：`dense 产 q/kv → attn 读、算、产 output → 下个 dense 读 output`。

| | copy 版（默认，proven） | segmented（本分支，`ATOM_AF_SEGMENTED=1`） |
|---|---|---|
| attn↔dense 边界 | attn 从**固定 buffer** 读、写**固定 StableOutputs slot**（解耦）；每步一个小 copy | attn 直接读 dense 输出在**共享池的地址**（zero-copy，无 copy） |
| attn 捕获 | 独立 runner + 独立池 | 和 dense 同 session、共享池、整个 forward 拆成段 |
| dense 捕获 | per-piece 自捕获，per-num_tokens 池 | 段捕获，跨 bucket **dedup**（`HipGraphDedupRegistry`） |
| 内存（tp8, ≤bs256, MTP=6） | ~14GB | **~17.6GB** |
| copy 成本 | 每层 ~1KB（`copy_per_step=("positions",)`） | 无 |
| 状态 | 稳定 | 正确（GSM8K ~0.96），本分支收尾 |

**为什么 segmented 曾经更贵（22GB→治理后 13GB）**：见 §3。

---

## 2. segmented 实现

- `atom/utils/attn_ffn_segmented_cudagraph.py`：`SegmentedCudaGraphCapture`
  一次 session 捕获整个 forward 为多个段（dense piece + attn core），共享一个 pinned
  pool。`run_segment(group_key, fn, ...)` 捕获一段；协调存活性 + **weak-ref 延迟释放**
  让 dense 段的输出跨段叠加复用（`_owned`/`_consumed`）。
- `atom/utils/hip_graph_dedup.py`：ROCm 无 cuda-python 的 **hipGraphExecUpdate dedup**
  （ctypes over libamdhip64）。结构相同、只差地址的 dense 段（同 num_tokens 跨 bucket）
  共享一个 hipGraphExec。attn 段键含 layer/bs/q_eff，唯一、不 dedup。
- `atom/utils/cuda_graph.py`：`CUDAGraphWrapper.__call__` 在有 active session 时把
  dense piece 路由到 `run_segment`。
- `atom/models/deepseek_v4.py`：`v4_core_attention` 在有 active session 时把 attn core
  路由到 `run_segment`。
- `atom/model_engine/model_runner.py`：`_capture_af_segmented(_ragged)` 逐 bucket 捕获，
  per-num_tokens 池；run_model 里按 `(bs,max_q_len,num_tokens_pad)` HIT 就 replay
  segmented 图、MISS 走 eager。

---

## 3. 内存结论（实测，别再重推）

pool 拆解（segmented, tp8, MTP=6）：
- **dense 段 ~1-2GB**：大内存在“输出”，weak-ref 延迟释放让其跨层叠加复用 → reserved 极小。
- **attn 段（曾 22GB）**：大内存在“内部并发 scratch”——`q_sa`（qk_norm 输出，
  `[num_tokens, H, D]`，∝ bs）+ sparse_attn 工作 buffer，**峰值同时 live ~100MB/层
  @bs512**。而分开捕获的 attn 图**跨图不复用** scratch → ×62 层 → 爆。
- **为什么 dense 不爆**：dense 是 inductor 编译的子图（buffer 复用被规划），且其大内存
  是可跨层复用的“输出”；attn 是 **custom op（inductor 黑盒）**，内部一堆 kernel 各自
  `torch.empty`，是“不可复用的并发 scratch”。
- **大 bs 特别贵**：attn scratch ∝ num_tokens = bs×6，再 ×62 层不复用 → bs=512 一个
  bucket 就 ~6.2GB，bs=256 ~3.1GB。

### 治理（本分支已做，安全、不碰精度）
1. **不录叠加的普通 piecewise 图**：segmented replay 只用 segmented 图，普通 per-piece
   图是死重（~11GB）。`model_runner` 里 segmented 时跳过 piecewise 捕获（保留 bucket
   簿记），MISS 回退走 eager 而非 PIECEWISE。
2. **capture sizes 砍到 ≤256**（`arg_utils.py` 默认 `[1,2,4,8,16,32,48,64,128,256]`）：
   砍掉 bs=512 的 attn（~6-9GB）。只影响并发>256 的步（fallthrough）。

效果：pool **36GB → 17.64GB**，Post-init 98.3% → 93.4%，GSM8K ~0.96。

### 走不通的路（别再试）
- **共享 q_sa/scratch buffer 复用**：跨图复用只在 torch 的 `default_capture_stream` 上
  发生（见 `tools/ut_attn_qsa_share.py` 微基准），但 attn 段换到该流 + 共享 buffer 会
  **腐蚀精度**（混流 replay 顺序松 + 共享 buffer = 竞争；真实 attn 的 pool 叠加复用会
  覆盖仍需的 scratch）。q_sa 单独共享也崩。**zero-copy(混流)与 buffer 复用(需严格串行)
  天生互斥。**
- 结论：segmented 的 attn 只能靠**减 bucket**省，不能靠共享。

---

## 4. 用法

```bash
# segmented zero-copy（本分支主路径）
ATOM_AF_SEGMENTED=1 AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 \
python -m atom.entrypoints.openai_server --model /data/DeepSeek-V4-Pro-DSpark/ \
  --method dspark --num-speculative-tokens 5 --kv_cache_dtype bf16 -tp 8 \
  --trust-remote-code \
  --dspark-config '{"confidence_schedule":true,"ragged":true,"ragged_graph_sizes":"6"}' \
  --cudagraph-mode AF_PIECEWISE

# 不设 ATOM_AF_SEGMENTED → copy 版（~14GB，proven）
```

### 诊断 env（默认关，PR 前可清）
- `ATOM_AF_SEG_WEAKREF=0`：关 dense 段 weak-ref 延迟释放。
- `ATOM_ATTN_CAP_DBG=1`：capture 期间按 num_tokens 打 attn 各子 op 推高 reserved 多少。
- `[af-seg]` 内存日志（`log_segment_mem_stats`）：每类段 reserved/held 汇总。

---

## 5. 后续
- 收尾：PR 前把 §4 的诊断探针（`_cap_dbg`、`_SEG_MEM` 埋点、`ut_attn_qsa_share.py`）
  清掉。
- rebase 到最新 main（~16 万行 divergence，需连 AF_PIECEWISE 前置一起移植）。
- 若还要更省：capture sizes 砍到 ≤128（再省 ~3GB attn），代价是并发 128-256 fallthrough。
