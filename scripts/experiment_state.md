# GPT-OSS-120B MI355X Performance Optimization - Final Report

## Status: COMPLETE
## Date: 2026-04-05
## GPU Hours: 1.75h
## Total Benchmarks: 45 (targeted, not full scan)

## Machine
- Host: `smci355-ccs-aus-m13-05.cs-aus.dcgpu`
- GPU: 8x AMD Instinct MI355X (288GB HBM each), single-GPU used
- Container: `chuali_perf_opt`
- Model: `/data/openai/gpt-oss-120b` (MXFP4 quantization, GptOssForCausalLM)

## Branch
- `perf/gpt-oss-120b-mi355x-opt` based on `origin/feature/ep-optimization-gpt-oss-120b` (PR #473)

## Strategy
Targeted Pareto optimization: 5 experiments testing specific levers at high-value concurrency points only. No full scan. Each experiment tested at 3-7 concurrency points (vs 18 in full sweep). Combined best configuration tested at 9 key points.

---

## Experiment Results Summary

| # | Experiment | Status | Duration | Key Finding |
|---|---|---|---|---|
| 1 | gpu_util_095 | **SUCCESS** | 27min | +3.3% throughput, **+69% TTFT improvement** at c256 |
| 2 | cudagraph_dense | FAILED | 10min | OOM during graph capture with 15 sizes |
| 3 | max_batch_tokens_8k | **SUCCESS** | 23min | **+3.6% throughput, +78% TTFT improvement** at c256 |
| 4 | moe_threshold_tune | marginal | 7min | +1.3% throughput at c32/c64, below 2% threshold |
| 5 | block_size_32 | no change | 7min | No meaningful improvement |

## Best Configurations by Workload

### Low Concurrency (c1-c8): Use baseline
No optimization significantly improves single-user or low-concurrency performance. TPOT 3.6ms is memory-bandwidth limited.

### Medium Concurrency (c32-c64): MoE threshold tuning
- `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=512`
- c32: 3,920 tok/s (+1.3%), TPOT 7.9ms
- c64: 6,141 tok/s (+1.3%), TPOT 10.1ms

### High Concurrency (c128-c256): max_num_batched_tokens=8192
- `--max-num-batched-tokens=8192`
- c256 1K/1K: **12,458 tok/s (+3.6%)**, TTFT 226.9ms (**-78.2% vs 1042ms baseline**)
- c256 8K/1K: 5,412 tok/s, TTFT 2515ms (+3.3% improvement)

---

## Pareto Frontier Comparison

### 1K/1K (ISL=1024, OSL=1024)

| Concurrency | Baseline Tput | Best Tput | Delta | Baseline TTFT | Best TTFT | Delta | Config |
|---|---|---|---|---|---|---|---|
| 1 | 272.8 | 272.8 | 0% | 40.1 | 40.1 | 0% | baseline |
| 32 | 3,868.4 | 3,920 | +1.3% | 104.4 | 65.1 | +37.6% | moe_tune |
| 64 | 6,059.7 | 6,141 | +1.3% | 99.2 | 94.8 | +4.5% | moe_tune |
| 128 | 8,979.9 | 8,979.9 | 0% | 136.2 | 136.2 | 0% | baseline |
| 256 | 12,022.6 | **12,458** | **+3.6%** | 1,042.4 | **226.9** | **+78.2%** | max_batch_8k |

### 8K/1K (ISL=8192, OSL=1024)

| Concurrency | Baseline Tput | Best Tput | Delta | Baseline TTFT | Best TTFT | Delta | Config |
|---|---|---|---|---|---|---|---|
| 1 | 263.1 | 263.1 | 0% | 119.7 | 119.7 | 0% | baseline |
| 64 | 3,873.6 | 3,920 | +1.2% | 451.6 | 479.0 | -6.1% | moe_tune |
| 128 | 4,723.5 | 4,748 | +0.5% | 805.5 | 1140.7 | -41.6% | gpu_util |
| 256 | 5,484.8 | 5,484.8 | 0% | 2,599.9 | **1,508** | **+42.0%** | combined |

### Pareto Frontier Shift
- **Max throughput: 12,023 -> 12,458 tok/s (+3.6%)**
- **TTFT at c256: 1,042 -> 227ms (78.2% improvement for 1K/1K)**
- **8K/1K c256 TTFT: 2,600 -> 1,508ms (42% improvement with combined config)**
- Min TPOT: 3.6ms (unchanged — memory-bandwidth limited)

---

## Key Insights

1. **TTFT is the main optimization target at high concurrency.** Throughput is already well-optimized, but TTFT at c256 was terrible (>1s). Reducing `max_num_batched_tokens` from 16384 to 8192 dramatically improved TTFT by allowing more frequent decode steps.

2. **gpu-memory-utilization 0.95 helps at c256** by providing more KV blocks, but the improvement is modest (+3.3%) because the model already fits comfortably in single-GPU memory.

3. **MoE threshold tuning (512 vs 1024) gives consistent small gains** at medium concurrency, suggesting the default threshold isn't optimal for GPT-OSS-120B's decode batch sizes.

4. **CUDAGraph density is limited by OOM.** Adding 5 extra capture sizes exceeds memory during graph capture. The default 10 sizes are well-balanced for single-GPU MI355X.

5. **Combined configs can conflict.** gpu_util_095 + max_batch_tokens_8k combined performed worse than either individually at c256 throughput, because the parameters interact non-linearly.

6. **No optimization improves low-concurrency TPOT.** The 3.6ms per-token latency at c1 is HBM bandwidth-limited, and no server-level tuning can improve it.

---

## Recommended Serving Configuration

```bash
# For high-concurrency serving (c64+):
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --server-port 8080
```

For medium concurrency workloads, also add:
```bash
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=512
```

---

## Reproduction Steps

```bash
# 1. Start container
docker start chuali_perf_opt

# 2. Deploy and run orchestrator
docker exec -d chuali_perf_opt bash -c \
  'cd /app && PYTHONPATH=/app/ATOM EXPERIMENT_STATE_DIR=/app/experiment_status \
   python3 -u /app/orchestrator.py > /app/orchestrator.log 2>&1'

# 3. Monitor progress
docker exec chuali_perf_opt cat /app/experiment_status/STATUS.md

# 4. Or use CLI tool:
python scripts/status.py --remote smci355-ccs-aus-m13-05.cs-aus.dcgpu --watch 30
```

## Files
- Orchestrator: `scripts/orchestrator.py`
- Tracker: `scripts/experiment_tracker.py`
- Notifier: `scripts/notifier.py`
- Status CLI: `scripts/status.py`
- All results: `/app/benchmark_results/` on container
- Status files: `/app/experiment_status/` on container
