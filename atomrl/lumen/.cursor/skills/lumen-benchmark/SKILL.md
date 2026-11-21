---
name: lumen-benchmark
description: "Write, run, and analyze Lumen performance benchmarks. Covers GPU timing, multi-GPU distributed setup, FP8 pipelines, comm-compute overlap, wgrad delay, and tracing. Use when creating benchmarks, diagnosing perf results, or adding new benchmark cases for Lumen features."
---

# Lumen Benchmark Guide

## Mission

Benchmarks exercise **actual Lumen implementations**, not standalone PyTorch. Every test must import and call a Lumen API — `_DeferredWgrad`, `FP8ParamManager`, `SdmaTpComm`, `quantized_linear`, `fused_moe_triton`, etc. If the test does not call Lumen code, it does not belong here.

## Project Layout

```
Lumen/benchmarks/
├── __init__.py
├── conftest.py              # markers (CUDA, AITER), _cleanup_dist fixture
├── bench_utils.py           # cuda_timer, track_cuda_memory, print_report, trace_fn
├── bench_kernel_launch.py   # single-GPU kernel-launch reduction
├── bench_rope_fusion.py     # RoPE fusion latency
├── bench_fp8_param_allgather.py  # FP8 param memory + pipelined gather
├── bench_comm_overlap.py    # multi-GPU comm-compute overlap (NCCL + SDMA)
├── bench_wgrad_delay.py     # deferred wgrad overlap
├── run_traces.py            # CLI for torch.profiler traces
├── README.md
└── traces/                  # git-ignored trace JSON output
```

## Model Dimensions

Use **Llama 3.1 8B** dimensions as the default:

```python
HIDDEN = 4096          # hidden_size
FFN_HIDDEN = 14336     # intermediate_size
NUM_HEADS = 32         # num_attention_heads
NUM_KV_HEADS = 8       # num_key_value_heads
HEAD_DIM = 128         # head_dim
```

## Timing — Use `cuda_timer`

Always time with CUDA events, never `time.time()`:

```python
from benchmarks.bench_utils import cuda_timer, print_report_with_table

result = cuda_timer(fn, warmup=10, iters=30, label="my op", trim_pct=10.0)
```

For multi-GPU tests, pass `dist_barrier=True` to align ranks before each iteration.

For overlap benchmarks, use `print_overlap_summary()` after `print_report()`.

## Distributed Setup — Critical Pattern

**Backend string**: Always `"cpu:gloo,cuda:nccl"` with `device_id`. Never `"nccl"` or `"nccl:gloo"`.

```python
def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
```

**Skip marker**: `_DIST = pytest.mark.skipif("RANK" not in os.environ, ...)`

**Cleanup**: Handled by `conftest.py`'s session-scoped `_cleanup_dist` fixture. Never call `dist.destroy_process_group()` manually in individual test files.

**SDMA tests**: Set `os.environ["MORI_ENABLE_SDMA"] = "1"` in the `_setup` fixture.

**Run commands**:

```bash
# Single GPU
pytest benchmarks/bench_foo.py -v -s

# Multi-GPU NCCL
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_foo.py -v -s -k RealComm

# Multi-GPU SDMA
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_foo.py -v -s -k SdmaComm
```

## Overlap Benchmark Pattern

Comm-compute overlap tests follow this structure:

1. **Measure compute alone** — e.g. wgrad GEMM
2. **Measure comm alone** — e.g. NCCL allreduce
3. **Measure sequential** — compute then comm
4. **Measure overlapped** — compute on default stream, comm on secondary stream
5. **Report** using `print_overlap_summary(t_compute=, t_comm=, t_seq=, t_ovl=)`

Key insight: When wgrad << allreduce, `overlap_ratio` is low but `hidden_ms` and `speedup` are the correct metrics.

## FP8 Pipeline Benchmark Pattern

Single-layer FP8 is often slower than BF16 (dequant overhead). Show the **multi-layer pipeline** where allgather(layer i+1) overlaps dequant+GEMM(layer i):

```python
# _fp8_pipelined:
dist.all_gather_into_tensor(fp8_outs[0], fp8_shards[0])
torch.cuda.synchronize()
for i in range(N):
    if i + 1 < N:
        with torch.cuda.stream(comm_stream):
            dist.all_gather_into_tensor(fp8_outs[i + 1], fp8_shards[i + 1])
    w = dequantize_param_from_fp8(fp8_outs[i], fp8_scales[i], torch.bfloat16)
    F.linear(x, w)
    if i + 1 < N:
        torch.cuda.current_stream().wait_stream(comm_stream)
```

## Tracing

Use `trace_fn()` or `trace_context()` from `bench_utils`:

```python
from benchmarks.bench_utils import trace_fn, trace_context, TRACE_DIR

# Single callable
trace_fn(fn, f"{TRACE_DIR}/my_trace.json", warmup=3, active=1)

# Multi-stream code
with trace_context(f"{TRACE_DIR}/overlap.json") as prof:
    # ... overlapped code ...
```

Open traces at `chrome://tracing` or https://ui.perfetto.dev.

## Test Comments — Expected Effects

Every test must have a docstring/comment explaining:
1. **What Lumen feature** is being tested
2. **Expected effect** (e.g. "FP8 allgather halves bandwidth")
3. **Why** (e.g. "1 byte vs 2 bytes per param element")

## Anti-Patterns

| Do NOT | Do Instead |
|--------|-----------|
| `backend="nccl"` or `"nccl:gloo"` | `backend="cpu:gloo,cuda:nccl"` with `device_id` |
| Manual `dist.destroy_process_group()` in test files | Rely on `conftest.py` `_cleanup_dist` |
| `time.time()` for GPU timing | `cuda_timer()` with CUDA events |
| Fabricated/simulated kernels | Call actual Lumen APIs |
| Forget `dist_barrier=True` in multi-GPU timing | Always pass `dist_barrier=True` |
| Missing `MORI_ENABLE_SDMA=1` for SDMA tests | Set in `_setup` fixture |
| Inline imports | All imports at top of file |

## Detailed Reference

For the full error catalog, conftest patterns, and per-benchmark details → see [reference.md](reference.md).
