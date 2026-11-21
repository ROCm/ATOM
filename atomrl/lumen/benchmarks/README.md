# Lumen Benchmarks

Performance benchmarks validating Lumen's key feature optimizations.

All benchmarks use **Llama 3.1 8B** dimensions by default:
`hidden_size=4096`, `intermediate_size=14336`, `num_attention_heads=32`,
`num_key_value_heads=8`, `head_dim=128`.

The three E2E-style distributed benchmarks share a common profile helper:
`bench_e2e_fusion.py`, `bench_wgrad_delay.py`, and
`bench_fp8_param_allgather.py`.

- Base profile: `LUMEN_E2E_PROFILE=default|backend_gap|pipeline_gain`
- Dimension overrides: `LUMEN_E2E_BATCH`, `LUMEN_E2E_SEQ`,
  `LUMEN_E2E_HIDDEN`, `LUMEN_E2E_FFN`
- `LUMEN_E2E_NUM_CHUNKS` applies only to `bench_e2e_fusion.py`

Default selectors keep their original collection scope. If
`LUMEN_E2E_PROFILE` is set before launch, those selectors use that profile.
The explicit `backend_gap_experiment` and `pipeline_gain_experiment`
selectors pin those base profiles directly, while per-dimension env
overrides still apply.

## Benchmarks

| # | File | Lumen Features Exercised | Requirements |
|---|------|--------------------------|-------------|
| 1 | `bench_kernel_launch.py` | FP8 quantized_linear (7 scaling modes), fused MoE, FP8 activation store, attention backends (csrc/triton), norm+GEMM pipeline | 1 GPU + AITER |
| 2 | `bench_comm_overlap.py` | SdmaTpComm async allgather/reduce-scatter, column/row parallel overlap patterns, NCCL vs SDMA comparison, fixed-profile shape sweep | 2+ GPUs |
| 3 | `bench_fp8_param_allgather.py` | FP8ParamManager, dequant hooks, param compression SNR, **Layer 2 FP8 all-gather** (quant→gather→dequant), profiled E2E gather+forward / pipeline experiments, tail latency profiling, multi-GPU scaling efficiency | 1 GPU (sections 1-5), 2+ GPUs (sections 6-8), 4+ GPUs (sections 9-10) |
| 4 | `bench_rope_fusion.py` | apply_rotary_pos_emb (1D), fused_rope (Q+K), 2D vision RoPE, 3D video RoPE, GQA configs, NeoX vs GPT-J | 1 GPU + AITER |
| 5 | `bench_wgrad_delay.py` | **Tier 0** explicit `defer(weight, ...)` / `execute()` mechanism (single-GPU); **Tier 1** real-module overlap (`LumenColumnParallelLinear`, `backward_dw()`); **Tier 2** Megatron-style (`-k MegatronStyle`); `NCCLvsSdma`/profile selectors use the same real-module path (not legacy `defer(weight, ...)`); multi-layer `_DeferredWgrad` demo; `gradient_accumulation_fusion` | 1 GPU / 2+ GPUs for distributed overlap |
| 6 | `bench_fused_pipeline.py` | Pipelined AG+GEMM / GEMM+RS overlap, NCCL vs SDMA backend, backward overlap isolation, chunk sweep | 2+ GPUs |
| 7 | `bench_e2e_fusion.py` | Single-layer end-to-end **pure pipeline** transformer layer, fixed six-profile **shape sweep**, TP scaling sweep, explicit size/profile experiments, bandwidth reporting | 2+ GPUs (8 for TP scaling / shape sweep) |

## Quick Start

### Single GPU

```bash
# Run all single-GPU benchmarks
pytest benchmarks/ -v -s -k "not Distributed and not NCCL and not Sdma"

# Individual benchmarks
python -m benchmarks.bench_kernel_launch
python -m benchmarks.bench_rope_fusion
python -m benchmarks.bench_fp8_param_allgather
python -m benchmarks.bench_wgrad_delay
```

### Multi-GPU

```bash
# 2-GPU communication overlap
torchrun --nproc_per_node=2 -m benchmarks.bench_comm_overlap

# 8-GPU
torchrun --nproc_per_node=8 -m benchmarks.bench_comm_overlap

# Fixed shape sweep summary (NCCL always, SDMA when available)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_comm_overlap.py -v -s -k ShapeSweep
```

### Fused Pipeline

```bash
# Forward overlap
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd
# NCCL vs SDMA
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k SdmaColumn
# Backward isolation
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k BackwardOverlap
```

### FP8 Param All-Gather — E2E / Tail Latency / Scaling

```bash
# Default E2E distributed FP8 param tests
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E

# Explicit size experiments (separate from -k E2E)
# These currently cover latency + pipelined gather only, not correctness.
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k backend_gap_experiment
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k pipeline_gain_experiment

# Tail latency profiling (8 GPUs recommended)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k TailLatency

# Multi-GPU scaling efficiency (4+ GPUs)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k Scaling

# Fixed-profile FP8 shape sweep (8 GPUs; average + tail summaries on rank 0)
# Reuses the same fixed six profiles as bench_e2e_fusion.py (tokens × FFN sweep).
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k ShapeSweep
```

### Wgrad Delay — Distributed Overlap

`bench_wgrad_delay.py` is organized in three tiers:

- **Tier 0**: Low-level ``_DeferredWgrad`` mechanism checks and sanity.
- **Tier 1**: Real Lumen module APIs (e.g. ``LumenColumnParallelLinear``) and
  ``backward_dw()`` overlap benchmarks — not the legacy ``dwg.defer(weight, ...)`` API.
- **Tier 2**: Megatron-style stacks via ``TestMegatronStyleWgradDelay``; run under
  ``torchrun`` and select with ``-k MegatronStyle``, for example:

```bash
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k MegatronStyle
```

```bash
# Default NCCL vs SDMA comparison
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k NCCLvsSdma

# Other distributed overlap sections follow LUMEN_E2E_PROFILE too
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm

# Explicit size experiments on the single-layer NCCL vs SDMA comparison
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k backend_gap_experiment
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k pipeline_gain_experiment
```

### E2E Transformer Layer

These `torchrun` commands need a working `torch.distributed` environment: matching process/GPU count, suitable backends, and a rendezvous that your PyTorch build and OS support. If launch fails before tests run—e.g. `DistStoreError` or libuv/TCP-store messages during rendezvous—that is usually an environment or PyTorch runtime configuration issue, not evidence that the benchmark implementation is wrong.

```bash
# Default single-layer end-to-end pure pipeline benchmark
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TransformerLayer

# Explicit size experiments (separate from -k TransformerLayer)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k backend_gap_experiment
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k pipeline_gain_experiment

# TP scaling sweep (requires 8 GPUs)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TPScaling

# Shape sweep: see how tokens/FFN change comm overlap and memory behavior (8 GPUs)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k ShapeSweep
```

### Shared E2E Profile Overrides

```bash
export LUMEN_E2E_PROFILE=backend_gap
export LUMEN_E2E_BATCH=4
export LUMEN_E2E_SEQ=2048
export LUMEN_E2E_HIDDEN=4096
export LUMEN_E2E_FFN=28672
export LUMEN_E2E_NUM_CHUNKS=4  # bench_e2e_fusion.py only
```

The default selectors `TransformerLayer`, `RealComm`, `SdmaComm`,
`NCCLvsSdma`, and `E2E` keep their original scope. The explicit
`backend_gap_experiment` / `pipeline_gain_experiment` selectors live in
separate classes, so they are opt-in.

Use the other two files for the complementary cases:

```bash
# Isolated AG+GEMM / GEMM+RS pipeline micro-benchmarks
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd

# Delayed-wgrad / multi-layer overlap benchmarks
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k "Pipeline or RealComm or SdmaComm"
```

## Tracing

Generate Chrome/Perfetto-compatible `.json` trace files that visualize the GPU
timeline (kernel names, durations, stream concurrency).

```bash
# All trace groups
python -m benchmarks.run_traces

# Single group
python -m benchmarks.run_traces --only kernel   # kernel launch
python -m benchmarks.run_traces --only comm     # comm-compute overlap
python -m benchmarks.run_traces --only wgrad    # wgrad delay
python -m benchmarks.run_traces --only rope     # RoPE fusion
python -m benchmarks.run_traces --only fp8      # FP8 param

# Single trace function (or multiple)
python -m benchmarks.run_traces --run trace_fused_moe
python -m benchmarks.run_traces --run trace_fused_moe trace_fused_qk_rope

# List all available trace function names
python -m benchmarks.run_traces --list
```

Trace files are written to `benchmarks/traces/` (git-ignored). Open them at
`chrome://tracing` or https://ui.perfetto.dev.

| Group | Trace Files | What to Look For |
|-------|-------------|------------------|
| kernel | `kernel_fused_moe.json`, `kernel_manual_moe.json`, `kernel_scaling_*.json`, `kernel_attn_*.json` | Kernel count reduction, kernel names, fused vs decomposed |
| comm | `comm_column_*.json`, `comm_pipeline_*.json` | Stream concurrency: allgather on comm stream overlapping GEMM on compute stream |
| wgrad | `wgrad_sequential.json`, `wgrad_overlapped.json`, `wgrad_4layer_*.json` | Deferred wgrad on secondary stream overlapping forward on default stream |
| rope | `rope_fused_s2048.json`, `rope_pytorch_s2048.json`, `rope_fused_qk.json` | Single fused kernel vs 4+ decomposed PyTorch ops |
| fp8 | `fp8_quant.json`, `fp8_dequant.json`, `fp8_forward_*.json` | Quant/dequant kernel overhead, BF16 vs FP8 param forward pipeline |

## Feature Coverage

### 1. Single-GPU Kernel Launch Reduction (`bench_kernel_launch.py`)

| Feature | What's Measured |
|---------|----------------|
| `quantized_linear` scaling modes | Latency of all 7 modes: none, delayed, dynamic, per_token, blockwise, blockwise2d, mxfp8 |
| `fused_moe_triton` | End-to-end fused MoE vs individual `fused_topk` + per-expert GEMMs + `fused_unpermute` |
| `fp8_activation_store` | `LumenGatedMLP(fp8_activation_store=True)` vs `False`: fwd, bwd, and activation memory |
| Attention backends | `aiter_csrc` vs `aiter_triton`: causal, GQA (32Q/8KV), sliding window, seqlen sweep |
| Norm + GEMM pipeline | `rmsnorm` → `quantized_linear` with different FP8 scaling modes |

### 2. Multi-GPU Comm-Compute Overlap (`bench_comm_overlap.py`)

| Feature | What's Measured |
|---------|----------------|
| Column-parallel overlap | Allgather on comm stream, local-shard GEMM on compute stream |
| Row-parallel overlap | Reduce-scatter on comm stream, GEMM on compute stream |
| `SdmaTpComm` async | `allgather_dim0_async` / `reduce_scatter_dim0_async` with true SDMA hardware |
| NCCL vs SDMA | Direct comparison of overlap ratios |
| **Shape sweep** (`-k ShapeSweep`) | Fixed six-profile summary across `tokens x ffn`: rank-0 column/row tables for NCCL, plus SDMA when available |

Overlap ratio: `1 - (T_overlapped / (T_comm + T_compute))`

The comm-overlap sweep reuses the same fixed six representative profiles as
`bench_e2e_fusion.py`, so `comm_bound_small`, `default`, `backend_gap`, and the
other shape labels keep the same `tokens x ffn` meaning across the micro-
benchmark and the E2E pure-pipeline benchmark.

### 3. FP8 Param All-Gather (`bench_fp8_param_allgather.py`)

| Feature | What's Measured |
|---------|----------------|
| `FP8ParamManager` | Applied to `LumenGatedMLP`, `LumenFusedMLP`, multi-layer stacks |
| Dequant hooks | Forward latency with BF16 vs FP8 params + dequant overhead |
| Param compression | ~2x memory reduction, per-layer bandwidth savings |
| Round-trip SNR | Quantization quality (>15 dB param, >10 dB output) |
| **Layer 2 FP8 all-gather** | `fp8_allgather_weight`: quant→gather→per-shard-dequant pipeline via NCCL or SDMA |
| **Tail latency profiling** | Per-rank p95/p99/max latency, cross-rank worst-case tail for BF16 vs FP8 |
| **Scaling efficiency** | Group-size sweep (2→4→8 GPUs): BF16 vs FP8 gather+GEMM and pipelined FP8 |
| **Shape sweep** (`-k ShapeSweep`) | 8-GPU-oriented opt-in sweep over the same fixed six profiles as `bench_e2e_fusion.py`; rank-0 tables report average and tail metrics for single-layer and pipelined FP8 gather paths |

### 4. RoPE Fusion (`bench_rope_fusion.py`)

| Feature | What's Measured |
|---------|----------------|
| `apply_rotary_pos_emb` | 1D RoPE across S={128..8192}, NeoX vs GPT-J interleaved |
| `fused_rope` | Q+K fused vs separate `apply_rotary_pos_emb` x2 |
| `apply_rotary_pos_emb_2d` | Vision 2D RoPE (14x14, 16x16, 32x32) |
| `apply_rotary_pos_emb_3d` | Video 3D RoPE (4x8x8) |
| GQA configs | H_kv = {1, 4, 8, 32} with H_q = 32 |

### 5. Wgrad Delay (`bench_wgrad_delay.py`)

| Tier / area | What's measured |
|-------------|-----------------|
| **Tier 0** | `_DeferredWgrad` `defer`/`execute` vs eager; stream overlap demos; multi-layer `_DeferredWgrad` pipeline scheduling |
| **Tier 1** | Real `LumenColumnParallelLinear` + `backward_dw()` vs NCCL allreduce; multi-layer pipelined wgrad + comm |
| **Tier 2** (`-k MegatronStyle`) | Megatron-style column/row TP + sequence parallel; NCCL and SDMA realism paths |
| Distributed comparisons | Real-module + SDMA allreduce (`SdmaComm`); NCCL vs SDMA (`NCCLvsSdma`); E2E profile experiment selectors |
| `gradient_accumulation_fusion` | `w.main_grad.add_(dw)` vs `w.grad = dw` in `quantized_linear` backward |

### 6. Fused Pipeline Comm-GEMM Overlap (`bench_fused_pipeline.py`)

| Feature | What's Measured |
|---------|----------------|
| Column-parallel AG+GEMM | Fused vs sequential forward latency, overlap ratio |
| Row-parallel GEMM+RS | Fused vs sequential forward latency, overlap ratio |
| NCCL vs SDMA backend | Head-to-head fused pipeline latency comparison |
| Backward overlap isolation | dgrad+RS and AG+dgrad measured separately |
| Chunk sweep | Pipeline chunk count (1, 2, 4, 8) sensitivity |

### 7. End-to-End Transformer Layer Fusion (`bench_e2e_fusion.py`)

| Feature | What's Measured |
|---------|----------------|
| Single-layer fwd+bwd | Naive vs Fused NCCL vs Fused SDMA total latency for the **pure pipeline** path |
| Forward-only | Isolated forward with all fusion strategies |
| Backward-only | Isolated backward for the pure pipeline path (no delayed wgrad) |
| TP scaling | TP=2/4/8 latency, speedup, bandwidth at each scale |
| Bandwidth utilization | AG and RS effective bandwidth in GB/s |
| **Shape sweep** (`-k ShapeSweep`) | Fixed profiles across tokens/FFN: rank-0 tables for latency, effective comm bandwidth, and `peak_delta` working-set memory |

The sweep shows when fusion is communication-bound, when compute amortizes overlap overhead, and when the benefit is primarily `peak_delta` memory reduction rather than latency.

`bench_e2e_fusion.py` intentionally excludes delayed-wgrad scheduling so the
results reflect only the chunked comm-GEMM pipeline itself. For delayed-wgrad
benefits, use `bench_wgrad_delay.py`. For isolated AG/RS pipeline micro-
benchmarks and chunk sensitivity, use `bench_fused_pipeline.py`.
