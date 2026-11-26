# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark 3 — FP8 param all-gather reduces memory.

Exercises Lumen's ``FP8ParamManager`` feature:

  * **Quantize params**: Convert BF16 model weights → FP8 (E4M3).
  * **Dequant hooks**: Forward pre-hooks that dequantize before compute.
  * **Memory savings**: ~2x reduction in param storage and all-gather volume.
  * **Numerical quality**: Round-trip quant→dequant SNR.
  * **End-to-end**: ``LumenGatedMLP`` with FP8 params — latency & accuracy.
  * **Section 6 — Multi-GPU NCCL**: Real ``all_gather`` with FP8 vs BF16 shards,
    including the full quant → gather → dequant pipeline latency.
  * **Section 7 — Multi-GPU SDMA**: ``SdmaTpComm`` allgather with FP8 param
    shards on dedicated DMA engines.
  * **Section 8 — E2E distributed**: Full-model FP8 param allgather + forward
    correctness across ranks.
  * **Section 9 — Tail latency**: Per-rank p95/max tail latency profiling
    for FP8 all-gather across rank counts.
  * **Section 10 — Scaling efficiency**: Multi-GPU scaling comparison
    (2→4→8 GPU subgroups) for FP8 vs BF16 all-gather + GEMM.

Run single-GPU::

    python -m benchmarks.bench_fp8_param_allgather
    pytest benchmarks/bench_fp8_param_allgather.py -v -s

Run multi-GPU (2 GPU) — NCCL::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k NCCL

Run multi-GPU (2 GPU) — SDMA (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k SDMA

Run multi-GPU (2 GPU) — E2E::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E

Run multi-GPU (8 GPU) — all distributed tests::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k "NCCL or SDMA or E2E"

Run multi-GPU (8 GPU) — tail latency & scaling::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k "TailLatency or Scaling"

Run multi-GPU (8 GPU) — fixed-profile FP8 shape sweep (``-k ShapeSweep``).
Targets eight ranks for the same comparison mode as the other 8-GPU sections.
The sweep reuses the fixed six profiles from ``bench_e2e_fusion.py`` (via
``get_e2e_fusion_shape_sweep()``), not a separate shape list.
Rank 0 prints compact summaries with both average and tail metrics (e.g. p95)
for the single-layer and pipelined gather paths::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k ShapeSweep

Run explicit size experiments (2+ GPUs; 8 GPUs shown)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k backend_gap_experiment
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k pipeline_gain_experiment

Override the active profile or individual dimensions::

    # Profiles: default | backend_gap | pipeline_gain
    export LUMEN_E2E_PROFILE=backend_gap
    export LUMEN_E2E_BATCH=4
    export LUMEN_E2E_SEQ=2048
    export LUMEN_E2E_HIDDEN=4096
    export LUMEN_E2E_FFN=28672

Run multi-GPU (8 GPU) — individual sections::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k NCCL
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k SDMA
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s \
        -k "NCCL or E2E or TailLatency or Scaling or backend_gap_experiment or \
        pipeline_gain_experiment"
"""

from __future__ import annotations

import contextlib
import math
import os
import statistics

import pytest
import torch
import torch.distributed as dist
from atomrl.lumen.quantize.fp8_params import (
    _dequantize_gathered_fp8_chunk,
    _split_dim0_chunks,
    quantize_param_to_fp8,
)
from torch import nn

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    format_bytes,
    print_report,
    print_report_with_table,
    print_table,
    require_cuda,
)
from benchmarks.conftest import CUDA
from benchmarks.e2e_fusion_profiles import (
    E2EFusionProfile,
    format_e2e_fusion_profile,
    fp8_param_e2e_input_shape,
    fp8_param_e2e_weight_shape,
    fp8_param_pipeline_layer_shapes,
    get_e2e_fusion_profile,
    get_e2e_fusion_shape_sweep,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
HIDDEN = 4096
FFN_HIDDEN = 14336
N_LAYERS = 32
DEFAULT_PROFILE = get_e2e_fusion_profile()

# Timing parameters — overridable via LUMEN_BENCH_WARMUP / LUMEN_BENCH_ITERS
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0


def _make_profiled_fp8_e2e_inputs(
    device: torch.device,
    profile: E2EFusionProfile,
):
    shape = fp8_param_e2e_weight_shape(profile)
    weight_full = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    x = torch.randn(
        *fp8_param_e2e_input_shape(profile), device=device, dtype=torch.bfloat16
    )
    return shape, weight_full, x


def _make_profiled_fp8_pipeline_inputs(
    device: torch.device,
    profile: E2EFusionProfile,
    n_layers: int,
):
    layer_shapes = fp8_param_pipeline_layer_shapes(profile)
    weights = []
    xs = []
    for _ in range(n_layers):
        for shape in layer_shapes:
            weights.append(torch.randn(*shape, device=device, dtype=torch.bfloat16))
            xs.append(
                torch.randn(
                    *fp8_param_e2e_input_shape(profile, shape[1]),
                    device=device,
                    dtype=torch.bfloat16,
                )
            )
    return layer_shapes, weights, xs


def _profiled_fp8_title(prefix: str, profile: E2EFusionProfile, world: int) -> str:
    return f"{prefix} ({format_e2e_fusion_profile(profile)}, world={world})"


def _profile_with_num_chunks(
    profile: E2EFusionProfile, num_chunks: int
) -> E2EFusionProfile:
    return E2EFusionProfile(
        name=profile.name,
        batch=profile.batch,
        seq=profile.seq,
        hidden=profile.hidden,
        ffn=profile.ffn,
        num_chunks=num_chunks,
    )


def _gather_scales_once(
    scale: torch.Tensor, device: torch.device, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    scale_scalar = scale.float().reshape(1).to(device)
    all_scales = torch.empty(world_size, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(all_scales, scale_scalar, group=group)
    return all_scales


def _allgather_dim0_chunk(
    chunk: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    full_shape = list(chunk.shape)
    full_shape[0] *= world_size
    full_chunk = torch.empty(full_shape, dtype=chunk.dtype, device=chunk.device)
    dist.all_gather_into_tensor(full_chunk, chunk.contiguous(), group=group)
    return full_chunk


def _bf16_chunked_linear(
    x: torch.Tensor,
    bf16_shard: torch.Tensor,
    *,
    num_chunks: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    outputs = []
    for bf16_chunk in _split_dim0_chunks(bf16_shard, num_chunks):
        gathered_chunk = _allgather_dim0_chunk(bf16_chunk, group=group)
        outputs.append(torch.nn.functional.linear(x, gathered_chunk))
    return torch.cat(outputs, dim=-1)


def _fp8_chunked_linear_sequential(
    x: torch.Tensor,
    fp8_shard: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_chunks: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    per_rank_scales = _gather_scales_once(scale, fp8_shard.device, group=group)
    outputs = []
    for fp8_chunk in _split_dim0_chunks(fp8_shard, num_chunks):
        gathered_chunk = _allgather_dim0_chunk(fp8_chunk, group=group)
        weight_chunk = _dequantize_gathered_fp8_chunk(
            gathered_chunk, per_rank_scales, torch.bfloat16
        )
        outputs.append(torch.nn.functional.linear(x, weight_chunk))
    return torch.cat(outputs, dim=-1)


def _fp8_chunked_linear_pipelined(
    x: torch.Tensor,
    fp8_shard: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_chunks: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    fp8_chunks = _split_dim0_chunks(fp8_shard, num_chunks)
    per_rank_scales = _gather_scales_once(scale, fp8_shard.device, group=group)
    comm_stream = torch.cuda.Stream(device=fp8_shard.device)
    compute_stream = torch.cuda.current_stream(fp8_shard.device)
    # Keep chunk all-gathers on comm_stream ordered after the scale all-gather
    # already enqueued on the default stream.
    comm_stream.wait_stream(compute_stream)
    gathered_chunks = [None] * len(fp8_chunks)
    outputs = [None] * len(fp8_chunks)

    def _launch_gather(chunk_idx: int):
        gathered_chunks[chunk_idx] = _allgather_dim0_chunk(
            fp8_chunks[chunk_idx], group=group
        )

    with torch.cuda.stream(comm_stream):
        _launch_gather(0)

    for chunk_idx in range(len(fp8_chunks)):
        compute_stream.wait_stream(comm_stream)
        if chunk_idx + 1 < len(fp8_chunks):
            with torch.cuda.stream(comm_stream):
                _launch_gather(chunk_idx + 1)
        weight_chunk = _dequantize_gathered_fp8_chunk(
            gathered_chunks[chunk_idx], per_rank_scales, torch.bfloat16
        )
        outputs[chunk_idx] = torch.nn.functional.linear(x, weight_chunk)

    compute_stream.wait_stream(comm_stream)
    return torch.cat(outputs, dim=-1)


def _fp8_multi_weight_chunked_pipelined(
    xs: list[torch.Tensor],
    fp8_shards: list[torch.Tensor],
    fp8_scales: list[torch.Tensor],
    *,
    num_chunks: int,
    group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    chunk_lists = [
        _split_dim0_chunks(fp8_shard, num_chunks) for fp8_shard in fp8_shards
    ]
    per_rank_scales = [
        _gather_scales_once(scale, fp8_shards[0].device, group=group)
        for scale in fp8_scales
    ]
    chunk_tasks = [
        (weight_idx, chunk_idx)
        for weight_idx in range(len(fp8_shards))
        for chunk_idx in range(num_chunks)
    ]
    comm_stream = torch.cuda.Stream(device=fp8_shards[0].device)
    compute_stream = torch.cuda.current_stream(fp8_shards[0].device)
    # Keep chunk all-gathers on comm_stream ordered after the per-weight scale
    # all-gathers already enqueued on the default stream.
    comm_stream.wait_stream(compute_stream)
    gathered_chunks = [[None] * num_chunks for _ in fp8_shards]
    outputs = [[None] * num_chunks for _ in fp8_shards]

    def _launch_gather(weight_idx: int, chunk_idx: int):
        gathered_chunks[weight_idx][chunk_idx] = _allgather_dim0_chunk(
            chunk_lists[weight_idx][chunk_idx], group=group
        )

    with torch.cuda.stream(comm_stream):
        _launch_gather(*chunk_tasks[0])

    for task_idx, (weight_idx, chunk_idx) in enumerate(chunk_tasks):
        compute_stream.wait_stream(comm_stream)
        if task_idx + 1 < len(chunk_tasks):
            with torch.cuda.stream(comm_stream):
                _launch_gather(*chunk_tasks[task_idx + 1])
        weight_chunk = _dequantize_gathered_fp8_chunk(
            gathered_chunks[weight_idx][chunk_idx],
            per_rank_scales[weight_idx],
            torch.bfloat16,
        )
        outputs[weight_idx][chunk_idx] = torch.nn.functional.linear(
            xs[weight_idx], weight_chunk
        )

    compute_stream.wait_stream(comm_stream)
    return [torch.cat(weight_outputs, dim=-1) for weight_outputs in outputs]


def classify_fp8_tail_note(*, avg_ms: float, p95_ms: float) -> str:
    ratio = p95_ms / max(avg_ms, 1e-6)
    return "tail-heavy" if ratio >= 1.10 else "tail-stable"


def build_fp8_single_shape_row(
    *,
    profile_name: str,
    tokens: int,
    ffn: int,
    chunks: int = 1,
    bf16_avg_ms: float,
    bf16_p95_ms: float,
    bf16_max_ms: float,
    fp8_avg_ms: float,
    fp8_p95_ms: float,
    fp8_max_ms: float,
    fp8_speedup: float,
    p95_bf16_over_fp8: float,
    chunk_gain_ms: float = 0.0,
) -> dict[str, object]:
    """One sweep row.

    ``p95_bf16_over_fp8`` is BF16 p95 / FP8 p95
    (cross-path tail comparison, not an FP8-only tail metric).
    """
    return {
        "profile_name": profile_name,
        "tokens": tokens,
        "ffn": ffn,
        "chunks": chunks,
        "bf16_avg_ms": bf16_avg_ms,
        "bf16_p95_ms": bf16_p95_ms,
        "bf16_max_ms": bf16_max_ms,
        "fp8_avg_ms": fp8_avg_ms,
        "fp8_p95_ms": fp8_p95_ms,
        "fp8_max_ms": fp8_max_ms,
        "fp8_speedup": fp8_speedup,
        "p95_bf16_over_fp8": p95_bf16_over_fp8,
        "chunk_gain_ms": chunk_gain_ms,
    }


def build_fp8_pipeline_shape_row(
    *,
    profile_name: str,
    tokens: int,
    ffn: int,
    chunks: int = 1,
    bf16_avg_ms: float,
    bf16_p95_ms: float,
    fp8_seq_avg_ms: float,
    fp8_seq_p95_ms: float,
    fp8_pipe_avg_ms: float,
    fp8_pipe_p95_ms: float,
    fp8_pipe_max_ms: float,
    pipeline_speedup_vs_bf16: float,
    pipeline_speedup_vs_fp8_seq: float,
    p95_bf16_over_fp8_pipe: float,
    chunk_gain_ms: float = 0.0,
) -> dict[str, object]:
    """One sweep row.

    ``p95_bf16_over_fp8_pipe`` is BF16 p95 / FP8-pipelined p95, i.e. a
    cross-path tail ratio analogous to single-layer ``p95_bf16_over_fp8``.
    """
    return {
        "profile_name": profile_name,
        "tokens": tokens,
        "ffn": ffn,
        "chunks": chunks,
        "bf16_avg_ms": bf16_avg_ms,
        "bf16_p95_ms": bf16_p95_ms,
        "fp8_seq_avg_ms": fp8_seq_avg_ms,
        "fp8_seq_p95_ms": fp8_seq_p95_ms,
        "fp8_pipe_avg_ms": fp8_pipe_avg_ms,
        "fp8_pipe_p95_ms": fp8_pipe_p95_ms,
        "fp8_pipe_max_ms": fp8_pipe_max_ms,
        "pipeline_speedup_vs_bf16": pipeline_speedup_vs_bf16,
        "pipeline_speedup_vs_fp8_seq": pipeline_speedup_vs_fp8_seq,
        "p95_bf16_over_fp8_pipe": p95_bf16_over_fp8_pipe,
        "chunk_gain_ms": chunk_gain_ms,
        "note": classify_fp8_tail_note(avg_ms=fp8_pipe_avg_ms, p95_ms=fp8_pipe_p95_ms),
    }


def _bench_single_layer_gather_forward_for_profile(
    device: torch.device,
    rank: int,
    world: int,
    profile: E2EFusionProfile,
) -> tuple[tuple[int, ...], BenchResult, BenchResult]:
    """Single-layer gather+forward timing shared by E2E tests and the shape sweep (no printing)."""
    shape, weight_full, x = _make_profiled_fp8_e2e_inputs(device, profile)
    dist.broadcast(weight_full, src=0)
    num_chunks = profile.num_chunks

    fp8_full, scale = quantize_param_to_fp8(weight_full)
    fp8_shard = fp8_full.chunk(world, dim=0)[rank].contiguous()
    bf16_shard = weight_full.chunk(world, dim=0)[rank].contiguous()

    for _ in range(3):
        _bf16_chunked_linear(x, bf16_shard, num_chunks=num_chunks)
        _fp8_chunked_linear_sequential(x, fp8_shard, scale, num_chunks=num_chunks)
    torch.cuda.synchronize()

    def _bf16_pipeline():
        _bf16_chunked_linear(x, bf16_shard, num_chunks=num_chunks)

    def _fp8_pipeline():
        _fp8_chunked_linear_sequential(x, fp8_shard, scale, num_chunks=num_chunks)

    r_bf16 = cuda_timer(
        _bf16_pipeline,
        label="BF16 gather+GEMM",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
        dist_barrier=True,
    )
    r_fp8 = cuda_timer(
        _fp8_pipeline,
        label="FP8 gather+dequant+GEMM",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
        dist_barrier=True,
    )
    speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
    r_bf16.extra["chunks"] = num_chunks
    r_fp8.extra["speedup"] = round(speedup, 2)
    r_fp8.extra["chunks"] = num_chunks
    return shape, r_bf16, r_fp8


def _bench_pipelined_gather_forward_for_profile(
    device: torch.device,
    rank: int,
    world: int,
    profile: E2EFusionProfile,
) -> tuple[BenchResult, BenchResult, BenchResult]:
    """Multi-layer pipelined gather+forward timing shared by E2E tests and shape sweep (no printing)."""
    n_layers = 4
    _, weights, xs = _make_profiled_fp8_pipeline_inputs(device, profile, n_layers)
    num_chunks = profile.num_chunks

    bf16_shards = []
    fp8_shards = []
    fp8_scales = []

    for w in weights:
        dist.broadcast(w, src=0)
        fp8_w, sc = quantize_param_to_fp8(w)
        bf16_shards.append(w.chunk(world, dim=0)[rank].contiguous())
        fp8_shards.append(fp8_w.chunk(world, dim=0)[rank].contiguous())
        fp8_scales.append(sc)

    n_weights = len(bf16_shards)

    for i in range(n_weights):
        _bf16_chunked_linear(xs[i], bf16_shards[i], num_chunks=num_chunks)
        _fp8_chunked_linear_sequential(
            xs[i], fp8_shards[i], fp8_scales[i], num_chunks=num_chunks
        )
    torch.cuda.synchronize()

    def _bf16_sequential():
        for i in range(n_weights):
            _bf16_chunked_linear(xs[i], bf16_shards[i], num_chunks=num_chunks)

    def _fp8_sequential():
        for i in range(n_weights):
            _fp8_chunked_linear_sequential(
                xs[i], fp8_shards[i], fp8_scales[i], num_chunks=num_chunks
            )

    def _fp8_pipelined():
        _fp8_multi_weight_chunked_pipelined(
            xs,
            fp8_shards,
            fp8_scales,
            num_chunks=num_chunks,
        )

    r_bf16 = cuda_timer(
        _bf16_sequential,
        label=f"BF16 sequential {n_layers}L",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
        dist_barrier=True,
    )
    r_fp8_seq = cuda_timer(
        _fp8_sequential,
        label=f"FP8 sequential {n_layers}L",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
        dist_barrier=True,
    )
    r_fp8_pipe = cuda_timer(
        _fp8_pipelined,
        label=f"FP8 pipelined {n_layers}L",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
        dist_barrier=True,
    )
    r_bf16.extra["chunks"] = num_chunks
    r_fp8_seq.extra["chunks"] = num_chunks
    r_fp8_pipe.extra["chunks"] = num_chunks
    return r_bf16, r_fp8_seq, r_fp8_pipe


_SHAPE_SWEEP_EXPECTED_WORLD = 8


def _skip_shape_sweep_unless_8gpu(world: int) -> None:
    if world != _SHAPE_SWEEP_EXPECTED_WORLD:
        pytest.skip(
            "ShapeSweep is defined for the 8-GPU comparison mode "
            f"(world_size={world}); rerun with torchrun --nproc_per_node=8"
        )


def print_fp8_single_layer_shape_sweep_summary(
    rank: int, rows: list[dict[str, object]]
) -> None:
    """Emit a compact rank-0 summary for the single-layer shape sweep."""
    if rank != 0:
        return
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FP8 Param Shape Sweep — single-layer gather+forward")
    print(sep)
    if not rows:
        print("  (no summary rows yet — sweep timing collection pending)")
    else:
        for row in rows:
            print(
                f"  {row['profile_name']!s:<18}  "
                f"T={row['tokens']!s:<5} FFN={row['ffn']!s:<6} chunks={row.get('chunks', 1)!s:<2}  "
                f"BF16 ms avg/p95/max={row['bf16_avg_ms']:.3f}/{row['bf16_p95_ms']:.3f}/{row['bf16_max_ms']:.3f}  "
                f"FP8 ms avg/p95/max={row['fp8_avg_ms']:.3f}/{row['fp8_p95_ms']:.3f}/{row['fp8_max_ms']:.3f}  "
                f"speedup={row['fp8_speedup']:.2f}x  "
                f"p95_BF16/FP8={row['p95_bf16_over_fp8']:.2f}  "
                f"chunk_gain_ms={row.get('chunk_gain_ms', 0.0):.3f}  "
                f"(BF16 p95 / FP8 p95)"
            )
    print(sep)


def print_fp8_pipeline_shape_sweep_summary(
    rank: int, rows: list[dict[str, object]]
) -> None:
    """Emit a compact rank-0 summary for the pipelined shape sweep (BF16 vs FP8-seq vs FP8-pipe)."""
    if rank != 0:
        return
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FP8 Param Shape Sweep — pipelined gather+forward")
    print(sep)
    if not rows:
        print("  (no summary rows yet — sweep timing collection pending)")
    else:
        for row in rows:
            print(
                f"  {row['profile_name']!s:<18}  "
                f"T={row['tokens']!s:<5} FFN={row['ffn']!s:<6} chunks={row.get('chunks', 1)!s:<2}  "
                f"BF16 ms avg/p95={row['bf16_avg_ms']:.3f}/{row['bf16_p95_ms']:.3f}  "
                f"FP8-seq avg/p95={row['fp8_seq_avg_ms']:.3f}/{row['fp8_seq_p95_ms']:.3f}  "
                f"FP8-pipe avg/p95/max={row['fp8_pipe_avg_ms']:.3f}/"
                f"{row['fp8_pipe_p95_ms']:.3f}/{row['fp8_pipe_max_ms']:.3f}  "
                f"pipe_spd_BF16={row['pipeline_speedup_vs_bf16']:.2f}x  "
                f"pipe_spd_FP8seq={row['pipeline_speedup_vs_fp8_seq']:.2f}x  "
                f"p95_BF16/FP8pipe={row['p95_bf16_over_fp8_pipe']:.2f}  "
                f"chunk_gain_ms={row.get('chunk_gain_ms', 0.0):.3f}  "
                f"note={row['note']!s}"
            )
    print(sep)


def _build_mlp_stack(n_layers=4, hidden=HIDDEN, ffn=FFN_HIDDEN):
    """Build an nn.Linear stack mimicking transformer MLP layers.

    FP8ParamManager operates on nn.Linear (checks ``isinstance(module,
    nn.Linear)``), so we use standard Linear layers to model the
    gate/up/down projections of each transformer layer.
    """
    layers = []
    for _ in range(n_layers):
        layers.extend(
            [
                nn.Linear(hidden, ffn, bias=False),  # gate_proj
                nn.Linear(hidden, ffn, bias=False),  # up_proj
                nn.Linear(ffn, hidden, bias=False),  # down_proj
                nn.Linear(hidden, hidden, bias=False),  # qkv_proj
            ]
        )
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Multi-GPU infrastructure
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _suppress_dist_stderr():
    if os.environ.get("LUMEN_BENCH_VERBOSE_DIST") == "1":
        yield
        return

    try:
        saved_stderr = os.dup(2)
    except OSError:
        yield
        return

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    with _suppress_dist_stderr():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )


_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


def _sdma_available():
    try:
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


_SDMA = pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)


def _new_subgroup_for_size(group_size: int):
    """Create a subgroup in identical global order on every rank.

    ``torch.distributed.new_group`` must be invoked in the same sequence on
    all processes. Creating only the "local" subgroup on each rank can
    deadlock during process-group setup.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % group_size != 0:
        raise ValueError(f"world_size={world} not divisible by group_size={group_size}")

    subgroup = None
    for start in range(0, world, group_size):
        ranks = list(range(start, start + group_size))
        with _suppress_dist_stderr():
            group = dist.new_group(ranks)
        if rank in ranks:
            subgroup = group

    if subgroup is None:
        raise RuntimeError(
            f"failed to assign subgroup for rank {rank}, group_size={group_size}"
        )
    return subgroup


# ---------------------------------------------------------------------------
# 1. FP8ParamManager on nn.Linear stacks (matching Lumen shapes)
# ---------------------------------------------------------------------------


@CUDA
class TestFP8ParamManagerCompression:
    """FP8ParamManager applied to nn.Linear stacks with Lumen-scale shapes."""

    # Expected: ~2x memory reduction. BF16 uses 2 bytes/element; FP8 uses
    # 1 byte/element plus a small per-tensor scale (4 bytes). For large weight
    # matrices (e.g., 4096x14336), the scale overhead is negligible, yielding
    # close to 2x compression. This directly halves all-gather volume in TP.
    def test_single_layer_savings(self):
        """Quantize a single layer's linears to FP8."""
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = _build_mlp_stack(n_layers=1).to(device="cuda", dtype=torch.bfloat16)
        bf16_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

        mgr = FP8ParamManager()
        n_quant = mgr.quantize_params(model)

        fp8_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        scale_overhead = n_quant * 4
        total_fp8 = fp8_bytes + scale_overhead
        ratio = bf16_bytes / total_fp8

        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  FP8 Param Compression — 1 layer ({n_quant} tensors)")
        print(sep)
        print(f"  BF16 params:   {format_bytes(bf16_bytes):>10s}")
        print(f"  FP8  params:   {format_bytes(total_fp8):>10s}")
        print("  ──────────────────────────────")
        print(f"  Compression:   {ratio:.2f}x")
        print(f"  Saved:         {format_bytes(bf16_bytes - total_fp8):>10s}")
        print(sep)
        assert ratio > 1.9, f"Expected ~2x, got {ratio:.2f}x"

    # Expected: Same ~2x compression as single layer, confirming that
    # FP8ParamManager scales across the full model. The per-tensor scale
    # overhead becomes even more negligible with more parameters, so the
    # compression ratio should be at least 1.9x across 4 layers (16 linears).
    def test_multi_layer_savings(self):
        """Quantize a 4-layer stack to FP8."""
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = _build_mlp_stack(n_layers=4).to(device="cuda", dtype=torch.bfloat16)
        bf16_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

        mgr = FP8ParamManager()
        n_quant = mgr.quantize_params(model)
        fp8_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        saved = mgr.memory_savings_bytes(model)

        total_fp8 = fp8_bytes + n_quant * 4
        ratio = bf16_bytes / total_fp8
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  FP8 Param Compression — 4 layers ({n_quant} tensors)")
        print(sep)
        print(f"  BF16 params:   {format_bytes(bf16_bytes):>10s}")
        print(f"  FP8  params:   {format_bytes(total_fp8):>10s}")
        print("  ──────────────────────────────")
        print(f"  Compression:   {ratio:.2f}x")
        print(f"  Saved:         {format_bytes(saved):>10s}")
        print(sep)
        assert ratio > 1.9


# ---------------------------------------------------------------------------
# 2. Dequant hooks — forward latency with FP8 params
# ---------------------------------------------------------------------------


@CUDA
class TestFP8DequantHookLatency:
    """Forward latency of nn.Linear with FP8 param dequant hooks vs BF16."""

    # Expected: FP8 params with dequant hooks add small overhead (<10%) over
    # BF16 forward. The dequant hook runs a fast FP8→BF16 cast before each
    # nn.Linear forward. For large matrices, GEMM dominates and the cast is
    # nearly free. The benefit is 2x less memory, enabling larger models to
    # fit in GPU memory and halving all-gather communication in TP training.
    def test_bf16_vs_fp8_forward(self):
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        x = torch.randn(2, 2048, HIDDEN, device="cuda", dtype=torch.bfloat16)

        # BF16 baseline
        model_bf16 = nn.Sequential(
            nn.Linear(HIDDEN, FFN_HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(FFN_HIDDEN, HIDDEN, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        r_bf16 = cuda_timer(lambda: model_bf16(x), label="Linear forward (BF16 params)")

        # FP8 params with dequant hooks
        model_fp8 = nn.Sequential(
            nn.Linear(HIDDEN, FFN_HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(FFN_HIDDEN, HIDDEN, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        mgr = FP8ParamManager()
        mgr.quantize_params(model_fp8)
        mgr.register_dequant_hooks(model_fp8)

        r_fp8 = cuda_timer(
            lambda: model_fp8(x), label="Linear forward (FP8 params + dequant)"
        )

        overhead = (r_fp8.avg_ms - r_bf16.avg_ms) / max(r_bf16.avg_ms, 1e-6) * 100
        r_fp8.extra["overhead"] = f"{overhead:+.1f}%"
        print_report("FP8 Param Forward Latency", [r_bf16, r_fp8])


# ---------------------------------------------------------------------------
# 3. Quant/dequant primitive latency
# ---------------------------------------------------------------------------


@CUDA
class TestFP8QuantDequantLatency:
    """Latency of quantize_param_to_fp8 / dequantize_param_from_fp8."""

    # Expected: Quant and dequant are both memory-bound element-wise ops.
    # Quant (BF16→FP8) finds amax then divides+casts; dequant (FP8→BF16)
    # divides by scale then casts. Dequant should be slightly faster than
    # quant because it skips the absmax reduction pass. Larger shapes (FFN gate)
    # take proportionally longer due to more memory traffic.
    @pytest.mark.parametrize(
        "shape,name",
        [
            ((HIDDEN, HIDDEN), "QKV"),
            ((FFN_HIDDEN, HIDDEN), "FFN gate"),
            ((HIDDEN, FFN_HIDDEN), "FFN down"),
        ],
    )
    def test_quant_dequant(self, shape, name):
        from atomrl.lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        weight = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        r_q = cuda_timer(
            lambda: quantize_param_to_fp8(weight), label=f"quantize {name} {shape}"
        )
        fp8_w, scale = quantize_param_to_fp8(weight)
        r_d = cuda_timer(
            lambda: dequantize_param_from_fp8(fp8_w, scale),
            label=f"dequantize {name} {shape}",
        )
        print_report(f"FP8 quant/dequant: {name}", [r_q, r_d])


# ---------------------------------------------------------------------------
# 4. All-gather bandwidth simulation (per-layer)
# ---------------------------------------------------------------------------


@CUDA
class TestFP8AllgatherBandwidth:
    """Simulate all-gather volume for a full 7B model: BF16 vs FP8."""

    # Expected: ~2x bandwidth reduction. A 7B model has ~7 billion BF16
    # params = ~14 GB. FP8 params = ~7 GB + negligible scale overhead.
    # In TP all-gather, each GPU broadcasts its shard to all peers; FP8
    # halves the bytes on the interconnect, directly reducing communication
    # time which is often the TP bottleneck at scale.
    def test_7b_allgather_volume(self):
        per_layer_shapes = [
            ("gate", FFN_HIDDEN, HIDDEN),
            ("up", FFN_HIDDEN, HIDDEN),
            ("down", HIDDEN, FFN_HIDDEN),
            ("qkv", HIDDEN * 3, HIDDEN),
        ]

        total_bf16 = 0
        total_fp8 = 0
        for name, rows, cols in per_layer_shapes:
            numel = rows * cols
            total_bf16 += numel * 2 * N_LAYERS
            total_fp8 += (numel * 1 + 4) * N_LAYERS  # 1 byte data + 4 bytes scale

        ratio = total_bf16 / total_fp8
        saved = total_bf16 - total_fp8
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  All-Gather Volume — {N_LAYERS}-layer 7B model")
        print(sep)
        print(f"  BF16 volume:   {format_bytes(total_bf16):>10s}")
        print(f"  FP8  volume:   {format_bytes(total_fp8):>10s}")
        print("  ──────────────────────────────")
        print(f"  BW reduction:  {ratio:.2f}x")
        print(f"  Saved/step:    {format_bytes(saved):>10s}")
        print(sep)
        assert ratio > 1.9


# ---------------------------------------------------------------------------
# 5. Numerical quality — round-trip SNR
# ---------------------------------------------------------------------------


@CUDA
class TestFP8ParamSNR:
    """Verify numerical quality of FP8 param compression."""

    # Expected: SNR > 15 dB. FP8 E4M3 has 3 mantissa bits (vs BF16's 7),
    # giving ~23 dB theoretical SNR for uniform-distributed data. For
    # normally-distributed weights, per-tensor scaling preserves most dynamic
    # range. Values near zero suffer more quantization noise, but the overall
    # SNR for transformer weights typically exceeds 18-25 dB.
    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_roundtrip_snr(self, shape):
        from atomrl.lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        weight = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        fp8_w, scale = quantize_param_to_fp8(weight)
        recovered = dequantize_param_from_fp8(fp8_w, scale, torch.bfloat16)

        signal_power = (weight.float() ** 2).mean()
        noise_power = ((weight.float() - recovered.float()) ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / noise_power).item()

        print(f"\n  FP8 param round-trip SNR {shape}: {snr_db:.1f} dB")
        assert snr_db > 15, f"FP8 param SNR too low: {snr_db:.1f} dB"

    # Expected: Model output SNR > 10 dB. Quantization error accumulates
    # through multiple layers (Linear → SiLU → Linear), but per-tensor FP8
    # preserves enough precision for training convergence. Lower than per-param
    # SNR because errors compound through nonlinearities. This validates that
    # FP8 param storage is viable for training without significant accuracy loss.
    def test_linear_stack_output_snr(self):
        """Compare model output with BF16 vs FP8 params."""
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.SiLU(),
            nn.Linear(512, 256, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = model(x).clone()

        mgr = FP8ParamManager()
        mgr.quantize_params(model)
        mgr.register_dequant_hooks(model)

        with torch.no_grad():
            fp8_out = model(x)

        signal = (ref_out.float() ** 2).mean()
        noise = ((ref_out.float() - fp8_out.float()) ** 2).mean()
        snr_db = 10 * torch.log10(signal / noise).item()

        print(f"\n  Model output SNR (BF16 vs FP8 params): {snr_db:.1f} dB")
        assert snr_db > 10, f"Output SNR too low: {snr_db:.1f} dB"


# ---------------------------------------------------------------------------
# 5. Timed all-gather: BF16 vs FP8 param shards (M2 acceptance criterion)
# ---------------------------------------------------------------------------


def _is_dist_initialized():
    """Return True if torch.distributed is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
class TestFP8AllGatherDistributed:
    """Measure actual all_gather time for BF16 vs FP8 param shards.

    Directly addresses M2 acceptance: 'FP8 param all-gather reduces memory'
    and 'Compare memory footprint with/without FP8 param all-gather.'

    These tests require ``torchrun --nproc_per_node=N`` to be meaningful.
    On single-GPU, they fall back to a local simulation that measures the
    bandwidth difference of gathering larger (BF16) vs smaller (FP8) buffers.
    """

    # Expected: FP8 all-gather should take roughly half the time of BF16
    # all-gather because the data volume is halved (1 byte/element vs 2).
    # All-gather is bandwidth-bound on the interconnect, so halving the
    # payload directly halves the transfer time (minus fixed latency).
    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_allgather_bf16_vs_fp8(self, shape):
        from atomrl.lumen.quantize.fp8_params import quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight_bf16)

        if _is_dist_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            bf16_shard = weight_bf16.chunk(world_size)[rank].contiguous()
            fp8_shard = fp8_w.chunk(world_size)[rank].contiguous()

            bf16_gather_list = [torch.empty_like(bf16_shard) for _ in range(world_size)]
            fp8_gather_list = [torch.empty_like(fp8_shard) for _ in range(world_size)]

            r_bf16 = cuda_timer(
                lambda: torch.distributed.all_gather(bf16_gather_list, bf16_shard),
                label=f"all_gather BF16 {shape}",
            )
            r_fp8 = cuda_timer(
                lambda: torch.distributed.all_gather(fp8_gather_list, fp8_shard),
                label=f"all_gather FP8 {shape}",
            )
        else:
            tp_size = 8
            bf16_shard = weight_bf16[: shape[0] // tp_size].contiguous()
            fp8_shard = fp8_w[: shape[0] // tp_size].contiguous()

            bf16_full = torch.empty_like(weight_bf16)
            fp8_full = torch.empty(shape, device="cuda", dtype=fp8_w.dtype)

            r_bf16 = cuda_timer(
                lambda: torch.cat([bf16_shard] * tp_size, dim=0, out=bf16_full),
                label=f"simulated all_gather BF16 {shape}",
            )
            r_fp8 = cuda_timer(
                lambda: torch.cat([fp8_shard] * tp_size, dim=0, out=fp8_full),
                label=f"simulated all_gather FP8 {shape}",
            )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        bf16_bytes = bf16_shard.nelement() * bf16_shard.element_size()
        fp8_bytes = fp8_shard.nelement() * fp8_shard.element_size()
        bw_reduction = bf16_bytes / max(fp8_bytes, 1)

        r_fp8.extra["speedup"] = round(speedup, 2)
        r_fp8.extra["bandwidth_reduction"] = f"{bw_reduction:.1f}x"
        print_report(f"All-Gather BF16 vs FP8 {shape}", [r_bf16, r_fp8])
        print(f"  Bandwidth reduction: {bw_reduction:.1f}x")
        print(f"  Time speedup: {speedup:.2f}x")


# ---------------------------------------------------------------------------
# 6. Multi-GPU NCCL: real all-gather BF16 vs FP8 + full pipeline
# ---------------------------------------------------------------------------


@_DIST
class TestFP8AllGatherNCCL:
    """Real NCCL all-gather with FP8 vs BF16 param shards.

    Measures three key scenarios:
      1. Raw all-gather bandwidth: FP8 shard vs BF16 shard.
      2. Full pipeline: quantize → all-gather FP8 → dequantize vs
         plain BF16 all-gather (the realistic training hot path).
      3. Multi-layer: pipeline across N layers to amortise any fixed costs.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k NCCL
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_raw_allgather_bf16_vs_fp8(self, shape):
        """Raw NCCL all_gather_into_tensor: BF16 (2 B/elem) vs FP8 (1 B/elem)."""
        from atomrl.lumen.quantize.fp8_params import quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight_bf16)

        bf16_shard = weight_bf16.chunk(self.world)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world)[self.rank].contiguous()

        bf16_out = torch.empty_like(weight_bf16)
        fp8_out = torch.empty(shape, device=self.device, dtype=fp8_w.dtype)

        for _ in range(3):
            dist.all_gather_into_tensor(bf16_out, bf16_shard)
            dist.all_gather_into_tensor(fp8_out, fp8_shard)
        torch.cuda.synchronize()

        r_bf16 = cuda_timer(
            lambda: dist.all_gather_into_tensor(bf16_out, bf16_shard),
            label=f"NCCL all_gather BF16 {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8 = cuda_timer(
            lambda: dist.all_gather_into_tensor(fp8_out, fp8_shard),
            label=f"NCCL all_gather FP8  {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        bw_ratio = (bf16_shard.nelement() * bf16_shard.element_size()) / max(
            fp8_shard.nelement() * fp8_shard.element_size(), 1
        )
        r_fp8.extra["speedup"] = round(speedup, 2)
        r_fp8.extra["bw_reduction"] = f"{bw_ratio:.1f}x"

        if self.rank == 0:
            print_report_with_table(
                f"NCCL All-Gather BF16 vs FP8 {shape} (world={self.world})",
                [r_bf16, r_fp8],
            )

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_full_pipeline_quant_gather_dequant(self, shape):
        """Full pipeline: quant→gather→dequant (FP8) vs plain gather (BF16).

        Compares three paths:
          1. BF16 gather (baseline)
          2. FP8 sequential: gather → dequant
          3. FP8 overlapped: gather(current) on comm_stream while
             dequant(previous) runs on compute_stream — models the
             realistic multi-layer pipeline where layer N's gather
             overlaps with layer N-1's dequant.
        """
        num_chunks = get_e2e_fusion_profile().num_chunks
        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, scale = quantize_param_to_fp8(weight_bf16)
        x = torch.randn(2, 2048, shape[1], device=self.device, dtype=torch.bfloat16)

        bf16_shard = weight_bf16.chunk(self.world)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world)[self.rank].contiguous()

        for _ in range(3):
            _bf16_chunked_linear(x, bf16_shard, num_chunks=num_chunks)
            _fp8_chunked_linear_sequential(
                x,
                fp8_shard,
                scale,
                num_chunks=num_chunks,
            )
        torch.cuda.synchronize()

        def _bf16_pipeline():
            _bf16_chunked_linear(x, bf16_shard, num_chunks=num_chunks)

        def _fp8_sequential():
            _fp8_chunked_linear_sequential(x, fp8_shard, scale, num_chunks=num_chunks)

        def _fp8_overlapped():
            _fp8_chunked_linear_pipelined(x, fp8_shard, scale, num_chunks=num_chunks)

        r_bf16 = cuda_timer(
            _bf16_pipeline,
            label=f"BF16 gather {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_seq = cuda_timer(
            _fp8_sequential,
            label=f"FP8 sequential {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_ovl = cuda_timer(
            _fp8_overlapped,
            label=f"FP8 overlapped {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        seq_speedup = r_bf16.avg_ms / max(r_seq.avg_ms, 1e-6)
        ovl_speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_bf16.extra["chunks"] = num_chunks
        r_seq.extra["vs_bf16"] = round(seq_speedup, 2)
        r_seq.extra["chunks"] = num_chunks
        r_ovl.extra["speedup"] = round(ovl_speedup, 2)
        r_ovl.extra["chunks"] = num_chunks
        r_ovl.extra["chunk_gain_ms"] = round(r_seq.avg_ms - r_ovl.avg_ms, 3)

        if self.rank == 0:
            print_report_with_table(
                f"Full Pipeline BF16 vs FP8 {shape} (world={self.world}, chunks={num_chunks})",
                [r_bf16, r_seq, r_ovl],
            )

    def test_multi_layer_pipeline(self):
        """Pipeline across 4 transformer layers (gate+up+down+qkv per layer).

        Compares three paths:
          1. BF16 gather (baseline — all 16 weights gathered in BF16)
          2. FP8 sequential: gather all 16 in FP8, then dequant all 16
          3. FP8 pipelined: gather[i] on comm_stream overlapped with
             dequant[i-1] on compute_stream — hides dequant latency
             behind the next weight's gather
        """
        profile = get_e2e_fusion_profile()
        n_layers = 4
        r_bf16, r_seq, r_pipe = _bench_pipelined_gather_forward_for_profile(
            self.device,
            self.rank,
            self.world,
            profile,
        )
        layer_shapes = fp8_param_pipeline_layer_shapes(profile)
        total_bf16_bytes = (
            sum(
                rows * cols * torch.tensor([], dtype=torch.bfloat16).element_size()
                for rows, cols in layer_shapes
            )
            * n_layers
        )
        total_fp8_bytes = sum(rows * cols for rows, cols in layer_shapes) * n_layers
        seq_vs_bf16 = r_bf16.avg_ms / max(r_seq.avg_ms, 1e-6)
        pipe_speedup = r_seq.avg_ms / max(r_pipe.avg_ms, 1e-6)
        r_seq.extra["vs_bf16"] = round(seq_vs_bf16, 2)
        r_seq.extra["bf16_vol"] = format_bytes(total_bf16_bytes)
        r_seq.extra["fp8_vol"] = format_bytes(total_fp8_bytes)
        r_pipe.extra["speedup"] = round(pipe_speedup, 2)
        r_pipe.extra["vs_bf16"] = round(r_bf16.avg_ms / max(r_pipe.avg_ms, 1e-6), 2)
        r_pipe.extra["chunk_gain_ms"] = round(r_seq.avg_ms - r_pipe.avg_ms, 3)

        if self.rank == 0:
            print_report_with_table(
                f"Multi-Layer Pipeline ({n_layers}L, world={self.world}, chunks={profile.num_chunks})",
                [r_bf16, r_seq, r_pipe],
            )


# ---------------------------------------------------------------------------
# 7. Multi-GPU SDMA: FP8 param all-gather on dedicated DMA engines
# ---------------------------------------------------------------------------


@_SDMA
class TestFP8AllGatherSDMA:
    """SDMA all-gather with FP8 param shards.

    SDMA uses dedicated hardware DMA engines (separate from compute SMs),
    providing zero-contention allgather that can overlap perfectly with
    compute.  FP8 params halve the DMA transfer volume.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k SDMA
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_sdma_allgather_bf16_vs_fp8(self, shape):
        """SDMA allgather_dim0: BF16 shard vs FP8 shard."""
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm
        from atomrl.lumen.quantize.fp8_params import quantize_param_to_fp8

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight_bf16)

        bf16_shard = weight_bf16.chunk(self.world, dim=0)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world, dim=0)[self.rank].contiguous()

        comm = SdmaTpComm(dist.group.WORLD)

        for _ in range(3):
            comm.allgather_dim0(bf16_shard)
            comm.allgather_dim0(fp8_shard)
        torch.cuda.synchronize()

        r_bf16 = cuda_timer(
            lambda: comm.allgather_dim0(bf16_shard),
            label=f"SDMA allgather BF16 {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fp8 = cuda_timer(
            lambda: comm.allgather_dim0(fp8_shard),
            label=f"SDMA allgather FP8  {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        r_fp8.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"SDMA All-Gather BF16 vs FP8 {shape} (world={self.world})",
                [r_bf16, r_fp8],
            )

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_sdma_async_allgather_overlap_with_dequant(self, shape):
        """Overlap SDMA allgather (FP8 shard) with dequant of previous layer."""
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm
        from atomrl.lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        weight_bf16 = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, scale = quantize_param_to_fp8(weight_bf16)
        fp8_shard = fp8_w.chunk(self.world, dim=0)[self.rank].contiguous()

        comm = SdmaTpComm(dist.group.WORLD)
        sdma_stream = torch.cuda.Stream(device=self.device)
        compute_stream = torch.cuda.current_stream(self.device)

        for _ in range(3):
            comm.allgather_dim0(fp8_shard)
        torch.cuda.synchronize()

        r_gather = cuda_timer(
            lambda: comm.allgather_dim0(fp8_shard),
            label=f"SDMA gather FP8 {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        gathered_fp8 = comm.allgather_dim0(fp8_shard)
        r_dequant = cuda_timer(
            lambda: dequantize_param_from_fp8(gathered_fp8, scale, torch.bfloat16),
            label=f"dequant {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _sequential():
            g = comm.allgather_dim0(fp8_shard)
            dequantize_param_from_fp8(g, scale, torch.bfloat16)

        def _overlapped():
            comm.allgather_dim0_async(fp8_shard, stream=sdma_stream)
            dequantize_param_from_fp8(gathered_fp8, scale, torch.bfloat16)
            comm.wait_allgather_dim0(stream=sdma_stream)
            compute_stream.wait_stream(sdma_stream)

        r_seq = cuda_timer(
            _sequential,
            label=f"sequential {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_ovl = cuda_timer(
            _overlapped,
            label=f"overlapped {shape}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"SDMA Gather+Dequant Overlap {shape} (world={self.world})",
                [r_gather, r_dequant, r_seq, r_ovl],
            )


# ---------------------------------------------------------------------------
# 8. E2E distributed: full model FP8 param allgather + forward correctness
# ---------------------------------------------------------------------------


@_DIST
class TestFP8ParamE2EDistributed:
    """End-to-end: all ranks hold FP8 shards, allgather, dequant, forward.

    Verifies that the distributed FP8 param pipeline produces numerically
    correct results (SNR > 10 dB vs BF16 reference) and measures the
    total latency of the gather-dequant-forward pipeline.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k E2E
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_gather_dequant_forward_correctness(self):
        """Allgather FP8 shards → dequant → Linear forward, check SNR vs BF16."""
        from atomrl.lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        torch.manual_seed(42)
        H_in, H_out = 256, 512
        weight_full = torch.randn(H_out, H_in, device=self.device, dtype=torch.bfloat16)
        x = torch.randn(2, 32, H_in, device=self.device, dtype=torch.bfloat16)

        dist.broadcast(weight_full, src=0)
        dist.broadcast(x, src=0)

        ref_out = torch.nn.functional.linear(x, weight_full)

        fp8_full, scale = quantize_param_to_fp8(weight_full)
        fp8_shard = fp8_full.chunk(self.world, dim=0)[self.rank].contiguous()

        gathered_list = [torch.empty_like(fp8_shard) for _ in range(self.world)]
        dist.all_gather(gathered_list, fp8_shard)
        fp8_gathered = torch.cat(gathered_list, dim=0)

        weight_recovered = dequantize_param_from_fp8(
            fp8_gathered, scale, torch.bfloat16
        )
        fp8_out = torch.nn.functional.linear(x, weight_recovered)

        signal = (ref_out.float() ** 2).mean()
        noise = ((ref_out.float() - fp8_out.float()) ** 2).mean()
        snr_db = 10 * torch.log10(signal / noise).item()

        if self.rank == 0:
            print(f"\n  E2E distributed FP8 param forward SNR: {snr_db:.1f} dB")
        assert snr_db > 10, f"Distributed FP8 param SNR too low: {snr_db:.1f} dB"

    def _run_gather_dequant_forward_latency(
        self, profile: E2EFusionProfile = DEFAULT_PROFILE
    ):
        """Time the full pipeline: scatter shards → allgather FP8 → dequant → GEMM."""
        shape, r_bf16, r_fp8 = _bench_single_layer_gather_forward_for_profile(
            self.device, self.rank, self.world, profile
        )

        if self.rank == 0:
            print_report_with_table(
                _profiled_fp8_title(f"E2E Gather+Forward {shape}", profile, self.world),
                [r_bf16, r_fp8],
            )

    def _run_multi_layer_pipelined_gather_forward(
        self, profile: E2EFusionProfile = DEFAULT_PROFILE
    ):
        """Multi-layer pipeline: overlap allgather(layer i+1) with dequant+GEMM(layer i).

        The single-layer E2E test is sequential (gather→dequant→GEMM), so the
        dequant overhead is fully exposed and FP8 appears slower than BF16.
        In real training, layers are processed in sequence. This test pipelines
        them: while compute processes layer i (dequant+GEMM on compute stream),
        the comm stream pre-fetches layer i+1's FP8 shard via allgather.
        The dequant cost is hidden behind the overlapped communication.
        """
        n_layers = 4
        r_bf16, r_fp8_seq, r_fp8_pipe = _bench_pipelined_gather_forward_for_profile(
            self.device, self.rank, self.world, profile
        )

        sp_seq = r_bf16.avg_ms / max(r_fp8_seq.avg_ms, 1e-6)
        sp_pipe = r_bf16.avg_ms / max(r_fp8_pipe.avg_ms, 1e-6)
        pipe_vs_seq = r_fp8_seq.avg_ms / max(r_fp8_pipe.avg_ms, 1e-6)
        r_fp8_seq.extra["vs_bf16"] = round(sp_seq, 2)
        r_fp8_pipe.extra["vs_bf16"] = round(sp_pipe, 2)
        r_fp8_pipe.extra["vs_fp8_seq"] = round(pipe_vs_seq, 2)

        if self.rank == 0:
            print_report_with_table(
                _profiled_fp8_title(
                    f"E2E Pipelined Gather+Forward ({n_layers}L)", profile, self.world
                ),
                [r_bf16, r_fp8_seq, r_fp8_pipe],
            )
            saved_ms = r_fp8_seq.avg_ms - r_fp8_pipe.avg_ms
            print(
                f"  Pipeline saves {saved_ms:.3f} ms vs FP8 sequential "
                f"({pipe_vs_seq:.2f}x speedup)"
            )
            if sp_pipe >= 1.0:
                print(f"  FP8 pipelined beats BF16: {sp_pipe:.2f}x")
            else:
                gap = r_fp8_pipe.avg_ms - r_bf16.avg_ms
                print(
                    f"  FP8 pipelined still {gap:.3f} ms behind BF16 "
                    f"(dequant overhead not fully hidden)"
                )
            print()

    def test_gather_dequant_forward_latency(self):
        self._run_gather_dequant_forward_latency()

    def test_multi_layer_pipelined_gather_forward(self):
        self._run_multi_layer_pipelined_gather_forward()


@_DIST
class TestFP8ParamProfileExperiments(TestFP8ParamE2EDistributed):
    test_gather_dequant_forward_correctness = None
    test_gather_dequant_forward_latency = None
    test_multi_layer_pipelined_gather_forward = None

    def test_backend_gap_experiment_gather_dequant_forward_latency(self):
        self._run_gather_dequant_forward_latency(
            profile=get_e2e_fusion_profile("backend_gap")
        )

    def test_pipeline_gain_experiment_gather_dequant_forward_latency(self):
        self._run_gather_dequant_forward_latency(
            profile=get_e2e_fusion_profile("pipeline_gain")
        )

    def test_backend_gap_experiment_multi_layer_pipelined_gather_forward(self):
        self._run_multi_layer_pipelined_gather_forward(
            profile=get_e2e_fusion_profile("backend_gap")
        )

    def test_pipeline_gain_experiment_multi_layer_pipelined_gather_forward(self):
        self._run_multi_layer_pipelined_gather_forward(
            profile=get_e2e_fusion_profile("pipeline_gain")
        )


# ---------------------------------------------------------------------------
# 8b. Shape sweep — six fixed E2E fusion profiles (opt-in: ``-k ShapeSweep``)
# ---------------------------------------------------------------------------


@_DIST
class TestFP8ParamShapeSweep:
    """Sweep FP8 param gather paths across ``get_e2e_fusion_shape_sweep()`` profiles.

    Matches the 8-GPU-oriented workflow used for TailLatency / E2E sections.
    Select with ``pytest ... -k ShapeSweep`` (see module docstring).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_single_layer_shape_sweep(self):
        _skip_shape_sweep_unless_8gpu(self.world)
        rows: list[dict[str, object]] = []
        for profile in get_e2e_fusion_shape_sweep():
            assert profile.name
            _, r_bf16, r_fp8 = _bench_single_layer_gather_forward_for_profile(
                self.device, self.rank, self.world, profile
            )
            _, _, r_fp8_unchunked = _bench_single_layer_gather_forward_for_profile(
                self.device,
                self.rank,
                self.world,
                _profile_with_num_chunks(profile, 1),
            )
            fp8_speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
            # Cross-path p95 ratio (BF16 vs FP8), not an intrinsic FP8 tail statistic.
            p95_bf16_over_fp8 = r_bf16.p95_ms / max(r_fp8.p95_ms, 1e-6)
            rows.append(
                build_fp8_single_shape_row(
                    profile_name=profile.name,
                    tokens=profile.tokens,
                    ffn=profile.ffn,
                    chunks=profile.num_chunks,
                    bf16_avg_ms=r_bf16.avg_ms,
                    bf16_p95_ms=r_bf16.p95_ms,
                    bf16_max_ms=r_bf16.max_ms,
                    fp8_avg_ms=r_fp8.avg_ms,
                    fp8_p95_ms=r_fp8.p95_ms,
                    fp8_max_ms=r_fp8.max_ms,
                    fp8_speedup=fp8_speedup,
                    p95_bf16_over_fp8=p95_bf16_over_fp8,
                    chunk_gain_ms=r_fp8_unchunked.avg_ms - r_fp8.avg_ms,
                )
            )
        print_fp8_single_layer_shape_sweep_summary(self.rank, rows)

    def test_pipelined_shape_sweep(self):
        _skip_shape_sweep_unless_8gpu(self.world)
        rows: list[dict[str, object]] = []
        for profile in get_e2e_fusion_shape_sweep():
            assert profile.name
            r_bf16, r_fp8_seq, r_fp8_pipe = _bench_pipelined_gather_forward_for_profile(
                self.device, self.rank, self.world, profile
            )
            pipeline_speedup_vs_bf16 = r_bf16.avg_ms / max(r_fp8_pipe.avg_ms, 1e-6)
            pipeline_speedup_vs_fp8_seq = r_fp8_seq.avg_ms / max(
                r_fp8_pipe.avg_ms, 1e-6
            )
            p95_bf16_over_fp8_pipe = r_bf16.p95_ms / max(r_fp8_pipe.p95_ms, 1e-6)
            rows.append(
                build_fp8_pipeline_shape_row(
                    profile_name=profile.name,
                    tokens=profile.tokens,
                    ffn=profile.ffn,
                    chunks=profile.num_chunks,
                    bf16_avg_ms=r_bf16.avg_ms,
                    bf16_p95_ms=r_bf16.p95_ms,
                    fp8_seq_avg_ms=r_fp8_seq.avg_ms,
                    fp8_seq_p95_ms=r_fp8_seq.p95_ms,
                    fp8_pipe_avg_ms=r_fp8_pipe.avg_ms,
                    fp8_pipe_p95_ms=r_fp8_pipe.p95_ms,
                    fp8_pipe_max_ms=r_fp8_pipe.max_ms,
                    pipeline_speedup_vs_bf16=pipeline_speedup_vs_bf16,
                    pipeline_speedup_vs_fp8_seq=pipeline_speedup_vs_fp8_seq,
                    p95_bf16_over_fp8_pipe=p95_bf16_over_fp8_pipe,
                    chunk_gain_ms=r_fp8_seq.avg_ms - r_fp8_pipe.avg_ms,
                )
            )
        print_fp8_pipeline_shape_sweep_summary(self.rank, rows)


# ---------------------------------------------------------------------------
# 9. Tail latency across ranks: per-rank p95/max for FP8 all-gather
# ---------------------------------------------------------------------------


def _per_rank_latencies(fn, warmup=10, iters=50, dist_barrier=True):
    """Collect per-iteration latencies on this rank, return sorted list."""
    for _ in range(warmup):
        if dist_barrier:
            dist.barrier()
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if dist_barrier:
            dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


@_DIST
class TestFP8AllGatherTailLatency:
    """Per-rank tail latency (p95/p99/max) for FP8 vs BF16 all-gather.

    Reports per-rank statistics and cross-rank max (the true system tail).
    Requires 2+ GPUs.

    Run::

        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k TailLatency
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    @pytest.mark.parametrize("shape", [(HIDDEN, HIDDEN), (FFN_HIDDEN, HIDDEN)])
    def test_tail_latency_bf16_vs_fp8(self, shape):
        """Compare p95/p99/max latency per rank for BF16 vs FP8 all-gather."""
        from atomrl.lumen.quantize.fp8_params import quantize_param_to_fp8

        weight = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        fp8_w, _ = quantize_param_to_fp8(weight)

        bf16_shard = weight.chunk(self.world)[self.rank].contiguous()
        fp8_shard = fp8_w.chunk(self.world)[self.rank].contiguous()

        bf16_out = torch.empty_like(weight)
        fp8_out = torch.empty(shape, device=self.device, dtype=fp8_w.dtype)

        iters = 100

        bf16_times = _per_rank_latencies(
            lambda: dist.all_gather_into_tensor(bf16_out, bf16_shard),
            warmup=_WARMUP,
            iters=iters,
        )
        fp8_times = _per_rank_latencies(
            lambda: dist.all_gather_into_tensor(fp8_out, fp8_shard),
            warmup=_WARMUP,
            iters=iters,
        )

        def _stats(times):
            s = sorted(times)
            p95_idx = min(math.ceil(0.95 * len(s)) - 1, len(s) - 1)
            p99_idx = min(math.ceil(0.99 * len(s)) - 1, len(s) - 1)
            return {
                "avg": statistics.mean(times),
                "p95": s[p95_idx],
                "p99": s[p99_idx],
                "max": s[-1],
            }

        bf16_s = _stats(bf16_times)
        fp8_s = _stats(fp8_times)

        bf16_p95_t = torch.tensor(bf16_s["p95"], device=self.device)
        fp8_p95_t = torch.tensor(fp8_s["p95"], device=self.device)
        bf16_max_t = torch.tensor(bf16_s["max"], device=self.device)
        fp8_max_t = torch.tensor(fp8_s["max"], device=self.device)

        dist.all_reduce(bf16_p95_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(fp8_p95_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(bf16_max_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(fp8_max_t, op=dist.ReduceOp.MAX)

        if self.rank == 0:
            sep = "=" * 72
            print(f"\n{sep}")
            print(
                f"  Tail Latency — FP8 vs BF16 All-Gather {shape} (world={self.world})"
            )
            print(sep)
            print(
                f"  {'Metric':<25s}  {'BF16 (ms)':>12s}  {'FP8 (ms)':>12s}  {'Reduction':>10s}"
            )
            print(f"  {'─' * 25}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
            for label, bv, fv in [
                ("Rank-0 avg", bf16_s["avg"], fp8_s["avg"]),
                ("Rank-0 p95", bf16_s["p95"], fp8_s["p95"]),
                ("Rank-0 p99", bf16_s["p99"], fp8_s["p99"]),
                ("Rank-0 max", bf16_s["max"], fp8_s["max"]),
                ("Cross-rank p95 max", bf16_p95_t.item(), fp8_p95_t.item()),
                ("Cross-rank max", bf16_max_t.item(), fp8_max_t.item()),
            ]:
                ratio = bv / max(fv, 1e-6)
                print(f"  {label:<25s}  {bv:>12.3f}  {fv:>12.3f}  {ratio:>9.2f}x")
            print(sep)

    def test_tail_latency_full_pipeline(self):
        """Tail latency for the full quant→gather→dequant pipeline."""
        shape = (FFN_HIDDEN, HIDDEN)
        num_chunks = get_e2e_fusion_profile().num_chunks
        weight = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        shard = weight.chunk(self.world)[self.rank].contiguous()
        fp8_w, scale = quantize_param_to_fp8(weight)
        fp8_shard = fp8_w.chunk(self.world)[self.rank].contiguous()
        x = torch.randn(2, 2048, HIDDEN, device=self.device, dtype=torch.bfloat16)

        bf16_out = torch.empty_like(weight)
        bf16_shard = shard.clone()

        iters = 100

        bf16_times = _per_rank_latencies(
            lambda: dist.all_gather_into_tensor(bf16_out, bf16_shard),
            warmup=_WARMUP,
            iters=iters,
        )
        fp8_times = _per_rank_latencies(
            lambda: _fp8_chunked_linear_pipelined(
                x, fp8_shard, scale, num_chunks=num_chunks
            ),
            warmup=_WARMUP,
            iters=iters,
        )

        def _p95(times):
            s = sorted(times)
            return s[min(math.ceil(0.95 * len(s)) - 1, len(s) - 1)]

        bf16_p95 = torch.tensor(_p95(bf16_times), device=self.device)
        fp8_p95 = torch.tensor(_p95(fp8_times), device=self.device)
        dist.all_reduce(bf16_p95, op=dist.ReduceOp.MAX)
        dist.all_reduce(fp8_p95, op=dist.ReduceOp.MAX)

        if self.rank == 0:
            print(
                f"\n  Full Pipeline Tail — {shape} (world={self.world}, chunks={num_chunks})\n"
                f"  BF16 cross-rank p95: {bf16_p95.item():.3f} ms\n"
                f"  FP8  cross-rank p95: {fp8_p95.item():.3f} ms\n"
                f"  Tail reduction: {bf16_p95.item() / max(fp8_p95.item(), 1e-6):.2f}x"
            )


# ---------------------------------------------------------------------------
# 10. Scaling efficiency: compare FP8 vs BF16 across GPU counts
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 4,
    reason="Needs 4+ GPUs — run with torchrun --nproc_per_node=N (N>=4)",
)
class TestFP8AllGatherScalingEfficiency:
    """Multi-GPU scaling efficiency for FP8 vs BF16 all-gather + GEMM.

    Creates process sub-groups of size 2, 4, and (if available) 8 to
    measure how each mode scales. Reports per-group-size latency and
    scaling efficiency relative to the smallest group.

    Run::

        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fp8_param_allgather.py -v -s -k Scaling
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_scaling_allgather_gemm(self):
        """Sweep group sizes (2,4,8) for BF16 vs FP8 gather+GEMM."""
        num_chunks = get_e2e_fusion_profile().num_chunks
        shape = (FFN_HIDDEN, HIDDEN)
        weight_full = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        dist.broadcast(weight_full, src=0)
        x = torch.randn(2, 2048, HIDDEN, device=self.device, dtype=torch.bfloat16)

        group_sizes = [s for s in [2, 4, 8] if s <= self.world]
        bf16_results = {}
        fp8_results = {}

        for gs in group_sizes:
            subgroup = _new_subgroup_for_size(gs)

            local_rank_in_group = self.rank % gs

            fp8_full, scale = quantize_param_to_fp8(weight_full)
            bf16_shard = weight_full.chunk(gs, dim=0)[local_rank_in_group].contiguous()
            fp8_shard = fp8_full.chunk(gs, dim=0)[local_rank_in_group].contiguous()

            for _ in range(5):
                _bf16_chunked_linear(
                    x, bf16_shard, num_chunks=num_chunks, group=subgroup
                )
                _fp8_chunked_linear_sequential(
                    x, fp8_shard, scale, num_chunks=num_chunks, group=subgroup
                )
            torch.cuda.synchronize()

            def _bf16_fn(bf16_shard=bf16_shard, subgroup=subgroup):
                _bf16_chunked_linear(
                    x, bf16_shard, num_chunks=num_chunks, group=subgroup
                )

            def _fp8_fn(fp8_shard=fp8_shard, scale=scale, subgroup=subgroup):
                _fp8_chunked_linear_sequential(
                    x, fp8_shard, scale, num_chunks=num_chunks, group=subgroup
                )

            r_bf16 = cuda_timer(
                _bf16_fn,
                label=f"BF16 gather+GEMM (gs={gs})",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r_fp8 = cuda_timer(
                _fp8_fn,
                label=f"FP8 gather+dequant+GEMM (gs={gs})",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r_bf16.extra["chunks"] = num_chunks
            r_fp8.extra["chunks"] = num_chunks

            bf16_results[gs] = r_bf16
            fp8_results[gs] = r_fp8

            dist.destroy_process_group(subgroup)

        if self.rank == 0:
            baseline_gs = group_sizes[0]
            bf16_base = bf16_results[baseline_gs].avg_ms
            fp8_base = fp8_results[baseline_gs].avg_ms

            sep = "=" * 80
            print(f"\n{sep}")
            print(
                f"  Scaling Efficiency — FP8 vs BF16 Gather+GEMM {shape} (chunks={num_chunks})"
            )
            print(f"  Baseline group size: {baseline_gs}")
            print(sep)
            print(
                f"  {'GPUs':<6s}  "
                f"{'BF16 (ms)':>10s}  {'BF16 eff':>9s}  "
                f"{'FP8 (ms)':>10s}  {'FP8 eff':>9s}  "
                f"{'FP8 speedup':>11s}"
            )
            print(
                f"  {'─' * 6}  "
                f"{'─' * 10}  {'─' * 9}  "
                f"{'─' * 10}  {'─' * 9}  "
                f"{'─' * 11}"
            )

            for gs in group_sizes:
                bf16_ms = bf16_results[gs].avg_ms
                fp8_ms = fp8_results[gs].avg_ms
                bf16_eff = (bf16_base / max(bf16_ms, 1e-6)) * 100
                fp8_eff = (fp8_base / max(fp8_ms, 1e-6)) * 100
                speedup = bf16_ms / max(fp8_ms, 1e-6)

                print(
                    f"  {gs:<6d}  "
                    f"{bf16_ms:>10.3f}  {bf16_eff:>8.1f}%  "
                    f"{fp8_ms:>10.3f}  {fp8_eff:>8.1f}%  "
                    f"{speedup:>10.2f}x"
                )

            print(sep)
            print(
                "  Ideal scaling: latency stays flat as group size grows\n"
                "  (all-gather volume scales with group size, but shard\n"
                "  size shrinks proportionally — net traffic per rank stays\n"
                "  constant if bandwidth scales linearly with group size)"
            )
            print()

    def test_scaling_pipelined_fp8(self):
        """Sweep group sizes for pipelined FP8 gather+dequant+GEMM.

        Uses fp8_allgather_weight_pipelined to overlap communication
        with dequant across multiple layers.
        """
        num_chunks = get_e2e_fusion_profile().num_chunks
        layer_shapes = [
            (FFN_HIDDEN, HIDDEN),
            (FFN_HIDDEN, HIDDEN),
            (HIDDEN, FFN_HIDDEN),
            (HIDDEN, HIDDEN),
        ]

        weights = []
        for s in layer_shapes:
            w = torch.randn(*s, device=self.device, dtype=torch.bfloat16)
            dist.broadcast(w, src=0)
            weights.append(w)

        xs = [
            torch.randn(2, 2048, s[1], device=self.device, dtype=torch.bfloat16)
            for s in layer_shapes
        ]

        group_sizes = [s for s in [2, 4, 8] if s <= self.world]
        results = {}

        for gs in group_sizes:
            subgroup = _new_subgroup_for_size(gs)
            local_rank = self.rank % gs

            shards = [w.chunk(gs, dim=0)[local_rank].contiguous() for w in weights]
            fp8_shards = []
            fp8_scales = []
            for shard in shards:
                fp8_w, scale = quantize_param_to_fp8(shard)
                fp8_shards.append(fp8_w)
                fp8_scales.append(scale)

            for _ in range(3):
                _fp8_multi_weight_chunked_pipelined(
                    xs,
                    fp8_shards,
                    fp8_scales,
                    num_chunks=num_chunks,
                    group=subgroup,
                )
            torch.cuda.synchronize()

            def _pipeline_fn(
                fp8_scales=fp8_scales, fp8_shards=fp8_shards, subgroup=subgroup
            ):
                _fp8_multi_weight_chunked_pipelined(
                    xs,
                    fp8_shards,
                    fp8_scales,
                    num_chunks=num_chunks,
                    group=subgroup,
                )

            r = cuda_timer(
                _pipeline_fn,
                label=f"FP8 pipelined {len(layer_shapes)}L (gs={gs})",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r.extra["chunks"] = num_chunks
            results[gs] = r

            dist.destroy_process_group(subgroup)

        if self.rank == 0:
            baseline_gs = group_sizes[0]
            base_ms = results[baseline_gs].avg_ms

            print(
                f"\n  FP8 Pipelined Scaling — {len(layer_shapes)} layers (chunks={num_chunks})"
            )
            print(
                f"  {'GPUs':<6s}  {'Avg (ms)':>10s}  {'Scaling eff':>11s}  {'P95 (ms)':>10s}"
            )
            print(f"  {'─' * 6}  {'─' * 10}  {'─' * 11}  {'─' * 10}")
            for gs in group_sizes:
                r = results[gs]
                eff = (base_ms / max(r.avg_ms, 1e-6)) * 100
                print(
                    f"  {gs:<6d}  {r.avg_ms:>10.3f}  {eff:>10.1f}%  {r.p95_ms:>10.3f}"
                )
            print()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()

    from atomrl.lumen.quantize.fp8_params import (
        FP8ParamManager,
        dequantize_param_from_fp8,
        fp8_allgather_weight,
        quantize_param_to_fp8,
    )

    results: list[BenchResult] = []

    # Quant/dequant latency
    for name, shape in [("QKV", (HIDDEN * 3, HIDDEN)), ("FFN", (FFN_HIDDEN, HIDDEN))]:
        w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        r_q = cuda_timer(lambda w=w: quantize_param_to_fp8(w), label=f"quantize {name}")
        fp8_w, sc = quantize_param_to_fp8(w)
        r_d = cuda_timer(
            lambda fp8_w=fp8_w, sc=sc: dequantize_param_from_fp8(fp8_w, sc),
            label=f"dequantize {name}",
        )
        results.extend([r_q, r_d])

    # Linear stack with FP8 params
    model = _build_mlp_stack(n_layers=4).to(device="cuda", dtype=torch.bfloat16)
    bf16_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    mgr = FP8ParamManager()
    n_quant = mgr.quantize_params(model)
    fp8_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_fp8 = fp8_bytes + n_quant * 4
    ratio = bf16_bytes / total_fp8

    r_mem = BenchResult(name="4-layer BF16→FP8 compression", avg_ms=0)
    r_mem.extra["bf16"] = format_bytes(bf16_bytes)
    r_mem.extra["fp8"] = format_bytes(total_fp8)
    r_mem.extra["ratio"] = round(ratio, 2)
    results.append(r_mem)

    # Layer 2: FP8 all-gather (single-GPU simulation)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        shape = (FFN_HIDDEN, HIDDEN)
        w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        shard = w.chunk(dist.get_world_size())[dist.get_rank()].contiguous()
        num_chunks = get_e2e_fusion_profile().num_chunks
        r_l2 = cuda_timer(
            lambda: fp8_allgather_weight(shard, num_chunks=num_chunks),
            label="FP8 allgather Layer 2",
            dist_barrier=True,
        )
        r_l2.extra["chunks"] = num_chunks
        results.append(r_l2)

    print_report("Lumen FP8 Param All-Gather", results)
    print_table("Summary Table", results)
    print(
        f"  Compression: {ratio:.2f}x  ({n_quant} tensors, saved {format_bytes(bf16_bytes - total_fp8)})"
    )
    print()


if __name__ == "__main__":
    main()
