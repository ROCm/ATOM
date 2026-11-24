# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark — End-to-end transformer layer pure-pipeline strategies.

Measures the latency of a single transformer layer (column-parallel +
row-parallel) under three schemes:

  * **Naive**: Sequential AG -> GEMM_up -> GEMM_down -> RS.
  * **Fused NCCL**: Pipelined comm-GEMM overlap with NCCL backend.
  * **Fused SDMA**: Pipelined comm-GEMM overlap with SDMA backend.

This file intentionally measures the **pure pipeline** path only:
``PipelinedAllgatherGemm`` / ``PipelinedGemmReduceScatter`` with eager
wgrad in backward.  It does **not** benchmark delayed wgrad execution in a
single-layer "backward then immediate execute()" schedule, because that mixes
two distinct optimizations and can make the results misleading.

For delayed-wgrad benchmarks and multi-layer pipeline schedules, use
``benchmarks/bench_wgrad_delay.py``. For isolated pipeline micro-benchmarks
and chunk sweeps, use ``benchmarks/bench_fused_pipeline.py``.

The built-in **shape sweep** compares fixed ``tokens`` / ``ffn`` combinations
(``hidden=4096``, ``num_chunks=4``). ``batch`` and ``seq`` matter only through
``tokens = batch * seq``; the benchmark sees flattened ``[tokens, hidden]``
activations.

Requires >= 2 GPUs with NCCL. TP scaling requires 8 GPUs.

Run shape sweep (8 GPUs recommended for the full NCCL vs SDMA comparison)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k ShapeSweep

Run E2E layer benchmarks::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TransformerLayer
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TransformerLayer

Run TP scaling sweep (requires 8 GPUs)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TPScaling

Run explicit size experiments (2+ GPUs; 8 GPUs shown)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k backend_gap_experiment
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k pipeline_gain_experiment

Override the active profile or individual dimensions::

    # Profiles: default | backend_gap | pipeline_gain
    export LUMEN_E2E_PROFILE=backend_gap
    export LUMEN_E2E_BATCH=4
    export LUMEN_E2E_SEQ=2048
    export LUMEN_E2E_HIDDEN=4096
    export LUMEN_E2E_FFN=28672
    export LUMEN_E2E_NUM_CHUNKS=4

Run all::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.bench_utils import (
    compute_bandwidth_gb_s,
    cuda_timer,
    print_bandwidth_summary,
    print_bench_warnings,
    print_report_with_table,
    track_cuda_memory,
)
from benchmarks.e2e_fusion_profiles import (
    E2EFusionProfile,
    format_e2e_fusion_profile,
    get_e2e_fusion_profile,
    get_e2e_fusion_shape_sweep,
)

DEFAULT_PROFILE = get_e2e_fusion_profile()
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0

_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


def peak_delta_mb(mem_info: dict[str, int]) -> float:
    """Convert ``peak_delta`` bytes to **mebibytes** (MiB), i.e. divide by 1024².

    Memory sweep columns use MiB for working-set deltas; comm volume uses decimal MB
    (see :func:`shape_comm_mb`).
    """
    return round(mem_info["peak_delta"] / (1024 * 1024), 1)


def shape_comm_mb(profile: E2EFusionProfile) -> float:
    """Logical AG+RS payload size in **decimal megabytes** (1e6 bytes), not MiB.

    Matches the shape-sweep ``total_comm_mb`` definition (bf16 activations, / 1e6).
    """
    ag_bytes = profile.tokens * profile.hidden * 2
    rs_bytes = profile.tokens * profile.hidden * 2
    return (ag_bytes + rs_bytes) / 1e6


def validate_tokens_divisible(*, tokens: int, world_size: int) -> None:
    if tokens % world_size != 0:
        raise ValueError(f"tokens={tokens} not divisible by world_size={world_size}")


def classify_backend_note(
    *,
    speedup: float | None,
    mem_savings_mb: float | None,
) -> str:
    if speedup is None or mem_savings_mb is None:
        return "n/a"
    if speedup >= 1.03:
        return "latency win"
    if speedup < 0.97:
        return "negative optimization"
    if mem_savings_mb >= 32.0:
        return "memory win only"
    return "neutral"


def _measure_peak_delta_mb(fn, *, device: torch.device) -> float:
    torch.cuda.empty_cache()
    with track_cuda_memory(device=device) as mem_info:
        fn()
    return peak_delta_mb(mem_info)


def build_shape_sweep_row(
    *,
    profile_name: str,
    tokens: int,
    ffn: int,
    naive_ms: float,
    nccl_ms: float,
    nccl_speedup: float,
    nccl_ag_bw: float,
    nccl_rs_bw: float,
    nccl_peak_delta_mb: float,
    nccl_mem_savings_mb: float,
    sdma_ms: float | None,
    sdma_speedup: float | None,
    sdma_ag_bw: float | None,
    sdma_rs_bw: float | None,
    sdma_peak_delta_mb: float | None,
    sdma_mem_savings_mb: float | None,
    naive_peak_delta_mb: float,
    total_comm_mb: float,
) -> dict[str, object]:
    nccl_note = classify_backend_note(
        speedup=nccl_speedup, mem_savings_mb=nccl_mem_savings_mb
    )
    if sdma_ms is None:
        sdma_note = "n/a"
    else:
        sdma_note = classify_backend_note(
            speedup=sdma_speedup, mem_savings_mb=sdma_mem_savings_mb
        )
    return {
        "profile_name": profile_name,
        "tokens": tokens,
        "ffn": ffn,
        "naive_ms": naive_ms,
        "nccl_ms": nccl_ms,
        "nccl_speedup": nccl_speedup,
        "nccl_ag_bw": nccl_ag_bw,
        "nccl_rs_bw": nccl_rs_bw,
        "nccl_peak_delta_mb": nccl_peak_delta_mb,
        "nccl_mem_savings_mb": nccl_mem_savings_mb,
        "sdma_ms": sdma_ms,
        "sdma_speedup": sdma_speedup,
        "sdma_ag_bw": sdma_ag_bw,
        "sdma_rs_bw": sdma_rs_bw,
        "sdma_peak_delta_mb": sdma_peak_delta_mb,
        "sdma_mem_savings_mb": sdma_mem_savings_mb,
        "naive_peak_delta_mb": naive_peak_delta_mb,
        "total_comm_mb": total_comm_mb,
        "nccl_note": nccl_note,
        "sdma_note": sdma_note,
        "comm_mb": total_comm_mb,
    }


def _print_shape_sweep_summary(measurements: list[dict[str, object]]) -> None:
    """Emit two compact rank-0 tables: latency/speedup/notes and eff. BW + peak_delta."""
    rows = [m["shape_sweep_row"] for m in measurements]

    print("\n" + "=" * 132)
    print("E2E Fusion Shape Sweep — performance (fwd+bwd)")
    print("=" * 132)
    print(
        f"{'Profile':<20} {'Tokens':>7} {'FFN':>7} {'Naive ms':>10} {'NCCL ms':>10} "
        f"{'NCCL sp':>9} {'NCCL note':<18} {'SDMA ms':>10} {'SDMA sp':>9} {'SDMA note':<18} {'Comm MB':>9}"
    )
    print("-" * 132)
    for r in rows:
        sdma_ms = r["sdma_ms"]
        if sdma_ms is None:
            sdma_ms_s = "n/a"
            sdma_sp_s = "n/a"
            sdma_note_s = "n/a"
        else:
            sdma_ms_s = f"{float(sdma_ms):.3f}"
            sdma_sp_s = f"{float(r['sdma_speedup']):.2f}x"
            sdma_note_s = str(r["sdma_note"])
        print(
            f"{r['profile_name']!s:<20} {int(r['tokens']):>7d} {int(r['ffn']):>7d} "
            f"{float(r['naive_ms']):>10.3f} {float(r['nccl_ms']):>10.3f} "
            f"{float(r['nccl_speedup']):>8.2f}x {r['nccl_note']!s:<18} "
            f"{sdma_ms_s:>10} {sdma_sp_s:>9} {sdma_note_s:<18} {float(r['comm_mb']):>9.1f}"
        )
    print("=" * 132)

    print("\n" + "=" * 128)
    print("E2E Fusion Shape Sweep — bandwidth + memory")
    print(
        "(AG/RS columns: effective step-normalized BW, GB/s; peak columns: track_cuda_memory peak_delta, MiB)"
    )
    print("=" * 128)
    print(
        f"{'Profile':<20} {'NCCL AG eff BW':>15} {'NCCL RS eff BW':>15} "
        f"{'SDMA AG eff BW':>15} {'SDMA RS eff BW':>15} "
        f"{'Naive peak ΔMB':>16} {'NCCL peak ΔMB':>16} {'SDMA peak ΔMB':>16}"
    )
    print("-" * 128)
    for r in rows:
        nccl_ag = float(r["nccl_ag_bw"])
        nccl_rs = float(r["nccl_rs_bw"])
        if r["sdma_ag_bw"] is None:
            sdma_ag_cell = "n/a"
            sdma_rs_cell = "n/a"
            sdma_peak_cell = "n/a"
        else:
            sdma_ag_cell = f"{float(r['sdma_ag_bw']):.1f}"
            sdma_rs_cell = f"{float(r['sdma_rs_bw']):.1f}"
            sdma_peak_cell = f"{float(r['sdma_peak_delta_mb']):.1f}"
        print(
            f"{r['profile_name']!s:<20} {nccl_ag:>15.1f} {nccl_rs:>15.1f} "
            f"{sdma_ag_cell:>15} {sdma_rs_cell:>15} "
            f"{float(r['naive_peak_delta_mb']):>16.1f} {float(r['nccl_peak_delta_mb']):>16.1f} {sdma_peak_cell:>16}"
        )
    print("=" * 128)


def _sdma_available():
    try:
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


_SDMA_DIST = pytest.mark.skipif(
    '"RANK" not in os.environ or not _sdma_available()',
    reason="Multi-GPU + mori SDMA required",
)


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )


class _AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        world = dist.get_world_size(group)
        ctx.world = world
        chunks = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(chunks, tensor.contiguous(), group=group)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        shard = grad_output.shape[0] // ctx.world
        out = torch.empty(
            shard,
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        dist.reduce_scatter_tensor(out, grad_output.contiguous(), group=ctx.group)
        return out, None


class _ReduceScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, world, device, group):
        ctx.group = group
        shard = tensor.shape[0] // world
        out = torch.empty(shard, *tensor.shape[1:], device=device, dtype=tensor.dtype)
        dist.reduce_scatter_tensor(out, tensor.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        world = dist.get_world_size(ctx.group)
        grad_output_contig = grad_output.contiguous()
        chunks = [torch.empty_like(grad_output_contig) for _ in range(world)]
        dist.all_gather(chunks, grad_output_contig, group=ctx.group)
        return torch.cat(chunks, dim=0), None, None, None


def _distributed_allgather(tensor, group=None):
    return _AllGatherFunc.apply(tensor, group)


def _reduce_scatter(tensor, world, device, group=None):
    return _ReduceScatterFunc.apply(tensor, world, device, group)


def _new_tp_process_group(tp_size: int):
    """Return this rank's tensor-parallel ``ProcessGroup`` of size *tp_size*.

    **All** global ranks must call this in the same order with the same
    *tp_size* values.  PyTorch requires :func:`torch.distributed.new_group` to
    be invoked in **identical sequence** on every process: each call wires one
    subgroup, and non-members still participate in that collective.  Calling
    ``new_group`` only with "my" ranks (so different ranks hit different
    ``new_group`` arguments at the same program point) deadlocks—exactly what
    the TP scaling benchmark used to do for ``tp_size`` in ``{2, 4}``.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % tp_size != 0:
        raise ValueError(f"world_size={world} not divisible by tp_size={tp_size}")
    tp_group = None
    for start in range(0, world, tp_size):
        ranks = list(range(start, start + tp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            tp_group = g
    if tp_group is None:
        raise RuntimeError(
            f"failed to assign TP subgroup for rank {rank}, tp_size={tp_size}"
        )
    return tp_group


class _E2ETransformerLayerFusionBase:
    """E2E single transformer layer: naive vs pure-pipeline NCCL/SDMA.

    This benchmark is intentionally single-layer and does not include delayed
    wgrad scheduling. Delayed wgrad is benchmarked separately in
    ``benchmarks/bench_wgrad_delay.py`` where its overlap can be measured in a
    realistic multi-layer or comm-overlap schedule.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{self.local_rank}")
        yield
        dist.barrier()

    def _make_layer(self, profile: E2EFusionProfile):
        w_up = torch.randn(
            profile.ffn // self.world,
            profile.hidden,
            device=self.device,
            dtype=torch.bfloat16,
        )
        w_down = torch.randn(
            profile.hidden,
            profile.ffn // self.world,
            device=self.device,
            dtype=torch.bfloat16,
        )
        return w_up, w_down

    def _naive_fwd(self, x_local, w_up, w_down):
        gathered = _distributed_allgather(x_local)
        up = F.linear(gathered, w_up)
        down = F.linear(up, w_down)
        return _reduce_scatter(down, self.world, self.device)

    def _make_pipelines(self, backend, profile: E2EFusionProfile):
        from atomrl.lumen.modules.comm_overlap import (
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
        )

        ag = PipelinedAllgatherGemm(num_chunks=profile.num_chunks, comm=backend)
        rs = PipelinedGemmReduceScatter(num_chunks=profile.num_chunks, comm=backend)
        return ag, rs

    def _fused_fwd(self, x_local, w_up, w_down, ag, rs, w_up_param, w_down_param):
        from atomrl.lumen.modules.comm_overlap import (
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )

        up = fused_column_parallel_forward(
            x_local,
            w_up,
            None,
            ag,
            rs,
            w_up_param,
            False,
            False,
            None,
            False,
        )
        return fused_row_parallel_forward(
            up,
            w_down,
            rs,
            ag,
            w_down_param,
            False,
            False,
            None,
        )

    def _measure_comparison(
        self,
        fwd_only=False,
        bwd_only=False,
        profile: E2EFusionProfile = DEFAULT_PROFILE,
    ) -> dict[str, object]:
        from atomrl.lumen.modules.comm_overlap import NcclCommBackend

        validate_tokens_divisible(tokens=profile.tokens, world_size=self.world)
        S_local = profile.tokens // self.world
        w_up, w_down = self._make_layer(profile)
        w_up_param = torch.nn.Parameter(w_up.clone())
        w_down_param = torch.nn.Parameter(w_down.clone())
        nccl = NcclCommBackend(dist.group.WORLD)
        ag, rs = self._make_pipelines(nccl, profile)

        ag_bytes = profile.tokens * profile.hidden * 2
        rs_bytes = profile.tokens * profile.hidden * 2
        needs_grad = not fwd_only

        if bwd_only:
            x_naive = torch.randn(
                S_local,
                profile.hidden,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            y_naive_saved = self._naive_fwd(x_naive, w_up_param, w_down_param)

            x_fused = torch.randn(
                S_local,
                profile.hidden,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            y_fused_saved = self._fused_fwd(
                x_fused,
                w_up,
                w_down,
                ag,
                rs,
                w_up_param,
                w_down_param,
            )

            def _naive():
                x_naive.grad = None
                w_up_param.grad = None
                w_down_param.grad = None
                y_naive_saved.sum().backward(retain_graph=True)

            def _fused_nccl():
                x_fused.grad = None
                w_up_param.grad = None
                w_down_param.grad = None
                y_fused_saved.sum().backward(retain_graph=True)

        else:

            def _naive():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(
                    S_local,
                    profile.hidden,
                    device=self.device,
                    dtype=torch.bfloat16,
                    requires_grad=needs_grad,
                )
                y = self._naive_fwd(x, w_up_param, w_down_param)
                if needs_grad:
                    y.sum().backward()
                return y

            def _fused_nccl():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(
                    S_local,
                    profile.hidden,
                    device=self.device,
                    dtype=torch.bfloat16,
                    requires_grad=needs_grad,
                )
                y = self._fused_fwd(x, w_up, w_down, ag, rs, w_up_param, w_down_param)
                if needs_grad:
                    y.sum().backward()
                return y

        for _ in range(3):
            _naive()
        torch.cuda.synchronize()
        for _ in range(3):
            _fused_nccl()
        torch.cuda.synchronize()

        label_suffix = "fwd" if fwd_only else ("bwd-only" if bwd_only else "fwd+bwd")

        r_naive = cuda_timer(
            _naive,
            label=f"naive {label_suffix}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_nccl,
            label=f"fused NCCL {label_suffix}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)

        results = [r_naive, r_fused]
        r_sdma = None

        if _sdma_available() and not bwd_only:
            from atomrl.lumen.modules.comm_overlap import SdmaCommBackend
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            os.environ["MORI_ENABLE_SDMA"] = "1"
            sdma_comm = SdmaTpComm(dist.group.WORLD)
            sdma = SdmaCommBackend(sdma_comm)
            ag_s, rs_s = self._make_pipelines(sdma, profile)

            def _fused_sdma():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(
                    S_local,
                    profile.hidden,
                    device=self.device,
                    dtype=torch.bfloat16,
                    requires_grad=needs_grad,
                )
                y = self._fused_fwd(
                    x, w_up, w_down, ag_s, rs_s, w_up_param, w_down_param
                )
                if needs_grad:
                    y.sum().backward()
                return y

            # Keep the backend transition explicit: some ranks can exit the
            # NCCL timer slightly earlier than others, so fence before starting
            # mori SDMA warmup to avoid overlapping the two transports.
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(3):
                _fused_sdma()
            torch.cuda.synchronize()
            dist.barrier()

            r_sdma = cuda_timer(
                _fused_sdma,
                label=f"fused SDMA {label_suffix}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            sdma_speedup = r_naive.avg_ms / max(r_sdma.avg_ms, 1e-6)
            r_sdma.extra["speedup"] = round(sdma_speedup, 2)
            results.append(r_sdma)

        torch.cuda.synchronize()
        dist.barrier()

        for _ in range(3):
            _naive()
        torch.cuda.synchronize()
        naive_peak_delta_mb = _measure_peak_delta_mb(_naive, device=self.device)
        dist.barrier()

        for _ in range(3):
            _fused_nccl()
        torch.cuda.synchronize()
        nccl_peak_delta_mb = _measure_peak_delta_mb(_fused_nccl, device=self.device)
        dist.barrier()

        nccl_mem_savings_mb = round(naive_peak_delta_mb - nccl_peak_delta_mb, 1)

        sdma_ms = None
        sdma_speedup_val = None
        sdma_peak_delta_mb = None
        sdma_mem_savings_mb = None

        if r_sdma is not None:
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(3):
                _fused_sdma()
            torch.cuda.synchronize()
            sdma_peak_delta_mb = _measure_peak_delta_mb(_fused_sdma, device=self.device)
            dist.barrier()
            sdma_ms = r_sdma.avg_ms
            sdma_speedup_val = r_naive.avg_ms / max(r_sdma.avg_ms, 1e-6)
            sdma_mem_savings_mb = round(naive_peak_delta_mb - sdma_peak_delta_mb, 1)

        total_comm_mb = shape_comm_mb(profile)
        ag_bw_nccl = compute_bandwidth_gb_s(ag_bytes, r_fused.avg_ms)
        rs_bw_nccl = compute_bandwidth_gb_s(rs_bytes, r_fused.avg_ms)
        ag_bw_sdma = (
            compute_bandwidth_gb_s(ag_bytes, r_sdma.avg_ms)
            if r_sdma is not None
            else None
        )
        rs_bw_sdma = (
            compute_bandwidth_gb_s(rs_bytes, r_sdma.avg_ms)
            if r_sdma is not None
            else None
        )

        shape_sweep_row = build_shape_sweep_row(
            profile_name=profile.name,
            tokens=profile.tokens,
            ffn=profile.ffn,
            naive_ms=r_naive.avg_ms,
            nccl_ms=r_fused.avg_ms,
            nccl_speedup=speedup,
            nccl_ag_bw=ag_bw_nccl,
            nccl_rs_bw=rs_bw_nccl,
            nccl_peak_delta_mb=nccl_peak_delta_mb,
            nccl_mem_savings_mb=nccl_mem_savings_mb,
            sdma_ms=sdma_ms,
            sdma_speedup=sdma_speedup_val,
            sdma_ag_bw=ag_bw_sdma,
            sdma_rs_bw=rs_bw_sdma,
            sdma_peak_delta_mb=sdma_peak_delta_mb,
            sdma_mem_savings_mb=sdma_mem_savings_mb,
            naive_peak_delta_mb=naive_peak_delta_mb,
            total_comm_mb=total_comm_mb,
        )

        return {
            "profile": profile,
            "label_suffix": label_suffix,
            "rank": self.rank,
            "world": self.world,
            "results": results,
            "ag_bytes": ag_bytes,
            "rs_bytes": rs_bytes,
            "naive_ms": r_naive.avg_ms,
            "nccl_ms": r_fused.avg_ms,
            "nccl_speedup": speedup,
            "ag_bw_nccl_gb_s": ag_bw_nccl,
            "rs_bw_nccl_gb_s": rs_bw_nccl,
            "sdma_ms": sdma_ms,
            "sdma_speedup": sdma_speedup_val,
            "ag_bw_sdma_gb_s": ag_bw_sdma,
            "rs_bw_sdma_gb_s": rs_bw_sdma,
            "total_comm_mb": total_comm_mb,
            "peak_delta_naive_mb": naive_peak_delta_mb,
            "peak_delta_nccl_mb": nccl_peak_delta_mb,
            "peak_delta_sdma_mb": sdma_peak_delta_mb,
            "nccl_mem_savings_mb_vs_naive": nccl_mem_savings_mb,
            "sdma_mem_savings_mb_vs_naive": sdma_mem_savings_mb,
            "shape_sweep_row": shape_sweep_row,
        }

    def _print_detailed_comparison(self, measurement: dict[str, object]) -> None:
        if self.rank != 0:
            return
        profile = measurement["profile"]
        label_suffix = measurement["label_suffix"]
        results = measurement["results"]
        ag_bytes = measurement["ag_bytes"]
        rs_bytes = measurement["rs_bytes"]
        r_fused = results[1]
        speedup = measurement["nccl_speedup"]
        print_report_with_table(
            (
                "E2E Transformer Layer Pure Pipeline "
                f"{label_suffix} ({format_e2e_fusion_profile(profile)}, world={self.world})"
            ),
            results,
        )
        print_bandwidth_summary(
            label="AG", bytes_transferred=ag_bytes, time_ms=r_fused.avg_ms
        )
        print_bandwidth_summary(
            label="RS", bytes_transferred=rs_bytes, time_ms=r_fused.avg_ms
        )
        print_bench_warnings(result=r_fused, speedup=speedup)

    def _run_comparison(
        self,
        fwd_only=False,
        bwd_only=False,
        profile: E2EFusionProfile = DEFAULT_PROFILE,
    ):
        m = self._measure_comparison(
            fwd_only=fwd_only, bwd_only=bwd_only, profile=profile
        )
        self._print_detailed_comparison(m)


@_DIST
class TestE2ETransformerLayerFusion(_E2ETransformerLayerFusionBase):
    def test_single_layer_fwd_bwd(self):
        self._run_comparison(fwd_only=False, bwd_only=False)

    def test_single_layer_fwd_only(self):
        self._run_comparison(fwd_only=True, bwd_only=False)

    def test_single_layer_bwd_only(self):
        self._run_comparison(fwd_only=False, bwd_only=True)


@_DIST
class TestE2EFusionExperiments(_E2ETransformerLayerFusionBase):
    def test_backend_gap_experiment_fwd_bwd(self):
        self._run_comparison(
            fwd_only=False,
            bwd_only=False,
            profile=get_e2e_fusion_profile("backend_gap"),
        )

    def test_pipeline_gain_experiment_fwd_bwd(self):
        self._run_comparison(
            fwd_only=False,
            bwd_only=False,
            profile=get_e2e_fusion_profile("pipeline_gain"),
        )


@_DIST
class TestE2EShapeSweep(_E2ETransformerLayerFusionBase):
    def test_shape_sweep_fwd_bwd(self):
        rows: list[dict[str, object]] = []
        for profile in get_e2e_fusion_shape_sweep():
            rows.append(
                self._measure_comparison(
                    fwd_only=False, bwd_only=False, profile=profile
                )
            )
            dist.barrier()

        if self.rank == 0:
            _print_shape_sweep_summary(rows)


@_SDMA_DIST
class TestE2ETransformerLayerSdmaOnly:
    """Isolated SDMA pure-pipeline measurement — no NCCL comparison overhead."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{self.local_rank}")
        yield
        dist.barrier()

    def _run_sdma(
        self,
        fwd_only=False,
        bwd_only=False,
        profile: E2EFusionProfile = DEFAULT_PROFILE,
    ):
        from atomrl.lumen.modules.comm_overlap import (
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            SdmaCommBackend,
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        validate_tokens_divisible(tokens=profile.tokens, world_size=self.world)
        S_local = profile.tokens // self.world
        w_up = torch.randn(
            profile.ffn // self.world,
            profile.hidden,
            device=self.device,
            dtype=torch.bfloat16,
        )
        w_down = torch.randn(
            profile.hidden,
            profile.ffn // self.world,
            device=self.device,
            dtype=torch.bfloat16,
        )
        w_up_param = torch.nn.Parameter(w_up.clone())
        w_down_param = torch.nn.Parameter(w_down.clone())

        sdma_comm = SdmaTpComm(dist.group.WORLD)
        sdma = SdmaCommBackend(sdma_comm)
        ag = PipelinedAllgatherGemm(num_chunks=profile.num_chunks, comm=sdma)
        rs = PipelinedGemmReduceScatter(num_chunks=profile.num_chunks, comm=sdma)

        ag_bytes = profile.tokens * profile.hidden * 2
        rs_bytes = profile.tokens * profile.hidden * 2
        needs_grad = not fwd_only

        if bwd_only:
            x_saved = torch.randn(
                S_local,
                profile.hidden,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            up = fused_column_parallel_forward(
                x_saved,
                w_up,
                None,
                ag,
                rs,
                w_up_param,
                False,
                False,
                None,
                False,
            )
            y_saved = fused_row_parallel_forward(
                up,
                w_down,
                rs,
                ag,
                w_down_param,
                False,
                False,
                None,
            )

            def _fused():
                x_saved.grad = None
                w_up_param.grad = None
                w_down_param.grad = None
                y_saved.sum().backward(retain_graph=True)

        else:

            def _fused():
                w_up_param.grad = None
                w_down_param.grad = None
                x = torch.randn(
                    S_local,
                    profile.hidden,
                    device=self.device,
                    dtype=torch.bfloat16,
                    requires_grad=needs_grad,
                )
                up = fused_column_parallel_forward(
                    x,
                    w_up,
                    None,
                    ag,
                    rs,
                    w_up_param,
                    False,
                    False,
                    None,
                    False,
                )
                y = fused_row_parallel_forward(
                    up,
                    w_down,
                    rs,
                    ag,
                    w_down_param,
                    False,
                    False,
                    None,
                )
                if needs_grad:
                    y.sum().backward()
                return y

        for _ in range(3):
            _fused()
        torch.cuda.synchronize()

        label = "fwd" if fwd_only else ("bwd-only" if bwd_only else "fwd+bwd")
        r = cuda_timer(
            _fused,
            label=f"SDMA only {label}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        if self.rank == 0:
            print_report_with_table(
                f"E2E SDMA-Only Pure Pipeline {label} ({format_e2e_fusion_profile(profile)}, world={self.world})",
                [r],
            )
            print_bandwidth_summary(
                label="AG", bytes_transferred=ag_bytes, time_ms=r.avg_ms
            )
            print_bandwidth_summary(
                label="RS", bytes_transferred=rs_bytes, time_ms=r.avg_ms
            )
            print_bench_warnings(result=r)

    def test_single_layer_fwd_bwd(self):
        self._run_sdma(fwd_only=False)

    def test_single_layer_fwd_only(self):
        self._run_sdma(fwd_only=True)

    def test_single_layer_bwd_only(self):
        self._run_sdma(bwd_only=True)


@_DIST
class TestTPScaling:
    """TP scaling sweep: measure pure pipeline benefit at TP=2, 4, 8.

    For each *tp_size*, this rank joins a disjoint TP subgroup of that width
    (e.g. with 8 GPUs and TP=2: subgroups ``{0,1}``, ``{2,3}``, …).  It then
    times a minimal "column + row parallel" FFN block in two ways:

    * **Naive**: monolithic all-gather → two ``F.linear`` → reduce-scatter,
      with autograd-compatible comm (see :func:`_distributed_allgather`).
    * **Fused**: pipelined comm–GEMM overlap
      (:class:`~atomrl.lumen.modules.comm_overlap.PipelinedAllgatherGemm` /
      :class:`~atomrl.lumen.modules.comm_overlap.PipelinedGemmReduceScatter`).

    Forward + backward are timed with eager wgrad in backward. Delayed wgrad is
    benchmarked separately in ``benchmarks/bench_wgrad_delay.py``.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{self.local_rank}")
        yield
        dist.barrier()

    @pytest.mark.skipif(
        "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 8,
        reason="Requires 8 GPUs for TP scaling sweep",
    )
    def test_tp_scaling_sweep(self):
        from atomrl.lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )

        profile = DEFAULT_PROFILE
        tp_sizes = [2, 4, 8]
        sweep_results: list[dict[str, object]] = []

        for tp_size in tp_sizes:
            tp_group = _new_tp_process_group(tp_size)

            validate_tokens_divisible(tokens=profile.tokens, world_size=tp_size)
            S_local = profile.tokens // tp_size
            w_up = torch.randn(
                profile.ffn // tp_size,
                profile.hidden,
                device=self.device,
                dtype=torch.bfloat16,
            )
            w_down = torch.randn(
                profile.hidden,
                profile.ffn // tp_size,
                device=self.device,
                dtype=torch.bfloat16,
            )
            w_up_param = torch.nn.Parameter(w_up.clone())
            w_down_param = torch.nn.Parameter(w_down.clone())

            backend = NcclCommBackend(tp_group)
            ag = PipelinedAllgatherGemm(num_chunks=profile.num_chunks, comm=backend)
            rs = PipelinedGemmReduceScatter(num_chunks=profile.num_chunks, comm=backend)

            ag_bytes = profile.tokens * profile.hidden * 2
            rs_bytes = profile.tokens * profile.hidden * 2

            def _naive(
                s_local=S_local,
                w_u=w_up_param,
                w_d=w_down_param,
                tp_g=tp_group,
                tp_s=tp_size,
            ):
                w_u.grad = None
                w_d.grad = None
                x = torch.randn(
                    s_local,
                    profile.hidden,
                    device=self.device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                gathered = _distributed_allgather(x, group=tp_g)
                up = F.linear(gathered, w_u)
                down = F.linear(up, w_d)
                y = _reduce_scatter(down, tp_s, self.device, group=tp_g)
                y.sum().backward()
                return y

            def _fused(
                s_local=S_local,
                w_u=w_up,
                w_d=w_down,
                _ag=ag,
                _rs=rs,
                w_u_p=w_up_param,
                w_d_p=w_down_param,
            ):
                w_u_p.grad = None
                w_d_p.grad = None
                x = torch.randn(
                    s_local,
                    profile.hidden,
                    device=self.device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                up = fused_column_parallel_forward(
                    x,
                    w_u,
                    None,
                    _ag,
                    _rs,
                    w_u_p,
                    False,
                    False,
                    None,
                    False,
                )
                y = fused_row_parallel_forward(
                    up,
                    w_d,
                    _rs,
                    _ag,
                    w_d_p,
                    False,
                    False,
                    None,
                )
                y.sum().backward()
                return y

            for _ in range(3):
                _naive()
            torch.cuda.synchronize()
            for _ in range(3):
                _fused()
            torch.cuda.synchronize()

            r_naive = cuda_timer(
                _naive,
                label=f"TP={tp_size} naive",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )
            r_fused = cuda_timer(
                _fused,
                label=f"TP={tp_size} fused",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
            ag_bw = compute_bandwidth_gb_s(ag_bytes, r_fused.avg_ms)
            rs_bw = compute_bandwidth_gb_s(rs_bytes, r_fused.avg_ms)

            sweep_results.append(
                {
                    "tp": tp_size,
                    "naive_ms": r_naive.avg_ms,
                    "fused_ms": r_fused.avg_ms,
                    "speedup": speedup,
                    "ag_bw": ag_bw,
                    "rs_bw": rs_bw,
                }
            )

            dist.barrier()

        if self.rank == 0:
            print("\n" + "=" * 72)
            print(f"TP Scaling Summary ({format_e2e_fusion_profile(profile)}, fwd+bwd)")
            print("=" * 72)
            print(
                f"{'TP':>4s}  {'Naive ms':>10s}  {'Fused ms':>10s}  "
                f"{'Speedup':>8s}  {'AG BW':>10s}  {'RS BW':>10s}"
            )
            print("-" * 72)
            for row in sweep_results:
                print(
                    f"{row['tp']:>4d}  {row['naive_ms']:>10.3f}  {row['fused_ms']:>10.3f}  "
                    f"{row['speedup']:>7.2f}x  {row['ag_bw']:>8.1f} GB/s  {row['rs_bw']:>8.1f} GB/s"
                )
            print("=" * 72)
            for row in sweep_results:
                if row["speedup"] < 1.0:
                    print(
                        f"  \u26a0 WARNING: TP={row['tp']} — fused is slower than naive "
                        f"({row['speedup']:.2f}x)"
                    )
