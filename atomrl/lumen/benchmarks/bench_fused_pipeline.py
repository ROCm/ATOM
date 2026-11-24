# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark — Fused pipelined comm-GEMM overlap.

Measures the latency and overlap ratio of the fused pipeline path
(``_FusedColumnParallelForward`` / ``_FusedRowParallelForward``) vs:

  * **Sequential baseline**: AG/RS then GEMM (no overlap).
  * **Fused pipeline**: Real AG+GEMM / GEMM+RS overlap in forward,
    dGEMM+RS / AG+dGEMM overlap in backward.

Both column-parallel and row-parallel are benchmarked.

Requires >= 2 GPUs with the PyTorch NCCL backend
(RCCL underneath on AMD).

Run forward-only::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd

Run forward + backward::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd_bwd
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd_bwd

Run chunk scaling sweep::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k chunk_sweep

Backend matrix envs::

    LUMEN_FUSED_PIPELINE_SDMA_ONLY=column_fwd|row_fwd|column_fwd_bwd|row_fwd_bwd|all
    LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS=1,2,4,8

Run all::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from atomrl.lumen.modules.comm_overlap import (
    NcclCommBackend,
    PipelinedAllgatherGemm,
    PipelinedGemmReduceScatter,
    fused_row_parallel_forward,
)
from atomrl.lumen.ops.quantize.gemm_primitives import (
    compute_dgrad_bf16,
    compute_wgrad_bf16,
)
from atomrl.lumen.ops.quantize.linear import dispatch_gemm

from benchmarks.bench_utils import (
    BenchResult,
    compute_bandwidth_gb_s,  # noqa: F401
    cuda_timer,
    print_bandwidth_summary,  # noqa: F401
    print_bench_warnings,
    print_overlap_summary,
    print_report_with_table,
    print_table,
)

# Dimensions — Llama 3.1 8B shapes
B, S = 2, 2048
H = 4096
FFN = 14336
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0

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


_SDMA_DIST = pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
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


def _bf16_dispatch_gemm(inp, weight, bias=None):
    return dispatch_gemm(inp, weight, None, None, "none", bias=bias)


def _clone_chunk(inp, _weight, _bias):
    return inp.clone()


def _run_row_forward_mirrored_baseline(input_parallel, weight, pipeline_rs):
    return pipeline_rs.forward(input_parallel, weight, _bf16_dispatch_gemm)


def _run_row_backward_mirrored_baseline(
    input_parallel, weight, grad_output, pipeline_ag
):
    grad_input = pipeline_ag.forward(
        grad_output,
        weight=None,
        bias=None,
        gemm_fn=lambda chunk, _weight, _bias: compute_dgrad_bf16(chunk, weight),
    )
    grad_gathered = pipeline_ag.forward(
        grad_output,
        weight=None,
        bias=None,
        gemm_fn=_clone_chunk,
    )
    grad_weight = compute_wgrad_bf16(grad_gathered, input_parallel)
    return grad_input, grad_weight


def _run_row_full_mirrored_baseline(input_parallel, weight, pipeline_rs, pipeline_ag):
    output = _run_row_forward_mirrored_baseline(input_parallel, weight, pipeline_rs)
    grad_output = torch.ones_like(output)
    grad_input, grad_weight = _run_row_backward_mirrored_baseline(
        input_parallel,
        weight,
        grad_output,
        pipeline_ag,
    )
    return output, grad_input, grad_weight


def _analysis_result(name, avg_ms, **extra):
    return BenchResult(name=name, avg_ms=avg_ms, extra=extra)


def _parse_backend_matrix_chunks(default):
    raw = os.environ.get("LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS")
    if raw is None:
        return tuple(default)

    chunks = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS must be a comma-separated list of integers, got {raw!r}"
            ) from exc
        if value < 1:
            raise ValueError(
                f"LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS values must be >= 1, got {value}"
            )
        chunks.append(value)

    if not chunks:
        raise ValueError(
            "LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS must contain at least one integer"
        )
    return tuple(chunks)


def _backend_matrix_note(result: BenchResult, skipped_reason: str | None = None) -> str:
    if skipped_reason is not None:
        return f"skip:{skipped_reason}"
    if result.cv_pct > 2.0:
        return "unstable"
    return "-"


def _build_backend_matrix_summary_row(
    backend,
    phase,
    chunks,
    avg_ms,
    rccl_avg_ms,
    note="-",
):
    return {
        "backend": backend,
        "phase": phase,
        "chunks": chunks,
        "avg_ms": avg_ms,
        "rccl_avg_ms": rccl_avg_ms,
        "speedup_vs_rccl": rccl_avg_ms / max(avg_ms, 1e-6),
        "note": note,
    }


def _print_backend_matrix_summary(world: int, rows: list[dict[str, object]]) -> None:
    sep = "=" * 112
    print(f"\n{sep}")
    print(f"Fused Pipeline Backend Matrix (world={world})")
    print(sep)
    print(
        f"{'Backend':<8} {'Phase':<16} {'Chunks':>6} {'Avg ms':>10} {'vs RCCL':>9} {'Note':>12}"
    )
    print("-" * 112)
    for row in rows:
        print(
            f"{row['backend']:<8} {row['phase']:<16} {row['chunks']:>6} "
            f"{row['avg_ms']:>10.3f} {row['speedup_vs_rccl']:>8.2f}x {row['note']:>12}"
        )
    print(sep)


def _print_row_full_analysis_table(
    *, title, forward_ms, ag_dgrad_ms, ag_wgrad_ms, baseline_total_ms, fused_total_ms
):
    delta_ms = fused_total_ms - baseline_total_ms
    speedup = baseline_total_ms / max(fused_total_ms, 1e-6)
    print_table(
        title,
        [
            _analysis_result("baseline forward (GEMM+RS)", forward_ms),
            _analysis_result("baseline backward (AG+dgrad)", ag_dgrad_ms),
            _analysis_result("baseline backward (AG+wgrad)", ag_wgrad_ms),
            _analysis_result("baseline full fwd+bwd", baseline_total_ms),
            _analysis_result(
                "fused full fwd+bwd",
                fused_total_ms,
                delta_ms=round(delta_ms, 3),
                speedup=round(speedup, 2),
            ),
        ],
    )


@_DIST
class TestFusedPipelineColumnFwd:
    """Column-parallel forward: AG + GEMM overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_fused_vs_sequential_fwd(self):
        from atomrl.lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        x_local = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)

        for _ in range(3):
            gathered = _distributed_allgather(x_local)
            _ = F.linear(gathered, w)
        torch.cuda.synchronize()

        def _seq():
            gathered = _distributed_allgather(x_local)
            return F.linear(gathered, w)

        r_seq = cuda_timer(
            _seq,
            label="sequential (AG->GEMM)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        r_comm = cuda_timer(
            lambda: _distributed_allgather(x_local),
            label="AG alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        gathered = _distributed_allgather(x_local)
        r_compute = cuda_timer(
            lambda: F.linear(gathered, w),
            label="GEMM alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _fused():
            return fused_column_parallel_forward(
                x_local.clone(),
                w.clone(),
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )

        r_fused = cuda_timer(
            _fused,
            label="fused pipeline (AG+GEMM)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_fused.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_fused.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Column-Parallel Forward (world={self.world})",
                [r_comm, r_compute, r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="GEMM",
                comm_label="allgather",
            )

    @pytest.mark.parametrize("num_chunks", [1, 2, 4, 8])
    def test_chunk_sweep_fwd(self, num_chunks):
        from atomrl.lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        x_local = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())

        if S_local * self.world % num_chunks != 0:
            pytest.skip(f"S={S_local * self.world} not divisible by {num_chunks}")

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=num_chunks, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=num_chunks, comm=backend)

        def _fused():
            return fused_column_parallel_forward(
                x_local.clone(),
                w.clone(),
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )

        r = cuda_timer(
            _fused,
            label=f"fused N={num_chunks}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        if self.rank == 0:
            print_report_with_table(
                f"Column Fwd Chunk Sweep N={num_chunks} (world={self.world})",
                [r],
            )


@_DIST
class TestFusedPipelineFwdBwd:
    """Full forward + backward cycle latency."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_fused_column_fwd_bwd(self):
        from atomrl.lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)
        w_param = torch.nn.Parameter(w.clone())

        def _seq_fwd_bwd():
            x = torch.randn(
                S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True
            )
            gathered = _distributed_allgather(x)
            y = F.linear(gathered, w)
            y.sum().backward()
            return x.grad

        def _fused_fwd_bwd():
            x = torch.randn(
                S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True
            )
            y = fused_column_parallel_forward(
                x,
                w_param,
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )
            y.sum().backward()
            return x.grad

        for _ in range(3):
            _seq_fwd_bwd()
            _fused_fwd_bwd()
        torch.cuda.synchronize()

        r_seq = cuda_timer(
            _seq_fwd_bwd,
            label="sequential fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_fwd_bwd,
            label="fused fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Column Fwd+Bwd (world={self.world})",
                [r_seq, r_fused],
            )


@_DIST
class TestFusedPipelineRowFwd:
    """Row-parallel forward: GEMM + RS overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_fused_row_vs_sequential_fwd(self):
        from atomrl.lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_row_parallel_forward,
        )

        S_full = B * S
        H_tp = H // self.world
        x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)

        for _ in range(3):
            gemm_out = F.linear(x, w)
            _reduce_scatter(gemm_out, self.world, self.device)
        torch.cuda.synchronize()

        def _seq():
            return _reduce_scatter(F.linear(x, w), self.world, self.device)

        r_seq = cuda_timer(
            _seq,
            label="sequential (GEMM->RS)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        r_compute = cuda_timer(
            lambda: F.linear(x, w),
            label="GEMM alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        gemm_out = F.linear(x, w)
        r_comm = cuda_timer(
            lambda: _reduce_scatter(gemm_out, self.world, self.device),
            label="RS alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _fused():
            return fused_row_parallel_forward(
                x.clone(),
                w.clone(),
                pipeline_rs,
                pipeline_ag,
                w_param,
                False,
                False,
                None,
            )

        r_fused = cuda_timer(
            _fused,
            label="fused pipeline (GEMM+RS)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_comm.avg_ms + r_compute.avg_ms
        overlap_ratio = 1 - (r_fused.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)
        r_fused.extra["speedup"] = round(speedup, 2)

        if self.rank == 0:
            print_report_with_table(
                f"Fused Row-Parallel Forward (world={self.world})",
                [r_comm, r_compute, r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_compute.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="GEMM",
                comm_label="reduce_scatter",
            )


@_SDMA_DIST
class TestFusedPipelineSdmaColumn:
    """RCCL vs SDMA backend matrix for fused pipelined overlap."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def _make_backends(self):
        from atomrl.lumen.modules.comm_overlap import SdmaCommBackend
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        rccl = NcclCommBackend(dist.group.WORLD)
        sdma_comm = SdmaTpComm(dist.group.WORLD)
        sdma = SdmaCommBackend(sdma_comm)
        return rccl, sdma

    def _make_pipelines(self, comm_backend, num_chunks):
        return (
            PipelinedAllgatherGemm(num_chunks=num_chunks, comm=comm_backend),
            PipelinedGemmReduceScatter(num_chunks=num_chunks, comm=comm_backend),
        )

    def _sync_all_ranks(self):
        torch.cuda.synchronize()
        dist.barrier()

    def _require_matching_chunks(self, chunks):
        expected = [tuple(chunks) if self.rank == 0 else None]
        dist.broadcast_object_list(expected, src=0)
        if tuple(chunks) != expected[0]:
            raise RuntimeError(
                "LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS must match across all ranks"
            )
        return expected[0]

    def _validate_phase_chunks(self, phase, num_chunks):
        total_tokens = B * S
        if phase.startswith("column"):
            s_local = total_tokens // self.world
            if s_local % num_chunks != 0:
                pytest.skip(
                    f"phase={phase} requires S_local={s_local} divisible by chunks={num_chunks}"
                )
            return

        if total_tokens % num_chunks != 0:
            pytest.skip(
                f"phase={phase} requires S_full={total_tokens} divisible by chunks={num_chunks}"
            )
        chunk_full = total_tokens // num_chunks
        if chunk_full % self.world != 0:
            pytest.skip(
                f"phase={phase} requires chunk_full={chunk_full} divisible by "
                f"tp_size={self.world} for chunks={num_chunks}"
            )

    def _bench_backend_pair(self, title, phase, rccl_run, sdma_run):
        for _ in range(3):
            rccl_run()
            sdma_run()
        torch.cuda.synchronize()

        r_rccl = cuda_timer(
            rccl_run,
            label=f"RCCL {phase}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_sdma = cuda_timer(
            sdma_run,
            label=f"SDMA {phase}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_rccl.extra["speedup_vs_rccl"] = 1.0
        r_sdma.extra["speedup_vs_rccl"] = round(
            r_rccl.avg_ms / max(r_sdma.avg_ms, 1e-6), 2
        )

        if self.rank == 0:
            print_report_with_table(title, [r_rccl, r_sdma])
            print_bench_warnings(
                result=r_sdma, speedup=r_rccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
            )
        return r_rccl, r_sdma

    def _bench_fused_column_fwd(
        self, rccl_backend, sdma_backend, num_chunks, fused_column_parallel_forward
    ):
        self._validate_phase_chunks("column_fwd", num_chunks)
        S_local = (B * S) // self.world
        x_local = torch.randn(S_local, H, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        ag_rccl, rs_rccl = self._make_pipelines(rccl_backend, num_chunks)
        ag_sdma, rs_sdma = self._make_pipelines(sdma_backend, num_chunks)

        def _run(ag, rs):
            return fused_column_parallel_forward(
                x_local.clone(),
                w.clone(),
                None,
                ag,
                rs,
                w_param,
                False,
                False,
                None,
                False,
            )

        return self._bench_backend_pair(
            f"Fused Column Fwd: RCCL vs SDMA (world={self.world}, chunks={num_chunks})",
            "fused column fwd",
            lambda: _run(ag_rccl, rs_rccl),
            lambda: _run(ag_sdma, rs_sdma),
        )

    def _bench_fused_row_fwd(
        self, rccl_backend, sdma_backend, num_chunks, fused_row_parallel_forward
    ):
        self._validate_phase_chunks("row_fwd", num_chunks)
        S_full = B * S
        H_tp = H // self.world
        x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        ag_rccl, rs_rccl = self._make_pipelines(rccl_backend, num_chunks)
        ag_sdma, rs_sdma = self._make_pipelines(sdma_backend, num_chunks)

        def _run(rs, ag):
            return fused_row_parallel_forward(
                x.clone(),
                w.clone(),
                rs,
                ag,
                w_param,
                False,
                False,
                None,
            )

        return self._bench_backend_pair(
            f"Fused Row Fwd: RCCL vs SDMA (world={self.world}, chunks={num_chunks})",
            "fused row fwd",
            lambda: _run(rs_rccl, ag_rccl),
            lambda: _run(rs_sdma, ag_sdma),
        )

    def _bench_fused_column_fwd_bwd(
        self, rccl_backend, sdma_backend, num_chunks, fused_column_parallel_forward
    ):
        self._validate_phase_chunks("column_fwd_bwd", num_chunks)
        S_local = (B * S) // self.world
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        ag_rccl, rs_rccl = self._make_pipelines(rccl_backend, num_chunks)
        ag_sdma, rs_sdma = self._make_pipelines(sdma_backend, num_chunks)

        def _run(ag, rs):
            x = torch.randn(
                S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True
            )
            y = fused_column_parallel_forward(
                x,
                w_param,
                None,
                ag,
                rs,
                w_param,
                False,
                False,
                None,
                False,
            )
            y.sum().backward()
            return x.grad

        return self._bench_backend_pair(
            f"Fused Column Fwd+Bwd: RCCL vs SDMA (world={self.world}, chunks={num_chunks})",
            "fused column fwd+bwd",
            lambda: _run(ag_rccl, rs_rccl),
            lambda: _run(ag_sdma, rs_sdma),
        )

    def _bench_fused_row_fwd_bwd(
        self, rccl_backend, sdma_backend, num_chunks, fused_row_parallel_forward
    ):
        self._validate_phase_chunks("row_fwd_bwd", num_chunks)
        S_full = B * S
        H_tp = H // self.world
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        ag_rccl, rs_rccl = self._make_pipelines(rccl_backend, num_chunks)
        ag_sdma, rs_sdma = self._make_pipelines(sdma_backend, num_chunks)

        def _run(rs, ag):
            x = torch.randn(
                S_full,
                H_tp,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            y = fused_row_parallel_forward(
                x,
                w_param,
                rs,
                ag,
                w_param,
                False,
                False,
                None,
            )
            y.sum().backward()
            return y, x.grad, w_param.grad

        return self._bench_backend_pair(
            f"Fused Row Fwd+Bwd: RCCL vs SDMA (world={self.world}, chunks={num_chunks})",
            "fused row fwd+bwd",
            lambda: _run(rs_rccl, ag_rccl),
            lambda: _run(rs_sdma, ag_sdma),
        )

    def test_nccl_vs_sdma_fused_suite(self):
        """Run all SDMA fused backend-matrix benchmarks in one test method.

        Keep one SDMA handle set alive across the suite. Mori allocates KFD
        queues during handle construction; recreating SDMA handles across
        separate test methods can exhaust that pool and later surface as
        AllGather timeouts on 8-GPU runs.

        Set ``LUMEN_FUSED_PIPELINE_SDMA_ONLY`` to one of ``column_fwd``,
        ``row_fwd``, ``column_fwd_bwd``, or ``row_fwd_bwd`` to run a single
        phase while still keeping one pytest node / one SDMA handle lifecycle.
        Set ``LUMEN_FUSED_PIPELINE_BACKEND_CHUNKS`` to a comma-separated chunk
        list such as ``1,2,4,8`` to sweep chunk counts.
        """
        from atomrl.lumen.modules.comm_overlap import (
            fused_column_parallel_forward,
            fused_row_parallel_forward,
        )

        selected_chunks = self._require_matching_chunks(
            _parse_backend_matrix_chunks((1, 2, 4, 8))
        )
        selected_phase = (
            os.environ.get("LUMEN_FUSED_PIPELINE_SDMA_ONLY", "all").strip().lower()
        )
        phase_codes = {
            "all": 0,
            "column_fwd": 1,
            "row_fwd": 2,
            "column_fwd_bwd": 3,
            "row_fwd_bwd": 4,
        }
        valid_phases = set(phase_codes)
        phase_code_value = phase_codes.get(selected_phase, -1)
        phase_code = torch.tensor(
            [phase_code_value], device=self.device, dtype=torch.int32
        )
        phase_min = phase_code.clone()
        phase_max = phase_code.clone()
        dist.all_reduce(phase_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(phase_max, op=dist.ReduceOp.MAX)
        if phase_min.item() != phase_max.item():
            raise RuntimeError(
                "LUMEN_FUSED_PIPELINE_SDMA_ONLY must match across all ranks"
            )
        if phase_code_value < 0:
            raise ValueError(
                "LUMEN_FUSED_PIPELINE_SDMA_ONLY must be one of "
                f"{sorted(valid_phases)}, got {selected_phase!r}"
            )

        rccl_backend, sdma_backend = self._make_backends()
        summary_rows = []

        def _record_summary(phase, chunks, rccl_result, sdma_result):
            if self.rank != 0:
                return
            summary_rows.append(
                _build_backend_matrix_summary_row(
                    backend="RCCL",
                    phase=phase,
                    chunks=chunks,
                    avg_ms=rccl_result.avg_ms,
                    rccl_avg_ms=rccl_result.avg_ms,
                    note=_backend_matrix_note(rccl_result),
                )
            )
            summary_rows.append(
                _build_backend_matrix_summary_row(
                    backend="SDMA",
                    phase=phase,
                    chunks=chunks,
                    avg_ms=sdma_result.avg_ms,
                    rccl_avg_ms=rccl_result.avg_ms,
                    note=_backend_matrix_note(sdma_result),
                )
            )

        for num_chunks in selected_chunks:
            if selected_phase in {"all", "column_fwd"}:
                r_rccl, r_sdma = self._bench_fused_column_fwd(
                    rccl_backend,
                    sdma_backend,
                    num_chunks,
                    fused_column_parallel_forward,
                )
                _record_summary("column_fwd", num_chunks, r_rccl, r_sdma)
                self._sync_all_ranks()

            if selected_phase in {"all", "row_fwd"}:
                r_rccl, r_sdma = self._bench_fused_row_fwd(
                    rccl_backend,
                    sdma_backend,
                    num_chunks,
                    fused_row_parallel_forward,
                )
                _record_summary("row_fwd", num_chunks, r_rccl, r_sdma)
                self._sync_all_ranks()

            if selected_phase in {"all", "column_fwd_bwd"}:
                r_rccl, r_sdma = self._bench_fused_column_fwd_bwd(
                    rccl_backend,
                    sdma_backend,
                    num_chunks,
                    fused_column_parallel_forward,
                )
                _record_summary("column_fwd_bwd", num_chunks, r_rccl, r_sdma)
                self._sync_all_ranks()

            if selected_phase in {"all", "row_fwd_bwd"}:
                r_rccl, r_sdma = self._bench_fused_row_fwd_bwd(
                    rccl_backend,
                    sdma_backend,
                    num_chunks,
                    fused_row_parallel_forward,
                )
                _record_summary("row_fwd_bwd", num_chunks, r_rccl, r_sdma)
                self._sync_all_ranks()

        if self.rank == 0 and summary_rows:
            _print_backend_matrix_summary(self.world, summary_rows)


@_DIST
class TestFusedPipelineBackwardOverlap:
    """Backward overlap microbenchmarks plus aligned full row fwd+bwd comparison."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_column_backward_dgrad_rs_overlap(self):
        """Column-parallel backward: dgrad + RS overlap."""
        from atomrl.lumen.modules.comm_overlap import (
            NcclCommBackend,
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        S_local = (B * S) // self.world
        w = torch.randn(FFN // self.world, H, device=self.device, dtype=torch.bfloat16)
        w_param = torch.nn.Parameter(w.clone())
        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=backend)

        def _seq_bwd():
            x = torch.randn(
                S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True
            )
            gathered = _distributed_allgather(x)
            y = F.linear(gathered, w)
            y.sum().backward()
            return x.grad

        def _fused_bwd():
            x = torch.randn(
                S_local, H, device=self.device, dtype=torch.bfloat16, requires_grad=True
            )
            y = fused_column_parallel_forward(
                x,
                w_param,
                None,
                pipeline_ag,
                pipeline_rs,
                w_param,
                False,
                False,
                None,
                False,
            )
            y.sum().backward()
            return x.grad

        for _ in range(3):
            _seq_bwd()
            _fused_bwd()
        torch.cuda.synchronize()

        r_seq = cuda_timer(
            _seq_bwd,
            label="sequential column bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_bwd,
            label="fused column bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_seq.avg_ms / max(r_fused.avg_ms, 1e-6)
        overlap_ratio = 1 - (r_fused.avg_ms / max(r_seq.avg_ms, 1e-6))
        r_fused.extra["speedup"] = round(speedup, 2)
        r_fused.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report_with_table(
                f"Column Backward Overlap (world={self.world})",
                [r_seq, r_fused],
            )
            print_overlap_summary(
                t_compute=r_seq.avg_ms,
                t_comm=0,
                t_seq=r_seq.avg_ms,
                t_ovl=r_fused.avg_ms,
                compute_label="seq bwd",
                comm_label="(fused)",
            )
            print_bench_warnings(
                result=r_fused, speedup=speedup, overlap_ratio=overlap_ratio
            )

    def test_row_full_fwd_bwd_mirrored_baseline_matches_fused(self):
        """Reference full row fwd+bwd path should mirror fused BF16 semantics."""
        S_full = B * S
        H_tp = H // self.world
        torch.manual_seed(2026 + self.rank)
        x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)

        ref_backend = NcclCommBackend(dist.group.WORLD)
        ref_pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=ref_backend)
        ref_pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=ref_backend)
        ref_out, ref_grad_input, ref_grad_weight = _run_row_full_mirrored_baseline(
            x,
            w,
            ref_pipeline_rs,
            ref_pipeline_ag,
        )

        fused_backend = NcclCommBackend(dist.group.WORLD)
        fused_pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=fused_backend)
        fused_pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=fused_backend)
        x_fused = x.clone().detach().requires_grad_(True)
        w_param = torch.nn.Parameter(w.clone())
        y_fused = fused_row_parallel_forward(
            x_fused,
            w_param,
            fused_pipeline_rs,
            fused_pipeline_ag,
            w_param,
            False,
            False,
            None,
        )
        y_fused.sum().backward()

        torch.testing.assert_close(y_fused.detach(), ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(x_fused.grad, ref_grad_input, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w_param.grad, ref_grad_weight, rtol=1e-2, atol=1e-2)

    def test_row_full_fwd_bwd_mirrored_baseline_vs_fused(self):
        """Row-parallel full fwd+bwd: mirrored baseline vs fused."""
        S_full = B * S
        H_tp = H // self.world
        w = torch.randn(FFN, H_tp, device=self.device, dtype=torch.bfloat16)

        baseline_backend = NcclCommBackend(dist.group.WORLD)
        baseline_pipeline_rs = PipelinedGemmReduceScatter(
            num_chunks=4, comm=baseline_backend
        )
        baseline_pipeline_ag = PipelinedAllgatherGemm(
            num_chunks=4, comm=baseline_backend
        )

        def _baseline_full():
            x = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
            return _run_row_full_mirrored_baseline(
                x,
                w,
                baseline_pipeline_rs,
                baseline_pipeline_ag,
            )

        fused_backend = NcclCommBackend(dist.group.WORLD)
        fused_pipeline_rs = PipelinedGemmReduceScatter(num_chunks=4, comm=fused_backend)
        fused_pipeline_ag = PipelinedAllgatherGemm(num_chunks=4, comm=fused_backend)

        def _fused_full():
            x = torch.randn(
                S_full,
                H_tp,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            w_param = torch.nn.Parameter(w.clone())
            y = fused_row_parallel_forward(
                x,
                w_param,
                fused_pipeline_rs,
                fused_pipeline_ag,
                w_param,
                False,
                False,
                None,
            )
            y.sum().backward()
            return y, x.grad, w_param.grad

        for _ in range(3):
            _baseline_full()
            _fused_full()
        torch.cuda.synchronize()

        r_baseline = cuda_timer(
            _baseline_full,
            label="mirrored baseline fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_fused = cuda_timer(
            _fused_full,
            label="fused fwd+bwd",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_baseline.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)

        analysis_backend = NcclCommBackend(dist.group.WORLD)
        analysis_pipeline_rs = PipelinedGemmReduceScatter(
            num_chunks=4, comm=analysis_backend
        )
        analysis_pipeline_ag = PipelinedAllgatherGemm(
            num_chunks=4, comm=analysis_backend
        )
        x_analysis = torch.randn(S_full, H_tp, device=self.device, dtype=torch.bfloat16)
        y_analysis = _run_row_forward_mirrored_baseline(
            x_analysis, w, analysis_pipeline_rs
        )
        grad_output_analysis = torch.ones_like(y_analysis)

        r_forward = cuda_timer(
            lambda: _run_row_forward_mirrored_baseline(
                x_analysis, w, analysis_pipeline_rs
            ),
            label="baseline forward (GEMM+RS)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_ag_dgrad = cuda_timer(
            lambda: analysis_pipeline_ag.forward(
                grad_output_analysis,
                weight=None,
                bias=None,
                gemm_fn=lambda chunk, _weight, _bias: compute_dgrad_bf16(chunk, w),
            ),
            label="baseline backward (AG+dgrad)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        r_ag_wgrad = cuda_timer(
            lambda: compute_wgrad_bf16(
                analysis_pipeline_ag.forward(
                    grad_output_analysis,
                    weight=None,
                    bias=None,
                    gemm_fn=_clone_chunk,
                ),
                x_analysis,
            ),
            label="baseline backward (AG+wgrad)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        if self.rank == 0:
            print_report_with_table(
                f"Full Row Fwd+Bwd: Mirrored Baseline vs Fused (world={self.world})",
                [r_baseline, r_fused],
            )
            _print_row_full_analysis_table(
                title=f"Full Row Fwd+Bwd Breakdown (world={self.world})",
                forward_ms=r_forward.avg_ms,
                ag_dgrad_ms=r_ag_dgrad.avg_ms,
                ag_wgrad_ms=r_ag_wgrad.avg_ms,
                baseline_total_ms=r_baseline.avg_ms,
                fused_total_ms=r_fused.avg_ms,
            )
            print_bench_warnings(result=r_fused, speedup=speedup)
