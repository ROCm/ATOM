# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Chunked comm-GEMM overlap engines for TP parallelism.

Provides pipelined allgather+GEMM and GEMM+reduce-scatter engines that split
collectives into N chunks and interleave them with computation using the
split-FPROP/BPROP pipelining algorithm, optimised for AMD MI300X and mori SDMA.

Two communication backends are supported:

- :class:`SdmaCommBackend` — wraps mori SDMA via :class:`SdmaTpComm`
- :class:`NcclCommBackend` — wraps ``torch.distributed``

Usage::

    backend = SdmaCommBackend(sdma_comm)  # or NcclCommBackend(tp_group)
    ag_pipe = PipelinedAllgatherGemm(num_chunks=4, comm=backend)
    output = ag_pipe.forward(input_local, weight, bias, gemm_fn)
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.distributed as dist

from atomrl.lumen.ops.quantize.gemm_primitives import (
    compute_dgrad_bf16,
    compute_dgrad_fp8,
    compute_wgrad_bf16,
    make_wgrad_closure_bf16,
)
from atomrl.lumen.ops.quantize.linear import dispatch_gemm, quantize_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UserBufferPool — reusable staging buffers
# ---------------------------------------------------------------------------


class UserBufferPool:
    """Reusable buffer pool for pipeline chunk staging.

    Buffers are keyed by ``(name, dtype)`` and reallocated only when the
    required element count exceeds the current allocation.  This avoids
    per-step ``torch.empty`` calls in the hot path.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._buffers: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def get(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Return a buffer with at least ``shape`` elements, reusing when possible."""
        key = (name, dtype)
        numel_needed = 1
        for s in shape:
            numel_needed *= s
        buf = self._buffers.get(key)
        if buf is not None and buf.numel() >= numel_needed:
            return buf.view(-1)[:numel_needed].view(shape)
        self._buffers[key] = torch.empty(shape, dtype=dtype, device=self._device)
        return self._buffers[key]

    def reset(self) -> None:
        """Release all cached buffers."""
        self._buffers.clear()


# ---------------------------------------------------------------------------
# CommBackend — abstract transport protocol
# ---------------------------------------------------------------------------


class CommBackend(ABC):
    """Protocol for chunk-level TP communication."""

    @property
    @abstractmethod
    def tp_size(self) -> int: ...

    @property
    @abstractmethod
    def tp_rank(self) -> int: ...

    @abstractmethod
    def allgather_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        """Allgather a single chunk into ``output_buf``.

        ``output_buf`` has shape ``[tp_size * chunk_seq, ...]`` where
        ``chunk_seq`` is ``input_chunk.shape[0]``.

        Returns a CUDA event that fires when the operation completes.
        """
        ...

    @abstractmethod
    def reduce_scatter_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        """Reduce-scatter a single chunk into ``output_buf``.

        ``input_chunk`` has shape ``[S_chunk, ...]``.
        ``output_buf`` has shape ``[S_chunk // tp_size, ...]``.

        Returns a CUDA event that fires when the operation completes.
        """
        ...


class NcclCommBackend(CommBackend):
    """Communication backend using NCCL via ``torch.distributed``."""

    def __init__(self, tp_group: dist.ProcessGroup):
        self._group = tp_group
        self._tp_size = dist.get_world_size(group=tp_group)
        self._tp_rank = dist.get_rank(group=tp_group)

    @property
    def tp_size(self) -> int:
        return self._tp_size

    @property
    def tp_rank(self) -> int:
        return self._tp_rank

    def allgather_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        with torch.cuda.stream(stream):
            chunk_list = list(output_buf.chunk(self._tp_size, dim=0))
            dist.all_gather(chunk_list, input_chunk.contiguous(), group=self._group)
        return stream.record_event()

    def reduce_scatter_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        with torch.cuda.stream(stream):
            dist.reduce_scatter_tensor(
                output_buf, input_chunk.contiguous(), group=self._group
            )
        return stream.record_event()


class SdmaCommBackend(CommBackend):
    """Communication backend using mori SDMA via :class:`SdmaTpComm`."""

    def __init__(self, sdma_comm):
        self._comm = sdma_comm

    @property
    def tp_size(self) -> int:
        return self._comm.npes

    @property
    def tp_rank(self) -> int:
        return self._comm.my_pe

    def allgather_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        return self._comm.allgather_dim0_chunk(input_chunk, output_buf, stream)

    def reduce_scatter_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        return self._comm.reduce_scatter_dim0_chunk(input_chunk, output_buf, stream)


# ---------------------------------------------------------------------------
# PipelinedAllgatherGemm — split-FPROP for column-parallel
# ---------------------------------------------------------------------------


class PipelinedAllgatherGemm:
    """Chunked allgather overlapped with GEMM for column-parallel forward.

    Timeline (4 chunks, TP=8)::

        comm_stream:  [AG_c0] [AG_c1] [AG_c2] [AG_c3]
        compute:               [GEMM0] [GEMM1] [GEMM2] [GEMM3]
                               ^ starts when AG_c0 done

    Each ``AG_chunk[i]`` gathers ``S_local / num_chunks`` rows from all TP
    ranks.  The GEMM for chunk *i* starts as soon as chunk *i*'s gather
    completes, while chunk *i+1*'s gather runs concurrently on the comm
    stream.  Double-buffered staging avoids write-after-read hazards.
    """

    def __init__(self, num_chunks: int, comm: CommBackend):
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self._num_chunks = num_chunks
        self._comm = comm
        self._comm_stream: torch.cuda.Stream | None = None
        self._buf_pool: UserBufferPool | None = None

    def forward(
        self,
        input_local: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gemm_fn: Callable,
    ) -> torch.Tensor:
        """Run pipelined allgather + GEMM.

        Args:
            input_local: ``[S_local, H]`` — local sequence shard.
            weight: ``[O_local, H]`` — column-parallel weight shard.
            bias: Optional bias for the first GEMM chunk.
            gemm_fn: ``(input, weight, bias) -> output`` wrapping ``_do_gemm``.

        Returns:
            ``[S_full, O_local]`` where ``S_full = S_local * tp_size``.
        """
        N = self._num_chunks
        tp_size = self._comm.tp_size

        if N == 1:
            return self._forward_single_chunk(input_local, weight, bias, gemm_fn)

        S_local = input_local.shape[0]
        H = input_local.shape[1]

        chunk_local = S_local // N
        if chunk_local * N != S_local:
            raise ValueError(
                f"S_local={S_local} not divisible by num_chunks={N}. "
                f"Adjust sequence length or chunk count."
            )
        chunk_full = chunk_local * tp_size

        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=input_local.device)
        if self._buf_pool is None:
            self._buf_pool = UserBufferPool(input_local.device)

        compute_stream = torch.cuda.current_stream(input_local.device)
        comm_stream = self._comm_stream

        buf_a = self._buf_pool.get("ag_a", (chunk_full, H), input_local.dtype)
        buf_b = self._buf_pool.get("ag_b", (chunk_full, H), input_local.dtype)
        bufs = [buf_a, buf_b]

        local_chunks = list(input_local.split(chunk_local, dim=0))

        output_chunks: list[torch.Tensor] = []

        comm_stream.wait_stream(compute_stream)
        ev_prev = self._comm.allgather_chunk(local_chunks[0], bufs[0], comm_stream)

        for i in range(1, N):
            compute_stream.wait_event(ev_prev)

            out_i = gemm_fn(bufs[(i - 1) % 2], weight, bias if i == 1 else None)
            output_chunks.append(out_i)

            comm_stream.wait_stream(compute_stream)
            ev_prev = self._comm.allgather_chunk(
                local_chunks[i], bufs[i % 2], comm_stream
            )

        compute_stream.wait_event(ev_prev)
        out_last = gemm_fn(bufs[(N - 1) % 2], weight, None)
        output_chunks.append(out_last)

        result = torch.cat(output_chunks, dim=0)

        # Reorder from chunk-interleaved layout to rank-contiguous layout.
        # Each allgather chunk produces [tp_size * chunk_local, ...] with rows
        # ordered as [rank0, rank1, ...].  After concatenating N chunks the
        # layout is [c0_r0, c0_r1, ..., c1_r0, c1_r1, ...] which differs from
        # the monolithic allgather order [r0_c0, r0_c1, ..., r1_c0, r1_c1, ...].
        # Reshape → transpose → reshape converts between the two.
        out_cols = result.shape[1:]
        result = (
            result.view(N, tp_size, chunk_local, *out_cols)
            .transpose(0, 1)
            .contiguous()
            .view(-1, *out_cols)
        )
        return result

    def _forward_single_chunk(
        self,
        input_local: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gemm_fn: Callable,
    ) -> torch.Tensor:
        """Degenerate case: single chunk = monolithic allgather then GEMM."""
        tp_size = self._comm.tp_size
        S_local = input_local.shape[0]
        H = input_local.shape[1]

        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=input_local.device)
        if self._buf_pool is None:
            self._buf_pool = UserBufferPool(input_local.device)

        compute_stream = torch.cuda.current_stream(input_local.device)
        comm_stream = self._comm_stream

        gathered = self._buf_pool.get(
            "ag_full", (S_local * tp_size, H), input_local.dtype
        )

        comm_stream.wait_stream(compute_stream)
        ev = self._comm.allgather_chunk(input_local, gathered, comm_stream)
        compute_stream.wait_event(ev)

        return gemm_fn(gathered, weight, bias)


# ---------------------------------------------------------------------------
# PipelinedGemmReduceScatter — split-BPROP for row-parallel
# ---------------------------------------------------------------------------


class PipelinedGemmReduceScatter:
    """Chunked GEMM overlapped with reduce-scatter for row-parallel forward.

    Timeline (4 chunks)::

        compute:  [GEMM_c0] [GEMM_c1] [GEMM_c2] [GEMM_c3]
        comm:                [RS_c0]   [RS_c1]   [RS_c2]   [RS_c3]
                             ^ starts when GEMM_c0 done

    The GEMM output is computed in chunks along dim 0 (sequence dimension).
    Each chunk's reduce-scatter launches as soon as its GEMM completes.
    """

    def __init__(self, num_chunks: int, comm: CommBackend):
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self._num_chunks = num_chunks
        self._comm = comm
        self._comm_stream: torch.cuda.Stream | None = None
        self._buf_pool: UserBufferPool | None = None

    def forward(
        self,
        input_full: torch.Tensor,
        weight: torch.Tensor,
        gemm_fn: Callable,
    ) -> torch.Tensor:
        """Run pipelined GEMM + reduce-scatter.

        Args:
            input_full: ``[S_full, H_local]`` — full sequence, partitioned hidden.
            weight: ``[O, H_local]`` — row-parallel weight.
            gemm_fn: ``(input, weight, bias) -> output``.

        Returns:
            ``[S_local, O]`` where ``S_local = S_full // tp_size``.
        """
        N = self._num_chunks
        tp_size = self._comm.tp_size

        if N == 1:
            return self._forward_single_chunk(input_full, weight, gemm_fn)

        S_full = input_full.shape[0]
        chunk_full = S_full // N
        if chunk_full * N != S_full:
            raise ValueError(f"S_full={S_full} not divisible by num_chunks={N}.")
        if chunk_full % tp_size != 0:
            raise ValueError(
                f"chunk_full={chunk_full} (S_full={S_full} // num_chunks={N}) "
                f"must be divisible by tp_size={tp_size} for reduce_scatter."
            )
        chunk_local = chunk_full // tp_size

        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=input_full.device)
        if self._buf_pool is None:
            self._buf_pool = UserBufferPool(input_full.device)

        compute_stream = torch.cuda.current_stream(input_full.device)
        comm_stream = self._comm_stream

        # Reorder input from rank-contiguous [r0_c0..cN, r1_c0..cN, ...] to
        # chunk-interleaved [c0_r0..rP, c1_r0..rP, ...] so that per-chunk
        # reduce-scatter produces results equivalent to a monolithic RS.
        trailing = input_full.shape[1:]
        input_reordered = (
            input_full.view(tp_size, N, chunk_local, *trailing)
            .transpose(0, 1)
            .contiguous()
            .view(S_full, *trailing)
        )
        input_chunks = list(input_reordered.split(chunk_full, dim=0))

        rs_events: list[torch.cuda.Event] = []
        rs_bufs: list[torch.Tensor] = []

        gemm_out_prev = gemm_fn(input_chunks[0], weight, None)

        for i in range(1, N):
            out_dim = gemm_out_prev.shape[-1]
            rs_buf = torch.empty(
                chunk_local,
                out_dim,
                dtype=gemm_out_prev.dtype,
                device=input_full.device,
            )
            comm_stream.wait_stream(compute_stream)
            ev = self._comm.reduce_scatter_chunk(gemm_out_prev, rs_buf, comm_stream)
            rs_events.append(ev)
            rs_bufs.append(rs_buf)

            gemm_out_prev = gemm_fn(input_chunks[i], weight, None)

        out_dim = gemm_out_prev.shape[-1]
        rs_buf = torch.empty(
            chunk_local, out_dim, dtype=gemm_out_prev.dtype, device=input_full.device
        )
        comm_stream.wait_stream(compute_stream)
        ev = self._comm.reduce_scatter_chunk(gemm_out_prev, rs_buf, comm_stream)
        rs_events.append(ev)
        rs_bufs.append(rs_buf)

        for ev in rs_events:
            compute_stream.wait_event(ev)

        return torch.cat(rs_bufs, dim=0)

    def forward_standalone(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pipeline only the reduce-scatter (no GEMM), for row-parallel post-GEMM.

        Args:
            tensor: ``[S_full, O]`` — GEMM output to reduce-scatter.

        Returns:
            ``[S_local, O]``.
        """
        N = self._num_chunks
        tp_size = self._comm.tp_size

        S_full = tensor.shape[0]
        chunk_full = S_full // N
        if chunk_full * N != S_full:
            raise ValueError(f"S_full={S_full} not divisible by num_chunks={N}.")
        chunk_local = chunk_full // tp_size
        if chunk_local * tp_size != chunk_full:
            raise ValueError(
                f"chunk_full={chunk_full} not divisible by tp_size={tp_size}."
            )

        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=tensor.device)

        compute_stream = torch.cuda.current_stream(tensor.device)
        comm_stream = self._comm_stream

        # Reorder to chunk-interleaved layout (see forward() for rationale).
        trailing = tensor.shape[1:]
        tensor_reordered = (
            tensor.view(tp_size, N, chunk_local, *trailing)
            .transpose(0, 1)
            .contiguous()
            .view(S_full, *trailing)
        )
        chunks = list(tensor_reordered.split(chunk_full, dim=0))

        rs_events: list[torch.cuda.Event] = []
        rs_bufs: list[torch.Tensor] = []

        for chunk in chunks:
            out_dim = chunk.shape[-1]
            rs_buf = torch.empty(
                chunk_local, out_dim, dtype=chunk.dtype, device=tensor.device
            )
            comm_stream.wait_stream(compute_stream)
            ev = self._comm.reduce_scatter_chunk(
                chunk.contiguous(), rs_buf, comm_stream
            )
            rs_events.append(ev)
            rs_bufs.append(rs_buf)

        for ev in rs_events:
            compute_stream.wait_event(ev)

        return torch.cat(rs_bufs, dim=0)

    def _forward_single_chunk(
        self,
        input_full: torch.Tensor,
        weight: torch.Tensor,
        gemm_fn: Callable,
    ) -> torch.Tensor:
        """Degenerate: single GEMM then single RS."""
        tp_size = self._comm.tp_size
        S_full = input_full.shape[0]
        S_local = S_full // tp_size

        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=input_full.device)

        compute_stream = torch.cuda.current_stream(input_full.device)
        comm_stream = self._comm_stream

        gemm_out = gemm_fn(input_full, weight, None)
        out_dim = gemm_out.shape[-1]
        rs_buf = torch.empty(
            S_local, out_dim, dtype=gemm_out.dtype, device=input_full.device
        )
        comm_stream.wait_stream(compute_stream)
        ev = self._comm.reduce_scatter_chunk(gemm_out, rs_buf, comm_stream)
        compute_stream.wait_event(ev)
        return rs_buf


# ---------------------------------------------------------------------------
# Autograd functions for pipelined comm-GEMM overlap
# ---------------------------------------------------------------------------


class _PipelinedGatherForSP(torch.autograd.Function):
    """Autograd-aware pipelined allgather for sequence-parallel layout.

    FWD: ``[S/TP, H] → [S, H]`` via pipelined allgather (identity GEMM).
    BWD: ``[S, H] → [S/TP, H]`` via pipelined reduce-scatter.

    This wraps only the communication; the GEMM is applied by the caller
    after this function returns.  In the forward direction this means the
    AG and GEMM are **not** overlapped (the overlap version is used when
    calling ``PipelinedAllgatherGemm.forward()`` directly with a real
    ``gemm_fn``).  However, autograd correctness requires the AG and GEMM
    to be separate nodes so that the GEMM's backward can produce
    ``grad_input_gathered`` which this function's backward then scatters.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, pipeline_ag, pipeline_rs_for_bwd):
        ctx.pipeline_rs = pipeline_rs_for_bwd
        # clone() is required: the pipeline engine uses double-buffered
        # workspace.  Without clone, output_chunks alias the buffers and
        # later allgather iterations overwrite earlier entries.
        return pipeline_ag.forward(
            input_,
            weight=None,
            bias=None,
            gemm_fn=lambda inp, w, b: inp.clone(),
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = ctx.pipeline_rs.forward_standalone(grad_output)
        return grad_input, None, None


class _PipelinedScatterForSP(torch.autograd.Function):
    """Autograd-aware pipelined reduce-scatter for sequence-parallel layout.

    FWD: ``[S, O] → [S/TP, O]`` via pipelined reduce-scatter.
    BWD: ``[S/TP, O] → [S, O]`` via pipelined allgather.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, pipeline_rs, pipeline_ag_for_bwd):
        ctx.pipeline_ag = pipeline_ag_for_bwd
        return pipeline_rs.forward_standalone(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # clone() required to avoid buffer aliasing (see _PipelinedGatherForSP)
        grad_input = ctx.pipeline_ag.forward(
            grad_output,
            weight=None,
            bias=None,
            gemm_fn=lambda inp, w, b: inp.clone(),
        )
        return grad_input, None, None


def pipelined_gather_for_sp(
    input_: torch.Tensor,
    pipeline_ag: PipelinedAllgatherGemm,
    pipeline_rs: PipelinedGemmReduceScatter,
) -> torch.Tensor:
    """Autograd-aware pipelined allgather along dim 0 for SP layout.

    Forward gathers ``[S/TP, H] → [S, H]``.
    Backward scatters ``[S, H] → [S/TP, H]``.
    """
    return _PipelinedGatherForSP.apply(input_, pipeline_ag, pipeline_rs)


def pipelined_scatter_for_sp(
    input_: torch.Tensor,
    pipeline_rs: PipelinedGemmReduceScatter,
    pipeline_ag: PipelinedAllgatherGemm,
) -> torch.Tensor:
    """Autograd-aware pipelined reduce-scatter along dim 0 for SP layout.

    Forward scatters ``[S, O] → [S/TP, O]``.
    Backward gathers ``[S/TP, O] → [S, O]``.
    """
    return _PipelinedScatterForSP.apply(input_, pipeline_rs, pipeline_ag)


class _FusedColumnParallelForward(torch.autograd.Function):
    """Fused AG+GEMM forward / dGEMM+RS backward for column-parallel.

    Forward: PipelinedAllgatherGemm with real gemm_fn (comm-GEMM overlap).
    Backward: PipelinedGemmReduceScatter with dgrad as gemm_fn (dgrad-RS overlap),
              plus deferred wgrad with AG for input reconstitution.

    Supports both BF16 (scaling_type="none") and FP8 modes. In FP8 mode,
    weight is quantized once before the pipeline loop; activation chunks
    are quantized per-chunk inside ``gemm_fn``.
    """

    @staticmethod
    def forward(
        ctx,
        input_local,
        weight,
        bias,
        pipeline_ag,
        pipeline_rs,
        weight_ref,
        gaf,
        delay_wgrad,
        deferred_wgrad,
        has_bias,
        scaling_manager,
        scaling_type,
        fp8_dtype,
        block_size,
    ):
        if scaling_type != "none":
            weight_2d = weight.reshape(-1, weight.shape[-1])
            weight_desc = quantize_input(
                weight_2d,
                scaling_type,
                fp8_dtype,
                block_size,
                manager=scaling_manager,
                tensor_id="col_weight",
            )

            def gemm_fn(inp, w, b):
                inp_2d = inp.reshape(-1, inp.shape[-1])
                inp_desc = quantize_input(
                    inp_2d,
                    scaling_type,
                    fp8_dtype,
                    block_size,
                    manager=scaling_manager,
                    tensor_id="col_act",
                )
                return dispatch_gemm(
                    inp_desc,
                    weight_desc,
                    scaling_type=scaling_type,
                    bias=b,
                )

        else:
            weight_desc = None

            def gemm_fn(inp, w, b):
                return dispatch_gemm(inp, w, None, None, "none", bias=b)

        output = pipeline_ag.forward(input_local, weight, bias, gemm_fn)
        ctx.save_for_backward(input_local, weight)
        ctx.pipeline_ag = pipeline_ag
        ctx.pipeline_rs = pipeline_rs
        ctx.weight_ref = weight_ref
        ctx.gaf = gaf
        ctx.delay_wgrad = delay_wgrad
        ctx.deferred_wgrad = deferred_wgrad
        ctx.has_bias = has_bias
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.weight_fp8 = weight_desc.data if weight_desc is not None else None
        ctx.weight_scale = weight_desc.scale if weight_desc is not None else None
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input_local, weight = ctx.saved_tensors
        scaling_type = ctx.scaling_type

        if scaling_type != "none" and ctx.weight_fp8 is not None:
            _weight_fp8 = ctx.weight_fp8
            _weight_scale = ctx.weight_scale

            def dgrad_fn(chunk, w, b):
                return compute_dgrad_fp8(
                    chunk,
                    _weight_fp8,
                    _weight_scale,
                    ctx.scaling_type,
                    ctx.scaling_manager,
                    ctx.fp8_dtype,
                    ctx.block_size,
                )

        else:

            def dgrad_fn(chunk, w, b):
                return compute_dgrad_bf16(chunk, w)

        grad_input = ctx.pipeline_rs.forward(grad_output, weight, dgrad_fn)

        if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
            _input_local = input_local
            _grad_output = grad_output
            _pipeline_ag = ctx.pipeline_ag
            _weight_ref = ctx.weight_ref
            _gaf = ctx.gaf

            def _deferred():
                input_gathered = _pipeline_ag.forward(
                    _input_local,
                    weight=None,
                    bias=None,
                    gemm_fn=lambda inp, w, b: inp.clone(),
                )
                closure = make_wgrad_closure_bf16(
                    _grad_output,
                    input_gathered,
                    _weight_ref,
                    _gaf,
                )
                closure()

            ctx.deferred_wgrad.defer(_deferred)
            grad_weight = None
        else:
            input_gathered = ctx.pipeline_ag.forward(
                input_local,
                weight=None,
                bias=None,
                gemm_fn=lambda inp, w, b: inp.clone(),
            )
            grad_weight = compute_wgrad_bf16(grad_output, input_gathered)
            if ctx.gaf and hasattr(ctx.weight_ref, "main_grad"):
                ctx.weight_ref.main_grad.add_(grad_weight)
                grad_weight = None

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        # 14 forward args → 14 backward returns
        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FusedRowParallelForward(torch.autograd.Function):
    """Fused GEMM+RS forward / AG+dGEMM backward for row-parallel.

    Forward: PipelinedGemmReduceScatter with real gemm_fn (GEMM-RS overlap).
    Backward: PipelinedAllgatherGemm with dgrad as gemm_fn (AG-dgrad overlap),
              plus deferred wgrad with AG for grad_output reconstitution.

    Supports both BF16 (scaling_type="none") and FP8 modes.
    """

    @staticmethod
    def forward(
        ctx,
        input_parallel,
        weight,
        pipeline_rs,
        pipeline_ag,
        weight_ref,
        gaf,
        delay_wgrad,
        deferred_wgrad,
        scaling_manager,
        scaling_type,
        fp8_dtype,
        block_size,
    ):
        if scaling_type != "none":
            weight_2d = weight.reshape(-1, weight.shape[-1])
            weight_desc = quantize_input(
                weight_2d,
                scaling_type,
                fp8_dtype,
                block_size,
                manager=scaling_manager,
                tensor_id="row_weight",
            )

            def gemm_fn(inp, w, b):
                inp_2d = inp.reshape(-1, inp.shape[-1])
                inp_desc = quantize_input(
                    inp_2d,
                    scaling_type,
                    fp8_dtype,
                    block_size,
                    manager=scaling_manager,
                    tensor_id="row_act",
                )
                return dispatch_gemm(
                    inp_desc,
                    weight_desc,
                    scaling_type=scaling_type,
                    bias=b,
                )

        else:
            weight_desc = None

            def gemm_fn(inp, w, b):
                return dispatch_gemm(inp, w, None, None, "none", bias=b)

        output = pipeline_rs.forward(input_parallel, weight, gemm_fn)
        ctx.save_for_backward(input_parallel, weight)
        ctx.pipeline_ag = pipeline_ag
        ctx.pipeline_rs = pipeline_rs
        ctx.weight_ref = weight_ref
        ctx.gaf = gaf
        ctx.delay_wgrad = delay_wgrad
        ctx.deferred_wgrad = deferred_wgrad
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.weight_fp8 = weight_desc.data if weight_desc is not None else None
        ctx.weight_scale = weight_desc.scale if weight_desc is not None else None
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input_parallel, weight = ctx.saved_tensors
        scaling_type = ctx.scaling_type

        if scaling_type != "none" and ctx.weight_fp8 is not None:
            _weight_fp8 = ctx.weight_fp8
            _weight_scale = ctx.weight_scale

            def dgrad_fn(chunk, w, b):
                return compute_dgrad_fp8(
                    chunk,
                    _weight_fp8,
                    _weight_scale,
                    ctx.scaling_type,
                    ctx.scaling_manager,
                    ctx.fp8_dtype,
                    ctx.block_size,
                )

        else:

            def dgrad_fn(chunk, w, b):
                return compute_dgrad_bf16(chunk, weight)

        grad_input = ctx.pipeline_ag.forward(
            grad_output,
            weight=None,
            bias=None,
            gemm_fn=dgrad_fn,
        )

        if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
            _grad_output = grad_output
            _input_parallel = input_parallel
            _pipeline_ag = ctx.pipeline_ag
            _weight_ref = ctx.weight_ref
            _gaf = ctx.gaf

            def _deferred():
                grad_gathered = _pipeline_ag.forward(
                    _grad_output,
                    weight=None,
                    bias=None,
                    gemm_fn=lambda inp, w, b: inp.clone(),
                )
                closure = make_wgrad_closure_bf16(
                    grad_gathered,
                    _input_parallel,
                    _weight_ref,
                    _gaf,
                )
                closure()

            ctx.deferred_wgrad.defer(_deferred)
            grad_weight = None
        else:
            grad_gathered = ctx.pipeline_ag.forward(
                grad_output,
                weight=None,
                bias=None,
                gemm_fn=lambda inp, w, b: inp.clone(),
            )
            grad_weight = compute_wgrad_bf16(grad_gathered, input_parallel)
            if ctx.gaf and hasattr(ctx.weight_ref, "main_grad"):
                ctx.weight_ref.main_grad.add_(grad_weight)
                grad_weight = None

        # 12 forward args → 12 backward returns
        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_column_parallel_forward(
    input_local,
    weight,
    bias,
    pipeline_ag,
    pipeline_rs,
    weight_ref,
    gaf,
    delay_wgrad,
    deferred_wgrad,
    has_bias,
    scaling_manager=None,
    scaling_type="none",
    fp8_dtype=None,
    block_size=128,
):
    return _FusedColumnParallelForward.apply(
        input_local,
        weight,
        bias,
        pipeline_ag,
        pipeline_rs,
        weight_ref,
        gaf,
        delay_wgrad,
        deferred_wgrad,
        has_bias,
        scaling_manager,
        scaling_type,
        fp8_dtype,
        block_size,
    )


def fused_row_parallel_forward(
    input_parallel,
    weight,
    pipeline_rs,
    pipeline_ag,
    weight_ref,
    gaf,
    delay_wgrad,
    deferred_wgrad,
    scaling_manager=None,
    scaling_type="none",
    fp8_dtype=None,
    block_size=128,
):
    return _FusedRowParallelForward.apply(
        input_parallel,
        weight,
        pipeline_rs,
        pipeline_ag,
        weight_ref,
        gaf,
        delay_wgrad,
        deferred_wgrad,
        scaling_manager,
        scaling_type,
        fp8_dtype,
        block_size,
    )


try:
    from atomrl.lumen.ops.quantize.linear import _mark_allow_in_graph

    _mark_allow_in_graph(_FusedColumnParallelForward)
    _mark_allow_in_graph(_FusedRowParallelForward)
except ImportError:
    pass
