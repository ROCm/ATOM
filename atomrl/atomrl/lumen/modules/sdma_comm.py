# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""SDMA-based TP communication primitives for Lumen modules.

When ``--use-sdma`` is enabled, these autograd functions replace Megatron's
``torch.distributed``-based mappings with mori's SDMA engine.  The SDMA
engine offloads data movement to a dedicated hardware copy engine, freeing
the compute units for overlapping work.

Supported primitives (each mirrors the corresponding Megatron mapping):

- :func:`sdma_gather_from_sequence_parallel_region`
  (fwd: all-gather dim 0, bwd: reduce-scatter dim 0)
- :func:`sdma_reduce_scatter_to_sequence_parallel_region`
  (fwd: reduce-scatter dim 0, bwd: all-gather dim 0)
- :func:`sdma_reduce_from_tensor_model_parallel_region`
  (fwd: all-reduce SUM, bwd: identity)
- :func:`sdma_gather_from_tensor_model_parallel_region`
  (fwd: all-gather last dim, bwd: split last dim)
- :func:`sdma_copy_to_tensor_model_parallel_region`
  (fwd: identity, bwd: all-reduce SUM)
"""

import logging
from math import prod
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDMA TP Context — manages mori shmem init for the TP process group
# ---------------------------------------------------------------------------


class SdmaTpContext:
    """Process-level singleton that registers the TP group with mori shmem."""

    _instance: Optional["SdmaTpContext"] = None

    _group_registered: bool = False

    def __init__(self, tp_group: torch.distributed.ProcessGroup):
        from mori import shmem

        group_name = "lumen_tp"
        if not SdmaTpContext._group_registered:
            torch._C._distributed_c10d._register_process_group(group_name, tp_group)
            SdmaTpContext._group_registered = True
        shmem.shmem_torch_process_group_init(group_name)
        self.my_pe: int = shmem.shmem_mype()
        self.npes: int = shmem.shmem_npes()
        self._tp_group = tp_group
        logger.info("SdmaTpContext: PE %d / %d", self.my_pe, self.npes)

    @classmethod
    def get(cls, tp_group: torch.distributed.ProcessGroup) -> "SdmaTpContext":
        if cls._instance is None:
            cls._instance = cls(tp_group)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._group_registered = False


# ---------------------------------------------------------------------------
# Cached SDMA handle wrappers
# ---------------------------------------------------------------------------


class _TpSdmaAllgather:
    """Cached ``AllgatherSdma`` handle that works with any dtype by
    reinterpreting tensor bytes as float32 for the SDMA transport.
    """

    def __init__(self, ctx: SdmaTpContext):
        self._ctx = ctx
        self._handle = None
        self._capacity_f32: int = 0
        self._gathered_buf: torch.Tensor | None = None

    def _ensure(self, n_f32: int) -> None:
        if self._handle is not None and n_f32 <= self._capacity_f32:
            return
        from mori.ccl import AllgatherSdma

        elem_size = 4  # float32
        padding_elems = 64
        self._handle = AllgatherSdma(
            self._ctx.my_pe,
            self._ctx.npes,
            input_buffer_size=(n_f32 + padding_elems) * elem_size,
            output_buffer_size=self._ctx.npes * (n_f32 + padding_elems) * elem_size,
            copy_output_to_user=True,
        )
        self._capacity_f32 = n_f32
        logger.debug(
            "_TpSdmaAllgather: (re)alloc for %d f32 elems (PE %d/%d)",
            n_f32,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flat all-gather. Returns ``(npes, n_elems)`` in *tensor.dtype*."""
        tensor = tensor.contiguous()
        orig_dtype = tensor.dtype
        orig_numel = tensor.numel()
        flat = tensor.view(-1)

        if orig_dtype != torch.float32:
            flat_f32 = flat.view(torch.float32)
        else:
            flat_f32 = flat
        n_f32 = flat_f32.numel()

        self._ensure(n_f32)
        npes = self._ctx.npes

        if (
            self._gathered_buf is None
            or self._gathered_buf.numel() < n_f32 * npes
            or self._gathered_buf.device != tensor.device
        ):
            self._gathered_buf = torch.empty(
                n_f32 * npes,
                dtype=torch.float32,
                device=tensor.device,
            )

        stream = torch.cuda.current_stream(tensor.device)
        self._handle(flat_f32, self._gathered_buf, n_f32, stream)
        stream.synchronize()

        gathered = self._gathered_buf[: n_f32 * npes]
        if orig_dtype != torch.float32:
            gathered = gathered.view(orig_dtype)

        return gathered[: orig_numel * npes].reshape(npes, orig_numel)

    def start_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async all-gather. Call :meth:`wait_async` to complete."""
        tensor = tensor.contiguous()
        orig_dtype = tensor.dtype
        orig_numel = tensor.numel()
        flat = tensor.view(-1)

        if orig_dtype != torch.float32:
            flat_f32 = flat.view(torch.float32)
        else:
            flat_f32 = flat
        n_f32 = flat_f32.numel()

        self._ensure(n_f32)
        npes = self._ctx.npes

        if (
            self._gathered_buf is None
            or self._gathered_buf.numel() < n_f32 * npes
            or self._gathered_buf.device != tensor.device
        ):
            self._gathered_buf = torch.empty(
                n_f32 * npes,
                dtype=torch.float32,
                device=tensor.device,
            )

        s = stream if stream is not None else torch.cuda.current_stream(tensor.device)
        self._async_meta = {
            "orig_dtype": orig_dtype,
            "orig_numel": orig_numel,
            "n_f32": n_f32,
            "npes": npes,
        }
        return self._handle.start_async(flat_f32, self._gathered_buf, n_f32, s)

    def wait_async(self, stream=None) -> torch.Tensor:
        """Wait for async all-gather and return ``(npes, n_elems)`` in original dtype."""
        meta = getattr(self, "_async_meta", None)
        if meta is None:
            raise RuntimeError("wait_async called without prior start_async")
        s = (
            stream
            if stream is not None
            else torch.cuda.current_stream(self._gathered_buf.device)
        )
        self._handle.wait_async(s)
        gathered = self._gathered_buf[: meta["n_f32"] * meta["npes"]]
        if meta["orig_dtype"] != torch.float32:
            gathered = gathered.view(meta["orig_dtype"])
        result = gathered[: meta["orig_numel"] * meta["npes"]].reshape(
            meta["npes"], meta["orig_numel"]
        )
        del self._async_meta
        return result


class _TpSdmaAllreduce:
    """Cached ``AllreduceSdma`` handle (SUM) for a specific dtype.

    The mori SDMA allreduce kernel natively supports uint32, int32,
    float16, and bfloat16.  When ``dtype`` is ``torch.float32``, this
    wrapper transparently casts to bfloat16 for the reduction and casts
    the result back.
    """

    _NATIVE_DTYPES = frozenset(
        {torch.uint32, torch.int32, torch.float16, torch.bfloat16}
    )

    def __init__(self, ctx: SdmaTpContext, dtype: torch.dtype = torch.float32):
        if dtype != torch.float32 and dtype not in self._NATIVE_DTYPES:
            raise ValueError(
                f"_TpSdmaAllreduce: unsupported dtype {dtype}. "
                f"Supported: float32 (via bf16), {sorted(str(d) for d in self._NATIVE_DTYPES)}"
            )
        self._ctx = ctx
        self._user_dtype = dtype
        self._wire_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        self._handle = None
        self._capacity: int = 0
        self._out_buf: torch.Tensor | None = None
        self._async_output_buf: torch.Tensor | None = None

    def _ensure(self, n_elems: int) -> None:
        if self._handle is not None and n_elems <= self._capacity:
            return
        from mori.ccl import AllreduceSdma

        elem_size = torch.tensor([], dtype=self._wire_dtype).element_size()
        npes = self._ctx.npes
        self._handle = AllreduceSdma(
            self._ctx.my_pe,
            npes,
            input_buffer_size=n_elems * elem_size,
            output_buffer_size=npes * (n_elems // npes + 64) * elem_size,
            copy_output_to_user=True,
            dtype=self._wire_dtype,
        )
        self._capacity = n_elems
        self._out_buf = None
        self._async_output_buf = None
        logger.debug(
            "_TpSdmaAllreduce: (re)alloc for %d elems, "
            "user_dtype=%s wire_dtype=%s (PE %d/%d)",
            n_elems,
            self._user_dtype,
            self._wire_dtype,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    @property
    def _needs_cast(self) -> bool:
        return self._user_dtype != self._wire_dtype

    def _get_out_buf(self, template: torch.Tensor) -> torch.Tensor:
        """Return a cached flat output buffer matching *template*'s size/dtype/device."""
        n = template.numel()
        if (
            self._out_buf is None
            or self._out_buf.numel() < n
            or self._out_buf.device != template.device
            or self._out_buf.dtype != template.dtype
        ):
            self._out_buf = torch.empty_like(template)
        return self._out_buf[:n]

    def _ensure_async_output(self, n_elems: int, device: torch.device) -> torch.Tensor:
        """Return a flat output buffer for async allreduce, reusing when possible."""
        if (
            self._async_output_buf is None
            or self._async_output_buf.numel() < n_elems
            or self._async_output_buf.device != device
        ):
            self._async_output_buf = torch.empty(
                n_elems,
                dtype=self._wire_dtype,
                device=device,
            )
        return self._async_output_buf[:n_elems]

    def inplace(self, tensor: torch.Tensor) -> None:
        flat = tensor.contiguous().reshape(-1)
        n = flat.numel()
        self._ensure(n)
        stream = torch.cuda.current_stream(tensor.device)
        if self._needs_cast:
            buf = flat.to(self._wire_dtype)
            out = self._get_out_buf(buf)
            self._handle(buf, out, n, stream)
            tensor.copy_(out.to(self._user_dtype).reshape(tensor.shape))
        else:
            out = self._get_out_buf(flat)
            self._handle(flat, out, n, stream)
            tensor.copy_(out.reshape(tensor.shape))
        stream.synchronize()

    def start_async_inplace(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async in-place allreduce.  Caller must call :meth:`wait_async`.

        Mori's ``AllreduceSdma`` only exposes an out-of-place ``start_async``,
        so we manage an output buffer internally and copy the result back in
        :meth:`wait_async`.
        """
        flat = tensor.contiguous().reshape(-1)
        self._ensure(flat.numel())
        s = stream if stream is not None else torch.cuda.current_stream(tensor.device)
        self._async_orig_shape = tensor.shape
        self._async_stream = s
        if self._needs_cast:
            wire_flat = flat.to(self._wire_dtype)
            self._async_buf = wire_flat
            self._async_tensor = tensor
            out = self._ensure_async_output(wire_flat.numel(), wire_flat.device)
            started = self._handle.start_async(wire_flat, out, wire_flat.numel(), s)
            if not started:
                self._clear_async_state()
            return started
        self._async_flat = flat
        self._async_tensor = tensor
        out = self._ensure_async_output(flat.numel(), flat.device)
        started = self._handle.start_async(flat, out, flat.numel(), s)
        if not started:
            self._clear_async_state()
        return started

    def wait_async(self, stream=None) -> None:
        """Wait for a previously started async allreduce and copy result back."""
        s = stream if stream is not None else getattr(self, "_async_stream", None)
        try:
            duration = self._handle.wait_async(s)
        except Exception:
            self._clear_async_state()
            raise
        if duration is not None and duration < 0:
            self._clear_async_state()
            raise RuntimeError("AllreduceSdma wait_async failed")
        out = self._async_output_buf
        try:
            if self._needs_cast and hasattr(self, "_async_buf"):
                n = self._async_buf.numel()
                self._async_tensor.copy_(
                    out[:n].to(self._user_dtype).reshape(self._async_orig_shape)
                )
            elif hasattr(self, "_async_flat"):
                n = self._async_flat.numel()
                self._async_tensor.copy_(out[:n].reshape(self._async_orig_shape))
            else:
                raise RuntimeError(
                    "AllreduceSdma wait_async missing async tensor state"
                )
        finally:
            self._clear_async_state()

    def _clear_async_state(self) -> None:
        for name in (
            "_async_buf",
            "_async_flat",
            "_async_tensor",
            "_async_orig_shape",
            "_async_stream",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def reset_flags(self) -> None:
        """Reset SDMA synchronization flags on the underlying handle."""
        if self._handle is not None:
            self._handle.reset_flags()


# ---------------------------------------------------------------------------
# SdmaTpComm — high-level TP communication manager
# ---------------------------------------------------------------------------


class SdmaTpComm:
    """Manages SDMA handles for TP communication.

    Provides high-level methods that match the semantics of Megatron's
    TP mapping primitives but route through mori SDMA.

    Use :meth:`get` to obtain the (cached) singleton for a given TP group.

    Args:
        tp_group: TP process group.
        rs_chunk_method: Backend for chunk-level reduce-scatter.
            ``"allreduce"`` uses AllreduceSdma + local shard extraction.
            ``"nccl"`` falls back to ``torch.distributed.reduce_scatter_tensor``.
    """

    _instance: Optional["SdmaTpComm"] = None

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        rs_chunk_method: str = "allreduce",
    ):
        self._ctx = SdmaTpContext.get(tp_group)
        self._tp_group = tp_group
        self._ag = _TpSdmaAllgather(self._ctx)
        self._ag_chunk = _TpSdmaAllgather(self._ctx)
        self._ar_handles: dict = {}
        self._rs_output_buf: torch.Tensor | None = None
        self._rs_chunk_method = rs_chunk_method
        self._rs_chunk_ar: _TpSdmaAllreduce | None = None
        self._rs_chunk_buf: torch.Tensor | None = None

    @property
    def npes(self) -> int:
        return self._ctx.npes

    @property
    def my_pe(self) -> int:
        return self._ctx.my_pe

    def _get_ar(self, dtype: torch.dtype) -> _TpSdmaAllreduce:
        if dtype not in self._ar_handles:
            self._ar_handles[dtype] = _TpSdmaAllreduce(self._ctx, dtype)
        return self._ar_handles[dtype]

    def reset_allreduce_flags(self) -> None:
        """Reset SDMA flags on all cached allreduce handles.

        Call between switching from synchronous to asynchronous allreduce
        on the same handle to avoid stale flag state.
        """
        for ar in self._ar_handles.values():
            ar.reset_flags()

    # -- Primitives --

    def allgather_dim0(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather along dim 0. ``[S/TP, ...] → [S, ...]``."""
        shape = tensor.shape
        gathered = self._ag(tensor)  # (npes, numel_per_pe)
        return gathered.reshape(self.npes * shape[0], *shape[1:])

    def allgather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather along last dim. ``[*, D/TP] → [*, D]``."""
        shape = tensor.shape
        batch_shape = shape[:-1]
        d_local = shape[-1]
        gathered = self._ag(tensor)  # (npes, numel_per_pe)
        reshaped = gathered.reshape(self.npes, *batch_shape, d_local)
        return reshaped.movedim(0, -2).reshape(*batch_shape, self.npes * d_local)

    def allreduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Out-of-place all-reduce SUM."""
        result = tensor.clone()
        ar = self._get_ar(tensor.dtype)
        ar.inplace(result)
        return result

    def allreduce_sum_inplace(self, tensor: torch.Tensor) -> None:
        """In-place all-reduce SUM."""
        ar = self._get_ar(tensor.dtype)
        ar.inplace(tensor)

    def reduce_scatter_dim0(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce-scatter along dim 0. ``[S, ...] → [S/TP, ...]``.

        Implemented as all-reduce(SUM) + local chunk extraction.
        """
        reduced = self.allreduce_sum(tensor)
        chunk_size = tensor.shape[0] // self.npes
        start = self.my_pe * chunk_size
        return reduced[start : start + chunk_size].contiguous()

    # -- Async primitives (for comm-GEMM overlap) --

    def allgather_dim0_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async all-gather along dim 0. Call :meth:`wait_allgather_dim0` to complete."""
        shape = tensor.shape
        self._ag_async_shape = shape
        return self._ag.start_async(tensor, stream)

    def wait_allgather_dim0(self, stream=None) -> torch.Tensor:
        """Wait for async all-gather and return ``[S, ...]``."""
        gathered = self._ag.wait_async(stream)  # (npes, numel_per_pe)
        shape = getattr(self, "_ag_async_shape", None)
        if shape is None:
            raise RuntimeError(
                "wait_allgather_dim0 called without prior allgather_dim0_async"
            )
        del self._ag_async_shape
        return gathered.reshape(self.npes * shape[0], *shape[1:])

    def allreduce_sum_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async in-place all-reduce SUM.

        Returns ``True`` on successful launch and raises ``RuntimeError`` if the
        underlying async start fails.
        """
        tensor = tensor.contiguous()
        ar = self._get_ar(tensor.dtype)
        started = ar.start_async_inplace(tensor, stream)
        if not started:
            raise RuntimeError(
                f"AllreduceSdma async start failed for dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
        self._ar_async_dtype = tensor.dtype
        return True

    def wait_allreduce_sum(self, stream=None) -> None:
        """Wait for async all-reduce to complete."""
        dtype = getattr(self, "_ar_async_dtype", None)
        if dtype is None:
            raise RuntimeError(
                "wait_allreduce_sum called without prior allreduce_sum_async"
            )
        ar = self._get_ar(dtype)
        try:
            ar.wait_async(stream)
        finally:
            del self._ar_async_dtype

    # -- Chunk-level primitives (for pipelined comm-GEMM overlap) --

    def allgather_dim0_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        """All-gather a single chunk for pipelined overlap.

        Args:
            input_chunk: ``[S_chunk, H]`` — local chunk from one PE.
            output_buf: ``[S_chunk * npes, H]`` — pre-allocated output.
            stream: CUDA stream on which to run the collective.

        Returns:
            CUDA event recorded on *stream* after the operation completes.
        """
        input_chunk = input_chunk.contiguous()
        orig_dtype = input_chunk.dtype
        flat_in = input_chunk.view(-1)
        if orig_dtype != torch.float32:
            flat_in = flat_in.view(torch.float32)
        n_f32 = flat_in.numel()
        npes = self._ctx.npes

        self._ag_chunk._ensure(n_f32)

        flat_out = output_buf.view(-1)
        if orig_dtype != torch.float32:
            flat_out = flat_out.view(torch.float32)

        self._ag_chunk._handle(flat_in, flat_out[: n_f32 * npes], n_f32, stream)
        return stream.record_event()

    def reduce_scatter_dim0_chunk(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        """Reduce-scatter a single chunk for pipelined overlap.

        .. note::

            When ``rs_chunk_method="allreduce"``, the result is written to an
            internal buffer; *input_chunk* is not modified.

        Args:
            input_chunk: ``[S_chunk, ...]`` — chunk to reduce-scatter.
            output_buf: ``[S_chunk // npes, ...]`` — pre-allocated output.
            stream: CUDA stream on which to run the collective.

        Returns:
            CUDA event recorded on *stream* after the operation completes.
        """
        if self._rs_chunk_method == "nccl":
            return self._rs_chunk_nccl(input_chunk, output_buf, stream)
        return self._rs_chunk_allreduce(input_chunk, output_buf, stream)

    def _rs_chunk_nccl(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        with torch.cuda.stream(stream):
            torch.distributed.reduce_scatter_tensor(
                output_buf,
                input_chunk.contiguous(),
                group=self._tp_group,
            )
        return stream.record_event()

    def _rs_chunk_allreduce(
        self,
        input_chunk: torch.Tensor,
        output_buf: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        """Reduce-scatter via AllreduceSdma + local shard extraction."""
        input_chunk = input_chunk.contiguous()
        dtype = input_chunk.dtype
        npes = self._ctx.npes

        if self._rs_chunk_ar is None:
            self._rs_chunk_ar = _TpSdmaAllreduce(self._ctx, dtype)

        ar = self._rs_chunk_ar
        flat = input_chunk.view(-1)
        wire_dtype = ar._wire_dtype
        if ar._needs_cast:
            flat = flat.to(wire_dtype)
        n_elems = flat.numel()
        ar._ensure(n_elems)

        if (
            self._rs_chunk_buf is None
            or self._rs_chunk_buf.numel() < n_elems
            or self._rs_chunk_buf.device != flat.device
            or self._rs_chunk_buf.dtype != wire_dtype
        ):
            self._rs_chunk_buf = torch.empty(
                n_elems,
                dtype=wire_dtype,
                device=flat.device,
            )

        out_flat = self._rs_chunk_buf[:n_elems]
        ar._handle(flat, out_flat, n_elems, stream)

        reduced = out_flat.view(input_chunk.shape)
        chunk_size = input_chunk.shape[0] // npes
        start = self._ctx.my_pe * chunk_size
        with torch.cuda.stream(stream):
            shard = reduced[start : start + chunk_size]
            if ar._needs_cast:
                shard = shard.to(dtype)
            output_buf.copy_(shard)
        return stream.record_event()

    def reduce_scatter_dim0_async(self, tensor: torch.Tensor, stream=None) -> bool:
        """Start async reduce-scatter along dim 0.

        Call :meth:`wait_reduce_scatter_dim0` to complete.  The caller
        must keep *tensor* alive until wait returns (mori holds a raw
        pointer to the device buffer).
        """
        tensor = tensor.contiguous()
        ar = self._get_ar(tensor.dtype)
        wire_dtype = ar._wire_dtype
        wire_tensor = tensor.to(wire_dtype) if ar._needs_cast else tensor
        flat_wire = wire_tensor.reshape(-1)
        n_elems = flat_wire.numel()
        ar._ensure(n_elems)
        if (
            self._rs_output_buf is None
            or self._rs_output_buf.numel() < n_elems
            or self._rs_output_buf.device != flat_wire.device
            or self._rs_output_buf.dtype != wire_dtype
        ):
            self._rs_output_buf = torch.empty(
                n_elems, dtype=wire_dtype, device=flat_wire.device
            )
        self._rs_async_shape = tensor.shape
        self._rs_user_dtype = tensor.dtype
        self._rs_input_buf = wire_tensor
        return ar._handle.start_async(flat_wire, self._rs_output_buf, n_elems, stream)

    def wait_reduce_scatter_dim0(self, stream=None) -> torch.Tensor:
        """Wait for async reduce-scatter and return ``[S/TP, ...]``."""
        shape = getattr(self, "_rs_async_shape", None)
        if shape is None:
            raise RuntimeError(
                "wait_reduce_scatter_dim0 called without prior reduce_scatter_dim0_async"
            )
        user_dtype = self._rs_user_dtype
        ar = self._get_ar(user_dtype)
        ar._handle.wait_async(stream)
        full = self._rs_output_buf[: prod(shape)].reshape(shape)
        chunk_size = shape[0] // self.npes
        start = self.my_pe * chunk_size
        result = full[start : start + chunk_size].contiguous()
        if ar._needs_cast:
            result = result.to(user_dtype)
        del self._rs_async_shape, self._rs_user_dtype, self._rs_input_buf
        return result

    @classmethod
    def get(cls, tp_group: torch.distributed.ProcessGroup) -> "SdmaTpComm":
        if cls._instance is None:
            cls._instance = cls(tp_group)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------


class _SdmaGatherFromSequenceParallelRegion(torch.autograd.Function):
    """FWD: all-gather dim 0 via SDMA.  BWD: reduce-scatter dim 0."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        return comm.allgather_dim0(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.comm.reduce_scatter_dim0(grad_output), None


class _SdmaReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """FWD: reduce-scatter dim 0 via SDMA.  BWD: all-gather dim 0."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        return comm.reduce_scatter_dim0(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.comm.allgather_dim0(grad_output), None


class _SdmaReduceFromTensorModelParallelRegion(torch.autograd.Function):
    """FWD: all-reduce SUM via SDMA.  BWD: identity (pass-through)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        return comm.allreduce_sum(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _SdmaGatherFromTensorModelParallelRegion(torch.autograd.Function):
    """FWD: all-gather last dim via SDMA.  BWD: split last dim (local chunk)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        ctx.d_local = input_.shape[-1]
        return comm.allgather_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        d_local = ctx.d_local
        start = ctx.comm.my_pe * d_local
        return grad_output[..., start : start + d_local].contiguous(), None


class _SdmaCopyToTensorModelParallelRegion(torch.autograd.Function):
    """FWD: identity (pass-through).  BWD: all-reduce SUM via SDMA."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm: SdmaTpComm):
        ctx.comm = comm
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.comm.allreduce_sum(grad_output), None


# ---------------------------------------------------------------------------
# Public API — SDMA-accelerated Megatron mapping functions
# ---------------------------------------------------------------------------


def sdma_gather_from_sequence_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaGatherFromSequenceParallelRegion.apply(input_, comm)


def sdma_reduce_scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaReduceScatterToSequenceParallelRegion.apply(input_, comm)


def sdma_reduce_from_tensor_model_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaReduceFromTensorModelParallelRegion.apply(input_, comm)


def sdma_gather_from_tensor_model_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaGatherFromTensorModelParallelRegion.apply(input_, comm)


def sdma_copy_to_tensor_model_parallel_region(
    input_: torch.Tensor,
    comm: SdmaTpComm,
) -> torch.Tensor:
    return _SdmaCopyToTensorModelParallelRegion.apply(input_, comm)
