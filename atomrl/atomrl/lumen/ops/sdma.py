# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen SDMA collective wrappers over ``mori.ccl``.

Provides thin, lazy-initialized wrappers around mori's SDMA-based collective
primitives (AllreduceSdma, AllgatherSdma) so that the rest of Lumen never
imports ``mori`` directly.  This module also centralizes shmem lifecycle
management (``shmem_torch_process_group_init`` / ``shmem_finalize``) so that
the one-time-per-process initialization is done exactly once regardless of
how many call-sites use SDMA.

Availability check::

    from atomrl.lumen.ops.sdma import is_sdma_available
    if is_sdma_available():
        from atomrl.lumen.ops.sdma import sdma_allgather_max, SdmaContext
"""

import functools
import glob
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def _has_kfd_sdma_engines() -> bool:
    """Check KFD topology for SDMA XGMI engines.

    The anvil transport layer creates SDMA queues via ``hsaKmtCreateQueueExt``
    using XGMI engine IDs derived from a MI300X-specific OAM map.  If the
    system has no SDMA XGMI engines (e.g. no ``/dev/kfd``, different GPU
    family, or VMs without KFD passthrough), queue creation will ``exit(1)``
    inside C++ and kill the process.

    Returns ``True`` only when at least one KFD topology node reports
    ``num_sdma_xgmi_engines > 0``.
    """
    if not os.path.exists("/dev/kfd"):
        return False

    for props_path in sorted(
        glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
    ):
        try:
            with open(props_path) as f:
                for line in f:
                    if (
                        line.startswith("num_sdma_xgmi_engines")
                        and int(line.split()[-1]) > 0
                    ):
                        return True
        except (OSError, ValueError):
            continue

    return False


@functools.lru_cache(maxsize=1)
def is_sdma_available() -> bool:
    """Return True if mori SDMA primitives can be imported **and** the
    hardware exposes SDMA XGMI engines via the KFD topology.

    The two-level check avoids a fatal ``exit(1)`` from the C++ anvil
    layer when ``hsaKmtCreateQueueExt`` is called on unsupported hardware.
    """
    try:
        import mori.shmem  # noqa: F401
        from mori.ccl import AllgatherSdma, AllreduceSdma  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False

    if not _has_kfd_sdma_engines():
        logger.debug(
            "mori importable but no SDMA XGMI engines found in "
            "/sys/class/kfd/kfd/topology — skipping SDMA"
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Shmem lifecycle (process-level singleton)
# ---------------------------------------------------------------------------


class SdmaContext:
    """Process-level singleton managing mori shmem initialization and PE info.

    Usage::

        ctx = SdmaContext.get()   # lazy init on first call
        print(ctx.my_pe, ctx.npes)
    """

    _instance: Optional["SdmaContext"] = None

    def __init__(self):
        from mori import shmem

        shmem.shmem_torch_process_group_init("default")
        self.my_pe: int = shmem.shmem_mype()
        self.npes: int = shmem.shmem_npes()
        logger.info(
            "SdmaContext initialized: PE %d / %d",
            self.my_pe,
            self.npes,
        )

    @classmethod
    def get(cls) -> "SdmaContext":
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (for testing or process-group re-init)."""
        cls._instance = None


# ---------------------------------------------------------------------------
# AllgatherSdma handle cache
# ---------------------------------------------------------------------------


class SdmaAllgather:
    """Reusable AllgatherSdma handle with automatic buffer management.

    Wraps ``mori.ccl.AllgatherSdma`` and caches the handle so that
    repeated calls with the same (or smaller) tensor size reuse the
    same symmetric-memory buffers.

    Args:
        ctx: A :class:`SdmaContext` (defaults to the process singleton).
    """

    def __init__(self, ctx: SdmaContext | None = None):
        self._ctx = ctx or SdmaContext.get()
        self._handle = None
        self._capacity_elems: int = 0
        self._gathered: torch.Tensor | None = None

    @property
    def npes(self) -> int:
        return self._ctx.npes

    def _ensure_handle(self, n_elems: int) -> None:
        """(Re-)create the AllgatherSdma handle if the current buffer is
        too small for *n_elems* float32 elements per PE.
        """
        if self._handle is not None and n_elems <= self._capacity_elems:
            return

        from mori.ccl import AllgatherSdma

        elem_size = 4  # float32
        input_buf_bytes = n_elems * elem_size
        output_buf_bytes = self._ctx.npes * (n_elems + 64) * elem_size

        self._handle = AllgatherSdma(
            self._ctx.my_pe,
            self._ctx.npes,
            input_buffer_size=input_buf_bytes,
            output_buffer_size=output_buf_bytes,
            copy_output_to_user=True,
        )
        self._capacity_elems = n_elems
        logger.debug(
            "SdmaAllgather: (re-)allocated handle for %d elems " "(PE %d/%d)",
            n_elems,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    def __call__(
        self,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Allgather *tensor* across all PEs.  Returns a 2-D tensor
        ``(npes, n_elems)`` on the same device as *tensor*.
        """
        tensor = tensor.contiguous()
        n_elems = tensor.numel()
        npes = self._ctx.npes
        self._ensure_handle(n_elems)

        if (
            self._gathered is None
            or self._gathered.numel() < n_elems * npes
            or self._gathered.device != tensor.device
        ):
            self._gathered = torch.empty(
                n_elems * npes,
                dtype=torch.float32,
                device=tensor.device,
            )

        if stream is None:
            stream = torch.cuda.current_stream(tensor.device)
        self._handle(tensor, self._gathered, n_elems, stream)
        stream.synchronize()
        return self._gathered[: n_elems * npes].reshape(npes, n_elems)


# ---------------------------------------------------------------------------
# AllreduceSdma handle cache
# ---------------------------------------------------------------------------


_SDMA_ALLREDUCE_NATIVE_DTYPES = frozenset(
    {torch.uint32, torch.int32, torch.float16, torch.bfloat16}
)


class SdmaAllreduce:
    """Reusable AllreduceSdma handle with automatic buffer management.

    Wraps ``mori.ccl.AllreduceSdma``.  The reduction operation is
    element-wise **SUM** (the only op mori's SDMA kernel supports).

    The mori SDMA allreduce kernel natively supports uint32, int32,
    float16, and bfloat16.  When ``dtype`` is ``torch.float32``, this
    wrapper transparently casts to bfloat16 for the reduction and casts
    the result back to float32.

    Args:
        dtype: Element type (default ``torch.float32``).
        ctx: A :class:`SdmaContext` (defaults to the process singleton).
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        ctx: SdmaContext | None = None,
    ):
        if dtype != torch.float32 and dtype not in _SDMA_ALLREDUCE_NATIVE_DTYPES:
            raise ValueError(
                f"SdmaAllreduce: unsupported dtype {dtype}. "
                f"Supported: float32 (via bf16), {sorted(str(d) for d in _SDMA_ALLREDUCE_NATIVE_DTYPES)}"
            )
        self._ctx = ctx or SdmaContext.get()
        self._user_dtype = dtype
        self._wire_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        self._handle = None
        self._capacity_elems: int = 0
        self._out_buf: torch.Tensor | None = None

    @property
    def npes(self) -> int:
        return self._ctx.npes

    def _ensure_handle(self, n_elems: int) -> None:
        if self._handle is not None and n_elems <= self._capacity_elems:
            return

        from mori.ccl import AllreduceSdma

        elem_size = torch.tensor([], dtype=self._wire_dtype).element_size()
        input_buf_bytes = n_elems * elem_size
        npes = self._ctx.npes
        output_buf_bytes = npes * (n_elems // npes + 64) * elem_size

        self._handle = AllreduceSdma(
            self._ctx.my_pe,
            npes,
            input_buffer_size=input_buf_bytes,
            output_buffer_size=output_buf_bytes,
            copy_output_to_user=True,
            dtype=self._wire_dtype,
        )
        self._capacity_elems = n_elems
        self._out_buf = None
        logger.debug(
            "SdmaAllreduce: (re-)allocated handle for %d elems, "
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

    def inplace(
        self,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """In-place SUM allreduce on *tensor*."""
        tensor = tensor.contiguous()
        n_elems = tensor.numel()
        self._ensure_handle(n_elems)
        if stream is None:
            stream = torch.cuda.current_stream(tensor.device)
        flat = tensor.reshape(-1)
        if self._needs_cast:
            buf = flat.to(self._wire_dtype)
            out = self._get_out_buf(buf)
            self._handle(buf, out, n_elems, stream)
            tensor.copy_(out.to(self._user_dtype).reshape(tensor.shape))
        else:
            out = self._get_out_buf(flat)
            self._handle(flat, out, n_elems, stream)
            tensor.copy_(out.reshape(tensor.shape))
        stream.synchronize()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Out-of-place SUM allreduce: ``output = sum(input across PEs)``."""
        input_tensor = input_tensor.contiguous()
        n_elems = input_tensor.numel()
        self._ensure_handle(n_elems)
        if stream is None:
            stream = torch.cuda.current_stream(input_tensor.device)
        if self._needs_cast:
            buf_in = input_tensor.to(self._wire_dtype)
            buf_out = torch.empty_like(buf_in)
            self._handle(buf_in, buf_out, n_elems, stream)
            stream.synchronize()
            output_tensor.copy_(buf_out.to(self._user_dtype))
        else:
            self._handle(input_tensor, output_tensor, n_elems, stream)
            stream.synchronize()


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


class SdmaAll2all:
    """Reusable All2allSdma handle with automatic buffer management.

    Wraps ``mori.ccl.All2allSdma`` and caches the handle so that
    repeated calls with the same (or smaller) tensor size reuse the
    same symmetric-memory buffers.

    The All2all semantics: each PE sends ``count`` elements to every
    other PE.  Input layout is ``[npes * count]`` where chunk *i* goes
    to PE *i*.  Output layout is the same: chunk *i* was received from
    PE *i*.

    Args:
        ctx: A :class:`SdmaContext` (defaults to the process singleton).
    """

    def __init__(self, ctx: SdmaContext | None = None):
        self._ctx = ctx or SdmaContext.get()
        self._handle = None
        self._capacity_bytes: int = 0

    @property
    def npes(self) -> int:
        return self._ctx.npes

    def _ensure_handle(self, total_bytes: int) -> None:
        if self._handle is not None and total_bytes <= self._capacity_bytes:
            return

        from mori.ccl import All2allSdma

        self._handle = All2allSdma(
            self._ctx.my_pe,
            self._ctx.npes,
            input_buffer_size=total_bytes,
            output_buffer_size=total_bytes,
            copy_output_to_user=True,
        )
        self._capacity_bytes = total_bytes
        logger.debug(
            "SdmaAll2all: (re-)allocated handle for %d bytes (PE %d/%d)",
            total_bytes,
            self._ctx.my_pe,
            self._ctx.npes,
        )

    def __call__(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """All-to-all *input_tensor* into *output_tensor* via SDMA.

        Both tensors must be contiguous 1-D with the same numel.  The
        data is reinterpreted as ``uint32`` for the SDMA transport.
        """
        input_tensor = input_tensor.contiguous()
        npes = self._ctx.npes
        total_bytes = input_tensor.numel() * input_tensor.element_size()
        self._ensure_handle(total_bytes)

        count_per_pe_bytes = total_bytes // npes
        assert count_per_pe_bytes % 4 == 0, (
            f"Per-PE chunk size ({count_per_pe_bytes} bytes) must be "
            f"divisible by 4 for uint32 reinterpret"
        )
        count_u32_per_pe = count_per_pe_bytes // 4

        in_flat = input_tensor.view(-1).view(torch.uint32)
        out_flat = output_tensor.view(-1).view(torch.uint32)

        if stream is None:
            stream = torch.cuda.current_stream(input_tensor.device)
        self._handle(in_flat, out_flat, count_u32_per_pe, stream)
        stream.synchronize()


def sdma_allgather_max(
    packed: torch.Tensor,
    allgather: SdmaAllgather,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Allgather *packed* via SDMA, then return element-wise MAX across PEs.

    This emulates ``torch.distributed.all_reduce(op=MAX)`` using SDMA
    transport: the data path goes through the SDMA engine (allgather),
    and only the final MAX reduction is computed locally on-GPU.

    Args:
        packed: 1-D float32 tensor of values on this rank.
        allgather: A pre-created :class:`SdmaAllgather` handle.
        stream: CUDA stream (defaults to current stream).

    Returns:
        1-D float32 tensor with the global element-wise MAX.
    """
    gathered = allgather(packed, stream=stream)  # (npes, n_elems)
    return gathered.amax(dim=0)
