# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A pinned host <-> device copy as one `hipMemcpyAsync`.

`Tensor.copy_(..., non_blocking=True)` between pinned host memory and the GPU
also records the pinned block's use for PyTorch's caching host allocator, and
that path queries the stream's capture status. In a serving runner process that
query costs tens of microseconds (vs ~0.2 us in a fresh process), and a decode
step makes several such copies. The buffers passed here are persistent (a
packed staging arena, D2H slots that hold their source until the copy has been
waited on), so the allocator bookkeeping is not needed.
"""

import ctypes

import torch

H2D = 1  # hipMemcpyHostToDevice
D2H = 2  # hipMemcpyDeviceToHost

_memcpy_async = None


def _fn():
    global _memcpy_async
    if _memcpy_async is None:
        from aiter.jit.utils.hip_runtime import load_hip_runtime

        f = load_hip_runtime().hipMemcpyAsync
        f.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        f.restype = ctypes.c_int
        _memcpy_async = f
    return _memcpy_async


def copy_async(
    dst: torch.Tensor,
    src: torch.Tensor,
    kind: int,
    stream: torch.cuda.Stream | None = None,
) -> bool:
    """Enqueue ``dst.copy_(src)`` on ``stream`` (default: the current one) as a
    plain memcpy. Returns False, doing nothing, unless both are contiguous with
    the same dtype and element count (the caller then falls back to copy_)."""
    if (
        dst.dtype != src.dtype
        or dst.numel() != src.numel()
        or not dst.is_contiguous()
        or not src.is_contiguous()
    ):
        return False
    nbytes = src.numel() * src.element_size()
    if nbytes:
        s = (stream if stream is not None else torch.cuda.current_stream()).cuda_stream
        rc = _fn()(dst.data_ptr(), src.data_ptr(), nbytes, kind, s)
        if rc != 0:
            raise RuntimeError(f"hipMemcpyAsync returned {rc}")
    return True
