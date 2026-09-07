# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers for creating ROCm streams with an explicit CU mask."""

import ctypes
from collections.abc import Sequence

import torch


def create_hip_stream_with_cu_mask(
    mask_bits: Sequence[int],
) -> torch.cuda.ExternalStream:
    """Create a normal-priority HIP stream with the supplied CU mask.

    ROCclr allocates CU-masked streams outside its ordinary hardware-queue
    pool.  This is useful when a latency-sensitive auxiliary stream must not
    share an AQL ring with unrelated logical streams after the ordinary pool
    reaches ``GPU_MAX_HW_QUEUES``.
    """
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("CU-masked streams require a ROCm PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("CU-masked streams require an available ROCm device")
    if not mask_bits:
        raise ValueError("CU mask must contain at least one word")

    words = [int(word) for word in mask_bits]
    if any(word < 0 or word > 0xFFFFFFFF for word in words):
        raise ValueError("CU mask words must be uint32 values")

    hip = ctypes.CDLL("libamdhip64.so")
    create_stream = hip.hipExtStreamCreateWithCUMask
    create_stream.restype = ctypes.c_int
    create_stream.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]

    raw_stream = ctypes.c_void_p()
    mask = (ctypes.c_uint * len(words))(*words)
    result = create_stream(ctypes.byref(raw_stream), len(words), mask)
    if result != 0:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: HIP error {result}")
    if raw_stream.value is None:
        raise RuntimeError("hipExtStreamCreateWithCUMask returned a null stream")
    return torch.cuda.ExternalStream(raw_stream.value)


def create_full_device_hip_stream() -> torch.cuda.ExternalStream:
    """Create a dedicated normal-priority HIP stream spanning every device CU."""
    cu_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if cu_count <= 0:
        raise RuntimeError(f"invalid device CU count: {cu_count}")

    full_words, remaining_bits = divmod(cu_count, 32)
    mask_bits = [0xFFFFFFFF] * full_words
    if remaining_bits:
        mask_bits.append((1 << remaining_bits) - 1)
    return create_hip_stream_with_cu_mask(mask_bits)
