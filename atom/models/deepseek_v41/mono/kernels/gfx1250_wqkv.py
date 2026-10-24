# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared hidden MXFP8 -> wqkv_a consumer using the eight-wave GEMV."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, check_e8m0, ptr_arg

from atom.models.deepseek_v41.mono.kernels.gfx1250_gemv import (
    THREADS,
    TILE_N,
    gemv_device,
    gemv_lds_bytes,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_norm_wqkv import HIDDEN, OUTPUT


def _kernel_name(rows):
    return f"atom_gfx1250_shared_wqkv_r{rows}_k8"


@lru_cache(maxsize=2)
def _launcher(rows):
    @flyc.jit
    def launch(
        x: fx.Pointer,
        xs: fx.Pointer,
        weight: fx.Pointer,
        scale: fx.Pointer,
        out: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(name=_kernel_name(rows), known_block_size=[THREADS, 1, 1])
        def kernel(
            x: fx.Pointer,
            xs: fx.Pointer,
            weight: fx.Pointer,
            scale: fx.Pointer,
            out: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(gemv_lds_bytes(rows, HIDDEN))
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            # Unused RoPE/debug operands never issue a load.
            gemv_device(
                x,
                xs,
                weight,
                scale,
                out,
                out,
                out,
                out,
                lds,
                tid,
                bid,
                rows,
                HIDDEN,
                OUTPUT,
            )

        kernel(x, xs, weight, scale, out).launch(
            grid=(OUTPUT // TILE_N, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch


def _arguments(x, xs, weight, scale, out):
    rows = x.shape[0] if x.ndim == 2 else 0
    owner = "gfx1250 shared wqkv"
    if rows not in (1, 2):
        raise ValueError(f"[{owner}] requires rows1/2")
    specs = [
        (x, (rows, HIDDEN), torch.float8_e4m3fn, "x"),
        (weight, (OUTPUT, HIDDEN), torch.float8_e4m3fn, "weight"),
        (out, (rows, OUTPUT), torch.bfloat16, "out"),
    ]
    check_e8m0(xs, "xs", owner, (rows, HIDDEN // 32))
    check_e8m0(scale, "scale", owner, (OUTPUT // 32, HIDDEN // 32))
    for t, shape, dtype, name in specs:
        if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
            raise ValueError(f"[{owner}] {name} requires contiguous {dtype} {shape}")
    tensors = (x, xs, weight, scale, out)
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError(f"[{owner}] buffers must share one GPU")
    if (
        torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        != "gfx1250"
    ):
        raise RuntimeError(f"[{owner}] requires gfx1250")
    lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
    if any(
        lo < t.data_ptr() + t.numel() * t.element_size() and t.data_ptr() < hi
        for t in tensors[:-1]
    ):
        raise ValueError(f"[{owner}] output must not overlap inputs")
    return tuple(ptr_arg(t) for t in tensors) + (
        torch.cuda.current_stream(x.device).cuda_stream,
    )


def shared_wqkv_gfx1250(x, xs, weight, scale, out):
    """Consume finite hidden FP8/E8M0 and native-preshuffled wqkv weights.

    Buffers are caller-owned and graph compatible. The independent launch
    establishes the same GEMV compute and precision as the fused norm stage.
    """
    _run_compiled(_launcher(x.shape[0]), *_arguments(x, xs, weight, scale, out))
    return out
