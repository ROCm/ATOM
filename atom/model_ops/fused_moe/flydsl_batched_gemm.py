# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Thin torch-facing wrapper around the FlyDSL strided-batched MXFP8xMXFP4 GEMM.

Wraps ``compile_mxfp8_gemm`` from FlyDSL's ``kernels/gemm/mxfp4_preshuffle.py``
(branch ``support_f8f4_batchgemm``): a per-expert strided-batched w4a8 GEMM where
``batch`` maps to grid.z. Used by the batched grouped-expert MoE path
(``rccl_batched_experts.py``) to compute, for every local expert e:

    C[e] = A[e] @ B[e]^T     A[e]: [M, K] MXFP8 e4m3
                             B[e]: [N, K/2] CK-preshuffled MXFP4 e2m1
                             C[e]: [M, N] bf16

with per-32 e8m0 block scales (K-256 granular). The compile step is cached
(keyed by the static dims N, K, batch, tile sizes); only the runtime launch runs
per call.

This module imports FlyDSL lazily so the file loads on machines without it; the
public entry raises a clear error if FlyDSL / the batch kernel is unavailable.
"""

import logging
from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger("atom")

_flydsl_compile_fn = None
_flydsl_import_done = False


def _get_batched_compile_fn():
    """Lazy-import compile_mxfp8_gemm from FlyDSL; cache the result (or None)."""
    global _flydsl_compile_fn, _flydsl_import_done
    if _flydsl_import_done:
        return _flydsl_compile_fn
    _flydsl_import_done = True
    # The FlyDSL pip package ships only the compiler/runtime, not the kernel
    # sources, so the MXFP8xMXFP4 batch kernel is vendored under
    # flydsl_kernels/ next to this file. Prefer the vendored copy; fall back to
    # an upstream flydsl.kernels install if one is present.
    try:
        from .flydsl_kernels.mxfp4_preshuffle import compile_mxfp8_gemm

        _flydsl_compile_fn = compile_mxfp8_gemm
        logger.info("[FlyDSL] loaded vendored batched MXFP8xMXFP4 GEMM compiler")
        return _flydsl_compile_fn
    except Exception as e:  # pragma: no cover - env-dependent
        logger.info(f"[FlyDSL] vendored batched GEMM import failed: {e}")
    try:
        from flydsl.kernels.gemm.mxfp4_preshuffle import compile_mxfp8_gemm

        _flydsl_compile_fn = compile_mxfp8_gemm
        logger.info("[FlyDSL] loaded upstream batched MXFP8xMXFP4 GEMM compiler")
    except Exception as e:  # pragma: no cover - env-dependent
        logger.info(f"[FlyDSL] batched MXFP8xMXFP4 GEMM not available: {e}")
    return _flydsl_compile_fn


def is_available() -> bool:
    """Whether the FlyDSL batched w4a8 GEMM kernel can be used."""
    return _get_batched_compile_fn() is not None


@lru_cache(maxsize=64)
def _compile(
    N: int,
    K: int,
    batch: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    out_dtype: str,
):
    compile_fn = _get_batched_compile_fn()
    if compile_fn is None:
        raise RuntimeError(
            "[FlyDSL] batched MXFP8xMXFP4 GEMM compiler is not available. "
            "Install FlyDSL (support_f8f4_batchgemm branch) to use "
            "ATOM_RCCL_MOE_IMPL=flydsl_batched_gemm."
        )
    return compile_fn(
        N=N,
        K=K,
        batch=batch,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype=out_dtype,
    )


def batched_mxfp8_mxfp4_gemm(
    a: Tensor,  # [E, M, K]   MXFP8 e4m3 (1 byte/elem)
    b: Tensor,  # [E, N, K/2] CK-preshuffled MXFP4 e2m1
    scale_a: Tensor,  # [E, M, K/32] e8m0 (shuffled)
    scale_b: Tensor,  # [E, N, K/32] e8m0 (shuffled)
    out: Optional[Tensor] = None,  # [E, M, N] bf16
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run the strided-batched w4a8 GEMM ``C[e] = A[e] @ B[e]^T`` for all E experts.

    Shapes (E = local experts, M = tokens-per-expert capacity, N = out feature,
    K = in feature):
      a:       [E, M, K]     MXFP8 e4m3
      b:       [E, N, K/2]   CK-preshuffled MXFP4 e2m1
      scale_a: [E, M, K/32]  e8m0
      scale_b: [E, N, K/32]  e8m0
      out:     [E, M, N]     bf16 (allocated if None)

    K must be a multiple of 256 and N a multiple of ``tile_n`` (FlyDSL kernel
    constraints). Returns ``out``.
    """
    assert a.dim() == 3 and b.dim() == 3, (a.shape, b.shape)
    E, M, K = a.shape
    Eb, N, Khalf = b.shape
    assert Eb == E, f"expert dim mismatch: a {E} vs b {Eb}"
    assert Khalf * 2 == K, f"MXFP4 weight K/2 mismatch: {Khalf}*2 != {K}"
    if K % 256 != 0:
        raise RuntimeError(f"[FlyDSL] K ({K}) must be a multiple of 256")
    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL] N ({N}) not a multiple of tile_n ({tile_n})")

    odt = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    exe = _compile(N, K, E, tile_m, tile_n, tile_k, odt)

    if out is None:
        out = torch.empty((E, M, N), dtype=out_dtype, device=a.device)

    import flydsl.compiler as flyc

    def _as_u8(t: Tensor) -> Tensor:
        # launch_gemm's arg_a/arg_b/scales are byte-typed tensor args; view fp8
        # codes as uint8. (This kernel's get_iter expects real tensors, not raw
        # pointers, so we pass flattened torch tensors directly to flyc.compile.)
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    out_c = out.contiguous()
    # FlyDSL launch_gemm wants a bias slot even when unused (epilogue=none).
    dummy_bias = torch.empty(0, dtype=out.dtype, device=out.device)
    args = (
        out_c.view(-1),
        _as_u8(a.contiguous()).view(-1),
        _as_u8(b.contiguous()).view(-1),
        _as_u8(scale_a.contiguous()).view(-1),
        _as_u8(scale_b.contiguous()).view(-1),
        dummy_bias,
        M,
        N,
        torch.cuda.current_stream(),
    )
    # First call compiles+runs; subsequent calls fast-dispatch via cached fn.
    # The FIRST compile must NOT happen inside a CUDA-graph capture region — it
    # launches its own kernels/allocs and corrupts the graph. LL warmup runs the
    # same static (N, K, E, tiles) shape before capture, so exe._cf is normally
    # already populated here. Fail loudly if we somehow reach capture uncompiled
    # (e.g. a shape that warmup didn't exercise) rather than silently break.
    cf = getattr(exe, "_cf", None)
    if cf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[FlyDSL] batched w4a8 GEMM hit an uncompiled shape during CUDA "
                f"graph capture (N={N}, K={K}, batch={E}). Warm up this shape "
                "eagerly before capture (run a dummy decode at the same graph_bs)."
            )
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)
    if out_c is not out:
        out.copy_(out_c)
    return out
