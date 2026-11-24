# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrappers of the Gemma add + RMSNorm kernels."""

import functools
import struct

import torch

from . import add_rmsnorm as _add_rmsnorm
from . import add_rmsnorm_mxfp8 as _add_rmsnorm_mxfp8
from .common import H


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``), later calls run the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _add_rmsnorm_kernel():
    return _add_rmsnorm.compile_add_rmsnorm()


@functools.cache
def _add_rmsnorm_mxfp8_kernel():
    return _add_rmsnorm_mxfp8.compile_add_rmsnorm_mxfp8()


def _f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def add_rmsnorm_ok(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
    """Shapes the FlyDSL kernels take: bf16 [m, 6144] rows with unit column stride; not under torch.compile (the
    DSpark draft): Dynamo cannot trace the launch."""
    return (
        not torch.compiler.is_compiling()
        and x.dim() == 2
        and x.shape[1] == H
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and residual.shape == x.shape
        and x.stride(1) == 1
        and residual.stride(1) == 1
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )


def gemma_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``gemma_fused_add_rmsnorm`` (out, x + residual) bit for bit."""
    m = x.shape[0]
    out = torch.empty((m, H), dtype=x.dtype, device=x.device)
    res_out = torch.empty((m, H), dtype=x.dtype, device=x.device)
    if m:
        _run_compiled(
            _add_rmsnorm_kernel(),
            x.data_ptr(),
            residual.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            res_out.data_ptr(),
            res_out.data_ptr(),  # the unused arg_aux
            m,
            x.stride(0),
            residual.stride(0),
            H,
            _f32_bits(eps),
            torch.cuda.current_stream(),
        )
    return out, res_out


def _out_or_empty(out, shape, dtype, device) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    assert tuple(out.shape) == tuple(shape) and out.dtype == dtype
    assert out.is_contiguous()
    return out


def gemma_add_rmsnorm_mxfp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    code_tile_rows: int,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mxfp8_norm_quant.gemma_add_rmsnorm_mxfp8_quant`` with AITER-layout scales and tiled codes (``code_tile_rows``
    > 0) bit for bit: (codes, scales, x + residual), padding rows untouched."""
    m = x.shape[0]
    assert code_tile_rows > 0 and code_tile_rows % 16 == 0
    rows = (m + code_tile_rows - 1) // code_tile_rows * code_tile_rows
    q = _out_or_empty(q_out, (rows, H), torch.float8_e4m3fn, x.device)
    scale = _out_or_empty(
        scale_out, ((rows + 31) // 32 * 32, H // 32), torch.uint8, x.device
    )
    res_out = torch.empty((m, H), dtype=x.dtype, device=x.device)
    if m:
        _run_compiled(
            _add_rmsnorm_mxfp8_kernel(),
            x.data_ptr(),
            residual.data_ptr(),
            weight.data_ptr(),
            q.data_ptr(),
            scale.data_ptr(),
            res_out.data_ptr(),
            res_out.data_ptr(),  # the unused arg_aux
            q.numel(),
            scale.numel(),
            m,
            x.stride(0),
            residual.stride(0),
            res_out.stride(0),
            H,
            _f32_bits(eps),
            torch.cuda.current_stream(),
        )
    return q, scale, res_out


def warmup_add_rmsnorm_mxfp8(device=None) -> None:
    """Compile the MXFP8-out add + RMSNorm on ``device``: tiled and row-major codes (no variant depends on the row
    count, the code tile or eps)."""
    device = torch.device("cuda") if device is None else torch.device(device)
    x = torch.zeros((32, H), dtype=torch.bfloat16, device=device)
    w = torch.zeros(H, dtype=torch.bfloat16, device=device)
    gemma_add_rmsnorm_mxfp8(x, x, w, 1e-6, 16)
    gemma_add_rmsnorm_mxfp8_row_major(x, x, w, 1e-6)
    torch.accelerator.synchronize()


@functools.cache
def _add_rmsnorm_mxfp8_row_major_kernel():
    from .add_rmsnorm_mxfp8_row_major import compile_add_rmsnorm_mxfp8_row_major
    return compile_add_rmsnorm_mxfp8_row_major()


def gemma_add_rmsnorm_mxfp8_row_major(
    x, residual, weight, eps, q_out=None, scale_out=None, res_out=None,
):
    """``gemma_add_rmsnorm_mxfp8`` with row-major codes (the tiled norm's arithmetic at every M): (codes, scales,
    residual out); AITER-layout scales, scale padding untouched."""
    assert add_rmsnorm_ok(x, residual, weight) and weight.shape == (H,)
    assert x.device == residual.device == weight.device
    m = x.shape[0]
    q = _out_or_empty(q_out, (m,H), torch.float8_e4m3fn, x.device)
    scale = _out_or_empty(scale_out, ((m+31)//32*32,H//32), torch.uint8, x.device)
    res = _out_or_empty(res_out, (m,H), torch.bfloat16, x.device)
    if m:
        _run_compiled(
            _add_rmsnorm_mxfp8_row_major_kernel(),
            x.data_ptr(),residual.data_ptr(),weight.data_ptr(),q.data_ptr(),scale.data_ptr(),
            res.data_ptr(),res.data_ptr(),q.numel(),scale.numel(),m,x.stride(0),
            residual.stride(0),H,H,_f32_bits(eps),torch.cuda.current_stream(),
        )
    return q,scale,res


@functools.cache
def code_tile_rows_for(rows: int, gemm_weight: torch.Tensor) -> int:
    """The code layout the MXFP8 GEMM of ``gemm_weight`` (gemm.mxfp8_common, preshuffled) reads at ``rows`` rows: its
    decode row's tile, or 0 for row-major codes."""
    from amd_kernels.gemm.mxfp8_common import gemm as G

    n, k = gemm_weight.shape
    return G.prequant_code_tile_rows(rows, n, k, torch.bfloat16)


_warmed_for: set = set()


def warmup_for_gemm(device) -> None:
    """Compile both code layouts of ``gemma_add_rmsnorm_mxfp8_for`` on ``device`` (once per device).
    ``gemm.mxfp8_common.warmup_mxfp8_gemm_weight`` calls it for every K 6144 weight, so the producer is warm before
    graph capture even when its first call comes later (SP prefill forwards skip it)."""
    device = torch.device(device)
    if device not in _warmed_for:
        _warmed_for.add(device)
        warmup_add_rmsnorm_mxfp8(device)


def gemma_add_rmsnorm_mxfp8_for(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The add + Gemma RMSNorm + MXFP8 quant whose codes the MXFP8 GEMM of ``gemm_weight`` reads as they are
    (``gemm.mxfp8_common.mxfp8_gemm_prequant``): (codes, scales, x + residual), (16, 16) codes padded to its decode
    row's tile or row-major. Its first call on a device outside graph capture compiles both layouts."""
    if not torch.cuda.is_current_stream_capturing():
        warmup_for_gemm(x.device)
    tile = code_tile_rows_for(x.shape[0], gemm_weight)
    if tile:
        return gemma_add_rmsnorm_mxfp8(x, residual, weight, eps, tile)
    return gemma_add_rmsnorm_mxfp8_row_major(x, residual, weight, eps)
