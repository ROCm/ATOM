# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Original bf16 weights and a compact, bit-exact activation layout."""

import functools

import torch

from .kernel import compile_bf16_gemm_packed_rows_shuffled
from .kernel_direct import compile_bf16_gemm_packed_rows
from .pack import compile_pack_bf16_rows
from .reference import K, N

# Sweep at 3/4/8/16/32/64 rows: keep direct stores below 8;
# LDS output transpose wins above that, especially at 32/64 rows.
DIRECT_MAX_ROWS = 7


def padded_rows(n: int) -> int:
    assert 0 <= n <= 64
    return 16 if n <= 16 else (32 if n <= 32 else 64)


@functools.cache
def _gemm(mt: int, shuffled: bool):
    if shuffled:
        return compile_bf16_gemm_packed_rows_shuffled(mt)
    return compile_bf16_gemm_packed_rows(mt)


@functools.cache
def _pack(mt: int):
    return compile_pack_bf16_rows(mt)


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def pack_bf16_rows(x: torch.Tensor, row_indices: torch.Tensor, packed: torch.Tensor) -> None:
    """Gather selected rows and pad them into the MFMA layout, with no rounding.

    Indices must be in [0, x.shape[0]); checking their values would synchronize.
    The caller owns the packed tensor. This function is CUDA-graph safe.
    """
    n = row_indices.numel()
    pad = padded_rows(n)
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.shape[0] <= N
    assert row_indices.dtype == torch.int64 and row_indices.ndim == 1 and row_indices.is_contiguous()
    assert packed.shape == (pad, K) and packed.dtype == x.dtype and packed.is_contiguous()
    assert x.device == row_indices.device == packed.device
    _run(
        _pack(pad // 16), x.data_ptr(), row_indices.data_ptr(), packed.data_ptr(),
        n, x.shape[0], torch.cuda.current_stream(),
    )


def bf16_gemm_packed_rows(
    packed: torch.Tensor, weight: torch.Tensor, n: int, out: torch.Tensor,
) -> torch.Tensor:
    """FP32-accumulated bf16 projection of 0-64 packed rows; original weight layout."""
    pad = padded_rows(n)
    assert packed.shape == (pad, K) and packed.dtype == torch.bfloat16 and packed.is_contiguous()
    assert weight.shape == (N, K) and weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert out.shape == (n, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert packed.device == weight.device == out.device
    if n:
        _run(
            _gemm(pad // 16, n > DIRECT_MAX_ROWS), packed.data_ptr(), weight.data_ptr(),
            out.data_ptr(), n, torch.cuda.current_stream(),
        )
    return out


def warmup_bf16_gemm_packed_rows(device: torch.device) -> None:
    """Compile the four GEMMs and three pack variants before graph capture."""
    w = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
    x = torch.zeros(64, K, dtype=torch.bfloat16, device=device)
    for n in (1, 8, 17, 33):
        p = torch.empty(padded_rows(n), K, dtype=torch.bfloat16, device=device)
        indices = torch.arange(n, device=device)
        pack_bf16_rows(x, indices, p)
        out = torch.empty(n, N, dtype=torch.bfloat16, device=device)
        bf16_gemm_packed_rows(p, w, n, out)
    torch.cuda.synchronize(device)

