# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows and kernels of the QKV projection N 2560 x K 6144 (TP4), registered with the common
dispatcher (``gemm.mxfp8_common.gemm._SHAPE_HOSTS``): its windows ahead of the tuned table, the producer rows of
pre-quantized codes, the deferred (split-K partial) rows of the fused norm / RoPE / insert chain, and the
launches of its two kernels (``rect_128x160``, ``prefill_192x256``)."""

import functools

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _DECODE_KERNEL, _Config

_RECT_N2560_K6144_KERNEL = "rect_n2560_k6144"
_PREFILL_N2560_K6144_192_KERNEL = "prefill_n2560_k6144_192x256"


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """The rows ahead of the tuned table (``_select_config``), or None."""
    # A 192-row tile fills the CU round without a second round of 128-row tiles.
    if out_dtype == torch.bfloat16 and 3073 <= m <= 4800:
        return _Config(192, 256, 128, 1, 1, 1, _PREFILL_N2560_K6144_192_KERNEL, 2, 2)
    # A 128x160 tile fills one CU round on the narrow QKV projection.
    if out_dtype == torch.bfloat16 and 1536 < m <= 2048:
        return _Config(128, 160, 128, 1, 1, 1, _RECT_N2560_K6144_KERNEL, 2, 2)
    return None


def select_prequant(m: int, out_dtype: torch.dtype) -> _Config | None:
    """A codes input can avoid the K-wave reduction paid by tiny bf16 rows
    (``_select_prequant_config``)."""
    if out_dtype == torch.bfloat16 and 9 <= m <= 16:
        return _Config(16, 16, 128, 0, 0, 1, _DECODE_KERNEL, 1, 0)
    return None


def select_deferred(m: int, config: _Config) -> _Config:
    """QKV partials consumed by the fused norm/RoPE/insert reduction.

    Choose the K split for the two-launch chain rather than a standalone GEMM. Only
    TP4 QKV's measured windows use different tiles; a row's code rows
    (``ceil(m / tile) * tile``) never exceed the producer's (its tile from
    ``prequant_code_tile_rows``), so it reads a prefix of the producer's codes.
    """
    if 1 <= m <= 16:
        return _Config(16, 32, 128, 0, 0, 3, _DECODE_KERNEL, 1, 0)
    if config.kernel == _DECODE_KERNEL:
        # one m-block of 32-column tiles at 33-80 rows (48 / 64 / 80 rows, 240 workgroups):
        # chain -2.3..-6.7% vs two m-blocks of 32x64 / 48x64
        if 33 <= m <= 80:
            return _Config(48 if m <= 48 else (64 if m <= 64 else 80), 32, 128, 0, 0, 3,
                           _DECODE_KERNEL, 1, 0)
        if 81 <= m <= 96:
            return _Config(48, 64, 128, 0, 0, 3, _DECODE_KERNEL, 1, 0)
        if 97 <= m <= 128:
            return _Config(64, 64, 128, 0, 0, 3, _DECODE_KERNEL, 1, 0)
        # three 48-row m-blocks cover 129-144 rows with 12.5% fewer bytes per CU than
        # three 64-row ones (chain -6.1..-6.8%)
        if 129 <= m <= 144:
            return _Config(48, 64, 128, 0, 0, 2, _DECODE_KERNEL, 3, 0)
        # to 192 since the split-2 reduce-insert takes one token pair from 177 tokens
        # (chain -1.3..-2.6% vs the standalone row at 177-192)
        if 145 <= m <= 192:
            return _Config(64, 64, 128, 0, 0, 2, _DECODE_KERNEL, 3, 0)
        # 3-4 m-blocks re-read each W column: cached W loads (w_cm 0) beat the
        # non-temporal default there (chain -1.8..-6.3%); four K-waves with
        # the park-all K-reduce instead of two (chain -0.9..-2.2%)
        if 193 <= m <= 256:
            return _Config(64, 80, 128, 0, 0, 2, _DECODE_KERNEL, 4, 0, w_cm=0)
        if 257 <= m <= 288:
            return _Config(96, 64, 128, 0, 0, 2, _DECODE_KERNEL, 4, 0, w_cm=0)
    return config


@functools.cache
def _prefill_n2560_k6144_192_launcher():
    from .prefill_192x256 import compile_mxfp8_gemm_n2560_k6144_192x256
    return compile_mxfp8_gemm_n2560_k6144_192x256()


def _launch_prefill_n2560_k6144_192(activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled
    m, n = out.shape
    _run_compiled(
        _prefill_n2560_k6144_192_launcher(),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        m, n, 0, ((m + 191) // 192) * (n // 256), torch.cuda.current_stream(),
    )


@functools.cache
def _rect_n2560_k6144_launcher():
    from .rect_128x160 import compile_mxfp8_gemm_n2560_k6144

    return compile_mxfp8_gemm_n2560_k6144()


def _launch_rect_n2560_k6144(activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled

    assert activation.shape[1] == 6144 and out.shape[1] == 2560
    assert out.dtype == torch.bfloat16
    _run_compiled(
        _rect_n2560_k6144_launcher(),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        activation.shape[0], 2560, torch.cuda.current_stream(),
    )


LAUNCHERS = {
    _PREFILL_N2560_K6144_192_KERNEL: lambda config, a, sa, w, sw, out: _launch_prefill_n2560_k6144_192(a, sa, w, sw, out),
    _RECT_N2560_K6144_KERNEL: lambda config, a, sa, w, sw, out: _launch_rect_n2560_k6144(a, sa, w, sw, out),
}
