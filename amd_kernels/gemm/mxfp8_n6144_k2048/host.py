# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows and kernels of the o_proj N 6144 x K 2048 (TP4), registered with the common dispatcher
(``gemm.mxfp8_common.gemm._SHAPE_HOSTS``): its windows ahead of the tuned table (the 128x192 tile here, the
256x192 persistent tile of ``gemm/mxfp8_n6144_k8192`` at K 2048), the producer rows of pre-quantized codes, and
the decode row whose activation stages through LDS (``decode_astage``)."""

import functools

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _DECODE_KERNEL, _Config
from amd_kernels.gemm.mxfp8_n6144_k8192.host import _PREFILL_N6144_K8192_192_KERNEL

_N6144_K2048_128_KERNEL = "n6144_k2048_128x192"


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """The rows ahead of the tuned table (``_select_config``), or None."""
    # The TP4 o_proj at 3329-3968 rows (14-16 m-blocks, the last one at most half full): the
    # persistent 256x192 tile at K 2048, 448-512 tiles in two rounds, instead of the AITER
    # 128x256 row's three (graph -4.5..-12.7 %, eager -3.6..-12.6 %). From
    # 3969 rows both fill their rounds and the eager gain is gone (4032 -1 % overlapping, 4095 0).
    if out_dtype == torch.bfloat16 and 3328 < m <= 3968:
        return _Config(256, 192, 128, 1, 1, 1, _PREFILL_N6144_K8192_192_KERNEL, 2, 2)
    # The TP4 o_proj, the same 128x192 kernel at K 2048: one round of up to 256 tiles
    # instead of 192 256x128 tiles (row-major codes; above 1024 rows a second round loses).
    if out_dtype == torch.bfloat16 and 768 < m <= 1024:
        return _Config(128, 192, 128, 1, 1, 1, _N6144_K2048_128_KERNEL, 2, 2)
    return None


def select_prequant(m: int, out_dtype: torch.dtype) -> _Config | None:
    """A codes input can avoid the K-wave reduction paid by tiny bf16 rows
    (``_select_prequant_config``)."""
    if out_dtype == torch.bfloat16 and 1 <= m <= 32:
        return _Config(16 if m <= 16 else 32, 32, 128, 0, 0, 1, _DECODE_KERNEL, 1, 0)
    return None


def decode_launcher(config: _Config, a_bf16: bool):
    """The decode row whose activation stages through LDS, or None (the common decode kernel)."""
    if (
        not a_bf16 and config.kernel == _DECODE_KERNEL
        and (config.tile_m, config.tile_n, config.waves_m, config.split_k) == (64, 48, 1, 1)
        and not config.la_w and not config.la_a and config.w_cm is None
    ):
        # M97-128 keeps its producer's 64-row code tile and deterministic K order.
        from .decode_astage import compile_mxfp8_gemm_n6144_k2048_astage

        return compile_mxfp8_gemm_n6144_k2048_astage()
    return None


@functools.cache
def _n6144_k2048_128_launcher():
    from .tile_128x192 import compile_mxfp8_gemm_n6144_k2048_128x192
    return compile_mxfp8_gemm_n6144_k2048_128x192()


def _launch_n6144_k2048_128(activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled
    assert activation.shape[1] == 2048 and out.shape[1] == 6144 and out.dtype == torch.bfloat16
    _run_compiled(
        _n6144_k2048_128_launcher(),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        activation.shape[0], 6144, torch.cuda.current_stream(),
    )


LAUNCHERS = {
    _N6144_K2048_128_KERNEL: lambda config, a, sa, w, sw, out: _launch_n6144_k2048_128(a, sa, w, sw, out),
}
