# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows and kernels of the full o_proj N 6144 x K 8192 (``o_proj_full`` of the TP4
sequence-parallel prefill), registered with the common dispatcher (``gemm.mxfp8_common.gemm._SHAPE_HOSTS``): its
windows ahead of the tuned table, the 256x192 persistent tile (also the o_proj's K 2048 row at 3329-3968, see
``gemm/mxfp8_n6144_k2048``), and its one-tile and split-2 forms (the split-2 form with its ready words,
``split2_ready``)."""

import functools

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _DECODE_FQ_KERNEL, _DECODE_KERNEL, _Config, _fourwave_grid_max

_PREFILL_N6144_K8192_192_KERNEL = "prefill_n6144_k8192_256x192"
_PREFILL_N6144_K8192_192_ONE_KERNEL = "prefill_n6144_k8192_256x192_one_tile"
_PREFILL_N6144_K8192_192_SPLIT2_KERNEL = "prefill_n6144_k8192_256x192_split2"


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """The rows ahead of the tuned table (``_select_config``), or None."""
    # One-tile epilogue: measured graph/eager gain on the lower full-o_proj window.
    if out_dtype == torch.bfloat16 and 1281 <= m <= 1536:
        return _Config(256, 192, 128, 1, 1, 1, _PREFILL_N6144_K8192_192_ONE_KERNEL, 2, 2)

    if out_dtype == torch.bfloat16 and 1281 <= m <= 2048:
        return _Config(256, 192, 128, 1, 1, 1, _PREFILL_N6144_K8192_192_KERNEL, 2, 2)

    # The full o_proj at 417-1024 rows (SP: 1024 rows at 4096 tokens): 64-128 256x192
    # tiles, two K splits each on one XCD with a fixed-role fp32 hand-off (128-256 workgroups
    # pulling 30 % less each than the one-launch 128x192 / 128x256 rows):
    # 1024 -6.4 %, 897 -6.1 %, 769-896 -11 %, 417-768 -15..-22 %, graph and eager.
    if out_dtype == torch.bfloat16 and 416 < m <= 1024:
        return _Config(256, 192, 128, 1, 1, 1, _PREFILL_N6144_K8192_192_SPLIT2_KERNEL, 2, 2)
    # Only the one measured token count overrides the tuned table.
    if allow_decode and out_dtype == torch.bfloat16 and m == 4:
        return _Config(8, 32, 128, 0, 0, 1, _DECODE_FQ_KERNEL, 2, 0, 6, 3)
    # The decode rows of M 289..416 (K-waves 2 -> 4, sched_barriers on; 112x96 above 384 rows
    # to stay one wave): -7.5..-9.3 % against the table's 96x96 row.
    if allow_decode and out_dtype == torch.bfloat16 and 288 < m <= 416:
        return _Config(96 if m <= 384 else 112, 96, 128, 0, 0, 1, _DECODE_KERNEL, 4, 1)
    return None


@functools.cache
def _prefill_n6144_k8192_192_launcher(k: int = 8192, grid_max: int = 256):
    from .prefill_256x192 import compile_mxfp8_gemm_n6144_k8192_256x192
    return compile_mxfp8_gemm_n6144_k8192_256x192(k, grid_max)


@functools.cache
def _prefill_n6144_k8192_192_one_launcher():
    from .prefill_256x192_one_tile import compile_mxfp8_gemm_n6144_k8192_256x192_one_tile
    return compile_mxfp8_gemm_n6144_k8192_256x192_one_tile()


def _launch_prefill_n6144_k8192_192(activation, activation_scale, weight, weight_scale, out, *, one_tile=False):
    from amd_kernels.common.tensor_shim import _run_compiled
    m, n = out.shape
    n_work = ((m + 255) // 256) * (n // 192)
    if one_tile and n_work > 256:
        raise ValueError("one-tile prefill GEMM requires at most 256 workgroups")
    k = activation.shape[1]
    if one_tile and k != 8192:
        raise ValueError("the one-tile 256x192 GEMM is compiled for K 8192 only")
    # more tiles than CUs (the K 2048 row at 3329-3968): inside DBO the plain grid, as the
    # 4-wave rows; at most 256 tiles the persistent and the plain grid are the same launch
    grid_max = _fourwave_grid_max() if n_work > 256 else 256
    _run_compiled(
        _prefill_n6144_k8192_192_one_launcher() if one_tile else _prefill_n6144_k8192_192_launcher(k, grid_max),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        m, n, 0, n_work, torch.cuda.current_stream(),
    )


@functools.cache
def _prefill_n6144_k8192_192_split2_launcher(p_steps: int = 32):
    from .prefill_256x192_split2 import (
        compile_mxfp8_gemm_n6144_k8192_256x192_split2,
    )
    return compile_mxfp8_gemm_n6144_k8192_256x192_split2(p_steps)


_SPLIT2_PARTIAL_FLOATS = 256 * 192  # one tile's fp32 partial
_SPLIT2_WORD_STRIDE = 32  # one ready word per tile, each on its own 128-byte line


_SPLIT2_WORDS: dict[tuple[torch.device, int], torch.Tensor] = {}


def _split2_ready_words(device: torch.device) -> torch.Tensor:
    """Eager calls reuse stream-local words; each captured call owns private words.

    A graph may replay on any stream, beside another graph or an eager call. Never borrow
    an eager stream's words during capture.
    """
    from .split2_ready import clear
    if torch.cuda.is_current_stream_capturing():
        words = torch.empty(256 * _SPLIT2_WORD_STRIDE, dtype=torch.int32, device=device)
        clear(words)
        return words
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    if key not in _SPLIT2_WORDS:
        words = torch.empty(256 * _SPLIT2_WORD_STRIDE, dtype=torch.int32, device=device)
        clear(words)
        _SPLIT2_WORDS[key] = words
    return _SPLIT2_WORDS[key]


def _launch_prefill_n6144_k8192_192_split2(activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled
    m, n = out.shape
    tiles = ((m + 255) // 256) * (n // 192)
    if 2 * tiles > 256:
        raise ValueError("the split-2 prefill GEMM needs both splits of every tile in one round")
    partials = torch.empty((tiles, _SPLIT2_PARTIAL_FLOATS), dtype=torch.float32, device=out.device)
    _run_compiled(
        _prefill_n6144_k8192_192_split2_launcher(),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        m, n, 0, 2 * tiles, partials.view(-1),
        _split2_ready_words(out.device),
        torch.cuda.current_stream(),
    )


LAUNCHERS = {
    _PREFILL_N6144_K8192_192_KERNEL: lambda config, a, sa, w, sw, out: _launch_prefill_n6144_k8192_192(a, sa, w, sw, out),
    _PREFILL_N6144_K8192_192_ONE_KERNEL: lambda config, a, sa, w, sw, out: _launch_prefill_n6144_k8192_192(a, sa, w, sw, out, one_tile=True),
    _PREFILL_N6144_K8192_192_SPLIT2_KERNEL: lambda config, a, sa, w, sw, out: _launch_prefill_n6144_k8192_192_split2(a, sa, w, sw, out),
}
