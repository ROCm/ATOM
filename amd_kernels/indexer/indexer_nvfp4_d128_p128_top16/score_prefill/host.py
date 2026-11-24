# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the FlyDSL NVFP4 prefill index scorer."""

import functools
import math

import torch

from amd_kernels.warm_up import launcher

from .. import constants

from . import kernel as _kernel
from ..host_stream import current_stream as _current_stream

# Workgroups to aim for: query tiles see causal windows of different lengths, so more, shorter segments (>= 4 blocks)
# balance the tail; 2048 was best of a 512..4096 sweep.
TARGET_WORKGROUPS = 2048
# segments are added until the last round of resident workgroups is this full (5 x 13K on 120K, 4 heads: -6.4%)
MIN_ROUND_FILL = 0.9
WORKGROUPS_PER_CU = 2
# Segments per query tile: small query counts on long contexts need many short segments to fill the GPU.
MAX_SEGMENTS = 1024


_NVFP4_E4M3_LUT: dict[torch.device, torch.Tensor] = {}


def nvfp4_e4m3_lut(device: torch.device) -> torch.Tensor:
    """[256] int64: byte j of entry s is e4m3_rn_satfinite(m_j * e4m3(s)) for the E2M1 magnitudes m = 0, 0.5, 1,
    1.5, 2, 3, 4, 6: the fp8 key of code magnitude j under block scale byte s."""
    lut = _NVFP4_E4M3_LUT.get(device)
    if lut is None:
        # device ops only (no host copy): a first call under graph capture is fine
        scales = torch.arange(256, dtype=torch.uint8, device=device)
        scales = scales.view(torch.float8_e4m3fn).float()
        code = torch.arange(8, device=device)
        exp, man = code // 2, (code % 2).float()
        mags = torch.where(
            exp == 0, man * 0.5, torch.exp2((exp - 1).float()) * (1.0 + man * 0.5)
        )
        # e2m1 * e4m3 is exact in fp32; saturate, then round once to E4M3.
        values = (scales[:, None] * mags[None, :]).clamp(-448.0, 448.0)
        lut = values.to(torch.float8_e4m3fn).view(torch.int64).reshape(256)
        _NVFP4_E4M3_LUT[device] = lut
    return lut


@functools.cache
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
@launcher
def _get_kernel(heads: int, staged: bool):
    return _kernel.compile_score_prefill(heads, staged)


@functools.cache
def _get_kernel_causal():
    from .score_causal_d128_p128_tq512 import compile_score_causal
    return compile_score_causal()


def _kernel_shared():
    # imported late: it shares helpers with the decode kernel, whose host imports this module
    from . import kernel_shared

    return kernel_shared


@functools.cache
@launcher
def _get_kernel_shared(heads: int, mq: int, fp8: bool):
    return _kernel_shared().compile_score_prefill_shared(heads, mq, fp8)


# Small work (<= 128 query rows x heads, or blocks x 256-row tiles x requests x heads below SMALL_WORK) takes
# ``kernel_shared`` with 128-row tiles on either route; larger work takes it with 256-row tiles on the exact route,
# ``kernel.py`` on the fp8 route. ``kernel_shared`` aims at SMALL_TARGET workgroups below MID_WORK (a one-block
# workgroup exposes its whole prologue), at TARGET_WORKGROUPS above.
SMALL_WORK = 4096
MID_WORK = 16384
SMALL_TARGET = 512
# Tests / benchmarks: pin the kernel (True: kernel_shared), its 32-row tiles per wave, or the workgroup target.
PIN_SHARED: bool | None = None
PIN_MQ: int | None = None
PIN_TARGET: int | None = None
# kernel.py stages a row's scores to 64 B stores (``kernel.SPAN``); False (or constants.INDEX_SCORE_STAGED = False)
# stores each 4-block group as it completes, the same bytes.
STAGED_STORES = True


def supports_nvfp4_index_score_prefill(
    q_idx: torch.Tensor, score: torch.Tensor
) -> bool:
    """gfx950, 1, 2 or 4 index heads, contiguous e4m3 query rows [total_q, H, 128]."""
    from ..gfx950 import on_gfx950

    heads = q_idx.shape[1]
    return (
        on_gfx950()
        and q_idx.dtype == torch.float8_e4m3fn
        and heads in (1, 2, 4)
        and q_idx.shape[2] == _kernel.D
        and q_idx.stride(2) == 1
        and q_idx.stride(1) == _kernel.D
        and q_idx.stride(0) == heads * _kernel.D
        and score.shape[0] == heads
        and score.stride(2) == 1
    )


def nvfp4_index_score_prefill(
    q_idx: torch.Tensor,  # [total_q, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, total_q, >= cdiv(max_seq_len, 128)] fp32
    block_table: torch.Tensor,  # [batch, >= max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    init_blocks: int,
    local_blocks: int,
    max_query_len: int,
    max_seq_len: int,
    exact: bool = False,
    target_workgroups: int | None = None,
    causal_grid: bool = False,
    stream=None,
) -> None:
    """The prefill index scores (``reference.prefill_scores_reference``) for 1, 2 or 4 index heads, fp8 keys or
    (``exact``) exact f16 keys.

    Args:
        target_workgroups: the large-work workgroup target instead of TARGET_WORKGROUPS.
        causal_grid: allows the compact one-head grid of live causal tasks (the page-gather caller) when the target
            covers four-block segments; other shapes keep their grid.
        stream: the launch stream (default the current one).
    """
    batch = cu_seqlens_q.shape[0] - 1
    total_q, heads, _ = q_idx.shape
    assert _cdiv(max_seq_len, _kernel.BLK) <= score.shape[2]
    exe, nseg, qt, grid, shared = _score_plan(
        heads, batch, max_query_len, max_seq_len, q_idx.device, exact,
        target_workgroups, causal_grid,
    )
    lut = () if shared else (nvfp4_e4m3_lut(q_idx.device).data_ptr(),)
    _run_compiled(
        exe,
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        *lut,
        score.data_ptr(),
        block_table.data_ptr(),
        cu_seqlens_q.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(block_table.stride(0)),
        int(score.stride(1)),
        int(nseg),
        int(qt),
        int(total_q),
        int(batch),
        int(((batch - 1) * block_table.stride(0) + block_table.shape[1]) * 4),
        int(init_blocks),
        int(local_blocks),
        int(score.stride(0)),
        int(grid),
        stream if stream is not None else _current_stream(),
    )


def _score_plan(heads, batch, max_query_len, max_seq_len, device, exact=False,
                target_workgroups=None, causal_grid=False):
    """The scorer, segments, query tiles, grid and route of a call; shared by the standalone and gathered chains."""
    max_blocks = _cdiv(max_seq_len, _kernel.BLK)
    work = max_blocks * _cdiv(max_query_len, 256) * batch * heads
    small = max_query_len * heads <= 128 or work < SMALL_WORK
    shared = exact or (small if PIN_SHARED is None else PIN_SHARED)
    mq = PIN_MQ or (1 if small else 2)
    target = PIN_TARGET or (
        SMALL_TARGET
        if shared and work < MID_WORK
        else (target_workgroups or TARGET_WORKGROUPS)
    )
    tile_q = _kernel_shared().tile_q(mq) if shared else _kernel.TILE_Q
    qt = _cdiv(max_query_len * heads, tile_q)
    # kernel.py: segments of whole 4-block store groups; shared: single blocks
    min_seg = 1 if shared else _kernel.GROUP
    max_nseg = max(1, min(MAX_SEGMENTS, max_blocks // min_seg))
    nseg = max(1, min(max_nseg, target // max(1, qt * batch)))
    slots = WORKGROUPS_PER_CU * _num_cus(device)
    while nseg < max_nseg:
        rounds = qt * batch * nseg / slots
        if rounds / math.ceil(rounds) >= MIN_ROUND_FILL:
            break
        nseg += 1
    compact = (
        causal_grid and not shared and batch == 1 and heads == 1
        and qt in (32, 64)
        and qt * _cdiv(max_blocks, _kernel.GROUP) <= target
    )
    grid = _cdiv(qt * nseg * batch, 8) * 8
    if compact:
        groups = _cdiv(max_seq_len, _kernel.TILE_Q)
        tiles = min(qt, groups)
        # Covers the last partial tile and shorter replayed query lengths.
        payload = tiles * groups - tiles * (tiles - 1) // 2
        # Fewer active tasks underfill the two-workgroup-per-CU scorer.
        compact = payload >= slots
        if compact:
            grid = _cdiv(payload + tiles, 8) * 8
    exe = (
        _get_kernel_causal() if compact
        else _get_kernel_shared(heads, mq, not exact) if shared
        else _get_kernel(heads, STAGED_STORES and constants.INDEX_SCORE_STAGED)
    )
    return exe, nseg, qt, grid, shared
