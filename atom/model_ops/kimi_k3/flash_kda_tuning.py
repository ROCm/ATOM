# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 KDA prefill choices FlashKDA makes for its own shapes: the segment
length of a multi-sequence step and the gfx950 pass A / pass B schedules."""

from __future__ import annotations

import functools

import torch


def balanced_chunks_per_seg(
    cu_seqlens: torch.Tensor, num_heads: int, head_v_dim: int
) -> int | None:
    """Segment length sized on the step's total chunks, or None to leave the
    choice to FlashKDA (one sequence, or no segmentation).

    FlashKDA's ``_choose_chunks_per_seg`` sizes segments as if every sequence
    were as long as the longest: [8K, 4K, 2K, 2K] gets 8 segments of 64
    chunks where a lone 16K sequence gets 16 of 32, for the same work. This
    applies its thresholds and limits to the total instead, so one sequence
    keeps FlashKDA's answer.
    """
    if cu_seqlens.shape[0] < 3:
        return None
    from aiter.ops.triton._triton_kernels.kimi_delta_attn import flash_kda as fk

    seqs = fk._seq_bounds(cu_seqlens)
    chunk = fk.FLASH_KDA_CHUNK
    chunks = [-(-(eos - bos) // chunk) for bos, eos in seqs]
    n_max, n_total = max(chunks), sum(chunks)
    blocks = num_heads * max(1, head_v_dim // 32)
    if len(seqs) * blocks > fk.get_num_sms() or n_max < fk._SEG_MIN_CHUNKS:
        return None
    # FlashKDA's own choice for one sequence of the step's total length (it
    # also applies FlashKDA's CHUNK_DELTA_ATTN_FLASH_KDA_SEG override), capped
    # at the longest sequence: past it nothing is segmented either way.
    return max(
        1,
        min(
            n_max, fk._choose_chunks_per_seg(n_total, len(seqs), num_heads, head_v_dim)
        ),
    )


@functools.cache
def apply_flash_kda_schedules() -> None:
    """Process-wide gfx950 FlashKDA schedule overrides, once per process.
    Called from the prefill path, not at model build: FlashKDA does not import
    without a GPU.

    Pass A always takes its published wide tile. FlashKDA takes it only above
    2 blocks per CU, which H=12 with at most 16 segments never reaches, yet
    the wide tile is faster there; same arithmetic per element.

    Pass B runs the prefetching scan (subject to aiter's
    ``AITER_FDA_SCAN_PREFETCH``), which FlashKDA enables on gfx1250 only. Its
    schedule differs, which changes numerics.
    """
    from aiter.ops.triton._triton_kernels.kimi_delta_attn import flash_kda as fk

    if fk.DEVICE_ARCH != "gfx950":
        return
    fk._SCAN_PREFETCH = fk.AITER_FDA_SCAN_PREFETCH
    if getattr(fk._k2_gluon_schedule, "atom_k2ab_wide", False):
        return
    published = fk._k2_gluon_schedule

    @functools.cache
    def schedule(W, num_segs, H, kernel="k2_ab_fused_gluon"):
        if kernel == "k2_ab_fused_gluon":
            wide = fk.chunk_delta_attn_tuned_config(
                f"{kernel}_wide", fk._K2_GLUON_FALLBACK, backend="gluon"
            )
            if W % wide.kwargs["BW"] == 0:
                return wide
        return published(W, num_segs, H, kernel)

    schedule.atom_k2ab_wide = True
    fk._k2_gluon_schedule = schedule
