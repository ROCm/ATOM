# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pack a PP/TP1 producer's PAGEs into the consumer TP rank's physical layout.

The wire unit is a complete destination PAGE, not a logical token row: NVFP4
scales and SHUFFLE/page-16 draft planes need not be token-contiguous. Reuse
MLA landing's bounded slots, credits, scatter and completion protocol with
block_size=1 (one wire row is one physical PAGE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.kv_transfer.disaggregation.mha_head_transfer import consumer_head_slices
from atom.kv_transfer.disaggregation.pd_producer import (
    mla_staging_pool_bytes,
    send_worker_count,
)
from atom.utils import envs

MHA_LANDING_LAYOUT = "mha_pages_v1"
MHA_SLOT_BYTES = 8 << 20
MHA_POOL_SLOTS = 32
MHA_SCATTER_SEGMENTS = 4096


def mha_landing_model(config) -> bool:
    from atom.config import _is_minimax_m3_config

    return bool(
        envs.ATOM_PD_MHA_LANDING
        and getattr(config, "decode_context_parallel_size", 1) == 1
        and _is_minimax_m3_config(getattr(config, "hf_config", None))
    )


def mha_staging_reserve_bytes(config) -> int:
    if (
        not mha_landing_model(config)
        or getattr(config, "tensor_parallel_size", 1) != 1
        or config.kv_transfer_config.get("kv_role") not in ("kv_producer", "kv_both")
    ):
        return 0
    return mla_staging_pool_bytes(send_worker_count(config.kv_transfer_config))


def mha_landing_configured(config) -> bool:
    return bool(
        mha_landing_model(config)
        and getattr(config, "tensor_parallel_size", 1) > 1
        and config.kv_transfer_config.get("kv_role") == "kv_consumer"
    )


def mha_landing_reserve_bytes(config) -> int:
    if not mha_landing_configured(config):
        return 0
    return MHA_POOL_SLOTS * MHA_SLOT_BYTES + 3 * 8 * MHA_SCATTER_SEGMENTS


@dataclass(frozen=True)
class HeadPage:
    region: int
    consumer_region: int
    src_bytes: int
    dst_bytes: int
    first: int
    src_stride: int
    width: int


def head_page(
    region, consumer_region, src_bytes, dst_bytes, src_runs, dst_runs, rank, tp
):
    """Validate head splitting with the direct path's exact byte-layout rules."""
    cuts = consumer_head_slices(src_bytes, dst_bytes, src_runs, dst_runs, rank, tp)
    first, _, width = cuts[0]
    stride = cuts[1][0] - first if len(cuts) > 1 else src_bytes
    if any(x % 4 for x in (src_bytes, dst_bytes, first, stride, width)):
        raise ValueError("MHA landing requires 4-byte aligned head slices")
    if dst_bytes != width * len(cuts) or any(
        (s, d, n) != (first + i * stride, i * width, width)
        for i, (s, d, n) in enumerate(cuts)
    ):
        raise ValueError("MHA landing requires uniform head runs covering a PAGE")
    return HeadPage(region, consumer_region, src_bytes, dst_bytes, first, stride, width)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_head_pages(
        src,
        pages,
        dst,
        SRC_WORDS: tl.constexpr,
        DST_WORDS: tl.constexpr,
        FIRST: tl.constexpr,
        STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        page = tl.program_id(0)
        word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        block_id = tl.load(pages + page).to(tl.int64)
        src_offset = (
            block_id * SRC_WORDS + FIRST + word // WIDTH * STRIDE + word % WIDTH
        )
        value = tl.load(src + src_offset, mask=word < DST_WORDS)
        tl.store(dst + page * DST_WORDS + word, value, mask=word < DST_WORDS)

except ImportError:  # CPU layout/protocol tests do not require Triton.
    _gather_head_pages = None


def gather_head_pages(
    source: torch.Tensor, pages: torch.Tensor, out: torch.Tensor, plan: HeadPage
):
    """Copy only this rank's byte slices; no dequantization or KV format change."""
    if source.dtype != torch.uint8 or out.dtype != torch.uint8:
        raise ValueError("MHA gather requires byte views")
    if not source.is_contiguous() or not out.is_contiguous():
        raise ValueError("MHA gather requires contiguous PAGE and staging views")
    if source.numel() % plan.src_bytes or out.numel() != pages.numel() * plan.dst_bytes:
        raise ValueError("MHA gather PAGE/staging size mismatch")
    if not pages.numel():
        return
    if source.is_cuda:
        _gather_head_pages[(pages.numel(), triton.cdiv(plan.dst_bytes // 4, 256))](
            source.view(torch.int32),
            pages,
            out.view(torch.int32),
            SRC_WORDS=plan.src_bytes // 4,
            DST_WORDS=plan.dst_bytes // 4,
            FIRST=plan.first // 4,
            STRIDE=plan.src_stride // 4,
            WIDTH=plan.width // 4,
            BLOCK=256,
            num_warps=1,
        )
    else:
        # Same layout on CPU for byte-exact protocol tests with a synchronous NIC.
        src = source.reshape(-1, plan.src_bytes)
        dst = out.view(pages.numel(), plan.dst_bytes)
        for offset in range(0, plan.dst_bytes, plan.width):
            first = plan.first + offset // plan.width * plan.src_stride
            dst[:, offset : offset + plan.width] = src[
                pages, first : first + plan.width
            ]
