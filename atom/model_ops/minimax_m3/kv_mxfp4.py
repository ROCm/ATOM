# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 page geometry of MiniMax-M3's sparse KV and index caches
(``ATOM_M3_SPARSE_KV_MXFP4=1`` / ``--index-cache-dtype mxfp4``).

One 128-token page of one head holds, for K (or V, or the index key), the
128-wide token rows as e2m1 data (two elements a byte, the low nibble the even
element) and one e8m0 scale per 32 elements::

    [128 tokens x 64 B data][128 tokens x 4 B scale]      = 8704 B

This is the page the aiter writer (``fused_qknorm_idxrqknorm`` with
``kv_cache_dtype="mxfp4"``) produces and ``pa_decode_mxfp4`` /
``score_flydsl_fp4`` read: 68 B a token, 136 B for K and V together, against
the 264 B an fp8 K/V token costs (128 B of values and a 4 B scale, twice).
"""

import torch

PAGE_TOKENS = 128
HEAD_DIM = 128
DATA_BYTES = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // 32
DATA_PAGE_BYTES = PAGE_TOKENS * DATA_BYTES
PAGE_BYTES = PAGE_TOKENS * (DATA_BYTES + SCALE_BYTES)


def page_views(pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(data, scale) views of uint8 pages ``[..., PAGE_BYTES]``.

    data ``[..., 128, 64]`` and scale ``[..., 128, 4]`` keep the leading dims
    and their strides, so a ``[pages, heads, 8704]`` pool field becomes the
    ``[pages, heads, 128, 64]`` / ``[pages, heads, 128, 4]`` pair the kernels
    take, and a ``[pages, 8704]`` index cache the ``[pages, 128, 64]`` pair.
    """
    if pages.dtype != torch.uint8 or pages.shape[-1] != PAGE_BYTES:
        raise ValueError(
            f"MXFP4 pages are uint8 [..., {PAGE_BYTES}], got {pages.dtype} "
            f"{tuple(pages.shape)}"
        )
    if pages.stride(-1) != 1:
        raise ValueError("MXFP4 pages need a contiguous byte axis")
    lead, lead_strides = tuple(pages.shape[:-1]), tuple(pages.stride()[:-1])
    off = pages.storage_offset()
    data = pages.as_strided(
        (*lead, PAGE_TOKENS, DATA_BYTES), (*lead_strides, DATA_BYTES, 1), off
    )
    scale = pages.as_strided(
        (*lead, PAGE_TOKENS, SCALE_BYTES),
        (*lead_strides, SCALE_BYTES, 1),
        off + DATA_PAGE_BYTES,
    )
    return data, scale
