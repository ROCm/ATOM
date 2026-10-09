# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU gather of preshuffled DSA index pages onto a DCP shard plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.kv_transfer.disaggregation.sharded_transfer import DCPShardPlan


@dataclass(frozen=True)
class DCPIndexGatherIndices:
    """GPU projection of a shared DCP shard plan for preshuffled index pages."""

    dst_pages: int
    src_block_id_per_token: torch.Tensor
    src_token: torch.Tensor
    src_token_tile: torch.Tensor
    src_token_in_tile: torch.Tensor
    valid: torch.Tensor


def prepare_dcp_index_gather_indices(
    plan: DCPShardPlan, device: torch.device
) -> DCPIndexGatherIndices:
    """Project the shared token plan to reusable GPU index tensors."""

    if plan.interleave_size != 1:
        raise ValueError(
            "Preshuffled index staging currently supports "
            f"interleave=1, got {plan.interleave_size}"
        )
    src_token = torch.as_tensor(plan.src_token, device=device, dtype=torch.int64)
    return DCPIndexGatherIndices(
        dst_pages=plan.dst_pages,
        src_block_id_per_token=torch.as_tensor(
            plan.src_block_id_per_run, device=device, dtype=torch.int64
        ),
        src_token=src_token,
        src_token_tile=torch.div(src_token, 16, rounding_mode="floor"),
        src_token_in_tile=src_token.remainder(16),
        valid=torch.as_tensor(plan.valid, device=device, dtype=torch.bool),
    )


def gather_dcp_preshuffled_index_pages(
    source: torch.Tensor,
    staging: torch.Tensor,
    indices: DCPIndexGatherIndices,
    index_head_dim: int,
    scheduler_block_size: int,
    block_ratio: int,
) -> int:
    """Convert producer index rows into compact consumer preshuffled pages.

    Prefill commonly allocates one-token physical pages, but the indexer views
    each ``block_ratio``-sized group as one scheduler block and writes MFMA
    preshuffled data across that contiguous group. Reconstruct that grouped
    page before gathering. Already-quantized key and scale bytes move directly;
    no dequantization or requantization occurs.

    Scale layout matches ``aligned_index_cache_dim``: one fp32 scale per token
    after the fp8 key bytes, not ``index_head_dim // 128`` scales.
    """

    source_page_size = source.shape[1]
    if scheduler_block_size % 16 or index_head_dim % 16:
        raise ValueError(
            "Preshuffled index staging requires scheduler block size and "
            f"head_dim multiples of 16, got {scheduler_block_size=} and "
            f"{index_head_dim=}"
        )
    if source_page_size * block_ratio != scheduler_block_size:
        raise ValueError(
            f"Source physical page {source_page_size} × ratio {block_ratio} "
            f"does not match scheduler block {scheduler_block_size}"
        )
    if indices.dst_pages == 0:
        return 0
    dst_pages = indices.dst_pages
    if staging.ndim != 2:
        raise ValueError(
            f"Index staging must be a 2-D [pages, bytes] slot, got {tuple(staging.shape)}"
        )
    if dst_pages > staging.shape[0]:
        raise ValueError(
            f"Index staging holds {staging.shape[0]} pages, needs {dst_pages}"
        )

    aligned_index_dim = source.shape[2]
    page_bytes = scheduler_block_size * aligned_index_dim * source.element_size()
    if page_bytes > staging.shape[1]:
        raise ValueError(
            f"Index staging slot is {staging.shape[1]} bytes wide, needs {page_bytes}"
        )
    if source.shape[0] % block_ratio:
        raise ValueError(
            f"Source physical page count {source.shape[0]} is not divisible by "
            f"block_ratio={block_ratio}"
        )
    source_bytes = source.view(torch.uint8).reshape(
        source.shape[0] // block_ratio, page_bytes
    )
    output = staging[:dst_pages, :page_bytes]
    output.zero_()

    token_tiles = scheduler_block_size // 16
    column_tiles = index_head_dim // 16
    source_keys = source_bytes[:, : scheduler_block_size * index_head_dim].reshape(
        source_bytes.shape[0], token_tiles, column_tiles, 16, 16
    )
    selected_keys = source_keys[
        indices.src_block_id_per_token,
        indices.src_token_tile,
        :,
        indices.src_token_in_tile,
        :,
    ]
    selected_keys.masked_fill_(~indices.valid[:, None, None], 0)
    output[:, : scheduler_block_size * index_head_dim].reshape(
        dst_pages, token_tiles, column_tiles, 16, 16
    ).copy_(
        selected_keys.reshape(dst_pages, token_tiles, 16, column_tiles, 16).permute(
            0, 1, 3, 2, 4
        )
    )

    # One fp32 scale per token, matching the aligned index-cache layout.
    key_bytes = scheduler_block_size * index_head_dim
    scale_bytes = scheduler_block_size * torch.float32.itemsize
    source_scales = source_bytes[:, key_bytes : key_bytes + scale_bytes].view(
        torch.float32
    )
    selected_scales = source_scales.reshape(
        source_bytes.shape[0], scheduler_block_size
    )[indices.src_block_id_per_token, indices.src_token]
    selected_scales.masked_fill_(~indices.valid, 0)
    output[:, key_bytes : key_bytes + scale_bytes].view(torch.float32).reshape(
        dst_pages, scheduler_block_size
    ).copy_(selected_scales.reshape(dst_pages, scheduler_block_size))

    return dst_pages
