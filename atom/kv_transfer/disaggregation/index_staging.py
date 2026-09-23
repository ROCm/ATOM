# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU gather of preshuffled DSA index pages onto a DCP shard plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from atom.kv_transfer.disaggregation.sharded_transfer import DCPShardPlan

# One program copies one destination token. 256 B covers typical MLA token
# widths (576 B FP8 latent+rope) in a few unrolled loads without wasting
# registers on a 1024 B tile.
_MLA_GATHER_MAX_BLOCK = 256


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


@triton.jit
def _gather_dcp_mla_pages_kernel(
    source_ptr,
    dest_ptr,
    src_block_id_ptr,
    src_token_ptr,
    valid_ptr,
    src_page_bytes,
    TOKEN_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one destination token's MLA bytes, or zero an invalid tail slot."""

    token_id = tl.program_id(0).to(tl.int64)
    valid = tl.load(valid_ptr + token_id).to(tl.int1)
    src_block = tl.load(src_block_id_ptr + token_id).to(tl.int64)
    src_token = tl.load(src_token_ptr + token_id).to(tl.int64)
    src_base = src_block * src_page_bytes + src_token * TOKEN_BYTES
    dst_base = token_id * TOKEN_BYTES
    for start in range(0, TOKEN_BYTES, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < TOKEN_BYTES
        src = (source_ptr + src_base + offs).to(tl.pointer_type(tl.uint8))
        dst = dest_ptr + dst_base + offs
        data = tl.load(src, mask=mask & valid, other=0)
        tl.store(dst, data, mask=mask)


def _mla_gather_block(token_bytes: int) -> int:
    return min(_MLA_GATHER_MAX_BLOCK, triton.next_power_of_2(max(int(token_bytes), 1)))


def gather_dcp_mla_pages(
    source: torch.Tensor,
    staging: torch.Tensor,
    indices: DCPIndexGatherIndices,
    scheduler_block_size: int,
) -> int:
    """Pack token-contiguous MLA bytes into complete DCP destination pages.

    The direct interleave=1 path issues one RDMA descriptor per token. Gathering
    the same bytes first lets the transport send pages, coalescing consecutive
    destination blocks. Quantized values, scales and padding are copied as bytes.
    """
    if not source.is_contiguous():
        raise ValueError("MLA staging requires contiguous source pages")
    if source.shape[0] == 0 or scheduler_block_size <= 0:
        raise ValueError("MLA staging requires source pages and a positive block size")
    page_bytes = source.stride(0) * source.element_size()
    if page_bytes % scheduler_block_size:
        raise ValueError("MLA page bytes must be divisible by the scheduler block size")
    dst_pages = indices.dst_pages
    if (
        staging.ndim != 2
        or staging.shape[0] < dst_pages
        or staging.shape[1] != page_bytes
    ):
        raise ValueError(
            "MLA staging must hold compact destination pages of the source byte width"
        )
    if not staging.is_contiguous():
        raise ValueError("MLA staging pages must be contiguous")
    if not dst_pages:
        return 0
    token_bytes = page_bytes // scheduler_block_size
    n_tokens = dst_pages * scheduler_block_size
    if indices.src_block_id_per_token.numel() != n_tokens:
        raise ValueError(
            f"MLA gather plan has {indices.src_block_id_per_token.numel()} tokens, "
            f"expected {n_tokens}"
        )
    source_bytes = source.view(torch.uint8).reshape(-1)
    dest = staging[:dst_pages].reshape(-1)
    valid = indices.valid
    if valid.dtype == torch.bool:
        valid = valid.view(torch.uint8)
    elif valid.dtype != torch.uint8:
        valid = valid.to(torch.uint8)
    _gather_dcp_mla_pages_kernel[(n_tokens,)](
        source_bytes,
        dest,
        indices.src_block_id_per_token,
        indices.src_token,
        valid,
        page_bytes,
        TOKEN_BYTES=token_bytes,
        BLOCK=_mla_gather_block(token_bytes),
        num_warps=1,
    )
    return dst_pages


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
