# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton gather of token-contiguous MLA pages onto a DCP shard plan."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One program copies one destination token. 256 B covers typical MLA token
# widths (576 B FP8 latent+rope) in a few unrolled loads without wasting
# registers on a 1024 B tile.
_MLA_GATHER_MAX_BLOCK = 256


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


def gather_dcp_mla_pages(
    source_bytes: torch.Tensor,
    dest: torch.Tensor,
    src_block_id_per_token: torch.Tensor,
    src_token: torch.Tensor,
    valid: torch.Tensor,
    page_bytes: int,
    token_bytes: int,
    n_tokens: int,
) -> None:
    """Launch the per-token MLA page gather kernel."""

    block = min(
        _MLA_GATHER_MAX_BLOCK, triton.next_power_of_2(max(int(token_bytes), 1))
    )
    _gather_dcp_mla_pages_kernel[(n_tokens,)](
        source_bytes,
        dest,
        src_block_id_per_token,
        src_token,
        valid,
        page_bytes,
        TOKEN_BYTES=token_bytes,
        BLOCK=block,
        num_warps=1,
    )
