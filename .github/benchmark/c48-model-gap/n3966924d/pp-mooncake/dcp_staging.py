# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather DCP tokens into a contiguous, registered sender slot."""

import torch


def stage_dcp_tokens(
    cache: torch.Tensor,
    source_blocks: torch.Tensor,
    out: torch.Tensor,
    *,
    source_logical_size: int,
    source_kernel_size: int,
    local_token_start: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> None:
    """Gather into out on the caller's stream, respecting packed layer strides."""
    token_bytes = out.shape[1]
    cache_bytes = cache.view(torch.uint8)
    stride = cache_bytes.stride(0)
    if stride % token_bytes:
        raise ValueError("MLA staging requires a whole-token block stride")
    stride_tokens = stride // token_bytes
    rows = (cache.shape[0] - 1) * stride_tokens + source_kernel_size
    flat = cache_bytes.as_strided((rows, token_bytes), (token_bytes, 1))
    local = torch.arange(
        local_token_start,
        local_token_start + out.shape[0],
        dtype=torch.int64,
        device=cache.device,
    )
    position = (
        local // interleave_size * dcp_size + dcp_rank
    ) * interleave_size + local % interleave_size
    logical_offset = position % source_logical_size
    physical = (
        source_blocks[position // source_logical_size]
        * (source_logical_size // source_kernel_size)
        + logical_offset // source_kernel_size
    )
    indices = physical * stride_tokens + logical_offset % source_kernel_size
    torch.index_select(flat, 0, indices, out=out)


def staged_write_spans(
    destination_blocks: list[int],
    block_size: int,
    token_start: int,
    num_tokens: int,
) -> list[tuple[int, int, int, int]]:
    """Return (staging token offset, destination block, offset, count)."""
    if block_size <= 0 or token_start < 0 or num_tokens < 0:
        raise ValueError("Invalid staging token range")
    if token_start + num_tokens > len(destination_blocks) * block_size:
        raise ValueError("Staged write exceeds destination blocks")
    spans = []
    copied = 0
    while copied < num_tokens:
        index, offset = divmod(token_start + copied, block_size)
        count = min(block_size - offset, num_tokens - copied)
        spans.append((copied, destination_blocks[index], offset, count))
        copied += count
    return spans
