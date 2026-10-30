# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dtype-preserving packing and stream-ordered Ulysses communication."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_fields_kernel(
    source,
    output,
    TOKENS: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    SPEC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    peer = tl.program_id(1)
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = flat // WIDTH, flat % WIDTH
    source_col = tl.full((BLOCK,), 0, tl.int32)
    start = 0
    for field in tl.static_range(len(SPEC)):
        offset = SPEC[field][0]
        full_width = SPEC[field][1]
        shards = SPEC[field][2] or WORLD
        size = full_width // shards
        base = offset + (peer // (WORLD // shards)) * size
        source_col = tl.where(
            (col >= start) & (col < start + size), base + col - start, source_col
        )
        start += size
    value = tl.load(source + row * STRIDE + source_col, row < TOKENS, other=0)
    tl.store(output + peer * TOKENS * WIDTH + flat, value, row < TOKENS)


def pack_fields(source: torch.Tensor, spec: tuple, world: int) -> torch.Tensor:
    """Pack [tokens, fields] into [destination, tokens, local fields].

    ``spec`` contains (column offset, full width, head shards) triples. Zero
    shards means world; one replicates the field. Fewer shards than ranks
    replicate each KV head to its consecutive query-head owners. This is a
    pure copy, including indexer keys; no payload arithmetic or cast occurs.
    The source may have a padded row stride.
    """
    assert source.ndim == 2 and source.stride(1) == 1
    width = sum(size // (shards or world) for _, size, shards in spec)
    output = source.new_empty((world, source.shape[0], width))
    if output.numel():
        _pack_fields_kernel[(triton.cdiv(source.shape[0] * width, 1024), world)](
            source,
            output,
            source.shape[0],
            source.stride(0),
            width,
            world,
            tuple(spec),
            1024,
        )
    return output


def all_to_all_into(output: torch.Tensor, source: torch.Tensor, group) -> None:
    """Equal-split all-to-all on the caller's stream, with a PyTorch fallback.

    PyTorch's ProcessGroupNCCL uses a separate stream. PyNccl lets packing,
    communication and attention stay ordered on one stream and is also usable
    during HIP graph capture. Availability is fixed for the whole group.
    """
    communicator = getattr(group, "device_communicator", None)
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is None or pynccl.disabled:
        torch.distributed.all_to_all_single(output, source, group=group.device_group)
        return
    assert source.is_contiguous() and output.is_contiguous()
    assert source.shape == output.shape and source.shape[0] == group.world_size
    assert source.dtype == output.dtype
    assert source.device == pynccl.device and output.device == source.device
    from aiter.dist.device_communicators.pynccl_wrapper import (
        buffer_type,
        cudaStream_t,
        ncclDataTypeEnum,
    )

    # Compute pointer offsets once instead of constructing two tensor views per
    # peer. That Python/dispatcher overhead is visible at SP8 in eager prefill.
    count = source.numel() // group.world_size
    stride = count * source.element_size()
    src, dst = source.data_ptr(), output.data_ptr()
    dtype = ncclDataTypeEnum.from_torch(source.dtype)
    stream = cudaStream_t(torch.cuda.current_stream(source.device).cuda_stream)
    pynccl.group_start()
    try:
        for peer in range(group.world_size):
            pynccl.nccl.ncclSend(
                buffer_type(src + peer * stride),
                count,
                dtype,
                peer,
                pynccl.comm,
                stream,
            )
            pynccl.nccl.ncclRecv(
                buffer_type(dst + peer * stride),
                count,
                dtype,
                peer,
                pynccl.comm,
                stream,
            )
    finally:
        pynccl.group_end()
