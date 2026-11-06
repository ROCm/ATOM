# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte gather primitives, independent of attention family and rank ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GatherLayout:
    """One source unit's strided slices covering a complete destination unit.

    Units may be contiguous token rows or physical PAGEs. Ownership is supplied
    separately as the source index tensor; this describes only physical bytes.
    """

    source_bytes: int
    unit_bytes: int
    first: int
    src_stride: int
    width: int

    def __post_init__(self):
        if min(self.source_bytes, self.unit_bytes, self.src_stride, self.width) <= 0:
            raise ValueError("Invalid gather byte geometry")
        if (
            self.first < 0
            or any(
                x % 4
                for x in (
                    self.source_bytes,
                    self.unit_bytes,
                    self.first,
                    self.src_stride,
                    self.width,
                )
            )
            or self.unit_bytes % self.width
        ):
            raise ValueError("Gather slices must cover aligned 4-byte units")
        end = (
            self.first
            + (self.unit_bytes // self.width - 1) * self.src_stride
            + self.width
        )
        if end > self.source_bytes:
            raise ValueError("Gather slices exceed the source unit")


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_strided_units(
        src,
        indices,
        dst,
        SRC_WORDS: tl.constexpr,
        DST_WORDS: tl.constexpr,
        FIRST: tl.constexpr,
        STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        unit = tl.program_id(0)
        word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        block_id = tl.load(indices + unit).to(tl.int64)
        src_offset = (
            block_id * SRC_WORDS + FIRST + word // WIDTH * STRIDE + word % WIDTH
        )
        value = tl.load(src + src_offset, mask=word < DST_WORDS)
        tl.store(dst + unit * DST_WORDS + word, value, mask=word < DST_WORDS)

except ImportError:  # CPU layout/protocol tests do not require Triton.
    _gather_strided_units = None


def gather_units(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """Copy only this rank's byte slices; no dequantization or KV format change."""
    if source.dtype != torch.uint8 or out.dtype != torch.uint8:
        raise ValueError("KV gather requires byte views")
    if not source.is_contiguous() or not out.is_contiguous():
        raise ValueError("KV gather requires contiguous PAGE and staging views")
    if (
        source.numel() % plan.source_bytes
        or out.numel() != indices.numel() * plan.unit_bytes
    ):
        raise ValueError("KV gather unit/staging size mismatch")
    if not indices.numel():
        return
    if plan.first == 0 and plan.width == plan.source_bytes == plan.unit_bytes:
        torch.index_select(
            source.view(-1, plan.source_bytes),
            0,
            indices,
            out=out.view(-1, plan.unit_bytes),
        )
    elif source.is_cuda:
        _gather_strided_units[
            (indices.numel(), triton.cdiv(plan.unit_bytes // 4, 256))
        ](
            source.view(torch.int32),
            indices,
            out.view(torch.int32),
            SRC_WORDS=plan.source_bytes // 4,
            DST_WORDS=plan.unit_bytes // 4,
            FIRST=plan.first // 4,
            STRIDE=plan.src_stride // 4,
            WIDTH=plan.width // 4,
            BLOCK=256,
            num_warps=1,
        )
    else:
        # Same layout on CPU for byte-exact protocol tests with a synchronous NIC.
        src = source.reshape(-1, plan.source_bytes)
        dst = out.view(indices.numel(), plan.unit_bytes)
        for offset in range(0, plan.unit_bytes, plan.width):
            first = plan.first + offset // plan.width * plan.src_stride
            dst[:, offset : offset + plan.width] = src[
                indices, first : first + plan.width
            ]
