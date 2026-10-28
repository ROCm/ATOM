# SPDX-License-Identifier: MIT
"""Owned copies of AITER's shared MegaMoE tail rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_tail_kernel(
    source,
    source_scale,
    out,
    out_scale,
    elements,
    rows,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offsets, offsets < elements, 0)
    tl.store(out + offsets, value, offsets < elements)
    if HAS_SCALE:  # noqa: SIM102 - discard the absent scale pointer at compile time
        # rows <= elements: the data offsets also cover every scale row.
        if pid * BLOCK < rows:
            scale = tl.load(source_scale + offsets, offsets < rows, 0)
            tl.store(out_scale + offsets, scale, offsets < rows)


def copy_tail(source, source_scale, out, out_scale):
    """Copy a contiguous rank/tile slice before the next shared-runtime launch.

    FP8 is copied as raw bytes, with its FP32 per-row scales in the same launch.
    BF16 final-norm rows need only the data copy. Caller-owned residual rows
    are already written by AITER's tail.
    """
    if source_scale is not None:
        source, out = source.view(torch.uint8), out.view(torch.uint8)
    _copy_tail_kernel[(triton.cdiv(out.numel(), 4096),)](
        source,
        source_scale,
        out,
        out_scale,
        out.numel(),
        out.shape[0],
        source_scale is not None,
        4096,
        num_warps=4,
    )
