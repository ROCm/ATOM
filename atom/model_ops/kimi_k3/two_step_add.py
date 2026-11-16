# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _two_step_add_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    c = tl.load(c_ptr + offs, mask=mask).to(tl.float32)
    ab = (a + b).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + offs, (ab + c).to(out_ptr.dtype.element_ty), mask=mask)


def two_step_add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``(a + b) + c`` over contiguous same-shape tensors in one launch,
    rounded to ``a``'s dtype after each add: bitwise the two torch adds."""
    assert a.shape == b.shape == c.shape
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
    out = torch.empty_like(a)
    grid = (triton.cdiv(a.numel(), 4096),)
    _two_step_add_kernel[grid](a, b, c, out, a.numel(), BLOCK=4096)
    return out
