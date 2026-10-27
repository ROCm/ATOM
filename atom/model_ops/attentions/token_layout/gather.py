# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Some rows of a token axis, out of several tensors in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row(source, output, row, out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    for base in tl.static_range(0, WIDTH, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        word = tl.load(source + row * WIDTH + offs, mask)
        tl.store(output + out * WIDTH + offs, word, mask)


@triton.jit
def _gather_rows(rows, sources, outputs, WIDTHS: tl.constexpr, BLOCK: tl.constexpr):
    out = tl.program_id(0)
    row = tl.load(rows + out).to(tl.int64)
    for i in tl.static_range(len(WIDTHS)):
        _copy_row(sources[i], outputs[i], row, out, WIDTHS[i], BLOCK)


def gather_rows(rows, *tensors, dim=0, out=None):
    """Each of `tensors` at `rows` of its token axis `dim`, every dim before
    it of size 1. A row is copied as 32-bit words, whatever its dtype. `out`:
    contiguous tensors, one per tensor, to write into instead of new ones."""
    if not tensors:
        return ()
    count = rows.numel()
    shapes = [(*t.shape[:dim], count, *t.shape[dim + 1 :]) for t in tensors]
    if out is None:
        outputs = tuple(t.new_empty(shape) for t, shape in zip(tensors, shapes))
    else:
        outputs = tuple(out)
        if [tuple(o.shape) for o in outputs] != shapes:
            raise ValueError("gather_rows outputs disagree with the rows they take")
    if count:
        _gather_rows[(count,)](
            rows,
            tuple(_words(t, dim) for t in tensors),
            tuple(_words(t, dim) for t in outputs),
            WIDTHS=tuple(_words(t, dim).shape[1] for t in tensors),
            BLOCK=1024,
        )
    return outputs


def _words(t, dim):
    return t.view(t.shape[: dim + 1].numel(), -1).view(torch.int32)
