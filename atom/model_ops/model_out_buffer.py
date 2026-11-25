# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_FINAL_NORM_INTO_OUTPUT=1 (default off): the final norm of a captured decode forward
writes straight into the runner's static graph output (`outputs[:n]`), so the runner's
`outputs[:n] = model(...)` is a copy of a view onto itself (torch skips it: same storage,
offset, sizes and strides) instead of a 1.6 us DtoD copy kernel + a graph gap a step.
Same bytes: the norm kernel is unchanged, only its output pointer differs."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

_TARGET: torch.Tensor | None = None


def enabled() -> bool:
    return os.environ.get("ATOM_FINAL_NORM_INTO_OUTPUT", "0") == "1"


@contextmanager
def target(buf: torch.Tensor):
    """Offer `buf` (the runner's output rows) to the model's final norm for one forward."""
    global _TARGET
    prev = _TARGET
    _TARGET = buf if enabled() else None
    try:
        yield
    finally:
        _TARGET = prev


def take(rows: int, hidden: int, dtype: torch.dtype, device) -> torch.Tensor | None:
    """The offered buffer, once, when it is exactly a contiguous [rows, hidden] `dtype` tensor."""
    global _TARGET
    buf = _TARGET
    if (
        buf is None
        or tuple(buf.shape) != (rows, hidden)
        or buf.dtype != dtype
        or buf.device != torch.device(device)
        or not buf.is_contiguous()
    ):
        return None
    _TARGET = None
    return buf
