# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reference kernel failure acceptance at the existing output boundary."""
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group
from atom.models.deepseek_v41.mono.gfx1250_diagnostics import _diagnostics
from atom.mono.runtime.step_begin import MISSING


@triton.jit
def _status(diag, fence, codes, sticky, LAYERS: tl.constexpr,
            FENCE_MISSING: tl.constexpr):
    failure = tl.load(diag + 256 * 8)
    missing = tl.load(fence + FENCE_MISSING) | tl.load(fence + FENCE_MISSING + 1)
    code = tl.maximum(tl.abs(failure), tl.where(missing != 0, 1, 0))
    code = tl.maximum(code, tl.load(sticky))
    x = tl.arange(0, 64)
    tl.store(codes + x, code, x < LAYERS)
    tl.store(codes + LAYERS, code)
    tl.store(sticky, code)


def finish_reference_step(metadata, diag, fence):
    cache = metadata.cache
    codes, _ = _diagnostics(cache, 1)
    _status[(1,)](diag, fence, codes, cache._gfx1250_diagnostic_sticky,
                 LAYERS=cache.geometry.layers, FENCE_MISSING=MISSING)
    group = get_tp_group()
    # Same rank agreement as the historical deferred-output contract.
    dist.all_reduce(codes[-1:], op=dist.ReduceOp.MAX, group=group.device_group)
    cache._gfx1250_diagnostic_sticky.copy_(codes[-1:])
