# SPDX-License-Identifier: MIT
"""Combine FFN branches directly into the native reduce-scatter input layout.

The output and padded send buffer belong to this invocation (or its captured
graph). No cross-call cache, peer publication, or alternate reduction arithmetic.
Aligned and unsupported inputs retain the original operation.
"""

import torch
import torch.distributed as dist
import triton as tr
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.deepseek_v41.row_sharded_mhc import (
    MIN_ROWS,
    _scatter_fake,
    v41_mhc_reduce_scatter,
)


def supports_fused_combine(routed, shared, tp_size):
    return (
        tp_size == 4
        and routed.ndim == shared.ndim == 2
        and routed.shape == shared.shape
        and routed.shape[0] >= MIN_ROWS
        and routed.shape[0] % tp_size != 0
        and routed.shape[1] == 5120
        and routed.dtype == shared.dtype == torch.bfloat16
        and routed.is_cuda
        and routed.device == shared.device
        and routed.is_contiguous()
        and shared.is_contiguous()
    )


@tr.jit(do_not_specialize=["ELEMENTS", "PADDED"])
def _combine_pad(
    Routed,
    Shared,
    Combined,
    ELEMENTS,
    PADDED,
    BLOCK: tl.constexpr,
):
    # H=5120 is a multiple of BLOCK=1024: every tile is entirely valid or padding.
    base = tl.program_id(0) * BLOCK
    index = base + tl.arange(0, BLOCK)
    routed = tl.load(Routed + index, base < ELEMENTS, other=0).to(tl.float32)
    shared = tl.load(Shared + index, base < ELEMENTS, other=0).to(tl.float32)
    tl.store(Combined + index, routed + shared, base < PADDED)


def _combined_scatter_fake(routed, shared):
    return _scatter_fake(routed)


@torch_compile_guard(mutates_args=[], gen_fake=_combined_scatter_fake)
def v41_ffn_combine_scatter(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    group = get_tp_group()
    if not supports_fused_combine(routed, shared, group.world_size):
        return v41_mhc_reduce_scatter(routed + shared)
    output = _scatter_fake(routed)
    combined = routed.new_empty((output.shape[0] * group.world_size, routed.shape[1]))
    _combine_pad[(tr.cdiv(combined.numel(), 1024),)](
        routed,
        shared,
        combined,
        routed.numel(),
        combined.numel(),
        BLOCK=1024,
        num_warps=4,
    )
    dist.reduce_scatter_tensor(output, combined, group=group.device_group)
    return output
