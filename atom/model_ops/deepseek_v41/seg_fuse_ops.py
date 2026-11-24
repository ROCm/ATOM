# SPDX-License-Identifier: MIT
"""Guarded ops of `model_options.seg_fuse` (see
`atom.models.deepseek_v41_plugin.seg_fuse`).

Imported only when an item that needs one is on: importing this file registers
a custom op and binds AITER's `module_dsv41_seg_fuse`.
"""

import logging

import torch
from aiter import dtypes
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.dsv41_seg_fuse import silu_mul_mxfp8


def _silu_mul_mxfp8_fake(combined: torch.Tensor, limit: float) -> list[torch.Tensor]:
    width = combined.shape[-1] // 2
    lead = combined.shape[:-1]
    return [
        combined.new_empty((*lead, width), dtype=dtypes.fp8),
        combined.new_empty((*lead, width // 32), dtype=torch.float8_e8m0fnu),
    ]


@torch_compile_guard(mutates_args=[], gen_fake=_silu_mul_mxfp8_fake)
def v41_silu_mul_mxfp8(combined: torch.Tensor, limit: float) -> list[torch.Tensor]:
    """`[gate | up]` bf16 -> `[x_fp8, x_scale]`, the pair `w2` reads: clamped
    SiLU-mul rounded to BF16, then the group-32 MXFP8 quant, in one launch."""
    global _launched
    if not _launched:
        # Once a process: item 7's line at model build only says the shared
        # expert was given this op, not that a forward reached it.
        _launched = True
        logging.getLogger("atom").info(
            "[SEG-FUSE] item 7: first launch (%d rows, width %d)",
            combined.shape[0],
            combined.shape[-1] // 2,
        )
    x_fp8, x_scale = silu_mul_mxfp8(combined, limit)
    return [x_fp8, x_scale]


_launched = False
