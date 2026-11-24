# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP4 routed MoE H6144 I768 E128 top4 with an optional MXFP8 shared expert, dispatched by token count (README)."""

import torch

from .prepared_input import MoeInput, MoeInputCompact, prepare_mxfp4_moe, gather_mxfp4_moe
from .shared_expert import SharedWeights, set_dbo_query
from .chain import MoeWeights, mxfp4_moe, warm_up_mxfp4_moe

# the public entry; private names (DISPATCH, mxfp4_moe_experts, ...) stay in .chain
__all__ = ["MoeWeights", "SharedWeights", "mxfp4_moe", "warm_up_mxfp4_moe", "MoeInput", "MoeInputCompact",
           "prepare_mxfp4_moe", "gather_mxfp4_moe", "set_dbo_query", "unsupported_reason"]

H, I, E, TOPK = 6144, 768, 128, 4  # noqa: E741
SWIGLU_OAI = (1.702, 7.0, 1.0)  # alpha, limit, beta (up + beta)


def unsupported_reason(w13, w13_scale, w2, w2_scale, shared_w13, shared_s13, shared_w2, shared_s2,
                       swiglu=SWIGLU_OAI) -> str | None:
    """None when ``MoeWeights.from_row_major`` (with ``SharedWeights``) takes these checkpoint-layout tensors:
    MXFP4 routed experts [E, 2I, H/2] / [E, H, I/2] + e8m0 scales, MXFP8 shared expert [2I, H] / [H, I] + scales,
    SwiGLU-OAI (alpha, limit, beta) as the kernels compute it."""
    if any(abs(a - b) > 1e-6 for a, b in zip(swiglu, SWIGLU_OAI)):
        return f"needs SwiGLU-OAI (alpha, limit, beta) {SWIGLU_OAI}, got {tuple(swiglu)}"
    for name, t, shape, dtype in (
        ("w13", w13, (E, 2 * I, H // 2), torch.uint8), ("w13_scale", w13_scale, (E, 2 * I, H // 32), torch.uint8),
        ("w2", w2, (E, H, I // 2), torch.uint8), ("w2_scale", w2_scale, (E, H, I // 32), torch.uint8),
        ("shared w13", shared_w13, (2 * I, H), torch.float8_e4m3fn), ("shared w13 scale", shared_s13, (2 * I, H // 32), None),
        ("shared w2", shared_w2, (H, I), torch.float8_e4m3fn), ("shared w2 scale", shared_s2, (H, I // 32), None),
    ):
        if t is None or tuple(t.shape) != shape or (dtype is not None and t.dtype != dtype):
            got = None if t is None else (tuple(t.shape), t.dtype)
            return f"{name}: needs {shape} {dtype or 'e8m0'}, got {got}"
    return None
