# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime aux capture shared by compiled and eager target forwards."""

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.utils.forward_context import get_forward_context


@torch_compile_guard(mutates_args=["buffer"], gen_fake=lambda tensor, buffer: None)
def capture_aux(tensor: torch.Tensor, buffer: torch.Tensor) -> None:
    """Read the live microbatch offset outside the compiled target region.

    Target and draft share embeddings. Draft invocations must not overwrite
    the target's aux rows, and TBO microbatches must write disjoint slices.
    """
    ctx = get_forward_context().context
    if not ctx.is_draft:
        tokens = ctx.ubatch_token_order
        if tokens is None:
            start = ctx.ubatch_token_offset
            buffer[start : start + tensor.shape[0]].copy_(tensor)
        else:
            buffer.index_copy_(0, tokens, tensor)
