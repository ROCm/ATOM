# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only decode admission shared by layer Mono and the native indexer."""

import torch

from atom.models.deepseek_v41.mono.config import MAX_MOE_ROWS, MAX_ROWS
from atom.utils.forward_context import get_forward_context


def step_supported(
    input_ids: torch.Tensor,
    inputs_embeds,
    dp_ranks: int = 0,
    *,
    max_rows: int = MAX_ROWS,
) -> bool:
    """Is this forward a decode step of up to ``max_rows`` rows? Its width,
    not the scheduled request count: a captured graph's width is the one
    its replays run, while the scheduled count differs from replay to replay.

    Under DP attention over ``dp_ranks`` ranks every rank runs its own requests
    (an idle one a dummy step), and the MoE spans them all: a step is mono on
    every rank or on none, of up to ``MAX_MOE_ROWS / dp_ranks`` rows a rank
    (the MoE's widest step their sum) and ``max_rows`` at most. It is only
    when no rank prefills
    (``running_tokens_are_unified``, agreed across DP); then every rank's width
    is the agreed ``running_tokens``, and an idle rank's step, requests or
    none, is a decode as well."""
    rows = input_ids.numel()
    max_rows = min(max_rows, MAX_MOE_ROWS // dp_ranks) if dp_ranks else max_rows
    if inputs_embeds is not None or not 0 < rows <= max_rows:
        return False
    fwd = get_forward_context()
    if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
        return False
    if dp_ranks and not fwd.context.running_tokens_are_unified:
        return False
    md = fwd.attn_metadata
    step = getattr(md, "step", None)
    if step is None or getattr(md, "image_mask", None) is not None:
        return False
    return step.decode and step.running_tokens == rows
