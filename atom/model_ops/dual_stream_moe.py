# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared dual-stream MoE dispatcher (custom op).

Registers `torch.ops.aiter.maybe_dual_stream_forward` once. Models that want
to use it (deepseek_v2, deepseek_v4, ...) register themselves in
`compilation_config.static_forward_context[prefix]` and dispatch via
`torch.ops.aiter.maybe_dual_stream_forward(hidden_states, prefix)`.

Caller contract (the MoE module looked up by `prefix`):
- has `_use_dual_stream: bool`
- has `single_stream_moe_forward(hidden_states) -> Tensor`
- has `dual_stream_moe_forward(hidden_states) -> Tensor`

Per-token-count gating (decode benefits from dual-stream, prefill doesn't):
controlled by `envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`.

Wrapping the dispatch as a torch custom op makes it opaque to
torch.compile / Dynamo (required because `torch.cuda.stream(...)` context
inside `dual_stream_moe_forward` cannot be traced); harmless in eager.
"""

import torch

from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op


def maybe_dual_stream_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    threshold = envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD
    num_tokens = hidden_states.shape[0]
    if self._use_dual_stream and 0 < num_tokens <= threshold:
        return self.dual_stream_moe_forward(hidden_states)
    return self.single_stream_moe_forward(hidden_states)


def _maybe_dual_stream_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="maybe_dual_stream_forward",
    op_func=maybe_dual_stream_forward,
    # Op returns a fresh tensor; never writes into `hidden_states`. Declaring
    # `mutates_args=["hidden_states"]` (the V2 original) misleads the
    # functionalization pass into inserting defensive input clones.
    mutates_args=(),
    fake_impl=_maybe_dual_stream_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
