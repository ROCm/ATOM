# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental TP QKV / replicated o_proj token transitions for MiniMax-M3.

Attention keeps TP's global-token metadata and local heads. Only its output
is exchanged for local tokens. FFNs gather those tokens, compute TP partials,
then reduce the attention/FFN increment through one all-reduce. The accumulated
replicated residual remains outside the reduction quantizer.
Shape-changing collectives stay opaque to piecewise compilation.
"""

import torch
from aiter import dtypes
from aiter.dist.parallel_state import get_tp_group

from atom.config import get_current_atom_config
from atom.distributed.ulysses_sp import _all_gather_tokens, ulysses_gather_heads
from atom.model_ops.minimax_m3.attention_fp8 import gather_heads_fp8
from atom.utils import mark_spliting_op
from atom.utils.custom_register import direct_register_custom_op


def enabled() -> bool:
    """Construction-time flag, snapshotted in Config and sent to every worker."""
    return bool(getattr(get_current_atom_config(), "m3_tp_replicated_o_proj", False))


def _pad_tokens(x: torch.Tensor, world: int) -> torch.Tensor:
    pad = (-x.shape[0]) % world
    if pad:
        return torch.cat((x, x.new_zeros((pad, *x.shape[1:]))), dim=0)
    return x


def _attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor,
    fp8_output: bool,
    world: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = (q.shape[0] + world - 1) // world
    return (
        torch.empty(
            (local, q.shape[1] * world),
            device=q.device,
            dtype=dtypes.fp8 if fp8_output else q.dtype,
        ),
        torch.empty((local, 1), device=q.device, dtype=torch.float32),
    )


@mark_spliting_op(is_custom=True, gen_fake=_attention_fake, mutates_args=[])
def minimax_m3_tp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor,
    fp8_output: bool,
    world: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    group = get_tp_group()
    assert group.world_size == world == 4
    # Call ordinary TP attention directly. No QKV exchange, metadata rewrite,
    # or padded positions/KV writes: padding starts AFTER attention has run.
    output = attn.impl.forward(
        query=q, key=k, value=v, position=positions, q_scale=None, qkv=qkv
    )
    output = _pad_tokens(output.reshape(q.shape[0], q.shape[1]), world)
    if output.shape[0] == 0:
        return _attention_fake(q, k, v, positions, layer_name, qkv, fp8_output, world)
    if fp8_output:
        return gather_heads_fp8(output, group=group)
    output = ulysses_gather_heads(output, group=group)
    # A fixed tuple schema keeps graph tracing independent of batch size. The
    # BF16 consumer ignores this scale; the FP8 consumer always receives it.
    return output, torch.ones(
        (output.shape[0], 1), device=q.device, dtype=torch.float32
    )


def _local_residual_fake(x: torch.Tensor, world: int) -> torch.Tensor:
    return x.new_empty(((x.shape[0] + world - 1) // world, x.shape[1]))


def minimax_m3_tp_local_residual(x: torch.Tensor, world: int) -> torch.Tensor:
    group = get_tp_group()
    assert group.world_size == world == 4
    padded = _pad_tokens(x, world)
    local = padded.shape[0] // world
    start = group.rank_in_group * local
    # Custom op outputs must not alias inputs, including the no-padding case.
    return padded[start : start + local].clone()


def _gather_fake(x: torch.Tensor, world: int) -> torch.Tensor:
    return x.new_empty((x.shape[0] * world, *x.shape[1:]))


def minimax_m3_tp_gather(x: torch.Tensor, world: int) -> torch.Tensor:
    """Gather dense/shared FFN inputs; byte views preserve FP8 bits."""
    group = get_tp_group()
    assert group.world_size == world == 4
    if x.numel() == 0:
        return _gather_fake(x, world)
    dtype = x.dtype
    x = x.contiguous()
    if x.element_size() == 1:
        x = x.view(torch.bfloat16)
    return _all_gather_tokens(x, group=group).view(dtype)


def _moe_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
    world: int,
) -> torch.Tensor:
    return _gather_fake(hidden_states, world)


def minimax_m3_tp_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
    world: int,
) -> torch.Tensor:
    layer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    group = get_tp_group()
    assert group.world_size == world == 4
    return layer.forward_impl(hidden_states, router_logits, token_group=group)


def _complete_fake(
    partial: torch.Tensor, attention: torch.Tensor, tokens: int, world: int
) -> torch.Tensor:
    return partial.new_empty((tokens, partial.shape[1]))


def minimax_m3_tp_complete(
    partial: torch.Tensor, attention: torch.Tensor, tokens: int, world: int
) -> torch.Tensor:
    """Reduce FFN partials and the owner's attention output in one collective.

    Every rank has all FFN rows but owns only one attention-output interval.
    The accumulated input residual is already replicated and stays outside
    this op. The next input norm (or final norm) adds it without a collective.
    QuickReduce therefore quantizes only this layer's increment.
    """
    group = get_tp_group()
    assert group.world_size == world == 4
    assert partial.shape == (attention.shape[0] * world, attention.shape[1])
    assert 0 <= tokens <= partial.shape[0]
    start = group.rank_in_group * attention.shape[0]
    partial[start : start + attention.shape[0]].add_(attention)
    if tokens == 0:
        return partial[:0].clone()
    # Drop padded rows before reduction; they never enter the next layer.
    return group.all_reduce(partial[:tokens].contiguous(), ca_fp8_quant=False)


# Only attention needs a piecewise split for dynamic attention metadata. Keep
# the other opaque operators inside captured FFN subgraphs, like moe_forward,
# rather than adding four graph boundaries per layer.
for _impl, _fake, _mutates in (
    (minimax_m3_tp_local_residual, _local_residual_fake, []),
    (minimax_m3_tp_gather, _gather_fake, []),
    (minimax_m3_tp_moe, _moe_fake, ["hidden_states"]),
    (minimax_m3_tp_complete, _complete_fake, ["partial"]),
):
    direct_register_custom_op(
        op_name=_impl.__name__, op_func=_impl, mutates_args=_mutates, fake_impl=_fake
    )
