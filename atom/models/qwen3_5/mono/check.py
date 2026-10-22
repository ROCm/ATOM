# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layer-by-layer comparison of the Qwen3.5 mono path against the original decoder layers.

Debug only (``ATOM_MONO_CHECK=1``, meant for ``--enforce-eager``). The original layers run the step as
the reference chain; every layer also runs as its mono kernel on exactly the reference's input, the f32
stream ``residual + bf16(rank sum)`` the original layer's fused all-reduce norm normalizes (the original
layer gets the reduced output on rank 0 and zeros elsewhere, so its all-reduce reproduces it). A GDN layer's
state slots are saved before the mono kernel runs and restored before the original layer does, so both
advance the same state once. Rank 0 logs, per layer and token, how far mono's output stream (and a GDN
layer's new recurrent state) is from the reference's, and how far its mixer's contribution to the stream
(``a`` minus the bf16 input: before the MoE, whose baseline kernels are not deterministic) is from the
reference's (the bf16 rank sum of the mixer output the post-attention norm takes). As the yardstick
for the stream, ``rerun`` is the reference's own MoE run again on the same input (its kernels are not
deterministic) against its first run. The step returns the
reference's hidden states. Verify steps (MTP) are not checked.
"""

import logging

import torch

from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.mono.runtime.compare import compare
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _clone(args):
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def forward_checked(
    runner, input_ids: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    assert runner.T == 1, "config_refusal keeps ATOM_MONO_CHECK off a drafter"
    pos, ctx, slots, bt, gdn = runner.step_inputs(input_ids, positions)
    kvd = get_forward_context().kv_cache_data
    lm = runner.lm
    n = input_ids.numel()
    built = runner.widths[n]
    built["mailboxes"].begin_step()
    hidden = lm.embed_tokens(input_ids)
    residual = None
    x = hidden.float()
    scr = {}
    for i, op in enumerate(built["ops"]):
        layer = lm.layers[i]
        v = op.variant
        lay = v.scr_layout
        # the f32 post-mixer stream, one value per (value, tag) pair
        a_mb = v.scratch[lay["a"] : lay["a"] + n * runner.cfg.hidden * 8]
        lc = kvd[f"layer_{runner.layer_nums[i]}"]
        state = None
        if op.full:
            kv = (lc.k_cache, lc.v_cache, lc.k_scale, lc.v_scale, pos, ctx, bt, slots)
            got = op.forward(
                x, kv=kv, rope=(runner.cos, runner.sin), layer=i, begin=False
            )
        else:
            conv, ssm = lc.k_cache, lc.v_cache
            s_in, s_out = gdn[0].long(), gdn[1].long()
            saved = conv[s_in].clone(), ssm[s_in].clone()
            got = op.forward(x, gdn_state=(conv, ssm) + gdn, layer=i, begin=False)
            state = ssm[s_out].clone()
            conv[s_in] = saved[0]
            ssm[s_in] = saved[1]
        got_a = (
            a_mb.view(torch.int32).view(-1, 2)[:, 0].contiguous().view(torch.float32)
        )
        got_a = got_a.view(n, -1)
        # the original layer on the same input
        if residual is not None and layer.fuse_allreduce_norm and runner.rank != 0:
            hidden = torch.zeros_like(hidden)
        hooks = [
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda _m, args: scr.update(mix=args[0].clone())
            ),
            layer.mlp.register_forward_pre_hook(
                lambda _m, args: scr.update(moe_in=_clone(args))
            ),
        ]
        try:
            hidden, residual = layer(positions, hidden, residual)
        finally:
            for h in hooks:
                h.remove()
        # the baseline's own spread: its MoE again on the same input
        again = layer.mlp(*scr.pop("moe_in"))
        if layer.fuse_allreduce_norm:
            again = tensor_model_parallel_all_reduce(again)
        mix = tensor_model_parallel_all_reduce(scr.pop("mix"))
        if layer.fuse_allreduce_norm:
            hidden = tensor_model_parallel_all_reduce(hidden)
        ref = residual.float() + hidden.float()
        rerun = residual.float() + again.float()
        xr = x.to(torch.bfloat16).float()
        if runner.rank == 0:
            for k in range(n):
                logger.info(
                    "Qwen3.5 mono check layer %d (%s) token %d/%d pos %d: stream %s | mixer %s"
                    " | rerun %s%s",
                    i,
                    "full" if op.full else "gdn",
                    k,
                    n,
                    int(pos[k]),
                    compare(got[k], ref[k]),
                    # the residual joins bf16-rounded (as the fused all-reduce norm's), so equal
                    # sums compare equal
                    compare(got_a[k] - xr[k], (xr[k] + mix[k].float()) - xr[k]),
                    compare(rerun[k], ref[k]),
                    (
                        ""
                        if state is None
                        else " | state " + compare(state[k], ssm[s_out[k]])
                    ),
                )
        x = ref
    return lm.norm.forward_static(
        lm.norm.weight.data, lm.norm.variance_epsilon, x, None
    ).to(torch.bfloat16)
