# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSpark draft LM head sharded over the DP group (pure DP attention, TP 1).

Under DP attention every rank holds the whole ``[V, dim]`` head, and the draft
block reads all of it for a handful of rows. This splits the vocab across the
DP group instead, the way the target's decode head does
(``ParallelLMHead._dp_sharded_logits``): the ranks all-gather their post-norm
rows and anchors, each projects every row onto its own ``V / dp`` slice, and
the Markov sampler reduces each position to ``(max, id)`` per shard with
``dspark_markov_argmax_tp`` over the DP group. Each rank keeps its own rows.

Same ids and confidences as the replicated path given the same logits: the
pick over shards is exact (see ``dspark_markov_argmax_tp``), and the shard
GEMM's logits must be bit-identical to the matching columns of the full GEMM
-- a property of the GEMM solutions picked for both shapes, which the 4-GPU
test checks.

Every collective here is sized from DP-agreed state only, so every rank takes
this path or none does; see ``can_use_dp_sharded_dspark_head``.
"""

import torch

from atom import plugin_host
from atom.distributed.comm_utils import gather_rows
from atom.model_ops.dspark_markov_argmax import (
    dspark_markov_argmax_tp,
    markov_weights_unsupported,
)
from atom.utils import envs


def can_use_dp_sharded_dspark_head(head, markov_head, context, dp_metadata, dp_group):
    """Whether this draft block samples off a DP vocab shard.

    Every DP rank must answer alike, or the gathers below deadlock. Each input
    is either process-wide static (the recipe's ``dspark_options.dp_sharded_head``
    and ``dp_sharded_head_max_rows``, head/markov weights and config, group
    size) or DP-agreed for the step:

    - ``running_tokens_are_unified`` -- the drafter states it for the DSpark
      block, whose height is ``running_bs`` (``ForwardMode.decide``'s DP
      reduction) times the configured width, and the warmup states it too;
    - ``running_tokens`` -- the same number on every rank when unified;
    - ``dp_metadata is not None`` -- set by ``_publish_draft_shape`` on every
      rank whenever ``data_parallel_size > 1``; absent outside a DP-reduced
      step (single GPU, plugin DP attention).

    ``is_draft`` / ``is_prefill`` are per-rank flags, but the draft block clears
    ``is_prefill`` and sets ``is_draft`` on every rank before it runs.
    """
    options = plugin_host.dspark_options()
    if not options.dp_sharded_head:
        return False
    if (
        head is None
        or head.tp_size != 1
        or head.bias is not None
        or not markov_head.fused_sample
        or markov_weights_unsupported(
            markov_head.markov_w1.weight, markov_head.markov_w2.weight
        )
        is not None
    ):
        return False
    if (
        context is None
        or dp_metadata is None
        or not getattr(context, "is_draft", False)
        or getattr(context, "is_prefill", False)
        or not getattr(context, "running_tokens_are_unified", False)
        or int(context.running_tokens) > options.dp_sharded_head_max_rows
    ):
        return False
    return dp_group.world_size > 1 and head.num_embeddings % dp_group.world_size == 0


def _gather_block_inputs(normed, anchor_ids, dp_group):
    """``[P * rows, T * dim]`` rows and ``[P * rows]`` anchors, rank-major, in
    one grouped collective (aiter's custom ``all_gatherv``, PyNCCL past it)."""
    comm = dp_group.device_communicator
    if comm is not None:
        return gather_rows(lambda ts: comm.all_gatherv(ts, dim=0), (normed, anchor_ids))
    return tuple(dp_group.all_gather(t, dim=0) for t in (normed, anchor_ids))


def dp_sharded_dspark_head_and_sample(
    normed: torch.Tensor,  # [B * T, dim] post-norm
    hc_hidden: torch.Tensor,  # [B, T, dim] pre-norm (confidence head's h_k)
    anchor_ids: torch.Tensor,  # [B]
    head_weight: torch.Tensor,  # [V, dim], replicated
    markov_head,
    confidence_head,
    dp_group,
    padded_rows: int,  # the DP-agreed request count, >= B
    mm=None,
):
    """``head_and_sample`` with the head's vocab split over ``dp_group``;
    returns this rank's ``(ids [B, T], confidence [B, T])``."""
    if mm is None:
        from aiter.tuned_gemm import tgemm

        mm = tgemm.mm
    B, T, dim = hc_hidden.shape
    world = dp_group.world_size
    rank = dp_group.rank_in_group
    vshard = head_weight.shape[0] // world
    vocab_start = rank * vshard
    # A loud tripwire: a height past the agreed one would hang every rank in
    # a mismatched gather instead.
    assert (
        B <= padded_rows
    ), f"DP sharded draft head: {B} rows > the DP-agreed {padded_rows}"
    rows = normed.view(B, T * dim)
    anchors = anchor_ids
    if B < padded_rows:
        pad = padded_rows - B
        rows = torch.cat([rows, rows.new_zeros(pad, T * dim)])
        anchors = torch.cat([anchors, anchors.new_zeros(pad)])
    all_rows, all_anchors = _gather_block_inputs(
        rows.contiguous(), anchors.contiguous(), dp_group
    )
    total = world * padded_rows
    # [P * rows * T, V / P] -> [P * rows, T, V / P]
    base_logits = mm(
        all_rows.view(total * T, dim), head_weight[vocab_start : vocab_start + vshard]
    ).view(total, T, vshard)
    markov_w1 = markov_head.markov_w1.weight
    markov_w2 = markov_head.markov_w2.weight
    use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER
    # Every rank samples every gathered row: position k + 1 needs position k's
    # id for its Markov bias on this rank's shard too.
    out_ids = anchor_ids.new_empty(total, T)
    markov_embeds = markov_w1.new_empty(total, T, markov_w1.shape[1])
    for k in range(T):
        dspark_markov_argmax_tp(
            base_logits[:, k],
            all_anchors if k == 0 else out_ids[:, k - 1],
            markov_w1,
            markov_w2,
            out_ids[:, k],
            vocab_start,
            dp_group,
            use_custom=use_custom,
            embed_out=markov_embeds[:, k],
        )
    own = slice(rank * padded_rows, rank * padded_rows + B)
    confidence = confidence_head(hc_hidden, markov_embeds[own])
    return out_ids[own], confidence
