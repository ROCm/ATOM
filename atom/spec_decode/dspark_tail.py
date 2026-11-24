# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Default-off switches for the V4.1 DSpark draft tail of a decode step.

After ROCm/ATOM#3645:

  ATOM_DSPARK_LEAN_HEAD=1        the draft's head + Markov block without the
                                 launches nothing reads (anchor copy, W1[x]
                                 stack, the confidence head while no
                                 confidence schedule consumes it)
  ATOM_V41_DRAFT_ATTN_NO_CAT=1   the BF16 draft attention's concatenating path
                                 reads the window and the block where they
                                 lie (no torch.cat)

Unset, nothing here runs: `install` returns before importing a model module.
The switches are put in at construction, beside the draft model's own code
rather than inside it, so the default path's files -- and with them its Triton
and torch.compile cache keys -- are untouched. Both give the same draft ids
and attention bytes as the paths they replace.
"""

import logging
import os

logger = logging.getLogger("atom")


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def install(drafter) -> tuple[str, ...]:
    """Put the switches that are set into ``drafter``'s model.

    Returns the names installed; ``()`` with none set, before any import.
    """
    lean_head = _flag("ATOM_DSPARK_LEAN_HEAD")
    # Only beside the Triton `sparse_attn`, its default.
    no_cat = (
        _flag("ATOM_V41_DRAFT_ATTN_NO_CAT")
        and os.environ.get("ATOM_USE_TRITON_ATTN", "1") == "1"
    )
    if not (lean_head or no_cat):
        return ()
    model = getattr(drafter, "model", None)
    from atom.models.deepseek_v41 import dspark as v41

    if not isinstance(model, v41.DeepseekV41DSpark):
        logger.warning(
            "dspark tail: the draft model is %s, not the V4.1 DSpark; no switch "
            "installed",
            type(model).__name__,
        )
        return ()
    installed = []
    if no_cat and _install_no_cat(v41):
        installed.append("ATOM_V41_DRAFT_ATTN_NO_CAT")
    if lean_head and _install_lean_head(model, drafter):
        installed.append("ATOM_DSPARK_LEAN_HEAD")
    logger.info("dspark tail switches installed: %s", ", ".join(installed) or "none")
    return tuple(installed)


# ---------------------------------------------------------------------------
# ATOM_V41_DRAFT_ATTN_NO_CAT
# ---------------------------------------------------------------------------


def _install_no_cat(v41) -> bool:
    from atom.model_ops.deepseek_v41 import dspark as ops

    if getattr(v41.draft_attention, "_dspark_tail", False):
        return True
    if v41.draft_attention is not ops.draft_attention:
        logger.warning(
            "dspark tail: ATOM_V41_DRAFT_ATTN_NO_CAT declined, the draft model "
            "does not call `deepseek_v41.dspark.draft_attention`"
        )
        return False
    from atom.model_ops.sparse_attn_two_buffers import sparse_attn_two_buffers

    concatenating = ops.draft_attention

    def draft_attention(query, context_kv, draft_kv, sink, step, scale):
        """`dspark.draft_attention`; its concatenating path reads the window
        and the block where they lie: the same bytes out, one launch fewer.
        The index list is laid out as the ring slots (their own index or -1),
        then the block rows."""
        if step.src_indices is None and query.is_cuda:
            return sparse_attn_two_buffers(
                query,
                context_kv,
                draft_kv,
                sink,
                step.indices,
                scale,
                first_only=context_kv.shape[1],
            )
        return concatenating(query, context_kv, draft_kv, sink, step, scale)

    draft_attention._dspark_tail = True
    # `DraftAttention.forward` looks the name up in its module at every call.
    v41.draft_attention = draft_attention
    return True


# ---------------------------------------------------------------------------
# ATOM_DSPARK_LEAN_HEAD
# ---------------------------------------------------------------------------


def dspark_markov_argmax_into(base_logits, prev_ids, markov_w1, markov_w2, out, embed_out):
    """``dspark_markov_argmax`` for a caller that owns the ``W1[prev_ids]`` rows.

    ``embed_out``, ``[B, r]`` with unit stride along ``r`` (its rows may be
    strided: a column of a ``[B, T, r]`` block), receives them instead of a
    fresh tensor a position, so a block of positions needs no stack afterwards.
    The same two launches and the same ids as ``dspark_markov_argmax``.
    """
    from atom.model_ops import dspark_markov_argmax as op

    op.check_markov_inputs(
        "dspark_markov_argmax_into", base_logits, prev_ids, markov_w1, markov_w2, out
    )
    if base_logits.shape[0] == 0 or not op._fused_is_supported(
        base_logits, prev_ids, out, markov_w1, markov_w2
    ):
        embed_out.copy_(
            op._torch_dspark_markov_argmax(
                base_logits, prev_ids, markov_w1, markov_w2, out
            )
        )
        return embed_out
    return op._launch(
        base_logits, prev_ids, markov_w1, markov_w2, out, 0, None, embed_out
    )


def markov_block(base_logits, anchor_ids, w1, w2, embeds, per_position):
    """The block's ids ``[B, T]``, left to right; position k reads column
    k - 1 (the anchors for k = 0) and writes column k."""
    B, T, _ = base_logits.shape
    out_ids = anchor_ids.new_empty(B, T)
    for k in range(T):
        dspark_markov_argmax_into(
            base_logits[:, k],
            anchor_ids if k == 0 else out_ids[:, k - 1],
            w1,
            w2,
            out_ids[:, k],
            embeds[:, k if per_position else 0],
        )
    return out_ids


def lean_head_ok(model) -> bool:
    """The lean head needs the fused Markov op on a bias-free head."""
    from atom.model_ops.dspark_markov_argmax import markov_weights_unsupported

    head = model.head
    markov = model.mtp[-1].markov_head
    return (
        head is not None
        and head.bias is None
        and markov.fused_sample
        and markov_weights_unsupported(
            markov.markov_w1.weight, markov.markov_w2.weight
        )
        is None
    )


def head_and_sample_lean(model, normed, hc_hidden, anchor_ids, want_conf):
    """The greedy, unsharded ``_head_and_sample`` minus the launches nothing
    reads.

    The same head GEMM and the same Markov positions on the same operands, so
    the same ids, bit for bit. What is gone: the copy of the anchors into
    column 0 (position 0 reads them where they are), the stack of the
    ``W1[x]`` rows (each position writes its column), and, while nothing
    consumes a confidence (``want_conf`` False), the confidence head itself and
    the clone `DraftGraph.run` makes of its result. Then the second value is
    None, which `DSparkProposer` already takes.
    """
    B, T, _ = hc_hidden.shape
    head, last = model.head, model.mtp[-1]
    markov = last.markov_head
    w1, w2 = markov.markov_w1.weight, markov.markov_w2.weight
    base_logits = head.get_logits(normed).view(B, T, -1)
    # The W1[x_{k-1}] rows: the confidence head's input, a column a position;
    # one scratch column when nobody reads them.
    embeds = w1.new_empty(B, T if want_conf else 1, w1.shape[1])
    out_ids = markov_block(base_logits, anchor_ids, w1, w2, embeds, want_conf)
    confidence = last.confidence_head(hc_hidden, embeds) if want_conf else None
    return out_ids, confidence


def _install_lean_head(model, drafter) -> bool:
    if "head_and_sample" in vars(model):
        return True
    # Whether anything reads the confidence head: only the confidence schedule
    # does (`DSparkProposer`).
    want_conf = bool(
        getattr(getattr(drafter.config, "dspark", None), "confidence_schedule", False)
    )
    original = model.head_and_sample

    def head_and_sample(out, anchor_ids, num_draft, *, greedy=False):
        # The block drawn through a bound `draft_sampling` (lossless) and the
        # TP-sharded head keep their own paths.
        if (
            (model.draft_sampling is not None and not greedy)
            or model._sharded_markov_sample()
            or not lean_head_ok(model)
        ):
            return original(out, anchor_ids, num_draft, greedy=greedy)
        return head_and_sample_lean(model, *out, anchor_ids, want_conf)

    # An instance attribute: found before the class's method, by
    # `DSparkProposer._block_head` and by `forward_spec` alike.
    object.__setattr__(model, "head_and_sample", head_and_sample)
    return True
