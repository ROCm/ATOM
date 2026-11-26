# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the ATOM plugin, SGLang 0.5.22: a greedy draft
(``SGLANG_DSPARK_USE_GREEDY_DRAFT=1``) folded into the draft CUDA graph runs
ATOM's sharded fused Markov argmax instead of SGLang's Markov chain.

SGLang's folded draft tail (``DsparkDraftSampler.__call__``) computes the
draft's base logits over the whole vocab (the LM head's ``[B*T, V]``
all-gather) and then, per block position, the fp32 Markov bias
``W1[x] @ W2^T`` ``[B, V]``, an argmax over ``[B, V]`` and a TP broadcast of
the ids. Native ATOM drafts the same greedy block off each rank's vocab shard
(``DeepseekV41DSpark._head_and_sample_sharded``): one fused bias-GEMV +
argmax per position (``dspark_markov_argmax_tp``) whose ranks exchange only
``(max, id)``, no step logits and no ``[B*T, V]`` gather.

The patch is installed by this plugin's V4.1 draft wrapper
(``attach_shared_modules``), the model of the process's only
``DsparkDraftSampler``: it offers ``atom_v41_greedy_draft_block``, samples
from the anchor and has no confidence head. For a greedy draft the patched
tail calls that hook, broadcasts the ``[B, gamma]`` block once over SGLang's
TP-sync site (the same agreement SGLang's per-step broadcasts give: every
rank takes rank 0's chain), and writes ``sampler.out`` -- the buffer the
folded proposal and the verify epilogue read -- exactly as the original
does. It keeps no step logits, as SGLang's greedy draft keeps none. A
sampled draft and a hook that declines (`atom_v41_greedy_draft_block`
returns None) run the original. Rounds whose draft is not graph-folded (more
requests than the captured buckets) keep SGLang's eager proposal.

Shape guard: the sampler attributes this reads and the source shape of the
tail it replaces (``DsparkDraftSampler.__call__`` and ``draft_markov_block``);
where they differ, the original stays and one line logs why. The plugin runs
on SGLang 0.5.22 only (`sglang_version_guard` refuses any other at launch,
before this installs).
"""

from __future__ import annotations

import logging
from functools import wraps

from atom.plugin.sglang.patches.sglang_version_guard import (
    missing_source_needles,
    sglang_version,
)

logger = logging.getLogger("atom")

_SAMPLER_ATTRS = (
    "model",
    "gamma",
    "query_token_num",
    "sample_from_anchor",
    "greedy_draft",
    "out",
    "confidence_out",
    "_tp_sync",
)


# What the replaced tail does, so the fused block stays its equivalent: the
# anchor is each request's first query row, the chain goes through
# ``draft_markov_block`` and lands in ``self.out``; a greedy chain syncs each
# step's ids over the DSpark draft TP-sync site, keeps no step logits and
# draws no noise (it returns before advancing the noise offset).
_CALL_NEEDLES = (
    "input_ids.view(bs, self.query_token_num)[:, 0]",
    "draft_markov_block(",
    "greedy_draft=self.greedy_draft",
    "self.out[: draft_tokens.numel()].copy_(",
    "self.confidence_out is not None",
)
_BLOCK_NEEDLES = (
    "tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT",
    "if greedy_draft:",
    "return torch.stack(draft_tokens, dim=1), None",
)


def _upstream_shape_error(sampler_cls, block_fn=None) -> str | None:
    if "__call__" not in vars(sampler_cls):
        return "DsparkDraftSampler defines no __call__"
    missing = missing_source_needles(
        sampler_cls.__init__, [f"self.{name} =" for name in _SAMPLER_ATTRS]
    )
    if missing is None:
        return "DsparkDraftSampler.__init__ has no source"
    if missing:
        return f"DsparkDraftSampler no longer sets {missing}"
    missing = missing_source_needles(sampler_cls.__call__, _CALL_NEEDLES)
    if missing is None:
        return "DsparkDraftSampler.__call__ has no source"
    if missing:
        return f"DsparkDraftSampler.__call__ no longer reads {missing}"
    if block_fn is None:
        from sglang.srt.speculative.dspark_components import dspark_draft

        block_fn = getattr(dspark_draft, "draft_markov_block", None)
        if block_fn is None:
            return "dspark_draft.draft_markov_block is missing"
    missing = missing_source_needles(block_fn, _BLOCK_NEEDLES)
    if missing is None:
        return "draft_markov_block has no source"
    if missing:
        return f"draft_markov_block no longer reads {missing}"
    return None


def fused_greedy_tail(sampler, hidden_states, input_ids) -> bool:
    """Run the greedy block through the model's ATOM hook into
    ``sampler.out``; False (nothing written) when it does not apply."""
    if not sampler.greedy_draft:
        return False
    from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

    bs = hidden_states.shape[0] // sampler.query_token_num
    anchor = input_ids.view(bs, sampler.query_token_num)[:, 0]
    tokens = sampler.model.atom_v41_greedy_draft_block(
        hidden_states, anchor, sampler.gamma
    )
    if tokens is None:
        return False
    tokens = sampler._tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, tokens)
    sampler.out[: tokens.numel()].copy_(tokens.reshape(-1))
    return True


def install_dspark_fused_greedy_draft_patch() -> bool:
    """Install the fused greedy tail (idempotent). Returns whether active."""
    from sglang.srt.speculative.dspark_components import dspark_draft_sampler

    cls = dspark_draft_sampler.DsparkDraftSampler
    if getattr(cls, "_atom_v41_fused_greedy", False):
        return True
    reason = _upstream_shape_error(cls)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark: fused greedy draft tail not installed "
            "(%s); a greedy draft runs SGLang's Markov chain",
            reason,
        )
        return False
    original = cls.__call__

    @wraps(original)
    def __call__(self, hidden_states, input_ids):
        if fused_greedy_tail(self, hidden_states, input_ids):
            return None
        return original(self, hidden_states, input_ids)

    cls.__call__ = __call__
    cls._atom_v41_fused_greedy = True
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: a graph-folded greedy draft runs ATOM's "
        "sharded fused Markov argmax (DsparkDraftSampler.__call__; sglang %s)",
        sglang_version(),
    )
    return True
