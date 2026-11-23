# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A verify head: a model's head that picks each row's id itself and verifies
the drafts on them, in place of the logits, the sampler and the rejection
sampler.

``VerifyHead`` is what ``ModelRunner`` asks of one, after the forward and
outside its graph. ``ShardedPicks`` is the part every vocab-sharded LM head
shares: each row's id over the TP ranks' shards in a buffer of its own and
the verify on them (``verify_ids``). ``ArgmaxVerifyHead`` is ATOM's Triton
picks for a drafted model whose logits are its ``ParallelLMHead`` of the
final rows (``logits_are_head``); the mono LM head launch is another
(``atom.mono.lm_head``), in front of it where it serves.
"""

from typing import NamedTuple

import numpy as np
import torch
from aiter.dist.parallel_state import get_dp_group

from atom.distributed.pcp_utils import get_pcp_world_size
from atom.model_engine.token_constraints import has_constraints
from atom.model_ops.embed_head import ParallelLMHead
from atom.model_ops.rejection_sampler import RELAXED_TOP_N
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.verify_ids import IdVerifiers, VerifyOutput
from atom.utils.forward_context import get_forward_context


class LmHeadOutput(NamedTuple):
    """A step's LM head output (``ModelRunner.run_lm_head``): a verify head's
    ids and spec status where they stand for the step, else the logits;
    neither on a step the model ran whole."""

    verified: VerifyOutput | None
    logits: torch.Tensor | None


class VerifyHead:
    """A verify head. ``sample_token_ids(hidden)``: a verify step's (output
    ids [requests, rq], spec status [3, requests]) as the rejection sampler
    lays them out, from the forward's final rows; None where it serves no
    such step. Its ids stand for a step whose rows are all greedy, or drawn
    from their temperature (``samples``), within their top-p too where it
    draws so at the step's width (``draws_top_p``); every TP rank's alike
    (``ids_agree``: the runner broadcasts none)."""

    samples = False
    ids_agree = False

    def sample_token_ids(self, hidden: torch.Tensor):
        raise NotImplementedError

    def draws_top_p(self, rows: int) -> bool:
        """Whether an S-row step's draws keep within each row's top-p."""
        return False

    def samples_step(self, batch, spec, rows: int) -> bool:
        """Whether its token ids stand for the step ``batch`` of ``rows``
        forward rows (``spec``: its speculative layout) in place of the
        logits: a verify that reads nothing of them but each row's pick (no
        logprobs, structured output or required channel; exact or forced
        acceptance; no prefill context parallelism), every request verifying
        the full draft span, its rows all greedy (forced acceptance's always:
        the engine admits its requests greedy) or all sampled the ways it
        draws. ATOM's verify of a sampled step is sample-and-match: a draw a
        row stands for it."""
        if (
            spec is None
            or batch.any_logprobs
            or has_constraints(batch)
            or RELAXED_TOP_N > 1
            or get_pcp_world_size() > 1
            or (np.asarray(spec.num_draft_tokens_np) != spec.num_spec_steps).any()
        ):
            return False
        bs = batch.total_seqs_num
        temperatures = batch.temperatures[:bs]
        if not temperatures.any():
            return True
        noise = getattr(batch, "needs_independent_noise", None)
        return bool(
            self.samples
            and (temperatures > SAMPLER_EPS).all()
            and (batch.top_ks[:bs] == -1).all()
            and ((batch.top_ps[:bs] >= 1.0).all() or self.draws_top_p(rows))
            and not (noise is not None and noise.any())
        )


class ShardedPicks:
    """A vocab-sharded LM head's picks on a verify step: each row's id over
    every rank, ``max_rows`` rows of the buffer the head writes (made with
    it, before the KV cache is sized), and the verifies on the ids
    (``verifiers``; None: no drafter, no picks). ``input_ids()``: the step's
    input ids, the drafts the verify reads."""

    def __init__(self, verifiers: IdVerifiers | None, input_ids, max_rows: int, device):
        self.verifiers, self.input_ids = verifiers, input_ids
        self.row_ids = torch.empty(max_rows, dtype=torch.int32, device=device)

    def picks(self, rows: int) -> bool:
        """Whether an S-row step picks its ids: whole verify requests within
        the buffers."""
        return (
            self.verifiers is not None
            and rows <= self.row_ids.shape[0]
            and self.verifiers.serves(rows)
        )

    def verify(self, rows: int):
        """The verify of an S-row step on the ids the head wrote
        (``row_ids``): (output ids, spec status)."""
        spec = get_forward_context().spec_decode_metadata
        # the verifiers' request stride is the speculative config's; the step
        # reads drafts at its own
        assert spec is None or spec.num_spec_steps + 1 == self.verifiers.rq, (
            spec.num_spec_steps,
            self.verifiers.rq,
        )
        return self.verifiers.of(rows).launch(self.row_ids[:rows], self.input_ids())


def argmax_lm_head(model) -> ParallelLMHead | None:
    """``model``'s LM head (its own, or the language model's it wraps) where
    its logits are that head of its final rows (``logits_are_head``), else
    None. Not under pure DP: there the logits may be a DP collective, which
    every DP rank must enter alike, while a rank falls back to them by its
    own batch. Nor over the PCP ranks (context or Ulysses sequence
    parallelism): no step there takes a head's ids."""
    while not hasattr(model, "lm_head") and hasattr(model, "language_model"):
        model = model.language_model
    head = getattr(model, "lm_head", None)
    if not (isinstance(head, ParallelLMHead) and head.logits_are_head):
        return None
    if head.tp_size == 1 and get_dp_group().world_size > 1:
        return None
    if get_pcp_world_size() > 1:
        return None
    return head


class ArgmaxVerifyHead(ShardedPicks, VerifyHead):
    """``lm_head``'s picks on a decode step of whole verify requests: each
    row's id over the vocab shards (``ParallelLMHead.argmax_of_logits``: the
    ranks' bests gathered, not the logits; a row of temperature T > 0 its
    draw of softmax(logits / T), ``verify_ids.HeadDraw``) and the verify on
    them; any other step None."""

    samples = ids_agree = True

    def __init__(
        self, lm_head: ParallelLMHead, verifiers: IdVerifiers, input_ids,
        max_rows: int,
    ):  # fmt: skip
        super().__init__(verifiers, input_ids, max_rows, lm_head.weight.device)
        self.lm_head = lm_head

    def sample_token_ids(self, hidden: torch.Tensor):
        rows = hidden.shape[0]
        context = get_forward_context().context
        if context is None or context.is_prefill or not self.picks(rows):
            return None
        self.lm_head.argmax_of_logits(
            self.lm_head.local_logits(hidden),
            out=self.row_ids[:rows],
            draw=self.verifiers.draw(),
        )
        return self.verify(rows)
