# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TileRT-path step's routing wrapper (as ``gfx1250.dispatch.SingleStepModel``) and its decode round.

``ModelRunner.run_model`` offers it each decode step (``step_override``). It serves the step (``TkStepRunner``: target,
sampling, verify and the next drafts on TkRank) or returns None and the production path runs. A served step bypasses
ATOM's sampler, rejection sampler and DSpark proposer: the drafter's ``compute_draft_kv`` / ``propose`` are wrapped to
do nothing / return the runner's drafts.

Served: one request (6 rows, a DSpark verify; not a warmup's dummy decode, not inside a graph capture), sampled
(temperature > 0, top_p 1, no top_k), no structured output, no required channel, no logprobs, inside the plan's
context. Sticky: a request TkRank admitted keeps its state there -- ATOM's caches do not get the KV it decodes -- so
every later step of it must be served; one that cannot raises instead of falling back. A step of another request (the
held one finished: ATOM sends no finish) re-admits.

ATOM's V4.1 state bookkeeping (``TkRound``, the metadata builder's ``DecodeRound``): the routing decision is made once
a step, when the builder lays the step out (``TkRound.service``), before ``prepare_state`` and ``step_override``; a step
TkStep will serve is laid out ``LEAN`` (no compress plans: no ATOM layer runs) and still gets the state probe. Its
speculative commit is TkStep's (``TkRound.committed``): ATOM's tentative state -- staged by the forward that does not
run -- is dropped, and the request's state cursor moved to where the verdict leaves it, as ``commit_tentative`` would
(``commit_served``), so the next step's probe passes and checks ATOM's position against TkRank's. Production steps
(no decision) keep every path unchanged.
"""

import logging
from functools import partial

import torch
from torch import nn

from atom.model_engine.model_hooks import DecodeRound
from atom.model_engine.token_constraints import has_constraints
from atom.models.deepseek_v41.mono.tk_step import ROWS, refusal
from atom.mono.runtime.step_override import StepOverrideModel
from atom.utils import envs
from atom.utils.forward_context import RoundService, get_forward_context

logger = logging.getLogger("atom")


def commit_served(cache, step, anchors) -> None:
    """A served verify step's speculative commit: ATOM's tentative state dropped (its staging was the forward's), each
    request's state cursor at its next step's start -- one past its accepted anchor's position, where
    ``TentativeState.commit`` puts it (prefix ``t`` of a request at ``position``: ``position + 1 + t``,
    ``speculative.py``) -- on the device, without a host round trip. The cursor's history ids are left: nothing of
    ATOM reads a held request's state again."""
    n = step.scheduled_bs
    rows = anchors[:n].to(step.positions.device, torch.int64)
    starts = step.positions.index_select(0, rows).to(cache.cursor.dtype) + 1
    cache.cursor[step.slots[:n].long(), 0] = starts.to(cache.cursor.device)
    cache.pending = None


class TkRound(DecodeRound):
    """The V4.1 metadata builder's (``builder``, ``use_decode_round``) decode round for ``TkStepModel`` (module
    docstring)."""

    def __init__(self, model, builder) -> None:
        self.model, self.builder = model, builder

    def service(self, batch, spans, running_bs, running_tokens, *, max_q_len, decode, tentative, is_prefill, resumes, image_rows):  # fmt: skip
        # a step the builder runs eagerly (its rows past the score workspace) gets no step override
        eager = partial(self.builder.step_needs_eager, batch)
        if self.model.decide(batch, running_bs, max_q_len, decode, eager):
            return RoundService.LEAN
        return RoundService.NONE

    def committed(self, metadata, anchors) -> bool:
        if not metadata.served_by_round:
            return False
        if self.model._served is None:
            raise RuntimeError(
                "V4.1 TileRT-path step: a step laid out for it was not served"
            )
        commit_served(metadata.cache, metadata.step, anchors)
        return True


class TkStepModel(StepOverrideModel, nn.Module):
    """The loaded V4.1 model, its decode steps offered to a ``TkStepRunner`` (``runner``: built at install, a
    collective over the TP group)."""

    def __init__(self, model: nn.Module, drafter, runner) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter
        self._runner = runner
        self._served = None
        self._decode_batch = None
        self._decision = None  # the batch of the step laid out for TkStep (``decide``), until ``step_override``
        self._why = None
        if drafter is not None:
            self._wrap_drafter(drafter)

    def _wrap_drafter(self, drafter) -> None:
        compute_draft_kv, propose = drafter.compute_draft_kv, drafter.propose

        def served_compute_draft_kv(*args, **kwargs):
            return (
                None if self._served is not None else compute_draft_kv(*args, **kwargs)
            )

        def served_propose(*args, **kwargs):
            if self._served is not None:
                return self._served.draft_ids(self._decode_batch.total_seqs_num)
            return propose(*args, **kwargs)

        drafter.compute_draft_kv, drafter.propose = (
            served_compute_draft_kv,
            served_propose,
        )

    def refusal(
        self, batch, running_bs: int, rows, decode: bool, eager=None
    ) -> str | None:
        """Why this step is not served (None: served): ``rows`` the request's verify rows, ``decode`` a DSpark verify
        of real requests (the builder's), ``eager()`` whether the runner will run it eagerly (no step override).
        """
        if envs.ATOM_MONO_CHECK:
            return "ATOM_MONO_CHECK"
        if batch is None or running_bs != 1 or rows != ROWS or not decode:
            return f"{running_bs} requests, {rows} rows, verify {decode}"
        # a warmup's dummy decode (ModelRunner.dummy_execution), or a step inside a stream capture: TkRank would
        # admit a request that is not one / replay its graphs inside another capture
        if batch.is_dummy_run or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            return "dummy run / graph capture"
        # build_for_cudagraph_capture's layout batch (a SimpleNamespace without the scheduler's sampling fields),
        # built before the capture starts: ATOM's own graph is captured as it is
        if getattr(batch, "temperatures", None) is None:
            return "graph capture layout (no sampling parameters)"
        if has_constraints(batch) or batch.any_logprobs:
            return "constraints / logprobs"
        if eager is not None and eager():
            return "an eager decode step"
        return self._runner.precheck(batch, self._anchor_position(batch))

    def decide(self, batch, running_bs: int, rows, decode: bool, eager=None) -> bool:
        """The step's routing decision, once, as the builder lays it out (``TkRound.service``): the previous step's
        drafts are proposed by now."""
        why = self.refusal(batch, running_bs, rows, decode, eager)
        self._decision = batch if why is None else None
        self._served = self._decode_batch = None
        if why is None:
            return True
        rid = None if batch is None or not batch.req_ids else batch.req_ids[0]
        if self._runner.holds(rid):
            raise RuntimeError(
                f"V4.1 TileRT-path step: request {rid} it served cannot fall back ({why})"
            )
        if why != self._why:
            logger.info("V4.1 TileRT-path step: not served (%s)", why)
            self._why = why
        return False

    @staticmethod
    def _anchor_position(batch) -> int:
        # a decode request's tokens: its anchor at num_tokens - (k + 1), its k placeholders after it
        return int(batch.context_lens[0]) - ROWS

    def step_override(self, input_ids, running_bs: int, batch):
        planned, self._decision = self._decision, None
        if planned is None:
            self._served = self._decode_batch = None
            return None
        # laid out LEAN for TkStep (``decide``): it serves it or nothing does
        if planned is not batch or input_ids.numel() != ROWS or get_forward_context().spec_decode_metadata is None:  # fmt: skip
            raise RuntimeError(
                f"V4.1 TileRT-path step: a step laid out for it runs another batch / {input_ids.numel()} rows / no verify"
            )
        fc = get_forward_context()
        rid = batch.req_ids[0]

        def kv_source():  # read only at the request's admission
            from atom.models.deepseek_v41.mono.tk_step import kv

            return kv.from_cache(
                fc.attn_metadata.cache,
                int(batch.state_slots_committed[0]),
                batch.block_tables[0],
            )

        self._runner.run_step(
            rid,
            input_ids,
            self._anchor_position(batch),
            float(batch.temperatures[0]),
            kv_source,
        )
        self._served, self._decode_batch = self._runner, batch
        return self._runner

    def forward(self, input_ids, positions, inputs_embeds=None):
        # a production forward (a prefill, an eager decode): not served
        if self._decision is not None:
            raise RuntimeError(
                "V4.1 TileRT-path step: a step laid out for it ran ATOM's forward (an eager decode?)"
            )
        self._served = self._decode_batch = None
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_tk_step(
    model: nn.Module, atom_config, drafter, metadata_builder=None
) -> nn.Module:
    """``TkStepModel`` over ``model``, its ``TkRound`` on ``metadata_builder`` (V4.1's, ``use_decode_round``), or
    ``model`` when this deployment cannot run it (logged)."""
    why = refusal(atom_config)
    if why is None and not envs.ATOM_DSV41_TK_CO:
        why = "ATOM_DSV41_TK_CO (TileRT's code objects) unset"
    if why is None and not hasattr(metadata_builder, "use_decode_round"):
        why = "no V4.1 metadata builder to take its decode round (ATOM's state bookkeeping)"
    if why is not None:
        logger.info("V4.1 TileRT-path step off: %s", why)
        return model
    from atom.models.deepseek_v41.mono.tk_step.runner import TkStepRunner

    runner = TkStepRunner.create(
        atom_config,
        atom_config.model,
        envs.ATOM_DSV41_TK_CO,
        envs.ATOM_DSV41_TK_TOKTAB_DIR or None,
        envs.ATOM_DSV41_TK_SWAP or None,
    )
    wrapped = TkStepModel(model, drafter, runner)
    metadata_builder.use_decode_round(TkRound(wrapped, metadata_builder))
    return wrapped
