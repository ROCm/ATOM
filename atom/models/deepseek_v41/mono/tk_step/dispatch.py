# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TileRT-path step's routing wrapper (as ``gfx1250.dispatch.SingleStepModel``).

``ModelRunner.run_model`` offers it each decode step (``step_override``). It serves the step (``TkStepRunner``: target,
sampling, verify and the next drafts on TkRank) or returns None and the production path runs. A served step bypasses
ATOM's sampler, rejection sampler and DSpark proposer: the drafter's ``compute_draft_kv`` / ``propose`` are wrapped to
do nothing / return the runner's drafts.

Served: one request (6 rows, a DSpark verify), sampled (temperature > 0, top_p 1, no top_k), no structured output,
no required channel, no logprobs, inside the plan's context. Sticky: a request TkRank admitted keeps its state there
-- ATOM's caches do not get the KV it decodes -- so every later step of it must be served; one that cannot raises
instead of falling back. A step of another request (the held one finished: ATOM sends no finish) re-admits.
"""

import logging

from torch import nn

from atom.model_engine.token_constraints import has_constraints
from atom.models.deepseek_v41.mono.tk_step import ROWS, refusal
from atom.mono.runtime.step_override import StepOverrideModel
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


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

    def refusal(self, input_ids, running_bs: int, batch) -> str | None:
        """Why this decode step is not served (None: served)."""
        if envs.ATOM_MONO_CHECK:
            return "ATOM_MONO_CHECK"
        if batch is None or running_bs != 1 or input_ids.numel() != ROWS:
            return f"{running_bs} requests, {input_ids.numel()} rows"
        if has_constraints(batch) or batch.any_logprobs:
            return "constraints / logprobs"
        if get_forward_context().spec_decode_metadata is None:
            return "not a DSpark verify"
        return self._runner.precheck(batch, self._anchor_position(batch))

    @staticmethod
    def _anchor_position(batch) -> int:
        # a decode request's tokens: its anchor at num_tokens - (k + 1), its k placeholders after it
        return int(batch.context_lens[0]) - ROWS

    def step_override(self, input_ids, running_bs: int, batch):
        why = self.refusal(input_ids, running_bs, batch)
        rid = None if batch is None or not batch.req_ids else batch.req_ids[0]
        if why is not None:
            if self._runner.holds(rid):
                raise RuntimeError(
                    f"V4.1 TileRT-path step: request {rid} it served cannot fall back ({why})"
                )
            if why != self._why:
                logger.info("V4.1 TileRT-path step: not served (%s)", why)
                self._why = why
            self._served = self._decode_batch = None
            return None
        fc = get_forward_context()

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
        self._served = self._decode_batch = None
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_tk_step(model: nn.Module, atom_config, drafter) -> nn.Module:
    """``TkStepModel`` over ``model``, or ``model`` when this deployment cannot run it (logged)."""
    why = refusal(atom_config)
    if why is None and not envs.ATOM_DSV41_TK_CO:
        why = "ATOM_DSV41_TK_CO (TileRT's code objects) unset"
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
    return TkStepModel(model, drafter, runner)
