# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The single-launch step's routing wrapper.

The step kernel does the whole decode step -- target forward, LM head, sampling, verify, the draft's context KV, its
three stages, head and Markov -- so a served step bypasses ``ModelRunner``'s sampler, rejection sampler and the
DSpark proposer's ``DraftGraph`` as well as the target forward (``SingleStepModel.step_ready``; the runner's three
call sites ask it). Until the runner binds a width, every step keeps the production path.
"""

import logging

from torch import nn

from atom.model_engine.token_constraints import has_constraints
from atom.models.deepseek_v41.mono.gfx1250 import refusal
from atom.models.deepseek_v41.mono.gfx1250.config import MAX_ROWS, ROWS_PER_REQ
from atom.models.deepseek_v41.mono.gfx1250.debug import CheckModel
from atom.mono.runtime.route import LazyRunner
from atom.mono.runtime.step_override import StepOverrideModel
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class SingleStepModel(CheckModel, StepOverrideModel, nn.Module):
    """The loaded V4.1 runtime model. ``ModelRunner`` offers it each decode step (``step_override``): it runs the
    step as the single launch, or declines and the production path runs (its forward, unchanged).

    A served step also sampled, verified and drafted: the drafter's ``compute_draft_kv`` and ``propose`` are wrapped
    at install so that on a served step they do nothing / return the kernel's drafts."""

    def __init__(self, model: nn.Module, drafter, atom_config) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter

        def create():
            # (imported on first use: the runner needs aiter and FlyDSL, the routing above must import without them)
            from atom.models.deepseek_v41.mono.gfx1250.runner import (
                SingleStepRunner,
            )

            return SingleStepRunner(model, drafter, atom_config)

        self._single = LazyRunner(create, "V4.1 single-launch step")
        self._served = None  # this step's runner when the single launch served it
        self._decode_batch = (
            None  # this step's batch when it is a decode step (``step_override``)
        )
        self._step_batch = None
        if drafter is not None:
            self._wrap_drafter(drafter)

    def _wrap_drafter(self, drafter) -> None:
        """On a served step the kernel already wrote the draft context KV and drew the next drafts."""
        compute_draft_kv, propose = drafter.compute_draft_kv, drafter.propose

        def served_compute_draft_kv(*args, **kwargs):
            if self._served is None:
                return compute_draft_kv(*args, **kwargs)
            return None

        def served_propose(*args, **kwargs):
            batch = self._decode_batch
            if self._served is not None:
                return self._served.draft_ids(batch.total_seqs_num)
            drafts = propose(*args, **kwargs)
            runner = self._single.runner
            if drafts is not None and runner is not None and batch is not None:
                # a decode batch the production drafter drew for: its requests' kept q no longer match (a prefill
                # batch's requests have none)
                runner.forget_drafts(batch.req_ids[: batch.total_seqs_num])
            return drafts

        drafter.compute_draft_kv, drafter.propose = (
            served_compute_draft_kv,
            served_propose,
        )

    def step_ready(self, rows: int) -> bool:
        return (
            0 < rows <= MAX_ROWS
            and rows % ROWS_PER_REQ == 0
            and self._single.ready(rows)
        )

    def step_override(self, input_ids, running_bs: int, batch):
        """``ModelRunner.run_model``, each decode step: run it as the single launch and return the runner (its
        ``verify_outputs`` / ``output_stream`` for postprocess), or None for the production path.
        """
        # the launch reads input_ids as 6 rows a request: any other step (a padded one, a ragged verify) takes the
        # production path
        n = input_ids.numel()
        runner = (
            self.single_step_runner(batch, n)
            if n == running_bs * ROWS_PER_REQ
            else None
        )
        if runner is not None:
            runner.run_step(input_ids, running_bs, get_forward_context(), batch)
        self._served, self._decode_batch = runner, batch
        return runner

    def single_step_runner(self, batch, rows: int):
        """The runner when this decode step runs as the single launch, else None (the production path). Only a
        DSpark verify with nothing the kernel does not produce: no structured output, no logprobs, no required
        channel (it returns ids and the verdict, not logits), and request shapes the kernel takes (``precheck``).
        """
        self._step_batch = batch  # (the ATOM_MONO_CHECK launch's sampling parameters)
        if envs.ATOM_MONO_CHECK:  # the production path serves, the launch only compares
            return None
        if batch is not None and (has_constraints(batch) or batch.any_logprobs):
            return None
        fc = get_forward_context()
        if fc.spec_decode_metadata is None or not self.step_ready(rows):
            return None
        runner = self._single.runner
        return runner if runner.precheck(batch, fc) else None

    @property
    def single_runner(self):
        return self._single.runner

    def forward(self, input_ids, positions, inputs_embeds=None):
        # a production forward (prefill, or an eager decode): not the single launch's step
        self._served = self._decode_batch = None
        if inputs_embeds is None and envs.ATOM_MONO_CHECK:
            return self.check_forward(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_single_step(model: nn.Module, atom_config, drafter) -> nn.Module:
    why = refusal(atom_config)
    if why is not None:
        logger.info("V4.1 single-launch step off: %s", why)
        return model
    return SingleStepModel(model, drafter, atom_config)
