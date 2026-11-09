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

from atom.models.deepseek_v41.mono.gfx1250_single.config import (
    MAX_ROWS,
    ROWS_PER_REQ,
    config_refusal,
)
from atom.models.deepseek_v41.mono.gfx1250_single.debug import ShadowModel
from atom.mono.runtime.route import LazyRunner
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class SingleStepModel(ShadowModel, nn.Module):
    """The loaded V4.1 runtime model. Its forward is unchanged (a served step does not call it); ``step_ready(rows)``
    tells the model runner whether this step runs as the single launch."""

    def __init__(self, model: nn.Module, drafter, atom_config) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter

        def create():
            from atom.models.deepseek_v41.mono.gfx1250_single.runner import (
                SingleStepRunner,
            )

            return SingleStepRunner(model, drafter, atom_config)

        self._single = LazyRunner(create, "V4.1 single-launch step")

    def step_ready(self, rows: int) -> bool:
        return (
            0 < rows <= MAX_ROWS
            and rows % ROWS_PER_REQ == 0
            and self._single.ready(rows)
        )

    def single_step_runner(self, batch, rows: int):
        """``ModelRunner``: the runner when this decode step runs as the single launch, else None (the production
        path). Only a DSpark verify with nothing the kernel does not produce: no structured output, no logprobs, no
        required channel (it returns ids and the verdict, not logits), and request shapes the kernel takes
        (``precheck``)."""
        if (
            envs.ATOM_MONO_SINGLE_SHADOW
        ):  # the production path serves, the launch only compares
            return None
        if batch is not None and (
            getattr(batch, "structured_requests", None)
            or batch.any_logprobs
            or getattr(batch, "required_channel", None) is not None
        ):
            return None
        fc = get_forward_context()
        if fc.spec_decode_metadata is None or not self.step_ready(rows):
            return None
        runner = self._single.runner
        return runner if runner.precheck(batch, fc) else None

    def set_step_batch(self, batch) -> None:
        """``ModelRunner.run_model``: this step's batch (the per-request sampling parameters the launch reads)."""
        self._step_batch = batch

    @property
    def single_runner(self):
        return self._single.runner

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None and envs.ATOM_MONO_SINGLE_SHADOW:
            return self.shadow_forward(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_single_step(model: nn.Module, atom_config, drafter) -> nn.Module:
    from atom.models.deepseek_v41.mono.kernels.dims import Dims

    why = config_refusal(atom_config, Dims)
    if why is not None:
        logger.info("V4.1 single-launch step off: %s", why)
        return model
    return SingleStepModel(model, drafter, atom_config)
