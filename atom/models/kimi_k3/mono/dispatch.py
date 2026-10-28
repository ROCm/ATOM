# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Kimi-K3 forwards take the mono path.

K3's registered class (``KimiK3ForConditionalGeneration``) is an uncompiled
wrapper, but the routing still cannot sit in its forward without editing a file
whose layer stack is ``@support_torch_compile``-decorated. So the installer
wraps the loaded model after load, the way V4.1's does
(``model_runner.mono_decode_installers``), and every other attribute passes
through to it. Nothing in ``atom/models/kimi_k3/model.py`` changes.

The configuration is checked once (``config.config_refusal``). A step is then
routed only when it is a pure, uniform speculative verify of at most
``MAX_ROWS`` rows (``runner.step_rows``). Every per-step input read here is
TP-uniform: the kernels wait in-kernel on every peer rank, so ranks that
disagreed would wait on each other for ever (a per-rank flag such as
``is_dummy_run`` must never gate this).
"""

import logging

import torch
from torch import nn

from atom.models.kimi_k3.mono.config import config_refusal
from atom.models.kimi_k3.mono.plan import kda_metadata, step_shape
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class MonoDecodeModel(nn.Module):
    """The loaded K3 model, with its supported steps routed to mono."""

    def __init__(self, model: nn.Module, drafter=None) -> None:
        super().__init__()
        self.model = model

        # Created on its first eligible step outside a graph capture
        # (``LazyRunner``: the allocation and the peer handshake cannot be
        # captured), then each width prepared on its first step. The import
        # sits in the factory rather than at module scope for the same
        # reason: resolving this installer must not pull ``runner`` -- and
        # through it K1/K2 and FlyDSL, 97 modules -- before
        # ``config_refusal`` can turn mono down. FlyDSL is not a declared
        # dependency, so a deployment without it would otherwise fail at
        # import even with ``ATOM_MONO_ENABLE=0``.
        # ``glm52/mono/kernels/__init__`` keeps its GPU wrappers lazy for
        # the same reason.
        def _build_runner():
            # An absent FlyDSL must refuse, not raise: `LazyRunner.ready`
            # catches only `MonoUnsupported`, so a bare ImportError here would
            # escape it and abort the decode warmup instead of leaving the
            # deployment on the production path. Converted at the import, which
            # is before this runner reaches any collective, so every rank
            # refuses at the same point.
            try:
                from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner
            except ImportError as error:
                raise MonoUnsupported(f"mono kernels unavailable: {error}") from error

            return K3MonoDecodeRunner(model, drafter)

        self._mono = LazyRunner(_build_runner, "Kimi-K3 mono decode")
        self._declines: set[str] = set()

    def _qlen(self, input_ids, intermediate_tensors, inputs_embeds) -> int | None:
        """This step's uniform query length if mono serves it, else None.

        A step turned away says so once per distinct reason: mono serving
        nothing and mono serving everything look the same from the outside.
        """
        if intermediate_tensors is not None or inputs_embeds is not None:
            return self._declined("pipeline or embedding input")
        context = get_forward_context()
        if context.context is None or context.context.is_prefill:
            return self._declined("prefill")
        if getattr(context, "ubatch_slices", None) is not None:
            return self._declined("TBO micro-batches")
        qlen, why = step_shape(kda_metadata(context), input_ids.numel())
        return qlen if qlen is not None else self._declined(why)

    def _declined(self, why: str) -> None:
        if why not in self._declines:
            self._declines.add(why)
            logger.info("Kimi-K3 mono decode declined a step: %s", why)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        qlen = self._qlen(input_ids, intermediate_tensors, inputs_embeds)
        # collective: the runner and this step's kernels are built on every
        # rank, so the row count and the query length must be TP-uniform
        served = qlen is not None and self._mono.ready(input_ids.numel(), qlen)
        if served:
            return self._mono_forward(input_ids, positions, qlen)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def _mono_forward(self, input_ids, positions, qlen):
        return self._mono.runner.forward(input_ids, positions, qlen)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    """``model`` wrapped for K3 mono decode, or ``model`` itself when this
    deployment is one mono does not serve."""
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("Kimi-K3 mono decode off: %s", why)
        return model
    logger.info("Kimi-K3 mono decode armed (TP%d)", atom_config.tensor_parallel_size)
    return MonoDecodeModel(model, drafter)
