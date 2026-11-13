# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which forwards of a model whose DSA indexers are its only mono part run
them on the mono kernel (``runner.IndexerRunner``).

The installer wraps the loaded model (``models.registry.mono_decode_installers``)
and changes nothing else of it: the layers stay the model's own, compiled or
not, and a served step only hooks their indexers (``bind.IndexerHook``). A step
is served when it is a decode -- plain or a uniform speculative verify -- of at
most the kernel's rows. Every per-step input read here is TP-uniform: the
kernel waits in-kernel on every peer rank, so ranks that disagreed would wait
on each other for ever.
"""

import logging

import torch
from torch import nn

from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.deployment import (
    common_refusals,
    first_refusal,
    resident_grid_refusals,
)
from atom.mono.runtime.route import LazyRunner
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot run its DSA indexers on mono, or None. The
    indexers' own shape and the DCP split are the binding's (``bind._refusal``),
    checked once the model is built."""
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    # no model weights are sharded by the kernel: any TP shards (``dims``)
    checks += common_refusals(
        atom_config, lambda *_: None, (1, 2, 4, 8), refuse_decode_cp=False
    )
    checks += resident_grid_refusals(atom_config)
    return first_refusal(checks)


def decode_qlen(context, rows: int) -> int | None:
    """A decode step's uniform query length (1, or a verify's draft + 1), or
    None for any other step."""
    if context.context is None or context.context.is_prefill:
        return None
    if getattr(context, "ubatch_slices", None) is not None:
        return None
    md = context.attn_metadata
    qlen = getattr(md, "max_seqlen_q", 0)
    lens = getattr(md, "context_lens", None)
    if qlen < 1 or lens is None or lens.numel() * qlen != rows:
        return None
    return qlen


class IndexerMonoModel(nn.Module):
    """The loaded model, its indexers on mono at the steps it serves."""

    def __init__(self, model: nn.Module, hook, metadata_builder) -> None:
        super().__init__()
        self.model = model

        # created on its first eligible step outside a graph capture
        # (``LazyRunner``); imported there so a deployment without FlyDSL
        # refuses rather than failing at import
        def _build_runner():
            try:
                from atom.mono.dsa_indexer.runner import IndexerRunner
            except ImportError as error:
                raise MonoUnsupported(f"mono kernels unavailable: {error}") from error
            return IndexerRunner.alone(model.model, hook, metadata_builder)

        self._mono = LazyRunner(_build_runner, "DSA indexer mono")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        args = (input_ids, positions, intermediate_tensors, inputs_embeds)
        rows = input_ids.numel()
        qlen = decode_qlen(get_forward_context(), rows)
        # collective: built on every rank, so ``rows`` and ``qlen`` are uniform
        if (
            qlen is None
            or intermediate_tensors is not None
            or inputs_embeds is not None
            or not self._mono.ready(rows, qlen)
        ):
            return self.model(*args)
        with self._mono.runner.serving(rows, qlen):
            return self.model(*args)

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
    """``model`` wrapped for its DSA indexers on mono, or ``model`` itself when
    this deployment is one mono does not serve."""
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("DSA indexer mono off: %s", why)
        return model
    # FlyDSL is optional: the binding (and through it the kernel) is imported
    # only once the deployment is armed
    try:
        from atom.mono.dsa_indexer.bind import (
            IndexerHook,
            indexer_modules,
            routing_refusal,
        )
    except ImportError as error:
        logger.info("DSA indexer mono off: mono kernels unavailable: %s", error)
        return model
    modules = indexer_modules(model.model)
    # every refusal the indexers decide now (all but their index caches, not
    # yet allocated): a refused deployment keeps their own forward untouched
    why = routing_refusal(atom_config, modules)
    if why:
        logger.info("DSA indexer mono off: %s", why)
        return model
    logger.info("DSA indexer mono armed (TP%d)", atom_config.tensor_parallel_size)
    # routed now, before the layer stack is first traced (its warmup): the
    # runner attaches to it once built
    return IndexerMonoModel(model, IndexerHook(modules), metadata_builder)
