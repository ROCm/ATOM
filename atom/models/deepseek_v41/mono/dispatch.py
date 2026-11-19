# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which V4.1 forwards take the mono path.

``install_mono_decode`` wraps the loaded model, the way the TBO
``UBatchWrapper`` does, and every other attribute passes through to it. The
runtime model's own forward dispatches to its compiled graphs (``runtime``);
mono routing sits outside them, before any compiled code runs.

The configuration is checked once (``config.config_refusal``). A target step
is then routed only when it is a decode (a DSpark verify of any length a
request, or a token a request) of up to ``MAX_ROWS`` rows (a graph's padding
rows included), however many requests they are and however they split them.
The DSpark draft model is wrapped the same way (``MonoDraftModel``): its block backbone of
up to ``MAX_REQUESTS`` requests runs on ``draft_runner``, after a prefill as
after a decode. Every per-step input read is TP-uniform: the kernels wait on
every peer rank, so ranks that disagreed would wait on each other for ever (a
per-rank flag such as ``is_dummy_run`` must never gate this).
"""

import logging

import torch
from torch import nn

from atom.models.deepseek_v41.mono import hi_gate
from atom.models.deepseek_v41.mono.config import (
    MAX_MOE_ROWS,
    MAX_REQUESTS,
    config_refusal,
    indexer_enabled,
)
from atom.models.deepseek_v41.mono.decode_step import step_supported
from atom.models.deepseek_v41.mono.draft_runner import DraftMonoRunner
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.deployment import dp_attention_ranks
from atom.mono.runtime.route import LazyRunner

logger = logging.getLogger("atom")


class MonoDecodeModel(nn.Module):
    """The loaded V4.1 runtime model, with its supported steps routed to mono."""

    def __init__(
        self,
        model: nn.Module,
        drafter,
        max_model_len: int,
        index_fp4: bool,
        dp_ranks: int,
        tp_size: int = 1,
        *,
        standalone_indexer: bool = True,
        indexer_workspace=None,
    ) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter
        # DP attention's ranks, 0 without it (``step_supported``)
        self._dp_ranks = dp_ranks
        self._tp_size = tp_size
        self._index_fp4 = index_fp4
        self._standalone_indexer = standalone_indexer
        # created on its first eligible step (``LazyRunner``)
        self._mono = LazyRunner(
            lambda: V41MonoDecodeRunner(
                model,
                drafter,
                max_model_len,
                index_fp4,
                standalone_indexer=standalone_indexer,
                indexer_workspace=indexer_workspace,
            ),
            "V4.1 mono decode",
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        if self.uses_layer_mono(input_ids, inputs_embeds):
            return self._mono.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def uses_layer_mono(self, input_ids, inputs_embeds=None):
        return (
            step_supported(input_ids, inputs_embeds, self._dp_ranks)
            and self.layer_rows_supported(input_ids.numel())
            and self._mono.ready(input_ids.numel())
        )

    def layer_rows_supported(self, rows):
        # At TP2, large FP4 decode batches are faster with native layers.
        # The outer wrapper still runs their standalone Mono indexer.
        return not (
            self._tp_size == 2
            and not self._dp_ranks
            and rows >= 40
            and self._standalone_indexer
            and indexer_enabled(self._index_fp4, self._tp_size)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class MonoDraftModel(nn.Module):
    """The loaded DSpark draft model, with the block backbone of up to
    ``MAX_REQUESTS`` requests at the full block width routed to mono (every
    batch size up to it is captured, so a replay's requests are all real). The
    call is inside ``DraftGraph``'s recording, so the runner binds on the first
    such call outside one (the graph's warmup).

    Under DP attention over ``dp_ranks`` ranks the requests are the agreed
    padded batch (``DSparkProposer`` stages the anchors at ``running_bs``,
    an idle rank's as padding), the same on every rank, of up to
    ``MAX_REQUESTS`` a rank while the MoE's rows over every rank stay within
    ``MAX_MOE_ROWS``."""

    def __init__(self, draft: nn.Module, dp_ranks: int = 0) -> None:
        super().__init__()
        self.model = draft
        self._max_requests = (
            min(MAX_REQUESTS, MAX_MOE_ROWS // (dp_ranks * draft.block_size))
            if dp_ranks
            else MAX_REQUESTS
        )
        self._mono = LazyRunner(lambda: DraftMonoRunner(draft), "V4.1 mono draft")

    def block_backbone(self, input_ids, positions, num_draft):
        width = self.model.block_size if num_draft is None else num_draft
        requests = input_ids.numel()
        if (
            self.model.draft_kv_dtype != "fp8"
            and 0 < requests <= self._max_requests
            and width == self.model.block_size
            and self._mono.ready(requests * width)
        ):
            return self._mono.runner.backbone(input_ids, positions)
        return self.model.block_backbone(input_ids, positions, num_draft)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    return _install_layer_decode(
        model, atom_config, drafter, metadata_builder=metadata_builder
    )


def _install_layer_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    """``model`` wrapped for mono decode (and ``drafter``'s DSpark draft model
    for its backbone), or ``model`` itself when this deployment is one mono
    does not serve."""
    from atom.models.deepseek_v41.mono import gfx1250

    if gfx1250.enabled():
        from atom.models.deepseek_v41.mono.gfx1250.dispatch import (
            install_single_step,
        )

        return install_single_step(model, atom_config, drafter)
    why = config_refusal(atom_config)
    if why is not None:
        if hi_gate.mode() == "1":
            raise RuntimeError(
                f"ATOM_DSV41_MONO_HI=1, but V4.1 mono decode is off: {why}"
            )
        logger.info("V4.1 mono decode off: %s", why)
        return model
    dp_ranks = dp_attention_ranks(atom_config)
    # The decoder layers on the high-interactivity monokernels where they serve
    # this deployment (``hi_gate``: ATOM_DSV41_MONO_HI, "auto" by default); any
    # other takes ATOM's mono decode below, unchanged.
    # A mismatch only the loaded model shows (``MonoUnsupported``, agreed over
    # the TP group before anything is built: ``install_hi``) falls back as the
    # gate's refusals do, on every rank.
    if hi_gate.decide(atom_config, drafter, metadata_builder, dp_ranks):
        from atom.models.deepseek_v41.mono.hi_dispatch import install_hi

        try:
            return install_hi(model, atom_config, drafter, metadata_builder, dp_ranks)
        except MonoUnsupported as why:
            if hi_gate.mode() == "1":
                raise
            hi_gate.log_once(logging.INFO, "V4.1 mono hi off: %s", why)
    if drafter is not None and hasattr(drafter.model, "block_backbone"):
        drafter.model = MonoDraftModel(drafter.model, dp_ranks)
    standalone = (
        indexer_enabled(
            atom_config.index_cache_dtype == "fp4", atom_config.tensor_parallel_size
        )
        and not dp_ranks
    )
    workspace = None
    if standalone:
        from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace

        workspace = IndexerWorkspace(
            model, atom_config.max_model_len, atom_config.tensor_parallel_size
        )
    wrapped = MonoDecodeModel(
        model,
        drafter,
        atom_config.max_model_len,
        atom_config.index_cache_dtype == "fp4",
        dp_ranks,
        atom_config.tensor_parallel_size,
        indexer_workspace=workspace,
    )
    # A successfully installed HI hook and gfx1250 return above. An auto HI
    # install refusal reaches this ordinary route and follows its indexer flag.
    if standalone:
        from atom.models.deepseek_v41.mono.indexer_dispatch import MonoIndexerModel

        return MonoIndexerModel(wrapped, model, atom_config, workspace=workspace)
    return wrapped
