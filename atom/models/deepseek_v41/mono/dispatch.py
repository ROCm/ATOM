# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which V4.1 forwards take the mono path.

``install_mono_decode`` wraps the loaded model, the way the TBO
``UBatchWrapper`` does, and every other attribute passes through to it. The
runtime model's own forward dispatches to its compiled graphs (``runtime``);
mono routing sits outside them, before any compiled code runs.

The configuration is checked once (``config.config_refusal``). A target step
is then routed only when it is a DSpark verify of up to ``MAX_ROWS`` rows (a
graph's padding rows included), however many requests they are. The DSpark
draft model is wrapped the same way (``MonoDraftModel``): its block backbone of
up to ``MAX_REQUESTS`` requests runs on ``draft_runner``, after a prefill as
after a decode. Every per-step input read is TP-uniform: the kernels wait on
every peer rank, so ranks that disagreed would wait on each other for ever (a
per-rank flag such as ``is_dummy_run`` must never gate this).
"""

import logging

import torch
from torch import nn

from atom.models.deepseek_v41.mono.config import (
    MAX_REQUESTS,
    MAX_ROWS,
    config_refusal,
)
from atom.models.deepseek_v41.mono.draft_runner import DraftMonoRunner
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from atom.models.deepseek_v41.mono.walk_plan import WalkPlanner
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def step_supported(input_ids: torch.Tensor, inputs_embeds) -> bool:
    """Is this forward a DSpark verify step of up to ``MAX_ROWS`` rows? Its
    width, not the scheduled request count: a captured graph's width is the one
    its replays run, while the scheduled count differs from replay to replay."""
    rows = input_ids.numel()
    if inputs_embeds is not None or not 0 < rows <= MAX_ROWS:
        return False
    fwd = get_forward_context()
    if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
        return False
    md = fwd.attn_metadata
    step = getattr(md, "step", None)
    if step is None or getattr(md, "image_mask", None) is not None:
        return False
    return step.decode and step.tentative and step.width == rows


class MonoDecodeModel(nn.Module):
    """The loaded V4.1 runtime model, with its supported steps routed to mono."""

    def __init__(
        self, model: nn.Module, drafter, max_model_len: int, index_fp4: bool
    ) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter
        # created on its first eligible step (``LazyRunner``)
        self._mono = LazyRunner(
            lambda: V41MonoDecodeRunner(model, drafter, max_model_len, index_fp4),
            "V4.1 mono decode",
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        if step_supported(input_ids, inputs_embeds) and self._mono.ready(
            input_ids.numel()
        ):
            return self._mono.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

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
    such call outside one (the graph's warmup)."""

    def __init__(self, draft: nn.Module) -> None:
        super().__init__()
        self.model = draft
        self._mono = LazyRunner(lambda: DraftMonoRunner(draft), "V4.1 mono draft")

    def block_backbone(self, input_ids, positions, num_draft):
        width = self.model.block_size if num_draft is None else num_draft
        requests = input_ids.numel()
        if (
            0 < requests <= MAX_REQUESTS
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
    model: nn.Module, atom_config, drafter, metadata_builder
) -> nn.Module:
    """``model`` wrapped for mono decode (and ``drafter``'s DSpark draft model
    for its backbone), or ``model`` itself when this deployment is one mono
    does not serve. On the FP4 index plane ``metadata_builder`` lays the score
    stage's task plans out with each step (``WalkPlanner``)."""
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("V4.1 mono decode off: %s", why)
        return model
    if drafter is not None and hasattr(drafter.model, "block_backbone"):
        drafter.model = MonoDraftModel(drafter.model)
    index_fp4 = atom_config.index_cache_dtype == "fp4"
    if index_fp4:
        metadata_builder.add_step_planner(
            WalkPlanner(ratio for ratio, _ in metadata_builder.geometry.compress_ratios)
        )
    return MonoDecodeModel(model, drafter, atom_config.max_model_len, index_fp4)
