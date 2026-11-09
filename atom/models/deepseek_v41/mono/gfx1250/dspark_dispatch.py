# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Route DSpark target and draft graphs to gfx1250 mono.

The same routing as ``mono.dispatch`` (gfx950): a target step is a decode of up
to ``MAX_ROWS`` rows, however many requests they are and however they split
them, and the draft's block backbone is up to ``MAX_REQUESTS`` requests at the
full block width. Of those, the widths in ``TARGET_WIDTHS`` / ``DRAFT_WIDTHS``
run on mono and every other one stays native. Every per-step input read is
TP-uniform (graph widths, not rank-local request lists): the kernels wait on
every peer rank.
"""

import logging

import torch
from torch import nn

from atom.models.deepseek_v41.mono.gfx1250.dspark_config import (
    DRAFT_ROWS,
    DRAFT_REQUESTS,
    DRAFT_WIDTHS,
    MAX_REQUESTS,
    MAX_ROWS,
    MONO_REQUESTS,
    TARGET_WIDTHS,
    config_refusal,
)
from atom.models.deepseek_v41.mono.gfx1250.dspark_draft_runner import DraftMonoRunner
from atom.models.deepseek_v41.mono.gfx1250.dspark_runner import V41MonoDecodeRunner
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def step_supported(input_ids: torch.Tensor, inputs_embeds) -> bool:
    """Is this forward a decode step of up to ``MAX_ROWS`` rows? Its graph
    width, not the scheduled request count (``mono.dispatch.step_supported``)."""
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
    return step.decode and step.width == rows


class MonoDecodeModel(nn.Module):
    """The target model, with its ``TARGET_WIDTHS`` decode steps on mono."""

    def __init__(self, model, drafter, max_model_len):
        super().__init__()
        self.model = model
        self._mono = LazyRunner(
            lambda: V41MonoDecodeRunner(model, drafter, max_model_len, False),
            "V4.1 gfx1250 mono decode",
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        rows = input_ids.numel()
        if (
            rows in TARGET_WIDTHS
            and step_supported(input_ids, inputs_embeds)
            and self._mono.ready(rows)
        ):
            return self._mono.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class MonoDraftModel(nn.Module):
    """The draft's full-width block backbone of ``DRAFT_REQUESTS`` requests on
    mono, after a prefill as after a decode."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._mono = LazyRunner(
            lambda: DraftMonoRunner(model), "V4.1 gfx1250 mono draft"
        )

    def block_backbone(self, input_ids, positions, num_draft):
        width = self.model.block_size if num_draft is None else num_draft
        rows = input_ids.numel() * width
        if (
            input_ids.numel() in DRAFT_REQUESTS
            and width == self.model.block_size == DRAFT_ROWS
            and self._mono.ready(rows)
        ):
            return self._mono.runner.backbone(input_ids, positions)
        return self.model.block_backbone(input_ids, positions, num_draft)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(model, atom_config, drafter, *, metadata_builder=None):
    """The regular mono installer and switch, with native fallback on refusal."""
    why = config_refusal(atom_config)
    draft = getattr(drafter, "model", None)
    if why is None and (
        len(getattr(model, "layers", ())) != 40
        or len(getattr(draft, "mtp", ())) != 3
        or getattr(draft, "block_size", None) != DRAFT_ROWS
        or not callable(getattr(draft, "block_backbone", None))
    ):
        why = "gfx1250 requires the 40-layer target and three-stage DSpark draft"
    if why is not None:
        logger.info("V4.1 gfx1250 mono decode off: %s", why)
        return model
    assert max(MONO_REQUESTS) <= MAX_REQUESTS
    wrapped = MonoDecodeModel(model, drafter, atom_config.max_model_len)
    draft_wrapped = MonoDraftModel(draft)
    # Immutable expert copies must be charged to KV profiling: bind and build
    # every enabled width now, before the runner profiles the KV cache.
    for owner, widths in (
        (wrapped._mono, TARGET_WIDTHS),
        (draft_wrapped._mono, DRAFT_WIDTHS),
    ):
        if not all(owner.ready(rows) for rows in widths):
            return model
    drafter.model = draft_wrapped
    logger.info(
        "V4.1 gfx1250 mono on: DSpark target %s / draft %s rows",
        TARGET_WIDTHS,
        DRAFT_WIDTHS,
    )
    return wrapped
