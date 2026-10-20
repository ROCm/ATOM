# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Qwen3.8-Flash-Next forwards take the mono path.

``install_mono_decode`` wraps the loaded model (as ``deepseek_v41.mono`` does)
and every other attribute passes through to it. The configuration is checked
once (``config.config_refusal``); a step is then routed when it is a plain
decode of up to ``MAX_ROWS`` rows, one token per request, with its QSA and PLE
metadata built (the warmup and profiling forwards that run before the state
pools exist have none). With MTP every decode step is a verify (k + 1 tokens
per request); one of up to ``GROUPS`` requests is routed (``groups``). Every
other forward, prefill and a wider verify included, is ATOM's own. The decode
steps' lm_head runs from MXFP4 (``head``).

With MTP the drafter's model is wrapped too (``MonoDraftModel``): the draft
steps of a routed verify run its decoder layer on mono (``draft_runner``),
every other draft forward ATOM's.
"""

import logging

import torch
from torch import nn

from atom.models.qwen4_exp_mtp import Qwen4ExpMTP
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

from .config import MAX_ROWS, config_refusal
from .draft_runner import Qwen4MonoDraftRunner
from .group_runner import Qwen4MonoGroupRunner
from .groups import group_draft_step_supported, group_step_supported
from .head import install_fp4_head
from .runner import Qwen4MonoDecodeRunner

logger = logging.getLogger("atom")

# with MTP, the most requests a routed verify step has: G groups of k + 1 rows
# fill at most a 16-row grouped build (``layout_wide.WIDE_ROWS``)
GROUPS = 4


def step_supported(
    input_ids: torch.Tensor, inputs_embeds, verify_rows: int = 0
) -> bool:
    """Is this forward a decode step of up to ``MAX_ROWS`` requests, one
    token each, whose caches mono can address? With MTP (``verify_rows`` =
    k + 1), a verify step of one request instead."""
    rows = input_ids.numel()
    if inputs_embeds is not None or not 0 < rows <= MAX_ROWS:
        return False
    fwd = get_forward_context()
    if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
        return False
    md = fwd.attn_metadata
    if getattr(md, "qsa_metadata", None) is None:
        return False
    if getattr(md, "ple_metadata", None) is None:
        return False
    gdn = getattr(md, "gdn_metadata", None)
    if gdn is None:
        return False
    if verify_rows:
        return (
            rows == verify_rows
            and md.max_seqlen_q == verify_rows
            and gdn.num_spec_decodes == 1
            and gdn.num_decodes == 0
            and gdn.spec_state_indices_tensor is not None
            and gdn.num_accepted_tokens is not None
        )
    if gdn.spec_state_indices_tensor is not None:
        return False
    if gdn.non_spec_state_indices_tensor is None:
        return False
    return md.max_seqlen_q == 1 and gdn.num_spec_decodes == 0


def draft_step_supported(input_ids: torch.Tensor, inputs_embeds, verify_rows) -> bool:
    """Is this drafter forward a draft step of a single-request verify of
    ``verify_rows`` (k + 1) rows: step 0 over the verify's rows, or with
    k >= 2 a later step, one row?"""
    rows = input_ids.numel()
    if inputs_embeds is not None:
        return False
    fwd = get_forward_context()
    c = fwd.context
    if c is None or c.is_prefill or not c.is_draft or fwd.ubatch_slices is not None:
        return False
    md = fwd.attn_metadata
    gdn = getattr(md, "gdn_metadata", None)
    if getattr(md, "qsa_metadata", None) is None or gdn is None:
        return False
    # the target's verify metadata: the draft steps keep its GDN part
    if gdn.num_spec_decodes != 1 or gdn.num_decodes != 0:
        return False
    if rows == verify_rows:
        return md.max_seqlen_q == verify_rows
    return rows == 1 and verify_rows > 2 and md.max_seqlen_q == 1


class MonoDraftModel(nn.Module):
    """ATOM's Qwen3.8-Flash-Next MTP drafter, the draft steps of a
    single-request verify routed to mono (``draft_step_supported``), and
    those of a verify of up to ``GROUPS`` requests
    (``group_draft_step_supported``)."""

    def __init__(self, draft: nn.Module, atom_config, metadata_builder) -> None:
        super().__init__()
        # not ``model``: the drafter reads its own ``model`` through this
        self.draft = draft
        self.verify_rows = atom_config.speculative_config.num_speculative_tokens + 1
        self.groups = GROUPS

        def create():
            return Qwen4MonoDraftRunner(
                draft, atom_config, metadata_builder, self.verify_rows, self.groups
            )

        self._mono = LazyRunner(create, "Qwen3.8 mono draft")

    def supported(self, input_ids, inputs_embeds) -> bool:
        if draft_step_supported(input_ids, inputs_embeds, self.verify_rows):
            return True
        return group_draft_step_supported(
            input_ids, inputs_embeds, self.verify_rows, self.groups
        )

    def forward(
        self, input_ids, positions, hidden_states, inputs_embeds=None, **kwargs
    ):
        if (
            not kwargs
            and self.supported(input_ids, inputs_embeds)
            and self._mono.ready(input_ids.numel())
        ):
            return self._mono.runner.forward(input_ids, hidden_states)
        return self.draft(input_ids, positions, hidden_states, inputs_embeds, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.draft, name)


class MonoDecodeModel(nn.Module):
    """The loaded Qwen3.8-Flash-Next model, its supported steps routed to
    mono."""

    def __init__(self, model: nn.Module, atom_config, metadata_builder) -> None:
        super().__init__()
        self.model = model
        spec = atom_config.speculative_config
        # a verify step's rows (k + 1) with MTP, else 0
        self.verify_rows = 0 if spec is None else spec.num_speculative_tokens + 1
        # with MTP, the most requests a routed verify step has
        self.groups = GROUPS if self.verify_rows else 1

        # created on its first eligible step (``LazyRunner``): the caches it
        # binds are allocated after the model is installed
        def create():
            if self.groups > 1:
                return Qwen4MonoGroupRunner(
                    model, atom_config, metadata_builder, self.groups
                )
            return Qwen4MonoDecodeRunner(model, atom_config, metadata_builder)

        self._mono = LazyRunner(create, "Qwen3.8 mono decode")

    def supported(self, input_ids, inputs_embeds) -> bool:
        if step_supported(input_ids, inputs_embeds, self.verify_rows):
            return True
        return self.groups > 1 and group_step_supported(
            input_ids, inputs_embeds, self.verify_rows, self.groups
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if (
            intermediate_tensors is None
            and self.supported(input_ids, inputs_embeds)
            and self._mono.ready(input_ids.numel())
        ):
            return self._mono.runner.forward(input_ids, positions)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, metadata_builder
) -> nn.Module:
    """``model`` wrapped for mono decode, or ``model`` itself when this
    deployment is one mono does not serve. With MTP the mono runs the target
    model's verify steps and ``drafter``'s model is wrapped for its draft
    steps (``MonoDraftModel``)."""
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("Qwen3.8 mono decode off: %s", why)
        return model
    # before the drafter's model is wrapped: it shares the target's lm_head
    install_fp4_head(model, drafter)
    if drafter is not None and isinstance(getattr(drafter, "model", None), Qwen4ExpMTP):
        drafter.model = MonoDraftModel(drafter.model, atom_config, metadata_builder)
    # B3 reads and writes the recurrent state in place as VK; the caches are
    # allocated after this, so ask for that layout now (ATOM's own GDN
    # kernels take either)
    metadata_builder.gdn_state_vk = True
    return MonoDecodeModel(model, atom_config, metadata_builder)
