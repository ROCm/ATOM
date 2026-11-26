# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Qwen3.8-Flash-Next forwards take the mono path.

``install_mono_decode`` (``models.registry.mono_decode_installers``) wraps the loaded model after load: a served MTP
verify step (``step_meta.verify_step``) runs every decoder layer as one mono program, reading and writing ATOM's GDN
pools and QSA caches in place; every other step runs ATOM's own model. The drafter's ``Qwen4ExpMTP`` is wrapped the
same way (``draft_model.MonoDraftModel``, its layer on ``draft_runner.DraftRunner``). Both are built here, before
warm-up and graph capture: the kernels of every served shape and the drafter's weights; the target's pointer tables
on the first eager step of each shape.
"""

import logging

import torch
from torch import nn

from atom.model_loader.weight_utils import local_model_dir
from atom.models.qwen38.mono.config import config_refusal, refused_by_switch_only
from atom.models.qwen38.mono.kernels.geometry import load_geometry

logger = logging.getLogger("atom")


class MonoTargetModel(nn.Module):
    """``Qwen4ExpForConditionalGeneration`` with its served verify steps on mono layer programs."""

    def __init__(self, model: nn.Module, runner) -> None:
        super().__init__()
        self.model = model
        self.runner = runner

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        meta = self.runner.step_meta(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if meta is None:
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return self.runner.forward(input_ids, meta)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    """``model`` wrapped for Qwen Mono decode, or ``model`` itself when Mono does not serve this deployment."""
    why = config_refusal(atom_config)
    if why is not None:
        if refused_by_switch_only(atom_config):
            # (an fp32 GDN state, BF16 draft caches or a record capacity Mono builds)
            logger.warning(
                "Qwen mono off: %s; the deployment is otherwise one Mono serves", why
            )
        else:
            logger.info("Qwen mono off: %s", why)
        return model
    from atom.models.qwen4_exp_mtp import Qwen4ExpMTP
    from atom.models.qwen38.mono.draft_model import MonoDraftModel
    from atom.models.qwen38.mono.draft_runner import DraftRunner
    from atom.models.qwen38.mono.runner import TargetRunner

    # (``--model`` may be a hub id: mono reads the checkpoint's files, so resolve its directory once)
    checkpoint = local_model_dir(atom_config.model)
    geo = load_geometry(checkpoint)
    if drafter is not None and isinstance(drafter.model, Qwen4ExpMTP):
        runner = DraftRunner(drafter.model.model, atom_config, geo, checkpoint)
        drafter.model = MonoDraftModel(drafter.model, runner)
    # the GDN verify ATOM picked (ReplaySSM by default under MTP, classic with ATOM_ENABLE_REPLAYSSM=0)
    replay = (
        int(metadata_builder.replayssm_cache_len)
        if getattr(metadata_builder, "replayssm", False)
        else 0
    )
    target = TargetRunner(model, atom_config, geo, checkpoint, replay=replay)
    return MonoTargetModel(model, target)
