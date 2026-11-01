# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Explicit TP4 DSpark entry for the complete gfx1250 mono implementation."""

import logging

from atom.models.deepseek_v41.mono.config import config_refusal
from atom.models.deepseek_v41.mono.dispatch import MonoDecodeModel, MonoDraftModel
from atom.mono.runtime.deployment import dp_attention_ranks, first_refusal

from .draft_runner import DraftMonoRunner
from .runner import V41MonoDecodeRunner

logger = logging.getLogger("atom")


def trial_refusal(config, model, drafter):
    """Keep admission within the measured TP4/BF16/FP8/DSpark5 recipe."""
    why = config_refusal(config, arch="gfx1250")
    if why is not None:
        return why
    spec = config.speculative_config
    draft = getattr(drafter, "model", None)
    return first_refusal([
        (config.tensor_parallel_size == 4, "gfx1250 DSpark requires TP4"),
        (not dp_attention_ranks(config), "gfx1250 DSpark has no DP attention"),
        (config.kv_cache_dtype == "bf16", "gfx1250 DSpark requires BF16 KV"),
        (config.index_cache_dtype == "fp8", "gfx1250 DSpark requires FP8 index"),
        (spec is not None and spec.method == "dspark"
         and spec.num_speculative_tokens == 5, "gfx1250 DSpark requires draft width 5"),
        (config.compilation_config.level == 0, "gfx1250 DSpark requires level 0"),
        (len(model.layers) == 40, "gfx1250 DSpark requires 40 target layers"),
        (draft is not None and hasattr(draft, "block_backbone")
         and len(draft.mtp) == 3, "gfx1250 DSpark requires 3 draft layers"),
    ])


def install_dspark_decode(model, atom_config, drafter):
    why = trial_refusal(atom_config, model, drafter)
    if why is not None:
        raise RuntimeError(f"Requested gfx1250 DSpark mono refused: {why}")
    drafter.model = MonoDraftModel(drafter.model, runner_type=DraftMonoRunner)
    wrapped = MonoDecodeModel(
        model, drafter, atom_config.max_model_len, False, 0,
        runner_type=V41MonoDecodeRunner,
    )
    # Allocate immutable expert copies and the two scratch buffers before KV
    # profiling; compile all graph widths before recording any graph.
    for owner in (wrapped._mono, drafter.model._mono):
        owner.runner = owner._create()
        for rows in owner.runner.mhtjstest_widths:
            if not owner.ready(rows):
                raise RuntimeError(f"Requested gfx1250 DSpark width {rows} declined")
    logger.info("gfx1250 DSpark mono: target/draft c1-c8 bound before KV profiling")
    return wrapped
