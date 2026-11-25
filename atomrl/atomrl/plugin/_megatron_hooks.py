# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Framework-neutral Megatron and Lumen hook implementations."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

logger = logging.getLogger(__name__)


def megatron_init(args: Namespace) -> None:
    """Create the Lumen FP8 manager after Megatron distributed init."""
    fp8_config = getattr(args, "atomrl_fp8_config", None)
    if fp8_config is None:
        logger.info("atomrl: args.atomrl_fp8_config not set; FP8 disabled")
        return

    from atomrl.core.config import QuantizationConfig
    from atomrl.quantization.fp8_training import FP8TrainingManager

    quant_config = (
        QuantizationConfig(**fp8_config) if isinstance(fp8_config, dict) else fp8_config
    )
    args._atomrl_fp8_manager = FP8TrainingManager(quant_config)


def model_provider(
    pre_process: bool,
    post_process: bool,
    vp_stage: int | None = None,
):
    """Build a Megatron GPTModel and apply Lumen FP8 when configured."""
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args as megatron_get_args

    args = megatron_get_args()
    model = GPTModel(
        config=core_transformer_config_from_args(args),
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )
    manager = getattr(args, "_atomrl_fp8_manager", None)
    if manager is not None:
        manager.enable(model)
    return model


def post_save_hook(
    args: Any,
    rollout_id: int,
    checkpoint_dir: str,
    hf_checkpoint_dir: str,
) -> None:
    """Default no-op checkpoint hook retained for host CLI compatibility."""
    logger.debug(
        "atomrl: post-save rollout=%d checkpoint=%s hf=%s",
        rollout_id,
        checkpoint_dir,
        hf_checkpoint_dir,
    )


def reset_fp8_state(args: Any, model: Any) -> None:
    """Reset delayed-scaling state before training or log-prob forwards."""
    manager = getattr(args, "_atomrl_fp8_manager", None)
    if manager is None:
        return
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        manager.reset_fp8_state(chunk)


def update_weight_post_write(
    args: Any,
    version_dir: str,
    rollout_engines: list,
) -> None:
    """Reload an HF checkpoint on ATOM once, from distributed rank zero."""
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    manager = getattr(args, "_atomrl_atom_manager", None)
    if manager is None:
        logger.debug("atomrl: no ATOM manager; checkpoint reload skipped")
        return
    manager.reload_weights_from_path(version_dir)
    logger.info("atomrl: ATOM weights reloaded from %s", version_dir)
