# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""vime CLI hook functions for AtomRL injection.

Usage — pass each function via its dotted import path::

    python train.py \\
      --custom-megatron-init-path  atomrl.plugin.vime.hooks.custom_megatron_init \\
      --custom-model-provider-path atomrl.plugin.vime.hooks.custom_model_provider \\
      --rollout-function-path      atomrl.plugin.vime.hooks.generate_rollout

All logic is delegated to existing AtomRL modules.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook 1: --custom-megatron-init-path
# Signature: def custom_init(args: Namespace) -> None
# Called at the end of vime/backends/megatron_utils/initialize.py:init()
# ---------------------------------------------------------------------------


def custom_megatron_init(args: Namespace) -> None:
    """Set up Lumen FP8 training after Megatron distributed init.

    Reads ``args.atomrl_fp8_config`` (a dict or QuantizationConfig).
    Stores the FP8TrainingManager on ``args._atomrl_fp8_manager`` so that
    ``custom_model_provider`` can call ``manager.enable(model)`` later.
    """
    fp8_config = getattr(args, "atomrl_fp8_config", None)
    if fp8_config is None:
        logger.info("atomrl: args.atomrl_fp8_config not set; FP8 disabled.")
        return

    from atomrl.core.config import QuantizationConfig
    from atomrl.quantization.fp8_training import FP8TrainingManager

    if isinstance(fp8_config, dict):
        quant_cfg = QuantizationConfig(**fp8_config)
    else:
        quant_cfg = fp8_config

    args._atomrl_fp8_manager = FP8TrainingManager(quant_cfg)
    logger.info("atomrl: FP8TrainingManager created and stored on args.")


# ---------------------------------------------------------------------------
# Hook 2: --custom-model-provider-path
# Signature: def custom_model_provider(
#     pre_process: bool, post_process: bool, vp_stage: int | None = None
# ) -> GPTModel
# Called from vime/backends/megatron_utils/model_provider.py
# ---------------------------------------------------------------------------


def custom_model_provider(
    pre_process: bool,
    post_process: bool,
    vp_stage: int | None = None,
):
    """Build a Megatron GPTModel and optionally enable Lumen FP8.

    Returns a ``megatron.core.models.gpt.GPTModel`` instance. If
    ``args._atomrl_fp8_manager`` was set by ``custom_megatron_init``,
    FP8 quantization is applied to the model before returning.
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args as megatron_get_args

    args = megatron_get_args()
    transformer_config = core_transformer_config_from_args(args)
    layer_spec = get_gpt_layer_local_spec()

    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )

    fp8_manager = getattr(args, "_atomrl_fp8_manager", None)
    if fp8_manager is not None:
        fp8_manager.enable(model)
        logger.info("atomrl: FP8 enabled on GPTModel.")

    return model


# ---------------------------------------------------------------------------
# Hook 3: --rollout-function-path
# Signature: def generate_rollout(
#     args, rollout_id: int, data_source, evaluation: bool = False
# ) -> RolloutFnTrainOutput | RolloutFnEvalOutput
# Called from vime/ray/rollout/rollout_manager.py
# ---------------------------------------------------------------------------


def generate_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
):
    """Drive ATOM rollout in place of vime's default vLLM rollout.

    Requires ``args._atomrl_atom_manager`` to be set by the trainer script
    before the first rollout call. The manager is an ``ATOMReplicaManager``
    instance that owns the ATOM Ray actors.
    """
    manager = getattr(args, "_atomrl_atom_manager", None)
    if manager is None:
        raise RuntimeError(
            "args._atomrl_atom_manager not set. Create ATOMReplicaManager "
            "and store it on args before the first rollout call."
        )

    import ray

    if evaluation:
        logger.info("atomrl: ATOM eval rollout (rollout_id=%d)", rollout_id)
        # Delegate to default vime eval or return empty result.
        # Users should override --eval-function-path separately if needed.
        from vime.rollout.base_types import RolloutFnEvalOutput

        return RolloutFnEvalOutput(data={})

    from vime.rollout.base_types import RolloutFnTrainOutput

    manager.wake_all(tags=["weights", "kv_cache"])

    samples_nested = data_source.get_samples(rollout_id)
    results: list[list[Any]] = []
    for group in samples_nested:
        group_out = []
        for sample in group:
            prompt = (
                sample if isinstance(sample, str) else getattr(sample, "prompt", "")
            )
            max_tokens = getattr(args, "rollout_max_response_len", 2048)
            futures = [
                server.generate.remote(
                    prompt=prompt,
                    sampling_params={"max_tokens": max_tokens},
                )
                for server in manager._servers
            ]
            out = ray.get(futures[0])
            group_out.append(out)
        results.append(group_out)

    manager.sleep_all()
    return RolloutFnTrainOutput(samples=results)
