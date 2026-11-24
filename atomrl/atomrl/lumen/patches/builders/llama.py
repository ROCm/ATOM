# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""LLaMA / generic Megatron CONFIG_BUILD and ARGS registrations."""

from __future__ import annotations

from atomrl.lumen.patches.registry import PatchPhase, register_patch


@register_patch(
    "llama_pretrain_args",
    PatchPhase.ARGS,
    description="LLaMA 3.1 pretrain mlperf Docker compatibility flags",
    tags=frozenset({"llama", "pretrain", "megatron"}),
    depends_on=("common_megatron_args",),
    default=False,
)
def add_llama_pretrain_args(parser):
    """Register LLaMA 3.1 pretrain-only CLI flags."""
    mlperf = parser.add_argument_group(title="mlperf")
    mlperf.add_argument(
        "--size",
        type=str,
        default="8b",
        choices=["8b"],
        help="Model size (for Docker compatibility).",
    )
    mlperf.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Number of nodes (Docker compat, unused by Megatron).",
    )
    mlperf.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="GPUs per node (Docker compat, unused by Megatron).",
    )
    return parser


@register_patch(
    "lumen_gpt_config",
    PatchPhase.CONFIG_BUILD,
    description="Lumen GPT defaults: no persist_layer_norm / bias_swiglu_fusion",
    tags=frozenset({"lumen", "builder"}),
    config_fields=("apply_rope_fusion", "persist_layer_norm", "bias_swiglu_fusion"),
)
def mutate_lumen_gpt_config(config, args) -> None:
    # core_transformer_config_from_args reads rope fusion off args, so a config
    # handed down by get_model can disagree with what the launcher asked for.
    config.apply_rope_fusion = getattr(args, "apply_rope_fusion", False)
    config.persist_layer_norm = False
    config.bias_swiglu_fusion = False
    if getattr(args, "lumen_fp8_activation_store", False):
        config.activation_func_fp8_input_store = True
