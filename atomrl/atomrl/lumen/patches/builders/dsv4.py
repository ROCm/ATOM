# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DSV4 CONFIG_BUILD / ARGS / MODEL_BUILD registrations."""

from __future__ import annotations

from atomrl.lumen.models.utils import safe_add_argument
from atomrl.lumen.patches.registry import PatchPhase, register_patch


@register_patch(
    "dsv4_config_core",
    PatchPhase.CONFIG_BUILD,
    description="Enable dsv4_mode and set unpadded vocab_size for hash routing",
    tags=frozenset({"dsv4", "builder"}),
    config_fields=("dsv4_mode", "vocab_size"),
)
def mutate_dsv4_config_core(config, args) -> None:
    config.dsv4_mode = True
    # ROCm Megatron's TransformerConfig predates unpadded vocab_size. tid2eid is
    # [vocab_size, topk] in the checkpoint even though embeddings use padded size.
    config.vocab_size = int(args.vocab_size)


@register_patch(
    "dsv4_config_pipeline",
    PatchPhase.CONFIG_BUILD,
    description="PP settings and mHC 4-D P2P shape exchange for DSV4",
    tags=frozenset({"dsv4", "builder"}),
    config_fields=("variable_seq_lengths", "batch_p2p_comm"),
)
def mutate_dsv4_config_pipeline(config, args) -> None:
    if getattr(args, "pipeline_model_parallel_size", 1) > 1:
        from atomrl.lumen.models.dsv4.megatron.pipeline import (
            install_dsv4_pipeline_shape_exchange,
        )

        install_dsv4_pipeline_shape_exchange()
        config.variable_seq_lengths = True
        config.batch_p2p_comm = False


@register_patch(
    "dsv4_spec_config",
    PatchPhase.CONFIG_BUILD,
    description="Runtime-only DSA top-k backend selector",
    tags=frozenset({"dsv4", "spec"}),
    config_fields=("dsv4_dsa_topk_backend",),
)
def mutate_dsv4_spec_config(config, args) -> None:
    config.dsv4_dsa_topk_backend = getattr(args, "dsv4_dsa_topk_backend", "torch")


@register_patch(
    "dsv4_pretrain_args",
    PatchPhase.ARGS,
    description="DSV4-specific Megatron CLI flags",
    tags=frozenset({"dsv4"}),
    default=False,
)
def _register_dsv4_pretrain_args(parser):
    """Register DSV4-specific Megatron CLI flags."""
    group = parser.add_argument_group(title="dsv4 pretrain")
    safe_add_argument(
        group,
        "--dsv4-dsa-topk-backend",
        type=str,
        default="torch",
        choices=["torch", "flashinfer"],
        help="Top-k backend for DSV4 DSA indexer.",
    )
    safe_add_argument(
        group,
        "--original-max-position-embeddings",
        type=int,
        default=4096,
        help="Original maximum position embeddings for YaRN RoPE (MLA).",
    )
    safe_add_argument(
        group, "--beta-fast", type=float, default=32, help="YaRN beta_fast (MLA)."
    )
    safe_add_argument(
        group, "--beta-slow", type=float, default=1, help="YaRN beta_slow (MLA)."
    )
    safe_add_argument(
        group,
        "--moe-router-freeze-gate",
        action="store_true",
        help="Freeze MoE router gate weights during training.",
    )
    safe_add_argument(
        group,
        "--freeze-e-score-correction-bias",
        action="store_true",
        help="Freeze MoE expert score correction bias during training.",
    )
    safe_add_argument(
        group,
        "--no-activation-func-clamp-shared-expert",
        action="store_false",
        dest="activation_func_clamp_shared_expert",
        default=True,
        help="Skip activation clamp inside shared expert MLP.",
    )
    return parser


def add_dsv4_pretrain_args(parser):
    """Apply DSV4 ARGS-phase patches to *parser*."""
    from atomrl.lumen.patches.builders import apply_args_patches

    apply_args_patches(parser, tags={"dsv4"})
    return parser


def _mori_enabled() -> bool:
    from atomrl.lumen.models.dsv4.megatron.moe_mori import mori_ep_enabled

    return mori_ep_enabled()


@register_patch(
    "dsv4_megatron_bootstrap",
    PatchPhase.MODEL_BUILD,
    description="One-time Megatron core init shims for DSV4 spec (optimizer, MoE, JIT)",
    tags=frozenset({"dsv4", "spec"}),
)
def install_dsv4_megatron_bootstrap(**_kwargs) -> None:
    from atomrl.lumen.models.dsv4.megatron.deepseek_v4 import (
        patch_dsv4_megatron_bootstrap,
    )

    patch_dsv4_megatron_bootstrap()


@register_patch(
    "dsv4_moe_mori",
    PatchPhase.MODEL_BUILD,
    description="MoEMori token dispatcher when LUMEN_DSV4_MOE_MORI=1",
    enabled=_mori_enabled,
    tags=frozenset({"dsv4", "spec", "moe"}),
)
def install_dsv4_moe_mori(**_kwargs) -> None:
    from atomrl.lumen.models.dsv4.megatron.moe_mori import patch_megatron_moe_mori

    patch_megatron_moe_mori()


def build_dsv4_transformer_block_spec(config, vp_stage):
    """Build DSV4 transformer block spec with temporary EAV module hooks."""
    from megatron.core.models.gpt import (
        experimental_attention_variant_module_specs as _eav_specs,
    )
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    from atomrl.lumen.models.dsv4.megatron.deepseek_v4 import (
        _dsv4_attention_module_spec,
    )
    from atomrl.lumen.models.dsv4.megatron.spec_provider import LumenDSV4SpecProvider

    _orig_get_spec = _eav_specs.get_experimental_attention_variant_module_spec
    _orig_get_backend = _eav_specs._get_backend_spec_provider

    def _patched_get_spec(config, backend=None):
        if config.experimental_attention_variant == "dsv4":
            return _dsv4_attention_module_spec(config, backend)
        return _orig_get_spec(config, backend)

    def _lumen_backend_spec_provider(config):
        return LumenDSV4SpecProvider()

    _eav_specs.get_experimental_attention_variant_module_spec = _patched_get_spec
    _eav_specs._get_backend_spec_provider = _lumen_backend_spec_provider
    try:
        return get_transformer_block_with_experimental_attention_variant_spec(
            config, vp_stage=vp_stage
        )
    finally:
        _eav_specs.get_experimental_attention_variant_module_spec = _orig_get_spec
        _eav_specs._get_backend_spec_provider = _orig_get_backend
