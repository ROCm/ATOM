# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Shared Megatron-LM training components for Lumen models.

This module consolidates the building blocks that are common to all
Megatron-LM-based training scripts (LLaMA2 SFT, LLaMA 3.1 pretraining, etc.).

Model-specific code (batch construction, dataset providers, model-specific CLI
arguments) remains in the per-model subpackages.
"""

import os
import warnings
from collections.abc import Callable
from functools import partial

import torch

# ---------------------------------------------------------------------------
# Megatron compatibility patches (must run before any Megatron model imports)
# ---------------------------------------------------------------------------
from atomrl.lumen.models.megatron_patches import (
    install_all as _install_megatron_patches,
)
from atomrl.lumen.patches.builders import apply_config_build, apply_model_build

_install_megatron_patches()


from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from atomrl.lumen.modules.attention_megatron import (
    LumenDotProductAttention,
)
from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

stimer = StragglerDetector()


# Backward-compatible re-exports (implementations live in patch registry).
from atomrl.lumen.patches.builders.megatron_model import (
    _NORM_ATTRS,
    GPT_LOCAL_MODEL_PATCHES,
    GPT_LOCAL_SPEC_PATCHES,
    GPT_LUMEN_MODEL_PATCHES,
    GPT_LUMEN_SPEC_PATCHES,
    _MegatronCompatibleTLLayerNorm,
    _MegatronCompatibleTLNorm,
    _MegatronCompatibleTLRMSNorm,
    _patch_all_norms,
    _patch_core_attention,
    _patch_fused_swiglu_mlp,
    _patch_layernorm,
    _patch_mla_attention,
    _patch_norms_in_spec,
    _patch_rmsnorm,
)

# Per-head QK norms sit one level down, inside the attention submodule spec, and
# normalise over head_dim instead of hidden_size.
_ATTN_NORM_ATTRS = ("q_layernorm", "k_layernorm")


# ---------------------------------------------------------------------------
# Override defaults for Lumen
# ---------------------------------------------------------------------------

from atomrl.lumen.patches.builders.megatron_args import (
    TE_FORCE_OVERRIDES as _TE_FORCE_OVERRIDES,
)
from atomrl.lumen.quantize.config import MXFP4_BLOCK_SIZE

_FP8_FORMAT_MAP = {"e4m3": "fp8_e4m3", "hybrid": "hybrid"}

# MX FP4 scales one block of 32 elements (OCP Microscaling spec); the FP4 GEMM
# and quantization kernels all assume that block length.
_MXFP4_BLOCK_SIZE = MXFP4_BLOCK_SIZE

_BACKEND_MAP = {
    "auto": ("aiter_csrc", "aiter_triton_fp8"),
    "triton": ("aiter_triton", "aiter_triton_fp8"),
    "csrc": ("aiter_csrc", "aiter_csrc_fp8"),
    "asm": ("aiter_csrc", "aiter_asm_fp8"),
}


def resolve_attn_backend(backend: str, fp8_attn: str) -> str:
    """Derive the concrete ``aiter_*`` backend string from user-facing flags.

    Args:
        backend: One of ``auto``, ``triton``, ``csrc``, ``asm``.
        fp8_attn: One of ``none``, ``dpa``, ``mha``.

    Returns:
        A concrete backend name like ``aiter_triton_fp8``.
    """
    bf16_be, fp8_be = _BACKEND_MAP.get(backend, ("aiter_csrc", "aiter_triton_fp8"))
    return fp8_be if fp8_attn in ("dpa", "mha") else bf16_be


def resolve_quant_format(args) -> str | None:
    """Return the effective Lumen quantisation format string, or None.

    MXFP4 has its own ``--linear-fp4`` gate. FP8 formats follow the same
    precedence :data:`atomrl.lumen.config._ARG_MAP` uses for the ``format`` field:
    ``--linear-fp8-format`` first, then Megatron's ``--fp8-format``.
    """
    if getattr(args, "linear_fp4", False):
        return "mxfp4"
    for attr in ("linear_fp8_format", "lumen_fp8_format"):
        value = getattr(args, attr, None)
        if value is not None:
            return value
    return None


def _override_te_args_for_lumen(args):
    """Configure Lumen FP8 settings from Megatron args.

    The ``--fp8-format`` value (``args.fp8``) is mapped to the Lumen
    :class:`QuantFormat` string and stored as ``args.lumen_fp8_format`` for
    :func:`apply_fp8_training`.  ``args.fp8`` is then set to ``None`` so
    that ``TransformerConfig`` uses Lumen's own FP8 code-paths.
    ``--linear-fp8-format`` overrides that mapping for FP8/MXFP8. MXFP4 is
    selected independently by ``--linear-fp4``.

    All other shared parameters (``fp8_margin``, ``fp8_recipe``,
    ``fp8_amax_history_len``, ``fp8_amax_compute_algo``, ``fp8_wgrad``,
    ``first_last_layers_bf16``, etc.) are kept as-is.
    """
    te_fp8 = getattr(args, "fp8", None)
    if te_fp8 is not None:
        args.lumen_fp8_format = _FP8_FORMAT_MAP.get(te_fp8, te_fp8)
    args.fp8 = None

    for attr, value in _TE_FORCE_OVERRIDES.items():
        setattr(args, attr, value)

    fp8_attn = getattr(args, "lumen_fp8_attn", "none")
    if getattr(args, "fp8_multi_head_attention", False):
        fp8_attn = "mha"
    elif getattr(args, "fp8_dot_product_attention", False) and fp8_attn == "none":
        fp8_attn = "dpa"
    args.lumen_fp8_attn = fp8_attn

    backend_base = getattr(args, "lumen_attn_backend", "auto")
    args.lumen_attn_backend = resolve_attn_backend(backend_base, fp8_attn)

    if getattr(args, "lumen_cross_entropy", False):
        _patch_cross_entropy()


_cross_entropy_patched = False


def _patch_cross_entropy():
    """Backward-compatible alias; see :func:`install_cross_entropy`."""
    from atomrl.lumen.patches.runtime.megatron_import import install_cross_entropy

    install_cross_entropy()


# ---------------------------------------------------------------------------
# Custom GPT builder that injects Lumen attention
# ---------------------------------------------------------------------------


def lumen_gpt_builder(
    args, pre_process, post_process, vp_stage=None, config=None, model_name="GPT"
):
    """Build a GPTModel with Lumen attention replacing the default
    DotProductAttention in every layer.

    When ``--lumen-linear`` is set, uses the Lumen spec provider for all
    linear, norm, and attention modules.  Otherwise, uses the Megatron-Core
    local spec and patches attention/norms post-hoc.

    Args:
        model_name: Label used in the startup log message (e.g. ``"LLaMA 3.1"``).
    """
    if getattr(args, "lumen_linear", False):
        return lumen_gpt_builder_with_spec(
            args,
            pre_process,
            post_process,
            vp_stage=vp_stage,
            config=config,
            model_name=model_name,
        )

    print_rank_0(f"building {model_name} model with Lumen attention ...")

    _override_te_args_for_lumen(args)

    # Rope fusion needs a kernel only TransformerEngine or Lumen's apex bridge
    # supplies, so it stays opt-in. Set on args because
    # core_transformer_config_from_args reads it from there.
    args.apply_rope_fusion = getattr(args, "lumen_fused_rope", False)

    if config is None:
        config = core_transformer_config_from_args(args)

    # Outside the `if` on purpose: these hold regardless of whether the config
    # was built here or handed down by Megatron's get_model, which passes one by
    # keyword. The registry's lumen_gpt_config patch carries the assignments.
    apply_config_build(config, args, tags={"lumen", "builder"})

    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
        use_kitchen=config.use_kitchen,
    )

    apply_model_build(
        config=config,
        args=args,
        spec=transformer_layer_spec,
        names=GPT_LOCAL_SPEC_PATCHES,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    apply_model_build(
        config=config,
        args=args,
        model=model,
        names=GPT_LOCAL_MODEL_PATCHES,
    )

    return model


def lumen_gpt_builder_with_spec(
    args, pre_process, post_process, vp_stage=None, config=None, model_name="GPT"
):
    """Build a GPTModel using the Lumen spec provider.

    Instead of patching individual modules post-hoc, this builder uses
    :class:`~atomrl.lumen.models.spec_provider.LumenSpecProvider` to produce a
    layer spec where *all* linear, norm, and attention modules are Lumen
    classes from the start.  This is the recommended path for full Lumen
    integration including FP8 parallel linear layers.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    from atomrl.lumen.models.spec_provider import LumenSpecProvider

    print_rank_0(f"building {model_name} model with Lumen spec provider ...")

    _override_te_args_for_lumen(args)

    if config is None:
        args.apply_rope_fusion = getattr(args, "lumen_fused_rope", False)
        config = core_transformer_config_from_args(args)
        apply_config_build(config, args, tags={"lumen", "builder"})

    # Monkey-patch the TE spec provider lookup so get_gpt_layer_with_transformer_engine_spec
    # picks up Lumen modules without modifying Megatron source.
    import megatron.core.models.gpt.gpt_layer_specs as _gls

    _orig_te_spec = getattr(_gls, "TESpecProvider", None)
    _gls.TESpecProvider = LumenSpecProvider
    _gls.HAVE_TE = True

    try:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=getattr(args, "num_experts", None),
            moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
            qk_layernorm=getattr(args, "qk_layernorm", False),
            multi_latent_attention=getattr(args, "multi_latent_attention", False),
            moe_use_legacy_grouped_gemm=getattr(
                args, "moe_use_legacy_grouped_gemm", False
            ),
            use_kitchen=getattr(config, "use_kitchen", False),
        )
    finally:
        if _orig_te_spec is not None:
            _gls.TESpecProvider = _orig_te_spec

    _subs = transformer_layer_spec.submodules
    _existing_map = getattr(_subs, "sharded_state_dict_keys_map", None) or {}
    _existing_map.update(
        {
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        }
    )
    _subs.sharded_state_dict_keys_map = _existing_map

    apply_model_build(
        config=config,
        args=args,
        spec=transformer_layer_spec,
        names=GPT_LUMEN_SPEC_PATCHES,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    apply_model_build(
        config=config,
        args=args,
        model=model,
        names=GPT_LUMEN_MODEL_PATCHES,
    )

    return model


_LN_KEY_REMAP_CANDIDATES = [
    (
        "input_layernorm.",
        [
            "self_attention.linear_qkv.layer_norm_",
            "self_attention.linear_qkv.base_layer.layer_norm_",
        ],
    ),
    (
        "pre_mlp_layernorm.",
        [
            "mlp.linear_fc1.layer_norm_",
            "mlp.linear_fc1.base_layer.layer_norm_",
        ],
    ),
]


def _install_layernorm_linear_ckpt_hook(model):
    """Remap legacy checkpoint keys for fused LayerNormLinear modules.

    Legacy Megatron checkpoints store norm weights under
    ``decoder.layers.N.input_layernorm.weight`` etc., but the fused
    ``LumenLayerNormLinear`` (used in the TE-style spec) expects them at
    ``decoder.layers.N.self_attention.linear_qkv.layer_norm_weight``.

    When LoRA is applied, the target has an extra ``base_layer.`` segment
    (e.g. ``linear_qkv.base_layer.layer_norm_weight``).
    """

    from megatron.core.transformer.transformer_layer import TransformerLayer

    model_keys = {n for n, _ in model.named_parameters()}
    model_keys |= {n for n, _ in model.named_buffers()}

    first_tl_prefix = None
    for mod_name, mod in model.named_modules():
        if isinstance(mod, TransformerLayer):
            first_tl_prefix = f"{mod_name}." if mod_name else ""
            break

    if first_tl_prefix is None:
        return

    _resolved_map: dict = {}
    for old_pfx, candidates in _LN_KEY_REMAP_CANDIDATES:
        for cand in candidates:
            probe = f"{first_tl_prefix}{cand}weight"
            if probe in model_keys:
                _resolved_map[old_pfx] = cand
                break

    if not _resolved_map:
        return

    def _remap_hook(state_dict, prefix, *_args, **_kwargs):
        for old_pfx, new_pfx in _resolved_map.items():
            for suffix in ("weight", "bias"):
                old_key = f"{prefix}{old_pfx}{suffix}"
                new_key = f"{prefix}{new_pfx}{suffix}"
                if old_key in state_dict and new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

    for module in model.modules():
        if isinstance(module, TransformerLayer):
            module._register_load_state_dict_pre_hook(_remap_hook)


def _enable_quantization_for_parallel_linear(
    model,
    scaling_manager=None,
    scaling_type="dynamic",
    fp8_dtype=None,
    block_size=None,
    fp8_mha=False,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    quant_config=None,
    _display_name="quantization",
):
    """Enable low-precision GEMMs on Lumen parallel linear modules.

    When *fp8_mha* is True, a shared :class:`Blockwise2DScaleManager` is
    attached to each ``LumenDotProductAttention`` (or MLA variant) so that
    QKV projection, dot-product attention and output projection share the
    same FP8 scale context within a single MHA block.

    When *delay_wgrad* is True, backward passes compute only dgrad
    immediately and defer wgrad to a later ``backward_dw()`` call.

    When *gradient_accumulation_fusion* is True, weight gradients
    accumulate directly into ``param.main_grad``.

    When *quant_config* is provided, per-module ScalingManagers are created
    with this config (ensuring correct amax_algo, history_len, etc.).
    """
    from atomrl.lumen.modules.grouped_linear import LumenGroupedLinear
    from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear
    from atomrl.lumen.modules.parallel_linear import (
        LumenColumnParallelLinear,
        LumenRowParallelLinear,
    )

    if fp8_dtype is None:
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()

    # Leaving this None keeps the linears on their init default of 128, and
    # _mxfp4_cached_weight would then compare a scale grid built at 32 against a
    # tile grid derived from 128 -- never equal, so the swizzle fusion is dead.
    if block_size is None and quant_config is not None:
        block_size = quant_config.block_size

    # Tell the fused SwiGLU quant bridge (LUMEN_FUSED_SWIGLU_QUANT) the global
    # activation scale granularity so its cached scale layout matches the fc2
    # GEMM that consumes it (blockwise2d needs a 2D 1×block scale, not 1D).
    from atomrl.lumen.models._swiglu_fp8_fuse import set_fused_swiglu_scaling

    set_fused_swiglu_scaling(scaling_type, block_size)

    # --first-last-layers-bf16 has to be applied here too: the other place that
    # honours it, quantize._patch_linear_layers, only sees Megatron's own linear
    # types, which --lumen-linear has already swapped out by then.
    from atomrl.lumen.quantize import is_under_bf16_prefix

    bf16_prefixes: set[str] = set()
    if quant_config is not None and quant_config.first_last_layers_bf16:
        from atomrl.lumen.quantize import _build_bf16_skip_prefixes

        if quant_config.num_layers > 0:
            bf16_prefixes = _build_bf16_skip_prefixes(model, quant_config)
        else:
            # Without the layer count the tail is unidentifiable, and the skip
            # rule would read as "every layer". Quantizing everything is the
            # documented-but-wrong behaviour; silently running the whole model
            # in BF16 would be worse and much harder to notice.
            warnings.warn(
                "first_last_layers_bf16 is set but num_layers is 0, so the BF16 "
                "tail cannot be located; quantizing every Lumen parallel linear.",
                stacklevel=2,
            )

    count = 0
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(
            module,
            (
                LumenColumnParallelLinear,
                LumenRowParallelLinear,
                LumenLayerNormLinear,
                LumenGroupedLinear,
            ),
        ):
            if bf16_prefixes and is_under_bf16_prefix(name, bf16_prefixes):
                skipped += 1
                continue
            bad_mxfp4_weight = (
                next(
                    (
                        weight
                        for weight in _mxfp4_weight_parameters(module)
                        if weight.shape[0] % _MXFP4_BLOCK_SIZE
                    ),
                    None,
                )
                if scaling_type == "mxfp4"
                else None
            )
            if bad_mxfp4_weight is not None:
                # The MXFP4 quantizer pads a weight's output rows up to 32 and
                # drops the original count, so forward's
                # output.view(..., weight.shape[0]) fails on the padded width --
                # a RuntimeError mid-step, with no fallback, for a shape that
                # was knowable here. N is hidden size / vocab / a TP shard, so
                # it is fixed for the run: decide once and leave the layer BF16.
                print_rank_0(
                    f"> {name}: output width {bad_mxfp4_weight.shape[0]} is not a multiple of "
                    f"{_MXFP4_BLOCK_SIZE}, leaving this layer in BF16"
                )
                skipped += 1
                continue
            _mgr = scaling_manager
            if _mgr is None and quant_config is not None:
                from atomrl.lumen.quantize import ScalingManager

                _mgr = ScalingManager(quant_config)
            module.enable_fp8(
                scaling_manager=_mgr,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
            )
            if hasattr(module, "gradient_accumulation_fusion"):
                module.gradient_accumulation_fusion = gradient_accumulation_fusion
            if hasattr(module, "delay_wgrad"):
                module.delay_wgrad = delay_wgrad
            count += 1

    if fp8_mha:
        from atomrl.lumen.quantize.scaling_manager import Blockwise2DScaleManager

        attn_count = 0
        for module in model.modules():
            if isinstance(
                module, (LumenDotProductAttention, LumenDotProductAttentionMLA)
            ):
                module.scale_manager = Blockwise2DScaleManager()
                attn_count += 1
        if attn_count > 0:
            print_rank_0(
                f"> Attached Blockwise2DScaleManager to {attn_count} attention modules for FP8 MHA"
            )

    if count > 0:
        note = f", {skipped} left in BF16" if skipped else ""
        print_rank_0(
            f"> Enabled {_display_name} (scaling={scaling_type}) "
            f"on {count} Lumen parallel linear modules{note}"
        )


def _mxfp4_weight_parameters(module):
    """Return the 2-D weights whose output widths this module's GEMMs expose."""
    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() == 2:
        return (weight,)

    # Grouped/MoE modules follow Megatron's checkpoint names: weight0..weightN.
    # They have no plain ``.weight``, but each expert independently reaches
    # quantize_input and therefore needs the same output-width contract.
    return tuple(
        param
        for param_name, param in module._parameters.items()
        if param_name.startswith("weight")
        and param_name[6:].isdigit()
        and param is not None
        and param.dim() == 2
    )


def _mxfp4_weight_shape_supported(module) -> bool:
    """Whether MXFP4 can run this module's GEMM without padding its output.

    ``quantize_input(is_weight=True)`` pads the weight's rows to the 32-element
    block and returns no record of the original count, so a width that is not a
    multiple of 32 reaches the forward's reshape as a wider tensor than the
    caller asked for. The reduction dim K is fine either way: both operands are
    padded along it, and zeros do not contribute.

    Grouped/MoE modules expose ``weight0``...``weightN`` instead of a plain
    weight; every expert is checked.
    """
    return all(
        weight.shape[0] % _MXFP4_BLOCK_SIZE == 0
        for weight in _mxfp4_weight_parameters(module)
    )


def enable_fp8_for_parallel_linear(
    model,
    scaling_manager=None,
    scaling_type="dynamic",
    fp8_dtype=None,
    block_size=None,
    fp8_mha=False,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    quant_config=None,
):
    """Enable FP8 GEMMs on Lumen parallel linear modules."""
    return _enable_quantization_for_parallel_linear(
        model,
        scaling_manager=scaling_manager,
        scaling_type=scaling_type,
        fp8_dtype=fp8_dtype,
        block_size=block_size,
        fp8_mha=fp8_mha,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        delay_wgrad=delay_wgrad,
        quant_config=quant_config,
        _display_name="FP8",
    )


def enable_fp4_for_parallel_linear(
    model,
    scaling_manager=None,
    scaling_type="mxfp4",
    fp8_dtype=None,
    block_size=32,
    fp8_mha=False,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    quant_config=None,
):
    """Enable MXFP4 GEMMs on Lumen parallel linear modules."""
    from atomrl.lumen.quantize import assert_mxfp4_arch_supported

    if scaling_type != "mxfp4":
        raise ValueError(
            f"FP4 enablement requires scaling_type='mxfp4', got {scaling_type!r}"
        )
    # This entry point patches the linears directly rather than going through
    # quantize.enable, so the support matrix has to be checked here too.
    assert_mxfp4_arch_supported()
    if block_size != _MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 requires block_size={_MXFP4_BLOCK_SIZE}, got {block_size}"
        )
    if quant_config is None:
        from atomrl.lumen.quantize import QuantConfig

        quant_config = QuantConfig.from_str(
            format="mxfp4",
            scaling="blockwise",
            block_size=_MXFP4_BLOCK_SIZE,
        )
    return _enable_quantization_for_parallel_linear(
        model,
        scaling_manager=scaling_manager,
        scaling_type=scaling_type,
        fp8_dtype=fp8_dtype,
        block_size=block_size,
        fp8_mha=fp8_mha,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        delay_wgrad=delay_wgrad,
        quant_config=quant_config,
        _display_name="MXFP4",
    )


# ---------------------------------------------------------------------------
# LoRA (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------------------------


def _patch_lora_for_layernorm_linear(model):
    """Backward-compatible alias; see :func:`patch_lora_for_layernorm_linear`."""
    from atomrl.lumen.patches.builders.megatron_model import (
        patch_lora_for_layernorm_linear,
    )

    patch_lora_for_layernorm_linear(model)


def apply_lora(model: GPTModel, args) -> None:
    """Wrap linear layers with LoRA adapters for parameter-efficient fine-tuning.

    Target modules controlled by ``--lora-target-modules``:

    * ``"attention"`` — QKV + output projection only (NeMo reference).
    * ``"attention_mlp"`` — attention + MLP (gate/up + down).
    * ``"all"`` (default) — attention + MLP + embedding + output layer.
    """
    from megatron.core.transformer.lora_adapter import (
        COLUMN_PARALLEL_LAYERS,
        LORA_LAYERS_MAPPING,
        ROW_PARALLEL_LAYERS,
        LoraAdapter,
    )

    from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear
    from atomrl.lumen.modules.parallel_linear import (
        LumenColumnParallelLinear,
        LumenRowParallelLinear,
    )

    if LumenLayerNormLinear not in LORA_LAYERS_MAPPING:
        LORA_LAYERS_MAPPING[LumenLayerNormLinear] = COLUMN_PARALLEL_LAYERS
    if LumenColumnParallelLinear not in LORA_LAYERS_MAPPING:
        LORA_LAYERS_MAPPING[LumenColumnParallelLinear] = COLUMN_PARALLEL_LAYERS
    if LumenRowParallelLinear not in LORA_LAYERS_MAPPING:
        LORA_LAYERS_MAPPING[LumenRowParallelLinear] = ROW_PARALLEL_LAYERS

    common = {
        "config": model.config,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }

    target = getattr(args, "lora_target_modules", "all")

    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "decoder") and model.decoder is not None:
        for layer in model.decoder.layers:
            layer.self_attention.linear_qkv = LoraAdapter(
                layer.self_attention.linear_qkv, **common
            )
            layer.self_attention.linear_proj = LoraAdapter(
                layer.self_attention.linear_proj, **common
            )
            if (
                target in ("all", "attention_mlp")
                and hasattr(layer, "mlp")
                and layer.mlp is not None
            ):
                layer.mlp.linear_fc1 = LoraAdapter(layer.mlp.linear_fc1, **common)
                layer.mlp.linear_fc2 = LoraAdapter(layer.mlp.linear_fc2, **common)

    if (
        target == "all"
        and hasattr(model, "embedding")
        and model.embedding is not None
        and hasattr(model.embedding, "word_embeddings")
    ):
        model.embedding.word_embeddings = LoraAdapter(
            model.embedding.word_embeddings, **common
        )
    if (
        target == "all"
        and hasattr(model, "output_layer")
        and model.output_layer is not None
    ):
        model.output_layer = LoraAdapter(model.output_layer, **common)

    from atomrl.lumen.patches.builders import apply_model_build

    apply_model_build(
        model=model,
        config=model.config,
        args=args,
        names={"lora_layernorm_linear"},
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"target={target}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------


def apply_fp8_training(model: GPTModel, args) -> None:
    """Enable FP8 quantised training via Lumen's non-invasive patching.

    .. deprecated:: Prefer :meth:`LumenConfig.enable` directly.
    """
    from atomrl.lumen.config import LumenConfig

    cfg = LumenConfig.from_args(args)

    dp_group = None
    if cfg.reduce_amax:
        import torch.distributed as dist
        from megatron.core import parallel_state

        if dist.is_initialized():
            dp_group = parallel_state.get_data_parallel_group()

    import atomrl.lumen.quantize as quant

    qcfg = cfg.quant_config
    quant.enable(model, config=qcfg, dp_group=dp_group)
    print_rank_0(
        f"> FP8 training enabled (format={cfg.format}, scaling={cfg.scaling}, "
        f"block_size={cfg.block_size}, amax_algo={cfg.amax_algo}, "
        f"reduce_amax={cfg.reduce_amax}, history={cfg.history_len}, "
        f"activation={cfg.quantize_activation}, grad_quant={cfg.quantize_grad})"
    )


def _find_scaling_manager(model):
    """Retrieve the ScalingManager from quant-patched modules."""
    for module in model.modules():
        sm = getattr(module, "_quant_manager", None)
        if sm is not None:
            return sm
    return None


def _enable_lumen_fp8_checkpoint(scaling_manager):
    """Monkey-patch tensor_parallel.checkpoint to preserve FP8 scaling state."""
    import megatron.core.tensor_parallel as tp_module
    import megatron.core.tensor_parallel.random as tp_random

    from atomrl.lumen.utils.checkpoint import _FP8ScalingContext

    if hasattr(tp_module, "_lumen_fp8_checkpoint_patched"):
        return

    _original = tp_random.checkpoint

    def _patched(function, distribute_saved_activations, *args):
        ctx = _FP8ScalingContext()
        ctx.save(scaling_manager)
        orig_fn = function

        def wrapped(*a, **kw):
            ctx.restore(scaling_manager)
            return orig_fn(*a, **kw)

        return _original(wrapped, distribute_saved_activations, *args)

    tp_random.checkpoint = _patched
    tp_module.checkpoint = _patched
    tp_module._lumen_fp8_checkpoint_patched = True
    tp_module._lumen_fp8_checkpoint_original = _original
    print_rank_0("> FP8-aware activation checkpointing enabled (Lumen)")


def apply_lumen_pre_quant(model: GPTModel, args) -> None:
    """Phase 1: Set module attributes BEFORE quant.enable() captures them.

    .. deprecated:: Prefer :meth:`LumenConfig.enable` which handles this automatically.
    """
    from atomrl.lumen.config import LumenConfig

    LumenConfig.from_args(args)._apply_pre_quant(model)


apply_lumen_optimizations = apply_lumen_pre_quant


def apply_lumen_post_quant(model: GPTModel, args) -> None:
    """Phase 2: Features requiring ScalingManager (created by quant.enable).

    .. deprecated:: Prefer :meth:`LumenConfig.enable` which handles this automatically.
    """
    from atomrl.lumen.config import LumenConfig

    cfg = LumenConfig.from_args(args)
    sm = (
        _find_scaling_manager(model)
        if (cfg.fp8_checkpoint or cfg.fp8_param_gather)
        else None
    )
    cfg._apply_post_quant(model, sm)


def get_cpu_offload_context(args):
    """Return CPU offload context manager (no-op if disabled)."""
    from atomrl.lumen.utils.cpu_offload import lumen_cpu_offload_context

    enabled = getattr(args, "lumen_cpu_offload", False)
    return lumen_cpu_offload_context(enabled=enabled)


def make_lumen_model_provider(
    model_builder: Callable,
    *,
    lora_applier: Callable = apply_lora,
    fp8_applier: Callable = apply_fp8_training,
):
    """Build the canonical Megatron model-provider assembly for Lumen.

    The returned callable keeps task semantics in the supplied ``model_builder``
    while applying the shared infrastructure assembly in a fixed order:

    1. Megatron LoRA (uses ``megatron.core.transformer.lora_adapter``,
       separate from PEFT — handled by *lora_applier*)
    2. ``LumenConfig.enable()`` — FP8ParamManager, norm patching, pre-quant,
       ``quant.enable``, post-quant (PEFT LoRA skipped via ``lora_rank=0``)
    3. Megatron-specific ``enable_fp8_for_parallel_linear`` (optional)
    """

    def model_provider(
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
    ):
        # pg_collection is accepted but not forwarded: Megatron's get_model
        # defaults it to ProcessGroupCollection.use_mpu_process_groups(), which is
        # the same set GPTModel derives from parallel_state when given None.
        import os
        from dataclasses import replace as _replace

        from atomrl.lumen.config import LumenConfig

        args = get_args()
        model = model_builder(args, pre_process, post_process, vp_stage, config=config)

        # 1. Megatron LoRA (not PEFT — stays separate)
        if getattr(args, "lora_rank", 0) > 0:
            lora_applier(model, args)
            if getattr(args, "lora_a2a", False):
                os.environ["LORA_A2A"] = "1"
                print_rank_0("> LoRA A2A communication optimisation enabled")

        # 1b. Install checkpoint key remapping for fused LayerNormLinear
        if getattr(args, "lumen_linear", False):
            _install_layernorm_linear_ckpt_hook(model)

        # The native parallel-linear pass below is gated on --lumen-linear, but
        # cfg.enable() is not. Asking for quantized linears without it produced
        # a partially quantized model -- norms patched, params wrapped, GEMMs
        # untouched -- and reported nothing, so the run looked like a working
        # FP8/FP4 run and trained in BF16. Fail before anything is patched.
        if (
            getattr(args, "linear_fp8", False) or getattr(args, "linear_fp4", False)
        ) and not getattr(args, "lumen_linear", False):
            _flag = (
                "--linear-fp4" if getattr(args, "linear_fp4", False) else "--linear-fp8"
            )
            raise ValueError(
                f"{_flag} quantizes Lumen's parallel linear layers, which requires "
                "--lumen-linear to install them. Without it the GEMMs stay BF16 "
                f"while the rest of the model is patched. Add --lumen-linear, or "
                f"drop {_flag}."
            )

        # 2. Unified LumenConfig.enable() — skip PEFT LoRA (handled above)
        cfg = LumenConfig.from_args(args)
        cfg = _replace(cfg, lora_rank=0)

        dp_group = None
        if cfg.reduce_amax:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if dist.is_initialized():
                dp_group = parallel_state.get_data_parallel_group()

        _original_model = model
        _manager, model = cfg.enable(model, dp_group=dp_group)
        assert model is _original_model, (
            f"quant.enable() returned a different model object "
            f"(type {type(model).__name__} vs {type(_original_model).__name__}). "
            f"This indicates unexpected model wrapping (e.g. LoRA) that may "
            f"break Megatron's parameter management."
        )

        # 3. Megatron-specific native parallel linear quantization
        fp8_enabled = getattr(args, "linear_fp8", False)
        fp4_enabled = getattr(args, "linear_fp4", False)
        if (fp8_enabled or fp4_enabled) and getattr(args, "lumen_linear", False):
            # The resolved recipe, not the raw --linear-fp8-scaling string: MX
            # formats carry scaling in the format, so an MXFP4 run passes
            # "blockwise" here and would silently run FP8 blockwise instead.
            if fp4_enabled:
                enable_fp4_for_parallel_linear(
                    model,
                    scaling_type=cfg.quant_config.recipe,
                    block_size=cfg.quant_config.block_size,
                    fp8_mha=getattr(args, "lumen_fp8_attn", "none") == "mha",
                    gradient_accumulation_fusion=getattr(
                        args, "lumen_gradient_accumulation_fusion", False
                    ),
                    delay_wgrad=getattr(args, "lumen_delay_wgrad", False),
                    quant_config=cfg.quant_config,
                )
            else:
                enable_fp8_for_parallel_linear(
                    model,
                    scaling_type=cfg.quant_config.recipe,
                    block_size=cfg.quant_config.block_size,
                    fp8_mha=getattr(args, "lumen_fp8_attn", "none") == "mha",
                    gradient_accumulation_fusion=getattr(
                        args, "lumen_gradient_accumulation_fusion", False
                    ),
                    delay_wgrad=getattr(args, "lumen_delay_wgrad", False),
                    quant_config=cfg.quant_config,
                )

        if getattr(args, "fp8_param_storage", False):
            from atomrl.lumen.models.fp8_param_storage import (
                shrink_frozen_weights_to_fp8,
            )

            shrink_frozen_weights_to_fp8(model)

        return model

    return model_provider


# ---------------------------------------------------------------------------
# Synthetic warmup + FP8 state reset
# ---------------------------------------------------------------------------

_warmup_step_counter = 0
_warmup_completed = False


def _get_synthetic_batch(args, *, zero_last_loss_mask=False):
    """Generate a synthetic batch for GPU kernel warmup.

    The loss_mask is zeroed out entirely so that the optimizer step
    receives zero gradients and trainable weights (e.g. LoRA) are not
    corrupted by synthetic data.  Forward + backward still execute
    (warming up GPU kernels and calibrating FP8 amax history).

    Args:
        zero_last_loss_mask: Legacy flag (kept for API compat); the
            entire loss_mask is now always zeroed.
    """
    seq_length = args.seq_length
    mbs = args.micro_batch_size

    tokens = torch.ones(mbs, seq_length, dtype=torch.long, device="cuda") * 3545
    tokens[:, -1] = 2
    labels = tokens.clone()
    loss_mask = torch.zeros(mbs, seq_length, dtype=torch.float, device="cuda")
    attention_mask = torch.ones(
        mbs, 1, seq_length, seq_length, dtype=torch.bool, device="cuda"
    )
    position_ids = (
        torch.arange(seq_length, dtype=torch.long, device="cuda")
        .unsqueeze(0)
        .expand(mbs, -1)
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def reset_fp8_state(model):
    """Reset FP8 scaling state in all Lumen quantised layers."""

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.apply(_reset)
    print_rank_0("> FP8 state reset after warmup")


_WARMUP_EVAL_STEPS = int(os.environ.get("LUMEN_WARMUP_EVAL_STEPS", "0"))


def _run_warmup_eval_pass(model, args):
    """Run synthetic forward passes in eval mode to prime the GPU allocator.

    The MLPerf reference runs ``warmup_validation_steps`` before real
    training so that the eval-time allocation pattern is already present
    in the allocator's block cache.  This prevents the first real eval
    from fragmenting the cache and causing a permanent step-time
    regression.

    The forward pass includes the loss computation path so that all
    eval-specific tensors (loss gather, metric buffers) are also
    pre-allocated in the cache.

    Controlled by ``LUMEN_WARMUP_EVAL_STEPS`` (default 0 = disabled).
    """
    n = _WARMUP_EVAL_STEPS
    if n <= 0:
        return

    print_rank_0(f"> Running {n} warmup eval forward passes to prime allocator ...")

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    unwrapped.eval()
    with torch.no_grad():
        for _ in range(n):
            tokens, labels, loss_mask, attention_mask, position_ids = (
                _get_synthetic_batch(args, zero_last_loss_mask=True)
            )
            output = unwrapped(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            if output is not None:
                losses = output.view(-1).float()
                lm = loss_mask.view(-1).float()
                _ = torch.sum(losses * lm)
                _ = lm.sum()
    unwrapped.train()

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_rank_0(f"> Warmup eval pass done ({n} steps). Allocator primed.")


# ---------------------------------------------------------------------------
# Loss function + early stopping
# ---------------------------------------------------------------------------

_val_loss_ema: float | None = None
_early_stop_logged = False


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """LM loss. Early stopping is handled collectively on the *reduced*
    validation loss by ``install_val_loss_early_stop_hook`` — NOT here. The
    previous per-step, per-rank training-loss EMA stop caused a DP desync
    (one rank crossed the threshold on its local loss and exited the train
    loop while others continued -> mismatched collectives -> NCCL deadlock)."""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
    return loss, num_tokens, {"lm loss": reporting}


# ---------------------------------------------------------------------------
# Forward step factory
# ---------------------------------------------------------------------------


def make_forward_step(
    get_batch_fn: Callable,
    loss_fn: Callable = loss_func,
    zero_last_loss_mask: bool = False,
):
    """Return a ``forward_step`` function suitable for :func:`megatron.training.pretrain`.

    Args:
        get_batch_fn: Model-specific batch constructor
            ``(data_iterator, vp_stage) -> (tokens, labels, loss_mask, attention_mask, position_ids)``.
        loss_fn: Loss function (defaults to :func:`loss_func`).
        zero_last_loss_mask: Forwarded to :func:`_get_synthetic_batch`.
    """

    def forward_step(data_iterator, model: GPTModel):
        global _warmup_step_counter, _warmup_completed

        args = get_args()
        timers = get_timers()
        warmup_steps = getattr(args, "warmup_steps", 0)

        timers("batch-generator", log_level=2).start()
        with stimer(bdata=True):
            if warmup_steps <= 0 and not _warmup_completed:
                _warmup_completed = True
                from atomrl.lumen.ops.quantize.linear import set_warmup_mode

                set_warmup_mode(False)
            if warmup_steps > 0 and not _warmup_completed:
                _warmup_step_counter += 1
                if _warmup_step_counter <= warmup_steps:
                    tokens, labels, loss_mask, attention_mask, position_ids = (
                        _get_synthetic_batch(
                            args, zero_last_loss_mask=zero_last_loss_mask
                        )
                    )
                    if data_iterator is not None:
                        try:
                            next(data_iterator)
                        except StopIteration:
                            pass
                else:
                    if getattr(args, "linear_fp8", False) or getattr(
                        args, "linear_fp4", False
                    ):
                        reset_fp8_state(model)
                    _run_warmup_eval_pass(model, args)
                    if getattr(args, "linear_fp8", False) or getattr(
                        args, "linear_fp4", False
                    ):
                        reset_fp8_state(model)
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    _warmup_completed = True
                    from atomrl.lumen.ops.quantize.linear import set_warmup_mode

                    set_warmup_mode(False)
                    print_rank_0(
                        f"> Synthetic warmup complete ({warmup_steps} steps). "
                        f"Resuming with real data."
                    )
                    vp_stage = get_attr_wrapped_model(model, "vp_stage")
                    tokens, labels, loss_mask, attention_mask, position_ids = (
                        get_batch_fn(data_iterator, vp_stage)
                    )
            else:
                vp_stage = get_attr_wrapped_model(model, "vp_stage")
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch_fn(
                    data_iterator, vp_stage
                )
        timers("batch-generator").stop()

        with stimer, get_cpu_offload_context(args):
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

        return output_tensor, partial(loss_fn, loss_mask, model=model)

    return forward_step


# ---------------------------------------------------------------------------
# Common CLI argument groups — see atomrl.lumen.patches.builders.megatron_args
# ---------------------------------------------------------------------------


# Backward-compatible re-exports (implementations live in dedicated modules).
from atomrl.lumen.models.fp8_param_storage import (
    _install_embedding_output_fp8_hooks,
    _patch_float16_module,
    _patch_load_checkpoint_for_fp8,
    _patch_meta_materializer,
    _shrink_frozen_weights_to_fp8,
    register_fp8_param_optimizer_hook,
)
from atomrl.lumen.patches.builders.megatron_args import add_common_megatron_args
from atomrl.lumen.patches.training.megatron_hooks import (
    install_fp8_param_gather_hook,
    install_fp8_param_storage_hook,
    install_hip_graphs_hook,
    install_mxfp4_weight_cache_hook,
    install_val_loss_early_stop_hook,
)

__all__ = [
    "_NORM_ATTRS",
    "_MegatronCompatibleTLLayerNorm",
    "_MegatronCompatibleTLNorm",
    "_MegatronCompatibleTLRMSNorm",
    "_install_embedding_output_fp8_hooks",
    "_patch_all_norms",
    "_patch_core_attention",
    "_patch_float16_module",
    "_patch_fused_swiglu_mlp",
    "_patch_layernorm",
    "_patch_load_checkpoint_for_fp8",
    "_patch_meta_materializer",
    "_patch_mla_attention",
    "_patch_norms_in_spec",
    "_patch_rmsnorm",
    "_shrink_frozen_weights_to_fp8",
    "add_common_megatron_args",
    "install_fp8_param_gather_hook",
    "install_fp8_param_storage_hook",
    "install_hip_graphs_hook",
    "install_mxfp4_weight_cache_hook",
    "install_val_loss_early_stop_hook",
    "register_fp8_param_optimizer_hook",
]
