# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Generic Megatron MODEL_BUILD registrations (GPT spec / post-build patches)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from atomrl.lumen.modules.attention_megatron import LumenDotProductAttention
from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA
from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp
from atomrl.lumen.patches.registry import PatchPhase, register_patch

GPT_LOCAL_SPEC_PATCHES = frozenset({"core_attention_spec", "norms_in_spec"})
GPT_LUMEN_SPEC_PATCHES = frozenset({"mla_attention_spec"})
GPT_LOCAL_MODEL_PATCHES = frozenset({"model_norms"})
GPT_LUMEN_MODEL_PATCHES = frozenset({"fused_swiglu_mlp"})


def patch_core_attention(spec) -> None:
    """Recursively replace ``core_attention`` with :class:`LumenDotProductAttention`."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(module=LumenDotProductAttention)
        if getattr(subs, "layer_specs", None):
            for layer_spec in subs.layer_specs:
                patch_core_attention(layer_spec)


class _MegatronCompatibleTLRMSNorm(torch.nn.Module):
    """Megatron-Core factory wrapper for :class:`LumenRMSNorm`."""

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        self._norm = LumenRMSNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class _MegatronCompatibleTLLayerNorm(torch.nn.Module):
    """Megatron-Core factory wrapper for :class:`LumenLayerNorm`."""

    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class _MegatronCompatibleTLNorm(torch.nn.Module):
    """Auto-detect RMSNorm vs LayerNorm from Megatron config."""

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        norm_type = getattr(config, "normalization", "LayerNorm")
        if norm_type == "RMSNorm":
            from atomrl.lumen.ops.normalization import LumenRMSNorm

            self._norm = LumenRMSNorm(hidden_size, eps=eps)
        else:
            from atomrl.lumen.ops.normalization import LumenLayerNorm

            self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


_NORM_ATTRS = (
    "input_layernorm",
    "pre_mlp_layernorm",
    "pre_cross_attn_layernorm",
    "post_cross_attn_layernorm",
    "final_layernorm",
)

# Per-head QK norms sit one level down, inside the attention submodule spec, and
# normalise over head_dim instead of hidden_size.
_ATTN_NORM_ATTRS = ("q_layernorm", "k_layernorm")


def patch_norms_in_spec(spec, norm_cls=None) -> None:
    """Replace norm classes in a spec tree with Lumen norm factories."""
    from megatron.core.transformer.identity_op import IdentityOp

    if norm_cls is None:
        norm_cls = _MegatronCompatibleTLNorm

    for attr in _NORM_ATTRS:
        cur = getattr(spec, attr, None)
        if cur is not None and cur is not IdentityOp:
            setattr(spec, attr, norm_cls)

    if hasattr(spec, "submodules") and spec.submodules is not None:
        for attr in _NORM_ATTRS:
            cur = getattr(spec.submodules, attr, None)
            if cur is not None and cur is not IdentityOp:
                setattr(spec.submodules, attr, norm_cls)

        for attn_attr in ("self_attention", "cross_attention"):
            attn_sub = getattr(
                getattr(spec.submodules, attn_attr, None), "submodules", None
            )
            if attn_sub is None:
                continue
            for attr in _ATTN_NORM_ATTRS:
                cur = getattr(attn_sub, attr, None)
                # L2Norm is a different normalisation, not an RMSNorm variant.
                if (
                    cur is None
                    or cur is IdentityOp
                    or getattr(cur, "__name__", "") == "L2Norm"
                ):
                    continue
                setattr(attn_sub, attr, norm_cls)

    layer_specs = getattr(spec, "layer_specs", None)
    if layer_specs is None and hasattr(spec, "submodules"):
        layer_specs = getattr(spec.submodules, "layer_specs", None)
    if layer_specs:
        for layer_spec in layer_specs:
            patch_norms_in_spec(layer_spec, norm_cls)


def patch_rmsnorm(model, grad_quant_type=None) -> None:
    """Replace Megatron RMSNorm modules with :class:`LumenRMSNorm`."""
    from megatron.training import print_rank_0

    from atomrl.lumen.ops.normalization import LumenRMSNorm

    count = 0
    for _name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "RMSNorm",
                "MegatronRMSNorm",
                "TENorm",
                "_MegatronCompatibleTLRMSNorm",
                "_MegatronCompatibleTLNorm",
            ):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-6))
                replacement = LumenRMSNorm(
                    hidden_size,
                    eps=eps,
                    grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                setattr(module, attr_name, replacement)
                count += 1

    print_rank_0(f"> Replaced {count} RMSNorm modules with LumenRMSNorm")


def patch_layernorm(model, grad_quant_type=None) -> None:
    """Replace Megatron LayerNorm modules with :class:`LumenLayerNorm`."""
    from megatron.training import print_rank_0

    from atomrl.lumen.ops.normalization import LumenLayerNorm

    count = 0
    for _name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "LayerNorm",
                "FusedLayerNorm",
                "WrappedTorchNorm",
                "_MegatronCompatibleTLLayerNorm",
                "_MegatronCompatibleTLNorm",
            ):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-5))
                replacement = LumenLayerNorm(
                    hidden_size,
                    eps=eps,
                    grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                if (
                    hasattr(child, "bias")
                    and child.bias is not None
                    and replacement.bias is not None
                ):
                    replacement.bias.data.copy_(child.bias.data)
                setattr(module, attr_name, replacement)
                count += 1

    print_rank_0(f"> Replaced {count} LayerNorm modules with LumenLayerNorm")


def patch_all_norms(model, normalization="RMSNorm", grad_quant_type=None) -> None:
    """Replace all norm modules with the appropriate Lumen implementation."""
    if normalization == "RMSNorm":
        patch_rmsnorm(model, grad_quant_type)
    else:
        patch_layernorm(model, grad_quant_type)


def patch_mla_attention(spec) -> None:
    """Replace ``core_attention`` with :class:`LumenDotProductAttentionMLA`."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(
                        module=LumenDotProductAttentionMLA
                    )
        if getattr(subs, "layer_specs", None):
            for layer_spec in subs.layer_specs:
                patch_mla_attention(layer_spec)


def patch_fused_swiglu_mlp(model) -> None:
    """Patch Megatron MLP forward to use AITER fused SwiGLU when available."""
    from megatron.core.transformer.mlp import MLP
    from megatron.training import print_rank_0

    patched = 0
    for module in model.modules():
        if not isinstance(module, MLP):
            continue
        if not getattr(module.config, "gated_linear_unit", False):
            continue
        if getattr(module.config, "add_bias_linear", False):
            continue
        if getattr(module.config, "tensor_model_parallel_size", 1) != 1:
            continue
        if getattr(module.config, "activation_func", None) is not F.silu:
            continue
        if getattr(module.config, "activation_func_clamp_value", None) is not None:
            continue
        if getattr(module.config, "glu_linear_offset", 0.0) != 0.0:
            continue
        if getattr(module.config, "activation_func_fp8_input_store", False):
            continue
        if getattr(module.config, "cpu_offloading", False) and getattr(
            module.config, "cpu_offloading_activations", False
        ):
            continue

        _orig_forward = module.forward

        def _make_fused_forward(mlp_module, orig_fwd):
            def _fused_forward(hidden_states, per_token_scale=None):
                if per_token_scale is not None:
                    return orig_fwd(hidden_states, per_token_scale=per_token_scale)
                w_fc1 = mlp_module.linear_fc1.weight
                w_fc2 = mlp_module.linear_fc2.weight

                orig_shape = hidden_states.shape
                x_2d = hidden_states.reshape(-1, orig_shape[-1]).contiguous()

                if x_2d.shape[0] > 64:
                    return orig_fwd(hidden_states, per_token_scale=per_token_scale)

                w_gate, w_up = w_fc1.chunk(2, dim=0)
                out = fused_gated_mlp(
                    x_2d,
                    w_up,
                    w_gate,
                    w_fc2,
                    activation="swiglu",
                )
                out = out.reshape(orig_shape[:-1] + (out.shape[-1],))
                return out, None

            return _fused_forward

        module.forward = _make_fused_forward(module, _orig_forward)
        patched += 1

    print_rank_0(f"Patched {patched} MLP modules with AITER fused SwiGLU forward")


def apply_core_attention_spec(*, spec=None, args=None, config=None, **kwargs) -> None:
    if spec is not None:
        patch_core_attention(spec)


def apply_norms_in_spec(*, spec=None, args=None, config=None, **kwargs) -> None:
    if spec is not None:
        patch_norms_in_spec(spec)


def apply_mla_attention_spec(*, spec=None, args=None, config=None, **kwargs) -> None:
    if spec is None or not getattr(args, "multi_latent_attention", False):
        return
    patch_mla_attention(spec)


def apply_model_norms(*, model=None, args=None, config=None, **kwargs) -> None:
    if model is None:
        return
    if not (
        getattr(args, "lumen_rmsnorm", False) or getattr(args, "lumen_norm", False)
    ):
        return
    grad_quant_type = getattr(args, "grad_quant_type", None)
    normalization = getattr(args, "normalization", "RMSNorm")
    patch_all_norms(model, normalization, grad_quant_type)


def apply_fused_swiglu_mlp(*, model=None, args=None, config=None, **kwargs) -> None:
    if model is None or not getattr(args, "lumen_fused_mlp", False):
        return
    patch_fused_swiglu_mlp(model)


def patch_lora_for_layernorm_linear(model) -> None:
    """Fix LoRA input for LumenLayerNormLinear base layers.

    When a LumenLayerNormLinear is wrapped by LoRA, the LoRA adapter's
    forward passes the *raw* (pre-norm) input to lora_a. But lora_a
    expects the *normalized* input — matching what a standalone
    ColumnParallelLinear would receive after a separate layernorm.

    This patch replaces the LoRA adapter forward to retrieve the cached
    normalized output from the base layer's forward (stored in thread-local
    by ``LumenLayerNormLinear.forward``), avoiding a redundant RMSNorm.
    Falls back to recomputing the norm if the cache is empty.
    """
    from megatron.core.transformer.lora_adapter import LoraAdapter
    from megatron.training import print_rank_0

    from atomrl.lumen.modules.layernorm_linear import (
        LumenLayerNormLinear,
        _pop_cached_ln_out,
    )

    patched = 0
    for module in model.modules():
        if not isinstance(module, LoraAdapter):
            continue
        base = module.base_layer
        if not isinstance(base, LumenLayerNormLinear):
            continue
        if module.lora_a is None:
            continue

        def _make_patched_forward(adapter, base_layer):
            def _patched_forward(input_tensor, *args, **kwargs):
                output = base_layer(input_tensor, *args, **kwargs)
                if adapter.lora_a is None:
                    return output

                normed_input = _pop_cached_ln_out()
                if normed_input is None:
                    normed_input = base_layer._norm(input_tensor)

                lora_a_out, _ = adapter.lora_a(normed_input)
                lora_b_out, _ = adapter.lora_b(lora_a_out)
                lora_drop_out = adapter.lora_dropout(lora_b_out)
                lora_out = adapter.lora_alpha * lora_drop_out

                if type(output) is torch.Tensor:
                    return output + lora_out

                out_tensor, bias = output
                return out_tensor + lora_out, bias

            return _patched_forward

        module.forward = _make_patched_forward(module, base)
        patched += 1

    if patched > 0:
        print_rank_0(
            f"> Patched {patched} LoRA adapters for LumenLayerNormLinear "
            f"(cached normalized input for lora_a)"
        )


def apply_lora_layernorm_linear(
    *, model=None, args=None, config=None, **kwargs
) -> None:
    if model is None:
        return
    patch_lora_for_layernorm_linear(model)


register_patch(
    "core_attention_spec",
    PatchPhase.MODEL_BUILD,
    description="Inject LumenDotProductAttention into local GPT layer spec",
    tags=frozenset({"megatron", "lumen", "builder", "spec"}),
    default=False,
)(apply_core_attention_spec)

register_patch(
    "norms_in_spec",
    PatchPhase.MODEL_BUILD,
    description="Inject Lumen norm factories into local GPT layer spec",
    tags=frozenset({"megatron", "lumen", "builder", "spec"}),
    default=False,
)(apply_norms_in_spec)

register_patch(
    "mla_attention_spec",
    PatchPhase.MODEL_BUILD,
    description="Inject LumenDotProductAttentionMLA when multi_latent_attention is set",
    tags=frozenset({"megatron", "lumen", "builder", "spec", "mla"}),
    default=False,
)(apply_mla_attention_spec)

register_patch(
    "model_norms",
    PatchPhase.MODEL_BUILD,
    description="Replace built RMSNorm/LayerNorm modules when --lumen-rmsnorm/--lumen-norm",
    tags=frozenset({"megatron", "lumen", "builder", "model"}),
    default=False,
)(apply_model_norms)

register_patch(
    "fused_swiglu_mlp",
    PatchPhase.MODEL_BUILD,
    description="AITER fused SwiGLU MLP forward when --lumen-fused-mlp",
    tags=frozenset({"megatron", "lumen", "builder", "model", "mlp"}),
    default=False,
)(apply_fused_swiglu_mlp)

register_patch(
    "lora_layernorm_linear",
    PatchPhase.MODEL_BUILD,
    description="LoRA forward fix for LumenLayerNormLinear base layers",
    tags=frozenset({"lora", "lumen", "megatron", "model"}),
    default=False,
)(apply_lora_layernorm_linear)

# Backward-compatible aliases for tests and legacy imports.
_patch_core_attention = patch_core_attention
_patch_norms_in_spec = patch_norms_in_spec
_patch_rmsnorm = patch_rmsnorm
_patch_layernorm = patch_layernorm
_patch_all_norms = patch_all_norms
_patch_mla_attention = patch_mla_attention
_patch_fused_swiglu_mlp = patch_fused_swiglu_mlp
_patch_lora_for_layernorm_linear = patch_lora_for_layernorm_linear
