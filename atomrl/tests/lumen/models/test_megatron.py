# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.models.megatron — Megatron-LM shared training components.

Covers:
  - _MegatronCompatibleTLRMSNorm: construction, forward, weight attribute
  - _MegatronCompatibleTLLayerNorm: construction, forward, weight attribute
  - _MegatronCompatibleTLNorm: auto-dispatch based on config.normalization
  - _patch_core_attention: recursive spec patching
  - _patch_norms_in_spec: replace norms in spec tree
  - _patch_rmsnorm / _patch_layernorm / _patch_all_norms: model-level replacement
  - _patch_mla_attention: MLA spec patching
  - _override_te_args_for_lumen: TE arg overrides and FP8 format mapping
  - _get_synthetic_batch: synthetic warmup batch generation
  - reset_fp8_state: FP8 state reset on Megatron model
  - loss_func: loss computation with early stopping
  - add_common_megatron_args: all argument groups and defaults
  - apply_fp8_training: QuantConfig construction and quant.enable dispatch
  - enable_fp8_for_parallel_linear: FP8 enablement on parallel linear modules
"""

import argparse
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from atomrl.lumen.models.megatron import (
    _FP8_FORMAT_MAP,
    _NORM_ATTRS,
    _TE_FORCE_OVERRIDES,
    _get_synthetic_batch,
    _MegatronCompatibleTLLayerNorm,
    _MegatronCompatibleTLNorm,
    _MegatronCompatibleTLRMSNorm,
    _override_te_args_for_lumen,
    _patch_all_norms,
    _patch_core_attention,
    _patch_mla_attention,
    _patch_norms_in_spec,
    add_common_megatron_args,
    enable_fp4_for_parallel_linear,
    enable_fp8_for_parallel_linear,
    reset_fp8_state,
)
from conftest import compute_snr as _compute_snr
from conftest import layernorm_ref as _layernorm_golden
from conftest import rmsnorm_ref as _rmsnorm_golden

# ===================================================================
# Megatron-compatible norm wrappers
# ===================================================================


class _FakeMegatronConfig:
    """Mimics a Megatron TransformerConfig with a normalization attribute."""

    def __init__(self, normalization="RMSNorm"):
        self.normalization = normalization


class TestMegatronCompatibleTLRMSNorm:
    def test_construction(self):
        config = _FakeMegatronConfig("RMSNorm")
        norm = _MegatronCompatibleTLRMSNorm(config, hidden_size=64, eps=1e-5)
        assert hasattr(norm, "weight")
        assert norm.weight.shape == (64,)

    def test_forward_matches_golden(self):
        """RMSNorm wrapper output should match pure-PyTorch golden."""
        config = _FakeMegatronConfig("RMSNorm")
        eps = 1e-6
        norm = _MegatronCompatibleTLRMSNorm(config, hidden_size=64, eps=eps).cuda()
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = norm(x)

        golden = _rmsnorm_golden(x, norm.weight.data, eps=eps)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"RMSNorm wrapper vs golden SNR: {snr:.1f} dB"

    def test_forward_not_identity(self):
        config = _FakeMegatronConfig("RMSNorm")
        norm = _MegatronCompatibleTLRMSNorm(config, hidden_size=32).cuda()
        x = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16) * 10.0
        out = norm(x)
        assert not torch.equal(out, x)


class TestMegatronCompatibleTLLayerNorm:
    def test_construction(self):
        config = _FakeMegatronConfig("LayerNorm")
        norm = _MegatronCompatibleTLLayerNorm(config, hidden_size=64, eps=1e-5)
        assert hasattr(norm, "weight")
        assert norm.weight.shape == (64,)

    def test_forward_matches_golden(self):
        """LayerNorm wrapper output should match pure-PyTorch golden."""
        config = _FakeMegatronConfig("LayerNorm")
        eps = 1e-5
        norm = (
            _MegatronCompatibleTLLayerNorm(config, hidden_size=64, eps=eps)
            .cuda()
            .bfloat16()
        )
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = norm(x)

        bias = norm._norm.bias.data if norm._norm.bias is not None else None
        golden = _layernorm_golden(x, norm.weight.data, bias=bias, eps=eps)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"LayerNorm wrapper vs golden SNR: {snr:.1f} dB"

    def test_forward_nonzero_bias_matches_golden(self):
        """LayerNorm wrapper output should be correct with non-zero bias."""
        config = _FakeMegatronConfig("LayerNorm")
        eps = 1e-5
        norm = (
            _MegatronCompatibleTLLayerNorm(config, hidden_size=64, eps=eps)
            .cuda()
            .bfloat16()
        )
        norm._norm.bias.data.uniform_(-0.5, 0.5)
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = norm(x)

        golden = _layernorm_golden(
            x, norm.weight.data, bias=norm._norm.bias.data, eps=eps
        )
        snr = _compute_snr(golden, out)
        assert (
            snr > 30
        ), f"LayerNorm wrapper (non-zero bias) vs golden SNR: {snr:.1f} dB"


class TestMegatronCompatibleTLNorm:
    def test_dispatches_rmsnorm(self):
        config = _FakeMegatronConfig("RMSNorm")
        norm = _MegatronCompatibleTLNorm(config, hidden_size=64)
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        assert isinstance(norm._norm, LumenRMSNorm)

    def test_dispatches_layernorm(self):
        config = _FakeMegatronConfig("LayerNorm")
        norm = _MegatronCompatibleTLNorm(config, hidden_size=64)
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        assert isinstance(norm._norm, LumenLayerNorm)

    def test_default_is_layernorm(self):
        config = SimpleNamespace()
        norm = _MegatronCompatibleTLNorm(config, hidden_size=64)
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        assert isinstance(norm._norm, LumenLayerNorm)

    def test_rmsnorm_output_matches_golden(self):
        """Auto-dispatch RMSNorm output should match golden."""
        config = _FakeMegatronConfig("RMSNorm")
        eps = 1e-6
        norm = _MegatronCompatibleTLNorm(config, hidden_size=128, eps=eps).cuda()
        torch.manual_seed(1)
        x = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
        out = norm(x)

        golden = _rmsnorm_golden(x, norm.weight.data, eps=eps)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Auto-dispatch RMSNorm vs golden SNR: {snr:.1f} dB"

    def test_layernorm_output_matches_golden(self):
        """Auto-dispatch LayerNorm output should match golden."""
        config = _FakeMegatronConfig("LayerNorm")
        eps = 1e-5
        norm = (
            _MegatronCompatibleTLNorm(config, hidden_size=128, eps=eps)
            .cuda()
            .bfloat16()
        )
        torch.manual_seed(1)
        x = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
        out = norm(x)

        bias = norm._norm.bias.data if norm._norm.bias is not None else None
        golden = _layernorm_golden(x, norm.weight.data, bias=bias, eps=eps)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Auto-dispatch LayerNorm vs golden SNR: {snr:.1f} dB"


# ===================================================================
# _patch_core_attention (spec patching)
# ===================================================================


class _FakeSubmodules:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeModuleSpec:
    def __init__(self, submodules=None, module=None):
        self.submodules = submodules
        self.module = module


class TestPatchCoreAttention:
    def test_patches_core_attention(self):
        from atomrl.lumen.modules.attention_megatron import LumenDotProductAttention

        sa_subs = _FakeSubmodules(core_attention=_FakeModuleSpec(module=nn.Identity))
        sa_spec = _FakeModuleSpec(submodules=sa_subs)
        top_subs = _FakeSubmodules(self_attention=sa_spec)
        spec = _FakeModuleSpec(submodules=top_subs)

        _patch_core_attention(spec)
        assert sa_subs.core_attention.module is LumenDotProductAttention

    def test_no_op_when_no_self_attention(self):
        top_subs = _FakeSubmodules()
        spec = _FakeModuleSpec(submodules=top_subs)
        _patch_core_attention(spec)

    def test_recursive_layer_specs(self):
        from atomrl.lumen.modules.attention_megatron import LumenDotProductAttention

        sa_subs1 = _FakeSubmodules(core_attention=_FakeModuleSpec(module=nn.Identity))
        sa_spec1 = _FakeModuleSpec(submodules=sa_subs1)
        layer1_subs = _FakeSubmodules(self_attention=sa_spec1)
        layer1 = _FakeModuleSpec(submodules=layer1_subs)

        sa_subs2 = _FakeSubmodules(core_attention=_FakeModuleSpec(module=nn.Identity))
        sa_spec2 = _FakeModuleSpec(submodules=sa_subs2)
        layer2_subs = _FakeSubmodules(self_attention=sa_spec2)
        layer2 = _FakeModuleSpec(submodules=layer2_subs)

        block_subs = _FakeSubmodules(layer_specs=[layer1, layer2])
        block = _FakeModuleSpec(submodules=block_subs)

        _patch_core_attention(block)
        assert sa_subs1.core_attention.module is LumenDotProductAttention
        assert sa_subs2.core_attention.module is LumenDotProductAttention


# ===================================================================
# _patch_mla_attention
# ===================================================================


class TestPatchMLAAttention:
    def test_patches_mla(self):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        sa_subs = _FakeSubmodules(core_attention=_FakeModuleSpec(module=nn.Identity))
        sa_spec = _FakeModuleSpec(submodules=sa_subs)
        top_subs = _FakeSubmodules(self_attention=sa_spec)
        spec = _FakeModuleSpec(submodules=top_subs)

        _patch_mla_attention(spec)
        assert sa_subs.core_attention.module is LumenDotProductAttentionMLA


# ===================================================================
# _patch_norms_in_spec
# ===================================================================


class TestPatchNormsInSpec:
    def test_patches_top_level_attrs(self):
        spec = _FakeSubmodules(
            input_layernorm=nn.LayerNorm,
            pre_mlp_layernorm=nn.LayerNorm,
            final_layernorm=nn.LayerNorm,
        )
        _patch_norms_in_spec(spec)
        assert spec.input_layernorm is _MegatronCompatibleTLNorm
        assert spec.pre_mlp_layernorm is _MegatronCompatibleTLNorm
        assert spec.final_layernorm is _MegatronCompatibleTLNorm

    def test_patches_submodule_attrs(self):
        sub = _FakeSubmodules(
            input_layernorm=nn.LayerNorm, pre_mlp_layernorm=nn.LayerNorm
        )
        spec = _FakeModuleSpec(submodules=sub)
        _patch_norms_in_spec(spec)
        assert sub.input_layernorm is _MegatronCompatibleTLNorm
        assert sub.pre_mlp_layernorm is _MegatronCompatibleTLNorm

    def test_custom_norm_cls(self):
        spec = _FakeSubmodules(input_layernorm=nn.LayerNorm)
        _patch_norms_in_spec(spec, norm_cls=_MegatronCompatibleTLRMSNorm)
        assert spec.input_layernorm is _MegatronCompatibleTLRMSNorm

    def test_recursive_through_layer_specs(self):
        layer_sub = _FakeSubmodules(input_layernorm=nn.LayerNorm)
        layer = _FakeModuleSpec(submodules=layer_sub)
        block = _FakeSubmodules(layer_specs=[layer])
        _patch_norms_in_spec(block)
        assert layer_sub.input_layernorm is _MegatronCompatibleTLNorm

    def test_patches_cross_attn_norm_attrs(self):
        """All five _NORM_ATTRS should be patched, including cross-attention ones."""
        spec = _FakeSubmodules(
            input_layernorm=nn.LayerNorm,
            pre_mlp_layernorm=nn.LayerNorm,
            pre_cross_attn_layernorm=nn.LayerNorm,
            post_cross_attn_layernorm=nn.LayerNorm,
            final_layernorm=nn.LayerNorm,
        )
        _patch_norms_in_spec(spec)
        for attr in _NORM_ATTRS:
            assert (
                getattr(spec, attr) is _MegatronCompatibleTLNorm
            ), f"{attr} not patched"


# ===================================================================
# _patch_rmsnorm / _patch_layernorm / _patch_all_norms
# ===================================================================


class _FakeRMSNorm(nn.Module):
    __qualname__ = "RMSNorm"

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x


_FakeRMSNorm.__name__ = "RMSNorm"


class _FakeLayerNorm(nn.Module):
    __qualname__ = "LayerNorm"

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x


_FakeLayerNorm.__name__ = "LayerNorm"


class TestPatchRMSNorm:
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_replaces_rmsnorm(self, mock_print):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        model = nn.Module()
        model.norm = _FakeRMSNorm(64)
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        _patch_rmsnorm(model)
        assert isinstance(model.norm, LumenRMSNorm)
        assert model.norm.weight.shape[0] == 64

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_copies_weight(self, mock_print):
        model = nn.Module()
        norm = _FakeRMSNorm(16)
        norm.weight.data.fill_(2.5)
        model.norm = norm
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        _patch_rmsnorm(model)
        assert torch.allclose(model.norm.weight.data, torch.full((16,), 2.5))

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_nested_replacement(self, mock_print):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        inner = nn.Module()
        inner.norm = _FakeRMSNorm(32)
        model = nn.Module()
        model.layer = inner
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        _patch_rmsnorm(model)
        assert isinstance(model.layer.norm, LumenRMSNorm)

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_grad_quant_type_forwarded(self, mock_print):
        model = nn.Module()
        model.norm = _FakeRMSNorm(32)
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        _patch_rmsnorm(model, grad_quant_type="fp8")
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        assert isinstance(model.norm, LumenRMSNorm)
        assert model.norm.grad_quant_type == "fp8"

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_epsilon_attribute_fallback(self, mock_print):
        """Norms with ``epsilon`` instead of ``eps`` should be patched correctly."""
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        model = nn.Module()
        custom_eps = 1e-4
        model.norm = _FakeRMSNormWithEpsilon(64, epsilon=custom_eps)
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        _patch_rmsnorm(model)
        assert isinstance(model.norm, LumenRMSNorm)
        assert model.norm.eps == custom_eps

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    @pytest.mark.parametrize("cls_name", ["RMSNorm", "MegatronRMSNorm", "TENorm"])
    def test_replaces_by_class_name(self, mock_print, cls_name):
        """_patch_rmsnorm should match all expected class names."""
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        class _DynRMSNorm(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(h))
                self.eps = 1e-6

            def forward(self, x):
                return x

        _DynRMSNorm.__name__ = cls_name
        _DynRMSNorm.__qualname__ = cls_name

        model = nn.Module()
        model.norm = _DynRMSNorm(32)
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        _patch_rmsnorm(model)
        assert isinstance(
            model.norm, LumenRMSNorm
        ), f"Expected replacement for {cls_name}"

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_patched_rmsnorm_forward_matches_golden(self, mock_print):
        from atomrl.lumen.models.megatron import _patch_rmsnorm

        model = nn.Module()
        model.norm = _FakeRMSNorm(64)
        model.norm.weight.data.uniform_(0.5, 1.5)
        _patch_rmsnorm(model)

        model.norm = model.norm.cuda()
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        golden = _rmsnorm_golden(x, model.norm.weight.data)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Patched RMSNorm vs golden SNR: {snr:.1f} dB"


class _FakeRMSNormWithEpsilon(nn.Module):
    """Uses ``epsilon`` instead of ``eps`` — matches some Megatron-Core norms."""

    __qualname__ = "MegatronRMSNorm"

    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, x):
        return x


_FakeRMSNormWithEpsilon.__name__ = "MegatronRMSNorm"


class _FakeLayerNormWithEpsilon(nn.Module):
    """Uses ``epsilon`` instead of ``eps`` — matches WrappedTorchNorm."""

    __qualname__ = "WrappedTorchNorm"

    def __init__(self, hidden_size, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.epsilon = epsilon

    def forward(self, x):
        return x


_FakeLayerNormWithEpsilon.__name__ = "WrappedTorchNorm"


class TestPatchLayerNorm:
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_replaces_layernorm(self, mock_print):
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        model = nn.Module()
        model.norm = _FakeLayerNorm(64)
        from atomrl.lumen.models.megatron import _patch_layernorm

        _patch_layernorm(model)
        assert isinstance(model.norm, LumenLayerNorm)

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_copies_weight_and_bias(self, mock_print):
        model = nn.Module()
        norm = _FakeLayerNorm(16)
        norm.weight.data.fill_(2.0)
        norm.bias.data.fill_(0.5)
        model.norm = norm
        from atomrl.lumen.models.megatron import _patch_layernorm

        _patch_layernorm(model)
        assert torch.allclose(model.norm.weight.data, torch.full((16,), 2.0))
        if model.norm.bias is not None:
            assert torch.allclose(model.norm.bias.data, torch.full((16,), 0.5))

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_epsilon_attribute_fallback(self, mock_print):
        """Norms with ``epsilon`` instead of ``eps`` should be patched correctly."""
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        model = nn.Module()
        custom_eps = 1e-3
        model.norm = _FakeLayerNormWithEpsilon(64, epsilon=custom_eps)
        from atomrl.lumen.models.megatron import _patch_layernorm

        _patch_layernorm(model)
        assert isinstance(model.norm, LumenLayerNorm)
        assert model.norm.eps == custom_eps

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    @pytest.mark.parametrize(
        "cls_name", ["LayerNorm", "FusedLayerNorm", "WrappedTorchNorm"]
    )
    def test_replaces_by_class_name(self, mock_print, cls_name):
        """_patch_layernorm should match all expected class names."""
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        class _DynLayerNorm(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(h))
                self.bias = nn.Parameter(torch.zeros(h))
                self.eps = 1e-5

            def forward(self, x):
                return x

        _DynLayerNorm.__name__ = cls_name
        _DynLayerNorm.__qualname__ = cls_name

        model = nn.Module()
        model.norm = _DynLayerNorm(32)
        from atomrl.lumen.models.megatron import _patch_layernorm

        _patch_layernorm(model)
        assert isinstance(
            model.norm, LumenLayerNorm
        ), f"Expected replacement for {cls_name}"

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_patched_layernorm_forward_matches_golden(self, mock_print):
        from atomrl.lumen.models.megatron import _patch_layernorm

        model = nn.Module()
        model.norm = _FakeLayerNorm(64)
        model.norm.weight.data.uniform_(0.5, 1.5)
        model.norm.bias.data.uniform_(-0.1, 0.1)
        _patch_layernorm(model)

        model.norm = model.norm.cuda().bfloat16()
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        bias = (
            model.norm.bias.data
            if hasattr(model.norm, "bias") and model.norm.bias is not None
            else None
        )
        golden = _layernorm_golden(x, model.norm.weight.data, bias=bias)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Patched LayerNorm vs golden SNR: {snr:.1f} dB"


class TestPatchAllNorms:
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_dispatch_rmsnorm(self, mock_print):
        model = nn.Module()
        model.norm = _FakeRMSNorm(32)
        _patch_all_norms(model, normalization="RMSNorm")
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        assert isinstance(model.norm, LumenRMSNorm)

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_dispatch_layernorm(self, mock_print):
        model = nn.Module()
        model.norm = _FakeLayerNorm(32)
        _patch_all_norms(model, normalization="LayerNorm")
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        assert isinstance(model.norm, LumenLayerNorm)


# ===================================================================
# _override_te_args_for_lumen
# ===================================================================


class TestOverrideTEArgs:
    def test_sets_fp8_to_none(self):
        args = SimpleNamespace(fp8="e4m3")
        _override_te_args_for_lumen(args)
        assert args.fp8 is None

    def test_maps_fp8_format(self):
        args = SimpleNamespace(fp8="e4m3")
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_format == "fp8_e4m3"

    def test_maps_hybrid_format(self):
        args = SimpleNamespace(fp8="hybrid")
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_format == "hybrid"

    def test_unknown_format_passthrough(self):
        args = SimpleNamespace(fp8="mxfp8_custom")
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_format == "mxfp8_custom"

    def test_force_overrides_applied(self):
        args = SimpleNamespace(fp8=None)
        _override_te_args_for_lumen(args)
        for attr, value in _TE_FORCE_OVERRIDES.items():
            assert getattr(args, attr) == value

    def test_transformer_impl_set_to_local(self):
        args = SimpleNamespace(fp8=None, transformer_impl="transformer_engine")
        _override_te_args_for_lumen(args)
        assert args.transformer_impl == "local"

    def test_lumen_cross_entropy_triggers_patch(self):
        args = SimpleNamespace(fp8=None, lumen_cross_entropy=True)
        with mock.patch(
            "atomrl.lumen.models.megatron._patch_cross_entropy"
        ) as mock_patch:
            _override_te_args_for_lumen(args)
            mock_patch.assert_called_once()

    def test_no_cross_entropy_patch_when_disabled(self):
        args = SimpleNamespace(fp8=None, lumen_cross_entropy=False)
        with mock.patch(
            "atomrl.lumen.models.megatron._patch_cross_entropy"
        ) as mock_patch:
            _override_te_args_for_lumen(args)
            mock_patch.assert_not_called()

    def test_no_fp8_attr_handled(self):
        args = SimpleNamespace(fp8=None)
        _override_te_args_for_lumen(args)
        assert args.fp8 is None
        assert not hasattr(args, "lumen_fp8_format")

    def test_fp8_attn_dpa_resolves_backend(self):
        args = SimpleNamespace(
            fp8=None, lumen_fp8_attn="dpa", lumen_attn_backend="auto"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_attn == "dpa"
        assert args.lumen_attn_backend == "aiter_triton_fp8"

    def test_fp8_attn_mha_resolves_backend(self):
        args = SimpleNamespace(
            fp8=None, lumen_fp8_attn="mha", lumen_attn_backend="auto"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_attn == "mha"
        assert args.lumen_attn_backend == "aiter_triton_fp8"

    def test_fp8_attn_none_keeps_bf16_backend(self):
        args = SimpleNamespace(
            fp8=None, lumen_fp8_attn="none", lumen_attn_backend="auto"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_attn_backend == "aiter_csrc"

    def test_triton_backend_with_dpa(self):
        args = SimpleNamespace(
            fp8=None, lumen_fp8_attn="dpa", lumen_attn_backend="triton"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_attn_backend == "aiter_triton_fp8"

    def test_csrc_backend_with_dpa(self):
        args = SimpleNamespace(
            fp8=None, lumen_fp8_attn="dpa", lumen_attn_backend="csrc"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_attn_backend == "aiter_csrc_fp8"

    def test_asm_backend_with_dpa(self):
        args = SimpleNamespace(fp8=None, lumen_fp8_attn="dpa", lumen_attn_backend="asm")
        _override_te_args_for_lumen(args)
        assert args.lumen_attn_backend == "aiter_asm_fp8"

    def test_megatron_fp8_dpa_promotes_to_dpa(self):
        args = SimpleNamespace(
            fp8=None, fp8_dot_product_attention=True, lumen_attn_backend="auto"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_attn == "dpa"
        assert args.lumen_attn_backend == "aiter_triton_fp8"

    def test_megatron_fp8_mha_promotes_to_mha(self):
        args = SimpleNamespace(
            fp8=None, fp8_multi_head_attention=True, lumen_attn_backend="auto"
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_attn == "mha"
        assert args.lumen_attn_backend == "aiter_triton_fp8"

    def test_megatron_both_fp8_flags_mha_wins(self):
        args = SimpleNamespace(
            fp8=None,
            fp8_dot_product_attention=True,
            fp8_multi_head_attention=True,
            lumen_attn_backend="auto",
        )
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_attn == "mha"
        assert args.lumen_attn_backend == "aiter_triton_fp8"

    def test_no_fp8_attn_leaves_backend_bf16(self):
        args = SimpleNamespace(fp8=None, lumen_attn_backend="triton")
        _override_te_args_for_lumen(args)
        assert args.lumen_fp8_attn == "none"
        assert args.lumen_attn_backend == "aiter_triton"


# ===================================================================
# _get_synthetic_batch
# ===================================================================


class TestGetSyntheticBatch:
    def test_shapes(self):
        args = SimpleNamespace(seq_length=128, micro_batch_size=2)
        tokens, labels, loss_mask, attention_mask, position_ids = _get_synthetic_batch(
            args
        )
        assert tokens.shape == (2, 128)
        assert labels.shape == (2, 128)
        assert loss_mask.shape == (2, 128)
        assert attention_mask.shape == (2, 1, 128, 128)
        assert position_ids.shape == (2, 128)

    def test_dtypes(self):
        args = SimpleNamespace(seq_length=64, micro_batch_size=1)
        tokens, labels, loss_mask, attention_mask, position_ids = _get_synthetic_batch(
            args
        )
        assert tokens.dtype == torch.long
        assert labels.dtype == torch.long
        assert loss_mask.dtype == torch.float
        assert attention_mask.dtype == torch.bool
        assert position_ids.dtype == torch.long

    def test_on_cuda(self):
        args = SimpleNamespace(seq_length=32, micro_batch_size=1)
        tokens, labels, _loss_mask, _attention_mask, _position_ids = (
            _get_synthetic_batch(args)
        )
        assert tokens.device.type == "cuda"
        assert labels.device.type == "cuda"

    def test_tokens_have_eos(self):
        args = SimpleNamespace(seq_length=16, micro_batch_size=2)
        tokens, _labels, _loss_mask, _attention_mask, _position_ids = (
            _get_synthetic_batch(args)
        )
        assert (tokens[:, -1] == 2).all()

    def test_labels_match_tokens(self):
        args = SimpleNamespace(seq_length=16, micro_batch_size=1)
        tokens, labels, _loss_mask, _attention_mask, _position_ids = (
            _get_synthetic_batch(args)
        )
        assert torch.equal(tokens, labels)

    def test_loss_mask_is_all_zeros(self):
        args = SimpleNamespace(seq_length=16, micro_batch_size=1)
        _tokens, _labels, loss_mask, _attention_mask, _position_ids = (
            _get_synthetic_batch(args)
        )
        assert (loss_mask == 0.0).all()

    def test_zero_last_loss_mask_flag_zeros_entire_mask(self):
        args = SimpleNamespace(seq_length=16, micro_batch_size=2)
        _tokens, _labels, loss_mask, _attention_mask, _position_ids = (
            _get_synthetic_batch(args, zero_last_loss_mask=True)
        )
        assert (loss_mask == 0.0).all()

    def test_attention_mask_all_true(self):
        args = SimpleNamespace(seq_length=8, micro_batch_size=1)
        _tokens, _labels, _loss_mask, attention_mask, _position_ids = (
            _get_synthetic_batch(args)
        )
        assert attention_mask.all()

    def test_position_ids_sequential(self):
        args = SimpleNamespace(seq_length=16, micro_batch_size=2)
        _tokens, _labels, _loss_mask, _attention_mask, position_ids = (
            _get_synthetic_batch(args)
        )
        expected = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)
        assert torch.equal(position_ids, expected)


# ===================================================================
# reset_fp8_state (Megatron version)
# ===================================================================


class TestResetFP8StateMegatron:
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_resets_fp8_initialized(self, mock_print):
        module = nn.Module()
        module.fp8_initialized = True
        model = nn.Sequential(module)
        reset_fp8_state(model)
        assert module.fp8_initialized is False

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_resets_quant_manager(self, mock_print):
        module = nn.Module()
        module._quant_manager = mock.MagicMock()
        model = nn.Sequential(module)
        reset_fp8_state(model)
        module._quant_manager.reset.assert_called_once()

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_resets_tl_scaling_manager(self, mock_print):
        module = nn.Module()
        module._tl_scaling_manager = mock.MagicMock()
        model = nn.Sequential(module)
        reset_fp8_state(model)
        module._tl_scaling_manager.reset.assert_called_once()

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_unwraps_ddp(self, mock_print):
        inner = nn.Module()
        inner.fp8_initialized = True
        wrapper = nn.Module()
        wrapper.module = inner
        reset_fp8_state(wrapper)
        assert inner.fp8_initialized is False


# ===================================================================
# loss_func
# ===================================================================


class TestLossFunc:
    @mock.patch("atomrl.lumen.models.megatron.get_args")
    def test_computes_loss(self, mock_get_args):
        mock_get_args.return_value = SimpleNamespace(val_loss_target=None)
        from atomrl.lumen.models.megatron import loss_func

        output_tensor = torch.tensor([0.5, 1.0, 0.2], device="cuda")
        loss_mask = torch.tensor([1.0, 1.0, 0.0], device="cuda")

        loss, num_tokens, report = loss_func(loss_mask, output_tensor)
        expected_loss = 0.5 * 1.0 + 1.0 * 1.0 + 0.2 * 0.0
        assert torch.isclose(
            loss, torch.tensor(expected_loss, device="cuda"), atol=1e-5
        )
        assert num_tokens.item() == 2
        assert "lm loss" in report
        report_tensor = report["lm loss"]
        assert torch.isclose(report_tensor[0], loss, atol=1e-6)
        assert report_tensor[1].item() == num_tokens.item()

    @mock.patch("atomrl.lumen.models.megatron.get_args")
    def test_all_masked(self, mock_get_args):
        mock_get_args.return_value = SimpleNamespace(val_loss_target=None)
        from atomrl.lumen.models.megatron import loss_func

        output_tensor = torch.tensor([0.5, 1.0], device="cuda")
        loss_mask = torch.tensor([0.0, 0.0], device="cuda")

        loss, num_tokens, report = loss_func(loss_mask, output_tensor)
        assert loss.item() == 0.0
        assert num_tokens.item() == 0
        report_tensor = report["lm loss"]
        assert report_tensor[0].item() == 0.0
        assert report_tensor[1].item() == 0

    @mock.patch("atomrl.lumen.models.megatron.get_args")
    def test_loss_gradients_flow(self, mock_get_args):
        mock_get_args.return_value = SimpleNamespace(val_loss_target=None)
        from atomrl.lumen.models.megatron import loss_func

        output_tensor = torch.tensor([0.5, 1.0, 0.2], device="cuda", requires_grad=True)
        loss_mask = torch.tensor([1.0, 1.0, 0.0], device="cuda")

        loss, _num_tokens, _report = loss_func(loss_mask, output_tensor)
        loss.backward()

        assert output_tensor.grad is not None
        assert torch.isfinite(output_tensor.grad).all()
        assert output_tensor.grad[0].item() != 0.0
        assert output_tensor.grad[1].item() != 0.0
        assert output_tensor.grad[2].item() == 0.0


# ===================================================================
# add_common_megatron_args
# ===================================================================


class TestAddCommonMegatronArgs:
    def _parse(self, cli_args=None):
        parser = argparse.ArgumentParser()
        add_common_megatron_args(parser)
        return parser.parse_args(cli_args or [])

    def test_backend_default(self):
        args = self._parse()
        assert args.backend == "megatron"

    def test_lumen_attn_backend_default(self):
        args = self._parse()
        assert args.lumen_attn_backend == "auto"

    def test_linear_fp4_has_dedicated_switch(self):
        args = self._parse(["--linear-fp4"])
        assert args.linear_fp4 is True
        assert args.linear_fp8 is False

    def test_fp8_format_rejects_mxfp4(self):
        with pytest.raises(SystemExit):
            self._parse(["--linear-fp8-format", "mxfp4"])

    def test_lumen_attn_backend_choices(self):
        for backend in ["auto", "triton", "csrc", "asm"]:
            args = self._parse(["--lumen-attn-backend", backend])
            assert args.lumen_attn_backend == backend

    def test_lumen_fp8_attn_default(self):
        args = self._parse()
        assert args.lumen_fp8_attn == "none"

    def test_lumen_fp8_attn_choices(self):
        for scope in ["none", "dpa", "mha"]:
            args = self._parse(["--lumen-fp8-attn", scope])
            assert args.lumen_fp8_attn == scope

    def test_grad_quant_type_choices(self):
        # Kept in step with the FSDP parser's --grad-quant-type: a recipe that
        # reaches only one of the two backends is the harder bug to spot.
        for gq in ["fp8", "mxfp8", "mxfp4"]:
            args = self._parse(["--grad-quant-type", gq])
            assert args.grad_quant_type == gq

    def test_grad_quant_type_still_accepts_the_retired_fp4_spelling(self):
        """Scripts written against "fp4" must not die at argparse.

        "fp4" has never computed anything -- ScalingManager raises
        NotImplementedError for it -- so dropping it from choices broke no
        working configuration, only moved the failure earlier and made it less
        legible. Keeping it means the error still arrives from the place that
        can name mxfp4 as the replacement. It is deliberately *not* aliased to
        mxfp4, which would quietly start quantizing gradients to a format the
        flag does not name.
        """
        from atomrl.lumen.quantize.scaling_manager import ScalingManager

        args = self._parse(["--grad-quant-type", "fp4"])
        assert args.grad_quant_type == "fp4"

        with pytest.raises(NotImplementedError, match="mxfp4"):
            ScalingManager.quantize_grad_tensor(
                torch.zeros(4, 4),
                grad_quant_type="fp4",
                block_size=32,
            )

    def test_linear_fp8_and_fp4_are_exclusive_at_the_parser(self):
        """Both gates at once has to fail before a model is built.

        The exclusion lived only in LumenConfig.from_args, which the FSDP and
        RL call sites bypass by reading args.linear_fp4 off the namespace.
        """
        assert self._parse(["--linear-fp4"]).linear_fp4 is True
        assert self._parse(["--linear-fp8"]).linear_fp8 is True
        with pytest.raises(SystemExit):
            self._parse(["--linear-fp8", "--linear-fp4"])

    def test_lumen_fp8_quant_type_default(self):
        args = self._parse()
        assert args.lumen_fp8_quant_type == "blockwise"

    def test_lumen_fp8_quant_type_choices(self):
        for qt in ["dynamic", "delayed", "blockwise", "per_token", "none", "mxfp8"]:
            args = self._parse(["--lumen-fp8-quant-type", qt])
            assert args.lumen_fp8_quant_type == qt

    def test_lumen_rmsnorm_default(self):
        args = self._parse()
        assert args.lumen_rmsnorm is False

    def test_lumen_norm_default(self):
        args = self._parse()
        assert args.lumen_norm is False

    def test_lumen_linear_default(self):
        args = self._parse()
        assert args.lumen_linear is False

    def test_lumen_cross_entropy_default(self):
        args = self._parse()
        assert args.lumen_cross_entropy is False

    def test_mxfp8_block_defaults(self):
        args = self._parse()
        assert args.mxfp8_block_m_fwd == 128
        assert args.mxfp8_block_n_fwd == 128
        assert args.mxfp8_block_m_dq_bwd == 128
        assert args.mxfp8_block_n_dq_bwd == 128
        assert args.mxfp8_block_m_dkv_bwd == 128
        assert args.mxfp8_block_n_dkv_bwd == 128
        assert args.mxfp8_quant_block_size == 128

    def test_lora_defaults(self):
        args = self._parse()
        assert args.lora_rank == 0
        assert args.lora_alpha == 32.0
        assert args.lora_dropout == 0.1
        assert args.lora_a2a is False

    def test_linear_fp8_defaults(self):
        args = self._parse()
        assert args.linear_fp8 is False
        assert args.linear_fp8_scaling == "delayed"
        assert args.linear_fp8_block_size == 128
        assert args.linear_fp8_amax_algo == "max"
        assert args.linear_fp8_reduce_amax is False
        assert args.use_sdma is False
        assert args.linear_fp8_amax_history == 16
        assert args.linear_fp8_margin == 0
        assert args.linear_fp8_activation is True
        assert args.linear_fp8_wgrad is True
        assert args.grad_quant_type is None

    def test_warmup_defaults(self):
        args = self._parse()
        assert args.warmup_steps == 0
        assert args.val_loss_target is None

    def test_use_sdma_flag(self):
        args = self._parse(["--use-sdma"])
        assert args.use_sdma is True

    def test_te_force_overrides_set(self):
        args = self._parse()
        assert args.transformer_impl == "local"
        assert args.fp8_param_gather is False

    def test_returns_parser(self):
        parser = argparse.ArgumentParser()
        result = add_common_megatron_args(parser)
        assert result is parser

    def test_no_linear_fp8_wgrad(self):
        args = self._parse(["--no-linear-fp8-wgrad"])
        assert args.linear_fp8_wgrad is False

    def test_no_linear_fp8_activation(self):
        args = self._parse(["--no-linear-fp8-activation"])
        assert args.linear_fp8_activation is False

    def test_lumen_fp8_quant_type_blockwise2d(self):
        args = self._parse(["--lumen-fp8-quant-type", "blockwise2d"])
        assert args.lumen_fp8_quant_type == "blockwise2d"

    def test_lumen_fp8_quant_type_all_choices(self):
        for qt in [
            "dynamic",
            "delayed",
            "blockwise",
            "blockwise2d",
            "per_token",
            "none",
            "mxfp8",
        ]:
            args = self._parse(["--lumen-fp8-quant-type", qt])
            assert args.lumen_fp8_quant_type == qt


# ===================================================================
# apply_fp8_training (Megatron version)
# ===================================================================


class TestApplyFP8TrainingMegatron:
    def _make_args(self, **overrides):
        defaults = {
            "lumen_fp8_format": "fp8_e4m3",
            "linear_fp8_scaling": "delayed",
            "linear_fp8_block_size": 128,
            "linear_fp8_amax_algo": "max",
            "linear_fp8_reduce_amax": False,
            "linear_fp8_amax_history": 16,
            "linear_fp8_margin": 0,
            "linear_fp8_activation": True,
            "linear_fp8_wgrad": True,
            "grad_quant_type": None,
            "first_last_layers_bf16": False,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
            "num_layers": 12,
            "use_sdma": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_enables_fp8(self, mock_print):
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args()

        from atomrl.lumen.models.megatron import apply_fp8_training as meg_apply_fp8

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            meg_apply_fp8(model, args)
            mock_enable.assert_called_once()
            config = mock_enable.call_args[1]["config"]
            assert config.format.value == "fp8_e4m3"
            assert config.scaling.value == "delayed"

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_use_sdma_forwarded(self, mock_print):
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(use_sdma=True)

        from atomrl.lumen.models.megatron import apply_fp8_training as meg_apply_fp8

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            meg_apply_fp8(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.use_sdma is True

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_num_layers_forwarded(self, mock_print):
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(num_layers=24)

        from atomrl.lumen.models.megatron import apply_fp8_training as meg_apply_fp8

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            meg_apply_fp8(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.num_layers == 24

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_reduce_amax_with_dp_group(self, mock_print):
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(linear_fp8_reduce_amax=True)

        mock_dp_group = mock.MagicMock()
        from atomrl.lumen.models.megatron import apply_fp8_training as meg_apply_fp8

        with (
            mock.patch("atomrl.lumen.quantize.enable") as mock_enable,
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "megatron.core.parallel_state.get_data_parallel_group",
                return_value=mock_dp_group,
            ),
        ):
            meg_apply_fp8(model, args)
            assert mock_enable.call_args[1]["dp_group"] is mock_dp_group


# ===================================================================
# enable_fp8_for_parallel_linear
# ===================================================================


class TestEnableFP8ForParallelLinear:
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_enables_on_lumen_modules(self, mock_print):
        from atomrl.lumen.modules.parallel_linear import LumenColumnParallelLinear

        class _MockLumenCol(LumenColumnParallelLinear):
            """Minimal subclass that passes isinstance checks without full init."""

            def __init__(self):
                nn.Module.__init__(self)
                self.enable_fp8 = mock.MagicMock()

        mock_linear = _MockLumenCol()
        model = nn.Sequential(mock_linear)

        enable_fp8_for_parallel_linear(model, scaling_type="blockwise")

        kwargs = mock_linear.enable_fp8.call_args.kwargs
        assert kwargs["scaling_manager"] is None
        assert kwargs["scaling_type"] == "blockwise"
        assert kwargs["block_size"] is None
        assert kwargs["fp8_dtype"] is not None

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_enables_on_row_parallel(self, mock_print):
        from atomrl.lumen.modules.parallel_linear import LumenRowParallelLinear

        class _MockLumenRow(LumenRowParallelLinear):
            def __init__(self):
                nn.Module.__init__(self)
                self.enable_fp8 = mock.MagicMock()

        mock_linear = _MockLumenRow()
        model = nn.Sequential(mock_linear)
        enable_fp8_for_parallel_linear(model, scaling_type="dynamic")
        kwargs = mock_linear.enable_fp8.call_args.kwargs
        assert kwargs["scaling_manager"] is None
        assert kwargs["scaling_type"] == "dynamic"
        assert kwargs["block_size"] is None
        assert kwargs["fp8_dtype"] is not None

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_enables_on_layernorm_linear(self, mock_print):
        from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear

        class _MockLumenLNLinear(LumenLayerNormLinear):
            def __init__(self):
                nn.Module.__init__(self)
                self.enable_fp8 = mock.MagicMock()

        mock_linear = _MockLumenLNLinear()
        model = nn.Sequential(mock_linear)
        enable_fp8_for_parallel_linear(model, scaling_type="delayed")
        kwargs = mock_linear.enable_fp8.call_args.kwargs
        assert kwargs["scaling_manager"] is None
        assert kwargs["scaling_type"] == "delayed"
        assert kwargs["block_size"] is None
        assert kwargs["fp8_dtype"] is not None

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_enables_on_grouped_linear(self, mock_print):
        from atomrl.lumen.modules.grouped_linear import LumenGroupedLinear

        class _MockLumenGrouped(LumenGroupedLinear):
            def __init__(self):
                nn.Module.__init__(self)
                self.enable_fp8 = mock.MagicMock()

        mock_linear = _MockLumenGrouped()
        model = nn.Sequential(mock_linear)
        enable_fp8_for_parallel_linear(model, scaling_type="blockwise")
        kwargs = mock_linear.enable_fp8.call_args.kwargs
        assert kwargs["scaling_manager"] is None
        assert kwargs["scaling_type"] == "blockwise"
        assert kwargs["block_size"] is None
        assert kwargs["fp8_dtype"] is not None

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_no_op_on_standard_linear(self, mock_print):
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        enable_fp8_for_parallel_linear(model)
        mock_print.assert_not_called()

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_block_size_taken_from_quant_config_when_unset(self, mock_print):
        from atomrl.lumen.modules.parallel_linear import LumenColumnParallelLinear
        from atomrl.lumen.quantize.config import QuantConfig

        class _MockLumenCol(LumenColumnParallelLinear):
            def __init__(self):
                nn.Module.__init__(self)
                self.enable_fp8 = mock.MagicMock()

        mock_linear = _MockLumenCol()
        model = nn.Sequential(mock_linear)

        enable_fp8_for_parallel_linear(
            model,
            scaling_type="mxfp4",
            block_size=None,
            quant_config=QuantConfig(block_size=32),
        )

        assert mock_linear.enable_fp8.call_args.kwargs["block_size"] == 32

    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_dedicated_fp4_enable_uses_mxfp4_recipe(self, mock_print):
        from atomrl.lumen.modules.parallel_linear import LumenColumnParallelLinear

        class _MockLumenCol(LumenColumnParallelLinear):
            def __init__(self):
                nn.Module.__init__(self)
                self.enable_fp8 = mock.MagicMock()

        mock_linear = _MockLumenCol()
        enable_fp4_for_parallel_linear(nn.Sequential(mock_linear))

        mock_linear.enable_fp8.assert_called_once()
        assert mock_linear.enable_fp8.call_args.kwargs["scaling_type"] == "mxfp4"
        assert mock_linear.enable_fp8.call_args.kwargs["block_size"] == 32


# ===================================================================
# model_provider -> parallel linear recipe
# ===================================================================


class TestModelProviderParallelLinearRecipe:
    """The native parallel linears must be configured from the resolved recipe.

    MX formats carry their scaling inside the format, so --linear-fp8-scaling
    still reads "blockwise" on an MXFP4 run. Forwarding that raw string made the
    native linears execute FP8 blockwise while the launcher believed it had
    asked for MXFP4 — a silent precision swap that raises nothing and only shows
    up as an unexplained loss curve.
    """

    @staticmethod
    def _args(fmt):
        return SimpleNamespace(
            linear_fp8=fmt != "mxfp4",
            linear_fp4=fmt == "mxfp4",
            lumen_linear=True,
            linear_fp8_format=None if fmt == "mxfp4" else fmt,
            linear_fp8_scaling="blockwise",
            linear_fp8_block_size=128,
            lora_rank=0,
            num_layers=4,
        )

    def _captured_kwargs(self, fmt):
        from atomrl.lumen.models.megatron import make_lumen_model_provider

        provider = make_lumen_model_provider(
            lambda args, pre_process, post_process, vp_stage, config=None: nn.Sequential(
                nn.Linear(16, 16)
            )
        )
        captured = {}
        with (
            mock.patch(
                "atomrl.lumen.models.megatron.get_args", return_value=self._args(fmt)
            ),
            mock.patch(
                "atomrl.lumen.models.megatron.enable_fp8_for_parallel_linear",
                side_effect=lambda model, **kw: captured.update(kw),
            ),
            mock.patch(
                "atomrl.lumen.models.megatron.enable_fp4_for_parallel_linear",
                side_effect=lambda model, **kw: captured.update(kw),
            ),
        ):
            provider()
        return captured

    def test_mxfp4_run_configures_linears_for_mxfp4(self):
        assert self._captured_kwargs("mxfp4")["scaling_type"] == "mxfp4"

    def test_mxfp8_run_configures_linears_for_mxfp8(self):
        assert self._captured_kwargs("mxfp8")["scaling_type"] == "mxfp8"

    def test_fp8_run_still_uses_its_scaling_string(self):
        assert self._captured_kwargs("fp8_e4m3")["scaling_type"] == "blockwise"

    def test_mxfp4_run_forwards_block_size_32(self):
        # Omitting block_size leaves the linears at their init default of 128,
        # which silently disables the MXFP4 scale-swizzle fusion.
        assert self._captured_kwargs("mxfp4")["block_size"] == 32

    @pytest.mark.parametrize("fmt", ["mxfp4", "fp8_e4m3"])
    def test_quantized_linears_without_lumen_linear_are_refused(self, fmt):
        """A half-quantized model must not start.

        The native pass is gated on --lumen-linear but cfg.enable() is not, so
        asking for FP8/FP4 linears without it left norms patched and parameters
        wrapped while every GEMM stayed BF16 -- and said nothing, so the run
        looked like a working quantized run.
        """
        from atomrl.lumen.models.megatron import make_lumen_model_provider

        args = self._args(fmt)
        args.lumen_linear = False
        provider = make_lumen_model_provider(
            lambda args, pre_process, post_process, vp_stage, config=None: nn.Sequential(
                nn.Linear(16, 16)
            )
        )
        with (
            mock.patch("atomrl.lumen.models.megatron.get_args", return_value=args),
            pytest.raises(ValueError, match="requires --lumen-linear"),
        ):
            provider()


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    def test_norm_attrs(self):
        expected = (
            "input_layernorm",
            "pre_mlp_layernorm",
            "pre_cross_attn_layernorm",
            "post_cross_attn_layernorm",
            "final_layernorm",
        )
        assert _NORM_ATTRS == expected

    def test_te_force_overrides_contains_expected(self):
        assert _TE_FORCE_OVERRIDES["transformer_impl"] == "local"
        assert _TE_FORCE_OVERRIDES["fp8_param_gather"] is False

    def test_fp8_format_map(self):
        assert _FP8_FORMAT_MAP["e4m3"] == "fp8_e4m3"
        assert _FP8_FORMAT_MAP["hybrid"] == "hybrid"


# ===================================================================
# loss_func: early stopping path
# ===================================================================


class TestLossFuncEarlyStop:
    def test_loss_func_does_not_stop_on_local_loss(self):
        import atomrl.lumen.models.megatron as meg_mod

        orig_ema = meg_mod._val_loss_ema
        orig_logged = meg_mod._early_stop_logged
        meg_mod._val_loss_ema = None
        meg_mod._early_stop_logged = False
        try:
            output_tensor = torch.tensor([0.01], device="cuda")
            loss_mask = torch.tensor([1.0], device="cuda")
            loss, _num_tokens, _report = meg_mod.loss_func(loss_mask, output_tensor)
            assert loss.item() == pytest.approx(0.01)
            assert meg_mod._val_loss_ema is None
            assert meg_mod._early_stop_logged is False
        finally:
            meg_mod._val_loss_ema = orig_ema
            meg_mod._early_stop_logged = orig_logged

    def test_eval_hook_stops_on_reduced_validation_loss(self):
        import megatron.training.training as train_mod
        from atomrl.lumen.patches.training.megatron_hooks import (
            install_val_loss_early_stop_hook,
        )

        orig_eval = train_mod.evaluate
        orig_flag = getattr(train_mod, "_lumen_val_early_stop_patched", False)
        train_mod._lumen_val_early_stop_patched = False
        train_mod.evaluate = lambda *a, **k: ({"lm loss": torch.tensor(0.5)},)
        args = SimpleNamespace(val_loss_target=1.0, iteration=7)
        try:
            with (
                mock.patch("megatron.training.get_args", return_value=args),
                mock.patch("megatron.training.print_rank_0"),
            ):
                install_val_loss_early_stop_hook()
                train_mod.evaluate()
            assert args.train_iters == 7
        finally:
            train_mod.evaluate = orig_eval
            train_mod._lumen_val_early_stop_patched = orig_flag


# ===================================================================
# _patch_cross_entropy
# ===================================================================


class TestPatchCrossEntropy:
    def test_patches_idempotently(self):
        # megatron._patch_cross_entropy is a backward-compatible alias now; the
        # guard flag it used to own lives with install_cross_entropy in the
        # patch registry, and megatron's copy is never written. Assert on the
        # registry's flag so this still tests the alias rather than a leftover.
        import atomrl.lumen.models.megatron as meg_mod
        import atomrl.lumen.patches.runtime.megatron_import as import_patches

        orig = import_patches._cross_entropy_patched
        import_patches._cross_entropy_patched = False

        try:
            with mock.patch("megatron.training.print_rank_0"):
                meg_mod._patch_cross_entropy()
                assert import_patches._cross_entropy_patched is True

                meg_mod._patch_cross_entropy()
                assert import_patches._cross_entropy_patched is True
        finally:
            import_patches._cross_entropy_patched = orig


# ===================================================================
# apply_lora (Megatron version)
# ===================================================================


class TestApplyLoraMegatron:
    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_wraps_decoder_layers(self, mock_print, mock_rank):
        from megatron.core.transformer.lora_adapter import LoraAdapter

        mock_linear_qkv = nn.Linear(16, 48)
        mock_linear_proj = nn.Linear(48, 16)
        mock_fc1 = nn.Linear(16, 32)
        mock_fc2 = nn.Linear(32, 16)

        sa = SimpleNamespace(linear_qkv=mock_linear_qkv, linear_proj=mock_linear_proj)
        mlp = SimpleNamespace(linear_fc1=mock_fc1, linear_fc2=mock_fc2)
        layer = SimpleNamespace(self_attention=sa, mlp=mlp)

        decoder = SimpleNamespace(layers=[layer])
        model_config = SimpleNamespace(sequence_parallel=False)

        _param = nn.Parameter(torch.randn(4))
        model = mock.MagicMock()
        model.config = model_config
        model.embedding = None
        model.decoder = decoder
        model.output_layer = None
        model.parameters = mock.Mock(side_effect=lambda: iter([_param]))
        model.modules.return_value = iter([model])

        args = SimpleNamespace(lora_rank=4, lora_alpha=16.0, lora_dropout=0.0)

        from atomrl.lumen.models.megatron import apply_lora as meg_apply_lora

        meg_apply_lora(model, args)

        assert isinstance(layer.self_attention.linear_qkv, LoraAdapter)
        assert isinstance(layer.self_attention.linear_proj, LoraAdapter)
        assert isinstance(layer.mlp.linear_fc1, LoraAdapter)
        assert isinstance(layer.mlp.linear_fc2, LoraAdapter)

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("atomrl.lumen.models.megatron.print_rank_0")
    def test_wraps_embedding_and_output_layer(self, mock_print, mock_rank):
        from megatron.core.transformer.lora_adapter import LoraAdapter

        mock_linear_qkv = nn.Linear(16, 48)
        mock_linear_proj = nn.Linear(48, 16)
        mock_fc1 = nn.Linear(16, 32)
        mock_fc2 = nn.Linear(32, 16)
        mock_embedding = nn.Embedding(100, 16)
        mock_output = nn.Linear(16, 100)

        sa = SimpleNamespace(linear_qkv=mock_linear_qkv, linear_proj=mock_linear_proj)
        mlp = SimpleNamespace(linear_fc1=mock_fc1, linear_fc2=mock_fc2)
        layer = SimpleNamespace(self_attention=sa, mlp=mlp)

        embedding = SimpleNamespace(word_embeddings=mock_embedding)
        decoder = SimpleNamespace(layers=[layer])
        model_config = SimpleNamespace(sequence_parallel=False)

        _param = nn.Parameter(torch.randn(4))
        model = mock.MagicMock()
        model.config = model_config
        model.embedding = embedding
        model.decoder = decoder
        model.output_layer = mock_output
        model.parameters = mock.Mock(side_effect=lambda: iter([_param]))
        model.modules.return_value = iter([model])

        args = SimpleNamespace(lora_rank=4, lora_alpha=16.0, lora_dropout=0.0)

        from atomrl.lumen.models.megatron import apply_lora as meg_apply_lora

        meg_apply_lora(model, args)

        assert isinstance(model.embedding.word_embeddings, LoraAdapter)
        assert isinstance(model.output_layer, LoraAdapter)


# ===================================================================
# make_forward_step
# ===================================================================


class TestMakeForwardStep:
    def test_returns_callable(self):
        from atomrl.lumen.models.megatron import make_forward_step

        mock_get_batch = mock.MagicMock()
        forward_step = make_forward_step(mock_get_batch)
        assert callable(forward_step)

    @mock.patch("atomrl.lumen.models.megatron.get_args")
    @mock.patch("atomrl.lumen.models.megatron.get_timers")
    @mock.patch("atomrl.lumen.models.megatron.stimer")
    @mock.patch("atomrl.lumen.models.megatron.get_attr_wrapped_model")
    def test_normal_path(self, mock_gawm, mock_stimer, mock_timers, mock_get_args):
        import atomrl.lumen.models.megatron as meg_mod

        orig_counter = meg_mod._warmup_step_counter
        orig_completed = meg_mod._warmup_completed
        meg_mod._warmup_step_counter = 0
        meg_mod._warmup_completed = True

        try:
            mock_get_args.return_value = SimpleNamespace(warmup_steps=0)
            mock_timer_obj = mock.MagicMock()
            mock_timers.return_value = mock_timer_obj
            mock_timer_obj.return_value = mock.MagicMock()
            mock_gawm.return_value = None

            batch = (
                torch.ones(1, 8, dtype=torch.long, device="cuda"),
                torch.ones(1, 8, dtype=torch.long, device="cuda"),
                torch.ones(1, 8, device="cuda"),
                torch.ones(1, 1, 8, 8, dtype=torch.bool, device="cuda"),
                torch.arange(8, device="cuda").unsqueeze(0),
            )
            mock_get_batch = mock.MagicMock(return_value=batch)

            mock_model = mock.MagicMock()
            mock_model.return_value = torch.tensor([0.5], device="cuda")

            forward_step = meg_mod.make_forward_step(mock_get_batch)
            output_tensor, loss_partial = forward_step(iter([None]), mock_model)

            assert output_tensor is not None
            assert callable(loss_partial)
        finally:
            meg_mod._warmup_step_counter = orig_counter
            meg_mod._warmup_completed = orig_completed

    @mock.patch("atomrl.lumen.models.megatron.get_args")
    @mock.patch("atomrl.lumen.models.megatron.get_timers")
    @mock.patch("atomrl.lumen.models.megatron.stimer")
    @mock.patch("atomrl.lumen.models.megatron.get_attr_wrapped_model")
    def test_output_matches_model_return(
        self, mock_gawm, mock_stimer, mock_timers, mock_get_args
    ):
        import atomrl.lumen.models.megatron as meg_mod

        orig_counter = meg_mod._warmup_step_counter
        orig_completed = meg_mod._warmup_completed
        meg_mod._warmup_step_counter = 0
        meg_mod._warmup_completed = True

        try:
            mock_get_args.return_value = SimpleNamespace(warmup_steps=0)
            mock_timer_obj = mock.MagicMock()
            mock_timers.return_value = mock_timer_obj
            mock_timer_obj.return_value = mock.MagicMock()
            mock_gawm.return_value = None

            expected_output = torch.tensor([1.5, 2.5], device="cuda")
            batch = (
                torch.ones(1, 8, dtype=torch.long, device="cuda"),
                torch.ones(1, 8, dtype=torch.long, device="cuda"),
                torch.ones(1, 8, device="cuda"),
                torch.ones(1, 1, 8, 8, dtype=torch.bool, device="cuda"),
                torch.arange(8, device="cuda").unsqueeze(0),
            )
            mock_get_batch = mock.MagicMock(return_value=batch)

            mock_model = mock.MagicMock()
            mock_model.return_value = expected_output

            forward_step = meg_mod.make_forward_step(mock_get_batch)
            output_tensor, loss_partial = forward_step(iter([None]), mock_model)

            assert torch.equal(output_tensor, expected_output)
            assert callable(loss_partial)
        finally:
            meg_mod._warmup_step_counter = orig_counter
            meg_mod._warmup_completed = orig_completed


# ===================================================================
# Norm wrapper benchmarks
# ===================================================================


class TestNormWrapperBenchmark:
    """Throughput benchmarks for Megatron-compatible norm wrappers."""

    @pytest.mark.parametrize("hidden", [1024, 4096, 8192])
    def test_rmsnorm_wrapper_throughput(self, hidden):
        config = _FakeMegatronConfig("RMSNorm")
        norm = _MegatronCompatibleTLRMSNorm(config, hidden_size=hidden).cuda()
        x = torch.randn(4, 256, hidden, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            norm(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 20

        start.record()
        for _ in range(iters):
            norm(x)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        total_bytes = 4 * 256 * hidden * 2 * 2
        bw_gb_s = (total_bytes / (avg_ms / 1000.0)) / (1024**3)
        print(
            f"\n[MegatronRMSNorm] hidden={hidden}: {avg_ms:.3f}ms, {bw_gb_s:.1f} GB/s"
        )

    @pytest.mark.parametrize("hidden", [1024, 4096, 8192])
    def test_layernorm_wrapper_throughput(self, hidden):
        config = _FakeMegatronConfig("LayerNorm")
        norm = _MegatronCompatibleTLLayerNorm(config, hidden_size=hidden).cuda()
        x = torch.randn(4, 256, hidden, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            norm(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 20

        start.record()
        for _ in range(iters):
            norm(x)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        total_bytes = 4 * 256 * hidden * 2 * 2
        bw_gb_s = (total_bytes / (avg_ms / 1000.0)) / (1024**3)
        print(
            f"\n[MegatronLayerNorm] hidden={hidden}: {avg_ms:.3f}ms, {bw_gb_s:.1f} GB/s"
        )
