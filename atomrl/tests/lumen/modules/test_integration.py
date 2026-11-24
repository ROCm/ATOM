# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Module-level integration tests — verify that Lumen modules compose correctly.

Covers:
  - LayerNorm + Linear pipeline (LumenLayerNormLinear)
  - Attention + FP8 forward pipeline
  - Fused MLP + FP8 activation store + backward
  - Full transformer-like block: norm → attention → norm → MLP
  - Zero-centered gamma in LayerNormLinear
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


_AITER = _CUDA


def _compute_snr(ref, test):
    ref_f, test_f = ref.float(), test.float()
    signal = ref_f.norm().pow(2)
    noise = (ref_f - test_f).norm().pow(2)
    if noise < 1e-12:
        return float("inf")
    return (10.0 * torch.log10(signal / (noise + 1e-12))).item()


# =========================================================================
# LayerNormLinear integration (with zero-centered gamma)
# =========================================================================


def _make_lnl_config(normalization="RMSNorm", zero_centered=False):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        normalization=normalization,
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=zero_centered,
    )


_LNL_PATCHES = [
    mock.patch(
        "atomrl.lumen.modules.layernorm_linear._get_tp_group", return_value=None
    ),
    mock.patch("atomrl.lumen.modules.layernorm_linear._pg_size", return_value=1),
    mock.patch("atomrl.lumen.modules.layernorm_linear._pg_rank", return_value=0),
    mock.patch(
        "atomrl.lumen.modules.layernorm_linear._use_sdma_from_args", return_value=False
    ),
    mock.patch(
        "atomrl.lumen.modules.layernorm_linear.divide", side_effect=lambda a, b: a // b
    ),
    mock.patch("atomrl.lumen.modules.layernorm_linear._initialize_affine_weight_gpu"),
    mock.patch(
        "atomrl.lumen.modules.layernorm_linear.set_tensor_model_parallel_attributes"
    ),
    mock.patch(
        "atomrl.lumen.modules.layernorm_linear.make_sharded_tensors_for_checkpoint",
        return_value={},
    ),
    mock.patch(
        "atomrl.lumen.modules.layernorm_linear.gather_from_sequence_parallel_region",
        side_effect=lambda x, **kw: x,
    ),
]


def _apply_lnl_patches(func):
    for p in reversed(_LNL_PATCHES):
        func = p(func)
    return func


@_CUDA
class TestZeroCenteredGamma:
    """Test zero-centered gamma in LayerNormLinear."""

    @_apply_lnl_patches
    def test_zero_centered_construction(self, *_):
        from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear

        config = _make_lnl_config("LayerNorm", zero_centered=True)
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.zero_centered_gamma is True

    @_apply_lnl_patches
    def test_zero_centered_forward_shape(self, *_):
        from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear

        config = _make_lnl_config("LayerNorm", zero_centered=True)
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        assert out.shape == (2, 16, 128)

    @_apply_lnl_patches
    def test_zero_centered_differs_from_normal(self, *_):
        """With gamma=0 init, zero-centered uses gamma+1=1 (active), non-zero-centered uses gamma=0 (dead)."""
        from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear

        config_zc = _make_lnl_config("LayerNorm", zero_centered=True)
        config_normal = _make_lnl_config("LayerNorm", zero_centered=False)

        m_zc = LumenLayerNormLinear(
            64,
            128,
            config=config_zc,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        m_normal = LumenLayerNormLinear(
            64,
            128,
            config=config_normal,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )

        with torch.no_grad():
            # GPU affine init is patched out, so the linear weight would stay
            # zero and both GEMMs would return 0 regardless of gamma.
            # perform_initialization is also off, so the bias buffer is
            # uninitialized and can dominate both outputs.
            torch.manual_seed(0)
            torch.nn.init.normal_(m_normal.weight, std=0.02)
            m_zc.ln_weight.zero_()
            m_normal.ln_weight.zero_()
            m_zc.weight.copy_(m_normal.weight)
            if m_zc.ln_bias is not None and m_normal.ln_bias is not None:
                m_zc.ln_bias.copy_(m_normal.ln_bias)
            if m_zc.bias is not None:
                m_zc.bias.zero_()
            if m_normal.bias is not None:
                m_normal.bias.zero_()

        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        out_zc, _ = m_zc(x)
        out_normal, _ = m_normal(x)

        # With ln_weight=0: zero-centered uses (0+1)=1 as gamma → non-zero output
        # Normal uses 0 as gamma → near-zero output after norm
        assert (
            out_zc.abs().mean() > out_normal.abs().mean() * 2
        ), "Zero-centered gamma with weight=0 should produce larger output than normal gamma=0"


# =========================================================================
# Attention module integration
# =========================================================================


@_CUDA
@_AITER
class TestAttentionModuleIntegration:
    """Test LumenAttention with various configurations."""

    def test_attention_forward_backward(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention()
        B, S, H, D = 2, 128, 8, 64
        q = torch.randn(
            B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = attn(q, k, v)
        assert out.shape == (B, S, H, D)
        out.sum().backward()
        assert q.grad is not None

    def test_attention_causal(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(causal=True)
        B, S, H, D = 2, 128, 8, 64
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        out = attn(q, k, v)
        assert out.shape == (B, S, H, D)
        assert torch.isfinite(out).all()

    def test_attention_gqa(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention()
        B, S = 2, 128
        q = torch.randn(B, S, 16, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
        out = attn(q, k, v)
        assert out.shape == (B, S, 16, 64)


# =========================================================================
# Fused MLP integration
# =========================================================================


@_CUDA
class TestFusedMLPIntegration:
    """Test fused MLP modules in realistic configurations."""

    def test_gated_mlp_train_step(self):
        """Simulate a mini training step: forward → loss → backward → param update."""
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(256, 512, fp8_activation_store=True).to("cuda")
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        for _ in range(3):
            x = torch.randn(4, 32, 256, device="cuda") * 0.1
            out = mlp(x)
            loss = out.pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert loss.item() < 1.0, "Loss should decrease after 3 steps"

    def test_ungated_mlp_train_step(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(256, 512, fp8_activation_store=True).to("cuda")
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        for _ in range(3):
            x = torch.randn(4, 32, 256, device="cuda") * 0.1
            out = mlp(x)
            loss = out.pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert loss.item() < 1.0


# =========================================================================
# Full transformer-like block: norm → attn → residual → norm → MLP → residual
# =========================================================================


class _MiniTransformerBlock(torch.nn.Module):
    """Minimal transformer block using Lumen modules for integration testing."""

    def __init__(self, hidden_dim, num_heads, mlp_hidden):
        super().__init__()
        from atomrl.lumen.modules.attention import LumenAttention
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = LumenRMSNorm(hidden_dim)
        self.attn = LumenAttention()
        self.norm2 = LumenRMSNorm(hidden_dim)
        self.mlp = LumenGatedMLP(hidden_dim, mlp_hidden)

    def forward(self, x):
        B, S, D = x.shape

        h = self.norm1(x)
        q = h.view(B, S, self.num_heads, self.head_dim)
        k = h.view(B, S, self.num_heads, self.head_dim)
        v = h.view(B, S, self.num_heads, self.head_dim)
        attn_out = self.attn(q, k, v).view(B, S, D)
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


@_CUDA
@_AITER
class TestMiniTransformerBlock:
    """End-to-end test: norm → attn → norm → MLP as a single block."""

    def test_forward_shape(self):
        block = _MiniTransformerBlock(hidden_dim=128, num_heads=4, mlp_hidden=256).to(
            device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
        out = block(x)
        assert out.shape == (2, 64, 128)

    def test_backward_all_grads(self):
        block = _MiniTransformerBlock(hidden_dim=128, num_heads=4, mlp_hidden=256).to(
            device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(
            2, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = block(x)
        out.sum().backward()

        assert x.grad is not None, "Input gradient missing"
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"

    def test_train_step_loss_decreases(self):
        block = _MiniTransformerBlock(hidden_dim=128, num_heads=4, mlp_hidden=256).to(
            device="cuda", dtype=torch.bfloat16
        )
        optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            x = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
            out = block(x)
            loss = out.pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"

    def test_no_nan_inf(self):
        block = _MiniTransformerBlock(hidden_dim=128, num_heads=4, mlp_hidden=256).to(
            device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        out = block(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

        out.sum().backward()
        for name, param in block.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(
                    param.grad
                ).all(), f"Gradient for {name} contains NaN or Inf"
