# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F


class TestFusedMLPOps:
    """Test fused MLP ops (PyTorch fallback path)."""

    @pytest.fixture
    def dims(self):
        return {"batch": 4, "seq": 32, "input": 64, "hidden": 128}

    def test_fused_gated_mlp_shape(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(dims["batch"], dims["seq"], dims["input"])
        w_gate = torch.randn(dims["hidden"], dims["input"])
        w_up = torch.randn(dims["hidden"], dims["input"])
        w_down = torch.randn(dims["input"], dims["hidden"])
        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation="swiglu")
        assert out.shape == (dims["batch"], dims["seq"], dims["input"])

    def test_fused_gated_mlp_correctness(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(dims["batch"], dims["seq"], dims["input"])
        w_gate = torch.randn(dims["hidden"], dims["input"])
        w_up = torch.randn(dims["hidden"], dims["input"])
        w_down = torch.randn(dims["input"], dims["hidden"])

        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation="swiglu")

        # Reference
        gate_out = F.linear(x, w_gate)
        up_out = F.linear(x, w_up)
        hidden = F.silu(gate_out) * up_out
        ref = F.linear(hidden, w_down)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_fused_mlp_shape(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_mlp

        x = torch.randn(dims["batch"], dims["seq"], dims["input"])
        w_up = torch.randn(dims["hidden"], dims["input"])
        w_down = torch.randn(dims["input"], dims["hidden"])
        out = fused_mlp(x, w_up, w_down, activation="gelu")
        assert out.shape == (dims["batch"], dims["seq"], dims["input"])

    def test_fused_mlp_correctness(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_mlp

        x = torch.randn(dims["batch"], dims["seq"], dims["input"])
        w_up = torch.randn(dims["hidden"], dims["input"])
        w_down = torch.randn(dims["input"], dims["hidden"])

        out = fused_mlp(x, w_up, w_down, activation="gelu")

        ref = F.linear(F.gelu(F.linear(x, w_up)), w_down)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_gated_activations(self, dims, activation):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(dims["batch"], dims["seq"], dims["input"])
        w_gate = torch.randn(dims["hidden"], dims["input"])
        w_up = torch.randn(dims["hidden"], dims["input"])
        w_down = torch.randn(dims["input"], dims["hidden"])
        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation=activation)
        assert out.shape == (dims["batch"], dims["seq"], dims["input"])
        assert torch.isfinite(out).all()


class TestLumenFusedMLPModule:

    def test_construction(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128, activation="gelu")
        assert mlp.input_size == 64
        assert mlp.hidden_size == 128
        assert mlp.w_up.shape == (128, 64)
        assert mlp.w_down.shape == (64, 128)

    def test_forward(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert mlp.w_up.grad is not None

    def test_repr(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128, activation="gelu", bias=False)
        r = repr(mlp)
        assert "64" in r and "128" in r


class TestLumenGatedMLPModule:

    def test_construction(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, activation="swiglu")
        assert mlp.w_gate.shape == (128, 64)
        assert mlp.w_up.shape == (128, 64)
        assert mlp.w_down.shape == (64, 128)

    def test_forward(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_no_bias(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, bias=False)
        assert mlp.bias_gate is None
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)
