# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import pytest
import torch
import torch.nn.functional as F

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")


@_CUDA
class TestFusedMLPOps:
    """Test the public AITER fused MLP kernels."""

    @pytest.fixture
    def dims(self):
        return {"batch": 4, "seq": 32, "input": 64, "hidden": 128}

    def test_fused_gated_mlp_shape(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(
            dims["batch"], dims["seq"], dims["input"], device="cuda"
        ).bfloat16()
        w_gate = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_up = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_down = (
            torch.randn(dims["input"], dims["hidden"], device="cuda")
            / dims["input"] ** 0.5
        ).bfloat16()
        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation="swiglu")
        assert out.shape == (dims["batch"], dims["seq"], dims["input"])

    def test_fused_gated_mlp_correctness(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(
            dims["batch"], dims["seq"], dims["input"], device="cuda"
        ).bfloat16()
        w_gate = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_up = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_down = (
            torch.randn(dims["input"], dims["hidden"], device="cuda")
            / dims["input"] ** 0.5
        ).bfloat16()

        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation="swiglu")

        # Reference
        gate_out = F.linear(x, w_gate)
        up_out = F.linear(x, w_up)
        hidden = F.silu(gate_out) * up_out
        ref = F.linear(hidden, w_down)

        torch.testing.assert_close(out, ref, atol=0.2, rtol=0.1)

    def test_fused_mlp_shape(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_mlp

        x = torch.randn(
            dims["batch"], dims["seq"], dims["input"], device="cuda"
        ).bfloat16()
        w_up = torch.randn(dims["hidden"], dims["input"], device="cuda").bfloat16()
        w_down = torch.randn(dims["input"], dims["hidden"], device="cuda").bfloat16()
        out = fused_mlp(x, w_up, w_down, activation="gelu")
        assert out.shape == (dims["batch"], dims["seq"], dims["input"])

    def test_fused_mlp_correctness(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_mlp

        x = torch.randn(
            dims["batch"], dims["seq"], dims["input"], device="cuda"
        ).bfloat16()
        w_up = torch.randn(dims["hidden"], dims["input"], device="cuda").bfloat16()
        w_down = torch.randn(dims["input"], dims["hidden"], device="cuda").bfloat16()

        out = fused_mlp(x, w_up, w_down, activation="gelu")

        ref = F.linear(F.gelu(F.linear(x, w_up)), w_down)
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.02)

    def test_fused_gated_mlp_backward(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        tensors = [
            torch.randn(
                dims["batch"],
                dims["seq"],
                dims["input"],
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            (
                torch.randn(
                    dims["hidden"],
                    dims["input"],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / dims["hidden"] ** 0.5
            ).requires_grad_(),
            (
                torch.randn(
                    dims["hidden"],
                    dims["input"],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / dims["hidden"] ** 0.5
            ).requires_grad_(),
            (
                torch.randn(
                    dims["input"],
                    dims["hidden"],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / dims["input"] ** 0.5
            ).requires_grad_(),
        ]
        x, w_up, w_gate, w_down = tensors
        refs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        x_ref, w_up_ref, w_gate_ref, w_down_ref = refs

        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation="swiglu")
        ref = F.linear(
            F.silu(F.linear(x_ref, w_gate_ref)) * F.linear(x_ref, w_up_ref),
            w_down_ref,
        )
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)

        for got, expected in zip(tensors, refs):
            torch.testing.assert_close(got.grad, expected.grad, atol=0.2, rtol=0.1)

    def test_fused_mlp_backward(self, dims):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_mlp

        tensors = [
            torch.randn(
                dims["batch"],
                dims["seq"],
                dims["input"],
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            (
                torch.randn(
                    dims["hidden"],
                    dims["input"],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / dims["hidden"] ** 0.5
            ).requires_grad_(),
            (
                torch.randn(
                    dims["input"],
                    dims["hidden"],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / dims["input"] ** 0.5
            ).requires_grad_(),
        ]
        x, w_up, w_down = tensors
        refs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        x_ref, w_up_ref, w_down_ref = refs

        out = fused_mlp(x, w_up, w_down, activation="gelu")
        ref = F.linear(F.gelu(F.linear(x_ref, w_up_ref)), w_down_ref)
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)

        for got, expected in zip(tensors, refs):
            torch.testing.assert_close(got.grad, expected.grad, atol=0.2, rtol=0.1)

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_gated_activations(self, dims, activation):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(
            dims["batch"], dims["seq"], dims["input"], device="cuda"
        ).bfloat16()
        w_gate = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_up = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_down = (
            torch.randn(dims["input"], dims["hidden"], device="cuda")
            / dims["input"] ** 0.5
        ).bfloat16()
        out = fused_gated_mlp(x, w_up, w_gate, w_down, activation=activation)
        gate = F.linear(x, w_gate)
        if activation == "swiglu":
            activated = F.silu(gate)
        elif activation == "geglu":
            activated = F.gelu(gate)
        else:
            activated = F.relu(gate)
        reference = F.linear(activated * F.linear(x, w_up), w_down)
        torch.testing.assert_close(out, reference, atol=0.2, rtol=0.1)

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_biased_gated_activations(self, dims, activation):
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp

        x = torch.randn(
            dims["batch"], dims["seq"], dims["input"], device="cuda"
        ).bfloat16()
        w_gate = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_up = (
            torch.randn(dims["hidden"], dims["input"], device="cuda")
            / dims["hidden"] ** 0.5
        ).bfloat16()
        w_down = (
            torch.randn(dims["input"], dims["hidden"], device="cuda")
            / dims["input"] ** 0.5
        ).bfloat16()
        bias_gate = torch.randn(dims["hidden"], device="cuda").bfloat16()
        bias_up = torch.randn(dims["hidden"], device="cuda").bfloat16()
        bias_down = torch.randn(dims["input"], device="cuda").bfloat16()

        result = fused_gated_mlp(
            x,
            w_up,
            w_gate,
            w_down,
            activation=activation,
            bias_gate=bias_gate,
            bias_up=bias_up,
            bias_down=bias_down,
        )
        gate = F.linear(x, w_gate, bias_gate)
        if activation == "swiglu":
            activated = F.silu(gate)
        elif activation == "geglu":
            activated = F.gelu(gate)
        else:
            activated = F.relu(gate)
        reference = F.linear(activated * F.linear(x, w_up, bias_up), w_down, bias_down)
        torch.testing.assert_close(result, reference, atol=0.2, rtol=0.1)


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

    def test_no_bias_cpu_forward_backward(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, bias=False)
        x = torch.randn(2, 16, 64, requires_grad=True)
        output = mlp(x)
        output.sum().backward()
        assert output.shape == (2, 16, 64)
        assert x.grad is not None
        assert all(parameter.grad is not None for parameter in mlp.parameters())

    @_CUDA
    def test_no_bias(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, bias=False).to(device="cuda", dtype=torch.bfloat16)
        assert mlp.bias_gate is None
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        out = mlp(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert mlp.w_gate.grad is not None
        assert mlp.w_up.grad is not None
        assert mlp.w_down.grad is not None
