# SPDX-License-Identifier: MIT

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.kimi_k3.activations import rmsnorm_gated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    ("tokens", "heads", "is_prefill"),
    [
        (4, 96, False),
        (342, 96, True),
        (448, 96, False),
        (448, 96, True),
        (512, 96, True),
        (601, 73, True),
        (1024, 96, True),
    ],
)
def test_rmsnorm_gated_matches_reference_with_strided_gate(tokens, heads, is_prefill):
    torch.manual_seed(20261002)
    dim = 128
    x = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device="cuda")
    gate_storage = torch.randn(
        tokens, heads, 3 * dim, dtype=torch.bfloat16, device="cuda"
    )
    gate = gate_storage[..., 2 * dim :]
    weight = torch.randn(dim, dtype=torch.bfloat16, device="cuda")
    x_before = x.clone()
    gate_before = gate.clone()

    actual = rmsnorm_gated(x, weight, gate, 1e-6, is_prefill=is_prefill)
    x_f32 = x.float()
    expected = (
        x_f32
        * torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight.float()
        * torch.sigmoid(gate.float())
    ).to(torch.bfloat16)

    assert not gate.is_contiguous()
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.equal(x, x_before)
    assert torch.equal(gate, gate_before)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    ("tokens", "heads"),
    [
        (4096, 96),
        (8192, 96),
        (16384, 96),
        (9474, 95),
    ],
)
def test_rmsnorm_gated_long_prefill_matches_reference(tokens, heads):
    torch.manual_seed(20261003)
    dim = 128
    x = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device="cuda")
    gate_storage = torch.randn(
        tokens, heads, dim + 1, dtype=torch.bfloat16, device="cuda"
    )
    gate = gate_storage[..., 1:]
    weight = torch.randn(dim, dtype=torch.bfloat16, device="cuda")

    actual = rmsnorm_gated(x, weight, gate, 1e-6, is_prefill=True)

    # Bound temporary fp32 memory while checking production 4K-16K prefills.
    for start in range(0, tokens, 256):
        stop = min(start + 256, tokens)
        x_chunk = x[start:stop].float()
        expected = (
            x_chunk
            * torch.rsqrt(x_chunk.square().mean(dim=-1, keepdim=True) + 1e-6)
            * weight.float()
            * torch.sigmoid(gate[start:stop].float())
        ).to(torch.bfloat16)
        torch.testing.assert_close(actual[start:stop], expected, rtol=0.02, atol=0.05)

    assert not gate.is_contiguous()
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_rmsnorm_gated_accepts_kda_output_view_without_copy():
    torch.manual_seed(20261004)
    tokens, heads, dim = 448, 96, 128
    kda_out = torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16, device="cuda")
    view = kda_out.squeeze(0)
    copied = torch.empty_like(view).copy_(view)
    gate_storage = torch.randn(
        tokens, heads, 3 * dim, dtype=torch.bfloat16, device="cuda"
    )
    gate = gate_storage[..., 2 * dim :]
    weight = torch.randn(dim, dtype=torch.bfloat16, device="cuda")

    from_view = rmsnorm_gated(view, weight, gate, 1e-6, is_prefill=True)
    from_copy = rmsnorm_gated(copied, weight, gate, 1e-6, is_prefill=True)

    assert view.is_contiguous()
    assert torch.equal(from_view, from_copy)
