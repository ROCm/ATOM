# SPDX-License-Identifier: MIT
"""V4 inverse-RoPE integration: positions, batch layout and rounding boundary."""

import pytest
import torch

from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("yarn", [False, True])
@pytest.mark.parametrize(
    "batch,length,heads,head_dim,strided",
    [
        (1, 1, 16, 512, False),
        (2, 5, 8, 512, True),
        (2, 5, None, 128, True),
        (2, 33, 4, 128, False),
        (1, 257, 16, 512, False),
    ],
)
def test_inverse_rotation_batch_positions_and_aliasing(
    yarn, batch, length, heads, head_dim, strided
):
    torch.manual_seed(433)
    rope = RotaryEmbedding(
        64,
        8192,
        base=160000 if yarn else 10000,
        original_length=65536 if yarn else 0,
        factor=16,
    ).cuda()
    tail_shape = (head_dim,) if heads is None else (heads, head_dim)
    storage = torch.randn(
        batch,
        length * (2 if strided else 1),
        *tail_shape,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden = storage[:, ::2] if strided else storage
    before = hidden.clone()
    untouched = storage[:, 1::2].clone() if strided else None
    # A non-contiguous position vector also tests the V4 kernel's flat ABI.
    positions = torch.arange(6000, 6000 + length * 2, device="cuda")[::2]
    freqs = rope.frequencies[positions]
    shape = [1, length] + [1] * (hidden.ndim - 3) + [32]
    cos, sin = freqs.real.double().view(shape), freqs.imag.double().view(shape)
    a, b = before[..., -64:].double().unflatten(-1, (32, 2)).unbind(-1)
    expected = torch.stack((a * cos + b * sin, b * cos - a * sin), -1)
    expected = expected.flatten(-2).to(hidden.dtype)

    with torch.inference_mode():
        result = rope(hidden, positions, inverse=True)
    assert result is hidden
    assert torch.equal(result[..., :-64], before[..., :-64])
    if strided:
        assert torch.equal(storage[:, 1::2], untouched)
    # Operator rounding tolerance; full-model quality is evaluated separately.
    torch.testing.assert_close(result[..., -64:], expected, rtol=1 / 128, atol=2**-16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_inverse_preserves_v4_fma_rounding_at_bf16_midpoint():
    rope = RotaryEmbedding(64, 38, base=10000).cuda()
    hidden = torch.zeros(1, 38, 16, 512, device="cuda", dtype=torch.bfloat16)
    hidden[0, 5, 13, 464] = -0.2470703125
    hidden[0, 5, 13, 465] = 0.5
    with torch.inference_mode():
        actual = rope(hidden, torch.arange(38, device="cuda"), inverse=True)
    # With these FP32 frequencies, a*cos+b*sin is 3.725e-9 below the
    # BF16 midpoint. V4 FMA rounds down; eager complex multiplication rounds up.
    assert actual[0, 5, 13, 464].item() == 0.0228271484375
