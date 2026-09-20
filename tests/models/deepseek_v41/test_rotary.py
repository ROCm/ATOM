# SPDX-License-Identifier: MIT
"""V4 cached/inverse RoPE: positions, batch layout and rounding boundaries."""

import pytest
import torch

pytest.importorskip("aiter", reason="the cached RoPE is an AITER kernel")

from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


def _rotate_and_check(rope, inverse, batch, length, heads, head_dim, strided):
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
    sign = -1 if inverse else 1
    expected = torch.stack((a * cos - sign * b * sin, b * cos + sign * a * sin), -1)
    expected = expected.flatten(-2).to(hidden.dtype)

    with torch.inference_mode():
        result = rope(hidden, positions, inverse=inverse)
    assert result is hidden
    assert torch.equal(result[..., :-64], before[..., :-64])
    if strided:
        assert torch.equal(storage[:, 1::2], untouched)
    # Operator rounding tolerance; full-model quality is evaluated separately.
    torch.testing.assert_close(result[..., -64:], expected, rtol=1 / 128, atol=2**-16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "batch,length,heads,head_dim,strided",
    [
        (1, 0, 16, 512, False),
        (2, 0, None, 128, False),
        (0, 5, 8, 512, False),
        (1, 1, 16, 512, False),
        (2, 5, 8, 512, True),
        (2, 5, None, 128, True),
        (2, 33, 4, 128, False),
        (1, 257, 16, 512, False),
    ],
)
def test_rotation_batch_positions_and_aliasing(
    inverse, batch, length, heads, head_dim, strided
):
    torch.manual_seed(433)
    rope = RotaryEmbedding(64, 8192, base=10000, original_length=0, factor=16).cuda()
    _rotate_and_check(rope, inverse, batch, length, heads, head_dim, strided)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("inverse", [False, True])
def test_yarn_frequencies_are_the_table_the_kernel_applies(inverse):
    """YaRN only rebuilds the frequency table.

    The expectation above reads that table back out of `rope.frequencies`, so
    the claim the extended table makes is the same one at every shape: the
    kernel rotates by what the table holds. One shape settles it; sweeping YaRN
    across the layout cases doubles the grid and re-asserts this sentence.
    """
    torch.manual_seed(433)
    rope = RotaryEmbedding(
        64, 8192, base=160000, original_length=65536, factor=16
    ).cuda()
    plain = RotaryEmbedding(64, 8192, base=10000, original_length=0, factor=16).cuda()
    assert not torch.equal(rope.frequencies, plain.frequencies)
    _rotate_and_check(rope, inverse, 2, 33, 4, 128, False)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_graph_replay_reads_updated_rope_positions(inverse, position_dtype):
    rope = RotaryEmbedding(64, 256, base=10000).cuda()
    x = torch.randn(1, 6, 16, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(6, device="cuda", dtype=position_dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            rope(x, positions, inverse=inverse)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = rope(x, positions, inverse=inverse)
    for offset in (17, 103):
        positions.copy_(torch.arange(offset, offset + 6, device="cuda"))
        x.normal_()
        expected = rope(x.clone(), positions, inverse=inverse)
        graph.replay()
        assert result is x
        assert torch.equal(result, expected)
