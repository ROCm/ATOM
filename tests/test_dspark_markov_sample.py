# SPDX-License-Identifier: MIT
"""Fused DSpark Markov block-sampling step vs the per-op reference.

The op replaces ``(base_logits + W1[x] @ W2^T).argmax(-1)`` with one fused
GEMV+argmax, so the assertions here are on the sampled ids, not on a bias
tensor the fused path never builds. Ids are checked for EXACT equality: the two
paths sum the same ``r`` exact products and differ only in accumulation order,
which cannot move an argmax unless two logits sit within a last-ulp
disagreement -- the random draws below keep the margin far wider than that, and
the tie case pins the tie-break with a bias that is exactly zero.
"""

import pytest
import torch

from atom.model_ops.dspark_markov_sample import (
    _MAX_BLOCK_ROW,
    _torch_dspark_markov_argmax,
    dspark_markov_argmax,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused sampler is a Triton kernel"
)

# Kimi-K3-DSpark's own rank; the vocab is cut down so the test stays quick while
# still spanning many V tiles (BLOCK_V=128).
RANK = 256


def _draw(num_rows, vocab_size, rank=RANK, dtype=torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(num_rows * 1000 + vocab_size)
    base = torch.randn(
        num_rows, vocab_size, generator=gen, device="cuda", dtype=torch.float32
    ).to(dtype)
    embed = torch.randn(
        num_rows, rank, generator=gen, device="cuda", dtype=torch.float32
    ).to(dtype)
    w2 = torch.randn(
        vocab_size, rank, generator=gen, device="cuda", dtype=torch.float32
    ).to(dtype)
    return base, embed, w2


@pytest.mark.parametrize(
    "num_rows,vocab_size",
    [
        (1, 4096),  # single request
        (8, 4096),  # below the MFMA M=16 floor, so padded-and-masked
        (64, 16384),  # the benchmarked --max-num-seqs
        (7, 16000),  # vocab not a multiple of BLOCK_V
    ],
)
def test_matches_reference(num_rows, vocab_size):
    base, embed, w2 = _draw(num_rows, vocab_size)
    torch.testing.assert_close(
        dspark_markov_argmax(base, embed, w2),
        _torch_dspark_markov_argmax(base, embed, w2),
        rtol=0,
        atol=0,
    )


def test_row_tiling_beyond_the_accumulator_cap():
    """A batch wider than BLOCK_ROW must tile rows, not widen the accumulator."""
    num_rows = _MAX_BLOCK_ROW * 3 + 5
    base, embed, w2 = _draw(num_rows, 8192)
    torch.testing.assert_close(
        dspark_markov_argmax(base, embed, w2),
        _torch_dspark_markov_argmax(base, embed, w2),
        rtol=0,
        atol=0,
    )


def test_ties_take_the_lowest_id():
    """torch.argmax returns the first maximum; so must the two-stage reduce.

    The bias is forced to exactly zero so the tie survives both paths bit for
    bit, and the duplicated maxima straddle a V-tile boundary so the tie-break
    is exercised inside a tile and across the cross-tile reduce.
    """
    vocab_size = 4096
    base = torch.full((2, vocab_size), -1.0, device="cuda", dtype=torch.bfloat16)
    base[0, [5, 130, 900]] = 3.0  # first tile wins, twice over
    base[1, [200, 201, 3000]] = 3.0
    embed = torch.zeros(2, RANK, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(vocab_size, RANK, device="cuda", dtype=torch.bfloat16)

    ids = dspark_markov_argmax(base, embed, w2)
    assert ids.tolist() == [5, 200]
    torch.testing.assert_close(
        ids, _torch_dspark_markov_argmax(base, embed, w2), rtol=0, atol=0
    )


def test_strided_base_logits_view():
    """vLLM slices [B, T, V] per position, so base_logits arrives non-contiguous."""
    num_rows, n_spec, vocab_size = 16, 7, 4096
    _, embed, w2 = _draw(num_rows, vocab_size)
    block = torch.randn(
        num_rows, n_spec, vocab_size, device="cuda", dtype=torch.bfloat16
    )
    for i in (0, n_spec - 1):
        base = block[:, i]
        assert base.stride(0) != vocab_size and base.stride(1) == 1
        torch.testing.assert_close(
            dspark_markov_argmax(base, embed, w2),
            _torch_dspark_markov_argmax(base, embed, w2),
            rtol=0,
            atol=0,
        )


def test_unsupported_layout_falls_back():
    """A transposed W2 has no stride-1 rank axis and must take the torch path."""
    base, embed, w2 = _draw(4, 2048)
    w2_t = w2.t().contiguous().t()  # [V, r] view whose rank stride is not 1
    assert w2_t.stride(1) != 1
    torch.testing.assert_close(
        dspark_markov_argmax(base, embed, w2_t),
        _torch_dspark_markov_argmax(base, embed, w2_t),
        rtol=0,
        atol=0,
    )
