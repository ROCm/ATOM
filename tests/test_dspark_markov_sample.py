# SPDX-License-Identifier: MIT
"""The Markov sampling op's contract: the destination, and the row it biased by.

The W1 gather lives inside stage 1 -- every tile program reads those rows to
form the bias, so a separate `index_select` would be a launch that reads what
the kernel is about to read anyway. That makes the gathered row the op's only
return value, and the confidence head downstream consumes it, so it has to be
the same row the bias was formed from rather than merely a plausible one.

The ids leave through `out` and nothing else. `out` is required for that
reason: `torch.compile` may functionalize the call, and the handle it would
hand back then is a copy, so a returned id tensor could not be trusted.
"""

import pytest
import torch

pytest.importorskip("aiter", reason="the Markov sampler is a Triton/AITER op")

from atom.model_ops.dspark_markov_sample import (
    _torch_dspark_markov_argmax,
    dspark_markov_argmax,
)


def _tables(rows, vocab, rank, device, seed=1201):
    torch.manual_seed(seed)
    kwargs = {"dtype": torch.bfloat16, "device": device}
    return (
        torch.randn(rows, vocab, **kwargs),
        torch.randint(0, vocab, (rows,), dtype=torch.int64, device=device),
        torch.randn(vocab, rank, **kwargs),
        torch.randn(vocab, rank, **kwargs),
    )


def _ids(rows, device, dtype=torch.int64):
    return torch.empty(rows, dtype=dtype, device=device)


def test_torch_body_returns_the_row_it_gathered():
    """The fallback every non-CUDA caller takes, and CI's only path here."""
    base, prev, w1, w2 = _tables(5, 64, 16, "cpu")
    ids = _ids(5, "cpu")
    embed = _torch_dspark_markov_argmax(base, prev, w1, w2, ids)
    assert torch.equal(embed, w1[prev])
    assert torch.equal(ids, (base + w1[prev].float() @ w2.float().t()).argmax(-1))


def test_shapes_are_checked_against_each_other():
    base, prev, w1, w2 = _tables(5, 64, 16, "cpu")
    ids = _ids(5, "cpu")
    with pytest.raises(ValueError, match="prev_ids"):
        dspark_markov_argmax(base, prev[:-1], w1, w2, ids)
    with pytest.raises(ValueError, match="markov_w1"):
        dspark_markov_argmax(base, prev, w1[:, :-1], w2, ids)
    with pytest.raises(ValueError, match="markov_w2"):
        dspark_markov_argmax(base, prev, w1, w2[:-1], ids)
    embed = dspark_markov_argmax(base[:0], prev[:0], w1, w2, ids[:0])
    assert embed.shape == (0, 16)


def test_ids_are_rejected_at_both_ends_unless_an_id_block_could_hold_them():
    """`prev_ids` indexes a table and `out` is indexed by one, so both are ids.

    A float `prev_ids` would gather by a reinterpreted address in the kernel
    and by an index error in the fallback -- two failures for one mistake, and
    only one of them loud.
    """
    base, prev, w1, w2 = _tables(5, 64, 16, "cpu")
    with pytest.raises(ValueError, match="prev_ids"):
        dspark_markov_argmax(base, prev.float(), w1, w2, _ids(5, "cpu"))
    with pytest.raises(ValueError, match="out"):
        dspark_markov_argmax(base, prev, w1, w2, _ids(4, "cpu"))
    with pytest.raises(ValueError, match="out"):
        dspark_markov_argmax(base, prev, w1, w2, _ids(5, "cpu", torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_fused_gathers_the_same_row_it_biases_with():
    """Fused against the torch body: the ids, and the row bit for bit.

    The row is the strict half -- a gather is exact, so anything but equality
    means stage 1 published a different row than it biased with, and the
    confidence head would drift with nothing pointing back here. The ids are
    equal up to accumulation order (see the op's module docstring); a tie
    closer than that would be the one thing this could not distinguish, which
    random logits make vanishingly unlikely.
    """
    base, prev, w1, w2 = _tables(7, 2048, 128, "cuda")
    ids, ref_ids = _ids(7, "cuda"), _ids(7, "cuda")
    embed = dspark_markov_argmax(base, prev, w1, w2, ids)
    ref_embed = _torch_dspark_markov_argmax(base, prev, w1, w2, ref_ids)
    # Armed: a batch whose rows all shared a previous token would pass with
    # the gather reading any single row.
    assert len(set(prev.tolist())) > 1
    assert torch.equal(embed, ref_embed)
    assert torch.equal(embed, w1[prev])
    assert torch.equal(ids, ref_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fused_reads_and_writes_the_strided_columns_it_is_given(dtype):
    """The production spelling: both ids are columns of one `[B, T + 1]` block.

    `out_ids[:, k]` carries stride `T + 1`, so a kernel that assumed unit
    stride would bias each row with another position's token on the way in and
    land each id in another position's slot on the way out -- in bounds, never
    a fault, and visible only as a draft that gets rejected more often.

    int32 is the one that matters: a draft's id block comes from
    `anchor_ids.new_empty`, so the store is narrowing as well as moving, and a
    kernel that only stored int64 would corrupt the neighbouring column.
    """
    torch.manual_seed(5)
    rows, width, vocab, rank = 6, 5, 2048, 128
    out_ids = torch.randint(0, vocab, (rows, width + 1), dtype=dtype, device="cuda")
    base = torch.randn(rows, vocab, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(vocab, rank, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(vocab, rank, dtype=torch.bfloat16, device="cuda")
    for k in range(width):
        prev = out_ids[:, k]
        # Armed: a contiguous column would make this the test above again.
        assert not prev.is_contiguous() and prev.stride(0) == width + 1
        ref_ids = _ids(rows, "cuda", dtype)
        ref_embed = _torch_dspark_markov_argmax(base, prev, w1, w2, ref_ids)
        embed = dspark_markov_argmax(base, prev, w1, w2, out_ids[:, k + 1])
        assert torch.equal(embed, ref_embed), k
        assert torch.equal(out_ids[:, k + 1], ref_ids), k
