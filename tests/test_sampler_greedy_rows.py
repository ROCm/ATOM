"""`_greedy_tokens` picks the right rows, whichever reducer answers.

The greedy fixup used to reduce `probs[greedy_mask]` -- a gather that
materializes a `[greedy, vocab]` copy before reducing anything. It now reduces
every row and keeps the selected answers, which is the same result only if the
row bookkeeping lines up: the gather produced answers in mask order, and so must
indexing the full answer.

That bookkeeping is ATOM's half and is what this covers. The reducer's half --
that a per-row argmax equals `torch.argmax`, ties included -- belongs to aiter's
own op tests and needs a GPU, which this suite does not have.
"""

import pytest
import torch

sampler = pytest.importorskip(
    "atom.model_ops.sampler",
    reason="sampler requires the Triton/AITER runtime",
    exc_type=ImportError,
)


def _torch_topk_select(input, topk, *, tie=None, **_kwargs):
    """`aiter.topk_select` over torch, for a runner with no GPU.

    The reduction is aiter's and is tested where a GPU can run it. Stubbing it
    leaves exactly the half this file is about. `tie="low"` is asserted rather
    than honoured -- `torch.argmax` already breaks ties that way, so a stub that
    silently accepted any `tie` would let the call site lose the promise without
    a test noticing.
    """
    assert tie == "low", f"the sampler must ask for the lowest-index tie, not {tie!r}"
    return None, torch.topk(input, topk, dim=-1, sorted=True).indices.to(torch.int32)


@pytest.fixture(autouse=True)
def _stub_selector(monkeypatch):
    monkeypatch.setattr(sampler, "topk_select", _torch_topk_select)


def _probs(rows: int, vocab: int, seed: int) -> torch.Tensor:
    return torch.rand(rows, vocab, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("rows,vocab", [(1, 8), (4, 16), (17, 61), (64, 129)])
@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 1.0])
def test_greedy_rows_match_the_gather_they_replaced(rows, vocab, frac):
    probs = _probs(rows, vocab, rows * 31 + vocab)
    mask = torch.zeros(rows, dtype=torch.bool)
    mask[: int(rows * frac)] = True

    got = sampler._greedy_tokens(probs, mask)

    assert got.shape == (int(rows * frac),)
    assert torch.equal(got, probs[mask].argmax(dim=-1))


def test_greedy_rows_follow_a_scattered_mask():
    """Not just a prefix: the rows kept must be the rows the mask names.

    A contiguous mask cannot tell "index the full answer" apart from "reduce the
    first N rows", which is the way this rewrite could have gone wrong.
    """
    probs = _probs(9, 32, 7)
    mask = torch.tensor([False, True, False, False, True, True, False, False, True])

    got = sampler._greedy_tokens(probs, mask)

    assert torch.equal(got, probs[mask].argmax(dim=-1))
    assert torch.equal(
        got, torch.stack([probs[r].argmax() for r in (1, 4, 5, 8)]).to(got.dtype)
    )


def test_greedy_rows_are_written_where_the_mask_points():
    """The assignment the callers make, which is where a row mix-up would land."""
    probs = _probs(6, 24, 3)
    mask = torch.tensor([False, True, True, False, False, True])
    next_tokens = torch.full((6,), -1, dtype=torch.long)

    next_tokens[mask] = sampler._greedy_tokens(probs, mask).to(next_tokens.dtype)

    for row in range(6):
        expected = int(probs[row].argmax()) if mask[row] else -1
        assert int(next_tokens[row]) == expected, f"row {row}"


def test_an_all_false_mask_selects_nothing():
    probs = _probs(5, 12, 11)
    mask = torch.zeros(5, dtype=torch.bool)

    assert sampler._greedy_tokens(probs, mask).numel() == 0
