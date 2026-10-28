# SPDX-License-Identifier: MIT
"""ATOM_SAMPLER_SPLIT_SOFTMAX: the split-row softmax against the eager
`(logits / t).softmax(-1, dtype=float32)` the sampler runs, on a GPU."""

import pytest
import torch

split_softmax = pytest.importorskip(
    "atom.model_ops.split_softmax",
    reason="split_softmax imports triton",
    exc_type=ImportError,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def eager(logits, temperatures):
    return (logits / temperatures.unsqueeze(-1)).softmax(dim=-1, dtype=torch.float32)


@pytest.mark.parametrize("rows", [1, 5, 35])
@pytest.mark.parametrize("cols", [248320, 151936, 4097, 100])
def test_matches_eager(rows, cols):
    torch.manual_seed(rows * cols)
    logits = torch.randn(rows, cols, device="cuda") * 6
    temperatures = 0.5 + torch.rand(rows, device="cuda")
    probs = split_softmax.split_softmax(logits, temperatures)
    assert probs.dtype == torch.float32 and probs.is_contiguous()
    # As close to the eager fp32 result as either is to fp64 (~5e-7 abs).
    torch.testing.assert_close(probs, eager(logits, temperatures), rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(
        probs.double().sum(-1),
        torch.ones(rows, device="cuda", dtype=torch.float64),
        rtol=0,
        atol=1e-6,
    )


def test_masked_columns_and_chunks():
    # -inf columns (masked vocab) get probability 0; a chunk that is all -inf
    # must not turn the row's sum into nan.
    logits = torch.randn(3, 248320, device="cuda")
    logits[:, -243:] = -float("inf")
    logits[1, 4096:8192] = -float("inf")
    temperatures = torch.ones(3, device="cuda")
    probs = split_softmax.split_softmax(logits, temperatures)
    assert torch.isfinite(probs).all()
    assert (probs[:, -243:] == 0).all() and (probs[1, 4096:8192] == 0).all()
    torch.testing.assert_close(probs, eager(logits, temperatures), rtol=1e-5, atol=1e-9)


def test_row_view_and_temperature_dtypes():
    # A row-strided view (index_select output is dense; a slice of a wider
    # buffer is not) and a temperature tensor that needs .contiguous().
    base = torch.randn(4, 300000, device="cuda")
    logits = base[:, :248320]
    temperatures = torch.rand(8, device="cuda")[::2] + 0.5
    assert split_softmax.can_split_softmax(logits, temperatures)
    torch.testing.assert_close(
        split_softmax.split_softmax(logits, temperatures),
        eager(logits, temperatures),
        rtol=1e-5,
        atol=1e-9,
    )


def test_can_split_softmax_turns_away():
    t = torch.ones(2, device="cuda")
    assert not split_softmax.can_split_softmax(
        torch.randn(2, 64, device="cuda", dtype=torch.bfloat16), t
    )
    assert not split_softmax.can_split_softmax(torch.randn(64, 2, device="cuda").t(), t)
    assert not split_softmax.can_split_softmax(
        torch.randn(0, 64, device="cuda"), torch.ones(0, device="cuda")
    )
    assert not split_softmax.can_split_softmax(
        torch.randn(2, 64, device="cuda"), torch.ones(3, device="cuda")
    )
