# SPDX-License-Identifier: MIT
"""Rows of a token axis out of several tensors in one launch
(`atom/model_ops/attentions/token_layout/gather.py`)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton on GPU")


def _tensors():
    torch.manual_seed(0)
    return (
        torch.randn(1, 10, 6, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 10, 3, 4, device="cuda"),
    )


def test_rows_come_out_in_new_tensors():
    from atom.model_ops.attentions.token_layout.gather import gather_rows

    rows = torch.tensor([7, 2, 2, 9], device="cuda", dtype=torch.int32)
    tensors = _tensors()
    for got, t in zip(gather_rows(rows, *tensors, dim=1), tensors):
        torch.testing.assert_close(got, t[:, rows.long()], rtol=0, atol=0)


def test_rows_land_in_the_callers_tensors():
    """What lets a request graph's inputs be gathered into in place."""
    from atom.model_ops.attentions.token_layout.gather import gather_rows

    rows = torch.tensor([3, 0, 8], device="cuda", dtype=torch.int32)
    tensors = _tensors()
    out = tuple(t.new_empty(1, 3, *t.shape[2:]) for t in tensors)
    returned = gather_rows(rows, *tensors, dim=1, out=out)
    assert all(r is o for r, o in zip(returned, out))
    for got, t in zip(out, tensors):
        torch.testing.assert_close(got, t[:, rows.long()], rtol=0, atol=0)


def test_outputs_of_another_shape_are_refused():
    from atom.model_ops.attentions.token_layout.gather import gather_rows

    rows = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    tensors = _tensors()
    out = tuple(t.new_empty(1, 3, *t.shape[2:]) for t in tensors)
    with pytest.raises(ValueError, match="disagree"):
        gather_rows(rows, *tensors, dim=1, out=out)
