# SPDX-License-Identifier: MIT
"""The one-launch DSpark confidence head (`atom.model_ops.dspark_confidence`).

The kernel is plain Triton, so its numerics are checked on any GPU by calling
it directly; `can_fuse_confidence` takes it on any GPU arch.
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from atom.model_ops.dspark_confidence import (
    can_fuse_confidence,
    fused_confidence,
)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(hidden, embed, weight):
    x = torch.cat([hidden, embed], dim=-1).double()
    return torch.sigmoid(x @ weight.double().t()).squeeze(-1)


@needs_gpu
@pytest.mark.parametrize("lead", [(1,), (3,), (7, 5), (224, 5), (256, 5)])
def test_fused_confidence_matches_fp64(lead):
    torch.manual_seed(0)
    hidden_size, rank = 5120, 256
    hidden = torch.randn(*lead, hidden_size, device="cuda", dtype=torch.bfloat16)
    embed = torch.randn(*lead, rank, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(1, hidden_size + rank, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    out = fused_confidence(hidden, embed, weight)
    assert out.shape == lead
    assert out.dtype == torch.float32
    torch.testing.assert_close(
        out.double(), _reference(hidden, embed, weight), atol=1e-5, rtol=0
    )


@needs_gpu
def test_fused_confidence_reads_strided_rows():
    """Rows of a wider buffer (the draft's hidden slice) are read in place."""
    torch.manual_seed(1)
    hidden_size, rank = 1024, 64
    wide = torch.randn(9, hidden_size + 512, device="cuda", dtype=torch.bfloat16)
    hidden = wide[:, :hidden_size]
    embed = torch.randn(9, rank, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1, hidden_size + rank, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        fused_confidence(hidden, embed, weight).double(),
        _reference(hidden, embed, weight),
        atol=1e-5,
        rtol=0,
    )


@needs_gpu
def test_fused_confidence_empty_batch():
    hidden = torch.empty(0, 512, device="cuda", dtype=torch.bfloat16)
    embed = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(1, 576, device="cuda", dtype=torch.bfloat16)
    assert fused_confidence(hidden, embed, weight).shape == (0,)


@needs_gpu
def test_gate_rejects_what_the_kernel_does_not_tile():
    h = torch.zeros(4, 512, device="cuda", dtype=torch.bfloat16)
    m = torch.zeros(4, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(1, 576, device="cuda", dtype=torch.bfloat16)
    assert can_fuse_confidence(h, m, w, None)  # on any GPU arch
    assert not can_fuse_confidence(h, m, w, torch.zeros(1, device="cuda"))
    assert not can_fuse_confidence(h.float(), m, w, None)
    assert not can_fuse_confidence(h[:, :500], m, w[:, :564], None)
    assert not can_fuse_confidence(h, m[:, :48], w[:, :560], None)
    assert not can_fuse_confidence(h.t().contiguous().t(), m, w, None)


@pytest.mark.parametrize(
    "hidden_size, rank, weight_cols",
    [(500, 64, 564), (512, 48, 560), (512, 64, 600)],
)
def test_fused_confidence_refuses_shapes_it_would_read_past(
    hidden_size, rank, weight_cols
):
    """The kernel loads without masks: a direct call with a shape it does not
    tile raises instead of reading out of bounds."""
    h = torch.zeros(4, hidden_size, dtype=torch.bfloat16)
    m = torch.zeros(4, rank, dtype=torch.bfloat16)
    w = torch.zeros(1, weight_cols, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="fused_confidence needs"):
        fused_confidence(h, m, w)


def test_gate_rejects_cpu_tensors():
    h = torch.zeros(4, 512, dtype=torch.bfloat16)
    m = torch.zeros(4, 64, dtype=torch.bfloat16)
    w = torch.zeros(1, 576, dtype=torch.bfloat16)
    assert not can_fuse_confidence(h, m, w, None)
