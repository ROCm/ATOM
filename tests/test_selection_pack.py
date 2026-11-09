# SPDX-License-Identifier: MIT
"""`pack_row_selections` against a host reference.

A sequence with more than one token (an MTP verify) puts other rows'
selections between its chosen ones, so the packed layout differs from the
source's."""

import numpy as np
import pytest
import torch

try:
    from atom.model_ops.attentions.selection_pack import pack_row_selections
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the selection pack is a Triton kernel")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton on GPU")

WIDTH = 512


def _case(num_seqs, tokens_per_seq, rng):
    lens = rng.integers(0, WIDTH + 1, size=num_seqs * tokens_per_seq)
    indptr = np.zeros(len(lens) + 1, dtype=np.int32)
    np.cumsum(lens, out=indptr[1:])
    indices = rng.integers(0, 1 << 20, size=int(indptr[-1]) + WIDTH, dtype=np.int32)
    rows = np.arange(num_seqs) * tokens_per_seq + tokens_per_seq - 1
    return rows, indptr, indices


def _pack(rows, indptr, indices, num_out):
    out_indptr = torch.full((num_out + 1,), -7, dtype=torch.int32, device="cuda")
    out_indices = torch.full((num_out * WIDTH,), -7, dtype=torch.int32, device="cuda")
    pack_row_selections(
        torch.from_numpy(rows).cuda(),
        torch.from_numpy(indptr).cuda(),
        torch.from_numpy(indices).cuda(),
        out_indptr,
        out_indices,
        WIDTH,
    )
    return out_indptr.cpu().numpy(), out_indices.cpu().numpy()


def _expected(rows, indptr, indices):
    starts, ends = indptr[rows], indptr[rows + 1]
    packed = np.concatenate([indices[s:e] for s, e in zip(starts, ends)])
    expect_indptr = np.zeros(len(rows) + 1, dtype=np.int32)
    np.cumsum(ends - starts, out=expect_indptr[1:])
    return expect_indptr, packed


@pytest.mark.parametrize("num_seqs", [1, 7, 256, 600])
@pytest.mark.parametrize("tokens_per_seq", [1, 2, 4])
def test_packs_each_rows_selection_in_sequence_order(num_seqs, tokens_per_seq):
    rng = np.random.default_rng(num_seqs * 10 + tokens_per_seq)
    rows, indptr, indices = _case(num_seqs, tokens_per_seq, rng)
    out_indptr, out_indices = _pack(rows, indptr, indices, num_seqs)
    expect_indptr, packed = _expected(rows, indptr, indices)
    assert out_indptr.tolist() == expect_indptr.tolist()
    assert out_indices[: len(packed)].tolist() == packed.tolist()


@pytest.mark.parametrize("tokens_per_seq", [1, 3])
def test_padding_rows_after_the_real_ones_leave_them_packed(tokens_per_seq):
    rng = np.random.default_rng(tokens_per_seq)
    real, padding = 37, 27
    rows, indptr, indices = _case(real, tokens_per_seq, rng)
    out_indptr, out_indices = _pack(
        np.pad(rows, (0, padding)), indptr, indices, real + padding
    )
    expect_indptr, packed = _expected(rows, indptr, indices)
    assert out_indptr[: real + 1].tolist() == expect_indptr.tolist()
    assert out_indices[: len(packed)].tolist() == packed.tolist()


def test_the_sources_are_left_as_they_were():
    rng = np.random.default_rng(0)
    rows, indptr, indices = _case(300, 2, rng)
    indptr_d = torch.from_numpy(indptr).cuda()
    indices_d = torch.from_numpy(indices).cuda()
    pack_row_selections(
        torch.from_numpy(rows).cuda(),
        indptr_d,
        indices_d,
        torch.empty(301, dtype=torch.int32, device="cuda"),
        torch.empty(300 * WIDTH, dtype=torch.int32, device="cuda"),
        WIDTH,
    )
    assert indptr_d.cpu().tolist() == indptr.tolist()
    assert indices_d.cpu().tolist() == indices.tolist()
