# SPDX-License-Identifier: MIT

"""Expanding `block_tables[bs]` into the per-query-row table the MQA kernel reads.

`_attach_v4_paged_decode_meta` used to build this on the host -- zero the whole
buffer, mask the rows whose `mqa_row_to_batch` is not -1, gather those from
`block_tables`, ship the result -- and now gathers it on the device from
tensors that are already there.

That swap rests on one property of the row -> seq map: **the rows it marks
invalid are a contiguous tail**. If they were interior, `index_select` over a
prefix would fill the wrong rows and only the padding rows -- which the kernel
skips because their `csa_n_committed_per_token` is 0 -- would hide it. So the tail
property is asserted directly, for both row layouts, before the gather is
compared against the host expansion it replaces.

The gather is exercised on CPU tensors; `torch.index_select` does not care, and
the arithmetic under test is the arithmetic the device runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

COLS = 16
MAX_BS = 64


def _block_tables(bs: int) -> np.ndarray:
    """`block_tables` as the buffer holds it: `bs` live rows, the rest stale."""
    rng = np.random.default_rng(bs)
    bt = np.full((MAX_BS, COLS), -999, dtype=np.int32)  # stale rows, must not leak
    bt[:bs] = rng.integers(1, 20_000, size=(bs, COLS)).astype(np.int32)
    return bt


def _per_token_layout(bs: int, tokens_per_seq: int, t_pad: int):
    """One row per query token: the map is `batch_id_per_token`, padded with -1."""
    t = bs * tokens_per_seq
    old = np.full(t_pad, -1, dtype=np.int32)
    old[:t] = np.repeat(np.arange(bs, dtype=np.int32), tokens_per_seq)
    return old, old[:t].copy(), t


def _rect_layout(bs: int, full_q: int, rect_bs: int):
    """dspark's rectangular layout: `full_q` slots per sequence."""
    old = np.full(rect_bs * full_q, -1, dtype=np.int32)
    old[: bs * full_q] = np.repeat(np.arange(bs, dtype=np.int32), full_q)
    new = (
        torch.arange(bs * full_q, dtype=torch.int32)
        .div_(full_q, rounding_mode="floor")
        .numpy()
    )
    return old, new, bs * full_q


LAYOUTS = [
    ("per-token bs=1", _per_token_layout(1, 8, 16)),
    ("per-token bs=50", _per_token_layout(50, 8, 512)),
    ("per-token unpadded", _per_token_layout(4, 3, 12)),
    ("rect full_q=4", _rect_layout(5, 4, 8)),
    ("rect no pad", _rect_layout(6, 2, 6)),
]


@pytest.mark.parametrize("name,layout", LAYOUTS, ids=[n for n, _ in LAYOUTS])
def test_invalid_rows_are_a_contiguous_tail(name, layout):
    old_map, _, n_valid = layout
    (invalid,) = np.nonzero(old_map < 0)
    assert np.array_equal(
        invalid, np.arange(n_valid, len(old_map))
    ), "gathering over a prefix assumes every -1 row sits past the valid ones"


@pytest.mark.parametrize("name,layout", LAYOUTS, ids=[n for n, _ in LAYOUTS])
def test_device_gather_matches_the_host_expansion(name, layout):
    old_map, new_idx, n_valid = layout
    bs = int(old_map.max()) + 1
    bt = _block_tables(bs)
    rows = len(old_map)

    # What the host used to build.
    want = np.zeros((rows, COLS), dtype=np.int32)
    valid = old_map >= 0
    want[valid] = bt[:bs][old_map[valid]]

    # What the device now builds, into a buffer left dirty by a prior step.
    dst = torch.full((rows, COLS), -7, dtype=torch.int32)
    torch.index_select(
        torch.from_numpy(bt), 0, torch.from_numpy(new_idx), out=dst[:n_valid]
    )
    dst[n_valid:].zero_()

    assert np.array_equal(dst.numpy(), want)
    assert (dst.numpy()[n_valid:] == 0).all()
    assert -999 not in dst.numpy(), "a stale block_tables row reached the output"


def test_a_gather_that_ignored_the_tail_would_be_caught():
    """Positive control: the comparison above has to fail on a wrong gather.

    Without it, `test_device_gather_matches_the_host_expansion` would pass
    against an implementation that never cleared the padding rows, since those
    rows are the only place the two can differ.
    """
    old_map, new_idx, n_valid = _per_token_layout(4, 3, 20)
    bs = int(old_map.max()) + 1
    bt = _block_tables(bs)
    want = np.zeros((len(old_map), COLS), dtype=np.int32)
    want[old_map >= 0] = bt[:bs][old_map[old_map >= 0]]

    dst = torch.full((len(old_map), COLS), -7, dtype=torch.int32)
    torch.index_select(
        torch.from_numpy(bt), 0, torch.from_numpy(new_idx), out=dst[:n_valid]
    )
    # ... and here the `dst[n_valid:].zero_()` is deliberately omitted.
    assert not np.array_equal(dst.numpy(), want)


def test_index_select_takes_an_int32_index_and_a_sliced_out():
    """The two properties the call site relies on, neither of them guaranteed.

    A silent promotion to int64 would allocate per step, and an `out=` that
    rejected a row slice would force a second copy.
    """
    src = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    idx = torch.tensor([2, 0], dtype=torch.int32)
    dst = torch.zeros((3, 3), dtype=torch.int32)
    out = torch.index_select(src, 0, idx, out=dst[:2])

    assert out.dtype == torch.int32
    assert dst.tolist() == [[6, 7, 8], [0, 1, 2], [0, 0, 0]]
    assert out.data_ptr() == dst.data_ptr(), "out= must write the caller's buffer"
