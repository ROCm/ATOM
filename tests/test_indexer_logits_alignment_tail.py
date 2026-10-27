# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`top_k_per_row_prefill` must not read the alignment padding of its buffer.

`sparse_attn_indexer` calls `fp8_mqa_logits(..., clean_logits=False)`, so the
scores buffer is whatever the allocator handed back. aiter rounds its width up
to a multiple of 256, which leaves columns `[kv_len, aligned)` holding nothing
in particular -- and `stride0` is the PADDED width, so a kernel that bounded a
row by `len` rather than by `rowEnd` would scan straight into them.

`tests/test_indexer_topk_row_window.py` covers the row window itself. What it
cannot cover is the padding, because its buffers are exactly as wide as the
window. Here `kv_len` is deliberately off a multiple of 256 so the tail is
real, and it is filled with a value that wins any top-k it is read into: if a
single padding column is ever looked at, it comes back in the indices.

The multi-block kernel gets its own pass. MI355X dispatch picks the one-block
path for every batch size this test can afford (the thresholds start at
seq >= 65536), so without `TOPK_FORCE_PATH=m` that kernel is never compiled,
let alone checked -- which is how a bound bug there would reach production.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises an aiter GPU kernel; needs a real GPU", allow_module_level=True
    )

top_k_per_row_prefill = pytest.importorskip("aiter.ops.topk").top_k_per_row_prefill

DEV = "cuda"
WINS_ANY_TOPK = 1e30
ALIGN = 256


def _buffer(rows, kv_len):
    """`[rows, aligned]` poisoned everywhere, real scores only inside the rows."""
    aligned = -(-kv_len // ALIGN) * ALIGN
    assert aligned > kv_len, f"kv_len={kv_len} leaves no padding to test"
    logits = torch.full((rows, aligned), WINS_ANY_TOPK, dtype=torch.float32, device=DEV)
    return logits, aligned


def _run(logits, starts, ends, k):
    idx = torch.empty((logits.shape[0], k), dtype=torch.int32, device=DEV)
    top_k_per_row_prefill(
        logits,
        starts,
        ends,
        idx,
        None,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k=k,
    )
    return idx


@pytest.mark.parametrize("kv_len", [3000, 4097, 8191, 16385])
@pytest.mark.parametrize("rows", [1, 5, 64])
@pytest.mark.parametrize("force_path", [None, "m"])
def test_no_index_lands_outside_the_row_window(kv_len, rows, force_path, monkeypatch):
    if force_path is not None:
        monkeypatch.setenv("TOPK_FORCE_PATH", force_path)
    k = 64
    logits, aligned = _buffer(rows, kv_len)

    # Give every row a different window, all strictly inside [0, kv_len).
    starts = torch.tensor(
        [(i * 37) % max(1, kv_len - k - 1) for i in range(rows)],
        dtype=torch.int32,
        device=DEV,
    )
    ends = torch.clamp(starts + k + 11, max=kv_len).to(torch.int32)
    for i in range(rows):
        s, e = int(starts[i]), int(ends[i])
        logits[i, s:e] = torch.linspace(0.0, 1.0, e - s, device=DEV)

    idx = _run(logits, starts, ends, k)

    for i in range(rows):
        s, e = int(starts[i]), int(ends[i])
        sel = idx[i][idx[i] >= 0]
        assert sel.numel() > 0
        assert int(sel.min()) >= s, f"row {i}: index {int(sel.min())} < start {s}"
        assert int(sel.max()) < e, f"row {i}: index {int(sel.max())} >= end {e}"
        assert int(sel.max()) < kv_len, (
            f"row {i}: index {int(sel.max())} reached the alignment padding "
            f"[{kv_len}, {aligned})"
        )


def test_the_poison_is_actually_reachable():
    """The control: widen one row's window over the padding and it comes back.

    Without this, a kernel that returned nothing but sentinels would satisfy
    every assertion above and the suite would be measuring its own absence.
    """
    kv_len, k = 4097, 8
    logits, aligned = _buffer(1, kv_len)
    logits[0, :kv_len] = 0.0
    starts = torch.tensor([0], dtype=torch.int32, device=DEV)
    ends = torch.tensor([aligned], dtype=torch.int32, device=DEV)  # over the padding

    idx = _run(logits, starts, ends, k)
    sel = idx[0][idx[0] >= 0]
    assert int(sel.max()) >= kv_len, (
        "the padding columns hold the only non-zero scores, so a kernel that "
        "reads them must select them -- if this fails the poison is not poison "
        "and the tests above prove nothing"
    )


def test_the_forced_pass_is_a_second_kernel(monkeypatch):
    """`TOPK_FORCE_PATH=m` has to actually reach the multi-block kernel.

    Without this the `force_path="m"` cases above would be the one-block kernel
    run twice, and would prove nothing about the path that a bound bug could
    hide in. aiter exposes the dispatch predicate, so ask it directly rather
    than inferring from the results -- an earlier version of this test compared
    run-to-run stability and was wrong, because neither path is deterministic.
    """
    from aiter.ops.topk import topk_use_mulblocks

    rows, stride0 = 5, 8448  # kv_len=8191 rounded to 256, one of the cases above
    assert not topk_use_mulblocks(rows, stride0), (
        "dispatch already selects the multi-block kernel at this shape, so "
        "forcing it adds no coverage -- pick a shape where it does not"
    )
    monkeypatch.setenv("TOPK_FORCE_PATH", "m")
    assert topk_use_mulblocks(rows, stride0), (
        "TOPK_FORCE_PATH is not honoured by this aiter build; the force_path "
        "cases above are the one-block kernel twice"
    )
