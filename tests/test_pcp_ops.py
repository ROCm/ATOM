# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU unit tests for the Prefill Context Parallel (PCP) pure helpers.

PCP's failure mode is not a wrong number, it is two places disagreeing about
which rows a rank owns. The draft-prefill crash this suite starts from was
exactly that: `compute_draft_kv` fed full-size q against attn_metadata already
reindexed to 1/pcp, and aiter aborted on `kv_indptr_prefix length must be N+1`.
So the tests here pin the *conventions* --- pad first, then round-robin; dense
pads with zeros while indptr repeats the last prefix-sum --- and, where two
functions have to agree, assert them against each other rather than separately.

Everything here is pure tensor work plus parallel-state reads that are
monkeypatched, so it runs on CPU with no process group.

Groups:
  * draft prefill split  --- `_pcp_split_draft_inputs`
  * row ownership        --- `pcp_round_robin_split` vs `pcp_round_robin_query_indices`
  * metadata padding     --- `pcp_pad_dense` vs `pcp_pad_indptr`
  * ragged reindex       --- `pcp_reindex_ragged`
"""

import pytest
import torch

try:
    import atom.distributed.pcp_utils as pu
    from atom.spec_decode import eagle_proposer as ep

    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - env without aiter/triton
    _IMPORT_ERR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERR is not None,
    reason=f"requires full atom import env: {_IMPORT_ERR}",
)

HID = 4
WS = 4


def _inputs(n):
    """Row i carries the value i everywhere, so a shard's provenance is readable."""
    ids = torch.arange(n, dtype=torch.int32)
    pos = torch.arange(n, dtype=torch.int32)
    hid = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, HID)
    return ids, pos, hid


def _split(monkeypatch, ids, pos, hid, rank, ws=WS):
    """Run the helper as PCP rank `rank` of `ws`.

    `get_pcp_world_size` is read by the helper itself (eagle_proposer's
    namespace); `get_pcp_rank` is read by `pcp_round_robin_split` further down
    (pcp_utils' namespace). The sizes are passed explicitly by the helper, so
    only the rank lookup needs patching on the callee side.
    """
    monkeypatch.setattr(ep, "get_pcp_world_size", lambda: ws)
    monkeypatch.setattr(pu, "get_pcp_rank", lambda: rank)
    return ep._pcp_split_draft_inputs(ids, pos, hid)


def test_pads_to_a_multiple_and_splits_round_robin(monkeypatch):
    """n_global=10, pcp=4 -> pad to 12 -> 3 rows per rank, stride 4."""
    ids, pos, hid = _inputs(10)
    seen = {}
    for rank in range(WS):
        with monkeypatch.context() as m:
            d_ids, d_pos, d_hid, n_global, ws = _split(m, ids, pos, hid, rank)
        assert (n_global, ws) == (10, WS)
        assert d_ids.shape[0] == 3, f"rank {rank} got {d_ids.shape[0]} rows"
        assert d_pos.shape[0] == 3 and d_hid.shape == (3, HID)
        seen[rank] = d_ids.tolist()

    # Round-robin, NOT contiguous chunks -- this is the convention the metadata
    # builder reindexes to, so a chunked split would silently mismatch it.
    assert seen[0] == [0, 4, 8]
    assert seen[1] == [1, 5, 9]
    assert seen[2] == [2, 6, 0]  # global row 10 is pad -> zero
    assert seen[3] == [3, 7, 0]  # global row 11 is pad -> zero

    # Every real row is owned by exactly one rank.
    real = [v for rows in seen.values() for v in rows]
    for row in range(10):
        assert real.count(row) == 1 or row == 0, f"row {row} owned {real.count(row)}x"


def test_all_three_tensors_shard_identically(monkeypatch):
    """ids / positions / hidden must stay row-aligned after the split."""
    ids, pos, hid = _inputs(10)
    d_ids, d_pos, d_hid, _, _ = _split(monkeypatch, ids, pos, hid, rank=2)
    assert d_ids.tolist() == d_pos.tolist()
    assert d_hid[:, 0].to(torch.int32).tolist() == d_ids.tolist()
    assert torch.equal(d_hid[:, 0], d_hid[:, -1])


def test_exact_multiple_needs_no_pad(monkeypatch):
    ids, pos, hid = _inputs(8)
    d_ids, _, _, n_global, _ = _split(monkeypatch, ids, pos, hid, rank=3)
    assert n_global == 8
    assert d_ids.tolist() == [3, 7]


def test_pcp1_is_a_no_op(monkeypatch):
    ids, pos, hid = _inputs(5)
    d_ids, d_pos, d_hid, n_global, ws = _split(monkeypatch, ids, pos, hid, rank=0, ws=1)
    assert (n_global, ws) == (5, 1)
    assert d_ids.tolist() == list(range(5))
    assert d_hid.shape == (5, HID)


def test_row_misalignment_is_rejected(monkeypatch):
    """The assert that turns a silent miscompute into a crash.

    PCP+TBO rewrites `context.positions` into per-request-group local stripes
    while ids/hidden stay global-length. Without this check the helper derives
    `n_pad` from the long tensor and pads the short one by it, producing
    garbage; now it refuses. (That combination is additionally rejected at
    startup in llm_engine.)
    """
    ids, pos, hid = _inputs(10)
    monkeypatch.setattr(ep, "get_pcp_world_size", lambda: WS)
    with pytest.raises(AssertionError, match="row-aligned"):
        ep._pcp_split_draft_inputs(ids, pos[:4], hid)
    with pytest.raises(AssertionError, match="row-aligned"):
        ep._pcp_split_draft_inputs(ids, pos, hid[:4])


# ---------------------------------------------------------------- ownership --
# `pcp_round_robin_split` extracts this rank's DATA rows; the metadata builder
# uses `pcp_round_robin_query_indices` to decide which METADATA rows to keep.
# They are separate functions with separate implementations (`view()[:, rank]`
# vs `arange(rank, n, pcp)`), and the whole class of bug this suite exists for
# is the two disagreeing. Assert them against each other, not just separately.


@pytest.mark.parametrize("n_global,ws", [(12, 4), (8, 4), (16, 8), (6, 2), (5, 1)])
def test_split_and_query_indices_agree_on_ownership(monkeypatch, n_global, ws):
    data = torch.arange(n_global, dtype=torch.int32)
    for rank in range(ws):
        monkeypatch.setattr(pu, "get_pcp_rank", lambda r=rank: r)
        rows = pu.pcp_round_robin_split(data, ws).tolist()
        owned = pu.pcp_round_robin_query_indices(n_global, ws, rank).tolist()
        assert rows == owned, f"rank {rank}/{ws}: data {rows} != metadata {owned}"


def test_ownership_partitions_every_row_exactly_once(monkeypatch):
    n_global, ws = 12, 4
    seen = []
    for rank in range(ws):
        seen += pu.pcp_round_robin_query_indices(n_global, ws, rank).tolist()
    assert sorted(seen) == list(range(n_global))


def test_pad_len_rounds_up_to_a_pcp_multiple():
    assert pu.pcp_pad_len(10, 4) == 12
    assert pu.pcp_pad_len(12, 4) == 12  # already aligned, no growth
    assert pu.pcp_pad_len(1, 8) == 8
    assert pu.pcp_pad_len(10, 1) == 10  # pcp=1 is a no-op
    # `multiple` stacks on top of pcp_size (used when a kernel also wants a
    # tile-aligned row count).
    assert pu.pcp_pad_len(10, 4, multiple=2) == 16


# ------------------------------------------------------------------ padding --
# pcp_pad_dense and pcp_pad_indptr share a (tensor, n_pad) signature but pad two
# different shapes, and pcp_utils carries a long comment warning not to confuse
# them. Pin the distinction: a padded dummy query must end up with an EMPTY KV
# segment, which only happens if indptr repeats its last prefix-sum value.


def test_pad_dense_appends_zero_rows():
    t = torch.tensor([[5.0, 5.0], [3.0, 3.0]])
    out = pu.pcp_pad_dense(t, 2)
    assert out.shape == (4, 2)
    assert torch.equal(out[:2], t)
    assert torch.count_nonzero(out[2:]) == 0


def test_pad_indptr_gives_dummy_queries_empty_segments():
    # kv_indptr [0,2,5,6] over kv_indices [a,b | c,d,e | f]
    kv_indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    kv_indices = torch.arange(6, dtype=torch.int32)
    padded = pu.pcp_pad_indptr(kv_indptr, 1)
    assert padded.tolist() == [0, 2, 5, 6, 6]
    # the dummy query's slice is empty -- it contributes nothing to attention
    lo, hi = int(padded[3]), int(padded[4])
    assert kv_indices[lo:hi].numel() == 0
    # and the real segments are untouched
    assert kv_indices[int(padded[0]) : int(padded[1])].tolist() == [0, 1]


def test_pad_helpers_are_no_ops_at_zero():
    t = torch.arange(3, dtype=torch.int32)
    assert pu.pcp_pad_dense(t, 0) is t
    assert pu.pcp_pad_indptr(t, 0) is t


# ----------------------------------------------------------- ragged reindex --
# `pcp_reindex_ragged` compacts per-query ragged metadata down to this rank's
# queries via a searchsorted gather map. Its docstring states the invariant
# exactly, so test that rather than hand-computed expected arrays.


def _ragged(lens):
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0, out=indptr[1:])
    indices = torch.arange(int(indptr[-1]), dtype=torch.int32) + 100
    return indptr, indices


@pytest.mark.parametrize("lens", [[2, 3, 1, 4, 0, 2], [1, 1, 1, 1], [0, 0, 5, 0]])
@pytest.mark.parametrize("rank,ws", [(0, 2), (1, 2), (2, 3)])
def test_reindex_ragged_preserves_each_owned_segment(lens, rank, ws):
    kv_indptr, kv_indices = _ragged(lens)
    owned = pu.pcp_round_robin_query_indices(len(lens), ws, rank)
    ip_local, idx_local = pu.pcp_reindex_ragged(kv_indptr, kv_indices, owned)

    assert ip_local.shape[0] == owned.shape[0] + 1
    assert int(ip_local[0]) == 0
    for i, g in enumerate(owned.tolist()):
        got = idx_local[int(ip_local[i]) : int(ip_local[i + 1])].tolist()
        want = kv_indices[int(kv_indptr[g]) : int(kv_indptr[g + 1])].tolist()
        assert got == want, f"owned query {i} (global {g}): {got} != {want}"


def test_reindex_ragged_handles_an_all_empty_shard():
    # every owned query has a zero-length segment -> the `total == 0` fast path
    kv_indptr, kv_indices = _ragged([0, 3, 0, 4])
    owned = torch.tensor([0, 2], dtype=torch.long)  # both empty
    ip_local, idx_local = pu.pcp_reindex_ragged(kv_indptr, kv_indices, owned)
    assert ip_local.tolist() == [0, 0, 0]
    assert idx_local.numel() == 0
