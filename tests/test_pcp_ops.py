# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU unit tests for the Prefill Context Parallel (PCP) pure helpers.

PCP fails when two places disagree about which rows a rank owns, so these pin
the conventions (pad first, then round-robin; dense pads with zeros, padding
queries get empty segments) and assert paired helpers against each other.
CPU-only: parallel-state reads are monkeypatched.

Groups:
  * draft prefill split  --- `_pcp_split_draft_inputs`
  * row ownership        --- `pcp_round_robin_split` vs `pcp_round_robin_query_indices`
  * metadata padding     --- `pcp_pad_dense`
  * V4 prefill indices   --- `_apply_pcp_reindex` and the PCP+TBO group build
                             vs the whole-batch build
  * grouped gather       --- `pcp_allgather_rerange_many`
  * DSpark context KV    --- `write_split_context_kv`, and every draft's
                             `write_context_kv` taking `pcp_num_tokens`
"""

import types
from types import SimpleNamespace

import numpy as np
import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

# atom.config first, so a circular import on the mainline path cannot be
# mistaken for an absent dependency below.
try:
    import atom.config  # noqa: F401  (import order, see above)
    import atom.distributed.pcp_utils as pu
    from atom.spec_decode import eagle_proposer as ep
except ImportError as _e:  # pragma: no cover - env without aiter/triton
    # Not `except Exception`: that is the exact mistake import_guard.py was
    # written about -- it swallows a SyntaxError in the module under test and
    # reports the silence as a skip.
    skip_if_dependency_missing(_e, "requires full atom import env")

HID = 4
WS = 4


def _import_or_skip_dep(name):
    """Skip only for a missing third-party dependency, so a renamed or broken
    atom module fails instead of skipping."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as e:
        skip_if_dependency_missing(e, name)


def _inputs(n):
    """Row i carries i+1, so provenance is readable and 0 means padding."""
    ids = torch.arange(1, n + 1, dtype=torch.int32)
    pos = torch.arange(1, n + 1, dtype=torch.int32)
    hid = torch.arange(1, n + 1, dtype=torch.float32)[:, None].repeat(1, HID)
    return ids, pos, hid


def _split(monkeypatch, ids, pos, hid, rank, ws=WS):
    """Run the helper as PCP rank `rank` of `ws` (world size read in
    eagle_proposer, rank in pcp_utils)."""
    monkeypatch.setattr(ep, "get_pcp_world_size", lambda: ws)
    monkeypatch.setattr(pu, "get_pcp_rank", lambda: rank)
    # The helper flags the pass as split for FusedMoE's PCP gather.
    ctx = SimpleNamespace(context=SimpleNamespace(pcp_split=False))
    monkeypatch.setattr(ep, "get_forward_context", lambda: ctx)
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
    assert seen[0] == [1, 5, 9]
    assert seen[1] == [2, 6, 10]
    assert seen[2] == [3, 7, 0]  # global row 10 is pad -> zero
    assert seen[3] == [4, 8, 0]  # global row 11 is pad -> zero

    # Every real row is owned by exactly one rank -- no exemptions, since 0 is
    # now only ever padding.
    real = [v for rows in seen.values() for v in rows]
    for row in range(1, 11):
        assert real.count(row) == 1, f"row {row} owned {real.count(row)}x"
    assert real.count(0) == 2, "exactly the two pad rows"


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
    assert d_ids.tolist() == [4, 8]


def test_pcp1_is_a_no_op(monkeypatch):
    ids, pos, hid = _inputs(5)
    d_ids, _, d_hid, n_global, ws = _split(monkeypatch, ids, pos, hid, rank=0, ws=1)
    assert (n_global, ws) == (5, 1)
    assert d_ids.tolist() == list(range(1, 6))
    assert d_hid.shape == (5, HID)


def test_row_misalignment_is_rejected(monkeypatch):
    """Positions shorter than ids/hidden (PCP+TBO) must be refused, not
    padded by the wrong amount."""
    ids, pos, hid = _inputs(10)
    monkeypatch.setattr(ep, "get_pcp_world_size", lambda: WS)
    ctx = SimpleNamespace(context=SimpleNamespace(pcp_split=False))
    monkeypatch.setattr(ep, "get_forward_context", lambda: ctx)
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


@pytest.mark.parametrize("rows_per_rank", [1, 2])
def test_split_shards_have_unit_strides(rows_per_rank):
    """A one-row shard counts as contiguous at any stride, but aiter's
    cached-positions RoPE checks `positions.view(1, n).stride(1) == 1` itself
    and aborts: a 2-token chunk under PCP8 leaves every rank one row."""
    n = rows_per_rank * WS
    pos = pu.pcp_round_robin_split(torch.arange(n), WS, 1)
    hid = pu.pcp_round_robin_split(torch.zeros(n, HID), WS, 1)
    assert pos.stride() == (1,) and pos.view(1, -1).stride(1) == 1
    assert hid.stride() == (HID, 1)


def test_pad_len_rounds_up_to_a_pcp_multiple():
    assert pu.pcp_pad_len(10, 4) == 12
    assert pu.pcp_pad_len(12, 4) == 12  # already aligned, no growth
    assert pu.pcp_pad_len(1, 8) == 8
    assert pu.pcp_pad_len(10, 1) == 10  # pcp=1 is a no-op
    # `multiple` stacks on top of pcp_size (used when a kernel also wants a
    # tile-aligned row count).
    assert pu.pcp_pad_len(10, 4, multiple=2) == 16


# ------------------------------------------------------------------ padding --


def test_pad_dense_appends_zero_rows():
    t = torch.tensor([[5.0, 5.0], [3.0, 3.0]])
    out = pu.pcp_pad_dense(t, 2)
    assert out.shape == (4, 2)
    assert torch.equal(out[:2], t)
    assert torch.count_nonzero(out[2:]) == 0


def test_pad_dense_is_a_no_op_at_zero():
    t = torch.arange(3, dtype=torch.int32)
    assert pu.pcp_pad_dense(t, 0) is t


# ------------------------------------------- V4 paged prefill indices split --
# Under PCP the builder builds the paged prefill indices for this rank's
# queries only, from their own positions and requests. That must equal the
# whole batch's build taken query by query; padding queries read nothing.

_V4_LENS = np.array([5, 7, 4], dtype=np.int32)
_V4_STARTS = np.array([0, 130, 300], dtype=np.int32)  # 130, 300: prefix hits
_V4_FLOOR = np.array([0, 128, 0], dtype=np.int32)  # request 1 replays
_V4_FIELDS = ("extend", "prefix_swa", "prefix_csa", "prefix_hca")


def _v4_builder(monkeypatch):
    from atom.model_ops.attentions import deepseek_v4_attn as v4
    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        CSA_RATIO,
        DENSE_RATIO,
        HCA_RATIO,
        UnifiedPoolGeometry,
    )
    from atom.model_ops.v4_kernels.paged_prefill_indices import (
        write_v4_paged_prefill_indices_reference,
    )

    monkeypatch.setattr(
        v4, "write_v4_paged_prefill_indices", write_v4_paged_prefill_indices_reference
    )
    cls = v4.DeepseekV4AttentionMetadataBuilder
    var = {}

    def stage(name, arr, pad_to=None, *, publication_group=None):
        # A fresh buffer per call: metadata built earlier keeps its views.
        n = arr.shape[0]
        buf = np.zeros(max(n, pad_to or 0), dtype=arr.dtype)
        buf[:n] = arr
        var[name] = SimpleNamespace(np=buf, gpu=torch.from_numpy(buf))
        return var[name].gpu[:n]

    builder = SimpleNamespace(
        device="cpu",
        window_size=8,
        index_topk=4,
        num_state_slots=4,
        pool_geometry=UnifiedPoolGeometry(
            [DENSE_RATIO, CSA_RATIO, HCA_RATIO],
            num_blocks=40,
            num_slots=4,
            ring_slots=11,
            block_size=256,
        ),
        hca_rows_per_block=2,
        model_runner=SimpleNamespace(forward_vars=var),
        _indexer_fp4=False,
        _stage=stage,
        _pcp_split_rows=cls._pcp_split_rows,
        _attach_v4_index_rows=lambda *a, **k: None,
        _stage_prefill_index_keys=lambda *a, **k: None,
    )
    for name in ("_build_paged_prefill_meta", "_stage_paged_prefill"):
        setattr(builder, name, getattr(cls, name).__get__(builder))
    return v4, cls, builder


def _v4_batch(reqs):
    """Requests `reqs` of the batch, rebased to their own token axis."""
    lens, starts = _V4_LENS[reqs], _V4_STARTS[reqs]
    cu = np.r_[0, np.cumsum(lens)].astype(np.int32)
    positions = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)])
    slots = torch.arange(3, dtype=torch.int32)[reqs]
    md = SimpleNamespace(
        batch_id_per_q_token=torch.from_numpy(
            np.repeat(np.arange(len(lens), dtype=np.int32), lens)
        ),
        state_slot_out=slots,
        state_slot_out_cpu=slots.numpy(),
        replay_start_cpu=_V4_FLOOR[reqs],
        n_committed_csa_per_seq_cpu=None,
    )
    kw = {
        "cu_q_per_seq_gpu": torch.from_numpy(cu[:-1]),
        "block_tables_gpu": (torch.arange(24, dtype=torch.int32).view(3, 8) + 1)[reqs],
    }
    return md, positions, cu, kw


def _stage_v4_batch(builder, positions, running=None, **kw):
    n = len(positions)
    builder._stage_paged_prefill(
        positions, _V4_LENS, _V4_STARTS, 3, n, running or n, _V4_FLOOR, **kw
    )


def _segment(md, field, q):
    indptr = getattr(md, f"kv_indptr_{field}")
    return getattr(md, f"kv_indices_{field}")[int(indptr[q]) : int(indptr[q + 1])]


def _v4_whole_batch(monkeypatch):
    """The builder, and the indices it builds for the whole batch unsplit."""
    v4, cls, builder = _v4_builder(monkeypatch)
    md, positions, _cu, kw = _v4_batch([0, 1, 2])
    _stage_v4_batch(builder, positions)
    builder._build_paged_prefill_meta(
        md,
        3,
        len(positions),
        len(positions),
        positions_gpu=torch.from_numpy(positions),
        **kw,
    )
    skip = md.skip_prefix_len_csa
    assert skip.max() > 0 and md.kv_indices_prefix_hca.numel() > skip.sum()
    return v4, cls, builder, md


def _assert_rank_segments(local, whole, n, ws, rank, first=0, running=None):
    """`local` holds `rank`'s round-robin share of the `n` tokens that start at
    token `first` of `whole` (padded from `running`), and its `kv` holds only
    those tokens."""
    skip = whole.skip_prefix_len_csa
    owned = range(rank, pu.pcp_pad_len(running or n, ws), ws)
    assert local.kv_indptr_extend.shape[0] == len(owned) + 1
    assert local.skip_prefix_len_csa.shape[0] == len(owned)
    for i, q in enumerate(owned):
        if q >= n:  # padding query
            assert local.skip_prefix_len_csa[i] == 0
            assert all(_segment(local, f, i).numel() == 0 for f in _V4_FIELDS)
            continue
        g = first + q
        assert local.skip_prefix_len_csa[i] == skip[g]
        for field in _V4_FIELDS:
            got, want = _segment(local, field, i), _segment(whole, field, g)
            assert got.shape == want.shape, (field, rank, q)
            if field == "extend":
                want = want - first
            elif field == "prefix_csa":
                # The head is the per-layer CSA top-k, written later.
                got, want = (
                    got[got.numel() - skip[g] :],
                    want[want.numel() - skip[g] :],
                )
            assert torch.equal(got, want), (field, rank, q)


@pytest.mark.parametrize("pad", [0, 2])  # running tokens beyond the scheduled ones
@pytest.mark.parametrize("ws", [2, 3, 4])
def test_v4_rank_paged_prefill_indices_match_the_whole_batch(monkeypatch, ws, pad):
    v4, cls, builder, whole = _v4_whole_batch(monkeypatch)
    monkeypatch.setattr(v4, "get_pcp_world_size", lambda: ws)
    for rank in range(ws):
        monkeypatch.setattr(v4, "get_pcp_rank", lambda r=rank: r)
        local, pos_l, cu_l, kw_l = _v4_batch([0, 1, 2])
        n = len(pos_l)
        _stage_v4_batch(builder, pos_l, n + pad, pcp_split=True)
        # Runner buffers span the running tokens; the tail is not scheduled.
        bid = local.batch_id_per_q_token
        local.batch_id_per_q_token = torch.cat([bid, bid.new_full((pad,), -1)])
        pos_run = torch.from_numpy(np.r_[pos_l, np.zeros(pad, pos_l.dtype)])
        cls._apply_pcp_reindex(builder, local, pos_run, 3, 3, n, n + pad, cu_l, **kw_l)
        _assert_rank_segments(local, whole, n, ws, rank, running=n + pad)


def _v4_tbo_step(builder, v4):
    """The whole batch as a PCP+TBO step: its groups, and the step metadata
    with each group's base metadata (`stage_ubatch_prefill`) hung on it."""
    from atom.utils.forward_context import AttentionMetaData

    md, positions, cu, kw = _v4_batch([0, 1, 2])
    groups = [
        pu.PcpBalGroup(rs0, rs1, int(cu[rs0]), int(cu[rs1]), 0)
        for rs0, rs1 in ((0, 1), (1, 3))
    ]
    full = AttentionMetaData(cu_seqlens_q=torch.from_numpy(cu), has_cached=False)
    full.__class__ = v4.AttentionMetaData_DSV4
    full.batch_id_per_q_token = md.batch_id_per_q_token
    full.state_slot_out = md.state_slot_out
    full.state_slot_out_cpu = md.state_slot_out_cpu
    full.replay_start_cpu = _V4_FLOOR
    full.replay_end_cpu = np.array([0, 132, 0], dtype=np.int32)
    full.ubatches = []
    for g in groups:
        g_md, _, g_cu, _ = _v4_batch(list(range(g.req_start, g.req_stop)))
        ub = AttentionMetaData(cu_seqlens_q=torch.from_numpy(g_cu), has_cached=False)
        ub.batch_id_per_q_token = g_md.batch_id_per_q_token
        full.ubatches.append(ub)
    builder.model_runner.forward_vars.update(
        {
            "cu_seqlens_q": SimpleNamespace(np=cu),
            "context_lens": SimpleNamespace(np=_V4_STARTS + _V4_LENS),
            "positions": SimpleNamespace(np=positions, gpu=torch.from_numpy(positions)),
            "block_tables": SimpleNamespace(gpu=kw["block_tables_gpu"]),
        }
    )
    return full, groups


# ws 6 / 8: the 5-token group leaves 1 / 3 ranks with no real query.
@pytest.mark.parametrize("ws", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("ubatch_idx", [0, 1])
def test_v4_tbo_group_paged_prefill_indices_match_the_whole_batch(
    monkeypatch, ws, ubatch_idx
):
    """PCP+TBO: each request group gets its own requests' block tables, query
    offsets and SWA floors, and its extend rows count from its first token."""
    v4, cls, builder, whole = _v4_whole_batch(monkeypatch)
    builder._build_compress_plans = lambda *a, **k: {}
    builder._apply_pcp_reindex = cls._apply_pcp_reindex.__get__(builder)
    monkeypatch.setattr(v4, "get_pcp_world_size", lambda: ws)
    for rank in range(ws):
        monkeypatch.setattr(v4, "get_pcp_rank", lambda r=rank: r)
        full, groups = _v4_tbo_step(builder, v4)
        grp = groups[ubatch_idx]
        full.ubatch_compress_plans = [
            cls._stage_pcp_group(builder, full, i, g, None)
            for i, g in enumerate(groups)
        ]
        ub = cls._finish_pcp_group(builder, full, ubatch_idx, grp)
        n = grp.tok_end - grp.tok_start
        _assert_rank_segments(ub, whole, n, ws, rank, first=grp.tok_start)


def test_v4_tbo_group_plans_are_the_whole_batch_plans_args(monkeypatch):
    """A PCP+TBO group's compression plans are built as the whole batch's
    (`stage_prefill`): its requests' lengths, no rejection slack, and the
    replay floors."""
    v4, cls, builder = _v4_builder(monkeypatch)
    builder.max_spec_steps = 3
    calls = []
    builder._build_compress_plans = lambda *a, **k: calls.append((a, k)) or {}
    full, groups = _v4_tbo_step(builder, v4)
    monkeypatch.setattr(v4, "get_pcp_world_size", lambda: 2)
    monkeypatch.setattr(v4, "get_pcp_rank", lambda: 0)
    for i, g in enumerate(groups):
        calls.clear()
        cls._stage_pcp_group(builder, full, i, g, None)
        (ext, ctx), kw = calls[0]
        rs = slice(g.req_start, g.req_stop)
        np.testing.assert_array_equal(ext, _V4_LENS[rs])
        np.testing.assert_array_equal(ctx, (_V4_STARTS + _V4_LENS)[rs])
        assert kw["extra_write"] == 0
        np.testing.assert_array_equal(kw["write_from"], full.replay_end_cpu[rs])


def _bytes(t):
    return t.contiguous().view(torch.uint8)


class _FakePcpGroup:
    """One process standing in for every PCP rank: a gather returns, per
    tensor, the rank-major concatenation of what each rank would pass in
    (`shards[rank][i]`), and is recorded. Like aiter's PyNCCL fallback, it
    takes only the dtypes NCCL has."""

    def __init__(self, shards, rank):
        self.shards, self.rank, self.calls = shards, rank, []
        self.device_communicator = self

    def all_gatherv(self, inputs, dim=0, sizes=None):
        from aiter.dist.device_communicators.pynccl_wrapper import ncclDataTypeEnum

        assert dim == 0 and sizes is None
        self.calls.append(len(inputs))
        for i, t in enumerate(inputs):
            ncclDataTypeEnum.from_torch(t.dtype)
            assert torch.equal(_bytes(t), _bytes(self.shards[self.rank][i]))
        return [
            torch.cat([s[i].view(t.dtype) for s in self.shards])
            for i, t in enumerate(inputs)
        ]


@pytest.mark.parametrize("ws", [2, 4, 6, 8])
def test_grouped_gather_restores_every_tensor_in_one_collective(monkeypatch, ws):
    """The extend K (fp8 bytes + bf16 RoPE tail) and its positions travel in
    one collective, and each comes back in global token order."""
    n = 5 * ws
    fulls = [
        torch.randint(0, 255, (n, 1, 8), dtype=torch.uint8).view(torch.float8_e4m3fn),
        torch.randn(n, 1, 4).to(torch.bfloat16),
        torch.arange(n, dtype=torch.int64) + 100,
    ]
    shards = [[pu.pcp_round_robin_split(t, ws, r) for t in fulls] for r in range(ws)]
    for rank in range(ws):
        group = _FakePcpGroup(shards, rank)
        monkeypatch.setattr(pu, "get_pcp_group", lambda g=group: g)
        got = pu.pcp_allgather_rerange_many(shards[rank], ws)
        assert group.calls == [len(fulls)]
        for g, f in zip(got, fulls):
            assert g.dtype == f.dtype and torch.equal(_bytes(g), _bytes(f))
        # The single-tensor gather is the same transport.
        group = _FakePcpGroup([[s[2]] for s in shards], rank)
        monkeypatch.setattr(pu, "get_pcp_group", lambda g=group: g)
        assert torch.equal(pu.pcp_allgather_rerange(shards[rank][2], ws), fulls[2])
        assert group.calls == [1]


def test_needs_id_sync_is_one_predicate_for_both_groups():
    """PCP and TP gate their id broadcast on the same question."""
    mr = _import_or_skip_dep("atom.model_engine.model_runner")
    assert mr._needs_id_sync(False, False) is False
    assert mr._needs_id_sync(True, False) is True
    assert mr._needs_id_sync(False, True) is True
    assert mr._needs_id_sync(True, True) is True


def test_settle_target_pcp_split_snapshots_pcp_split(monkeypatch):
    """Both forwards call it bare right after run_model (so it must not have
    picked up `forward`'s decorators) to keep the target's `pcp_split`."""
    mr = _import_or_skip_dep("atom.model_engine.model_runner")
    assert hasattr(mr.ModelRunner.forward, "__wrapped__")
    assert not hasattr(mr.ModelRunner._settle_target_pcp_split, "__wrapped__")
    ctx = SimpleNamespace(pcp_split=True, target_pcp_split=None)
    monkeypatch.setattr(mr, "get_forward_context", lambda: SimpleNamespace(context=ctx))
    monkeypatch.setattr(mr, "pcp_is_enabled", lambda: True)
    runner = SimpleNamespace(drafter=object())
    mr.ModelRunner._settle_target_pcp_split(runner)
    assert ctx.target_pcp_split is True
    ctx.pcp_split = False
    mr.ModelRunner._settle_target_pcp_split(runner)
    assert ctx.target_pcp_split is False
    # SP splits every pass too, but into contiguous shards, not round-robin.
    ctx.pcp_split = True
    monkeypatch.setattr(mr, "pcp_is_enabled", lambda: False)
    mr.ModelRunner._settle_target_pcp_split(runner)
    assert ctx.target_pcp_split is False


def test_aux_gathers_exactly_when_the_target_forward_split(monkeypatch):
    """After a row-split target prefill the aux hooks hold this rank's 1/pcp
    stripe; reading it as full rows silently halves DSpark's acceptance."""
    dr = _import_or_skip_dep("atom.spec_decode.drafter")
    ws, n = 4, 10  # pads to 12: 3 rows per rank
    gathers = []

    def fake_gather(xs, pcp_size):
        gathers.append(([x.shape[0] for x in xs], pcp_size))
        return [x.repeat(pcp_size, 1) for x in xs]

    ctx = SimpleNamespace(context=SimpleNamespace(target_pcp_split=True))
    monkeypatch.setattr(dr, "get_pcp_world_size", lambda: ws)
    monkeypatch.setattr(dr, "pcp_allgather_rerange_many", fake_gather)
    monkeypatch.setattr(dr, "get_forward_context", lambda: ctx)
    buffers = [torch.ones(16, 2) for _ in range(3)]
    drafter = SimpleNamespace(
        _captures_aux=True, _aux_buffers=buffers, _relayed_aux=None
    )
    hidden = torch.zeros(n, 2)

    aux = dr.Drafter.aux_for(drafter, hidden)
    # Every layer's stripe in one collective.
    assert gathers == [([3, 3, 3], ws)] and [a.shape[0] for a in aux] == [n] * 3

    gathers.clear()
    ctx.context.target_pcp_split = False
    aux = dr.Drafter.aux_for(drafter, hidden)
    assert gathers == [] and [a.shape[0] for a in aux] == [n] * 3


def test_draft_split_follows_the_target_not_the_current_pass(monkeypatch):
    """By step 0 the draft pass has cleared `pcp_split`; whether to split the
    draft prefill is the target forward's answer, `target_pcp_split`."""
    ctx = SimpleNamespace(
        context=SimpleNamespace(target_pcp_split=True, pcp_split=False)
    )
    monkeypatch.setattr(ep, "get_forward_context", lambda: ctx)
    assert ep._pcp_active_for_draft_model(None) is True
    ctx.context.target_pcp_split, ctx.context.pcp_split = False, True
    assert ep._pcp_active_for_draft_model(None) is False


def test_dspark_proposer_hands_the_model_its_own_stripe(monkeypatch):
    """After a row-split target prefill the proposer passes this rank's aux
    stripe and the row count to the model, which decides what to gather;
    otherwise the gathered aux rows, as before PCP."""
    dp = _import_or_skip_dep("atom.spec_decode.dspark_proposer")
    n, n_local = 10, 3  # 4 ranks pad to 12: 3 rows per rank
    positions = torch.arange(50, 50 + n + 6)
    bufs = [torch.randn(16, 3), torch.randn(16, 2)]
    calls = []
    model = SimpleNamespace(
        kv_pool_spec=None,
        write_context_kv=lambda aux, pos, slots, **kw: calls.append((aux, pos, kw)),
    )
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(served_by_round=False, slot_mapping=None),
        context=SimpleNamespace(scheduled_bs=2),
    )
    monkeypatch.setattr(dp, "get_forward_context", lambda: ctx)
    full = [b[:n] for b in bufs]
    drafter = SimpleNamespace(
        model=model,
        _captures_aux=True,
        _aux_buffers=bufs,
        _relayed_aux=None,
        aux_for=lambda h: full,
    )
    monkeypatch.setattr(dp, "pcp_aux_rows", lambda rows: n_local)
    drafter.aux_for = lambda h: pytest.fail("gathered the aux states")
    dp.DSparkProposer.compute_draft_kv(drafter, positions, torch.zeros(n, 7), None)
    ((aux, pos, kw),) = calls
    assert torch.equal(aux, torch.cat([b[:n_local] for b in bufs], dim=-1))
    assert pos is positions and kw == {"pcp_num_tokens": n}

    # TR_HYBRID: `positions` covers fewer rows than the target ran.
    calls.clear()
    dp.DSparkProposer.compute_draft_kv(
        drafter, positions[: n - 2], torch.zeros(n, 7), None
    )
    ((_, _, kw),) = calls
    assert kw == {"pcp_num_tokens": n - 2}

    calls.clear()
    monkeypatch.setattr(dp, "pcp_aux_rows", lambda rows: None)
    drafter.aux_for = lambda h: full
    dp.DSparkProposer.compute_draft_kv(drafter, positions, torch.zeros(n, 7), None)
    ((aux, pos, kw),) = calls
    assert torch.equal(aux, torch.cat(full, dim=-1)) and kw == {}


def _split_ctx(monkeypatch, ws, dummy=False):
    import atom.utils.forward_context as fcm

    ctx = SimpleNamespace(context=SimpleNamespace(is_dummy_run=dummy))
    monkeypatch.setattr(fcm, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(pu, "get_pcp_world_size", lambda: ws)


@pytest.mark.parametrize("width", [1, 2])  # bf16 window rows / fp8 2buff pair
def test_dspark_v4_split_context_kv_gathers_each_rank_s_own_kv(monkeypatch, width):
    """V4's window KV is made row by row, so each rank makes its own rows at
    their own positions, and every stage's rows (the 2buff pair under an fp8
    window) travel in one collective: the window gets exactly what making
    every row writes."""
    v4d = _import_or_skip_dep("atom.models.deepseek_v4_dspark")
    from atom.models.dspark_draft import DSparkDraftModel

    ws, n, n_local = 4, 10, 3
    positions = torch.arange(50, 50 + n + 6)  # the forward buffer runs past n
    aux_full = torch.randn(n, 5)
    _split_ctx(monkeypatch, ws)

    class Layer:
        def __init__(self, k):
            self.k, self.written = k, []

        def context_kv_rows(self, x, pos):
            kv = x.sum(-1, keepdim=True) * self.k + pos[:, None].float()
            return (kv, -kv)[:width]

        def write_context_kv_rows(self, rows, pos):
            self.written.append((rows, pos))

    layers = [Layer(1.0), Layer(-2.0)]
    model = SimpleNamespace(project_context=lambda a: a * 3, context_layers=layers)
    model.write_context_kv = types.MethodType(DSparkDraftModel.write_context_kv, model)
    model.write_split_context_kv = types.MethodType(
        v4d.DeepseekV4DSpark.write_split_context_kv, model
    )
    stripes = [
        pu.pcp_round_robin_split(pu.pcp_pad_dense(aux_full, 2), ws, r)
        for r in range(ws)
    ]
    sent = []
    for rank in range(ws):
        monkeypatch.setattr(pu, "get_pcp_rank", lambda r=rank: r)
        monkeypatch.setattr(
            pu,
            "pcp_allgather_rerange_many",
            lambda ts, w: sent.append(ts) or [t.repeat(w, 1) for t in ts],
        )
        model.write_context_kv(stripes[rank], positions, pcp_num_tokens=n)
    # One collective a rank, carrying every stage's rows.
    assert [len(ts) for ts in sent] == [width * len(layers)] * ws
    # Row t of the gather is rank t % ws's local row t // ws.
    gathered = [
        torch.stack([sent[r][i] for r in range(ws)], dim=1).reshape(ws * n_local, -1)
        for i in range(width * len(layers))
    ]
    ctx_full = model.project_context(aux_full)
    for i, layer in enumerate(layers):
        want = layer.context_kv_rows(ctx_full, positions[:n])
        for j in range(width):
            assert torch.allclose(gathered[i * width + j][:n], want[j])

    for layer in layers:
        layer.written.clear()
    monkeypatch.setattr(pu, "pcp_allgather_rerange_many", lambda ts, w: gathered)
    model.write_context_kv(stripes[0], positions, pcp_num_tokens=n)
    for i, layer in enumerate(layers):
        ((written, pos),) = layer.written
        assert len(written) == width and pos is positions
        for j, w in enumerate(written):
            assert torch.equal(w, gathered[i * width + j][:n])


def test_dspark_default_split_context_kv_gathers_the_aux_rows(monkeypatch):
    """A draft that does not make its KV row by row gathers the aux stripe and
    writes as before; a dummy run writes nothing either way."""
    from atom.models.dspark_draft import DSparkDraftModel

    ws, n = 4, 10
    _split_ctx(monkeypatch, ws)
    written = []
    gathers = []
    model = SimpleNamespace(
        project_context=lambda a: a + 1,
        context_layers=[
            SimpleNamespace(write_context_kv=lambda h, p, s: written.append((h, p, s)))
        ],
    )
    model.write_context_kv = types.MethodType(DSparkDraftModel.write_context_kv, model)
    model.write_split_context_kv = types.MethodType(
        DSparkDraftModel.write_split_context_kv, model
    )

    def gather(ts, w=None):
        gathers.append([t.shape[0] for t in ts])
        return [t.repeat(ws, 1) for t in ts]

    monkeypatch.setattr(pu, "pcp_allgather_rerange_many", gather)
    positions = torch.arange(n)
    slots = torch.arange(n) + 100
    model.write_context_kv(torch.ones(3, 5), positions, slots, pcp_num_tokens=n)
    ((h, p, s),) = written
    assert gathers == [[3]] and h.shape == (n, 5) and p is positions and s is slots

    written.clear()
    _split_ctx(monkeypatch, ws, dummy=True)
    model.write_context_kv(torch.ones(3, 5), positions, pcp_num_tokens=n)
    assert written == []


def test_every_dspark_draft_override_takes_pcp_num_tokens():
    """The proposer passes `pcp_num_tokens` to every draft model, and each
    override must act on it; read from source so a draft whose own extra deps
    are missing here is still checked. V4.1 refuses context parallel at
    startup (`validate_runtime_config`), so its override never sees it."""
    import ast
    import pathlib

    import atom

    def _uses(fn, name):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(fn))

    def _forwards_kwargs(fn):
        kw = fn.args.kwarg.arg
        return any(
            isinstance(n, ast.keyword) and n.arg is None and _uses(n.value, kw)
            for n in ast.walk(fn)
        )

    classes = []
    for path in pathlib.Path(atom.__file__).parent.rglob("*.py"):
        text = path.read_text()
        if "DSpark" not in text:
            continue
        tree = ast.parse(text)
        # `import X as Y` renames, this file's only.
        aliases = {
            a.asname: a.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for a in node.names
            if a.asname
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {getattr(b, "id", getattr(b, "attr", None)) for b in node.bases}
                classes.append((node, {aliases.get(b, b) for b in bases}))
    # Subclasses at any depth.
    drafts = {"DSparkDraftModel"}
    while True:
        more = {
            c.name for c, bases in classes if c.name not in drafts and bases & drafts
        }
        if not more:
            break
        drafts |= more
    for cls, _ in classes:
        if cls.name in ("DSparkDraftModel", "DeepseekV41DSpark") or (
            cls.name not in drafts
        ):
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "write_context_kv":
                args = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
                if "pcp_num_tokens" in args:
                    assert _uses(fn, "pcp_num_tokens"), cls.name
                else:
                    assert fn.args.kwarg and _forwards_kwargs(fn), cls.name
    assert {"DeepseekV4DSpark", "KimiK3DSparkDraft"} <= drafts
