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
  * folded MoE grid      --- `pcp_merged_tp_grid` vs `FusedMoEParallelConfig.make`
"""

import importlib
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

# atom.config first, so a circular import on the mainline path cannot be
# mistaken for an absent dependency below.
try:
    import atom.config
    import atom.distributed.pcp_utils as pu
    from atom.model_ops import moe as moe_mod
    from atom.spec_decode import eagle_proposer as ep

    # NOT `import aiter.dist.parallel_state as aps`: aiter's __init__ rebinds
    # its `dist` attribute to torch.distributed after the real subpackage is
    # already in sys.modules, so the `as` form's attribute walk lands on torch
    # and raises. The repo's own `from aiter.dist.parallel_state import X` is
    # unaffected (that reads sys.modules). The module object is what is needed
    # here: moe.py imports those names *inside* the function under test, so
    # only patching the real module reaches it.
    aps = importlib.import_module("aiter.dist.parallel_state")
except ImportError as _e:  # pragma: no cover - env without aiter/triton
    # Not `except Exception`: that is the exact mistake import_guard.py was
    # written about -- it swallows a SyntaxError in the module under test and
    # reports the silence as a skip.
    skip_if_dependency_missing(_e, "requires full atom import env")

HID = 4
WS = 4


def _inputs(n):
    """Row i carries i+1 everywhere, so a shard's provenance is readable.

    i+1, not i: padding fills with zero, so starting at 1 leaves 0 as a
    sentinel that cannot be confused with real row 0 -- otherwise the
    ownership assertions below have to exempt the one value most likely to
    expose an off-by-one.
    """
    ids = torch.arange(1, n + 1, dtype=torch.int32)
    pos = torch.arange(1, n + 1, dtype=torch.int32)
    hid = torch.arange(1, n + 1, dtype=torch.float32)[:, None].repeat(1, HID)
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


# --------------------------------------------------------------------------
# folded MoE grid --- `pcp_merged_tp_grid`
#
# DSV4's `moe_inter_dim`, the dimension the shared expert is cut along. 3072 /
# 128 = 24, so the folded grid can only cut it 1/2/4/8 ways before a block-quant
# scale block would straddle two ranks.
# --------------------------------------------------------------------------

INTER = 3072


def _cfg(**over):
    base = {
        "enable_dp_attention": False,
        "moe_ep_flatten_tp_across_dp": False,
        "enable_expert_parallel": False,
        "dp_logical_size": 0,
        "moe_all2all_backend": "auto",
        "enable_low_latency": False,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _as_rank(m, pcp_size, pcp_rank, tp_size, tp_rank, merge="1", cfg=None):
    """Present one (pcp, tp) rank position to BOTH folds under test.

    Patched in two namespaces on purpose: `pcp_merged_tp_grid` reads the pcp
    helpers and tp group out of `pcp_utils`, while `FusedMoEParallelConfig.make`
    reads its tp group out of `moe` and imports the pcp ones from aiter inside
    the call. Same numbers, so the two can be compared.
    """
    cfg = cfg if cfg is not None else _cfg()
    m.setenv("ATOM_PCP_MOE_MERGE", merge)
    tp = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank)
    m.setattr(pu, "get_pcp_world_size", lambda: pcp_size)
    m.setattr(pu, "get_pcp_rank", lambda: pcp_rank)
    m.setattr(pu, "get_tp_group", lambda: tp)
    m.setattr(moe_mod, "get_tp_group", lambda: tp)
    m.setattr(moe_mod, "get_current_atom_config", lambda: cfg)
    m.setattr(atom.config, "get_current_atom_config", lambda: cfg)
    m.setattr(aps, "get_prefill_context_model_parallel_world_size", lambda: pcp_size)
    m.setattr(aps, "get_prefill_context_model_parallel_rank", lambda: pcp_rank)
    return cfg


@pytest.mark.parametrize("pcp_size, tp_size", [(8, 1), (2, 4), (4, 2)])
def test_folded_grid_agrees_with_fused_moe(monkeypatch, pcp_size, tp_size):
    """The two folds must produce the same (size, rank) on every rank.

    This is the invariant, not the formula: routed and shared ride the same tp
    all_reduce + pcp collective, so if the two disagree about which slice a rank
    owns, one of them is double-counted on every layer of every token. `pcp8` is
    tp=1, where both flatten orders coincide and cannot be told apart -- the
    mixed configs are what actually discriminate.
    """
    for flat in range(pcp_size * tp_size):
        pcp_rank, tp_rank = divmod(flat, tp_size)
        with monkeypatch.context() as m:
            cfg = _as_rank(m, pcp_size, pcp_rank, tp_size, tp_rank)
            grid = pu.pcp_merged_tp_grid()
            fused = moe_mod.FusedMoEParallelConfig.make(tp_size, 1, cfg)
        assert grid is not None
        assert (grid.size, grid.rank) == (fused.tp_size, fused.tp_rank), (
            f"pcp{pcp_size}xtp{tp_size} flat={flat}: "
            f"grid={grid.size, grid.rank} fused={fused.tp_size, fused.tp_rank}"
        )
        assert grid.prescale == 1.0  # whole grid -> nothing to undo


def test_the_reversed_flatten_would_be_caught(monkeypatch):
    """The agreement test above has to be able to go red.

    Both orders are permutations of 0..N-1, so "every shard owned once" passes
    either way -- only agreement with moe.py separates them. Pin that the
    reversed mapping really does differ, so the test above is not vacuous.
    """
    pcp_size, tp_size = 2, 4
    differs = 0
    for flat in range(pcp_size * tp_size):
        pcp_rank, tp_rank = divmod(flat, tp_size)
        with monkeypatch.context() as m:
            _as_rank(m, pcp_size, pcp_rank, tp_size, tp_rank)
            grid = pu.pcp_merged_tp_grid()
        reversed_rank = tp_rank * pcp_size + pcp_rank  # the mapping to avoid
        differs += grid.rank != reversed_rank
    assert differs > 0


@pytest.mark.parametrize(
    "pcp_size, tp_size, want",
    [
        (8, 1, (8, 1.0)),  # 3072/8 = 384, a 128 multiple -> the whole grid
        (16, 1, (8, 0.5)),  # 3072/16 = 192 is not -> step down to 8
        (8, 2, (8, 0.5)),  # same ceiling reached from the other axis
        (2, 8, None),  # steps all the way down to tp -> stay on the TP group
    ],
)
def test_steps_down_to_a_block_aligned_grid(monkeypatch, pcp_size, tp_size, want):
    """A shard that would split a 128 scale block faults in the weight loader.

    `tp*pcp = 16` is where DSV4 hits it: 3072/16 = 192, and `narrow` then runs
    off the end of a 3-row scale. Step down instead of crashing at load.
    """
    with monkeypatch.context() as m:
        _as_rank(m, pcp_size, 0, tp_size, 0)
        grid = pu.pcp_merged_tp_grid(shard_dim=INTER, align=128)
    if want is None:
        assert grid is None
        return
    size, prescale = want
    assert (grid.size, grid.prescale) == (size, prescale)
    assert INTER % grid.size == 0 and (INTER // grid.size) % 128 == 0


def test_prescale_matches_how_often_a_shard_is_summed(monkeypatch):
    """`prescale` is only correct if the shards tile the ranks evenly.

    The collectives downstream sum all pcp*tp contributions blindly. Cut g ways,
    every shard must land on exactly pcp*tp/g ranks -- then and only then is the
    sum uniformly g/(pcp*tp) too large and a scalar can undo it.
    """
    pcp_size, tp_size = 16, 1
    held = {}
    prescale = None
    for flat in range(pcp_size * tp_size):
        pcp_rank, tp_rank = divmod(flat, tp_size)
        with monkeypatch.context() as m:
            _as_rank(m, pcp_size, pcp_rank, tp_size, tp_rank)
            grid = pu.pcp_merged_tp_grid(shard_dim=INTER, align=128)
        held[grid.rank] = held.get(grid.rank, 0) + 1
        prescale = grid.prescale
    assert sorted(held) == list(range(8)), "shards 0..7 all present"
    assert set(held.values()) == {2}, f"uneven: {held}"
    assert prescale == pytest.approx(1 / 2)


@pytest.mark.parametrize(
    "pcp_size, tp_size, merge",
    [(1, 8, "1"), (8, 1, "0")],  # no pcp to fold / fold switched off
)
def test_no_grid_when_the_fold_is_off(monkeypatch, pcp_size, tp_size, merge):
    with monkeypatch.context() as m:
        _as_rank(m, pcp_size, 0, tp_size, 0, merge=merge)
        assert pu.pcp_merged_tp_grid() is None


@pytest.mark.parametrize("flag", ["enable_dp_attention", "moe_ep_flatten_tp_across_dp"])
def test_flatten_tp_across_dp_is_refused(monkeypatch, flag):
    """moe.py folds DP into TP before pcp; nothing sums that axis back for a
    module carried only by the tp all_reduce + the pcp collective."""
    with monkeypatch.context() as m:
        _as_rank(m, 8, 0, 1, 0, cfg=_cfg(**{flag: True}))
        with pytest.raises(ValueError, match="flatten-TP-across-DP"):
            pu.pcp_merged_tp_grid()


def test_needs_id_sync_is_one_predicate_for_both_groups():
    """PCP and TP gate their id broadcast on the same question."""
    mr = pytest.importorskip("atom.model_engine.model_runner")
    assert mr._needs_id_sync(False, False) is False
    assert mr._needs_id_sync(True, False) is True
    assert mr._needs_id_sync(False, True) is True
    assert mr._needs_id_sync(True, True) is True


def test_shared_expert_stays_on_tp_when_requantized_online():
    """Online re-quantization gathers on the real TP group, which cannot rebuild
    a weight cut on the wider pcp*tp grid, so such a shared expert is not folded."""
    import ast
    import inspect
    import textwrap

    v4 = pytest.importorskip("atom.models.deepseek_v4")
    from atom.quant_spec import LayerQuantConfig, QuantType

    fp8 = LayerQuantConfig(
        quant_type=QuantType.per_Token, quant_dtype=torch.float8_e4m3fn
    )

    class QC:
        def __init__(self, online_quant, source, target):
            self.online_quant = online_quant
            self._source, self._target = source, target

        def get_layer_quant_config(self, name, use_online_quant=False):
            return self._target if use_online_quant else self._source

    prefix = "layers.3.ffn.shared_experts"
    bf16 = LayerQuantConfig.no_quant()
    assert v4._shared_expert_requantized_online(QC(True, bf16, fp8), prefix)
    assert not v4._shared_expert_requantized_online(QC(False, bf16, fp8), prefix)
    # A source already in the online target's format keeps its checkpoint weight.
    assert not v4._shared_expert_requantized_online(QC(True, fp8, fp8), prefix)
    assert not v4._shared_expert_requantized_online(None, prefix)

    # And MoE consults it before picking the grid.
    src = textwrap.dedent(inspect.getsource(v4.MoE.__init__))
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    assert "_shared_expert_requantized_online" in names
