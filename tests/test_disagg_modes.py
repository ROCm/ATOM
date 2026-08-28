# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for intra-GPU disagg constrained vs unconstrained modes.

Only the scheduler-level shm gating is exercised here; the IPC handshake
and CUDA stream pool are out of scope for the no-GPU test environment.
"""

import logging

import pytest
import torch
from conftest import MockConfig



@pytest.fixture
def prefill_scheduler_unconstrained():
    from atom.model_engine.scheduler import PrefillScheduler

    return PrefillScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def decode_scheduler_unconstrained():
    from atom.model_engine.scheduler import DecodeScheduler

    return DecodeScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def seq_factory():
    from atom.sampling_params import SamplingParams
    from atom.model_engine.sequence import Sequence

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


# ── Unconstrained: no shm handle attached ────────────────────────────────


def test_prefill_scheduler_skips_shm_when_name_empty(prefill_scheduler_unconstrained):
    assert prefill_scheduler_unconstrained._cu_shm is None


def test_decode_scheduler_skips_shm_when_name_empty(decode_scheduler_unconstrained):
    assert decode_scheduler_unconstrained._cu_shm is None


# ── Unconstrained: batches carry cu_stream_fraction=None ─────────────────


def test_unconstrained_prefill_batch_has_none_cu_fraction(
    prefill_scheduler_unconstrained, seq_factory
):
    """Without shm, PrefillScheduler must produce batches keyed by the
    plain (None) stream — never a fractional CU mask."""
    seq = seq_factory([10, 20, 30, 40])
    seq.block_table = [0, 1]
    seq.num_cached_tokens = 0
    prefill_scheduler_unconstrained.add(seq)

    batch, _ = prefill_scheduler_unconstrained.schedule()
    assert batch is not None
    assert batch.cu_stream_fraction is None


# ── Asymmetric rapidserve: reserved KV dump index ────────────────────────
#
# Under asymmetric rapidserve (prefill TP=N, decode TP=1) MLA's TP-replicated
# KV latent leaves every prefill rank holding an identical copy, but only the
# rank co-located with the target GPU may commit it. The others redirect their
# writes to block_manager.DUMP_INDEX, which must therefore never be handed out
# by any of the three pools.


class TestDumpIndexReservation:
    """The sink is a REGION, not an index.

    V4's absorbed prefill reads the KV back out of the paged cache during
    attention, so each dumped row needs its own scratch blocks — collapsing a
    batch onto block 0 makes every row read whichever token wrote last. The
    pool therefore holds out `dump_block_count(config)` blocks, sized for one
    prefill batch.
    """

    # 64 batched tokens / block_size 4 = 16 blocks, + 4 seqs for partial
    # blocks = 20 reserved. The pool must be bigger than that.
    DUMP = 20

    @staticmethod
    def _block_manager(**overrides):
        from atom.model_engine.block_manager import BlockManager

        return BlockManager(MockConfig(**overrides))

    def test_symmetric_keeps_full_block_budget(self):
        """No reservation without asymmetry — every other config is unaffected."""
        from atom.model_engine.block_manager import DUMP_INDEX

        bm = self._block_manager(num_kvcache_blocks=10)
        assert DUMP_INDEX in bm.free_block_ids_set
        assert len(bm.free_block_ids_set) == 10

    def test_region_size_covers_one_prefill_batch(self):
        from atom.model_engine.block_manager import dump_block_count

        assert dump_block_count(MockConfig()) == self.DUMP

    def test_region_rounds_partial_blocks_up_per_sequence(self):
        """Each row can waste part of a block, hence the + max_num_seqs term."""
        from atom.model_engine.block_manager import dump_block_count

        cfg = MockConfig(
            max_num_batched_tokens=65, kv_cache_block_size=4, max_num_seqs=3
        )
        assert dump_block_count(cfg) == 17 + 3

    def test_asymmetric_reserves_the_whole_region(self):
        bm = self._block_manager(num_kvcache_blocks=64, disagg_prefill_tp_size=8)
        assert not (set(range(self.DUMP)) & bm.free_block_ids_set)
        assert len(bm.free_block_ids_set) == 64 - self.DUMP

    def test_region_is_never_allocated(self):
        """Drain the pool: no scratch block may appear in any allocation."""
        bm = self._block_manager(num_kvcache_blocks=64, disagg_prefill_tp_size=8)
        drained = [bm.free_block_ids.popleft() for _ in range(len(bm.free_block_ids))]
        assert sorted(drained) == list(range(self.DUMP, 64))

    def test_pool_too_small_for_the_region_is_rejected(self):
        """Silently sharing scratch is the bug; fail at startup instead."""
        with pytest.raises(AssertionError, match="scratch blocks"):
            self._block_manager(num_kvcache_blocks=10, disagg_prefill_tp_size=8)

    def test_per_req_cache_groups_reserve_exactly_one_slot(self):
        """Unlike the paged pools: that pool's width is pinned to max_num_seqs."""
        from atom.model_engine.block_manager import DUMP_INDEX

        bm = self._block_manager(
            num_kvcache_blocks=64,
            num_per_req_cache_groups=4,
            disagg_prefill_tp_size=8,
        )
        assert DUMP_INDEX not in bm.free_per_req_cache_groups
        assert bm.free_per_req_cache_groups == [1, 2, 3]

    def test_per_req_cache_groups_unreserved_when_symmetric(self):
        bm = self._block_manager(num_kvcache_blocks=10, num_per_req_cache_groups=4)
        assert bm.free_per_req_cache_groups == [0, 1, 2, 3]

    def test_empty_per_req_pool_is_not_broken_by_reservation(self):
        """Models with no per-request state (pool size 0) must not underflow."""
        bm = self._block_manager(num_kvcache_blocks=64, disagg_prefill_tp_size=8)
        assert bm.free_per_req_cache_groups == []


class TestSwaDumpIndexReservation:
    @staticmethod
    def _pool(num_blocks, reserve):
        from atom.model_engine.swa_pool import SlidingWindowPool

        return SlidingWindowPool(
            num_blocks=num_blocks,
            window=8,
            block_size=4,
            max_num_batched_tokens=64,
            mtp_k=0,
            reserve_dump_blocks=reserve,
        )

    def test_single_sink_block_is_what_block_manager_asks_for(self):
        """SWA needs ONE sink block, not a per-row region.

        Prefill reads the SWA pool back only for the PREFIX region; the current
        chunk's window comes from the per-forward KV tensor
        (paged_prefill_indices.py:123-128 vs :130-147). With chunked prefill off
        and no prefix-cache hit nothing reads what dumped rows wrote, so they
        may alias. Sizing it like the compressed pool asked a 1088-block pool
        for 1536 blocks.
        """
        from atom.model_engine.block_manager import BlockManager, dump_block_count

        # SWA pool deliberately SMALLER than the region the compressed pool
        # reserves — the real config was 1088 vs 1536. Sizing them alike would
        # empty it; the point is that SWA gives up exactly one block.
        cfg = MockConfig(
            num_kvcache_blocks=4096,
            disagg_prefill_tp_size=8,
            num_swa_blocks=8,
            swa_window_size=8,
        )
        n_dump = dump_block_count(cfg)
        assert n_dump > 8, "pool must be smaller than the region to prove this"
        bm = BlockManager(cfg)
        assert len(bm.swa.free_block_ids_set) == 7

    def test_region_reserved(self):
        pool = self._pool(6, reserve=3)
        assert not (set(range(3)) & pool.free_block_ids_set)
        assert len(pool.free_block_ids_set) == 3

    def test_not_reserved_by_default(self):
        pool = self._pool(6, reserve=0)
        assert len(pool.free_block_ids_set) == 6

    def test_disabled_pool_stays_empty(self):
        """num_blocks=0 means SWA is off; reservation must not go negative."""
        pool = self._pool(0, reserve=3)
        assert not pool.enabled
        assert len(pool.free_block_ids_set) == 0

    def test_reservation_larger_than_pool_is_refused(self):
        """Reserving the whole pool leaves nothing allocatable — a silent hang.

        The real config hit this: a 1088-block SWA pool asked for a 1536-block
        dump region. Clamping produced an empty free list and no sequence could
        ever be admitted.
        """
        with pytest.raises(ValueError, match="does not fit in"):
            self._pool(6, reserve=10)

    def test_reservation_exactly_filling_the_pool_is_refused(self):
        with pytest.raises(ValueError, match="does not fit in"):
            self._pool(6, reserve=6)

    def test_reservation_leaving_one_block_is_allowed(self):
        pool = self._pool(6, reserve=5)
        assert len(pool.free_block_ids_set) == 1


# ── Asymmetric rapidserve: TP=1 weight twins ─────────────────────────────
#
# Prefill runs TP=N and decode TP=1, so a few weights (attention, shared
# experts, embed/head) have no counterpart among prefill's shards. Rather than
# let decode load them — which meant 8 concurrent checkpoint reads and a
# partial-load path that kept writing through IPC aliases into prefill's memory
# — prefill builds a TP=1 twin of those modules and exports it. Selecting the
# wrong module set is the failure that matters: twinning a FusedMoE would
# duplicate ~90GB that already aliases correctly, and MISSING a module leaves
# decode with an un-materialized meta parameter.


class TestTwinSelection:
    @staticmethod
    def _mod(cls_name, **attrs):
        m = torch.nn.Module()
        m.__class__ = type(cls_name, (torch.nn.Module,), {})
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def test_column_parallel_needs_a_twin(self):
        """tp_dim set and sharded => its TP=1 shape differs."""
        m = self._mod("ColumnParallelLinear", tp_dim=0, tp_size=8)
        assert m.tp_dim is not None and m.tp_size > 1

    def test_replicated_linear_needs_no_twin(self):
        """tp_dim None => already full size on every rank."""
        m = self._mod("ReplicatedLinear", tp_dim=None, tp_size=8)
        assert not (m.tp_dim is not None and m.tp_size > 1)

    def test_tp1_module_needs_no_twin(self):
        """Nothing is sharded at tp_size=1, so there is nothing to reconstruct."""
        m = self._mod("ColumnParallelLinear", tp_dim=0, tp_size=1)
        assert not (m.tp_dim is not None and m.tp_size > 1)

    def test_fused_moe_is_never_twinned(self):
        """The expensive case: MoE already aliases via flatten_tp_across_dp."""
        # decode_twins pulls in AITER-backed modules; other suites here
        # mock/unmock aiter mid-run, so skip rather than fail on a polluted
        # sys.modules.
        dt = pytest.importorskip("atom.model_engine.decode_twins")
        # needs_twin imports these lazily, so importorskip on decode_twins alone
        # is not enough to skip cleanly under a polluted sys.modules.
        pytest.importorskip("atom.model_ops.linear")
        pytest.importorskip("atom.model_ops.embed_head")

        moe = self._mod("FusedMoE", ep_size=8, ep_rank=0, tp_size=8)
        assert dt.needs_twin(moe) is False


class TestSoloGroup:
    """Twins are constructed under a patched single-rank group."""

    def test_reports_world_size_one(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        g = dt._SoloGroup()
        assert g.world_size == 1 and g.rank_in_group == 0

    def test_patch_is_restored(self):
        decode_twins = pytest.importorskip("atom.model_engine.decode_twins")
        linear = pytest.importorskip("atom.model_ops.linear")
        pytest.importorskip("atom.model_ops.embed_head")

        before = linear.get_tp_group
        with decode_twins._tp_group_of_one():
            assert linear.get_tp_group().world_size == 1
        assert linear.get_tp_group is before


class TestDualLoader:
    """One `loaded_weight` must populate both the shard and the twin."""

    def test_both_sides_receive_the_full_tensor(self):
        _dual_loader = pytest.importorskip(
            "atom.model_engine.decode_twins"
        )._dual_loader

        seen = {}

        def real_loader(param, loaded_weight, *a, **kw):
            seen["real"] = loaded_weight.shape

        twin = torch.nn.Module()
        twin.weight_loader = lambda p, w, *a, **kw: seen.__setitem__("twin", w.shape)
        tparam = torch.nn.Parameter(torch.zeros(32, 8))

        dual = _dual_loader(real_loader, twin, "weight", tparam)
        dual(torch.nn.Parameter(torch.zeros(4, 8)), torch.zeros(32, 8))

        # Both see the FULL checkpoint tensor; each narrows for itself.
        assert seen["real"] == (32, 8)
        assert seen["twin"] == (32, 8)

    def test_twin_without_a_loader_is_skipped(self):
        _dual_loader = pytest.importorskip(
            "atom.model_engine.decode_twins"
        )._dual_loader

        calls = []
        dual = _dual_loader(
            lambda p, w, *a, **kw: calls.append("real"),
            torch.nn.Module(),
            "weight",
            torch.nn.Parameter(torch.zeros(2)),
        )
        dual(torch.nn.Parameter(torch.zeros(2)), torch.zeros(2))
        assert calls == ["real"]


# ── Asymmetric rapidserve: decode-rank selection ─────────────────────────
#
# DisaggCoreManager's engine list is [prefill, decode_0 .. decode_N-1], so a
# request's target must be chosen from the decode slice only. Picking index 0
# would route a sequence to the prefill process, which owns no KV pool.


class TestBoundedRankSelection:
    class _Mgr:
        """Just the selector and the counters it reads."""

        def __init__(self, n_engines, strategy="least_requests"):
            from atom.model_engine.engine_core_mgr import CoreManager

            self.local_engine_count = n_engines
            self._dp_lb_strategy = strategy
            self._dp_lb_req_equiv = 0
            self._rank_reqs = [0] * n_engines
            self._rank_tokens = [0] * n_engines
            self._rank_rotation_cursor = 0
            self._select = CoreManager._select_dp_rank_locked.__get__(self)

    def test_never_selects_the_prefill_index(self):
        m = self._Mgr(9)  # 1 prefill + 8 decode
        picks = {m._select(lo=1, hi=9) for _ in range(50)}
        assert 0 not in picks
        assert picks <= set(range(1, 9))

    def test_round_robin_stays_in_range(self):
        m = self._Mgr(9, strategy="round_robin")
        picks = [m._select(lo=1, hi=9) for _ in range(24)]
        assert set(picks) == set(range(1, 9))
        assert picks[0] == 1 and picks[8] == 1  # wraps within the slice

    def test_prefers_the_least_loaded_decode_rank(self):
        m = self._Mgr(5)  # 1 prefill + 4 decode
        m._rank_reqs = [99, 3, 1, 7, 5]  # index 0 is prefill and must be ignored
        assert m._select(lo=1, hi=5) == 2

    def test_default_bounds_cover_every_engine(self):
        """Non-disagg callers pass no bounds and must be unaffected."""
        m = self._Mgr(4, strategy="round_robin")
        assert {m._select() for _ in range(12)} == {0, 1, 2, 3}


# ── Asymmetric rapidserve: per-row KV write ownership ────────────────────


class TestOwnedRowMask:
    class _Builder:
        def __init__(self, rank):
            # `backends` pulls in AITER-backed modules. Other suites here
            # mock/unmock aiter mid-run, so this import can land in a polluted
            # sys.modules; skip rather than fail the run when it does.
            backends = pytest.importorskip("atom.model_ops.attentions.backends")

            self.model_runner = type("R", (), {"rank": rank})()
            self._owned = backends.CommonAttentionBuilder.disagg_owned_rows.__get__(
                self
            )

    @staticmethod
    def _batch(target_ranks):
        return type("B", (), {"target_ranks": target_ranks})()

    def test_rank_owns_only_its_own_rows(self):
        b = self._batch([2, 5, 2, 7])
        assert list(self._Builder(2)._owned(b)) == [True, False, True, False]
        assert list(self._Builder(5)._owned(b)) == [False, True, False, False]

    def test_rank_with_no_rows_owns_nothing(self):
        b = self._batch([2, 5, 2])
        assert not any(self._Builder(4)._owned(b))

    def test_missing_target_info_returns_none(self):
        """Warmup/dummy batches carry no targets; caller masks everything."""
        assert self._Builder(0)._owned(type("B", (), {})()) is None
        assert self._Builder(0)._owned(self._batch([])) is None


# ── Asymmetric rapidserve: GPU pinning of decode ranks ───────────────────
#
# ModelRunner derives its GPU from
#   (data_parallel_rank_local * pp + pp_rank) * (tp * pcp) + rank
# (model_runner.py:970-983). Decode runs at tp=1, so data_parallel_rank_local is
# the ONLY term that varies between its ranks — omit it and all N processes
# compute device 0, pile onto GPU 0, and then try to open IPC handles belonging
# to prefill ranks on other GPUs. That surfaces far away as "storage on
# different device", so the arithmetic is pinned down here.


def _local_device_rank(dp_rank_local, rank, tp, pp=1, pp_rank=0, pcp=1):
    engine_index = dp_rank_local * pp + pp_rank
    return engine_index * (tp * pcp) + rank


class TestDecodeGpuPinning:
    def test_prefill_tp_ranks_span_all_gpus(self):
        got = [_local_device_rank(0, r, tp=8) for r in range(8)]
        assert got == list(range(8))

    def test_decode_dp_ranks_span_all_gpus(self):
        """tp=1, so the DP-local rank alone must carry the device index."""
        got = [_local_device_rank(k, 0, tp=1) for k in range(8)]
        assert got == list(range(8))

    def test_unset_dp_rank_local_collapses_onto_gpu0(self):
        """The exact bug: without dp_rank_local every decode rank lands on 0."""
        got = [_local_device_rank(0, 0, tp=1) for _ in range(8)]
        assert got == [0] * 8

    def test_decode_rank_k_pairs_with_prefill_rank_k(self):
        """Both sides must resolve to the same physical GPU for IPC to work."""
        for k in range(8):
            assert _local_device_rank(k, 0, tp=1) == _local_device_rank(0, k, tp=8)


# ── rapidserve: where decode's device cost comes from ────────────────────
#
# prefill's `non_torch` = (total - free) - its own reserved, i.e. every byte the
# DEVICE says is used that prefill does not own. On 8xMI355X / V4-Pro that came
# to 150.10 GB while decode's allocator held ~5 GB at import time, and the terms
# reconcile with device-used (111.64 + 150.10 = 261.7 vs 260.2 actually used) —
# so the accounting is self-consistent, NOT double counting.
#
# An earlier attempt netted the IPC-mapped weights out of `non_torch` on the
# theory that they were prefill's own pages counted twice. That was wrong: it
# handed out a KV budget that did not exist and turned a clean startup error
# into an OOM inside allocate_kv_cache. `non_torch` must stay as written.
#
# Which of map_cost vs load_cost dominates the 150 GB is still open; the
# instrumentation in import_model_weight_ipc_handles measures both.


GB = 1 << 30


def _non_torch(total, free, reserved):
    """Mirrors the expression in ModelRunner.get_num_blocks."""
    return max((total - free) - reserved, 0)


class TestNonTorchAccounting:
    def test_reports_memory_held_by_other_processes(self):
        assert _non_torch(288 * GB, 100 * GB, 180 * GB) == 8 * GB

    def test_peer_memory_is_not_netted_out(self):
        """Regression guard for the reverted 'double count' fix."""
        # Device used 260, prefill allocator 110 -> 150 unavailable to prefill.
        # Subtracting the ~105 GB decode has mapped would report 45 and OOM later.
        assert _non_torch(288 * GB, 28 * GB, 110 * GB) == 150 * GB

    def test_never_goes_negative(self):
        assert _non_torch(288 * GB, 290 * GB, 80 * GB) == 0

    def test_terms_reconcile_with_device_used(self):
        """peak_torch + non_torch should sum to device-used, not exceed it.

        111.64 + 150.10 = 261.74 against 260.16 GB used: agreement to ~1.6GB is
        what shows the two terms are not counting the same bytes twice.
        """
        total, free = 287.98, 27.82
        peak_torch, non_torch = 111.64, 150.10
        assert abs((peak_torch + non_torch) - (total - free)) < 2.0


# ── rapidserve: exported tensors must outlive the export call ────────────
#
# decode's parameters are CUDA IPC views into prefill's allocations, so anything
# prefill materialises during export has to stay referenced for the life of the
# process. `_export_tensor` calls `.contiguous()`, which returns a NEW tensor for
# a non-contiguous input — the handle is taken from that copy's storage, and if
# it is left to go out of scope the consumer reads freed memory.


class TestExportRetention:
    def test_contiguous_input_is_not_retained(self):
        """Nothing was materialised, so there is nothing extra to hold."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        retain = []
        t = torch.zeros(4, 4)
        try:
            ipc._export_tensor(t, retain)
        except RuntimeError:
            pass  # CPU tensor: _share_cuda_ raises, but only AFTER .contiguous()
        assert retain == []

    def test_materialised_copy_is_retained(self):
        """A non-contiguous input produces a copy that MUST be kept alive."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        retain = []
        base = torch.zeros(4, 8)
        view = base[:, ::2]  # strided
        assert not view.is_contiguous()
        try:
            ipc._export_tensor(view, retain)
        except RuntimeError:
            pass
        assert len(retain) == 1
        assert retain[0].is_contiguous()
        assert retain[0].shape == view.shape

    def test_retain_is_optional(self):
        """Callers that do not care must not crash."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        base = torch.zeros(4, 8)
        try:
            ipc._export_tensor(base[:, ::2])
        except RuntimeError:
            pass


# ── Twin module attributes must override, not just fill gaps ─────────────
#
# process_weights_after_loading emits more than tensors. LinearBase sets
# `is_output_padded` (linear.py:798) and `_output_size_before_padding` (:827),
# and forward slices padded columns off using both (:985). Whether padding is
# needed is decided from output_size — precisely what differs between the TP=N
# module and its TP=1 twin — so the consumer must take the TWIN's values. It
# cannot derive them: it never runs the hook, and __init__ leaves
# is_output_padded=False (linear.py:508).


class TestTwinModuleAttrs:
    @staticmethod
    def _twins_with(attrs):
        dt = pytest.importorskip("atom.model_engine.decode_twins")
        t = dt.DecodeTwins()
        twin = torch.nn.Module()
        for k, v in attrs.items():
            setattr(twin, k, v)
        t._twins["layers.0.attn.wo_b"] = twin
        return t

    def test_collects_post_processing_flags(self):
        t = self._twins_with({"is_output_padded": True})
        got = t.module_attr_overrides()["layers.0.attn.wo_b"]
        assert got["is_output_padded"] is True

    def test_includes_underscore_prefixed_names(self):
        """_module_meta_attrs skips these; dropping it breaks forward."""
        t = self._twins_with(
            {"is_output_padded": True, "_output_size_before_padding": 7168}
        )
        got = t.module_attr_overrides()["layers.0.attn.wo_b"]
        assert got["_output_size_before_padding"] == 7168

    def test_excludes_tensors_and_training_flag(self):
        t = self._twins_with(
            {"is_output_padded": True, "weight": torch.zeros(4), "training": False}
        )
        got = t.module_attr_overrides()["layers.0.attn.wo_b"]
        assert "weight" not in got and "training" not in got

    def test_false_value_still_travels(self):
        """A False must be carried; the consumer cannot tell it from a default."""
        t = self._twins_with({"is_output_padded": False})
        got = t.module_attr_overrides()["layers.0.attn.wo_b"]
        assert got["is_output_padded"] is False


# ── Ancestor hooks must run against the twins too ────────────────────────
#
# A post-load hook does not always live on the module owning the weight.
# DeepSeek-V4's attention dequantizes wo_a from FP8 to BF16 for the grouped-LoRA
# einsum (deepseek_v4.py:2453) — the hook is on the PARENT, and wo_a is a
# twinned child. Running only the twins' own hooks leaves the twin FP8, and
# decode dies in _wo_a_grouped_lora with "expected scalar type BFloat16 but
# found Float8_e4m3fn".


class _ParentWithHook(torch.nn.Module):
    """Stands in for DeepseekV4Attention: the hook mutates its child."""

    def __init__(self, child):
        super().__init__()
        self.wo_a = child
        self.ran_on = []

    def process_weights_after_loading(self):
        self.ran_on.append(id(self.wo_a))
        self.wo_a.dequantized = True


class TestHookOrdering:
    """Hooks must run in `named_modules()` order — parents before children.

    `load_model` walks the model that way, and V4 depends on it: the attention
    dequants wo_a FP8 -> BF16 (deepseek_v4.py:2453) and wo_a's own LinearBase
    hook must then see the BF16 result. Running the child first preshuffles the
    FP8 weight and the parent dequants a shuffled tensor — output that is
    structurally valid, numerically wrong, and never reaches EOS.
    """

    @staticmethod
    def _model_with_twin(order):
        """Parent + twinned child, both recording the order they ran in."""
        child = torch.nn.Module()
        twin = torch.nn.Module()
        twin.process_weights_after_loading = lambda: order.append("child_twin")

        parent = torch.nn.Module()
        parent.wo_a = child
        parent.process_weights_after_loading = lambda: order.append(
            "parent(%s)" % ("twin" if parent.wo_a is twin else "real")
        )

        model = torch.nn.Module()
        model.attn = parent
        return model, parent, child, twin

    def test_parent_runs_before_its_twinned_child(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        order = []
        model, parent, child, twin = self._model_with_twin(order)
        t = dt.DecodeTwins()
        t._twins["attn.wo_a"] = twin
        t.finalize(model)
        assert order == ["parent(twin)", "child_twin"], order

    def test_parent_hook_sees_the_twin(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        order = []
        model, parent, child, twin = self._model_with_twin(order)
        t = dt.DecodeTwins()
        t._twins["attn.wo_a"] = twin
        t.finalize(model)
        assert "parent(twin)" in order and "parent(real)" not in order

    def test_real_child_is_restored(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        order = []
        model, parent, child, twin = self._model_with_twin(order)
        t = dt.DecodeTwins()
        t._twins["attn.wo_a"] = twin
        t.finalize(model)
        assert parent.wo_a is child

    def test_restored_even_if_the_hook_raises(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        order = []
        model, parent, child, twin = self._model_with_twin(order)
        parent.process_weights_after_loading = lambda: (_ for _ in ()).throw(
            RuntimeError("hook failed")
        )
        t = dt.DecodeTwins()
        t._twins["attn.wo_a"] = twin
        with pytest.raises(RuntimeError, match="hook failed"):
            t.finalize(model)
        assert parent.wo_a is child

    def test_parent_without_twinned_children_is_not_rerun(self):
        """Only modules that actually own a twin get their hook re-run."""
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        order = []
        model = torch.nn.Module()
        model.other = torch.nn.Module()
        model.other.process_weights_after_loading = lambda: order.append("other")
        t = dt.DecodeTwins()
        t.finalize(model)
        assert order == []


class TestDecodeTwinsExportContract:
    """Every method the export path calls must exist and behave.

    Guard against edits that remove one: `export_model_weight_handles` reaches
    for `overrides()` and `module_attr_overrides()`, and `_build_and_load_model`
    for `finalize()` / `drop_replicated_bare()`. A missing method is an
    AttributeError deep in prefill's bootstrap that no static check catches.
    """

    REQUIRED = (
        "build",
        "finalize",
        "drop_replicated_bare",
        "overrides",
        "module_attr_overrides",
    )

    def test_all_required_methods_exist(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")
        missing = [m for m in self.REQUIRED if not hasattr(dt.DecodeTwins, m)]
        assert not missing, f"DecodeTwins is missing: {missing}"

    def test_overrides_covers_params_attrs_and_bare(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        t = dt.DecodeTwins()
        twin = torch.nn.Module()
        twin.weight = torch.nn.Parameter(torch.zeros(4, 4))
        # A post-load hook can turn a Parameter into a plain attribute.
        twin.weight_scale = torch.zeros(4)
        t._twins["attn.wq_b"] = twin
        t._bare_full["attn.attn_sink"] = torch.zeros(128)

        got = t.overrides()
        assert set(got) == {
            "attn.wq_b.weight",
            "attn.wq_b.weight_scale",
            "attn.attn_sink",
        }

    def test_overrides_skips_private_attrs(self):
        dt = pytest.importorskip("atom.model_engine.decode_twins")

        t = dt.DecodeTwins()
        twin = torch.nn.Module()
        twin._scratch = torch.zeros(2)
        t._twins["attn.wq_b"] = twin
        assert t.overrides() == {}


# ── Decode DP lockstep ───────────────────────────────────────────────────
#
# The MoE all_gather / reduce_scatter (moe.py:362) is a COLLECTIVE across the
# decode DP ranks, run once per MoE layer per step. Every rank must issue a
# forward on every step or the ones that do block forever on the ones that do
# not. That is exactly what happens while draining — ranks run out of sequences
# at different times — and it cost 7 in-flight requests in debug9.log, with the
# hang surfacing as aiter "shared memory broadcast block" timeouts.
#
# DPEngineCoreProc's loop prevents it: all-reduce `has_unfinished` each step and
# run `_execute_dummy_batch()` on any rank with nothing real to do.


class TestDecodeLockstep:
    def test_decode_engine_inherits_the_dp_loop(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")
        assert issubclass(ec.DecodeEngineCore, ec.DPEngineCoreProc)

    def test_lockstep_selected_when_a_dp_group_exists(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = object()  # stand-in for a real group
        assert obj._dp_lockstep is True

    def test_plain_loop_when_single_decode_rank(self):
        """Symmetric rapidserve: one decode engine, no peer to stay in step with."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None
        assert obj._dp_lockstep is False

    def test_missing_attribute_degrades_to_plain_loop(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        assert obj._dp_lockstep is False

    def test_step_reports_whether_it_ran(self):
        """The loop uses this to decide when a dummy batch is needed."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.scheduler = type("S", (), {"has_requests": lambda self: False})()
        assert obj._process_engine_step() is False


# ── Symmetric rapidserve init path ───────────────────────────────────────
#
# `--enable-rapidserve` without `--enable-dp-attention` runs ONE decode process
# at the same TP as prefill, so decode worker `r` sits on GPU r and pairs with
# prefill TP rank r. `disagg_decode_rank` is 0 for all of them, which is what
# made indexing the IPC handle files by decode rank alone collapse every worker
# onto prefill rank 0's file: workers 1..N-1 opened a handle exported on cuda:0
# from another GPU and died, and the engine only noticed two RPCs later, when
# the manager had already torn the broadcast shm down ("'NoneType' object is
# not subscriptable" out of aiter's shm_broadcast).
#
# Decode also has to keep taking the plain EngineCore loop here: with a single
# rank there is no collective to stay in step with, and DPEngineCoreProc's loop
# would all-reduce over a group that was never formed.


def _rapidserve_runner(
    *, rank, decode, decode_rank=0, tp=8, prefill_tp=0, device=None, pcp=1
):
    """A RapidServeModelRunner with only the attributes the pairing logic reads.

    `tp` is THIS process's TP size (8 for symmetric decode, 1 for asymmetric);
    `prefill_tp` is 0 for symmetric and the prefill TP size for asymmetric.
    `device` defaults to the correctly-paired GPU so a test only has to spell it
    out when it is deliberately breaking the pairing.
    """
    import types

    mr = pytest.importorskip("atom.model_engine.model_runner")

    obj = object.__new__(mr.RapidServeModelRunner)
    obj.rank = rank
    obj.config = types.SimpleNamespace(
        disagg_is_decode=decode,
        disagg_decode_rank=decode_rank,
        disagg_prefill_tp_size=prefill_tp,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
    )
    if device is None:
        device = decode_rank * tp * pcp + rank
    obj.device = torch.device(f"cuda:{device}")
    return obj


class TestDisaggPairRank:
    """The arithmetic itself. `disagg_types` is stdlib-only on purpose, so these
    keep running under the module-shadowing that skips the ModelRunner tests
    below when the whole suite runs in one process."""

    @staticmethod
    def _pair_rank(*args):
        from atom.model_engine.disagg_types import disagg_pair_rank

        return disagg_pair_rank(*args)

    def test_symmetric_decode_worker_pairs_with_its_own_gpu(self):
        """The regression: TP=8 decode, one process, decode_rank 0 throughout."""
        got = [self._pair_rank(0, r, 8) for r in range(8)]
        assert got == list(range(8))

    def test_symmetric_decode_does_not_collapse_onto_prefill_rank_0(self):
        """Every worker reading paths[0] is what killed ranks 1..7."""
        assert len({self._pair_rank(0, r, 8) for r in range(8)}) == 8

    def test_asymmetric_decode_indexes_by_decode_rank(self):
        """TP=1 decode: the worker rank is always 0, so decode_rank is the GPU."""
        got = [self._pair_rank(k, 0, 1) for k in range(8)]
        assert got == list(range(8))

    def test_matches_the_device_arithmetic_in_both_topologies(self):
        """The pair rank IS a GPU index, so it must equal _local_device_rank."""
        for r in range(8):  # symmetric: one process at TP=8
            assert self._pair_rank(0, r, 8) == _local_device_rank(0, r, tp=8)
        for k in range(8):  # asymmetric: eight processes at TP=1
            assert self._pair_rank(k, 0, 1) == _local_device_rank(k, 0, tp=1)

    def test_context_parallel_widens_the_stage(self):
        """stage_span is tp*pcp, matching _setup_device_and_distributed."""
        assert self._pair_rank(1, 1, 2 * 2) == 5


class TestDisaggPairRankWiring:
    """...and that RapidServeModelRunner actually feeds it both terms."""

    def test_symmetric_decode_worker_pairs_with_its_own_gpu(self):
        got = [
            _rapidserve_runner(rank=r, decode=True, tp=8)._disagg_pair_rank
            for r in range(8)
        ]
        assert got == list(range(8))

    def test_asymmetric_decode_indexes_by_decode_rank(self):
        got = [
            _rapidserve_runner(
                rank=0, decode=True, decode_rank=k, tp=1, prefill_tp=8
            )._disagg_pair_rank
            for k in range(8)
        ]
        assert got == list(range(8))

    def test_prefill_worker_uses_its_own_tp_rank(self):
        got = [
            _rapidserve_runner(rank=r, decode=False, tp=8)._disagg_pair_rank
            for r in range(8)
        ]
        assert got == list(range(8))


class TestDisaggPairingGuard:
    def test_symmetric_mismatch_is_caught(self):
        """The guard used to be asymmetric-only, so this bug walked straight past.

        Worker 3 of the symmetric decode process is on cuda:3; a pair rank of 0
        would have it open a handle exported on cuda:0.
        """
        runner = _rapidserve_runner(rank=3, decode=True, tp=8, device=0)
        with pytest.raises(RuntimeError, match="pairing broken"):
            runner._assert_disagg_pairing()

    def test_symmetric_correct_pairing_passes(self):
        for r in range(8):
            runner = _rapidserve_runner(rank=r, decode=True, tp=8)
            runner._assert_disagg_pairing()  # must not raise

    def test_asymmetric_correct_pairing_passes(self):
        for k in range(8):
            runner = _rapidserve_runner(
                rank=0, decode=True, decode_rank=k, tp=1, prefill_tp=8
            )
            runner._assert_disagg_pairing()  # must not raise

    def test_asymmetric_collapse_onto_gpu0_is_caught(self):
        """Forgetting data_parallel_rank_local lands every decode rank on cuda:0."""
        runner = _rapidserve_runner(
            rank=0, decode=True, decode_rank=5, tp=1, prefill_tp=8, device=0
        )
        with pytest.raises(RuntimeError, match="pairing broken"):
            runner._assert_disagg_pairing()


class TestSymmetricDecodeInit:
    def test_single_rank_forms_no_dp_group(self):
        """dp_size == 1: nothing to rendezvous with, so never build a group."""
        import types

        ec = pytest.importorskip("atom.model_engine.engine_core")

        def _boom():
            raise AssertionError("stateless_init_dp_group must not be called")

        obj = object.__new__(ec.DPEngineCoreProc)
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                data_parallel_rank=0,
                data_parallel_size=1,
                # None is what a non-DP launch leaves this at; the DP path
                # asserts on it, so the early return has to come first.
                data_parallel_rank_local=None,
                stateless_init_dp_group=_boom,
            )
        )
        obj._init_data_parallel(config)
        assert obj.dp_group is None
        assert obj.dp_rank == 0

    def test_symmetric_takes_the_plain_engine_loop(self, monkeypatch):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        picked = []
        monkeypatch.setattr(
            ec.EngineCore, "busy_loop", lambda self: picked.append("plain")
        )
        monkeypatch.setattr(
            ec.DPEngineCoreProc, "busy_loop", lambda self: picked.append("lockstep")
        )
        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None
        obj.busy_loop()
        assert picked == ["plain"]

    def test_asymmetric_takes_the_lockstep_loop(self, monkeypatch):
        """Load-bearing: dropping it reintroduces the drain deadlock."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        picked = []
        monkeypatch.setattr(
            ec.EngineCore, "busy_loop", lambda self: picked.append("plain")
        )
        monkeypatch.setattr(
            ec.DPEngineCoreProc, "busy_loop", lambda self: picked.append("lockstep")
        )
        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = object()
        obj.busy_loop()
        assert picked == ["lockstep"]

    def test_decode_never_attaches_a_prefill_delayer(self, monkeypatch):
        """Decode has no prefill to delay, and no scheduler yet at this point."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        monkeypatch.setattr(ec.envs, "ATOM_ENABLE_PREFILL_DELAYER", True)

        class _Scheduler:
            def set_prefill_delayer(self, delayer):
                raise AssertionError("decode must not get a prefill delayer")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.scheduler = _Scheduler()
        obj.dp_group = None
        assert obj._maybe_attach_prefill_delayer(None) is None


# ── Asymmetric rapidserve: twin/shard cross-check geometry ───────────────
#
# `verify_against_shards` asserts that a twin holds the same bytes its real
# module's TP shard received from the one shared `loaded_weight`. That is only
# meaningful if the shard geometry is mapped correctly — a wrong map either
# compares the wrong bytes (silent) or fails a correct run (fatal). Merged
# layers are the case worth pinning down: the real tensor is
# [gate_shard, up_shard] while the twin is [gate_full, up_full], so a single
# narrow reads across the partition boundary.


class TestShardViewGeometry:
    @staticmethod
    def _mod(**attrs):
        m = torch.nn.Module()
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    @staticmethod
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def test_column_parallel_single_partition(self):
        from atom.model_engine.decode_twins import shard_view

        full = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
        real = self._mod(tp_size=4, tp_dim=0, output_partition_sizes=[2])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[8])
        for rank in range(4):
            shard = full[2 * rank : 2 * rank + 2]
            view = shard_view(real, twin, self._p(full), self._p(shard), rank)
            assert torch.equal(view, shard)

    def test_merged_layer_walks_partitions(self):
        """The case a single narrow gets wrong: [gate|up] vs [gate_full|up_full]."""
        from atom.model_engine.decode_twins import shard_view

        gate = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        up = torch.arange(100, 104, dtype=torch.float32).reshape(4, 1)
        full = torch.cat([gate, up], 0)  # twin: 8 rows
        real = self._mod(tp_size=2, tp_dim=0, output_partition_sizes=[2, 2])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[4, 4])

        for rank in range(2):
            expect = torch.cat(
                [gate[2 * rank : 2 * rank + 2], up[2 * rank : 2 * rank + 2]], 0
            )
            view = shard_view(real, twin, self._p(full), self._p(expect), rank)
            assert torch.equal(view, expect)
        # and a naive contiguous narrow would NOT have matched rank 1
        naive = full.narrow(0, 2 * 1 * 2, 4)
        expect1 = torch.cat([gate[2:4], up[2:4]], 0)
        assert not torch.equal(naive, expect1)

    def test_row_parallel_narrows_input_dim(self):
        from atom.model_engine.decode_twins import shard_view

        full = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
        real = self._mod(tp_size=4, tp_dim=1, output_partition_sizes=[2])
        twin = self._mod(tp_size=1, tp_dim=1, output_partition_sizes=[2])
        for rank in range(4):
            shard = full[:, 2 * rank : 2 * rank + 2]
            view = shard_view(real, twin, self._p(full), self._p(shard), rank)
            assert torch.equal(view, shard)

    def test_vocab_parallel_uses_start_index(self):
        from atom.model_engine.decode_twins import shard_view

        full = torch.arange(12 * 2, dtype=torch.float32).reshape(12, 2)
        real = self._mod(tp_size=4, vocab_start_idx=6)
        twin = self._mod(tp_size=1)
        view = shard_view(real, twin, self._p(full), self._p(full[6:9]), 2)
        assert torch.equal(view, full[6:9])

    def test_replicated_param_compared_whole(self):
        """Same shape => not sharded; the twin must match exactly."""
        from atom.model_engine.decode_twins import shard_view

        t = torch.ones(3, 1)
        real = self._mod(tp_size=8, tp_dim=0, output_partition_sizes=[3])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[3])
        view = shard_view(real, twin, self._p(t), self._p(t), 0)
        assert torch.equal(view, t)

    def test_unmapped_geometry_is_skipped_not_guessed(self):
        """No tp_dim and no vocab index => None, so the caller skips it."""
        from atom.model_engine.decode_twins import shard_view

        full = torch.zeros(8, 2)
        real = self._mod(tp_size=4)
        twin = self._mod(tp_size=1)
        shard = self._p(torch.zeros(2, 2))
        assert shard_view(real, twin, self._p(full), shard, 0) is None

    def test_inconsistent_partition_sizes_are_skipped(self):
        """twin partition != real partition * tp_size => geometry unknown."""
        from atom.model_engine.decode_twins import shard_view

        real = self._mod(tp_size=4, tp_dim=0, output_partition_sizes=[2])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[6])
        view = shard_view(
            real, twin, self._p(torch.zeros(6, 1)), self._p(torch.zeros(2, 1)), 0
        )
        assert view is None

    def test_fp8_equality_does_not_raise(self):
        from atom.model_engine.decode_twins import _tensors_equal

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch build has no float8_e4m3fn")
        a = torch.zeros(4, dtype=torch.float8_e4m3fn)
        b = torch.zeros(4, dtype=torch.float8_e4m3fn)
        assert _tensors_equal(a, b)
        c = torch.ones(4, dtype=torch.float8_e4m3fn)
        assert not _tensors_equal(a, c)


class TestDiagnoseMismatch:
    """A failing cross-check has to say WHICH failure it is.

    'The twin holds the right bytes but the checker looked at the wrong offset'
    and 'the twin never received these bytes' need opposite fixes, and the only
    thing that separates them is whether the shard turns up somewhere else.
    """

    @staticmethod
    def _mod(**attrs):
        m = torch.nn.Module()
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    @staticmethod
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def test_reports_the_offset_that_does_match(self):
        from atom.model_engine.decode_twins import diagnose_mismatch

        full = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        real = self._mod(tp_size=4, tp_rank=3, tp_dim=0)
        twin = self._mod(tp_size=1)
        # the shard really living at index 3, but checked as rank 0
        msg = diagnose_mismatch(real, twin, self._p(full), self._p(full[6:8]), 0)
        assert "offset index [3]" in msg
        assert "tp_rank=3 (checked as 0)" in msg

    def test_reports_when_no_offset_matches(self):
        from atom.model_engine.decode_twins import diagnose_mismatch

        full = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        real = self._mod(tp_size=4, tp_rank=1, tp_dim=0)
        twin = self._mod(tp_size=1)
        alien = torch.full((2, 2), -1.0)
        msg = diagnose_mismatch(real, twin, self._p(full), self._p(alien), 1)
        assert "NO offset" in msg

    def test_survives_unmapped_geometry(self):
        """Must never raise — it runs on the failure path."""
        from atom.model_engine.decode_twins import diagnose_mismatch

        real = self._mod(tp_size=4, tp_rank=0, tp_dim=None)
        twin = self._mod(tp_size=1)
        msg = diagnose_mismatch(
            real, twin, self._p(torch.zeros(4, 1)), self._p(torch.zeros(2, 1)), 0
        )
        assert "tp_dim=None" in msg


class TestTwinLifetimeDetection:
    """Separate a lifetime fault from an arithmetic one.

    `_dual_loader` closes over the twin Parameter it will write into, which pins
    that object for the whole load. If anything rebinds `twin.<attr>` afterwards,
    the load lands on an orphan and export ships the un-written replacement — the
    bytes are not merely wrong, they were never written. Same symptom as a bad
    shard offset, opposite fix, so the checker has to name which one it is.
    """

    @staticmethod
    def _twin(written=True, rebind=False):
        twin = torch.nn.Module()
        full = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        twin.weight = torch.nn.Parameter(full, requires_grad=False)
        target = twin.weight
        if rebind:
            target = torch.nn.Parameter(full.clone(), requires_grad=False)
        twin._twin_targets = {"weight": target}
        twin._twin_writes = {"weight": 1} if written else {}
        return twin

    @staticmethod
    def _twins(twin, agree):
        from atom.model_engine.decode_twins import DecodeTwins

        obj = object.__new__(DecodeTwins)
        obj._twins = {"m": twin}
        obj._bare_full = {}
        obj._agree = {"m.weight": agree}
        obj._diag = []
        return obj

    def test_clean_pair_passes(self):
        self._twins(self._twin(), True).verify_against_shards(None, 0)

    def test_rebound_parameter_is_named_as_such(self):
        twins = self._twins(self._twin(rebind=True), True)
        with pytest.raises(RuntimeError, match="replaced after the dual loader"):
            twins.verify_against_shards(None, 0)

    def test_never_written_parameter_is_named_as_such(self):
        twins = self._twins(self._twin(written=False), True)
        with pytest.raises(RuntimeError, match="never fed by the dual loader"):
            twins.verify_against_shards(None, 0)

    def test_lifetime_faults_outrank_byte_mismatch(self):
        """An unwritten twin also mismatches; the root cause must win."""
        twins = self._twins(self._twin(written=False), False)
        with pytest.raises(RuntimeError, match="never fed by the dual loader"):
            twins.verify_against_shards(None, 0)

    def test_disagreement_is_reported(self):
        twins = self._twins(self._twin(), False)
        with pytest.raises(RuntimeError, match="disagree with the TP shard"):
            twins.verify_against_shards(None, 0)

    def test_unjudged_geometry_does_not_fail(self):
        """None means 'not mapped', which must never be read as a failure."""
        self._twins(self._twin(), None).verify_against_shards(None, 0)


class TestAgreeRecordedAtLoadTime:
    """The comparison must happen while both sides are still raw.

    `load_model` post-processes the real model before it returns, so FP8 weights
    are preshuffled by the time the load call finishes. Comparing after that
    measures the preshuffle and flags every quantized weight as corrupt. The
    recorder therefore runs inside the dual loader, and these tests pin that it
    is wired up and that a correct dual load records agreement.
    """

    @staticmethod
    def _pair():
        from atom.model_engine.decode_twins import DecodeTwins

        real = torch.nn.Module()
        real.tp_size, real.tp_dim, real.tp_rank = 2, 0, 1
        real.output_partition_sizes = [2]
        real.weight = torch.nn.Parameter(torch.zeros(2, 1), requires_grad=False)

        twin = torch.nn.Module()
        twin.tp_size, twin.tp_dim = 1, 0
        twin.output_partition_sizes = [4]
        twin.weight = torch.nn.Parameter(
            torch.arange(4, dtype=torch.float32).reshape(4, 1), requires_grad=False
        )

        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        return obj, real, twin

    def test_matching_load_records_agreement(self):
        twins, real, twin = self._pair()
        # rank 1 owns rows 2..3 of the twin
        real.weight.data.copy_(twin.weight.data[2:4])
        twins._agree_recorder("m", real, twin)(real.weight, "weight", ())
        assert twins._agree == {"m.weight": True}

    def test_mismatching_load_records_disagreement_and_diagnoses(self):
        twins, real, twin = self._pair()
        real.weight.data.fill_(-1.0)
        twins._agree_recorder("m", real, twin)(real.weight, "weight", ())
        assert twins._agree == {"m.weight": False}
        assert twins._diag and "m.weight" in twins._diag[0]

    def test_wrong_offset_is_recorded_as_disagreement(self):
        """Right bytes, wrong rank: still a disagreement, but diagnosable."""
        twins, real, twin = self._pair()
        real.weight.data.copy_(twin.weight.data[0:2])  # rank 0's rows, not rank 1's
        twins._agree_recorder("m", real, twin)(real.weight, "weight", ())
        assert twins._agree == {"m.weight": False}
        assert "offset index [0]" in twins._diag[0]

    def test_unmapped_geometry_records_none(self):
        twins, real, twin = self._pair()
        real.tp_dim = None
        twins._agree_recorder("m", real, twin)(real.weight, "weight", ())
        assert twins._agree == {"m.weight": None}

    def test_dual_loader_invokes_the_recorder(self):
        from atom.model_engine.decode_twins import _dual_loader

        seen = []
        twin = torch.nn.Module()
        twin.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        twin.weight_loader = lambda p, w, *a, **k: None
        real_param = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        load = _dual_loader(
            lambda p, w, *a, **k: None,
            twin,
            "weight",
            twin.weight,
            lambda rp, attr, args: seen.append(attr),
        )
        load(real_param, torch.zeros(2))
        assert seen == ["weight"]


class TestMergedPartitionScoping:
    """A merged layer's partitions arrive from different threads.

    `loading_core.py:187-198` loads weights through a ThreadPoolExecutor, and a
    merged layer's gate/up halves are separate checkpoint tensors on separate
    tasks — so both write one Parameter concurrently. Comparing the whole
    Parameter from inside one of those calls races the other, which showed up as
    sporadic per-rank disagreement on weights that were actually correct. Each
    partition has exactly one writer, so the comparison must be scoped to it.
    """

    @staticmethod
    def _real(**over):
        m = torch.nn.Module()
        m.tp_size, m.tp_dim, m.tp_rank = 2, 0, 0
        m.output_partition_sizes = [2, 2]
        for k, v in over.items():
            setattr(m, k, v)
        return m

    def test_int_shard_id_selects_that_partition(self):
        from atom.model_engine.decode_twins import _written_partition

        assert _written_partition(self._real(), (1,)) == 1
        assert _written_partition(self._real(), (0,)) == 0

    def test_no_shard_id_compares_whole_param(self):
        """The fused-tensor path writes every partition in one call."""
        from atom.model_engine.decode_twins import _written_partition

        assert _written_partition(self._real(), ()) is None

    def test_string_shard_id_is_not_a_partition_index(self):
        """QKV uses 'q'/'k'/'v'; indexing output_partition_sizes with it is wrong."""
        from atom.model_engine.decode_twins import _written_partition

        assert _written_partition(self._real(), ("q",)) is None

    def test_bool_is_not_treated_as_an_index(self):
        from atom.model_engine.decode_twins import _written_partition

        assert _written_partition(self._real(), (True,)) is None

    def test_unmerged_module_has_no_partition_scope(self):
        from atom.model_engine.decode_twins import _written_partition

        assert _written_partition(self._real(output_partition_sizes=[4]), (0,)) is None

    def test_out_of_range_shard_id_is_ignored(self):
        from atom.model_engine.decode_twins import _written_partition

        assert _written_partition(self._real(), (7,)) is None

    def test_half_written_merged_param_does_not_false_alarm(self):
        """The regression: gate written, up still empty -> gate must still agree."""
        from atom.model_engine.decode_twins import DecodeTwins

        real = torch.nn.Module()
        real.tp_size, real.tp_dim, real.tp_rank = 2, 0, 1
        real.output_partition_sizes = [2, 2]
        real.weight = torch.nn.Parameter(torch.zeros(4, 1), requires_grad=False)

        twin = torch.nn.Module()
        twin.tp_size, twin.tp_dim = 1, 0
        twin.output_partition_sizes = [4, 4]
        gate = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        up = torch.arange(100, 104, dtype=torch.float32).reshape(4, 1)
        twin.weight = torch.nn.Parameter(torch.cat([gate, up], 0), requires_grad=False)

        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        # only the gate half has landed; rank 1 owns gate rows 2..3
        real.weight.data[0:2] = gate[2:4]
        obj._agree_recorder("m", real, twin)(real.weight, "weight", (0,))
        assert obj._agree == {"m.weight#0": True}
        assert obj._diag == []


# ── Asymmetric rapidserve: masking the prefill KV WRITE ──────────────────
#
# prepare_block_tables masks the paged block table, but that is the READ side
# and it is published only when the batch has prefix-cache hits. Where prefill
# WRITES its KV is decided by slot_mapping, built in prepare_prefill from the
# raw batch.block_tables lists. Unmasked, every prefill TP rank commits every
# row into its own GPU's cache at block ids owned by a different decode rank's
# sequences — silent cross-sequence corruption.


class TestDisaggDumpRowPredicate:
    @staticmethod
    def _builder(masked, rank=3, targets=None):
        # Importing backends pulls in aiter, which the full-suite run mocks;
        # skip rather than fail on a polluted sys.modules (same as the other
        # backends test in this file).
        backends = pytest.importorskip("atom.model_ops.attentions.backends")

        # CommonAttentionBuilder is abstract; the predicate under test needs
        # only model_runner, so a minimal concrete subclass is enough.
        concrete = type(
            "Builder",
            (backends.CommonAttentionBuilder,),
            {
                "prepare_decode": lambda self, batch, bs: None,
                "build_for_cudagraph_capture": lambda self, bs: None,
            },
        )
        b = object.__new__(concrete)
        runner = type("R", (), {})()
        runner.rank = rank
        runner.config = MockConfig(
            disagg_prefill_tp_size=8 if masked else 0,
            disagg_is_decode=False,
        )
        b.model_runner = runner
        batch = type("B", (), {})()
        batch.target_ranks = targets
        return b, batch

    def test_disabled_outside_asymmetric_rapidserve(self):
        """Every other configuration must reach identical slot arithmetic."""
        b, batch = self._builder(masked=False, targets=[0, 1, 2])
        fn = b._disagg_dump_row_fn(batch)
        assert [fn(i) for i in range(3)] == [False, False, False]

    def test_owns_only_rows_targeting_this_rank(self):
        b, batch = self._builder(masked=True, rank=3, targets=[3, 5, 3, 0])
        fn = b._disagg_dump_row_fn(batch)
        assert [fn(i) for i in range(4)] == [False, True, False, True]

    def test_no_target_info_dumps_everything(self):
        """Warmup/dummy batches carry no targets; they must not reach a live pool."""
        b, batch = self._builder(masked=True, targets=None)
        fn = b._disagg_dump_row_fn(batch)
        assert all(fn(i) for i in range(4))

    def test_empty_target_list_dumps_everything(self):
        b, batch = self._builder(masked=True, targets=[])
        fn = b._disagg_dump_row_fn(batch)
        assert all(fn(i) for i in range(4))

    def test_row_past_end_of_ownership_dumps(self):
        b, batch = self._builder(masked=True, rank=0, targets=[0])
        fn = b._disagg_dump_row_fn(batch)
        assert fn(0) is False
        assert fn(1) is True

    def test_rank_zero_is_not_special(self):
        """rank 0 owning rows must not be confused with DUMP_INDEX == 0."""
        b, batch = self._builder(masked=True, rank=0, targets=[0, 1])
        fn = b._disagg_dump_row_fn(batch)
        assert [fn(i) for i in range(2)] == [False, True]


class TestDumpedRowSlotArithmetic:
    """A dumped row's slots must all land inside the reserved block."""

    @staticmethod
    def _slots(block_table, block_size, cached, seqlen):
        out = []
        first_blk = cached // block_size
        last_blk = (seqlen - 1) // block_size
        for blk_idx in range(first_blk, last_blk + 1):
            blk_start = block_table[blk_idx] * block_size
            off_start = cached % block_size if blk_idx == first_blk else 0
            last = blk_idx == last_blk
            off_end = ((seqlen - 1) % block_size) + 1 if last else block_size
            out.extend(range(blk_start + off_start, blk_start + off_end))
        return out

    def test_substituted_table_confines_writes_to_the_dump_block(self):
        from atom.model_engine.block_manager import DUMP_INDEX

        block_size, seqlen = 16, 40
        real = [7, 9, 4]
        dumped = [DUMP_INDEX] * len(real)
        got = self._slots(dumped, block_size, 0, seqlen)
        assert len(got) == seqlen  # same token count as the real row
        assert max(got) < block_size  # never leaves block 0
        assert min(got) >= 0
        # and it really differs from the unmasked write
        assert got != self._slots(real, block_size, 0, seqlen)
        # A 3-block row folded into one block MUST alias: 40 tokens into 16
        # slots. That is intended, not a bug to "fix" by widening the sink —
        # the writes are plain stores (no accumulation) into a block reserved
        # out of every pool, so whichever token lands last is irrelevant
        # because nothing ever reads it.
        assert len(set(got)) == block_size < len(got)

    def test_dumped_row_emits_one_slot_per_scheduled_token(self):
        """Length drives an assert in prepare_prefill; a short row would trip it."""
        from atom.model_engine.block_manager import DUMP_INDEX

        block_size, cached, seqlen = 16, 16, 40
        dumped = [DUMP_INDEX] * 3
        assert len(self._slots(dumped, block_size, cached, seqlen)) == seqlen - cached


class TestBlockScaleShardGeometry:
    """FP8 per-1x128 block scales shard at 1/128 the weight's granularity.

    These went unjudged for a while — 183 of them on a 61-layer V4 — because the
    partition walk demanded `sum(output_partition_sizes) == rows`, which counts
    WEIGHT rows. A scale has rows/128 of them, so every tp_dim==0 scale was
    skipped. They are the most numerically sensitive tensors in the twin set: a
    wrong block scale is fluent-but-wrong output, not a crash.
    """

    @staticmethod
    def _mod(**attrs):
        m = torch.nn.Module()
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    @staticmethod
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def test_unmerged_block_scale_is_judged(self):
        """wq_b: weight 8192x1536 per rank -> scale 64 rows per rank."""
        from atom.model_engine.decode_twins import shard_view

        real = self._mod(tp_size=8, tp_dim=0, output_partition_sizes=[8192])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[65536])
        full = torch.arange(512 * 12, dtype=torch.float32).reshape(512, 12)
        for rank in range(8):
            real.tp_rank = rank
            shard = full[rank * 64 : (rank + 1) * 64]
            view = shard_view(real, twin, self._p(full), self._p(shard), rank)
            assert view is not None, "block scale must not be skipped"
            assert torch.equal(view, shard)

    def test_merged_block_scale_walks_partitions(self):
        """gate_up: real scale [3 gate, 3 up]; twin [24 gate, 24 up]."""
        from atom.model_engine.decode_twins import shard_view

        real = self._mod(tp_size=8, tp_dim=0, output_partition_sizes=[384, 384])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[3072, 3072])
        gate = torch.arange(24, dtype=torch.float32).reshape(24, 1)
        up = torch.arange(100, 124, dtype=torch.float32).reshape(24, 1)
        full = torch.cat([gate, up], 0)  # 48 scale rows
        for rank in range(8):
            expect = torch.cat(
                [gate[3 * rank : 3 * rank + 3], up[3 * rank : 3 * rank + 3]], 0
            )
            view = shard_view(real, twin, self._p(full), self._p(expect), rank)
            assert view is not None
            assert torch.equal(view, expect)

    def test_weights_still_use_granularity_one(self):
        """The generalisation must not perturb the plain weight case."""
        from atom.model_engine.decode_twins import shard_view

        real = self._mod(tp_size=4, tp_dim=0, output_partition_sizes=[2])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[8])
        full = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
        for rank in range(4):
            shard = full[2 * rank : 2 * rank + 2]
            view = shard_view(real, twin, self._p(full), self._p(shard), rank)
            assert torch.equal(view, shard)

    def test_indivisible_rows_are_skipped_not_guessed(self):
        """rows that do not divide the weight partition => unknown geometry."""
        from atom.model_engine.decode_twins import shard_view

        real = self._mod(tp_size=8, tp_dim=0, output_partition_sizes=[8192])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[65536])
        # 100 rows does not divide 8192 evenly
        odd = torch.zeros(100, 4)
        big = torch.zeros(800, 4)
        assert shard_view(real, twin, self._p(big), self._p(odd), 0) is None

    def test_partition_not_divisible_by_granularity_is_skipped(self):
        from atom.model_engine.decode_twins import shard_view

        # sum=768, rows=6 -> gran=128, but 300 % 128 != 0
        real = self._mod(tp_size=8, tp_dim=0, output_partition_sizes=[300, 468])
        twin = self._mod(tp_size=1, tp_dim=0, output_partition_sizes=[2400, 3744])
        assert (
            shard_view(
                real, twin, self._p(torch.zeros(48, 1)), self._p(torch.zeros(6, 1)), 0
            )
            is None
        )


class TestScratchTableSubstitution:
    """Dumped rows get distinct scratch blocks, and both sides see the same ones.

    The write side (slot_mapping, built in prepare_prefill) and the read side
    (kv_indices, generated from the block-table buffer — aiter_mla.py:762) must
    resolve to identical physical blocks. V4's absorbed prefill reads its own
    KV back, so a mismatch, or two rows sharing scratch, is silent corruption
    that also poisons the owning rank through the TP all-reduce.
    """

    @staticmethod
    def _builder(masked=True, rank=0, targets=None, tables=None, **cfg):
        backends = pytest.importorskip("atom.model_ops.attentions.backends")

        concrete = type(
            "Builder",
            (backends.CommonAttentionBuilder,),
            {
                "prepare_decode": lambda self, batch, bs: None,
                "build_for_cudagraph_capture": lambda self, bs: None,
            },
        )
        b = object.__new__(concrete)
        runner = type("R", (), {})()
        runner.rank = rank
        runner.config = MockConfig(
            disagg_prefill_tp_size=8 if masked else 0,
            disagg_is_decode=False,
            **cfg,
        )
        b.model_runner = runner
        batch = type("B", (), {})()
        batch.target_ranks = targets
        batch.block_tables = tables or []
        return b, batch

    def test_owned_rows_keep_their_real_blocks(self):
        real = [[11, 12], [21, 22, 23]]
        b, batch = self._builder(rank=0, targets=[0, 0], tables=real)
        assert b.disagg_effective_block_tables(batch) == real

    def test_dumped_rows_do_not_alias_each_other(self):
        """The regression: one shared block made every dumped row read the last."""
        b, batch = self._builder(
            rank=0, targets=[1, 2, 3], tables=[[11, 12], [21, 22, 23], [31]]
        )
        eff = b.disagg_effective_block_tables(batch)
        flat = [blk for row in eff for blk in row]
        assert len(flat) == len(set(flat)), "scratch blocks must be distinct"
        assert eff == [[0, 1], [2, 3, 4], [5]]

    def test_shape_is_preserved_per_row(self):
        """Same block count per row, so slot_mapping length is unchanged."""
        tables = [[11, 12], [21, 22, 23], [31]]
        b, batch = self._builder(rank=7, targets=[0, 1, 2], tables=tables)
        eff = b.disagg_effective_block_tables(batch)
        assert [len(r) for r in eff] == [len(r) for r in tables]

    def test_mixed_batch_only_rewrites_non_owned(self):
        b, batch = self._builder(
            rank=1, targets=[0, 1, 2], tables=[[11, 12], [21], [31, 32]]
        )
        eff = b.disagg_effective_block_tables(batch)
        assert eff[1] == [21]  # owned, untouched
        assert eff[0] == [0, 1] and eff[2] == [2, 3]

    def test_deterministic_across_calls(self):
        """Write and read recompute independently; they must agree exactly."""
        b, batch = self._builder(
            rank=0, targets=[1, 2, 3], tables=[[11, 12], [21, 22, 23], [31]]
        )
        assert b.disagg_effective_block_tables(
            batch
        ) == b.disagg_effective_block_tables(batch)

    def test_disabled_returns_none(self):
        b, batch = self._builder(masked=False, targets=[0, 1], tables=[[11], [21]])
        assert b.disagg_effective_block_tables(batch) is None

    def test_no_target_info_dumps_every_row(self):
        b, batch = self._builder(targets=None, tables=[[11, 12], [21]])
        assert b.disagg_effective_block_tables(batch) == [[0, 1], [2]]

    def test_scratch_exhaustion_raises_instead_of_wrapping(self):
        """Wrapping the cursor would silently restore the aliasing bug."""
        # 8 batched tokens / block 4 = 2, + 1 seq = 3 scratch blocks available
        b, batch = self._builder(
            targets=[1],
            tables=[[1, 2, 3, 4]],
            max_num_batched_tokens=8,
            kv_cache_block_size=4,
            max_num_seqs=1,
        )
        with pytest.raises(RuntimeError, match="scratch exhausted"):
            b.disagg_effective_block_tables(batch)


class TestMergedBlockScaleRecording:
    """The crash from debug16: narrowing a scale with weight-row sizes.

    A merged layer's gate/up partitions are 384 WEIGHT rows each, but its
    per-1x128 scale has 3 rows each. `on_write` scoped the comparison to a
    partition using the raw weight sizes and tried `narrow(0, 0, 384)` on a
    6-row tensor. Every partition index must go through
    partition_sizes_in_param_units.
    """

    @staticmethod
    def _pair():
        from atom.model_engine.decode_twins import DecodeTwins

        real = torch.nn.Module()
        real.tp_size, real.tp_dim, real.tp_rank = 8, 0, 1
        real.output_partition_sizes = [384, 384]
        # scale: 768 weight rows / 128 = 6 rows
        real.weight_scale = torch.nn.Parameter(torch.zeros(6, 1), requires_grad=False)

        twin = torch.nn.Module()
        twin.tp_size, twin.tp_dim = 1, 0
        twin.output_partition_sizes = [3072, 3072]
        gate = torch.arange(24, dtype=torch.float32).reshape(24, 1)
        up = torch.arange(100, 124, dtype=torch.float32).reshape(24, 1)
        twin.weight_scale = torch.nn.Parameter(
            torch.cat([gate, up], 0), requires_grad=False
        )

        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        return obj, real, twin, gate, up

    def test_recording_a_merged_scale_partition_does_not_crash(self):
        twins, real, twin, gate, _ = self._pair()
        # rank 1's gate scale rows are twin gate rows 3..5
        real.weight_scale.data[0:3] = gate[3:6]
        twins._agree_recorder("m", real, twin)(real.weight_scale, "weight_scale", (0,))
        assert twins._agree == {"m.weight_scale#0": True}

    def test_second_partition_uses_the_scale_offset(self):
        twins, real, twin, _, up = self._pair()
        real.weight_scale.data[3:6] = up[3:6]
        twins._agree_recorder("m", real, twin)(real.weight_scale, "weight_scale", (1,))
        assert twins._agree == {"m.weight_scale#1": True}

    def test_wrong_scale_bytes_are_still_caught(self):
        twins, real, twin, _, _ = self._pair()
        real.weight_scale.data.fill_(-1.0)
        twins._agree_recorder("m", real, twin)(real.weight_scale, "weight_scale", (0,))
        assert twins._agree == {"m.weight_scale#0": False}

    def test_param_units_helper_reports_scale_sizes(self):
        from atom.model_engine.decode_twins import partition_sizes_in_param_units

        real = torch.nn.Module()
        real.output_partition_sizes = [384, 384]
        twin = torch.nn.Module()
        twin.output_partition_sizes = [3072, 3072]
        assert partition_sizes_in_param_units(real, twin, 6, 48) == ([3, 3], [24, 24])
        assert partition_sizes_in_param_units(real, twin, 768, 6144) == (
            [384, 384],
            [3072, 3072],
        )


class TestPostProcessingForwardCheck:
    """Bytes-in-right does not imply layout-out-right.

    verify_against_shards runs before finalize() and proves the twin was fed
    the same checkpoint bytes as the shard. process_weights_after_loading then
    preshuffles FP8 weights and applies ancestor dequant hooks, and a TP=1 twin
    could diverge there with no byte-level symptom. The only way to compare
    post-processed tensors is through the forward pass.
    """

    @staticmethod
    def _twins(mods):
        from atom.model_engine.decode_twins import DecodeTwins

        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = mods, {}, {}, []
        return obj

    @staticmethod
    def _linear(out_rows, in_size, weight):
        m = torch.nn.Module()
        m.tp_dim, m.input_size = 0, in_size
        m.output_partition_sizes = [out_rows]
        m.w = weight
        m.forward = lambda x: x.float() @ m.w.t()
        return m

    def test_matching_twin_reports_zero_error(self, caplog):
        """The twin's rank-1 output slice must equal the shard's output."""
        from atom.model_engine.decode_twins import DecodeTwins

        w = torch.randn(8, 4)
        real = self._linear(2, 4, w[2:4])  # rank 1 owns rows 2..3
        real.tp_size, real.tp_rank = 4, 1
        twin = self._linear(8, 4, w)
        twin.tp_size, twin.output_partition_sizes = 1, [8]
        twin.p = torch.nn.Parameter(w, requires_grad=False)

        model = torch.nn.Module()
        model.add_module("m", real)
        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        with caplog.at_level(logging.INFO, logger="atom"):
            obj.verify_post_processing(model, 1)
        msgs = [r.getMessage() for r in caplog.records]
        line = [m for m in msgs if "post-process" in m]
        assert line and "1 column-parallel" in line[0]
        assert "0 over tolerance" in line[0]

    def test_divergence_is_counted_against_tolerance(self, caplog):
        from atom.model_engine.decode_twins import DecodeTwins

        w = torch.randn(8, 4)
        real = self._linear(2, 4, w[2:4])
        real.tp_size, real.tp_rank = 4, 1
        twin = self._linear(8, 4, w.clone())
        twin.w[2:4] += 5.0  # rank 1's slice is wrong
        twin.tp_size, twin.output_partition_sizes = 1, [8]
        twin.p = torch.nn.Parameter(w, requires_grad=False)

        model = torch.nn.Module()
        model.add_module("m", real)
        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        with caplog.at_level(logging.INFO, logger="atom"):
            obj.verify_post_processing(model, 1)
        msgs = [r.getMessage() for r in caplog.records]
        line = [m for m in msgs if "post-process" in m]
        assert line and "1 over tolerance" in line[0]

    def test_reports_and_does_not_raise_on_divergence(self):
        """A wrong post-processed layout must be reported, not crash the boot."""
        from atom.model_engine.decode_twins import DecodeTwins

        w = torch.randn(8, 4)
        real = self._linear(2, 4, w[2:4])
        real.tp_size, real.tp_rank = 4, 1
        twin = self._linear(8, 4, torch.randn(8, 4))  # deliberately wrong
        twin.tp_size = 1
        twin.p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        model = torch.nn.Module()
        model.add_module("m", real)
        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        obj.verify_post_processing(model, 1)  # must not raise

    def test_row_parallel_modules_are_skipped(self):
        """tp_dim==1 would need the TP all-reduce; running one here is unsafe."""
        from atom.model_engine.decode_twins import DecodeTwins

        real = torch.nn.Module()
        real.tp_dim, real.tp_size, real.tp_rank, real.input_size = 1, 4, 0, 4
        real.output_partition_sizes = [2]
        real.forward = lambda x: (_ for _ in ()).throw(AssertionError("called"))
        twin = torch.nn.Module()
        twin.tp_size, twin.output_partition_sizes = 1, [8]

        model = torch.nn.Module()
        model.add_module("m", real)
        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        obj.verify_post_processing(model, 0)  # skipped => forward never called

    def test_unsharded_modules_are_skipped(self):
        from atom.model_engine.decode_twins import DecodeTwins

        real = torch.nn.Module()
        real.tp_dim, real.tp_size, real.input_size = 0, 1, 4
        real.output_partition_sizes = [8]
        real.forward = lambda x: (_ for _ in ()).throw(AssertionError("called"))
        twin = torch.nn.Module()
        twin.tp_size, twin.output_partition_sizes = 1, [8]

        model = torch.nn.Module()
        model.add_module("m", real)
        obj = object.__new__(DecodeTwins)
        obj._twins, obj._bare_full, obj._agree, obj._diag = {"m": twin}, {}, {}, []
        obj.verify_post_processing(model, 0)


# ── Asymmetric rapidserve: post-load module attributes across the IPC ────
#
# The consumer builds on meta and imports; it NEVER runs
# process_weights_after_loading. So every attribute that hook produces must
# travel in the payload, and the consumer's own value is a construction-time
# default rather than a computed result worth preserving. A "only fill what is
# None" policy restored 0 of 1343 modules in practice, because the flags that
# matter default to False, not None.


class TestModuleMetaAttrs:
    @staticmethod
    def _mod(**attrs):
        m = torch.nn.Module()
        for k, v in attrs.items():
            object.__setattr__(m, k, v)
        return m

    def test_false_bools_are_exported(self):
        """is_output_padded defaults False; it must still travel."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(is_output_padded=False))
        assert attrs["is_output_padded"] is False

    def test_true_bools_are_exported(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(is_output_padded=True))
        assert attrs["is_output_padded"] is True

    def test_layout_strings_are_exported(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(w13_swizzle_layout="foo"))
        assert attrs["w13_swizzle_layout"] == "foo"

    def test_padding_width_travels_despite_being_int_and_underscored(self):
        """_output_size_before_padding is read by forward (linear.py:986)."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(_output_size_before_padding=4096))
        assert attrs["_output_size_before_padding"] == 4096

    def test_parallel_config_ints_are_not_carried(self):
        """tp_size/tp_rank differ by design between a TP=8 and a TP=1 process."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(tp_size=8, tp_rank=3))
        assert "tp_size" not in attrs and "tp_rank" not in attrs

    def test_training_flag_is_not_carried(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        m = torch.nn.Module()
        assert "training" not in ipc._module_meta_attrs(m)


class TestStateSlotMasking:
    """The third pool: V4's per-request compressor tail.

    A prefill rank that does not own a row must not write that row's
    kv_state/score_state into its live slot — that slot belongs to whatever
    sequence its co-located decode rank put there. The mask used to live only
    in prepare_decode, where disagg_kv_write_masked is always False, so it never
    ran; the prefill call site is unguarded, so the helper must no-op by itself.
    """

    @staticmethod
    def _builder(masked=True, rank=0, targets=None, nrows=3):
        backends = pytest.importorskip("atom.model_ops.attentions.backends")

        concrete = type(
            "Builder",
            (backends.CommonAttentionBuilder,),
            {
                "prepare_decode": lambda self, batch, bs: None,
                "build_for_cudagraph_capture": lambda self, bs: None,
            },
        )
        b = object.__new__(concrete)
        runner = type("R", (), {})()
        runner.rank = rank
        runner.config = MockConfig(
            disagg_prefill_tp_size=8 if masked else 0, disagg_is_decode=False
        )
        b.model_runner = runner
        batch = type("B", (), {})()
        batch.target_ranks = targets
        batch.block_tables = [[1]] * nrows
        return b, batch

    def test_no_op_outside_asymmetric_rapidserve(self):
        """Called unguarded on the prefill path; must not touch other configs."""
        import numpy as np

        b, batch = self._builder(masked=False, targets=[0, 1, 2])
        slots = np.array([5, 6, 7], dtype=np.int32)
        assert b.disagg_mask_state_slots(batch, slots, 3) is slots

    def test_non_owned_rows_go_to_the_sink(self):
        import numpy as np

        from atom.model_engine.block_manager import DUMP_INDEX

        b, batch = self._builder(rank=1, targets=[0, 1, 2])
        out = b.disagg_mask_state_slots(batch, np.array([5, 6, 7]), 3)
        assert list(out) == [DUMP_INDEX, 6, DUMP_INDEX]

    def test_input_is_not_mutated(self):
        """Callers reuse the staging array; masking must not leak into it."""
        import numpy as np

        b, batch = self._builder(rank=1, targets=[0, 1, 2])
        slots = np.array([5, 6, 7])
        b.disagg_mask_state_slots(batch, slots, 3)
        assert list(slots) == [5, 6, 7]

    def test_no_target_info_dumps_every_row(self):
        import numpy as np

        from atom.model_engine.block_manager import DUMP_INDEX

        b, batch = self._builder(targets=None)
        out = b.disagg_mask_state_slots(batch, np.array([5, 6, 7]), 3)
        assert list(out) == [DUMP_INDEX] * 3

    def test_respects_scheduled_bs_bound(self):
        """Rows past scheduled_bs are not part of this batch."""
        import numpy as np

        b, batch = self._builder(rank=9, targets=[0, 1, 2])
        out = b.disagg_mask_state_slots(batch, np.array([5, 6, 7]), 2)
        assert out[2] == 7

    def test_owned_rows_keep_their_live_slot(self):
        import numpy as np

        b, batch = self._builder(rank=0, targets=[0, 0, 0])
        out = b.disagg_mask_state_slots(batch, np.array([5, 6, 7]), 3)
        assert list(out) == [5, 6, 7]


class TestKvDimShipping:
    """Dims KV binding needs that the CONSUMER cannot derive correctly.

    _prepare_kv_dims computes num_kv_heads as num_key_value_heads // world_size.
    Under asymmetric rapidserve prefill is TP=8 and decode TP=1, so decode
    derives an 8x larger value and would bind views that size over prefill's
    allocation. MLA/V4 never reads it, but the MHA backends size their KV with
    it, so the producer's value has to travel.
    """

    def test_num_kv_heads_is_shipped(self):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        assert "num_kv_heads" in mr._KV_DIM_ATTRS

    def test_swa_and_slot_dims_still_shipped(self):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        assert "num_swa_blocks" in mr._KV_DIM_ATTRS
        assert "max_per_req_cache_slots" in mr._KV_DIM_ATTRS

    def test_asymmetric_world_sizes_derive_different_head_counts(self):
        """The arithmetic that makes shipping necessary."""

        def derive(num_kv_heads_cfg, world_size):
            if num_kv_heads_cfg >= world_size:
                return num_kv_heads_cfg // world_size
            return 1

        assert derive(128, 8) == 16  # prefill
        assert derive(128, 1) == 128  # decode — 8x, and wrong for binding
        assert derive(128, 8) != derive(128, 1)

    def test_symmetric_world_sizes_agree(self):
        """Why symmetric rapidserve never hit this."""

        def derive(num_kv_heads_cfg, world_size):
            if num_kv_heads_cfg >= world_size:
                return num_kv_heads_cfg // world_size
            return 1

        assert derive(128, 8) == derive(128, 8)


class TestPrefixCachingIncompatibility:
    """Asymmetric rapidserve cannot use prefix caching.

    On a hit, prefill skips the cached tokens and reads their KV back from the
    pool — but that prefix lives only on the GPU of the rank that originally
    served the sequence. Other prefill TP ranks read their own GPU at those
    block ids and get unrelated data, and MLA's TP all-reduce spreads it to the
    owning rank too. Write-side masking cannot fix data that is not there.
    """

    @staticmethod
    def _decide(enable_dp_attention, tp, prefix_caching):
        """Mirror of the guard in DisaggCoreManager._spawn_disagg."""
        asymmetric = enable_dp_attention and tp > 1
        return False if (asymmetric and prefix_caching) else prefix_caching

    def test_disabled_under_asymmetric(self):
        assert self._decide(True, 8, True) is False

    def test_left_alone_under_symmetric(self):
        """Symmetric decode shares prefill's TP group and its GPUs."""
        assert self._decide(False, 8, True) is True

    def test_left_alone_without_rapidserve_topology(self):
        assert self._decide(True, 1, True) is True

    def test_already_off_stays_off(self):
        assert self._decide(True, 8, False) is False


class TestPrefillSwaReadArithmetic:
    """Why the SWA sink can be one block while the compressed sink cannot.

    Mirrors paged_prefill_indices.py:116-119. Prefill DOES compute sliding-
    window attention, but its window comes from `extend_indices` — rows in the
    per-forward KV tensor — not from the paged pool. The pool is read only for
    `prefix_swa_count` positions, and chunk_start is num_cached_tokens.

    A mirror test earns its place here because this arithmetic is the sole
    justification for the two pools being sized differently; if the kernel
    changes, the sizing decision has to be revisited.
    """

    @staticmethod
    def _counts(pos, chunk_start, win):
        token_pos_in_chunk = pos - chunk_start
        swa_low = max(pos - win + 1, 0)
        return (
            min(token_pos_in_chunk + 1, win),  # extend_count
            max(chunk_start - swa_low, 0),  # prefix_swa_count
        )

    def test_single_pass_prefill_never_reads_the_pool(self):
        """chunk_start == 0 => prefix_swa_count == 0 for every token."""
        win = 128
        for pos in (0, 1, 127, 128, 5000, 8191):
            _, prefix = self._counts(pos, chunk_start=0, win=win)
            assert prefix == 0, f"pos={pos} would page the SWA pool"

    def test_whole_window_comes_from_the_forward_tensor(self):
        win = 128
        for pos in (0, 63, 127, 128, 8191):
            extend, prefix = self._counts(pos, chunk_start=0, win=win)
            assert extend == min(pos + 1, win)
            assert extend + prefix == min(pos + 1, win)

    def test_only_the_first_win_tokens_of_a_chunk_read_the_pool(self):
        """chunk_start > 0 does NOT mean every token pages the pool.

        Only tokens whose window reaches back past the chunk boundary do, i.e.
        the first `win` of them. Deeper into the chunk the window is fully
        contained and prefix_swa_count returns to 0.
        """
        cs, win = 8192, 128
        assert self._counts(cs, cs, win)[1] == win - 1  # first token: 127
        assert self._counts(cs + 8, cs, win)[1] == 119
        assert self._counts(cs + win - 2, cs, win)[1] == 1  # last one that pages
        assert self._counts(cs + win - 1, cs, win)[1] == 0  # window now inside
        assert self._counts(cs + 800, cs, win)[1] == 0

    def test_prefix_cache_hit_pages_the_pool(self):
        """chunk_start = num_cached_tokens, so a hit pages — hence the guard."""
        _, prefix = self._counts(pos=4096, chunk_start=4096, win=128)
        assert prefix == 127

    def test_pool_read_is_bounded_by_the_window(self):
        """Never more than win-1 positions, whatever the chunk start."""
        for cs in (128, 4096, 8192):
            for off in range(0, 200, 7):
                assert self._counts(cs + off, cs, 128)[1] <= 127


class TestSwaSinkVsScratch:
    """The two pools need OPPOSITE substitutions, and mixing them corrupts.

    Compressed pool: prefill reads back what it writes (the absorbed MLA path
    runs over kv_indices), so dumped rows need distinct scratch blocks and the
    pool reserves a whole region for them.

    SWA pool: prefill never reads back its own window (that comes from the
    per-forward KV tensor), so dumped rows may alias one sink block — and the
    pool reserves exactly one. Handing SWA the region-based substitution indexes
    blocks 1..n_dump, which in that pool are LIVE and belong to other sequences.
    """

    @staticmethod
    def _builder(rank=0, targets=None, masked=True):
        backends = pytest.importorskip("atom.model_ops.attentions.backends")

        concrete = type(
            "Builder",
            (backends.CommonAttentionBuilder,),
            {
                "prepare_decode": lambda self, batch, bs: None,
                "build_for_cudagraph_capture": lambda self, bs: None,
            },
        )
        b = object.__new__(concrete)
        runner = type("R", (), {})()
        runner.rank = rank
        runner.config = MockConfig(
            disagg_prefill_tp_size=8 if masked else 0, disagg_is_decode=False
        )
        b.model_runner = runner
        batch = type("B", (), {})()
        batch.target_ranks = targets
        batch.block_tables = [[1]] * (len(targets) if targets else 0)
        return b, batch

    def test_sink_collapses_every_block_to_the_reserved_index(self):
        from atom.model_engine.block_manager import DUMP_INDEX

        b, batch = self._builder(rank=0, targets=[1, 2])
        out = b._disagg_sink_tables(batch, [[4, 5, 6], [7, 8]])
        assert out == [[DUMP_INDEX] * 3, [DUMP_INDEX] * 2]

    def test_sink_never_names_a_live_block(self):
        """The bug: a region-based substitution hands out 1..n, which are live."""
        from atom.model_engine.block_manager import DUMP_INDEX

        b, batch = self._builder(rank=0, targets=[1, 2, 3])
        out = b._disagg_sink_tables(batch, [[4, 5, 6], [7, 8], [9]])
        assert {blk for row in out for blk in row} == {DUMP_INDEX}

    def test_sink_preserves_row_lengths(self):
        b, batch = self._builder(rank=0, targets=[1, 2])
        src = [[4, 5, 6], [7, 8]]
        out = b._disagg_sink_tables(batch, src)
        assert [len(r) for r in out] == [len(r) for r in src]

    def test_sink_leaves_owned_rows_alone(self):
        """Decode reads these blocks, so the owning rank's write must land."""
        b, batch = self._builder(rank=1, targets=[1, 2])
        out = b._disagg_sink_tables(batch, [[4, 5, 6], [7, 8]])
        assert out[0] == [4, 5, 6]

    def test_sink_is_a_no_op_when_masking_is_off(self):
        b, batch = self._builder(masked=False, targets=[0, 1])
        src = [[4, 5], [6]]
        assert b._disagg_sink_tables(batch, src) is src

    def test_scratch_and_sink_differ_for_the_same_batch(self):
        """Guards against the two being unified again by a later refactor."""
        b, batch = self._builder(rank=0, targets=[1, 2])
        batch.block_tables = [[4, 5, 6], [7, 8]]
        scratch = b.disagg_effective_block_tables(batch)
        sink = b._disagg_sink_tables(batch, batch.block_tables)
        assert scratch != sink
        assert len({blk for row in scratch for blk in row}) == 5  # all distinct
        assert len({blk for row in sink for blk in row}) == 1


class TestV4PrefillPopulatorsAreMasked:
    """V4 prefill uses its OWN block-table populators, not the parent's.

    `_populate_block_tables` / `_populate_swa_block_tables` are duplicates of
    CommonAttentionBuilder.prepare_block_tables that V4's prepare_prefill calls
    unconditionally (the parent's runs only when has_cached, which is never true
    without prefix caching or chunking). Masking added only to the parent is
    silently dead here — which is how the compressed pool stayed unmasked for
    the whole rapidserve prefill path.
    """

    def test_populators_consult_the_masking_helpers(self):
        """Source-level: both must route through the disagg helpers."""
        import inspect

        v4 = pytest.importorskip("atom.model_ops.attentions.deepseek_v4_attn")
        cls = v4.DeepseekV4AttentionMetadataBuilder

        bt = inspect.getsource(cls._populate_block_tables)
        assert "disagg_effective_block_tables" in bt

        swa = inspect.getsource(cls._populate_swa_block_tables)
        assert "_disagg_sink_tables" in swa

    def test_compressed_uses_scratch_and_swa_uses_sink(self):
        """They must NOT use the same helper — opposite pools, opposite rules."""
        import inspect

        v4 = pytest.importorskip("atom.model_ops.attentions.deepseek_v4_attn")
        cls = v4.DeepseekV4AttentionMetadataBuilder

        bt = inspect.getsource(cls._populate_block_tables)
        swa = inspect.getsource(cls._populate_swa_block_tables)
        assert "_disagg_sink_tables" not in bt, "compressed pool is read back"
        assert "disagg_effective_block_tables" not in swa, "SWA pool is not"

    def test_prefill_calls_both_populators(self):
        """If prepare_prefill stops calling these, the masking moves with it."""
        import inspect

        v4 = pytest.importorskip("atom.model_ops.attentions.deepseek_v4_attn")
        src = inspect.getsource(
            v4.DeepseekV4AttentionMetadataBuilder.prepare_prefill
        )
        assert "_populate_block_tables(" in src
        assert "_populate_swa_block_tables(" in src


class TestDisaggMaskLogging:
    """Runtime evidence that the mask ran, throttled so it cannot flood.

    Added after the masking was reasoned about for many iterations without
    anyone confirming it executed — V4 prefill turned out to call a different
    populator, so every source-level argument had been about dead code.
    """

    @staticmethod
    def _builder(rank=0, targets=None, tables=None):
        backends = pytest.importorskip("atom.model_ops.attentions.backends")

        concrete = type(
            "Builder",
            (backends.CommonAttentionBuilder,),
            {
                "prepare_decode": lambda self, batch, bs: None,
                "build_for_cudagraph_capture": lambda self, bs: None,
            },
        )
        b = object.__new__(concrete)
        runner = type("R", (), {})()
        runner.rank = rank
        runner.config = MockConfig(
            disagg_prefill_tp_size=8, disagg_is_decode=False
        )
        b.model_runner = runner
        batch = type("B", (), {})()
        batch.target_ranks = targets
        batch.block_tables = tables or []
        return b, batch

    def test_logs_owned_and_dumped_counts(self, caplog):
        b, batch = self._builder(rank=1, targets=[0, 1, 2], tables=[[1], [2], [3]])
        with caplog.at_level(logging.INFO, logger="atom"):
            b.disagg_effective_block_tables(batch)
        msgs = [m.getMessage() for m in caplog.records]
        line = [m for m in msgs if "DISAGG-MASK" in m]
        assert line and "owned=1 dumped=2" in line[0]

    def test_throttled_after_a_few_batches(self, caplog):
        b, batch = self._builder(rank=0, targets=[0], tables=[[1]])
        with caplog.at_level(logging.INFO, logger="atom"):
            for _ in range(10):
                b.disagg_effective_block_tables(batch)
        lines = [m for m in caplog.records if "DISAGG-MASK" in m.getMessage()]
        assert len(lines) == 3

    def test_mixed_batches_get_their_own_budget(self, caplog):
        """Single-row batches must not exhaust the budget for mixed ones.

        The case the per-row scratch region exists for is rows with DIFFERENT
        target ranks in one batch. Early batches are all single-row, so a shared
        counter hides exactly the case worth seeing.
        """
        b, batch = self._builder(rank=0, targets=[0], tables=[[1]])
        with caplog.at_level(logging.INFO, logger="atom"):
            for _ in range(10):
                b.disagg_effective_block_tables(batch)
            batch.target_ranks = [0, 5]
            batch.block_tables = [[1], [2]]
            for _ in range(10):
                b.disagg_effective_block_tables(batch)
        msgs = [m.getMessage() for m in caplog.records]
        lines = [m for m in msgs if "DISAGG-MASK" in m]
        assert len([m for m in lines if "2 row(s)" in m]) == 3
        assert len([m for m in lines if "1 row(s)" in m]) == 3

    def test_reports_the_target_ranks_it_saw(self, caplog):
        """All-zero targets would mean assignments never reached prefill."""
        b, batch = self._builder(rank=0, targets=[0, 5], tables=[[1], [2]])
        with caplog.at_level(logging.INFO, logger="atom"):
            b.disagg_effective_block_tables(batch)
        msgs = [m.getMessage() for m in caplog.records]
        line = [m for m in msgs if "DISAGG-MASK" in m]
        assert "[0, 5]" in line[0]
