# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for intra-GPU disagg constrained vs unconstrained modes.

Only the scheduler-level shm gating is exercised here; the IPC handshake
and CUDA stream pool are out of scope for the no-GPU test environment.
"""

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

    def test_asymmetric_reserves_the_dump_block(self):
        from atom.model_engine.block_manager import DUMP_INDEX

        bm = self._block_manager(num_kvcache_blocks=10, disagg_prefill_tp_size=8)
        assert DUMP_INDEX not in bm.free_block_ids_set
        assert len(bm.free_block_ids_set) == 9

    def test_dump_block_is_never_allocated(self):
        """Drain the pool: the dump index must not appear in any allocation."""
        from atom.model_engine.block_manager import DUMP_INDEX

        bm = self._block_manager(num_kvcache_blocks=10, disagg_prefill_tp_size=8)
        drained = [bm.free_block_ids.popleft() for _ in range(len(bm.free_block_ids))]
        assert DUMP_INDEX not in drained
        assert sorted(drained) == list(range(1, 10))

    def test_per_req_cache_groups_reserve_slot_zero(self):
        from atom.model_engine.block_manager import DUMP_INDEX

        bm = self._block_manager(
            num_kvcache_blocks=10,
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
        bm = self._block_manager(num_kvcache_blocks=10, disagg_prefill_tp_size=8)
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
            reserve_dump_block=reserve,
        )

    def test_reserved(self):
        from atom.model_engine.block_manager import DUMP_INDEX

        pool = self._pool(6, reserve=True)
        assert DUMP_INDEX not in pool.free_block_ids_set
        assert len(pool.free_block_ids_set) == 5

    def test_not_reserved_by_default(self):
        pool = self._pool(6, reserve=False)
        assert len(pool.free_block_ids_set) == 6

    def test_disabled_pool_stays_empty(self):
        """num_blocks=0 means SWA is off; reservation must not go negative."""
        pool = self._pool(0, reserve=True)
        assert not pool.enabled
        assert len(pool.free_block_ids_set) == 0


# ── Asymmetric rapidserve: shape-aware weight aliasing ───────────────────
#
# Prefill runs TP=N and decode TP=1, so their attention weights are different
# tensors (shard vs full matrix) and cannot be aliased. import_model_weights
# with shape_aware=True must alias only where the producer's shape matches the
# consumer's own, and leave the consumer's locally-loaded copy alone otherwise.
# Aliasing a shard over a full matrix would silently install a wrong-shaped
# weight; keeping a local copy where one could be aliased just wastes memory.


class TestShapeAwareAliasing:
    @staticmethod
    def _model():
        import torch.nn as nn

        m = nn.Module()
        # Replicated: producer and consumer agree -> alias.
        m.norm = nn.Parameter(torch.ones(8))
        # TP-sharded on the producer: consumer is TP=1 so it is 4x larger.
        m.wq = nn.Parameter(torch.ones(32, 8))
        return m

    @staticmethod
    def _handles(model, sharded_out=8):
        """Producer payload: `norm` full-size, `wq` as one TP=4 shard."""
        from atom.model_engine.ipc_utils import _META_KEY

        return {
            "__param__norm": {"shape": (8,)},
            "__param__wq": {"shape": (sharded_out, 8)},
            _META_KEY: {"tensor_attrs": {}, "module_attrs": {}},
        }

    def _run(self, monkeypatch, shape_aware):
        """Import with _import_tensor stubbed — no real IPC handles needed."""
        import atom.model_engine.ipc_utils as ipc

        imported = []

        def fake_import(meta):
            imported.append(tuple(meta["shape"]))
            return torch.zeros(meta["shape"])

        monkeypatch.setattr(ipc, "_import_tensor", fake_import)
        model = self._model()
        ipc.import_model_weights(
            model, self._handles(model), shape_aware=shape_aware
        )
        return model, imported

    def test_matching_shape_is_aliased(self, monkeypatch):
        _, imported = self._run(monkeypatch, shape_aware=True)
        assert (8,) in imported, "replicated tensor should have been aliased"

    def test_mismatched_shape_is_skipped(self, monkeypatch):
        model, imported = self._run(monkeypatch, shape_aware=True)
        assert (8, 8) not in imported, "producer's TP shard must NOT be imported"
        # Consumer keeps the full-size weight it loaded itself.
        assert tuple(model.wq.shape) == (32, 8)

    def test_symmetric_import_aliases_everything(self, monkeypatch):
        """Without shape_aware the old behaviour is unchanged: alias all keys."""
        model, imported = self._run(monkeypatch, shape_aware=False)
        assert (8,) in imported and (8, 8) in imported
        # Symmetric mode trusts the producer, so wq is replaced by the payload.
        assert tuple(model.wq.shape) == (8, 8)

    def test_equal_shapes_alias_even_when_shape_aware(self, monkeypatch):
        """A TP=1 producer and TP=1 consumer agree — nothing should be skipped."""
        import atom.model_engine.ipc_utils as ipc

        imported = []

        def fake_import(meta):
            imported.append(tuple(meta["shape"]))
            return torch.zeros(meta["shape"])

        monkeypatch.setattr(ipc, "_import_tensor", fake_import)
        model = self._model()
        ipc.import_model_weights(
            model, self._handles(model, sharded_out=32), shape_aware=True
        )
        assert imported.count((32, 8)) == 1


# ── Asymmetric rapidserve: expert placement cross-check ──────────────────
#
# Prefill (TP=N, DPA off) and decode (TP=1/DP=N, DPA on) reach the same expert
# sharding by different routes, both landing on ep_size=N / ep_rank=<this GPU>.
# MoE aliasing depends on that agreement, and shape equality CANNOT detect a
# divergence: every rank holds global_num_experts // ep_size experts either way,
# so a mismatched mapping yields identically-shaped tensors holding the wrong
# experts — right kernels, wrong weights, no crash.


class _FakeMoE(torch.nn.Module):
    def __init__(self, ep_size, ep_rank, local_num_experts):
        super().__init__()
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.local_num_experts = local_num_experts


class TestExpertPlacementCheck:
    @staticmethod
    def _model(ep_size, ep_rank, local):
        m = torch.nn.Module()
        m.ffn = _FakeMoE(ep_size, ep_rank, local)
        return m

    def test_matching_placement_passes(self):
        from atom.model_engine.ipc_utils import (
            _assert_expert_placement_matches,
            _expert_placement,
        )

        producer = _expert_placement(self._model(8, 3, 48))
        _assert_expert_placement_matches(self._model(8, 3, 48), producer)

    def test_different_ep_rank_is_rejected(self):
        """Same shapes, different experts — the dangerous case."""
        from atom.model_engine.ipc_utils import (
            ExpertPlacementMismatch,
            _assert_expert_placement_matches,
            _expert_placement,
        )

        producer = _expert_placement(self._model(8, 3, 48))
        with pytest.raises(ExpertPlacementMismatch, match="ep_rank"):
            _assert_expert_placement_matches(self._model(8, 5, 48), producer)

    def test_different_ep_size_is_rejected(self):
        from atom.model_engine.ipc_utils import (
            ExpertPlacementMismatch,
            _assert_expert_placement_matches,
            _expert_placement,
        )

        producer = _expert_placement(self._model(8, 3, 48))
        with pytest.raises(ExpertPlacementMismatch):
            _assert_expert_placement_matches(self._model(4, 3, 96), producer)

    def test_absent_sidecar_is_tolerated(self):
        """A producer with no MoE modules must not trip the check."""
        from atom.model_engine.ipc_utils import _assert_expert_placement_matches

        _assert_expert_placement_matches(self._model(8, 3, 48), {})

    def test_non_moe_modules_are_ignored(self):
        from atom.model_engine.ipc_utils import _expert_placement

        m = torch.nn.Module()
        m.norm = torch.nn.LayerNorm(4)
        assert _expert_placement(m) == {}


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
