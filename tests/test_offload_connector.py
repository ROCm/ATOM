# SPDX-License-Identifier: MIT
# Tests for atom/kv_transfer/offload/* — focused on the OFFLOAD admission
# and load-queue path. GPU-dependent round-trip behavior is exercised by
# ad-hoc scripts (see project/005-kvcache-offload notes) since the CI
# environment mocks torch.cuda.

import os
import importlib.util
import sys
import types
from collections import deque

import pytest

# Connector __init__ enforces this, and it must be set before the module
# is imported (tests inherit the env from pytest's own process).
os.environ.setdefault("PYTHONHASHSEED", "0")

from atom.kv_transfer.disaggregation.base import KVConnectorRole
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload import (  # noqa: F401  (registers factory entry)
    OffloadConnectorBase,
    OffloadConnectorMetadata,
    OffloadConnectorSchedulerBase,
    OffloadReqMeta,
)
from atom.kv_transfer.offload.lmcache.lmcache_connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceStatus
from conftest import MockConfig

_OFFLOAD_CFG = MockConfig(
    kv_transfer_config={
        "kv_connector": "lmcache_offload",
        "kv_role": "offload",
        "cpu_bytes": 1024 * 1024 * 64,
    },
    tensor_parallel_size=1,
    enable_prefix_caching=True,
)


@pytest.fixture(autouse=True)
def stub_lmcache_and_torch(monkeypatch):
    """Keep offload unit tests independent of optional GPU/runtime deps."""
    if importlib.util.find_spec("lmcache") is None:
        fake_lmcache = types.SimpleNamespace(
            c_ops=object(), _backend_module="lmcache.c_ops"
        )
        monkeypatch.setitem(sys.modules, "lmcache", fake_lmcache)

    if importlib.util.find_spec("torch") is None:
        fake_torch = types.SimpleNamespace(
            uint8=object(),
            empty=lambda num_bytes, dtype, pin_memory: types.SimpleNamespace(
                data_ptr=lambda: 0
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)


# ── Factory wiring ────────────────────────────────────────────────────────


class TestFactoryRegistration:
    def test_lmcache_offload_registered(self):
        assert "lmcache_offload" in KVConnectorFactory._registry

    def test_worker_role_is_offload(self):
        w = KVConnectorFactory.create_connector(_OFFLOAD_CFG, role="worker")
        assert isinstance(w, LMCacheOffloadConnector)
        assert w.role == KVConnectorRole.OFFLOAD
        assert w.is_producer is False  # OFFLOAD is neither PD producer nor consumer

    def test_scheduler_role_is_offload(self):
        s = KVConnectorFactory.create_connector(_OFFLOAD_CFG, role="scheduler")
        assert isinstance(s, LMCacheOffloadConnectorScheduler)
        assert s.role == KVConnectorRole.OFFLOAD


# ── Scheduler-side: queue_save + mirror ───────────────────────────────────


class TestSchedulerSaveQueue:
    def test_queue_save_populates_mirror(self):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.queue_save("r1", [(11, 0xABCD), (12, 0xBCDE)])
        assert s.saved_hashes == {0xABCD, 0xBCDE}
        assert "r1" in s._pending_save
        assert s._pending_save["r1"].block_hashes == [0xABCD, 0xBCDE]

    def test_queue_save_multi_step_accumulates(self):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.queue_save("r1", [(11, 0xAAA)])
        s.queue_save("r1", [(12, 0xBBB)])
        assert s._pending_save["r1"].block_ids == [11, 12]
        assert s._pending_save["r1"].block_hashes == [0xAAA, 0xBBB]

    def test_queue_save_empty_is_noop(self):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.queue_save("r1", [])
        assert s.saved_hashes == set()
        assert s._pending_save == {}

    def test_build_connector_meta_drains(self):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.queue_save("r1", [(11, 0xAAA)])
        meta1 = s.build_connector_meta()
        assert meta1 is not None
        assert "r1" in meta1.reqs_to_save
        meta2 = s.build_connector_meta()
        assert meta2 is None  # drained


class TestSchedulerOffloadBlockLifetime:
    def test_finished_seq_with_pending_offload_save_defers_block_free(self, seq_factory):
        scheduler = _scheduler_with_offload_connector()
        seq = _seq_with_prompt(seq_factory, [10, 11, 12, 13, 14, 15, 16, 17])
        scheduler.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        fwd_output = ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(scheduler.eos_token_id,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        )

        finished = scheduler.postprocess([seq], fwd_output)

        assert finished == [seq]
        assert str(seq.id) in scheduler.offload_pending_save_req_ids
        assert str(seq.id) in scheduler.deferred_free_blocks
        assert seq.block_table

    def test_offload_save_completion_releases_deferred_blocks(self, seq_factory):
        scheduler = _scheduler_with_offload_connector()
        seq = _seq_with_prompt(seq_factory, [10, 11, 12, 13, 14, 15, 16, 17])
        scheduler.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        fwd_output = ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(scheduler.eos_token_id,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        )
        scheduler.postprocess([seq], fwd_output)

        scheduler._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_sending={str(seq.id)})
        )

        assert str(seq.id) not in scheduler.offload_pending_save_req_ids
        assert seq.id not in scheduler.deferred_free_blocks
        assert seq.block_table == []

    def test_finished_seq_without_offload_save_frees_immediately(self, seq_factory):
        scheduler = _scheduler_with_offload_connector()
        seq = _seq_with_prompt(seq_factory, [10, 11, 12])
        scheduler.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        fwd_output = ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(scheduler.eos_token_id,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        )

        scheduler.postprocess([seq], fwd_output)

        assert str(seq.id) not in scheduler.offload_pending_save_req_ids
        assert seq.id not in scheduler.deferred_free_blocks
        assert seq.block_table == []

    def test_offload_save_completion_before_finish_allows_immediate_free(
        self, seq_factory
    ):
        scheduler = _scheduler_with_offload_connector()
        seq = _seq_with_prompt(seq_factory, [10, 11, 12, 13, 14, 15, 16, 17])
        scheduler.block_manager.allocate(seq)
        scheduler.kv_connector.queue_save(str(seq.id), [(seq.block_table[0], 0xAAA)])
        scheduler.offload_pending_save_req_ids.add(str(seq.id))

        scheduler._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_sending={str(seq.id)})
        )

        seq.prefix_hashes_published = True
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
        fwd_output = ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(scheduler.eos_token_id,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        )
        scheduler.postprocess([seq], fwd_output)

        assert str(seq.id) not in scheduler.offload_pending_save_req_ids
        assert seq.id not in scheduler.deferred_free_blocks
        assert seq.block_table == []


# ── Scheduler-side: lookup + admission path ───────────────────────────────


def _seq_with_prompt(seq_factory, token_ids, block_size=4):
    return seq_factory(token_ids, block_size=block_size)


def _scheduler_with_offload_connector():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_manager = BlockManager(_OFFLOAD_CFG)
    scheduler.running = deque()
    scheduler.eos_token_id = _OFFLOAD_CFG.eos_token_id
    scheduler.stop_token_ids = _OFFLOAD_CFG.stop_token_ids
    scheduler.use_spec = False
    scheduler.mtp_k = 0
    scheduler.spec_stats = None
    scheduler.deferred_free_blocks = {}
    scheduler.offload_pending_save_req_ids = set()
    scheduler.finished_recving_kv_req_ids = []
    scheduler.kv_connector = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
    scheduler.kv_connector.bind_block_manager(scheduler.block_manager)
    return scheduler


class TestSchedulerLookup:
    def test_lookup_external_hits_chain_prefix(self):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.saved_hashes = {1, 2, 3}
        # Longest matching prefix from [1, 2, 99, 3]: [1, 2] then break.
        assert s.lookup_external_hits(None, [1, 2, 99, 3]) == 2
        assert s.lookup_external_hits(None, [1, 2, 3]) == 3
        assert s.lookup_external_hits(None, [99, 1]) == 0

    def test_get_num_new_matched_tokens_no_block_manager(self, seq_factory):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        seq = _seq_with_prompt(seq_factory, [1, 2, 3, 4, 5, 6, 7, 8])
        n_tok, async_load = s.get_num_new_matched_tokens(seq)
        assert (n_tok, async_load) == (0, False)

    def test_get_num_new_matched_tokens_no_external_hits(
        self, block_manager_prefix, seq_factory
    ):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.bind_block_manager(block_manager_prefix)
        # block_size=4 by MockConfig default; 3 blocks of distinct content.
        seq = _seq_with_prompt(
            seq_factory, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        )
        n_tok, async_load = s.get_num_new_matched_tokens(seq)
        assert (n_tok, async_load) == (0, False)

    def test_get_num_new_matched_tokens_external_after_eviction(
        self, block_manager_prefix, seq_factory
    ):
        """Simulate: prior request published 2 blocks to HBM + OFFLOAD.
        HBM then evicted them (manually clear hash_to_block_id) but
        OFFLOAD mirror retained them. New same-prefix request should see
        2 blocks worth of external hits.
        """
        bm = block_manager_prefix
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.bind_block_manager(bm)

        # 3-block prompt; last block is never considered by can_allocate
        # so only first 2 blocks are eligible for prefix-cache reuse.
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12 tokens / bs=4 → 3 blocks
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([5, 6, 7, 8], prefix=h1)
        # OFFLOAD mirror has both (as if a prior request had saved them).
        s.saved_hashes = {h1, h2}
        # HBM is empty (simulates eviction).
        assert h1 not in bm.hash_to_block_id

        seq = _seq_with_prompt(seq_factory, tokens)
        n_tok, async_load = s.get_num_new_matched_tokens(seq)
        # 2 external blocks × block_size=4 = 8 tokens.
        assert async_load is True
        assert n_tok == 8
        # Stash should hold the 2 hashes for update_state_after_alloc.
        stashed = s._external_match_stash[id(seq)]
        assert stashed == [h1, h2]

    def test_get_num_new_matched_tokens_hbm_takes_precedence(
        self, block_manager_prefix, seq_factory
    ):
        """If HBM has the first block and OFFLOAD has both, we should
        report only the residual (block 2) as external — block 1 is free
        in HBM, no need to H2D it.
        """
        bm = block_manager_prefix
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.bind_block_manager(bm)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([5, 6, 7, 8], prefix=h1)

        # Seed HBM with block 1 only — use a dummy seq to allocate +
        # hash, then mimic the post-publish state.
        warmup_seq = _seq_with_prompt(seq_factory, [1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(warmup_seq)
        bm.hash_blocks(warmup_seq, warmup_seq.num_tokens)
        # Confirm both block hashes are in HBM now (block 0 and 1, since
        # hash_blocks skips the last partial — but this seq has 2 full
        # blocks; the "last is never considered for reuse" rule lives in
        # can_allocate, not hash_blocks).
        assert h1 in bm.hash_to_block_id

        # OFFLOAD mirror has h2 too (from a hypothetical earlier save).
        s.saved_hashes = {h1, h2}
        # Simulate eviction of block 2 only (clear h2 from HBM).
        if h2 in bm.hash_to_block_id:
            del bm.hash_to_block_id[h2]

        new_seq = _seq_with_prompt(seq_factory, tokens)
        n_tok, async_load = s.get_num_new_matched_tokens(new_seq)
        # Block 1 = HBM hit (free), block 2 = external (needs load) → 4 tokens.
        assert async_load is True
        assert n_tok == 4


# ── Scheduler-side: update_state_after_alloc → load queue ────────────────


class TestSchedulerLoadQueue:
    def test_update_state_after_alloc_queues_load(
        self, block_manager_prefix, seq_factory
    ):
        bm = block_manager_prefix
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.bind_block_manager(bm)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([5, 6, 7, 8], prefix=h1)
        s.saved_hashes = {h1, h2}

        seq = _seq_with_prompt(seq_factory, tokens)
        n_tok, async_load = s.get_num_new_matched_tokens(seq)
        assert async_load and n_tok == 8

        # Mimic admission: BlockManager allocates fresh blocks for all
        # 3 prompt blocks (HBM was empty).
        bm.allocate(seq)
        s.update_state_after_alloc(seq)

        assert str(seq.id) in s._pending_load
        meta = s._pending_load[str(seq.id)]
        assert meta.block_hashes == [h1, h2]
        # dest_block_ids are seq.block_table[n_hbm : n_hbm + n_ext]; n_hbm=0
        # here so the first 2 fresh slots.
        assert meta.block_ids == list(seq.block_table[:2])

    def test_no_stash_after_alloc_is_noop(self, block_manager_prefix, seq_factory):
        s = LMCacheOffloadConnectorScheduler(_OFFLOAD_CFG)
        s.bind_block_manager(block_manager_prefix)
        seq = _seq_with_prompt(seq_factory, [1, 2, 3, 4])
        s.update_state_after_alloc(seq)  # never went through lookup
        assert s._pending_load == {}


# ── Worker-side: pool sizing + degenerate paths ──────────────────────────


class TestWorkerPool:
    def test_environment_check_rejects_bad_pythonhashseed(self, monkeypatch):
        monkeypatch.setenv("PYTHONHASHSEED", "42")
        with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
            LMCacheOffloadConnector(_OFFLOAD_CFG)

    def test_start_load_kv_before_pool_init_drops(self):
        w = LMCacheOffloadConnector(_OFFLOAD_CFG)
        # register_kv_caches never called — pool is empty.
        assert w._num_slots == 0
        meta = OffloadConnectorMetadata()
        meta.reqs_to_save["rX"] = OffloadReqMeta(block_ids=[1], block_hashes=[0xAAA])
        meta.reqs_to_load["rY"] = OffloadReqMeta(block_ids=[2], block_hashes=[0xBBB])
        w.start_load_kv(meta)
        done_save, done_load = w.get_finished()
        # Both reported as done so the scheduler doesn't stall, but no
        # GPU motion happened.
        assert done_save == {"rX"}
        assert done_load == {"rY"}

    def test_start_load_kv_empty_metadata_noop(self):
        w = LMCacheOffloadConnector(_OFFLOAD_CFG)
        w.start_load_kv(None)
        w.start_load_kv(OffloadConnectorMetadata())
        done_save, done_load = w.get_finished()
        assert done_save == set() and done_load == set()

    def test_register_kv_caches_mla_uses_block_bytes(self, monkeypatch):
        """MLA k_cache is token-major: [num_blocks * block_size, 1, 576].

        Offload slots must copy one logical paged block, not one token row.
        """

        class FakeTensor:
            is_cuda = True

            def __init__(
                self,
                shape=(8, 1, 576),
                stride=(576, 576, 1),
                element_size=2,
            ):
                self.shape = shape
                self._stride = stride
                self._element_size = element_size
                self.device = "cuda:0"

            def is_contiguous(self):
                return True

            def numel(self):
                n = 1
                for dim in self.shape:
                    n *= dim
                return n

            def element_size(self):
                return self._element_size

            def stride(self, dim):
                return self._stride[dim]

            def view(self, *args):
                return self

            def data_ptr(self):
                return 0xCAFE

        class FakeKV:
            k_cache = FakeTensor()
            v_cache = None

        class FakeCPUPool(FakeTensor):
            def __init__(self, num_bytes):
                super().__init__(shape=(num_bytes,), stride=(1,), element_size=1)

        fake_torch = types.SimpleNamespace(
            uint8=object(),
            empty=lambda num_bytes, dtype, pin_memory: FakeCPUPool(num_bytes),
        )
        fake_lmcache = types.SimpleNamespace(
            c_ops=object(), _backend_module="lmcache.c_ops"
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "lmcache", fake_lmcache)

        w = LMCacheOffloadConnector(_OFFLOAD_CFG)
        w.register_kv_caches({"layer0": FakeKV()})

        # block_size=4, row bytes=576 * bf16(2) => one MLA block is 4608 bytes.
        assert w._k_bytes_per_layer == [4 * 576 * 2]
        assert w._v_bytes_per_layer == [0]
        assert w.bytes_per_block == 4 * 576 * 2
