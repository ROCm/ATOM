# SPDX-License-Identifier: MIT
# Tests for per-request cache decoupling: unified block pool + per-request
# slot management. The first user is GDN recurrent state (Qwen3-Next /
# Qwen3.5); the same infra serves any future stateful attention type
# (e.g. DeepseekV4 ring buffer + compressor state) via the
# AttentionMetadataBuilder.compute_per_req_cache_bytes() hook.


from conftest import MockConfig
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence
from atom.model_engine.scheduler import Scheduler, ScheduledBatch

# ── helpers ────────────────────────────────────────────────────────────────


def per_req_cache_config(**overrides):
    """Config with per-request cache slot management enabled."""
    defaults = dict(
        kv_cache_block_size=4,
        num_kvcache_blocks=100,
        enable_prefix_caching=False,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        bos_token_id=1,
        eos_token_id=2,
        stop_token_ids=[],
        scheduler_delay_factor=0.0,
        speculative_config=None,
        per_req_cache_equiv_blocks=5,  # each stateful request costs 5 equiv blocks
        num_per_req_cache_groups=8,  # max 8 concurrent stateful requests
    )
    defaults.update(overrides)
    return MockConfig(**defaults)


def stateful_seq(token_ids, block_size=4, **kwargs):
    return Sequence(token_ids, block_size, has_per_req_cache=True, **kwargs)


def plain_seq(token_ids, block_size=4, **kwargs):
    return Sequence(token_ids, block_size, has_per_req_cache=False, **kwargs)


# ── BlockManager: per-req cache slot allocation ────────────────────────────


class TestBlockManagerPerReqCacheSlots:

    def test_disabled_no_slots(self):
        """Stateless config: no slots allocated, behaves like before."""
        bm = BlockManager(MockConfig(num_kvcache_blocks=50))
        assert len(bm.free_per_req_cache_groups) == 0
        assert bm.per_req_cache_equiv_blocks == 0

    def test_enabled_has_slots(self):
        bm = BlockManager(per_req_cache_config())
        assert len(bm.free_per_req_cache_groups) == 8
        assert bm.per_req_cache_equiv_blocks == 5

    def test_allocate_assigns_slot(self):
        bm = BlockManager(per_req_cache_config())
        seq = stateful_seq([1, 2, 3, 4])
        bm.allocate(seq)
        assert seq.per_req_cache_group >= 0
        assert seq.per_req_cache_group < 8
        assert len(bm.free_per_req_cache_groups) == 7

    def test_allocate_deducts_equiv_blocks(self):
        bm = BlockManager(per_req_cache_config())
        initial_free = len(bm.free_block_ids_set)
        seq = stateful_seq([1, 2, 3, 4])  # 1 KV block
        bm.allocate(seq)
        # 1 KV block + 5 equiv blocks = 6 total deducted
        assert len(bm.free_block_ids_set) == initial_free - 6
        assert seq.id in bm.per_req_cache_accounting
        assert len(bm.per_req_cache_accounting[seq.id]) == 5

    def test_deallocate_returns_slot_and_blocks(self):
        bm = BlockManager(per_req_cache_config())
        initial_free = len(bm.free_block_ids_set)
        seq = stateful_seq([1, 2, 3, 4])
        bm.allocate(seq)
        bm.deallocate(seq)
        assert seq.per_req_cache_group == -1
        assert len(bm.free_block_ids_set) == initial_free
        assert len(bm.free_per_req_cache_groups) == 8
        assert seq.id not in bm.per_req_cache_accounting

    def test_can_allocate_checks_both_kv_and_slot(self):
        """can_allocate must check KV blocks AND per-req cache slots."""
        bm = BlockManager(per_req_cache_config(num_kvcache_blocks=100))
        seq = stateful_seq([1, 2, 3, 4])
        assert bm.can_allocate(seq) is True

    def test_can_allocate_fails_not_enough_blocks(self):
        """Not enough free blocks for KV + per-req cache equiv."""
        bm = BlockManager(per_req_cache_config(num_kvcache_blocks=5))
        seq = stateful_seq([1, 2, 3, 4])  # needs 1 KV + 5 equiv = 6 blocks
        assert bm.can_allocate(seq) is False

    def test_can_allocate_fails_no_free_slots(self):
        """All per-req cache slots exhausted."""
        bm = BlockManager(per_req_cache_config(num_per_req_cache_groups=1))
        seq1 = stateful_seq([1, 2, 3, 4])
        bm.allocate(seq1)
        seq2 = stateful_seq([5, 6, 7, 8])
        assert bm.can_allocate(seq2) is False

    def test_plain_seq_ignores_per_req_cache(self):
        """Stateless sequence should not consume per-req cache slots."""
        bm = BlockManager(per_req_cache_config())
        initial_slots = len(bm.free_per_req_cache_groups)
        seq = plain_seq([1, 2, 3, 4])
        bm.allocate(seq)
        assert seq.per_req_cache_group == -1
        assert len(bm.free_per_req_cache_groups) == initial_slots
        assert seq.id not in bm.per_req_cache_accounting

    def test_multiple_allocate_deallocate(self):
        """Allocate and deallocate multiple stateful sequences."""
        bm = BlockManager(per_req_cache_config(num_kvcache_blocks=200))
        seqs = [stateful_seq([1, 2, 3, 4], id=i + 100) for i in range(8)]
        slots = set()
        for seq in seqs:
            bm.allocate(seq)
            slots.add(seq.per_req_cache_group)
        # All 8 slots used
        assert len(slots) == 8
        assert len(bm.free_per_req_cache_groups) == 0

        # Deallocate all
        for seq in seqs:
            bm.deallocate(seq)
        assert len(bm.free_per_req_cache_groups) == 8

    def test_slot_reuse_after_dealloc(self):
        """Freed slots can be reused."""
        bm = BlockManager(
            per_req_cache_config(num_per_req_cache_groups=2, num_kvcache_blocks=200)
        )
        s1 = stateful_seq([1, 2, 3, 4])
        s2 = stateful_seq([5, 6, 7, 8])
        bm.allocate(s1)
        bm.allocate(s2)
        assert len(bm.free_per_req_cache_groups) == 0

        slot1 = s1.per_req_cache_group
        bm.deallocate(s1)
        assert len(bm.free_per_req_cache_groups) == 1

        s3 = stateful_seq([9, 10, 11, 12])
        bm.allocate(s3)
        assert s3.per_req_cache_group == slot1  # reused

    def test_dynamic_competition(self):
        """KV and per-req cache compete for same pool —
        a long sequence reduces the per-req cache capacity."""
        bm = BlockManager(
            per_req_cache_config(num_kvcache_blocks=20, per_req_cache_equiv_blocks=5)
        )
        # Allocate a long plain sequence (16 tokens → 4 KV blocks)
        long_seq = plain_seq(list(range(16)))
        bm.allocate(long_seq)
        # 20 - 4 = 16 free blocks
        # stateful seq needs 1 KV + 5 equiv = 6 blocks
        assert bm.can_allocate(stateful_seq([1, 2, 3, 4]))
        s1 = stateful_seq([1, 2, 3, 4])
        bm.allocate(s1)  # 16 - 6 = 10 free
        s2 = stateful_seq([1, 2, 3, 4])
        bm.allocate(s2)  # 10 - 6 = 4 free
        s3 = stateful_seq([1, 2, 3, 4])
        assert bm.can_allocate(s3) is False  # 4 < 6


# ── Sequence: per_req_cache_group field ──────────────────────────────────────


class TestSequencePerReqCacheSlot:

    def test_default_slot_negative(self):
        seq = Sequence([1, 2, 3], 4, has_per_req_cache=True)
        assert seq.per_req_cache_group == -1
        assert seq.has_per_req_cache is True

    def test_plain_seq_no_slot(self):
        seq = Sequence([1, 2, 3], 4, has_per_req_cache=False)
        assert seq.per_req_cache_group == -1
        assert seq.has_per_req_cache is False

    def test_no_legacy_num_mamba_blocks(self):
        """Legacy `num_mamba_blocks` field must not exist (replaced by
        the per-req cache slot mechanism)."""
        seq = Sequence([1, 2, 3], 4, has_per_req_cache=True)
        assert not hasattr(seq, "num_mamba_blocks")


# ── ScheduledBatch: per_req_cache_groups ─────────────────────────────────────


class TestScheduledBatchPerReqCacheSlots:

    def test_per_req_cache_slots_collected(self):
        s1 = stateful_seq([1, 2, 3, 4])
        s1.per_req_cache_group = 3
        s1.status = s1.status  # keep as WAITING
        s2 = plain_seq([5, 6, 7, 8])
        seqs = {s1.id: s1, s2.id: s2}
        batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=[4, 4],
            total_tokens_num=8,
            total_seqs_num=2,
            total_seqs_num_prefill=2,
        )
        assert batch.per_req_cache_groups == [3]

    def test_no_stateful_seqs(self):
        s1 = plain_seq([1, 2, 3, 4])
        seqs = {s1.id: s1}
        batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=[4],
            total_tokens_num=4,
            total_seqs_num=1,
            total_seqs_num_prefill=1,
        )
        assert batch.per_req_cache_groups == []


# ── State index mapping ──────────────────────────────────────────────────


class TestStateIndexMapping:
    """Verify the slot_group → tensor index mapping logic used in gdn_attn
    (and in the future, any other per-req-cache attention builder)."""

    def test_non_spec_mapping(self):
        """Non-spec: tensor_index = slot_group * slots_per_group."""
        slots_per_group = 4  # 1 + 3 spec
        slot_group = 7
        base = slot_group * slots_per_group
        assert base == 28

    def test_spec_mapping(self):
        """Spec: contiguous indices [base, base+1, ..., base+num_spec]."""
        num_spec = 3
        slots_per_group = 1 + num_spec
        slot_group = 5
        base = slot_group * slots_per_group
        indices = list(range(base, base + 1 + num_spec))
        assert indices == [20, 21, 22, 23]

    def test_all_indices_in_range(self):
        """All generated indices must be < max_per_req_cache_slots."""
        max_num_seqs = 256
        num_spec = 3
        slots_per_group = 1 + num_spec
        max_per_req_cache_slots = max_num_seqs * slots_per_group
        # Check the last group
        last_group = max_num_seqs - 1
        base = last_group * slots_per_group
        indices = list(range(base, base + 1 + num_spec))
        assert all(0 <= i < max_per_req_cache_slots for i in indices)
        assert indices[-1] == max_per_req_cache_slots - 1


# ── Scheduler integration ────────────────────────────────────────────────


class TestSchedulerPerReqCacheIntegration:

    def test_prefill_stateful_seq(self):
        """Scheduler prefill allocates a per-req cache slot via block_manager."""
        sched = Scheduler(per_req_cache_config(num_kvcache_blocks=100))
        seq = stateful_seq([1, 2, 3, 4])
        sched.add(seq)
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 1
        assert seq.per_req_cache_group >= 0
        assert len(batch.per_req_cache_groups) == 1

    def test_preempt_releases_slot(self):
        """Preempted stateful sequence releases its per-req cache slot."""
        sched = Scheduler(per_req_cache_config(num_kvcache_blocks=100))
        seq = stateful_seq([1, 2, 3, 4])
        sched.add(seq)
        sched.schedule()
        assert seq.per_req_cache_group >= 0
        initial_slots = len(sched.block_manager.free_per_req_cache_groups)
        sched.preempt(seq)
        assert seq.per_req_cache_group == -1
        assert len(sched.block_manager.free_per_req_cache_groups) == initial_slots + 1

    def test_slot_exhaustion_blocks_prefill(self):
        """When all per-req cache slots are used, new stateful requests wait."""
        sched = Scheduler(
            per_req_cache_config(num_kvcache_blocks=200, num_per_req_cache_groups=2)
        )
        s1 = stateful_seq([1, 2, 3, 4])
        s2 = stateful_seq([5, 6, 7, 8])
        s3 = stateful_seq([9, 10, 11, 12])
        sched.extend([s1, s2, s3])
        batch, _ = sched.schedule()
        # Only 2 slots → only 2 prefilled
        assert batch.total_seqs_num_prefill == 2
        assert sched.get_num_unfinished_requests() == 3

    def test_mixed_stateful_and_plain(self):
        """Stateful and plain sequences coexist —
        plain sequences don't consume per-req cache slots."""
        sched = Scheduler(
            per_req_cache_config(num_kvcache_blocks=200, num_per_req_cache_groups=2)
        )
        s1 = stateful_seq([1, 2, 3, 4])
        s2 = plain_seq([5, 6, 7, 8])
        s3 = stateful_seq([9, 10, 11, 12])
        s4 = plain_seq([13, 14, 15, 16])
        sched.extend([s1, s2, s3, s4])
        batch, _ = sched.schedule()
        # All 4 should prefill — only 2 per-req cache slots needed
        assert batch.total_seqs_num_prefill == 4
