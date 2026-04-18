# SPDX-License-Identifier: MIT
# Tests for GDN recurrent state decoupling: unified block pool + mamba slot management.


from conftest import MockConfig
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence
from atom.model_engine.scheduler import Scheduler, ScheduledBatch

# ── helpers ────────────────────────────────────────────────────────────────


def mamba_config(**overrides):
    """Config with mamba state management enabled."""
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
        mamba_equiv_per_req=5,  # each mamba request costs 5 equiv blocks
        num_mamba_groups=8,  # max 8 concurrent mamba requests
        mamba_slots_per_group=1,  # no spec decode
    )
    defaults.update(overrides)
    return MockConfig(**defaults)


def mamba_seq(token_ids, block_size=4, **kwargs):
    return Sequence(token_ids, block_size, mamba_enabled=True, **kwargs)


def plain_seq(token_ids, block_size=4, **kwargs):
    return Sequence(token_ids, block_size, mamba_enabled=False, **kwargs)


# ── BlockManager: mamba slot allocation ────────────────────────────────────


class TestBlockManagerMambaSlots:

    def test_mamba_disabled_no_slots(self):
        """Non-mamba config: no slots allocated, behaves like before."""
        bm = BlockManager(MockConfig(num_kvcache_blocks=50))
        assert len(bm.free_mamba_slots) == 0
        assert bm.mamba_equiv_per_req == 0

    def test_mamba_enabled_has_slots(self):
        bm = BlockManager(mamba_config())
        assert len(bm.free_mamba_slots) == 8
        assert bm.mamba_equiv_per_req == 5

    def test_allocate_assigns_slot(self):
        bm = BlockManager(mamba_config())
        seq = mamba_seq([1, 2, 3, 4])
        bm.allocate(seq)
        assert seq.mamba_state_slot >= 0
        assert seq.mamba_state_slot < 8
        assert len(bm.free_mamba_slots) == 7

    def test_allocate_deducts_equiv_blocks(self):
        bm = BlockManager(mamba_config())
        initial_free = len(bm.free_block_ids_set)
        seq = mamba_seq([1, 2, 3, 4])  # 1 KV block
        bm.allocate(seq)
        # 1 KV block + 5 equiv blocks = 6 total deducted
        assert len(bm.free_block_ids_set) == initial_free - 6
        assert seq.id in bm.mamba_accounting
        assert len(bm.mamba_accounting[seq.id]) == 5

    def test_deallocate_returns_slot_and_blocks(self):
        bm = BlockManager(mamba_config())
        initial_free = len(bm.free_block_ids_set)
        seq = mamba_seq([1, 2, 3, 4])
        bm.allocate(seq)
        bm.deallocate(seq)
        assert seq.mamba_state_slot == -1
        assert len(bm.free_block_ids_set) == initial_free
        assert len(bm.free_mamba_slots) == 8
        assert seq.id not in bm.mamba_accounting

    def test_can_allocate_checks_both_kv_and_mamba(self):
        """can_allocate must check KV blocks AND mamba slots."""
        bm = BlockManager(mamba_config(num_kvcache_blocks=100))
        seq = mamba_seq([1, 2, 3, 4])
        assert bm.can_allocate(seq) is True

    def test_can_allocate_fails_not_enough_blocks(self):
        """Not enough free blocks for KV + mamba equiv."""
        bm = BlockManager(mamba_config(num_kvcache_blocks=5))
        seq = mamba_seq([1, 2, 3, 4])  # needs 1 KV + 5 equiv = 6 blocks
        assert bm.can_allocate(seq) is False

    def test_can_allocate_fails_no_mamba_slots(self):
        """All mamba slots exhausted."""
        bm = BlockManager(mamba_config(num_mamba_groups=1))
        seq1 = mamba_seq([1, 2, 3, 4])
        bm.allocate(seq1)
        seq2 = mamba_seq([5, 6, 7, 8])
        assert bm.can_allocate(seq2) is False

    def test_plain_seq_ignores_mamba(self):
        """Non-mamba sequence should not use mamba slots."""
        bm = BlockManager(mamba_config())
        initial_slots = len(bm.free_mamba_slots)
        seq = plain_seq([1, 2, 3, 4])
        bm.allocate(seq)
        assert seq.mamba_state_slot == -1
        assert len(bm.free_mamba_slots) == initial_slots
        assert seq.id not in bm.mamba_accounting

    def test_multiple_allocate_deallocate(self):
        """Allocate and deallocate multiple mamba sequences."""
        bm = BlockManager(mamba_config(num_kvcache_blocks=200))
        seqs = [mamba_seq([1, 2, 3, 4], id=i + 100) for i in range(8)]
        slots = set()
        for seq in seqs:
            bm.allocate(seq)
            slots.add(seq.mamba_state_slot)
        # All 8 slots used
        assert len(slots) == 8
        assert len(bm.free_mamba_slots) == 0

        # Deallocate all
        for seq in seqs:
            bm.deallocate(seq)
        assert len(bm.free_mamba_slots) == 8

    def test_slot_reuse_after_dealloc(self):
        """Freed slots can be reused."""
        bm = BlockManager(mamba_config(num_mamba_groups=2, num_kvcache_blocks=200))
        s1 = mamba_seq([1, 2, 3, 4])
        s2 = mamba_seq([5, 6, 7, 8])
        bm.allocate(s1)
        bm.allocate(s2)
        assert len(bm.free_mamba_slots) == 0

        slot1 = s1.mamba_state_slot
        bm.deallocate(s1)
        assert len(bm.free_mamba_slots) == 1

        s3 = mamba_seq([9, 10, 11, 12])
        bm.allocate(s3)
        assert s3.mamba_state_slot == slot1  # reused

    def test_dynamic_competition(self):
        """KV and mamba compete for same pool — long sequence reduces mamba capacity."""
        bm = BlockManager(mamba_config(num_kvcache_blocks=20, mamba_equiv_per_req=5))
        # Allocate a long plain sequence (16 tokens → 4 KV blocks)
        long_seq = plain_seq(list(range(16)))
        bm.allocate(long_seq)
        # 20 - 4 = 16 free blocks
        # mamba seq needs 1 KV + 5 equiv = 6 blocks
        assert bm.can_allocate(mamba_seq([1, 2, 3, 4]))
        s1 = mamba_seq([1, 2, 3, 4])
        bm.allocate(s1)  # 16 - 6 = 10 free
        s2 = mamba_seq([1, 2, 3, 4])
        bm.allocate(s2)  # 10 - 6 = 4 free
        s3 = mamba_seq([1, 2, 3, 4])
        assert bm.can_allocate(s3) is False  # 4 < 6


# ── Sequence: mamba_state_slot field ──────────────────────────────────────


class TestSequenceMambaSlot:

    def test_default_slot_negative(self):
        seq = Sequence([1, 2, 3], 4, mamba_enabled=True)
        assert seq.mamba_state_slot == -1
        assert seq.mamba_enabled is True

    def test_plain_seq_no_slot(self):
        seq = Sequence([1, 2, 3], 4, mamba_enabled=False)
        assert seq.mamba_state_slot == -1
        assert seq.mamba_enabled is False

    def test_no_num_mamba_blocks(self):
        """num_mamba_blocks should no longer exist on Sequence."""
        seq = Sequence([1, 2, 3], 4, mamba_enabled=True)
        assert not hasattr(seq, "num_mamba_blocks")


# ── ScheduledBatch: mamba_state_slots ─────────────────────────────────────


class TestScheduledBatchMambaSlots:

    def test_mamba_state_slots_collected(self):
        s1 = mamba_seq([1, 2, 3, 4])
        s1.mamba_state_slot = 3
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
        assert batch.mamba_state_slots == [3]

    def test_no_mamba_seqs(self):
        s1 = plain_seq([1, 2, 3, 4])
        seqs = {s1.id: s1}
        batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=[4],
            total_tokens_num=4,
            total_seqs_num=1,
            total_seqs_num_prefill=1,
        )
        assert batch.mamba_state_slots == []


# ── State index mapping ──────────────────────────────────────────────────


class TestStateIndexMapping:
    """Verify the slot_group → tensor index mapping logic used in gdn_attn."""

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
        """All generated indices must be < max_mamba_slots."""
        max_num_seqs = 256
        num_spec = 3
        slots_per_group = 1 + num_spec
        max_mamba_slots = max_num_seqs * slots_per_group
        # Check the last group
        last_group = max_num_seqs - 1
        base = last_group * slots_per_group
        indices = list(range(base, base + 1 + num_spec))
        assert all(0 <= i < max_mamba_slots for i in indices)
        assert indices[-1] == max_mamba_slots - 1


# ── Scheduler integration ────────────────────────────────────────────────


class TestSchedulerMambaIntegration:

    def test_prefill_mamba_seq(self):
        """Scheduler prefill allocates mamba slot via block_manager."""
        sched = Scheduler(mamba_config(num_kvcache_blocks=100))
        seq = mamba_seq([1, 2, 3, 4])
        sched.add(seq)
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 1
        assert seq.mamba_state_slot >= 0
        assert len(batch.mamba_state_slots) == 1

    def test_preempt_releases_mamba_slot(self):
        """Preempted mamba sequence releases its slot."""
        sched = Scheduler(mamba_config(num_kvcache_blocks=100))
        seq = mamba_seq([1, 2, 3, 4])
        sched.add(seq)
        sched.schedule()
        assert seq.mamba_state_slot >= 0
        initial_slots = len(sched.block_manager.free_mamba_slots)
        sched.preempt(seq)
        assert seq.mamba_state_slot == -1
        assert len(sched.block_manager.free_mamba_slots) == initial_slots + 1

    def test_mamba_slot_exhaustion_blocks_prefill(self):
        """When all mamba slots are used, new mamba requests wait."""
        sched = Scheduler(mamba_config(num_kvcache_blocks=200, num_mamba_groups=2))
        s1 = mamba_seq([1, 2, 3, 4])
        s2 = mamba_seq([5, 6, 7, 8])
        s3 = mamba_seq([9, 10, 11, 12])
        sched.extend([s1, s2, s3])
        batch, _ = sched.schedule()
        # Only 2 slots → only 2 prefilled
        assert batch.total_seqs_num_prefill == 2
        assert sched.get_num_unfinished_requests() == 3

    def test_mixed_mamba_and_plain(self):
        """Mamba and plain sequences coexist — plain doesn't consume slots."""
        sched = Scheduler(mamba_config(num_kvcache_blocks=200, num_mamba_groups=2))
        s1 = mamba_seq([1, 2, 3, 4])
        s2 = plain_seq([5, 6, 7, 8])
        s3 = mamba_seq([9, 10, 11, 12])
        s4 = plain_seq([13, 14, 15, 16])
        sched.extend([s1, s2, s3, s4])
        batch, _ = sched.schedule()
        # All 4 should prefill — only 2 mamba slots needed
        assert batch.total_seqs_num_prefill == 4
