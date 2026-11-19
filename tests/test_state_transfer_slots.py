# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D producer's state transfer slots: the top of the state pool, where a
finished request's state waits for its push while its live slot is reused."""

import pytest
from conftest import MockConfig

from atom.kv_transfer.disaggregation.types import ConnectorMetadata
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence


def config(**overrides):
    defaults = {
        "kv_cache_block_size": 4,
        "num_kvcache_blocks": 100,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 256,
        "max_model_len": 256,
        "pool_entries": {"state": 6},
        "state_transfer_slots": 2,
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


def stateful_seq(token_ids):
    return Sequence(token_ids, 4, has_per_req_cache=True)


def admitted(bm, token_ids=(1, 2, 3, 4)):
    seq = stateful_seq(list(token_ids))
    bm.allocate(seq)
    return seq


class TestBlockManagerTransferSlots:
    def test_transfer_slots_are_never_admitted_into(self):
        bm = BlockManager(config())
        assert (bm.num_state_slots, bm.num_state_transfer_slots) == (4, 2)
        seqs = [admitted(bm, range(4 * i, 4 * i + 4)) for i in range(4)]
        assert all(0 <= s.state_slot < 4 for s in seqs)
        assert bm.can_allocate(stateful_seq([20, 21, 22, 23])) < 0

    def test_a_finished_request_moves_and_frees_its_live_slot(self):
        bm = BlockManager(config())
        seq = admitted(bm)
        live = seq.state_slot
        assert bm.move_state_to_transfer_slot(seq) == (live, 4)
        assert seq.state_slots == [4]
        assert bm.state.num_free() == 4
        assert bm.pool_pressure()["transfer_slots_used"] == 1

    def test_deallocate_returns_the_transfer_slot_to_its_own_list(self):
        bm = BlockManager(config())
        seq = admitted(bm)
        bm.move_state_to_transfer_slot(seq)
        bm.deallocate(seq)
        assert bm.state.num_free() == 4
        occupancy = bm.transfer_slot_occupancy()
        assert (
            occupancy["transfer_slots_total"],
            occupancy["transfer_slots_used"],
        ) == (
            2,
            0,
        )
        other = admitted(bm, (5, 6, 7, 8))
        assert bm.move_state_to_transfer_slot(other)[1] == 5

    def test_a_full_transfer_list_leaves_the_request_where_it_is(self):
        bm = BlockManager(config())
        for i in range(2):
            assert bm.move_state_to_transfer_slot(admitted(bm, range(4 * i, 4 * i + 4)))
        seq = admitted(bm, (9, 10, 11, 12))
        live = seq.state_slot
        assert bm.move_state_to_transfer_slot(seq) is None
        assert seq.state_slots == [live]
        assert (bm.transfer_slot_moves, bm.transfer_slots_full) == (2, 1)

    def test_a_slot_still_being_written_or_forked_does_not_move(self):
        bm = BlockManager(config())
        loading = admitted(bm)
        loading.offload_joint.load_hash = 77
        forked = admitted(bm, (5, 6, 7, 8))
        forked.state_fork_src = 3
        for seq in (loading, forked):
            slots = list(seq.state_slots)
            assert bm.move_state_to_transfer_slot(seq) is None
            assert seq.state_slots == slots

    def test_transfer_slots_must_leave_room_for_one_request(self):
        with pytest.raises(ValueError):
            BlockManager(config(pool_entries={"state": 2}, state_transfer_slots=2))

    def test_no_state_pool_means_no_transfer_slots(self):
        bm = BlockManager(config(pool_entries={}))
        assert bm.num_state_transfer_slots == 0
        seq = Sequence([1, 2, 3, 4], 4, has_per_req_cache=False)
        bm.allocate(seq)
        assert bm.move_state_to_transfer_slot(seq) is None


class _Connector:
    """Records what the scheduler hands a staging producer, and when."""

    is_producer = True

    def __init__(self):
        self.staged = []
        self.advertised = []

    def stage_state_for_send(self, seq, src, dst):
        self.staged.append((seq.id, src, dst))

    def request_finished(self, seq):
        self.advertised.append(seq.state_slot)


class TestSchedulerStagesOnFinish:
    def finished(self, reason="stop"):
        sched = Scheduler(config())
        sched.kv_connector = _Connector()
        seq = stateful_seq([1, 2, 3, 4])
        sched.block_manager.allocate(seq)
        live = seq.state_slot
        seq.prefill_completed = True
        sched._mark_finished(seq, reason, seq.num_tokens)
        return sched, seq, live

    def test_the_peer_is_told_the_transfer_slot(self):
        sched, seq, live = self.finished()
        assert sched.kv_connector.staged == [(seq.id, live, 4)]
        assert sched.kv_connector.advertised == [4]
        assert sched.block_manager.state.num_free() == 4
        assert sched.block_manager.transfer_slot_moves == 1

    def test_an_abort_hands_nothing_off_so_nothing_moves(self):
        sched, _seq, live = self.finished("aborted")
        assert sched.kv_connector.staged == []
        assert sched.kv_connector.advertised == [live]


def test_state_stages_are_work_for_the_worker():
    meta = ConnectorMetadata()
    assert not meta.has_work()
    meta.state_stages = {7: (1, 4)}
    assert meta.has_work()
