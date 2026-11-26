# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D producer's state transfer slots: the top of the state pool, where a
finished request's state waits for its push while its live slot is reused."""

import logging

from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.engine_stats import EngineStats
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
        assert bm.transfer_slot_occupancy()["transfer_slots_used"] == 1

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

    def test_a_draft_request_gives_back_every_live_slot(self):
        """With DSpark a request also holds its rollback slots; all go back."""
        bm = BlockManager(
            config(pool_entries={"state": 8}, pool_entries_per_req={"state": 3})
        )
        seq = admitted(bm)
        assert len(seq.state_slots) == 3
        live = seq.state_slot
        assert bm.move_state_to_transfer_slot(seq) == (live, 6)
        assert seq.state_slots == [6]
        assert bm.state.num_free() == 6

    def test_a_full_transfer_list_leaves_the_request_where_it_is(self):
        bm = BlockManager(config())
        for i in range(2):
            assert bm.move_state_to_transfer_slot(admitted(bm, range(4 * i, 4 * i + 4)))
        seq = admitted(bm, (9, 10, 11, 12))
        live = seq.state_slot
        assert bm.move_state_to_transfer_slot(seq) is None
        assert seq.state_slots == [live]
        assert (bm.transfer_slot_moves, bm.transfer_slots_full) == (2, 1)

    def test_only_sized_slots_rebind_the_pools_release(self):
        """`deallocate` calls the pool's own `release_many`, as on main, unless
        there are transfer slots to route back to their list."""
        bm = BlockManager(config(state_transfer_slots=0))
        assert (bm.num_state_slots, bm.num_state_transfer_slots) == (6, 0)
        assert "release_many" not in vars(bm.state)
        sized = BlockManager(config())
        assert sized.state.release_many == sized._release_live_or_transfer_slots


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

    def test_only_sized_slots_get_a_stats_line(self, caplog):
        """Unsized, the cache stats print main's lines through main's method."""
        stats = Scheduler(config(state_transfer_slots=0)).engine_stats
        assert stats._pressure_logger is EngineStats._log_pressure
        with caplog.at_level(logging.INFO, logger="atom"):
            stats._log_cache()
        assert "[Pool Pressure]" in caplog.text
        assert "[Transfer Slots]" not in caplog.text
        caplog.clear()
        sched, _seq, _live = self.finished()
        with caplog.at_level(logging.INFO, logger="atom"):
            sched.engine_stats._log_cache()
        assert "[Pool Pressure]" in caplog.text
        assert "[Transfer Slots] 1/2 used, 1 moved, 0 found full" in caplog.text
