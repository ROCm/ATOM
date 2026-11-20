# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D receive of a hybrid model keeps the prefix the consumer already holds.

The producer pushes the per-request state whole, so the consumer's prefix needs
no state checkpoint behind it and must not restore one. The scheduler asks the
load's owner before allocation, once per admission pass.
"""

import types
from collections import deque
from unittest import mock

import test_moriio_connector as moriio_tests
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_RUNTIME,
    ckpt_config,
    make_block_manager,
)

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.multi import multi_connector
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence

PROMPT = list(range(100, 100 + 6 * BLOCK))
LONGER = PROMPT + list(range(900, 900 + 2 * BLOCK))


def receive(token_ids, **params):
    seq = Sequence(list(token_ids), BLOCK, has_per_req_cache=True)
    seq.kv_transfer_params = {"do_remote_prefill": True, **params}
    return seq


def consumer(**checkpointing):
    """A decode consumer that received PROMPT once; no checkpoint unless asked."""
    bm = make_block_manager(
        ckpt_config(**({"state_checkpoint_interval_tokens": 0} | checkpointing)),
        state_runtime=PAGED_COPY_RUNTIME,
    )
    first = receive(PROMPT)
    bm.allocate(first, bm.can_allocate(first))
    bm.register_received_prefix(first)
    bm.deallocate(first)
    return bm


class Leg:
    """A connector leg: its lookup answer and whether it brings the state."""

    def __init__(self, matched=True, whole=True, is_offload=False):
        self.matched, self.whole, self.is_offload = matched, whole, is_offload
        self.may_receive_suffix_and_state = whole

    def get_num_new_matched_tokens(self, seq):
        return (len(seq.prompt_token_ids), True) if self.matched else (0, False)

    def receives_suffix_and_state(self, seq):
        return self.whole

    def cancel_pending_load(self, seq):
        pass


def admit(connector, seq):
    """What the scheduler records from `connector` before `can_allocate`."""
    sched = types.SimpleNamespace(kv_connector=connector)
    Scheduler._query_connector_prefill_match(sched, seq, skip=False)
    return seq.state_from_peer


def composite(*legs):
    """The real multi scheduler over `legs`."""
    with mock.patch.object(
        multi_connector, "_build_subconnectors", return_value=list(legs)
    ):
        return multi_connector.MultiConnectorScheduler(types.SimpleNamespace())


def test_a_receive_keeps_the_prefix_without_a_checkpoint():
    bm = consumer()
    seq = receive(LONGER)
    assert admit(Leg(), seq)
    hit = bm.can_allocate(seq)
    # Every full prompt block of the earlier receive.
    assert hit == len(PROMPT) // BLOCK
    bm.allocate(seq, hit)
    assert seq.num_cached_tokens == len(PROMPT)
    # A fresh state slot that nothing restores: the producer fills it.
    assert seq.state_slot >= 0
    assert not bm.paged_state_checkpoints.restore_queued_for(seq.state_slot)
    assert not seq.replays_hit


def test_a_kept_prefix_demands_no_checkpoint():
    """With checkpointing on, the hit the gate lifts is not a hit the ladder
    lost, so a receive files no checkpoint demand."""
    bm = consumer(state_checkpoint_interval_tokens=BLOCK)
    seq = receive(LONGER)
    assert admit(Leg(), seq)
    assert bm.can_allocate(seq) == len(PROMPT) // BLOCK
    assert seq.checkpoint_demand_pos == 0
    assert bm.demands_recorded == 0


def test_a_receive_of_every_block_keeps_the_gate():
    bm = consumer()
    seq = receive(LONGER)
    assert not admit(Leg(whole=False), seq)
    assert bm.can_allocate(seq) == 0


def test_a_local_prefill_still_needs_a_checkpoint():
    bm = consumer()
    seq = receive(LONGER)
    assert not admit(Leg(matched=False), seq)
    assert bm.can_allocate(seq) == 0


def test_a_held_back_receive_that_prefills_locally_keeps_the_gate():
    """Asked again after a hold-back, a connector may answer "no remote load"
    while the params still say remote prefill; that pass prefills here."""
    bm = consumer()
    seq = receive(LONGER)
    leg = Leg()
    admit(leg, seq)
    assert bm.can_allocate(seq) == len(PROMPT) // BLOCK
    leg.matched = False
    admit(leg, seq)
    assert bm.can_allocate(seq) == 0


def test_an_offload_win_in_a_composite_keeps_the_gate():
    """The leg that won the lookup answers; an offload tier never brings the
    state, whatever the P/D leg beside it could do."""
    bm = consumer()
    seq = receive(LONGER)
    offload = Leg(whole=False, is_offload=True)
    assert composite(Leg(), offload).may_receive_suffix_and_state
    assert not composite(offload).may_receive_suffix_and_state
    assert not admit(composite(Leg(matched=False), offload), seq)
    assert bm.can_allocate(seq) == 0
    assert admit(composite(Leg(), offload), seq)
    assert bm.can_allocate(seq) == len(PROMPT) // BLOCK
    # No winner, no answer; and a connector that says nothing brings none.
    assert not composite(Leg()).receives_suffix_and_state(seq)
    assert not KVConnectorSchedulerBase.may_receive_suffix_and_state
    assert not KVConnectorSchedulerBase.receives_suffix_and_state(None, seq)


def moriio_consumer():
    """The real MoRIIO scheduler, as a write-mode decode of BLOCK-token pages."""
    return moriio_tests.SchedulerSlotTest.built(dcp=1, mode="write")


def test_a_producer_without_page_geometry_claims_no_shared_prefix():
    """Its receive moves every block, so the consumer must not hand it blocks
    the index holds: the gate decides, and with no checkpoint it keeps none."""
    bm = consumer()
    seq = receive(LONGER)
    assert not admit(moriio_consumer(), seq)
    hit = bm.can_allocate(seq)
    assert hit == 0
    bm.allocate(seq, hit)
    assert seq.num_cached_tokens == 0


def test_a_read_receives_behind_the_gate():
    """A read moves every block, so the scheduler never asks it, however
    alike the producer pages."""
    bm = consumer()
    seq = receive(LONGER, block_size=BLOCK, dcp_size=1)
    reader = moriio_tests.SchedulerSlotTest.built(dcp=1, mode="read")
    assert not admit(reader, seq)
    assert bm.can_allocate(seq) == 0


def test_a_skipped_pass_asks_nothing():
    """A pass that skips the lookup (a landed receive's first decode, an
    offload resume, a detached lookup) neither asks nor moves the answer."""
    seq = receive(LONGER)
    leg = Leg()
    assert admit(leg, seq)
    asked = []
    leg.get_num_new_matched_tokens = lambda s: asked.append(s) or (0, False)
    sched = types.SimpleNamespace(kv_connector=leg)
    assert not Scheduler._query_connector_prefill_match(sched, seq, skip=True)
    assert asked == [] and seq.state_from_peer


def test_a_preempted_receive_recomputes_behind_the_gate():
    """Preemption is the only way a received request reaches `can_allocate`
    again, and its recompute resumes nothing a peer pushed."""
    bm = consumer()
    seq = receive(LONGER)
    assert admit(Leg(), seq)
    bm.allocate(seq, bm.can_allocate(seq))
    sched = types.SimpleNamespace(
        _is_preemptable=lambda s: True,
        total_preemptions=0,
        _partial_prefill_count=0,
        _connector_release_stalled_save=lambda s: None,
        block_manager=bm,
        _behind_settled_resumes=lambda: 0,
        waiting=deque(),
        pd_admission=None,
    )
    assert Scheduler.preempt(sched, seq)
    assert not seq.state_from_peer
    assert bm.can_allocate(seq) == 0


def test_the_producers_hit_is_reported_even_when_it_is_zero():
    """`cached_tokens` is the producer's hit: the decode's local prefix only
    saved a transfer. A reported 0 is an answer, not a missing one."""
    sched = types.SimpleNamespace(
        _lookup_detach_declined=set(),
        _notify_connector_after_prefill_alloc=lambda seq: None,
    )
    for params, reported in (
        ({"prefix_cache_hit_tokens": 0}, 0),
        # A request no producer answered for records the local hit.
        ({}, len(PROMPT)),
    ):
        bm = consumer()
        seq = receive(LONGER, **params)
        admit(Leg(), seq)
        bm.allocate(seq, bm.can_allocate(seq))
        assert seq.num_cached_tokens == len(PROMPT)
        Scheduler._finish_prefill_admission(sched, seq)
        assert seq.prefix_cache_hit_tokens == reported
