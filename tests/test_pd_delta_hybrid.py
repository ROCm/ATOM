# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D receive of a hybrid model keeps the prefix the consumer already holds.

The producer pushes the per-request state whole, so the consumer's prefix needs
no state checkpoint behind it and must not restore one. The scheduler asks the
load's owner before allocation, once per admission pass.
"""

import ast
import logging
import types
from pathlib import Path

import pytest
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_RUNTIME,
    ckpt_config,
    make_block_manager,
)

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import ReqMeta
from atom.model_engine.prefix_hash import hash_block_size_for
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence

ROOT = Path(__file__).resolve().parents[1]
MORIIO = ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py"
PROMPT = list(range(100, 100 + 6 * BLOCK))
LONGER = PROMPT + list(range(900, 900 + 2 * BLOCK))
GEOMETRY = {"block_size": BLOCK, "dcp_size": 1}


def receive(token_ids, **params):
    seq = Sequence(list(token_ids), BLOCK, has_per_req_cache=True)
    seq.kv_transfer_params = {"do_remote_prefill": True, **params}
    return seq


def consumer():
    """A decode consumer that received PROMPT once, with no checkpoint."""
    bm = make_block_manager(
        ckpt_config(state_checkpoint_interval_tokens=0),
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
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = list(legs)
    multi._load_winner = {}
    return multi


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
    assert not admit(composite(Leg(matched=False), offload), seq)
    assert bm.can_allocate(seq) == 0
    assert admit(composite(Leg(), offload), seq)
    assert bm.can_allocate(seq) == len(PROMPT) // BLOCK
    # No winner, no answer; and a connector that says nothing brings none.
    assert not composite(Leg()).receives_suffix_and_state(seq)
    assert not KVConnectorSchedulerBase.receives_suffix_and_state(None, seq)


def lift(cls, names, **scope):
    """Methods of `cls` in the MoRIIO module, undecorated, for the CPU lane."""
    for node in ast.parse(MORIIO.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            items = [i for i in node.body if getattr(i, "name", None) in names]
            for item in items:
                item.decorator_list = []
            code = compile(ast.Module(items, []), str(MORIIO), "exec")
            exec(code, scope)  # noqa: S102
    return {name: scope[name] for name in names}


def moriio_scheduler(producer=False, dcp_size=1):
    """`MoRIIOConnectorScheduler`'s receive side, lifted for the CPU lane."""
    methods = lift(
        "MoRIIOConnectorScheduler",
        {
            "get_num_new_matched_tokens",
            "receives_suffix_and_state",
            "_same_page_geometry",
            "update_state_after_alloc",
        },
        Sequence=Sequence,
        logger=logging.getLogger("atom"),
    )
    sched = type("MoRIIOScheduler", (), methods)()
    sched.is_producer, sched.block_size, sched.dcp_size = producer, BLOCK, dcp_size
    sched.hash_block_size = hash_block_size_for(BLOCK, dcp_size)
    sched.transfer_id_to_request_id, sched.request_id_to_transfer_id = {}, {}
    sched._reqs_need_recv = {}
    return sched


def test_moriio_brings_the_state_only_from_a_producer_paging_alike():
    sched = moriio_scheduler()
    assert sched.receives_suffix_and_state(receive(LONGER, **GEOMETRY))
    # A producer that does not say, or pages otherwise, moves every block.
    for params in (
        {},
        {"block_size": 2 * BLOCK, "dcp_size": 1},
        {**GEOMETRY, "dcp_size": 2},
    ):
        assert not sched.receives_suffix_and_state(receive(LONGER, **params))
    plain = receive(LONGER, **GEOMETRY)
    plain.kv_transfer_params["do_remote_prefill"] = False
    assert not sched.receives_suffix_and_state(plain)
    producer = moriio_scheduler(producer=True)
    assert not producer.receives_suffix_and_state(receive(LONGER, **GEOMETRY))


def test_a_producer_without_page_geometry_claims_no_shared_prefix():
    """Its receive moves every block, so the consumer must not hand it blocks
    the index holds: the gate decides, and with no checkpoint it keeps none."""
    bm = consumer()
    seq = receive(LONGER)
    assert not admit(moriio_scheduler(), seq)
    hit = bm.can_allocate(seq)
    assert hit == 0
    bm.allocate(seq, hit)
    assert seq.num_cached_tokens == 0


@pytest.mark.parametrize("dcp", [1, 2])
def test_moriio_skips_only_a_prefix_numbered_alike(dcp, caplog):
    """`num_computed_blocks` counts block-table entries of `hash_block_size`
    tokens; a producer that pages otherwise, or does not say, gets every block."""
    sched = moriio_scheduler(dcp_size=dcp)
    held = 3 * sched.hash_block_size
    for params, kept in (
        ({"block_size": BLOCK, "dcp_size": dcp}, 3),
        ({}, 0),
        ({"block_size": 2 * BLOCK, "dcp_size": dcp}, 0),
        ({"block_size": BLOCK, "dcp_size": 3 - dcp}, 0),
    ):
        seq = receive(LONGER, **params)
        seq.num_cached_tokens = held
        caplog.clear()
        with caplog.at_level("WARNING", logger="atom"):
            sched.update_state_after_alloc(seq)
        assert seq.kv_transfer_params["num_computed_blocks"] == kept
        assert ("receiving every block" in caplog.text) == (kept == 0)


def test_both_ends_of_the_pairing_skip_the_kept_prefix():
    split = lift("MoRIIOConnector", {"_split_at_kept_prefix"}, ReqMeta=ReqMeta)[
        "_split_at_kept_prefix"
    ]
    meta = types.SimpleNamespace(
        num_computed_blocks=2,
        local_block_ids=[10, 11, 12, 13],
        remote_block_ids=[20, 21, 22, 23],
    )
    assert split(meta) == ([10, 11], [12, 13], [22, 23])
    meta.num_computed_blocks = 0
    assert split(meta) == ([], [10, 11, 12, 13], [20, 21, 22, 23])


def test_a_skipped_pass_drops_an_earlier_yes():
    """The pass that hands a finished receive to its first decode skips the
    lookup; the yes from its admission must not reach a later recompute."""
    bm = consumer()
    seq = receive(LONGER)
    leg = Leg()
    assert admit(leg, seq)
    sched = types.SimpleNamespace(kv_connector=leg)
    Scheduler._query_connector_prefill_match(sched, seq, skip=True)
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
        seq = receive(LONGER, **GEOMETRY, **params)
        admit(Leg(), seq)
        bm.allocate(seq, bm.can_allocate(seq))
        assert seq.num_cached_tokens == len(PROMPT)
        Scheduler._finish_prefill_admission(sched, seq)
        assert seq.prefix_cache_hit_tokens == reported
