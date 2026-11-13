# SPDX-License-Identifier: MIT
"""Successor credit corrections across unsent, unseen and reported work."""

import pickle

import pytest
from test_dp_load_balance import _chain_seq, _make_mgr, _note_charged, _route

from atom.kv_transfer.offload.tier import OffloadTierProfile
from atom.model_engine.dp_routing.share_cache import ShareCacheRouter
from atom.model_engine.engine_core_protocol import EngineCoreRequestType


def _pair(stage):
    mgr = _make_mgr(2, strategy="share_cache")
    mgr._share_cache_router = ShareCacheRouter(OffloadTierProfile(True, 0, 256, True), 512)
    sent = []
    mgr._send_request = lambda rank, data: sent.append((rank, pickle.loads(data)))
    producer = _chain_seq(1, 100000)
    follower = _chain_seq(2, 110000)
    producer.data_parallel_rank = follower.data_parallel_rank = 0
    _route(mgr, [producer])
    _note_charged(mgr, producer, 0)
    _route(mgr, [follower])
    _note_charged(mgr, follower, 0)
    assert follower.dp_est_cached_tokens > 0
    mgr._backlog.note_sent(0, [producer], mgr._seq_load)
    mgr._backlog.on_report(0, 1, 0, ())
    if stage != "unsent":
        mgr._backlog.note_sent(0, [follower], mgr._seq_load)
    if stage == "reported":
        mgr._backlog.on_report(0, 2, 10032, ())
    return mgr, follower, sent


@pytest.mark.parametrize("stage", ["unsent", "unseen", "reported"])
def test_successor_debt_remains_visible_after_rank_report(stage):
    mgr, follower, sent = _pair(stage)
    mgr._release_seq(1)
    assert mgr._seq_load[2][2] == 110000
    assert mgr._backlog_effective_tokens_locked()[0] == 110000
    if stage == "unsent":
        assert follower.dp_est_cached_tokens == 0
        assert not sent
        mgr._backlog.note_sent(0, [follower], mgr._seq_load)
        mgr._flush_credit_updates()
    assert sent[-1] == (0, (EngineCoreRequestType.DP_PREFIX_CREDIT, [(2, 0)]))
    mgr._backlog.ack(1, [(2, 0)])
    assert mgr._backlog.debts[2]
    # Apply the acknowledgement and its exact backlog snapshot atomically.
    mgr._backlog.ack(0, [(2, 0)])
    mgr._backlog.on_report(0, 2, 110000, ())
    assert mgr._backlog_effective_tokens_locked()[0] == 110000
    mgr._backlog.ack(0, [(2, 0)])
    assert mgr._backlog.reported_credit_tokens == [0, 0]
    mgr._release_seq(2)
    assert not mgr._backlog.unsent
    assert not mgr._backlog.debts
    assert not mgr._backlog.ordinals


def test_cancelled_unseen_successor_cannot_restore_debt_on_report():
    mgr, _, _ = _pair("unseen")
    mgr._release_seq(1)
    mgr._release_seq(2)
    mgr._backlog.on_report(0, 2, 0, ())
    assert mgr._backlog_effective_tokens_locked() == [0, 0]
    assert mgr._backlog.reported_credit_tokens == [0, 0]


def test_prefill_completion_retires_credit_state_but_keeps_request_charge():
    mgr, _, _ = _pair("reported")
    mgr._release_seq(1)
    mgr._mark_seq_prefill_complete(2)
    assert mgr._seq_load[2] == (0, 1, 0)
    assert not mgr._backlog.prompt_tokens
    assert not mgr._backlog.ordinals
    assert not mgr._backlog.debts
    assert mgr._backlog.reported_credit_tokens == [0, 0]


def test_send_after_concurrent_cancel_does_not_resurrect_credit_identity():
    mgr, follower, _ = _pair("unsent")
    mgr._release_seq(2)
    mgr._backlog.note_sent(0, [follower], mgr._seq_load)
    assert 2 not in mgr._backlog.ordinals
    assert 2 not in mgr._backlog.unsent
    mgr._backlog.on_report(0, 2, 0, ())
    assert mgr._backlog.unseen_tokens == [0, 0]


def test_reset_clears_credit_identity_before_sequence_id_reuse():
    mgr, _, _ = _pair("reported")
    mgr._release_seq(1)
    mgr.reset_dp_router()
    assert not mgr._backlog.unsent
    assert not mgr._backlog.prompt_tokens
    assert not mgr._backlog.credit_updates
    assert not mgr._backlog.credit_sent
    assert not mgr._backlog.debts
    assert not mgr._backlog.ordinals
    assert not any(mgr._backlog.unseen)
    assert mgr._backlog.reported_credit_tokens == [0, 0]


def test_older_ack_clears_only_its_revision_and_unknown_ack_changes_nothing():
    mgr, follower, _ = _pair("reported")
    # Two successive losses before the latest revision has been acknowledged.
    mgr._backlog.add_successor_debt(2, 0, 40000, 50032)
    mgr._flush_credit_updates()
    older_credit = follower.num_prompt_tokens - 50032
    mgr._backlog.add_successor_debt(2, 0, 59968, 110000)
    mgr._flush_credit_updates()
    assert mgr._backlog.reported_credit_tokens[0] == 99968
    mgr._backlog.ack(0, [(2, -1)])
    assert mgr._backlog.reported_credit_tokens[0] == 99968
    mgr._backlog.ack(0, [(2, older_credit)])
    assert mgr._backlog.reported_credit_tokens[0] == 59968
    assert mgr._backlog.credit_updates[2] == (0, 0)
    mgr._backlog.ack(0, [(2, older_credit)])
    assert mgr._backlog.reported_credit_tokens[0] == 59968
    mgr._backlog.ack(0, [(2, 0)])
    assert mgr._backlog.reported_credit_tokens[0] == 0


def test_failed_credit_send_keeps_debt_and_retries_before_ack():
    mgr, _, sent = _pair("reported")
    good_send = mgr._send_request

    def failed_send(*_args):
        raise ConnectionError("input transport unavailable")

    mgr._send_request = failed_send
    mgr._release_seq(1)
    assert mgr._backlog_effective_tokens_locked()[0] == 110000
    assert mgr._backlog.credit_updates[2] == (0, 0)
    assert not mgr._backlog.credit_sent
    mgr._backlog.ack(0, [(2, 0)])
    assert mgr._backlog.debts[2]
    mgr._send_request = good_send
    mgr._flush_credit_updates()
    assert sent[-1] == (0, (EngineCoreRequestType.DP_PREFIX_CREDIT, [(2, 0)]))
    mgr._backlog.ack(0, [(2, 0)])
    assert not mgr._backlog.debts
