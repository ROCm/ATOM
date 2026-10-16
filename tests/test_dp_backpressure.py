"""Fast CPU-only test of Fix A backpressure dispatch (ATOM_DP_DISPATCH_DEPTH).

Builds a bare CoreManager via __new__ (same pattern as tests/test_dp_load_balance.py),
stubs _send_request, and checks: (1) per-rank cap is honored, (2) the rest waits in
the pending pool, (3) completing a seq re-feeds the drained rank, (4) a fast rank
ends up processing MORE total requests than a slow one (the whole point of Fix A).
"""

import logging
import pickle
import queue
import threading
import time
from collections import deque
from threading import Lock
from types import SimpleNamespace

import pytest

from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_core_protocol import EngineCoreRequestType


class Seq:
    _n = 0

    def __init__(self, prompt=10):
        Seq._n += 1
        self.id = f"s{Seq._n}"
        self.num_prompt_tokens = prompt
        self.data_parallel_rank = None
        self.dp_session_id = None
        self.dp_parent_session_id = None
        self.stream_callback = None


def make_mgr(n_ranks=4, depth=3, fail_ranks=None):
    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "test"
    mgr._dp_lb_strategy = "least_requests"
    mgr._dp_lb_req_equiv = 512
    mgr._dp_session_affinity_enabled = False
    mgr.global_engine_count = n_ranks
    mgr.local_engine_count = n_ranks
    mgr.pp_size = 1
    mgr._rank_rotation_cursor = 0
    mgr._rank_reqs = [0] * n_ranks
    mgr._rank_tokens = [0] * n_ranks
    mgr._rank_routed_total = [0] * n_ranks
    mgr._dp_route_counters = {"explicit_total": 0, "load_balanced_total": 0}
    mgr._seq_load = {}
    mgr._lb_lock = Lock()
    mgr._input_send_lock = Lock()
    mgr._bp_depth = depth
    mgr._bp_pending = deque()
    mgr._bp_offline_batch = True
    mgr._bp_dispatch_paused = False
    mgr._seq_id_to_callback = {}
    mgr._flush_stream_batch_fn = None
    mgr.fail_ranks = set(fail_ranks or ())
    # capture what got sent to each rank
    mgr.sent = [[] for _ in range(n_ranks)]
    mgr._send_lock = Lock()

    def fake_send(dp_rank, payload):
        if dp_rank in mgr.fail_ranks:
            raise RuntimeError(f"send failed rank {dp_rank}")
        _type, seqs = pickle.loads(payload)
        assert _type == EngineCoreRequestType.ADD
        with mgr._send_lock:
            mgr.sent[dp_rank].extend(s.id for s in seqs)

    mgr._send_request = fake_send
    return mgr


def _accounted(mgr):
    with mgr._send_lock:
        sent = [sid for rank_ids in mgr.sent for sid in rank_ids]
    pending = [seq.id for seq in mgr._bp_pending]
    return sent, pending


def test_cap_and_pending():
    mgr = make_mgr(n_ranks=4, depth=3)
    seqs = [Seq() for _ in range(100)]
    mgr.add_request(seqs)
    # 4 ranks * depth 3 = 12 dispatched, rest pending.
    assert mgr._rank_reqs == [3, 3, 3, 3], mgr._rank_reqs
    assert sum(len(s) for s in mgr.sent) == 12
    assert len(mgr._bp_pending) == 88


def test_refill_feeds_fast_rank_more():
    mgr = make_mgr(n_ranks=4, depth=3)
    seqs = [Seq() for _ in range(40)]
    mgr.add_request(seqs)
    assert mgr._rank_reqs == [3, 3, 3, 3]

    # Simulate rank 0 completing fast: finish 10 of its seqs one at a time,
    # each completion triggering a refill (as the output thread does).
    def finish_one_on_rank(rank):
        # pick a charged seq currently on `rank`
        sid = next(s for s, (r, _, _) in mgr._seq_load.items() if r == rank)
        mgr._release_seq_load(sid)
        if mgr._bp_depth > 0 and mgr._bp_pending:
            mgr._bp_dispatch_or_refill(raise_on_error=False)
        return sid

    for _ in range(10):
        finish_one_on_rank(0)

    # Rank 0 should have been re-fed each time -> processed more than peers.
    assert len(mgr.sent[0]) > len(mgr.sent[1]), (len(mgr.sent[0]), mgr.sent[1:])
    # Every rank still capped at depth in-flight (never exceeded).
    assert all(r <= 3 for r in mgr._rank_reqs), mgr._rank_reqs
    # Pending shrank by the 10 refills.
    assert len(mgr._bp_pending) == 40 - 12 - 10, len(mgr._bp_pending)


def _init_with_strategy(strategy):
    mgr = CoreManager.__new__(CoreManager)
    mgr._init_shared_state(
        SimpleNamespace(dp_load_balance=strategy, max_num_seqs=1536),
        label="test",
        local_engine_count=2,
    )
    return mgr


def test_init_warns_when_backpressure_overrides_strategy(monkeypatch, caplog):
    def check(strategy, depth, overridden):
        monkeypatch.setenv("ATOM_DP_DISPATCH_DEPTH", str(depth))
        caplog.clear()
        mgr = None
        try:
            with caplog.at_level(logging.WARNING, logger="atom"):
                mgr = _init_with_strategy(strategy)
            assert mgr._bp_depth == depth
            hit = any(
                f"ignores dp_load_balance={strategy}" in r.getMessage()
                for r in caplog.records
            )
            assert hit is overridden
        finally:
            if mgr is not None:
                mgr.ctx.term()

    check("least_tokens", 8, True)
    check("round_robin", 8, True)
    check("least_requests", 8, False)
    check("least_tokens", 0, False)


def test_drains_everything():
    mgr = make_mgr(n_ranks=2, depth=2)
    seqs = [Seq() for _ in range(20)]
    mgr.add_request(seqs)
    # finish all in-flight repeatedly until pending + in-flight empty
    guard = 0
    while mgr._seq_load:
        sid = next(iter(mgr._seq_load))
        mgr._release_seq_load(sid)
        if mgr._bp_pending:
            mgr._bp_dispatch_or_refill(raise_on_error=False)
        guard += 1
        assert guard < 1000
    assert len(mgr._bp_pending) == 0
    dispatched = [sid for rank_ids in mgr.sent for sid in rank_ids]
    assert len(dispatched) == 20  # every seq dispatched exactly once
    assert len(set(dispatched)) == 20  # no duplicates


def test_online_add_does_not_enter_pending_pool():
    mgr = make_mgr(n_ranks=2, depth=1)
    mgr._bp_offline_batch = False
    seqs = [Seq() for _ in range(6)]
    mgr.add_request(seqs)
    assert len(mgr._bp_pending) == 0
    sent, pending = _accounted(mgr)
    assert pending == []
    assert sorted(sent) == sorted(seq.id for seq in seqs)


def test_partial_send_failure_requeues_later_ranks_and_pauses():
    # Rank 1 fails in the middle of the send loop. Rank 2 must still be
    # attempted. The failed seq goes back to the head of the pool, its charge
    # returns to zero, and a later refill does not dispatch the failed add.
    mgr = make_mgr(n_ranks=3, depth=1, fail_ranks={1})
    seqs = [Seq() for _ in range(6)]
    with pytest.raises(RuntimeError, match="send failed rank 1"):
        mgr.add_request(seqs)
    assert mgr._rank_reqs == [1, 0, 1]
    assert next(seq.id for seq in mgr._bp_pending) == seqs[1].id
    sent, pending = _accounted(mgr)
    assert sent == [seqs[0].id, seqs[2].id]
    assert sorted(sent + pending) == sorted(seq.id for seq in seqs)
    assert len(sent + pending) == len(set(sent + pending))
    assert mgr._bp_dispatch_paused
    mgr._bp_dispatch_or_refill(raise_on_error=False)
    sent_after, _pending_after = _accounted(mgr)
    assert sent_after == sent


def test_every_send_failing_returns_counters_to_zero():
    mgr = make_mgr(n_ranks=2, depth=1, fail_ranks={0, 1})
    seqs = [Seq() for _ in range(4)]
    with pytest.raises(RuntimeError):
        mgr.add_request(seqs)
    assert mgr._rank_reqs == [0, 0]
    assert not mgr._seq_load
    sent, pending = _accounted(mgr)
    assert sent == []
    assert sorted(pending) == sorted(seq.id for seq in seqs)
    assert len(pending) == len(set(pending))


def test_refill_send_failure_requeues_and_fails_a_stalled_run():
    mgr = make_mgr(n_ranks=2, depth=1)
    mgr.outputs_queue = queue.Queue()
    seqs = [Seq() for _ in range(4)]
    mgr.add_request(seqs)
    assert sum(mgr._rank_reqs) == 2
    mgr.fail_ranks.update({0, 1})
    for sid in list(mgr._seq_load):
        mgr._release_seq_load(sid)
    mgr._bp_dispatch_or_refill(raise_on_error=False)
    assert mgr._rank_reqs == [0, 0]
    sent, pending = _accounted(mgr)
    # The two originally sent seqs stay sent; the two that failed the refill
    # are back in the pool, once each.
    assert len(sent) == 2
    assert len(pending) == 2
    assert len(set(sent + pending)) == 4
    assert mgr._bp_dispatch_paused
    err = mgr.outputs_queue.get_nowait()
    assert isinstance(err, RuntimeError)


def test_abort_pending_seq_is_not_dispatched():
    mgr = make_mgr(n_ranks=2, depth=1)
    seqs = [Seq() for _ in range(4)]
    delivered = {}

    def callback(output):
        delivered[output.request_id] = output.finish_reason

    seqs[-1].stream_callback = callback
    mgr.add_request(seqs)
    target = seqs[-1]
    assert any(seq.id == target.id for seq in mgr._bp_pending)
    mgr.abort_request(target.id)
    assert all(seq.id != target.id for seq in mgr._bp_pending)
    assert delivered[target.id] == "aborted"
    guard = 0
    while mgr._seq_load or mgr._bp_pending:
        if mgr._seq_load:
            sid = next(iter(mgr._seq_load))
            mgr._release_seq_load(sid)
        if mgr._bp_pending:
            mgr._bp_dispatch_or_refill(raise_on_error=False)
        guard += 1
        assert guard < 1000
    sent, pending = _accounted(mgr)
    assert target.id not in sent
    assert pending == []
    assert len(sent) == 3


def test_concurrent_add_and_refill_dispatches_once():
    mgr = make_mgr(n_ranks=4, depth=2)
    seqs = [Seq() for _ in range(40)]
    errors = []

    def add_chunks():
        for i in range(0, len(seqs), 4):
            mgr.add_request(seqs[i : i + 4])

    def release_loop():
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                sid = None
                with mgr._lb_lock:
                    if mgr._seq_load:
                        sid = next(iter(mgr._seq_load))
                if sid is None:
                    if not adder.is_alive() and not mgr._bp_pending:
                        return
                    time.sleep(0.001)
                    continue
                mgr._release_seq_load(sid)
                mgr._bp_dispatch_or_refill(raise_on_error=False)
            raise AssertionError("timed out draining concurrent backpressure")
        except Exception as exc:  # noqa: BLE001 - surfaced to the test thread
            errors.append(exc)

    adder = threading.Thread(target=add_chunks)
    releaser = threading.Thread(target=release_loop)
    adder.start()
    releaser.start()
    adder.join()
    releaser.join()
    assert not errors, errors
    assert mgr._rank_reqs == [0, 0, 0, 0]
    assert not mgr._seq_load
    assert not mgr._bp_pending
    sent, pending = _accounted(mgr)
    assert pending == []
    assert sorted(sent) == sorted(seq.id for seq in seqs)
    assert len(sent) == len(set(sent))
