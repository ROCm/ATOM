"""Fast CPU-only test of Fix A backpressure dispatch (ATOM_DP_DISPATCH_DEPTH).

Builds a bare CoreManager via __new__ (same pattern as tests/test_dp_load_balance.py),
stubs _send_request, and checks: (1) per-rank cap is honored, (2) the rest waits in
the pending pool, (3) completing a seq re-feeds the drained rank, (4) a fast rank
ends up processing MORE total requests than a slow one (the whole point of Fix A).
"""
import pickle
from collections import deque
from threading import Lock

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


def make_mgr(n_ranks=4, depth=3):
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
    mgr._seq_id_to_callback = {}
    # capture what got sent to each rank
    mgr.sent = [[] for _ in range(n_ranks)]

    def fake_send(dp_rank, payload):
        _type, seqs = pickle.loads(payload)
        assert _type == EngineCoreRequestType.ADD
        mgr.sent[dp_rank].extend(s.id for s in seqs)

    mgr._send_request = fake_send
    return mgr


def test_cap_and_pending():
    mgr = make_mgr(n_ranks=4, depth=3)
    seqs = [Seq() for _ in range(100)]
    mgr.add_request(seqs)
    # 4 ranks * depth 3 = 12 dispatched, rest pending.
    assert mgr._rank_reqs == [3, 3, 3, 3], mgr._rank_reqs
    assert sum(len(s) for s in mgr.sent) == 12
    assert len(mgr._bp_pending) == 88
    print("OK cap+pending: dispatched=12 pending=88 reqs=", mgr._rank_reqs)


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
    print("OK refill: rank0 processed", len(mgr.sent[0]),
          "vs peers", [len(mgr.sent[i]) for i in (1, 2, 3)],
          "| in-flight", mgr._rank_reqs, "| pending", len(mgr._bp_pending))


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
    assert sum(len(s) for s in mgr.sent) == 20  # every seq dispatched exactly once
    assert len(set(sum(mgr.sent, []))) == 20    # no duplicates
    print("OK drain: all 20 dispatched exactly once, nothing stuck")


if __name__ == "__main__":
    test_cap_and_pending()
    test_refill_feeds_fast_rank_more()
    test_drains_everything()
    print("\nALL BACKPRESSURE TESTS PASSED")
