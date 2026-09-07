# SPDX-License-Identifier: MIT
# Unit tests for CoreManager DP request load-balancing (engine_core_mgr).
#
# These exercise the pure routing/bookkeeping helpers in isolation via
# ``CoreManager.__new__`` — no engine cores, sockets, or GPU are created.
# CoreManager gets ``EngineCoreRequestType`` from the lightweight
# ``engine_core_protocol`` module and only lazy-imports the heavy ``EngineCore``
# (which pulls aiter) inside ``launch_engine_core``, so importing CoreManager
# here works on the CPU-only / mocked CI runner without any sys.modules stub;
# conftest.py supplies the atom.* / zmq stubs the import chain needs.

import pickle
from threading import Lock

import pytest

from atom.model_engine.engine_core_mgr import (
    DP_LB_STRATEGIES,
    CoreManager,
)

# ── Helpers ────────────────────────────────────────────────────────────────


class _FakeSeq:
    """Minimal stand-in for Sequence: routing only reads id/num_prompt_tokens."""

    def __init__(
        self,
        seq_id,
        num_prompt_tokens=1,
        data_parallel_rank=None,
        dp_session_id=None,
        dp_parent_session_id=None,
    ):
        self.id = seq_id
        self.num_prompt_tokens = num_prompt_tokens
        self.dp_session_id = dp_session_id
        self.dp_parent_session_id = dp_parent_session_id
        if data_parallel_rank is not None:
            self.data_parallel_rank = data_parallel_rank


def _make_mgr(
    n_ranks,
    strategy="least_tokens",
    req_equiv=512,
    session_affinity=False,
):
    """Build a bare CoreManager with just the routing state initialized."""
    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "Engine Core Mgr"
    mgr.local_engine_count = n_ranks
    mgr._dp_lb_strategy = strategy
    mgr._dp_lb_req_equiv = req_equiv
    mgr._dp_session_affinity_enabled = session_affinity
    mgr._dp_session_owners = {}
    mgr._dp_session_prompt_tokens = {}
    mgr._dp_route_counters = {
        "affinity_new_total": 0,
        "affinity_owner_hit_total": 0,
        "affinity_spill_total": 0,
        "affinity_parent_ignored_total": 0,
        "explicit_total": 0,
        "load_balanced_total": 0,
    }
    mgr._rank_routed_total = [0] * n_ranks
    mgr._rank_rotation_cursor = 0
    mgr._rank_reqs = [0] * n_ranks
    mgr._rank_tokens = [0] * n_ranks
    mgr._seq_load = {}
    mgr._lb_lock = Lock()
    return mgr


def _route(mgr, seqs):
    """Replicate add_request's per-seq selection loop (hint > strategy)."""
    assigned = []
    with mgr._lb_lock:
        for seq in seqs:
            hint = getattr(seq, "data_parallel_rank", None)
            hint = int(hint) if hint is not None else None
            rank = mgr._select_dp_rank_for_seq_locked(seq, hint)
            mgr._charge_seq_load_locked(seq, rank)
            assigned.append(rank)
    return assigned


# ── Tests ──────────────────────────────────────────────────────────────────


def test_least_requests_spreads_uniformly():
    mgr = _make_mgr(4, strategy="least_requests")
    seqs = [_FakeSeq(i, num_prompt_tokens=10) for i in range(8)]
    _route(mgr, seqs)
    # 8 uniform requests across 4 ranks -> 2 each.
    assert mgr._rank_reqs == [2, 2, 2, 2]


def test_least_requests_tiebreak_prefers_fewer_prompt_tokens():
    # Equal in-flight request count -> the prompt-token load breaks the tie.
    mgr = _make_mgr(2, strategy="least_requests")
    mgr._rank_reqs = [1, 1]
    mgr._rank_tokens = [500, 100]
    with mgr._lb_lock:
        rank = mgr._select_dp_rank_locked()
    assert rank == 1  # equal reqs -> pick the lighter-token rank


def test_least_requests_count_dominates_token_tiebreak():
    # Request count is the primary key: a rank with fewer requests wins even
    # when it carries far more prompt tokens.
    mgr = _make_mgr(2, strategy="least_requests")
    mgr._rank_reqs = [1, 3]
    mgr._rank_tokens = [10_000, 100]
    with mgr._lb_lock:
        rank = mgr._select_dp_rank_locked()
    assert rank == 0  # fewer requests wins despite the larger token load


def test_least_requests_tiebreak_then_count_primacy_over_route():
    # Seed equal counts but skewed tokens; routing should first even the token
    # load (tie-break), then request-count primacy takes over.
    mgr = _make_mgr(2, strategy="least_requests")
    mgr._rank_reqs = [1, 1]
    mgr._rank_tokens = [1000, 100]
    ranks = _route(mgr, [_FakeSeq("a", 100), _FakeSeq("b", 100)])
    # a: reqs tie (1,1) -> tokens 1000 vs 100 -> rank 1; now reqs=[1,2].
    # b: reqs 1 vs 2 -> rank 0 has fewer requests -> rank 0 (count primary).
    assert ranks == [1, 0]


def test_least_tokens_avoids_heavy_rank():
    # Pure token balance (req_equiv=0): a long prompt should keep that rank out
    # of rotation until the others accumulate comparable token load.
    mgr = _make_mgr(2, strategy="least_tokens", req_equiv=0)
    first = _route(mgr, [_FakeSeq("big", num_prompt_tokens=1000)])[0]
    other = 1 - first
    # Next several small requests must all go to the lighter rank.
    ranks = _route(mgr, [_FakeSeq(f"s{i}", num_prompt_tokens=100) for i in range(5)])
    assert all(r == other for r in ranks)
    assert mgr._rank_tokens[other] == 500
    assert mgr._rank_tokens[first] == 1000


def test_least_tokens_combined_signal_counts_requests():
    # With req_equiv>0 a rank holding many tiny requests still looks loaded, so
    # routing balances request count even when token counts are equal-ish.
    mgr = _make_mgr(2, strategy="least_tokens", req_equiv=512)
    ranks = _route(mgr, [_FakeSeq(i, num_prompt_tokens=1) for i in range(6)])
    # 6 near-zero-token requests -> alternate evenly by the req_equiv term.
    assert mgr._rank_reqs == [3, 3]
    assert sorted(ranks) == [0, 0, 0, 1, 1, 1]


def test_affinity_new_session_uses_token_load_before_stable_tiebreak():
    mgr = _make_mgr(3, session_affinity=True)
    expected = mgr._stable_session_rank("session-root")
    mgr._rank_tokens[expected] = 500_000
    rank = _route(
        mgr,
        [_FakeSeq("root", 1000, dp_session_id="session-root")],
    )[0]
    assert rank != expected
    assert mgr._rank_tokens[rank] == 1000
    assert mgr._dp_session_owners["session-root"] == rank


def test_affinity_hash_is_deterministic_across_managers():
    first = _make_mgr(8, session_affinity=True)
    second = _make_mgr(8, session_affinity=True)
    assert first._stable_session_rank("stable-correlation-id") == (
        second._stable_session_rank("stable-correlation-id")
    )
    assert first._select_new_session_rank_locked("stable-correlation-id") == (
        second._select_new_session_rank_locked("stable-correlation-id")
    )


def test_affinity_existing_session_always_uses_cache_owner():
    mgr = _make_mgr(2, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    # Even extreme transient backlog cannot move an existing session. Replaying
    # a long agent prefix is more expensive than waiting for its cache owner.
    mgr._rank_tokens = [10_000_000, 0]
    rank = _route(mgr, [_FakeSeq("turn", 1000, dp_session_id="s")])[0]
    assert rank == 0
    assert mgr._dp_route_counters["affinity_spill_total"] == 0


def test_affinity_later_turn_charges_only_prompt_growth():
    mgr = _make_mgr(2, session_affinity=True, req_equiv=512)
    first = _FakeSeq("first", 100_000, dp_session_id="s")
    second = _FakeSeq("second", 101_500, dp_session_id="s")

    owner = _route(mgr, [first])[0]
    mgr._mark_seq_prefill_complete(first.id)
    assert mgr._rank_tokens[owner] == 0

    assert _route(mgr, [second]) == [owner]
    assert mgr._rank_tokens[owner] == 1_500
    assert mgr._seq_load[second.id][2] == 1_500


def test_affinity_new_session_accounts_for_decode_pressure():
    mgr = _make_mgr(2, session_affinity=True, req_equiv=512)
    mgr._rank_reqs = [4, 0]
    mgr._rank_tokens = [0, 1_000]

    rank = _route(mgr, [_FakeSeq("new", 100, dp_session_id="new-session")])[0]
    # score(rank0)=2048, score(rank1)=1000 before the new request is charged.
    assert rank == 1


def test_affinity_routes_across_global_ranks_on_multinode_coordinator():
    mgr = _make_mgr(2, session_affinity=True)
    # A coordinator owns two local engines but routes across all four global
    # ranks. Routing state must use the global width so a remote rank can win.
    mgr.global_engine_count = 4
    mgr._rank_reqs = [0, 0, 0, 0]
    mgr._rank_tokens = [1000, 1000, 1000, 0]
    mgr._rank_routed_total = [0, 0, 0, 0]

    rank = _route(mgr, [_FakeSeq("new", 100, dp_session_id="global-session")])[0]

    assert rank == 3
    assert mgr._dp_session_owners["global-session"] == 3
    assert len(mgr.get_dp_router_statistics()["requests_per_rank"]) == 4


def test_affinity_child_is_independent_of_parent_cache_owner():
    mgr = _make_mgr(4, session_affinity=True)
    parent_owner = 0
    mgr._dp_session_owners["parent"] = parent_owner
    mgr._rank_tokens[parent_owner] = 100_000
    rank = _route(
        mgr,
        [
            _FakeSeq(
                "child-seq",
                100,
                dp_session_id="child",
                dp_parent_session_id="parent",
            )
        ],
    )[0]
    assert rank != parent_owner
    assert mgr._dp_session_owners["child"] == rank
    assert mgr._dp_session_owners["parent"] == parent_owner
    assert mgr._dp_route_counters["affinity_parent_ignored_total"] == 1


def test_affinity_child_does_not_reserve_parent_owner():
    mgr = _make_mgr(3, session_affinity=True)
    child_rank = _route(
        mgr,
        [
            _FakeSeq(
                "child-seq",
                100,
                dp_session_id="child",
                dp_parent_session_id="parent",
            )
        ],
    )[0]
    parent_rank = _route(
        mgr,
        [_FakeSeq("parent-seq", 100, dp_session_id="parent")],
    )[0]
    # The child's first-turn charge participates in the parent's independent
    # placement instead of reserving or forcing the parent owner.
    assert parent_rank != child_rank
    assert mgr._dp_session_owners["parent"] == parent_rank


def test_explicit_rank_is_authoritative_and_becomes_session_owner():
    mgr = _make_mgr(4, session_affinity=True)
    mgr._rank_tokens = [0, 0, 0, 100_000]
    rank = _route(
        mgr,
        [
            _FakeSeq(
                "explicit",
                100,
                data_parallel_rank=3,
                dp_session_id="s",
            )
        ],
    )[0]
    assert rank == 3
    assert mgr._dp_session_owners["s"] == 3


def test_dp_router_statistics_exposes_locality_and_rank_load():
    mgr = _make_mgr(4, session_affinity=True)
    session = "observed-session"
    ranks = _route(
        mgr,
        [
            _FakeSeq("first", 100, dp_session_id=session),
            _FakeSeq("second", 200, dp_session_id=session),
        ],
    )
    owner = ranks[0]

    stats = mgr.get_dp_router_statistics()
    assert stats["affinity_new_total"] == 1
    assert stats["affinity_owner_hit_total"] == 1
    assert stats["affinity_spill_total"] == 0
    assert stats["requests_per_rank"][owner] == 2
    assert stats["inflight_requests_per_rank"][owner] == 2
    # First turn charges 100; the sticky second turn charges only its 100-token
    # growth, not the full cached 200-token prompt.
    assert stats["queued_prefill_tokens_per_rank"][owner] == 200
    assert stats["session_count_per_rank"][owner] == 1


def test_release_restores_counts():
    mgr = _make_mgr(3, strategy="least_tokens")
    seqs = [_FakeSeq(i, num_prompt_tokens=50) for i in range(6)]
    _route(mgr, seqs)
    for seq in seqs:
        mgr._release_seq_load(seq.id)
    assert mgr._rank_reqs == [0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0]
    assert mgr._seq_load == {}


def test_release_is_idempotent_no_leak_no_negative():
    mgr = _make_mgr(2, strategy="least_tokens")
    _route(mgr, [_FakeSeq("a", num_prompt_tokens=20)])
    mgr._release_seq_load("a")
    # Second release (e.g. abort after finish) must be a no-op.
    mgr._release_seq_load("a")
    # Unknown id must also be a no-op.
    mgr._release_seq_load("never-seen")
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [0, 0]


def test_first_output_releases_prefill_tokens_but_not_request_count():
    mgr = _make_mgr(2, strategy="least_tokens")
    seq = _FakeSeq("a", num_prompt_tokens=20)
    _route(mgr, [seq])
    rank = mgr._seq_load[seq.id][0]

    mgr._mark_seq_prefill_complete(seq.id)
    mgr._mark_seq_prefill_complete(seq.id)  # idempotent across stream chunks
    assert mgr._rank_tokens[rank] == 0
    assert mgr._rank_reqs[rank] == 1

    mgr._release_seq_load(seq.id)
    assert mgr._rank_reqs == [0, 0]


def test_burst_does_not_dogpile_one_rank():
    mgr = _make_mgr(4, strategy="least_tokens")
    # A burst of identical requests dispatched back-to-back (optimistic +1 in
    # the same locked loop) must spread, not all land on rank 0.
    _route(mgr, [_FakeSeq(i, num_prompt_tokens=128) for i in range(4)])
    assert mgr._rank_reqs == [1, 1, 1, 1]


def test_round_robin_ignores_load():
    mgr = _make_mgr(3, strategy="round_robin")
    # Pre-skew load; round_robin must still rotate purely by cursor.
    mgr._rank_tokens = [10_000, 0, 0]
    mgr._rank_reqs = [50, 0, 0]
    ranks = _route(mgr, [_FakeSeq(i) for i in range(6)])
    assert ranks == [0, 1, 2, 0, 1, 2]


def test_explicit_hint_takes_priority_and_is_charged():
    mgr = _make_mgr(4, strategy="least_tokens")
    ranks = _route(
        mgr,
        [
            _FakeSeq("h", num_prompt_tokens=30, data_parallel_rank=2),
            _FakeSeq("auto", num_prompt_tokens=30),
        ],
    )
    assert ranks[0] == 2
    # Hinted rank's load is counted so it participates in future balancing.
    assert mgr._rank_reqs[2] >= 1
    assert mgr._rank_tokens[2] >= 30


def test_dispatch_sends_explicit_hint_to_exact_engine_rank():
    class _RecordingSocket:
        def __init__(self):
            self.messages = []

        def send_multipart(self, parts, copy=False):
            self.messages.append(parts)

    mgr = _make_mgr(4, strategy="least_requests")
    mgr.engine_core_identities = [b"e0", b"e1", b"e2", b"e3"]
    mgr.input_sockets = [_RecordingSocket() for _ in range(4)]
    # Make rank 3 the least attractive load-balanced destination. An explicit
    # mesh hint must still route there exactly.
    mgr._rank_reqs = [0, 0, 0, 10]
    mgr._rank_tokens = [0, 0, 0, 10_000]

    seq = _FakeSeq("sticky", num_prompt_tokens=100, data_parallel_rank=3)
    mgr._dispatch_to_dp_ranks([seq])

    assert [len(sock.messages) for sock in mgr.input_sockets] == [0, 0, 0, 1]
    request_type, dispatched = pickle.loads(mgr.input_sockets[3].messages[0][1])
    assert request_type.name == "ADD"
    assert [item.id for item in dispatched] == [seq.id]


def test_invalid_hint_still_supported_via_add_request_validation():
    # _route mimics add_request; out-of-range hints are validated in
    # add_request itself, so here we only assert a valid hint routes exactly.
    mgr = _make_mgr(2, strategy="least_tokens")
    ranks = _route(mgr, [_FakeSeq("x", data_parallel_rank=1)])
    assert ranks == [1]


def test_reset_dp_router_clears_all_state():
    mgr = _make_mgr(3, strategy="least_tokens")
    mgr._dp_session_owners["session"] = 1
    mgr._dp_session_prompt_tokens["session"] = 1234
    _route(mgr, [_FakeSeq(i, num_prompt_tokens=40) for i in range(5)])
    mgr.reset_dp_router()
    assert mgr._rank_rotation_cursor == 0
    assert mgr._rank_reqs == [0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0]
    assert mgr._seq_load == {}
    assert mgr._dp_session_owners == {}
    assert mgr._dp_session_prompt_tokens == {}


def test_tie_break_rotates_starting_rank():
    # All ranks equal load -> successive picks rotate rather than always rank 0.
    mgr = _make_mgr(3, strategy="least_tokens")
    ranks = [mgr._select_dp_rank_locked() for _ in range(6)]
    assert ranks == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("strategy", ["round_robin", "least_requests", "least_tokens"])
def test_all_strategies_route_within_range(strategy):
    mgr = _make_mgr(4, strategy=strategy)
    ranks = _route(mgr, [_FakeSeq(i, num_prompt_tokens=i + 1) for i in range(20)])
    assert all(0 <= r < 4 for r in ranks)
    # Every dispatched request is accounted for.
    assert sum(mgr._rank_reqs) == 20


def test_resolve_and_validate_hints_rejects_out_of_range_without_side_effects():
    # An invalid hint anywhere in the batch must raise BEFORE any load is
    # charged, so a rejected batch cannot leak partial in-flight load.
    mgr = _make_mgr(4, strategy="least_requests")
    seqs = [_FakeSeq("ok", 100), _FakeSeq("bad", 100, data_parallel_rank=9)]
    with pytest.raises(ValueError):
        mgr._resolve_and_validate_hints(seqs)
    assert mgr._rank_reqs == [0, 0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0, 0]
    assert mgr._seq_load == {}


def test_resolve_and_validate_hints_accepts_and_returns_resolved():
    mgr = _make_mgr(4, strategy="least_requests")
    # In-range hint and no-hint (None) are both fine, and the resolved hints are
    # returned in order for the dispatch loop to reuse.
    hints = mgr._resolve_and_validate_hints(
        [_FakeSeq("a", 10, data_parallel_rank=3), _FakeSeq("b", 10)]
    )
    assert hints == [3, None]


def test_send_failure_rolls_back_undispatched_charge():
    # If a rank's send_multipart raises mid-batch, the seqs on ranks that were
    # NOT successfully handed off were charged but will never emit a finished
    # output to release them. Dispatch must roll those back so routing does not
    # skew permanently; already-sent ranks keep their (legitimate) charge.
    # dispatch pickles (EngineCoreRequestType.ADD, seqs) with the real enum.
    class _FakeSocket:
        def __init__(self, fail=False):
            self.fail = fail

        def send_multipart(self, parts, copy=False):
            if self.fail:
                raise RuntimeError("send failed")

    mgr = _make_mgr(4, strategy="least_requests")
    mgr.engine_core_identities = [b"e0", b"e1", b"e2", b"e3"]
    # Rank 2 fails; ranks 0/1 send first, rank 3 is never attempted.
    mgr.input_sockets = [
        _FakeSocket(),
        _FakeSocket(),
        _FakeSocket(fail=True),
        _FakeSocket(),
    ]
    seqs = [_FakeSeq(i, num_prompt_tokens=10) for i in range(4)]
    with pytest.raises(RuntimeError):
        mgr._dispatch_to_dp_ranks(seqs)
    # Ranks 0,1 dispatched -> stay charged; ranks 2 (failed) and 3 (never
    # attempted) -> rolled back to zero.
    assert mgr._rank_reqs == [1, 1, 0, 0]
    assert mgr._rank_tokens == [10, 10, 0, 0]
    assert set(mgr._seq_load.keys()) == {0, 1}


def test_reset_clears_even_with_charged_in_flight():
    # reset must fully clear counters even if requests are still charged (it
    # warns, but state must not be left dirty or go negative).
    mgr = _make_mgr(2, strategy="least_requests")
    _route(mgr, [_FakeSeq(i, num_prompt_tokens=50) for i in range(3)])
    assert mgr._seq_load  # charged
    mgr.reset_dp_router()
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [0, 0]
    assert mgr._seq_load == {}


def test_dp_lb_strategies_constant():
    assert DP_LB_STRATEGIES == ("round_robin", "least_requests", "least_tokens")
    assert "least_request" not in DP_LB_STRATEGIES  # guards against typos
