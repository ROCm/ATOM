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
from threading import Event, Lock, Thread

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
    mgr._prefix_router = None
    mgr._lb_lock = Lock()
    return mgr


def _route(mgr, seqs):
    """Replicate add_request's per-seq selection loop (hint > strategy)."""
    assigned = []
    with mgr._lb_lock:
        for seq in seqs:
            hint = getattr(seq, "data_parallel_rank", None)
            hint = int(hint) if hint is not None else None
            route = mgr._select_dp_rank_for_seq_locked(seq, hint)
            mgr._charge_seq_load_locked(seq, route.rank, credit=route.credit)
            assigned.append(route.rank)
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
        mgr._release_seq(seq.id)
    assert mgr._rank_reqs == [0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0]
    assert mgr._seq_load == {}


def test_release_is_idempotent_no_leak_no_negative():
    mgr = _make_mgr(2, strategy="least_tokens")
    _route(mgr, [_FakeSeq("a", num_prompt_tokens=20)])
    mgr._release_seq("a")
    # Second release (e.g. abort after finish) must be a no-op.
    mgr._release_seq("a")
    # Unknown id must also be a no-op.
    mgr._release_seq("never-seen")
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

    mgr._release_seq(seq.id)
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
    assert DP_LB_STRATEGIES == (
        "round_robin",
        "least_requests",
        "least_tokens",
        "prefix_spread",
    )
    assert "least_request" not in DP_LB_STRATEGIES  # guards against typos


def test_decode_bookkeeping_does_not_wait_for_an_unrelated_router_lock():
    mgr = _make_mgr(2, strategy="least_tokens")
    seq = _FakeSeq("a", num_prompt_tokens=20)
    _route(mgr, [seq])
    assert sum(mgr._rank_reqs) == 1
    assert sum(mgr._rank_tokens) == 20
    mgr._mark_seq_prefill_complete(seq.id)
    assert sum(mgr._rank_tokens) == 0
    completed = Event()

    def receive_decode_token():
        mgr._mark_seq_prefill_complete(seq.id)
        completed.set()

    with mgr._lb_lock:
        worker = Thread(target=receive_decode_token)
        worker.start()
        progressed_while_router_locked = completed.wait(timeout=2)
    worker.join(timeout=2)
    assert progressed_while_router_locked
    assert sum(mgr._rank_reqs) == 1
    assert sum(mgr._rank_tokens) == 0


# ── Prefix-spread routing ──────────────────────────────────────────────────


def _make_spread_mgr(n_ranks=2, tier=None):
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile
    from atom.model_engine.prefix_spread import PrefixSpreadRouter

    if tier is None:
        tier = OffloadTierProfile(loads=True, min_load_tokens=8192)
    mgr = _make_mgr(n_ranks, strategy="prefix_spread")
    mgr._prefix_router = PrefixSpreadRouter(32768, tier, max_seqs=256)
    mgr._dp_route_counters.update(
        dict.fromkeys(
            PrefixSpreadRouter.DECISIONS + ("prefix_moved_tokens",),
            0,
        )
    )
    return mgr


def _count(mgr, name):
    return mgr._dp_route_counters[name]


def _est(mgr, seq):
    """The prefix `seq` was credited (its prompt minus its prefill charge)."""
    return seq.num_prompt_tokens - mgr._seq_load[seq.id][2]


def _chain_seq(seq_id, tokens, first_block=0, session=None):
    import array

    from atom.model_engine.prefix_hash import ShippedChain

    seq = _FakeSeq(seq_id, num_prompt_tokens=tokens, dp_session_id=session)
    blocks = array.array("Q", range(first_block, first_block + tokens // 128))
    seq.prompt_hash_chain = ShippedChain(128, 0, tokens, 0, blocks)
    return seq


def test_prefix_spread_keeps_prefix_owner_without_a_backlog():
    mgr = _make_spread_mgr()
    first = _chain_seq(1, 100_000, session="s")
    (owner,) = _route(mgr, [first])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [0, 0]
    mgr._rank_reqs = [0, 0]
    mgr._rank_tokens[owner] = 30_000  # busier, but no backlog (<= 2 x idle)
    second = _chain_seq(2, 110_000, session="s")
    assert _route(mgr, [second]) == [owner]
    assert _count(mgr, "prefix_owner_total") == 1
    # Only the uncached extension is charged as prefill debt.
    assert mgr._seq_load[2][2] == 110_000 - 99_968


def test_short_prefix_keeps_affinity_even_below_the_load_threshold():
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 2048)])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [0, 0]
    mgr._rank_tokens[owner] = 30_000
    follower = _chain_seq(2, 3072)
    assert _route(mgr, [follower]) == [owner]
    assert _est(mgr, follower) == 2048
    assert mgr._seq_load[follower.id][2] == 1024


@pytest.mark.parametrize("minimum,moved", [(3072, 3072), (3073, 0), (4096, 0)])
def test_small_missing_prefix_still_places_on_the_idle_partial_holder(minimum, moved):
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    mgr = _make_spread_mgr(tier=OffloadTierProfile(True, minimum, 512))
    for seq_id, rank, length in ((1, 1, 13312), (2, 0, 16384)):
        warm = _chain_seq(seq_id, length)
        placed = mgr._prefix_router.place(warm, rank, 2)
        mgr._charge_seq_load_locked(warm, rank, credit=placed.credit)
        mgr._mark_seq_prefill_complete(seq_id)
    mgr._rank_tokens = [200_000, 0]
    request = _chain_seq(3, 16512)
    assert _route(mgr, [request]) == [1]
    assert _est(mgr, request) == 13312 + moved
    assert mgr._seq_load[request.id][2] == 16512 - 13312 - moved
    assert _count(mgr, "prefix_moved_tokens") == moved


@pytest.mark.parametrize("minimum,moved", [(4095, 4096), (4096, 4096), (4097, 0)])
def test_transfer_threshold_applies_to_the_chunk_aligned_difference(minimum, moved):
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    mgr = _make_spread_mgr(tier=OffloadTierProfile(True, minimum, 512))
    for seq_id, rank, length in ((1, 1, 12288), (2, 0, 16384)):
        warm = _chain_seq(seq_id, length)
        placed = mgr._prefix_router.place(warm, rank, 2)
        mgr._charge_seq_load_locked(warm, rank, credit=placed.credit)
        mgr._mark_seq_prefill_complete(seq_id)
    mgr._rank_tokens = [200_000, 0]
    request = _chain_seq(3, 16512)
    assert _route(mgr, [request]) == [1]
    assert _est(mgr, request) == 12288 + moved
    assert _count(mgr, "prefix_moved_tokens") == moved


def test_prefix_credit_is_not_rounded_to_the_old_sampling_interval():
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 12000)])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [0, 0]
    request = _chain_seq(2, 12544)
    assert _route(mgr, [request]) == [owner]
    # Complete 128-token hash blocks cover 11904 tokens, not 8192.
    assert _est(mgr, request) == 11904


def test_history_eviction_preserves_a_producers_follower_debt():
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    router.CAP = 4096
    producer = _chain_seq(1, 4096)
    placed = router.place(producer, 0, 2)
    mgr._charge_seq_load_locked(producer, 0, credit=placed.credit)
    follower = _chain_seq(2, 4224)
    mgr._rank_tokens[1] = 40_000
    assert _route(mgr, [follower]) == [0]
    assert _est(mgr, follower) == 4096
    unrelated = _chain_seq(3, 8192, first_block=10000)
    placed = router.place(unrelated, 1, 2)
    mgr._charge_seq_load_locked(unrelated, 1, credit=placed.credit)
    mgr._mark_seq_prefill_complete(unrelated.id)
    assert router.indexed_tokens > router.CAP  # active intervals are pinned
    assert router._rank_hits(producer.prompt_hash_chain, 2)[0][0] == 4096
    mgr._release_seq(producer.id)
    assert _est(mgr, follower) == 0
    assert mgr._seq_load[follower.id][2] == 4224
    mgr._release_seq(producer.id)
    assert mgr._seq_load[follower.id][2] == 4224
    mgr._mark_seq_prefill_complete(follower.id)
    assert not router.pending


def test_reindexing_an_evicted_prefix_keeps_its_inflight_producer():
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    router.CAP = 4096
    producer = _chain_seq(1, 4096)
    placed = router.place(producer, 0, 2)
    mgr._charge_seq_load_locked(producer, 0, credit=placed.credit)
    unrelated = _chain_seq(2, 8192, first_block=10000)
    placed = router.place(unrelated, 1, 2)
    mgr._charge_seq_load_locked(unrelated, 1, credit=placed.credit)
    mgr._mark_seq_prefill_complete(unrelated.id)
    newcomer = _chain_seq(3, 4096)
    placed = router.place(newcomer, 0, 2)
    mgr._charge_seq_load_locked(newcomer, 0, credit=placed.credit)
    assert router._rank_hits(producer.prompt_hash_chain, 2)[1] == 0
    mgr._release_seq(newcomer.id)
    assert router._rank_hits(producer.prompt_hash_chain, 2)[1] == 0
    mgr._release_seq(producer.id)
    assert not router.pending
    assert not router.seq_hashes


def test_new_follower_discovers_a_pinned_producer_above_the_history_budget():
    import array

    from atom.model_engine.prefix_hash import ShippedChain

    mgr = _make_spread_mgr(n_ranks=8)
    router = mgr._prefix_router
    router.CAP = 131072 * 16
    router.max_seqs = 32

    def request(seq_id, lineage):
        seq = _FakeSeq(seq_id, num_prompt_tokens=65536)
        seq.prompt_hash_chain = ShippedChain(
            16,
            0,
            65536,
            0,
            array.array("Q", range(lineage * 4096, (lineage + 1) * 4096)),
        )
        return seq

    for i in range(40):
        producer = request(i + 1, i)
        placed = router.place(producer, i % 8, 8)
        router.charge(producer.id, placed.credit)
    assert router.indexed_tokens > router.CAP
    mgr._rank_tokens = [65536] * 7 + [0]
    mgr._rank_reqs = [5] * 8
    follower = request(100, 0)
    route = router.select(
        follower, mgr._rank_tokens, mgr._rank_reqs, mgr._rank_load, lambda: 7
    )
    assert route.rank == 0
    assert route.credit == 65536
    router.charge(follower.id, route.credit)
    debt = router.release(1)
    assert debt == {100: 65536}
    for seq_id in list(router._in_flight):
        router.release(seq_id)
    assert not router.pending
    assert not router.seq_hashes
    assert not router.index
    assert router.indexed_tokens == 0


def test_explicit_placement_uses_the_same_tier_credit_as_a_move():
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    def placed(explicit):
        mgr = _make_spread_mgr(tier=OffloadTierProfile(True, 4096, 512))
        warm = _chain_seq(1, 16384)
        route = mgr._prefix_router.place(warm, 0, 2)
        mgr._prefix_router.charge(warm.id, route.credit)
        mgr._prefix_router.prefill_complete(warm.id)
        seq = _chain_seq(2, 20480)
        if explicit:
            return mgr._prefix_router.place(seq, 1, 2)
        return mgr._prefix_router.select(
            seq, [200000, 0], [0, 0], lambda r: r, lambda: 1
        )

    explicit, moved = placed(True), placed(False)
    assert explicit.rank == moved.rank == 1
    assert explicit.credit == moved.credit == 16384
    assert explicit.moved_tokens == moved.moved_tokens == 16384


def test_compressed_prefix_domains_do_not_mix_seeds_or_block_sizes():
    from atom.model_engine.prefix_hash import ShippedChain

    mgr = _make_spread_mgr()
    seq = _chain_seq(1, 4096)
    router = mgr._prefix_router
    route = router.place(seq, 0, 2)
    router.charge(seq.id, route.credit)
    router.prefill_complete(seq.id)
    chain = seq.prompt_hash_chain
    other_seed = ShippedChain(chain.hash_block_size, 1, 4096, 0, chain.hashes)
    other_blocks = ShippedChain(16, chain.cache_seed, 512, 0, chain.hashes)
    assert router._rank_hits(other_seed, 2) == ({}, None)
    assert router._rank_hits(other_blocks, 2) == ({}, None)


def test_reusing_a_completed_request_id_does_not_hide_its_tier_history():
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    warm = _chain_seq(9, 8192)
    placed = router.place(warm, 0, 2)
    router.charge(warm.id, placed.credit)
    router.prefill_complete(warm.id)
    router.release(warm.id)
    new = _chain_seq(9, 8192, first_block=10000)
    placed = router.place(new, 1, 2)
    router.charge(new.id, placed.credit)
    assert router._tier_hit(warm.prompt_hash_chain) == 8192
    router.release(new.id)
    assert not router.pending
    assert not router.seq_hashes
    assert not router._produced
    assert not router._active


def test_completed_reuse_does_not_refresh_the_owner_again():
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    warm = _chain_seq(1, 8192)
    placed = router.place(warm, 0, 2)
    router.charge(warm.id, placed.credit)
    router.prefill_complete(warm.id)
    reuse = _chain_seq(2, 8192)
    placed = router.place(reuse, 0, 2)
    router.charge(reuse.id, placed.credit)
    producer = _chain_seq(3, 8192)
    placed = router.place(producer, 1, 2)
    router.charge(producer.id, placed.credit)
    assert router._rank_hits(warm.prompt_hash_chain, 2)[1] == 1
    router.prefill_complete(reuse.id)
    assert router._rank_hits(warm.prompt_hash_chain, 2)[1] == 1


def test_an_older_request_cannot_confirm_a_later_reindexed_producer():
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    warm = _chain_seq(1, 8192)
    placed = router.place(warm, 0, 2)
    router.charge(warm.id, placed.credit)
    router.prefill_complete(warm.id)
    reuse = _chain_seq(2, 8192)
    placed = router.place(reuse, 0, 2)
    router.charge(reuse.id, placed.credit)
    router.CAP = 0
    router._trim_history()
    assert not router.index
    producer = _chain_seq(3, 8192)
    placed = router.place(producer, 0, 2)
    router.charge(producer.id, placed.credit)
    router.prefill_complete(reuse.id)
    assert router.pending
    assert router._tier_hit(warm.prompt_hash_chain) == 0
    router.release(producer.id)
    assert not router.pending
    assert not router.index


def test_prefix_spread_ignores_sessions_for_new_prefixes():
    mgr = _make_spread_mgr()
    seqs = [
        _chain_seq(i, 50_000, first_block=10_000 * i, session="same") for i in range(4)
    ]
    assert sorted(_route(mgr, seqs)) == [0, 0, 1, 1]
    assert _count(mgr, "prefix_cold_total") == 4


def test_busy_source_keeps_followers_unless_the_owner_is_backlogged():
    """While the prefix's source still prefills, a follower stays to reuse it
    from HBM -- unless the owner is backlogged and another rank is idle. Then
    it moves anyway and is charged its full prompt: the tier has nothing yet."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])  # still prefilling
    other = 1 - owner
    mgr._rank_tokens[other] = 40_000  # not idle
    assert _route(mgr, [_chain_seq(2, 110_000)]) == [owner]

    mgr._rank_tokens[other] = 0  # owner backlogged, other idle
    third = _chain_seq(3, 120_000)
    assert _route(mgr, [third]) == [other]
    assert _est(mgr, third) == 0
    assert mgr._seq_load[3][2] == 120_000


def test_a_fully_cached_follower_is_not_the_prefix_source():
    """A follower whose whole prompt is the prefix is charged 0 tokens and is
    still prefilling, but it only reuses the prefix: once the real source is
    done, the next request may move and load the prefix from the tier."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 98_304)])
    other = 1 - owner
    mgr._rank_tokens[other] = 40_000
    follower = _chain_seq(2, 98_304)
    assert _route(mgr, [follower]) == [owner]
    assert mgr._seq_load[2][2] == 0
    mgr._mark_seq_prefill_complete(1)  # the source is done; 2 still prefills
    mgr._rank_tokens = [0, 0]
    mgr._rank_tokens[owner] = 200_000
    third = _chain_seq(3, 110_000)
    assert _route(mgr, [third]) == [other]
    assert _est(mgr, third) == 98_304


def test_move_target_is_chosen_by_queued_prefill_not_combined_load():
    """The rank with no prefill work is the target even when its decode load
    makes it look heavier under the combined `least_tokens` score."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._mark_seq_prefill_complete(1)
    other = 1 - owner
    mgr._rank_tokens = [0, 0]
    mgr._rank_reqs = [0, 0]
    mgr._rank_tokens[owner], mgr._rank_reqs[owner] = 100_000, 10
    mgr._rank_reqs[other] = 250  # 0 queued prefill, many decodes
    assert mgr._rank_load(owner) < mgr._rank_load(other)
    assert _route(mgr, [_chain_seq(2, 110_000)]) == [other]


class _SinkSocket:
    def send_multipart(self, parts, copy=False):
        pass


def test_prefix_spread_tracks_load_through_dispatch():
    """Dispatch charges the in-flight load prefix spread scores ranks by."""
    mgr = _make_spread_mgr()
    mgr.engine_core_identities = [b"e0", b"e1"]
    mgr.input_sockets = [_SinkSocket(), _SinkSocket()]
    first = _chain_seq(1, 100_000)
    mgr._dispatch_to_dp_ranks([first])
    (owner,) = [r for r, n in enumerate(mgr._rank_reqs) if n]
    assert mgr._rank_tokens[owner] == 100_000
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens[owner] = 200_000  # backlogged
    moved = _chain_seq(2, 110_000)
    mgr._dispatch_to_dp_ranks([moved])
    assert mgr._seq_load[2][0] != owner
    assert _count(mgr, "prefix_moved_total") == 1


def test_prefix_spread_charges_full_prompt_when_a_moved_hit_is_not_loaded():
    """The connector only loads a hit of at least its minimum (8192 here,
    `OffloadTierProfile.min_load_tokens`). A smaller one that moves rank is
    recomputed there, so it is not credited."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 4_096)])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [0, 0]
    mgr._rank_reqs = [0, 0]
    mgr._rank_tokens[owner] = 200_000
    small = _chain_seq(2, 5_000)
    (rank,) = _route(mgr, [small])
    assert rank != owner
    assert mgr._seq_load[2][2] == 5_000
    # Staying on the owner credits the prefix whatever its size.
    mgr._mark_seq_prefill_complete(2)
    mgr._rank_tokens = [0, 0]
    mgr._rank_reqs = [0, 0]
    stay = _chain_seq(3, 5_000)
    assert _route(mgr, [stay]) == [rank]
    assert mgr._seq_load[3][2] == 5_000 - 4_992


def test_prefix_spread_moves_only_to_an_idle_rank():
    """Leave the prefix's rank only for a rank about to idle, and only from a
    real backlog; with both ranks busy, stay."""
    mgr = _make_spread_mgr()

    def place(case, q_owner, q_other):
        base = 100_000 * case
        mgr._rank_tokens = [0, 0]
        mgr._rank_reqs = [0, 0]
        (owner,) = _route(mgr, [_chain_seq(base, 100_000, first_block=base)])
        mgr._mark_seq_prefill_complete(base)
        other = 1 - owner
        mgr._rank_tokens = [0, 0]
        mgr._rank_reqs = [0, 0]
        mgr._rank_tokens[owner], mgr._rank_tokens[other] = q_owner, q_other
        (rank,) = _route(mgr, [_chain_seq(base + 1, 110_000, first_block=base)])
        mgr._mark_seq_prefill_complete(base + 1)
        return "owner" if rank == owner else "other"

    assert place(1, 100_000, 10_000) == "other"  # other about to idle
    assert place(2, 50_000, 10_000) == "owner"  # owner has no real backlog
    assert place(3, 500_000, 200_000) == "owner"  # both busy: stay
    mgr._prefix_router.idle_tokens = 8192  # the one knob: a smaller idle threshold
    assert place(4, 100_000, 10_000) == "owner"


def _dp4_idle_ratio_mgr(partial_completed=True):
    mgr = _make_spread_mgr(n_ranks=4)
    # rank0 ran the whole 100k prefix, rank2 its first 48k.
    mgr._prefix_router._record(_chain_seq(1, 100_000).prompt_hash_chain, 0, 1, 0)
    mgr._prefix_router._record(_chain_seq(2, 49_152).prompt_hash_chain, 2, 2, 0)
    mgr._prefix_router.prefill_complete(1)
    if partial_completed:
        mgr._prefix_router.prefill_complete(2)
    else:
        mgr._prefix_router.charge(2, 0)
    mgr._rank_tokens = [200_000, 0, 4_096, 8_192]
    mgr._rank_reqs = [0, 0, 0, 0]
    return mgr


def test_idle_ratio_prefers_idle_rank_holding_part_of_the_prefix():
    """Load first; among ranks about as idle, the one with more of the prefix."""
    mgr = _dp4_idle_ratio_mgr()
    seq = _chain_seq(3, 110_000)
    (rank,) = _route(mgr, [seq])
    assert rank == 2  # 4k above the lightest, holds 48k of the prefix
    assert _est(mgr, seq) == 99_968  # Local prefix plus transferable remainder.
    assert _count(mgr, "prefix_moved_tokens") == 99_968 - 49_152


def test_idle_ratio_load_wins_over_a_partial_prefix():
    mgr = _dp4_idle_ratio_mgr()
    mgr._rank_tokens[2] = 24_576  # still idle, but over idle / 2 above the lightest
    (rank,) = _route(mgr, [_chain_seq(3, 110_000)])
    assert rank == 1


def test_partial_holder_still_prefilling_is_not_an_hbm_hit():
    mgr = _dp4_idle_ratio_mgr(partial_completed=False)
    hits, owner = mgr._prefix_router._rank_hits(
        _chain_seq(3, 110_000).prompt_hash_chain, 4
    )
    assert owner == 0 and 2 not in hits
    (rank,) = _route(mgr, [_chain_seq(3, 110_000)])
    assert rank == 1  # no partial credit: the lightest idle rank


def test_followers_sharing_a_prefix_do_not_convoy_onto_one_rank():
    """Followers that only reuse a prefix the tier holds are not its busy
    source: the idle rule spreads them instead of pinning them all."""
    import array

    from atom.model_engine.prefix_hash import ShippedChain

    mgr = _make_spread_mgr()
    _route(mgr, [_chain_seq(1, 65_536)])
    mgr._mark_seq_prefill_complete(1)

    def follower(i):
        seq = _FakeSeq(100 + i, num_prompt_tokens=65_536 + 8_192)
        blocks = array.array(
            "Q", list(range(512)) + [10**6 + 64 * i + j for j in range(64)]
        )
        seq.prompt_hash_chain = ShippedChain(128, 0, 65_536 + 8_192, 0, blocks)
        return seq

    ranks = _route(mgr, [follower(i) for i in range(32)])
    assert set(ranks) == {0, 1}


def test_a_prefix_record_is_rolled_back_when_its_request_leaves_unprefilled():
    mgr = _make_spread_mgr()
    _route(mgr, [_chain_seq(1, 100_000)])
    mgr._release_seq(1)  # aborted / rejected before its first output
    assert not mgr._prefix_router.index
    second = _chain_seq(2, 100_000)
    _route(mgr, [second])
    assert _est(mgr, second) == 0  # no fake HBM hit
    mgr._mark_seq_prefill_complete(2)
    mgr._release_seq(2)  # finished normally: the record stays
    assert mgr._prefix_router.index


def test_an_aborted_move_leaves_no_hit_on_its_destination():
    """A moved request records the reused prefix on its destination only
    provisionally: aborted before its load and prefill, the destination does
    not keep a hit for KV that never arrived. A later request that completes
    there confirms the entry for good."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._mark_seq_prefill_complete(1)
    other = 1 - owner

    def moved(seq_id):
        mgr._rank_tokens = [0, 0]
        mgr._rank_reqs = [0, 0]
        mgr._rank_tokens[owner] = 200_000
        seq = _chain_seq(seq_id, 100_000)
        assert _route(mgr, [seq]) == [other]
        return seq

    moved(2)
    mgr._release_seq(2)  # aborted before its load
    hits, _ = mgr._prefix_router._rank_hits(_chain_seq(9, 100_000).prompt_hash_chain, 2)
    assert other not in hits

    moved(3)
    mgr._mark_seq_prefill_complete(3)
    mgr._release_seq(3)
    hits, _ = mgr._prefix_router._rank_hits(_chain_seq(9, 100_000).prompt_hash_chain, 2)
    assert hits.get(other) == 99_968


def test_a_producer_abort_hands_the_prefix_and_debt_to_its_follower():
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    other = 1 - owner
    mgr._rank_tokens[other] = 40_000
    follower = _chain_seq(2, 110_000)
    assert _route(mgr, [follower]) == [owner]
    assert _est(mgr, follower) == 99_968
    mgr._release_seq(1)
    assert mgr._seq_load[2][2] == 110_000
    assert mgr._rank_tokens[owner] == 110_000
    router = mgr._prefix_router
    hits, src_owner = router._rank_hits(follower.prompt_hash_chain, 2)
    assert src_owner == owner and hits[owner] == 109_952
    assert router.pending
    mgr._release_seq(1)  # repeated producer release cannot charge twice
    assert mgr._rank_tokens[owner] == 110_000
    mgr._mark_seq_prefill_complete(2)
    assert not router.pending
    mgr._release_seq(2)
    assert mgr._rank_tokens[owner] == 0
    assert router._rank_hits(follower.prompt_hash_chain, 2)[0][owner] == 109_952


def test_an_aborted_move_does_not_replace_a_local_prefill_producer():
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    mgr = _make_spread_mgr(tier=OffloadTierProfile())
    router = mgr._prefix_router
    local = _chain_seq(3, 8192)
    owner = _chain_seq(1, 8192)
    # C computes on rank1; completed A is the newest holder on rank0.
    placed = router.place(local, 1, 2)
    mgr._charge_seq_load_locked(local, 1, credit=placed.credit)
    placed = router.place(owner, 0, 2)
    mgr._charge_seq_load_locked(owner, 0, credit=placed.credit)
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens[0] = 100_000
    moved = _chain_seq(2, 12288)
    assert _route(mgr, [moved]) == [1]
    assert all(
        router.index[edge].get(1) == 3
        for edge, _ in router._path(local.prompt_hash_chain)
    )
    mgr._release_seq(2)
    assert all(
        router.index[edge].get(1) == 3
        for edge, _ in router._path(local.prompt_hash_chain)
    )
    mgr._mark_seq_prefill_complete(3)
    assert router._rank_hits(local.prompt_hash_chain, 2)[0][1] == 8192


def test_successive_producer_aborts_transfer_debt_without_leaking_it():
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._rank_tokens[1 - owner] = 40_000
    assert _route(mgr, [_chain_seq(2, 110_000), _chain_seq(3, 120_000)]) == [
        owner,
        owner,
    ]
    mgr._release_load(1)  # API abort precedes terminal output
    mgr._release_seq(1)
    assert mgr._seq_load[2][2] == 110_000
    mgr._release_seq(2)
    assert mgr._seq_load[3][2] == 120_000
    assert mgr._rank_tokens[owner] == 120_000
    mgr._release_seq(2)
    assert mgr._rank_tokens[owner] == 120_000
    mgr._mark_seq_prefill_complete(3)
    mgr._release_seq(3)
    assert mgr._rank_tokens[owner] == 0
    assert not mgr._prefix_router.pending
    assert not mgr._prefix_router._in_flight
    assert mgr._prefix_router.index


def test_producer_handoff_preserves_an_earlier_completed_prefix_credit():
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 4096)])
    mgr._mark_seq_prefill_complete(1)
    mgr._release_seq(1)
    assert _route(mgr, [_chain_seq(2, 8192), _chain_seq(3, 12288)]) == [owner, owner]
    assert mgr._seq_load[3][2] == 4096
    mgr._release_seq(2)
    # Only the provisional 4K..8K part loses credit, not the confirmed first 4K.
    assert mgr._seq_load[3][2] == 8192
    assert mgr._rank_tokens[owner] == 8192
    mgr._mark_seq_prefill_complete(3)
    mgr._release_seq(3)
    assert mgr._rank_tokens[owner] == 0


def test_an_abort_racing_the_first_stream_keeps_the_prefix():
    """The client disconnects after the engine finished the prefill but before
    the output thread saw its first STREAM: the abort (API thread) leaves the
    router alone, the STREAM confirms the prefix, and the terminal output
    releases the request with its prefix kept."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._release_load(1)  # abort_request
    assert mgr._prefix_router.prefilling(1)
    mgr._mark_seq_prefill_complete(1)  # the STREAM, on the output thread
    mgr._release_seq(1)  # the terminal output
    assert not mgr._prefix_router.prefilling(1)
    hits, src_owner = mgr._prefix_router._rank_hits(
        _chain_seq(9, 100_000).prompt_hash_chain, 2
    )
    assert src_owner == owner and hits[owner] == 99_968
    assert mgr._rank_reqs == [0, 0] and mgr._rank_tokens == [0, 0]


def test_prefix_spread_records_its_routes_like_every_strategy():
    """Cold requests go where `least_tokens` puts them (the rotation cursor
    advances once) and every decision lands in the route counters and the
    per-rank totals."""
    mgr = _make_spread_mgr()
    mgr._rank_tokens = [10, 0]
    cursor = mgr._rank_rotation_cursor
    (rank,) = _route(mgr, [_chain_seq(1, 4_096)])
    assert rank == 1 and mgr._rank_rotation_cursor == cursor + 1
    assert _count(mgr, "prefix_cold_total") == 1
    assert _count(mgr, "load_balanced_total") == 0
    assert mgr._rank_routed_total == [0, 1]
    assert mgr.get_dp_router_statistics()["prefix_cold_total"] == 1


def test_without_an_offload_tier_a_moved_request_is_charged_its_full_prompt():
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    mgr = _make_spread_mgr(tier=OffloadTierProfile())
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [0, 0]
    mgr._rank_tokens[owner] = 200_000
    moved = _chain_seq(2, 110_000)
    assert _route(mgr, [moved]) == [1 - owner]
    assert _est(mgr, moved) == 0
    assert mgr._seq_load[2][2] == 110_000


def test_offload_tier_profile_reads_the_kv_transfer_config(monkeypatch):
    from types import SimpleNamespace

    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "4096")

    def tier(kvc):
        return KVConnectorFactory.offload_tier(SimpleNamespace(kv_transfer_config=kvc))

    assert not tier({}).loads
    mp = {"kv_connector": "lmcache_mp", "kv_role": "offload"}
    assert tier(mp).loads
    assert tier(mp).min_load_tokens == 4096
    assert not tier({"kv_connector": "lmcache_mp", "kv_role": "kv_producer"}).loads
    assert not tier({"kv_connector": "lmcache_mp", "kv_role": "kv_consumer"}).loads
    assert not tier({"kv_connector": "mooncake_store", "kv_role": "kv_consumer"}).loads
    assert tier({"kv_connector": "lmcache_mp", "kv_role": "kv_both"}).loads
    pd = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    assert not tier({"kv_connector": "multi", "connectors": [pd]}).loads
    assert tier({"kv_connector": "mooncake_store", "kv_role": "offload"}).loads
    # Per-rank tiers (local CPU / disk): another rank's prefix is not loadable.
    dense = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
    assert not tier(dense).loads
    assert tier(dense).min_load_tokens == 4096
    # Shared-tier recovery also works under `multi`.
    both = tier({"kv_connector": "multi", "connectors": [pd, mp]})
    assert both.loads


@pytest.mark.parametrize(
    "local,hit,min_load,credit",
    [
        (4096, 8192, 512, 0),
        (4096, 16384, 512, 8192),
        (0, 12288, 512, 8192),
        (8192, 16384, 8192, 8192),
        (4096, 16384, 8193, 0),
    ],
)
def test_store_tier_credit_counts_only_aligned_loads(local, hit, min_load, credit):
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    tier = OffloadTierProfile(True, min_load, load_chunk_tokens=8192)
    assert tier.loadable_tokens(local, hit) == credit


def test_partial_store_hit_does_not_hide_recomputed_prefill(monkeypatch):
    from types import SimpleNamespace

    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "512")
    kvc = {
        "kv_connector": "mooncake_store",
        "kv_role": "offload",
        "kv_connector_extra_config": {"mooncake_store.chunk_tokens": 8192},
    }
    tier = KVConnectorFactory.offload_tier(SimpleNamespace(kv_transfer_config=kvc))
    assert tier.load_chunk_tokens == 8192
    mgr = _make_spread_mgr(n_ranks=3, tier=tier)
    router = mgr._prefix_router
    owner = _chain_seq(1, 8192)
    placed = router.place(owner, 0, 3)
    router.charge(owner.id, placed.credit)
    router.prefill_complete(owner.id)
    partial = _chain_seq(2, 4096)
    placed = router.place(partial, 1, 3)
    router.charge(partial.id, placed.credit)
    router.prefill_complete(partial.id)
    mgr._rank_tokens = [100_000, 0, 0]
    moved = _chain_seq(3, 12288)
    assert _route(mgr, [moved]) == [1]
    assert _est(mgr, moved) == 4096
    assert mgr._seq_load[moved.id][2] == 8192
    assert _count(mgr, "prefix_moved_tokens") == 0


def test_reset_keeps_the_prefix_index_across_batches():
    """`reset_dp_router` runs before every offline batch; the ranks' prefix
    caches outlive it, so the second batch still finds the shared prefix."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 65_536)])
    mgr._mark_seq_prefill_complete(1)
    mgr._release_seq(1)
    mgr.reset_dp_router()
    second = _chain_seq(2, 70_000)
    assert _route(mgr, [second]) == [owner]
    assert _est(mgr, second) == 65_536
    assert _count(mgr, "prefix_owner_total") == 1


def test_a_rank_full_of_decodes_is_not_idle():
    """A rank at `max_num_seqs` queues no prefill tokens but cannot start the
    request: the owner keeps it."""
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._mark_seq_prefill_complete(1)
    other = 1 - owner
    mgr._rank_tokens = [0, 0]
    mgr._rank_tokens[owner] = 200_000
    mgr._rank_reqs = [0, 0]
    mgr._rank_reqs[other] = 256
    assert _route(mgr, [_chain_seq(2, 110_000)]) == [owner]
    mgr._rank_reqs[other] = 255
    assert _route(mgr, [_chain_seq(3, 110_000)]) == [other]


def test_a_terminal_output_completes_the_prefill_only_after_forward():
    """The scheduler's own fact decides (`Sequence.prefill_completed`), not
    whether a token came out. An error mid-prefill left no prefix: the prefix
    entries are dropped. An abort after the last chunk completed did:
    the entries stay, though no first token ever came."""
    from types import SimpleNamespace

    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._release_terminal_seq(SimpleNamespace(id=1, prefill_completed=False))
    assert not mgr._prefix_router.index  # dropped: its KV was never written

    (owner,) = _route(mgr, [_chain_seq(2, 100_000)])
    mgr._release_terminal_seq(SimpleNamespace(id=2, prefill_completed=True))
    hits, src_owner = mgr._prefix_router._rank_hits(
        _chain_seq(9, 100_000).prompt_hash_chain, 2
    )
    assert src_owner == owner and hits[owner] == 99_968
    assert mgr._rank_reqs == [0, 0] and mgr._rank_tokens == [0, 0]


def test_an_explicit_rank_indexes_the_prefix_it_lands():
    """A proxy pins a prompt to a rank: that rank caches it, so a later
    unhinted request with the prefix finds its owner instead of going cold."""
    mgr = _make_spread_mgr()
    pinned = _chain_seq(1, 100_000)
    pinned.data_parallel_rank = 1
    assert _route(mgr, [pinned]) == [1]
    assert _count(mgr, "explicit_total") == 1
    mgr._mark_seq_prefill_complete(1)
    follower = _chain_seq(2, 110_000)
    assert _route(mgr, [follower]) == [1]
    assert _count(mgr, "prefix_owner_total") == 1
    assert _est(mgr, follower) == 99_968
    # A second pinned request credits what that rank holds, as a kept one.
    again = _chain_seq(3, 100_000)
    again.data_parallel_rank = 1
    _route(mgr, [again])
    assert _est(mgr, again) == 99_968


def test_reset_removes_provisional_holders_but_keeps_completed():
    mgr = _make_spread_mgr()
    _route(mgr, [_chain_seq(1, 65536)])
    mgr._mark_seq_prefill_complete(1)
    _route(mgr, [_chain_seq(2, 131072, first_block=1000)])
    router = mgr._prefix_router
    completed = {
        h: dict(holders)
        for h, holders in router.index.items()
        if not any((h, rank) in router.pending for rank in holders)
    }
    assert router.pending
    mgr.reset_dp_router()
    assert router.index == completed
    assert not router.pending
    assert not router.seq_hashes


def test_a_pending_restore_makes_its_target_the_provisional_owner():
    mgr = _make_spread_mgr()
    (owner,) = _route(mgr, [_chain_seq(1, 65536)])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens[owner] = 200000
    # Longer than the prefix, so the whole prefix is the tier hit (a prompt
    # that is entirely a hit steps back one token and recomputes its last).
    moved = _chain_seq(2, 65536 + 4096)
    (target,) = _route(mgr, [moved])
    assert target != owner
    probe = _chain_seq(9, 65536).prompt_hash_chain
    # While the move's load is in flight its target is the prefix's newest
    # (provisional) holder: followers reuse it there instead of loading again.
    hits, actual_owner = mgr._prefix_router._rank_hits(probe, 2)
    assert actual_owner == target
    assert hits[owner] == 65536
    mgr._mark_seq_prefill_complete(2)
    hits, _ = mgr._prefix_router._rank_hits(probe, 2)
    assert hits[target] == 65536


def test_a_follower_completing_confirms_the_prefix_it_reused():
    """B reuses the prefix A is still prefilling, then completes: rank0 really
    holds it. The entry must stop naming A as an unfinished producer, and a
    later request that recomputes it and aborts must not delete it."""
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    a = _chain_seq(1, 8192)
    (owner,) = _route(mgr, [a])
    b = _chain_seq(2, 8192)
    assert _route(mgr, [b]) == [owner]
    mgr._mark_seq_prefill_complete(2)
    assert all(
        seq_id not in router._in_flight
        for holders in router.index.values()
        for seq_id in holders.values()
    ), "a confirmed entry still names an unfinished producer"

    # A recomputes nothing it did not hold; a request charged the full prompt
    # (est_hit 0) joins and aborts, as does A.
    mgr._rank_tokens = [0, 0]
    mgr._rank_tokens[1 - owner] = 200_000
    c = _chain_seq(3, 8192)
    router.place(c, owner, 2)
    mgr._release_seq(1)
    mgr._release_seq(3)
    assert router._rank_hits(a.prompt_hash_chain, 2)[0].get(owner) == 8192


def test_the_tier_serves_a_prefix_any_finished_holder_put_there():
    """The newest holder of a prefix can be a request still prefilling it on
    another rank while a finished copy has long been in the tier. A request
    moved to an idle rank is credited that prefix."""
    mgr = _make_spread_mgr(n_ranks=3)
    router = mgr._prefix_router
    s = _chain_seq(1, 16384)
    placed = router.place(s, 0, 3)
    mgr._charge_seq_load_locked(s, 0, credit=placed.credit)
    mgr._mark_seq_prefill_complete(1)  # S finished: the tier holds the prefix
    s2 = _chain_seq(2, 16384)
    placed = router.place(s2, 0, 3)
    mgr._charge_seq_load_locked(
        s2, 0, credit=placed.credit
    )  # keeps S's owner busy, never completes
    p = _chain_seq(3, 16384)
    router.place(p, 1, 3)  # P recomputes it on rank1: the newest holder
    mgr._charge_seq_load_locked(p, 1)
    mgr._rank_tokens = [200_000, 200_000, 0]
    r = _chain_seq(4, 20480)
    assert _route(mgr, [r]) == [2]
    assert _est(mgr, r) == 16384


def test_tier_credit_follows_the_offload_schedulers_load_decision():
    """The router credits what the offload scheduler will load: a whole-prompt
    hit steps back one token before flooring to a chunk (G4)."""
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile
    from atom.kv_transfer.offload.load_math import loadable_hit, tier_load_tokens

    tier = OffloadTierProfile(True, 512, load_chunk_tokens=256)
    # Prompt 8192, fully hit: the scheduler loads 8192 - 256, not 8192.
    assert loadable_hit(8192, 8192, 256) == 7936
    assert tier.loadable_tokens(0, 8192, 8192) == 7936
    assert tier.loadable_tokens(0, 8192) == 8192  # no prompt length: no step back
    # An unaligned local prefix starts the load at the next boundary.
    assert tier_load_tokens(300, 7936, 256, 512) == 7936 - 512
    assert tier_load_tokens(300, 7936, 256, 8192) == 0


def test_lmcache_mp_declares_its_chunk_through_the_registry(monkeypatch):
    from types import SimpleNamespace

    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    monkeypatch.delenv("LMCACHE_CHUNK_SIZE", raising=False)
    mp = {
        "kv_connector": "lmcache_mp",
        "kv_role": "offload",
        "kv_connector_extra_config": {"lmcache.chunk_size": 512},
    }
    cfg = SimpleNamespace(kv_transfer_config=mp)
    assert KVConnectorFactory.offload_tier(cfg).load_chunk_tokens == 512
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "1024")
    cfg = SimpleNamespace(
        kv_transfer_config={"kv_connector": "lmcache_mp", "kv_role": "offload"}
    )
    assert KVConnectorFactory.offload_tier(cfg).load_chunk_tokens == 1024


def test_a_full_length_hit_is_not_credited_when_nothing_would_load():
    """An 8192-token prompt fully hit in an 8192-token-chunk Store: the
    connector steps back a token, floors to a chunk and loads nothing, so the
    moved request is charged its whole prompt."""
    from atom.kv_transfer.disaggregation.factory import OffloadTierProfile

    tier = OffloadTierProfile(True, 512, load_chunk_tokens=8192)
    assert tier.loadable_tokens(0, 8192, 8192) == 0
    mgr = _make_spread_mgr(n_ranks=2, tier=tier)
    owner = _chain_seq(1, 8192)
    placed = mgr._prefix_router.place(owner, 0, 2)
    mgr._charge_seq_load_locked(owner, 0, credit=placed.credit)
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [100_000, 0]
    moved = _chain_seq(2, 8192)
    assert _route(mgr, [moved]) == [1]
    assert _est(mgr, moved) == 0


def test_a_moved_request_loading_from_the_tier_is_its_own_holder_on_the_target():
    """M is moved to T and credited a prefix from the tier while another
    request P still prefills it elsewhere (an older finished copy made the tier
    ready). T's entry must name M, not P: when M aborts, followers that joined
    on T are handed the entry and the debt instead of keeping credit for a
    prefix no request is loading."""
    mgr = _make_spread_mgr(n_ranks=3)
    router = mgr._prefix_router
    s = _chain_seq(1, 16384)
    placed = router.place(s, 0, 3)
    mgr._charge_seq_load_locked(s, 0, credit=placed.credit)
    mgr._mark_seq_prefill_complete(1)  # a finished copy: the tier holds it
    p = _chain_seq(2, 16384)
    placed = router.place(p, 1, 3)
    mgr._charge_seq_load_locked(
        p, 1, credit=placed.credit
    )  # P prefills on rank1, the newest holder
    mgr._rank_tokens = [40_000, 200_000, 0]  # only rank2 is idle
    m = _chain_seq(3, 20480)
    assert _route(mgr, [m]) == [2]
    names = [holders[2] for holders in router.index.values() if 2 in holders]
    assert names and all(name == 3 for name in names)


def test_a_shorter_finished_copy_in_the_tier_is_credited():
    """The owner is still prefilling the whole prefix while another rank
    finished a shorter part of it earlier: that shorter part is in the tier and
    a request moved to an idle rank is credited it, not charged its whole
    prompt."""
    mgr = _make_spread_mgr(n_ranks=3)
    router = mgr._prefix_router
    short = _chain_seq(1, 8192)
    placed = router.place(short, 1, 3)
    mgr._charge_seq_load_locked(short, 1, credit=placed.credit)
    mgr._mark_seq_prefill_complete(1)  # 8k of P is in the tier
    full = _chain_seq(2, 16384)
    placed = router.place(full, 0, 3)
    mgr._charge_seq_load_locked(
        full, 0, credit=placed.credit
    )  # rank0 still prefilling all 16k
    mgr._rank_tokens = [200_000, 40_000, 0]  # only rank2 is idle
    moved = _chain_seq(3, 20480)
    assert _route(mgr, [moved]) == [2]
    assert _est(mgr, moved) == 8192


def test_owner_ties_go_to_the_rank_that_used_the_prefix_last():
    """Ranks 0 and 1 both hold P and rank 1 recorded it last; a request that
    then reuses P on rank 0 makes rank 0 the owner on a tie."""
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    for seq_id, rank in ((1, 0), (2, 1)):
        seq = _chain_seq(seq_id, 8192)
        placed = router.place(seq, rank, 2)
        mgr._charge_seq_load_locked(seq, rank, credit=placed.credit)
        mgr._mark_seq_prefill_complete(seq_id)
    probe = _chain_seq(9, 8192).prompt_hash_chain
    assert router._rank_hits(probe, 2)[1] == 1
    reuse = _chain_seq(3, 8192)
    placed = router.place(reuse, 0, 2)
    mgr._charge_seq_load_locked(reuse, 0, credit=placed.credit)
    assert router._rank_hits(probe, 2)[1] == 0


def test_an_owner_with_every_slot_taken_sends_its_prefix_to_an_idle_rank():
    """The owner queues no prefill but runs `max_seqs` decodes, so a request
    could not start there; with another rank idle the prefix's requests go
    there instead of piling onto the full owner."""
    mgr = _make_spread_mgr()
    router = mgr._prefix_router
    router.max_seqs = 32
    (owner,) = _route(mgr, [_chain_seq(1, 100_000)])
    mgr._mark_seq_prefill_complete(1)
    mgr._rank_tokens = [0, 0]
    mgr._rank_reqs = [0, 0]
    mgr._rank_reqs[owner] = 32
    ranks = _route(mgr, [_chain_seq(100 + i, 100_000) for i in range(40)])
    assert ranks.count(owner) == 0
    assert mgr._rank_reqs[owner] == 32


@pytest.mark.parametrize(
    "ranks,pp,builds", [(1, 1, False), (2, 2, False), (2, 1, True)]
)
def test_prefix_spread_is_checked_and_built_only_where_it_routes(
    monkeypatch, ranks, pp, builds
):
    """One DP rank, or pipeline parallelism, routes nothing: the strategy has
    no effect, so a missing prompt hash chain does not stop startup. With DP
    ranks to route between it does."""
    from types import SimpleNamespace

    mgr = _make_mgr(ranks, strategy="prefix_spread")
    mgr.global_engine_count = ranks
    config = SimpleNamespace(
        enable_prefix_caching=True,
        dp_load_balance="prefix_spread",
        pipeline_parallel_size=pp,
        max_num_seqs=256,
        kv_transfer_config=None,
    )
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "0")
    if builds:
        with pytest.raises(ValueError, match="ATOM_FRONTEND_PREFIX_HASH"):
            mgr._build_prefix_router(config)
        monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "1")
        assert mgr._build_prefix_router(config) is not None
    else:
        assert mgr._build_prefix_router(config) is None
