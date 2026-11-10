# SPDX-License-Identifier: MIT
# Unit tests for `--dp-load-balance share_cache` (atom/model_engine/share_cache.py)
# and its CoreManager bookkeeping (rank-reported prefill backlog).

import os
from collections import deque
from threading import Lock
from types import SimpleNamespace

from atom.model_engine.engine_core_mgr import DP_LB_STRATEGIES, CoreManager
from atom.model_engine.share_cache import ShareCacheRouter

HBS = 16


def _seq(seq_id, tokens):
    hashes = [hash(tuple(tokens[: (k + 1) * HBS])) for k in range(len(tokens) // HBS)]
    return SimpleNamespace(
        id=seq_id,
        num_prompt_tokens=len(tokens),
        dp_session_id=None,
        prompt_hash_chain=SimpleNamespace(hash_block_size=HBS, hashes=hashes),
    )


def _route(router, seq, backlog, reqs):
    r = router.select(seq, backlog, reqs)
    cost = seq.num_prompt_tokens - router.charge(seq.id)
    backlog[r.rank] += cost
    reqs[r.rank] += 1
    return r, cost


PREFIX = list(range(40_000))


def test_strategy_is_registered():
    assert "share_cache" in DP_LB_STRATEGIES


def test_cold_prompt_goes_to_least_backlog():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    route, cost = _route(router, _seq(1, PREFIX), [9_000, 0], [0, 0])
    assert route.rank == 1 and route.decision == "share_cache_cold_total"
    assert cost == len(PREFIX)


def test_prefix_in_tier_is_credited_on_any_rank():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq(1, PREFIX), backlog, reqs)
    router.prefill_complete(1)
    backlog[first.rank] += 100_000  # the producer's rank is busy now
    route, cost = _route(
        router, _seq(2, PREFIX + list(range(10**6, 10**6 + 2_000))), backlog, reqs
    )
    assert route.rank != first.rank
    # Credited the sampled prefix (multiple of STRIDE_TOKENS).
    assert cost == 42_000 - (40_000 // 1024) * 1024


def test_prefix_still_prefilling_is_credited_only_on_its_rank():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq(1, PREFIX), backlog, reqs)
    other = 1 - first.rank
    seq = _seq(2, PREFIX + list(range(10**6, 10**6 + 2_000)))
    hit = (40_000 // 1024) * 1024
    # The other rank pays the whole prompt; the producer's rank only the rest.
    route = router.select(seq, list(backlog), list(reqs))
    expected = min(
        (backlog[first.rank] + 42_000 - hit + 512 * reqs[first.rank], first.rank),
        (backlog[other] + 42_000 + 512 * reqs[other], other),
    )[1]
    assert route.rank == expected


def test_unprefilled_producer_drops_its_entries():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    _route(router, _seq(1, PREFIX), backlog, reqs)
    router.release(1)  # aborted before its prefill completed
    _, cost = _route(router, _seq(2, PREFIX + [1] * 1_000), [0, 0], [0, 0])
    assert cost == 41_000


def _bare_mgr(n=2):
    mgr = CoreManager.__new__(CoreManager)
    mgr.local_engine_count = n
    mgr._rank_tokens = [0] * n
    mgr._seq_load = {}
    mgr._lb_lock = Lock()
    mgr._backlog_report = [None] * n
    mgr._backlog_unseen = [deque() for _ in range(n)]
    mgr._backlog_unseen_tokens = [0] * n
    mgr._backlog_sent = [0] * n
    mgr._backlog_popped = [0] * n
    mgr._backlog_unsent = {}
    mgr._backlog_unsent_tokens = [0] * n
    return mgr


def test_backlog_is_report_plus_unreported_charges():
    mgr = _bare_mgr()
    mgr._rank_tokens = [7, 9]
    assert mgr._backlog_effective_tokens_locked() == [7, 9]  # no report yet
    mgr._seq_load = {"a": (0, 1, 1_000), "b": (0, 1, 2_000), "c": (0, 1, 4_000)}
    mgr._backlog_note_sent_locked(0, [SimpleNamespace(id="a"), SimpleNamespace(id="b")])
    mgr._backlog_report[0] = (1, 300)  # rank 0 ingested "a" and has 300 left
    mgr._backlog_pop_seen_locked(0)
    assert mgr._backlog_effective_tokens_locked()[0] == 300 + 2_000
    mgr._backlog_note_sent_locked(0, [SimpleNamespace(id="c")])
    mgr._backlog_report[0] = (3, 50)
    mgr._backlog_pop_seen_locked(0)
    assert mgr._backlog_effective_tokens_locked()[0] == 50


def test_report_ahead_of_note_sent_does_not_double_count():
    mgr = _bare_mgr()
    mgr._seq_load = {"a": (1, 1, 5_000)}
    mgr._backlog_report[1] = (1, 5_000)  # the rank ingested "a" before we noted it
    mgr._backlog_note_sent_locked(1, [SimpleNamespace(id="a")])
    assert mgr._backlog_effective_tokens_locked()[1] == 5_000


def test_a_charged_seq_not_sent_yet_counts_for_the_next_one_of_its_batch():
    mgr = _bare_mgr()
    mgr._seq_load = {"a": (0, 1, 30_000), "b": (0, 1, 30_000)}
    mgr._backlog_report = [(0, 0), (0, 0)]
    first, second = SimpleNamespace(id="a"), SimpleNamespace(id="b")
    mgr._backlog_note_charged_locked(first, 0)
    assert mgr._backlog_effective_tokens_locked() == [30_000, 0]  # rank 1 is lighter
    mgr._backlog_note_charged_locked(second, 1)
    # Sending moves the charges to the unreported ones, none counted twice.
    mgr._backlog_note_sent_locked(0, [first])
    mgr._backlog_note_sent_locked(1, [second])
    assert mgr._backlog_effective_tokens_locked() == [30_000, 30_000]
    assert mgr._backlog_unsent == {} and mgr._backlog_unsent_tokens == [0, 0]


def test_a_released_unsent_seq_gives_its_charge_back():
    mgr = _bare_mgr()
    mgr._seq_load = {"a": (0, 1, 30_000)}
    mgr._backlog_report = [(0, 0), (0, 0)]
    mgr._backlog_note_charged_locked(SimpleNamespace(id="a"), 0)
    assert mgr._backlog_drop_unsent_locked("a") == 30_000
    assert mgr._backlog_drop_unsent_locked("a") == 0  # idempotent
    assert mgr._backlog_effective_tokens_locked() == [0, 0]


def test_load_threshold_is_per_config_and_preserves_explicit_override(monkeypatch):
    from atom.config import Config
    from atom.kv_transfer.offload.config import min_load_tokens

    monkeypatch.delenv("OFFLOAD_MIN_LOAD_TOKENS", raising=False)
    store = {"kv_connector": "mooncake_store", "kv_role": "offload"}
    pd = {"kv_connector": "mooncake", "kv_role": "kv_producer"}

    def snapshot(kv):
        config = object.__new__(Config)
        config.kv_transfer_config = kv
        config._resolve_offload_min_load_tokens()
        return config

    first = snapshot(store)
    assert min_load_tokens(first) == 512
    assert (
        min_load_tokens(snapshot({"kv_connector": "multi", "connectors": [pd, store]}))
        == 512
    )
    assert min_load_tokens(snapshot({"kv_connector": "lmcache_mp"})) == 8192
    assert min_load_tokens(snapshot({})) == 8192
    assert "OFFLOAD_MIN_LOAD_TOKENS" not in os.environ
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "2048")
    assert min_load_tokens(snapshot(store)) == 2048
    assert min_load_tokens(snapshot({"kv_connector": "lmcache_mp"})) == 2048
    assert min_load_tokens(first) == 512  # workers retain their parent's snapshot
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
    assert min_load_tokens(snapshot(store)) == 0


def test_a_newcomer_does_not_take_a_prefix_from_an_owner_still_prefilling_it():
    """A prefills a system prompt on rank 0; B, same prompt, is placed on rank 1.
    A finishes, so the prefix is in the tier. B aborts before its first token:
    that must not make the prefix cold."""
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq("A", PREFIX), backlog, reqs)
    backlog[first.rank] = 10_000_000  # B goes to the other rank for backlog
    second, _ = _route(router, _seq("B", PREFIX), backlog, reqs)
    assert second.rank != first.rank
    router.prefill_complete("A")
    router.release("B")  # B left before prefilling anything
    router.select(_seq("C", PREFIX + [9] * 2_000), [0, 0], [0, 0])
    assert router.charge("C") > 0  # the prefix is still in the tier: credited


def test_the_newcomer_carries_the_entry_if_the_owner_leaves_unprefilled():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq("A", PREFIX), backlog, reqs)
    backlog[first.rank] = 10_000_000
    second, _ = _route(router, _seq("B", PREFIX), backlog, reqs)
    router.release("A")  # aborted before its prefill completed
    router.prefill_complete("B")
    probe = router.select(_seq("C", PREFIX + [9] * 2_000), [0, 0], [0, 0])
    assert router.charge("C") > 0
    assert probe.rank == second.rank  # B holds the prefix now


def _scheduler_with(waiting, running=()):
    from collections import deque

    class Sched:
        def __init__(self):
            self.waiting = deque(waiting)
            self.running = list(running)

        def pending_prefill_tokens(self, seq, *, estimate=False):
            return seq.tokens

    return Sched()


def test_the_backlog_report_skips_aborted_requests_and_bounds_its_scan():
    from atom.model_engine.prefill_backlog import prefill_backlog
    from atom.model_engine.sequence import SequenceStatus

    live = SimpleNamespace(
        status=SequenceStatus.WAITING, tokens=100, num_completion_tokens=0
    )
    dead = SimpleNamespace(
        status=SequenceStatus.ABORTED, tokens=10_000, num_completion_tokens=0
    )
    assert prefill_backlog(_scheduler_with([live, dead, live])) == 200  # no abort
    # 10 scanned, 90 extrapolated.
    assert prefill_backlog(_scheduler_with([live] * 100), scan=10) == 10_000
    decoding = SimpleNamespace(
        status=SequenceStatus.RUNNING, tokens=500, num_completion_tokens=3
    )
    prefilling = SimpleNamespace(
        status=SequenceStatus.RUNNING, tokens=40, num_completion_tokens=0
    )
    assert prefill_backlog(_scheduler_with([], [decoding, prefilling])) == 40


def test_a_promoted_backup_that_also_aborts_takes_the_entry_with_it():
    """A owns a cold prefix and B is its backup. A aborts, B is promoted; B then
    aborts before completing: nothing may keep crediting a prefix neither
    computed."""
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq("A", PREFIX), backlog, reqs)
    backlog[first.rank] = 10_000_000
    _route(router, _seq("B", PREFIX), backlog, reqs)
    router.release("A")
    router.release("B")
    assert not router.index and not router.backups
    router.select(_seq("C", PREFIX + [9] * 2_000), [0, 0], [0, 0])
    assert router.charge("C") == 0  # the prefix is cold again


def test_a_rank_that_did_not_find_the_prefix_drops_the_stale_entries():
    """The tier lost the prefix (failed save, eviction): the rank reports how
    much it reused, the entries past that are dropped, and the request that
    computes them becomes their producer."""
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    _route(router, _seq(1, PREFIX), backlog, reqs)
    router.prefill_complete(1)
    route, cost = _route(router, _seq(2, PREFIX + [7] * 2_000), backlog, reqs)
    assert cost < 42_000  # placed on the prefix the index lists
    # Its rank found only 8192 tokens.
    assert router.note_reuse(2, 8192)
    _, src_rank, src_seq = router._longest_hit(_seq(3, PREFIX).prompt_hash_chain)
    assert src_seq == 2 and src_rank == route.rank  # seq 2 produces the rest
    assert router.prefilling(2)
    # Once seq 2 completes, the whole prefix counts as loadable again.
    router.prefill_complete(2)
    _, cost3 = _route(router, _seq(3, PREFIX), backlog, reqs)
    assert cost3 == 40_000 - (40_000 // 1024) * 1024
    # A report at the estimate (one hash block of replay below it) drops nothing.
    assert not router.note_reuse(3, (40_000 // 1024) * 1024 - HBS)


def test_reuse_reports_leave_entries_of_a_producer_still_prefilling():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq(1, PREFIX), backlog, reqs)
    backlog[1 - first.rank] = 10**9  # keep the second on the producer's rank
    route, cost = _route(router, _seq(2, PREFIX + [7] * 2_000), backlog, reqs)
    assert route.rank == first.rank and cost < 42_000
    assert not router.note_reuse(2, 0)  # seq 1 is still prefilling it
    assert router._longest_hit(_seq(3, PREFIX).prompt_hash_chain)[2] == 1


def test_a_released_placement_forgets_its_reuse_report():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    _route(router, _seq(1, PREFIX), backlog, reqs)
    router.prefill_complete(1)
    _route(router, _seq(2, PREFIX), backlog, reqs)
    router.release(2)
    assert not router.note_reuse(2, 0)
    assert router._placed == {}


def test_the_manager_counts_stale_placements_from_rank_reports():
    mgr = _bare_mgr()
    mgr._store_router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    mgr._dp_route_counters = dict.fromkeys(ShareCacheRouter.DECISIONS, 0)
    backlog, reqs = [0, 0], [0, 0]
    _route(mgr._store_router, _seq(1, PREFIX), backlog, reqs)
    mgr._store_router.prefill_complete(1)
    _route(mgr._store_router, _seq(2, PREFIX), backlog, reqs)
    mgr._note_dp_reuse_locked([(2, 0), ("unknown", 0)])
    assert mgr._dp_route_counters["share_cache_stale_total"] == 1
