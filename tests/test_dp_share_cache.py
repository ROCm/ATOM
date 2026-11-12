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
        prompt_hash_chain=SimpleNamespace(
            hash_block_size=HBS, hashes=hashes, cache_seed=None
        ),
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
    # The shared radix index retains every complete hash block.
    assert cost == 42_000 - 40_000


def test_prefix_still_prefilling_is_credited_only_on_its_rank():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq(1, PREFIX), backlog, reqs)
    other = 1 - first.rank
    seq = _seq(2, PREFIX + list(range(10**6, 10**6 + 2_000)))
    hit = 40_000
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
    mgr._backlog_unseen_ids = [deque() for _ in range(n)]
    mgr._backlog_objects = {}
    mgr._backlog_prompt_tokens = {}
    mgr._backlog_sent_ordinals = {}
    mgr._backlog_credit_updates = {}
    mgr._backlog_credit_sent = {}
    mgr._backlog_credit_debts = {}
    mgr._backlog_reported_credit_tokens = [0] * n
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
    first, second = (
        SimpleNamespace(id="a", num_prompt_tokens=30000),
        SimpleNamespace(id="b", num_prompt_tokens=30000),
    )
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
    mgr._backlog_note_charged_locked(
        SimpleNamespace(id="a", num_prompt_tokens=30000), 0
    )
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
    assert not router.index and not router.pending
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
    _route_result, cost = _route(router, _seq(2, PREFIX + [7] * 2_000), backlog, reqs)
    assert cost < 42_000  # placed on the prefix the index lists
    # Its rank found only 8192 tokens.
    assert router.note_reuse(2, 8192)
    path = router._path(_seq(3, PREFIX).prompt_hash_chain)
    assert router._tier_hit(path) <= 8192 + HBS
    assert 2 in router._produced  # seq 2 computes its missing suffix
    assert router.prefilling(2)
    # Once seq 2 completes, the whole prefix counts as loadable again.
    router.prefill_complete(2)
    _, cost3 = _route(router, _seq(3, PREFIX), backlog, reqs)
    assert cost3 == 0
    # A report at the estimate (one hash block of replay below it) drops nothing.
    assert not router.note_reuse(3, 40_000 - HBS)


def test_reuse_reports_leave_entries_of_a_producer_still_prefilling():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq(1, PREFIX), backlog, reqs)
    backlog[1 - first.rank] = 10**9  # keep the second on the producer's rank
    _route_result, cost = _route(router, _seq(2, PREFIX + [7] * 2_000), backlog, reqs)
    assert _route_result.rank == first.rank and cost < 42_000
    assert not router.note_reuse(2, 0)  # seq 1 is still prefilling it
    assert 1 in router.pending.values()


def test_a_released_placement_forgets_its_reuse_report():
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    _route(router, _seq(1, PREFIX), backlog, reqs)
    router.prefill_complete(1)
    _route(router, _seq(2, PREFIX), backlog, reqs)
    router.release(2)
    assert not router.note_reuse(2, 0)
    assert router._reuse_pending == {}


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


def test_share_cache_refuses_a_store_split_into_per_nic_pools():
    """Each worker uses its own NIC's pool: with two pools a prefix one DP rank
    saved is a hit another rank's load misses, so placing a request on any rank
    would fail its load and recompute."""
    import pytest

    one = {
        "mooncake_store.master": "127.0.0.1:1",
        "mooncake_store.metadata": "http://127.0.0.1:2/metadata",
    }
    two = {
        "mooncake_store.pools": {
            f"nic{i}": {
                "master": f"127.0.0.{i + 1}:1",
                "metadata": f"http://127.0.0.{i + 1}:2/metadata",
            }
            for i in range(2)
        }
    }

    def build(extra, ranks):
        mgr = _bare_mgr(ranks)
        mgr.global_engine_count = ranks
        mgr._dp_session_affinity_enabled = False
        mgr._dp_lb_req_equiv = 512
        config = SimpleNamespace(
            enable_prefix_caching=True,
            kv_transfer_config={
                "kv_connector": "mooncake_store",
                "kv_connector_extra_config": extra,
            },
        )
        return mgr._build_store_router(config)

    assert isinstance(build(one, 2), ShareCacheRouter)
    assert isinstance(build(two, 1), ShareCacheRouter)  # one rank: nothing to share
    with pytest.raises(
        ValueError, match="Shared-prefix DP routing requires one shared"
    ):
        build(two, 2)


def test_a_same_rank_reuser_carries_the_prefix_if_its_producer_aborts():
    """B reuses, on A's rank, a prefix A is still prefilling. A aborts and B
    computes the prefix: it must stay in the index."""
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq("A", PREFIX), backlog, reqs)
    backlog[1 - first.rank] = 10_000_000  # B stays on A's rank
    second, cost = _route(router, _seq("B", PREFIX + [7] * 2_000), backlog, reqs)
    assert second.rank == first.rank and cost < 42_000  # reuses A's prefix
    router.release("A")
    router.prefill_complete("B")
    router.select(_seq("C", PREFIX + [9] * 2_000), [0, 0], [0, 0])
    assert router.charge("C") > 0


def test_every_backup_of_a_prefix_is_kept():
    """A owns a prefix; B, then C, compute it beside A. B completes, then A and
    C abort: B computed it, so it stays credited."""
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq("A", PREFIX), backlog, reqs)
    backlog[first.rank] = 10_000_000
    _route(router, _seq("B", PREFIX), backlog, reqs)
    _route(router, _seq("C", PREFIX), backlog, reqs)
    router.prefill_complete("B")
    router.release("A")
    router.release("C")
    router.select(_seq("D", PREFIX + [9] * 2_000), [0, 0], [0, 0])
    assert router.charge("D") > 0
    # And with A and B gone unprefilled, C (completing) still carries it.
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    first, _ = _route(router, _seq("A", PREFIX), backlog, reqs)
    backlog[first.rank] = 10_000_000
    _route(router, _seq("B", PREFIX), backlog, reqs)
    _route(router, _seq("C", PREFIX), backlog, reqs)
    router.release("A")
    router.release("B")
    router.prefill_complete("C")
    router.select(_seq("D", PREFIX + [9] * 2_000), [0, 0], [0, 0])
    assert router.charge("D") > 0
    assert set(router.seq_hashes) == {"D"}


def test_a_readmitted_partial_prefill_counts_in_the_backlog():
    from atom.model_engine.prefill_backlog import prefill_backlog
    from atom.model_engine.sequence import SequenceStatus

    readmitted = SimpleNamespace(
        status=SequenceStatus.RUNNING,
        tokens=700,
        num_completion_tokens=12,  # kept from before its preemption
        is_partial_prefill=True,
    )
    decoding = SimpleNamespace(
        status=SequenceStatus.RUNNING,
        tokens=500,
        num_completion_tokens=3,
        is_partial_prefill=False,
    )
    assert prefill_backlog(_scheduler_with([], [readmitted, decoding])) == 700


def test_reuse_reports_are_queued_ahead_of_the_step_outputs():
    """The engine flushes them first in `_postprocess`; a request's report then
    reaches the router before its first output completes its placement."""
    import queue

    from atom.model_engine.prefill_backlog import flush_dp_reuse

    out = queue.Queue()
    sched = SimpleNamespace(dp_reuse_reports=[(1, 4096)])
    flush_dp_reuse(sched, out)
    out.put_nowait(("STREAM", [(1, "token")]))
    flush_dp_reuse(sched, out)  # nothing new: nothing queued
    assert out.get_nowait() == ("DP_REUSE", [(1, 4096)])
    assert out.get_nowait()[0] == "STREAM" and out.empty()
    assert sched.dp_reuse_reports == []


def test_a_reuse_report_before_the_first_output_reassigns_the_prefix():
    """Report first (the queue order), then the first output: the request that
    recomputed the prefix owns it and, once complete, it is credited again."""
    router = ShareCacheRouter(min_load_tokens=512, req_equiv=512)
    backlog, reqs = [0, 0], [0, 0]
    _route(router, _seq(1, PREFIX), backlog, reqs)
    router.prefill_complete(1)
    _route(router, _seq(2, PREFIX), backlog, reqs)
    assert router.note_reuse(2, 0)
    router.prefill_complete(2)
    router.select(_seq(3, PREFIX), [0, 0], [0, 0])
    assert router.charge(3) > 0
