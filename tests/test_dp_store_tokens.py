# SPDX-License-Identifier: MIT
# Unit tests for `--dp-load-balance store_tokens` (atom/model_engine/store_aware.py)
# and its CoreManager bookkeeping (rank-reported prefill backlog).

from collections import deque
from threading import Lock
from types import SimpleNamespace

from atom.model_engine.engine_core_mgr import DP_LB_STRATEGIES, CoreManager
from atom.model_engine.store_aware import StoreAwareRouter

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
    assert "store_tokens" in DP_LB_STRATEGIES


def test_cold_prompt_goes_to_least_backlog():
    router = StoreAwareRouter(min_load_tokens=512, req_equiv=512)
    route, cost = _route(router, _seq(1, PREFIX), [9_000, 0], [0, 0])
    assert route.rank == 1 and route.decision == "store_cold_total"
    assert cost == len(PREFIX)


def test_prefix_in_tier_is_credited_on_any_rank():
    router = StoreAwareRouter(min_load_tokens=512, req_equiv=512)
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
    router = StoreAwareRouter(min_load_tokens=512, req_equiv=512)
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
    router = StoreAwareRouter(min_load_tokens=512, req_equiv=512)
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
    mgr._bl_report = [None] * n
    mgr._bl_unseen = [deque() for _ in range(n)]
    mgr._bl_unseen_tokens = [0] * n
    mgr._bl_sent = [0] * n
    mgr._bl_popped = [0] * n
    return mgr


def test_backlog_is_report_plus_unreported_charges():
    mgr = _bare_mgr()
    mgr._rank_tokens = [7, 9]
    assert mgr._bl_effective_tokens_locked() == [7, 9]  # no report yet
    mgr._seq_load = {"a": (0, 1, 1_000), "b": (0, 1, 2_000), "c": (0, 1, 4_000)}
    mgr._bl_note_sent_locked(0, [SimpleNamespace(id="a"), SimpleNamespace(id="b")])
    mgr._bl_report[0] = (1, 300)  # rank 0 ingested "a" and has 300 left
    mgr._bl_pop_seen_locked(0)
    assert mgr._bl_effective_tokens_locked()[0] == 300 + 2_000
    mgr._bl_note_sent_locked(0, [SimpleNamespace(id="c")])
    mgr._bl_report[0] = (3, 50)
    mgr._bl_pop_seen_locked(0)
    assert mgr._bl_effective_tokens_locked()[0] == 50


def test_report_ahead_of_note_sent_does_not_double_count():
    mgr = _bare_mgr()
    mgr._seq_load = {"a": (1, 1, 5_000)}
    mgr._bl_report[1] = (1, 5_000)  # the rank ingested "a" before we noted it
    mgr._bl_note_sent_locked(1, [SimpleNamespace(id="a")])
    assert mgr._bl_effective_tokens_locked()[1] == 5_000


def _delayer_for(monkeypatch, lb, env=None):
    from atom.model_engine.engine_core import EngineCore

    monkeypatch.delenv("ATOM_PREFILL_DELAYER_STALL_TICKS", raising=False)
    if env is not None:
        monkeypatch.setenv("ATOM_PREFILL_DELAYER_STALL_TICKS", str(env))
    captured = {}
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(
        cost_balance=False, set_prefill_delayer=lambda d: captured.setdefault("d", d)
    )
    config = SimpleNamespace(
        enable_rapidserve=False,
        dp_load_balance=lb,
        max_num_batched_tokens=16384,
        prefill_admit_policy="interval",
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )
    engine._init_prefill_delayer(config, cpu_group=object())
    return captured["d"]


def test_store_tokens_shortens_the_delayer_stall_bound(monkeypatch):
    assert _delayer_for(monkeypatch, "least_requests").stall_ticks == 10
    assert _delayer_for(monkeypatch, "store_tokens").stall_ticks == 4


def test_explicit_stall_ticks_env_wins_under_store_tokens(monkeypatch):
    assert _delayer_for(monkeypatch, "store_tokens", env=7).stall_ticks == 7
