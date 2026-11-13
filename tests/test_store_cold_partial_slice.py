from types import SimpleNamespace

from conftest import MockConfig

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus


class ConnectorView(SimpleNamespace, KVConnectorSchedulerBase):
    def admission_ready(self, seq):
        return True


def fixture(pending=True):
    s = Scheduler(
        MockConfig(
            max_num_seqs=8,
            max_num_batched_tokens=32768,
            num_kvcache_blocks=2048,
            kv_cache_block_size=256,
            max_model_len=262144,
        )
    )
    s.kv_connector = ConnectorView(
        is_offload=True,
        pending_load_partial_token_cap=8192,
        build_connector_meta=lambda: None,
        get_num_new_matched_tokens=lambda q: (0, False),
        update_state_after_alloc=lambda q: None,
        cancel_pending_load=lambda q: None,
    )
    cold = Sequence(list(range(65536)), block_size=256)
    s.block_manager.allocate(cold)
    cold.num_cached_tokens = 256
    cold.status = SequenceStatus.RUNNING
    cold.is_partial_prefill = True
    s.running.append(cold)
    s._partial_prefill_count = 1
    load = Sequence(list(range(100000, 228000)), block_size=256)
    s.block_manager.allocate(load)
    if pending:
        load.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        s._count_inflight_load(load)
        s.waiting.append(load)
    return s, cold, load


def sizes(s):
    b, q = s.schedule()
    return dict(zip(q, map(int, b.num_scheduled_tokens)))


def test_pending_bounds_cold_and_preserves_global_budget():
    s, cold, load = fixture()
    assert sizes(s) == {cold.id: 8192}
    assert s.max_num_batched_tokens == 32768
    assert load in s.waiting and len(load.block_table) > 0


def test_no_pending_unchanged():
    s, cold, _load = fixture(False)
    assert sizes(s) == {cold.id: 32768}


def test_cap_restores_when_pending_finishes():
    s, _cold, load = fixture()
    assert s._pending_store_partial_cap() == 8192
    s.finished_recving_kv_req_ids.append(load.id)
    assert s._pending_store_partial_cap() == 0


def test_failed_completion_not_pending():
    s, _cold, load = fixture()
    s.failed_recving_kv_req_ids.append(load.id)
    assert s._pending_store_partial_cap() == 0


def test_non_store_backend_unchanged():
    s, cold, _load = fixture()
    s.kv_connector.pending_load_partial_token_cap = 0
    assert sizes(s) == {cold.id: 32768}


def test_unowned_or_uncounted_not_pending():
    s, _cold, load = fixture()
    s._uncount_inflight_load(load)
    assert s._pending_store_partial_cap() == 0
    s._count_inflight_load(load)
    s.block_manager.deallocate(load)
    assert s._pending_store_partial_cap() == 0


def test_multiple_cold_partials_share_cap():
    s, _cold, _load = fixture()
    other = Sequence(list(range(300000, 365536)), block_size=256)
    s.block_manager.allocate(other)
    other.num_cached_tokens = 256
    other.is_partial_prefill = True
    other.status = SequenceStatus.RUNNING
    s.running.append(other)
    s._partial_prefill_count += 1
    assert sum(sizes(s).values()) == 8192


def test_replay_is_not_clipped():
    s, cold, _load = fixture()
    s._tick_cold_partial_cap = 8192
    assert not s._slice_cold_partial(cold, 0, 256)
    cold.offload_loaded = True
    assert not s._slice_cold_partial(cold, 256, 256)


def test_final_short_chunk_remains_exact():
    s, cold, _load = fixture()
    cold.num_cached_tokens = cold.num_tokens - 256
    assert sizes(s) == {cold.id: 256}


def test_cost_estimate_observes_cold_cap():
    s, _cold, _load = fixture()
    s._tick_cold_partial_cap = s._pending_store_partial_cap()
    assert s._estimate_prefill_chunk_cost() == s._chunk_cost(256, 8192)


def test_delayer_hold_still_blocks(monkeypatch):
    from atom.model_engine.prefill_delayer import PrefillDecision, PrefillDelayer

    s, _cold, _load = fixture()
    s.set_prefill_delayer(PrefillDelayer(1, None, 32768))
    monkeypatch.setattr(
        PrefillDelayer,
        "should_allow_prefill",
        lambda *a, **kw: PrefillDecision(allow=False, prefillable_ranks=0),
    )
    r = s.schedule()
    assert r is None or not r[1]


def test_cold_progresses_until_complete_with_load_pending():
    s, cold, _load = fixture()
    total = 0
    while cold.num_cached_tokens < cold.num_tokens:
        n = sizes(s)[cold.id]
        assert 0 < n <= 8192
        cold.num_cached_tokens += n
        total += n
    assert total == 65536 - 256


def test_multiconnector_only_prefill_exposes_store_cap():
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    c = object.__new__(MultiConnectorScheduler)
    c._connectors = [
        SimpleNamespace(is_offload=True, pending_load_partial_token_cap=8192)
    ]
    c.is_producer = True
    assert c.pending_load_partial_token_cap == 8192
    c.is_producer = False
    assert c.pending_load_partial_token_cap == 0


def test_cost_budget_can_reduce_slice_further(monkeypatch):
    from atom.model_engine.prefill_delayer import PrefillDecision, PrefillDelayer

    s, cold, _load = fixture()
    s.set_prefill_delayer(PrefillDelayer(1, None, 32768))
    s.cost_balance = True
    s._balance_min_tokens = 0
    monkeypatch.setattr(
        PrefillDelayer,
        "should_allow_prefill",
        lambda *a, **kw: PrefillDecision(allow=True, prefillable_ranks=1),
    )

    def balance(self, own):
        self._tick_cost_limit = self._chunk_cost(256, 4096)

    monkeypatch.setattr(Scheduler, "_balance_prefill_step", balance)
    counts = sizes(s)
    assert 0 < counts[cold.id] <= 4096
    assert s._tick_cost <= s._tick_cost_limit
