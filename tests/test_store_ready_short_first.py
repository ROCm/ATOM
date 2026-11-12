import inspect
import sys
from collections import deque
from copy import deepcopy
from types import SimpleNamespace

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Replay, SequenceStatus


class ConnectorView(SimpleNamespace, KVConnectorSchedulerBase):
    def admission_ready(self, seq):
        return True


class View(SimpleNamespace):
    @property
    def replay_until(self):
        return self.replay.until if self.replay is not None else 0


def fixture():
    ledger = []
    bm = BlockManager.__new__(BlockManager)
    bm.block_size = bm.hash_block_size = 128
    bm.state = SimpleNamespace(
        transfer=SimpleNamespace(replays=True), min_fork_tokens=0
    )
    bm.min_final_prefill_rows = bm.state_replay_tokens = 0
    bm.state_caches = [
        SimpleNamespace(applies=lambda seq: True, readable_midstep=False)
    ]
    bm.state_checkpoint_interval_tokens = 8192
    bm.chunks_cut_for_demand = bm.chunks_cut_for_end = 0
    bm.checkpoint_limit = lambda seq: 0
    bm._midstep_applies = lambda seq: True
    bm.enable_prefix_caching = False
    bm._event_log = None
    bm.publish_loaded_prefix = (
        lambda seq, **kwargs: ledger.append(("publish", seq.id)) or kwargs["end_token"]
    )
    s = Scheduler.__new__(Scheduler)
    s.block_manager = bm
    s.max_num_seqs = 64
    s.max_num_batched_tokens = 32768
    s._ready_short_admitted = s._ready_short_cold_stalls = 0
    s._tick_cost_limit = float("inf")
    s._tick_cost = 0
    s._balance_token_weight = 0
    s._balance_min_tokens = 4096
    s._tick_cold_partial_cap = 0
    s._checkpoint_holds = {}
    s.long_prefill_token_threshold = 0
    s.enable_chunked_prefill = True
    s._dynamic_chunk_limit = lambda *args: None
    s.replay_history_tokens = 3
    s.advance_on_schedule = False
    s.delay_factor = 0
    s.kv_connector = ConnectorView(is_offload=True, pending_load_partial_token_cap=8192)
    s._early_load = SimpleNamespace(
        entries={}, note_prefill=lambda sid: ledger.append(("prefill", sid))
    )
    s._record_cache_reuse = lambda seq: ledger.append(("reuse", seq.id))
    s._note_dp_reuse = lambda seq: None
    ready = View(
        id=2,
        status=SequenceStatus.WAITING_FOR_REMOTE_KVS,
        block_table=[1],
        num_tokens=90066,
        num_cached_tokens=0,
        offload_loaded_tokens=89856,
        offload_load_start_tokens=0,
        prefix_cache_hit_tokens=0,
        offload_loaded=False,
        offload_load_failed=False,
        offload_joint=SimpleNamespace(
            load_hash=-1, boundary_tokens=0, boundary_hash=-1
        ),
        has_per_req_cache=True,
        replays_hit=False,
        replay=None,
        token_ids=range(90066),
        checkpoint_demand_pos=0,
        checkpoint_end_pos=0,
        state_fork_src=-1,
        multimodal_data=None,
        is_partial_prefill=False,
        _counted_as_inflight_load=True,
    )
    cold = deepcopy(ready)
    cold.id = 1
    cold.status = SequenceStatus.RUNNING
    cold.num_tokens = 90000
    cold.num_cached_tokens = 0
    cold.has_per_req_cache = False
    cold.is_partial_prefill = True
    cold._counted_as_inflight_load = False
    cold.offload_loaded_tokens = 0
    s.running = deque([cold])
    s.waiting = deque([ready])
    s._partial_prefill_count = 1
    s._num_parked_remote_kv = 1
    s.finished_recving_kv_req_ids = [2]
    s.failed_recving_kv_req_ids = []
    return s, ready, cold, ledger


def stage(s):
    s._prioritize_ready_remote_kv_requests()
    selected, widths = {}, []
    counts = s._schedule_ready_short_prefills(selected, widths, 0, 0)
    return selected, widths, counts


def test_actual_terminal_resolves_and_admits_before_cold():
    s, ready, cold, ledger = fixture()
    s._tick_cost_limit = 64_000_000
    selected, widths, (n, used) = stage(s)
    assert list(selected) == [ready.id] and widths == [338]
    assert ready.offload_loaded and ready.status == SequenceStatus.RUNNING
    assert ready.num_cached_tokens == 89728 and ready.replay == Replay(
        89856, True, (89725, 89726, 89727), 89728
    )
    assert not s.waiting and list(s.running) == [cold, ready]
    assert s.finished_recving_kv_req_ids == [] and s._num_parked_remote_kv == 0
    assert ledger == [("publish", 2), ("prefill", 2), ("reuse", 2)]
    # The original multi-cold floor counterexample: ready really occupies the
    # batch first. Remaining cold sizing cannot manufacture 4096 floor rows.
    cold.num_tokens = 128
    other = deepcopy(cold)
    other.id = 3
    other.num_cached_tokens = 90000
    other.num_tokens = 100000
    for seq in (cold, other):
        result = s._schedule_running_partial(seq, selected, widths, n, used, 0)
        if result is not None:
            n, used, _ = result
    assert list(selected) == [2, 1, 3]
    assert widths == [338, 128, 240]
    assert s._tick_cost <= s._tick_cost_limit


def test_resolve_then_budget_reject_retains_allocated_resume_for_retry():
    s, ready, _, ledger = fixture()
    original = s._prefill_chunk_for_budget
    s._prefill_chunk_for_budget = lambda *args, **kwargs: None
    selected, widths, _ = stage(s)
    assert not selected and not widths
    assert list(s.waiting) == [ready] and ready.offload_loaded
    assert ready.block_table == [1] and s._num_parked_remote_kv == 1
    assert s.finished_recving_kv_req_ids == [] and ledger == [("publish", 2)]
    s._prefill_chunk_for_budget = original
    selected, widths, _ = stage(s)
    assert list(selected) == [2] and widths == [338]
    assert s._num_parked_remote_kv == 0
    assert list(s.running).count(ready) == 1
    assert ledger.count(("publish", 2)) == 1


def test_loading_failed_aborted_large_fork_and_full_slots_do_not_prioritize():
    for case in ("loading", "failed", "aborted", "large", "fork", "full"):
        s, ready, _, ledger = fixture()
        if case == "loading":
            s.finished_recving_kv_req_ids.clear()
        elif case == "failed":
            s.failed_recving_kv_req_ids.append(2)
            ready.offload_load_failed = True
        elif case == "aborted":
            ready.status = SequenceStatus.ABORTED
        elif case == "large":
            ready.num_tokens += 5000
        elif case == "fork":
            ready.state_fork_src = 1
        elif case == "full":
            s.max_num_seqs = len(s.running)
        selected, widths, _ = stage(s)
        assert not selected and not widths, case
        assert list(s.waiting) == [ready], case
        assert not any(event[0] == "prefill" for event in ledger), case


def test_actual_ready_floor_exception_stops_other_ready_and_cold():
    s, ready, cold, _ = fixture()
    s._tick_cost_limit = 10_000_000
    other = deepcopy(ready)
    other.id = 3
    s.waiting.append(other)
    s.finished_recving_kv_req_ids.append(3)
    s._num_parked_remote_kv += 1
    selected, widths, (n, used) = stage(s)
    assert list(selected) == [2] and widths == [338]
    assert s._tick_cost > s._tick_cost_limit  # Existing ready floor, not a new target.
    assert list(s.waiting) == [other] and s._ready_short_admitted == 1
    assert s._schedule_running_partial(cold, selected, widths, n, used, 0) is None


def test_already_running_restored_small_replay_precedes_new_ready():
    s, ready, _cold, _ = fixture()
    warm = deepcopy(ready)
    warm.id = 3
    warm.status = SequenceStatus.RUNNING
    warm.offload_loaded = True
    warm.is_partial_prefill = True
    warm.num_cached_tokens = 89728
    warm.replay = Replay(89856, True, (), 89728)
    s.running.append(warm)
    s._partial_prefill_count += 1
    selected, widths, _ = stage(s)
    assert list(selected) == [3, 2]
    assert widths == [338, 338]
    assert list(s.running).count(warm) == 1
    warm.status = SequenceStatus.ABORTED
    assert s._ready_short_demand(warm, running=True) == 0


def schedule_fixture(s):
    """Keep real scheduling/admission bodies; isolate unrelated model packing."""
    s._schedule_tick = 0
    s._rejected = []
    s.enable_dynamic_chunking = False
    s.prefill_delayer = None
    s.cost_balance = False
    s._arrived_requests = 0
    s.block_manager.complete_previous_state_batch = lambda: None
    s._promote_ready_remote_kv_requests = lambda: None
    s._early_load_tier_prefixes = lambda: 0
    s._early_load.entries = {}
    s._group_prefill_tick = lambda decision: True
    s._prefix_bypass_limit = lambda: 4
    s._unschedulable_reason = lambda seq: None
    s._lookup_detach_enabled = lambda seq: False
    s._offload_lookup_pending = lambda seq: False
    s._query_connector_prefill_match = lambda seq, skip: False
    s._local_prefill_coalescing = False
    s._probe_hbm_prefix = lambda seq: SimpleNamespace(num_cached_blocks=0)
    s._allocate_admitted_prefill = lambda *args: True
    s._finish_prefill_admission = lambda seq: None
    s._confirm_remote_load_after_alloc = lambda *args: False
    s._adjust_prefill_chunk_after_alloc = lambda seq, chunk: chunk
    s._commit_calibration_arm = lambda: None
    s._inflight_prefix_wait = {}
    s._waiting_prefills = lambda: iter(s.waiting)
    s._passed_delay = lambda now: True
    s.block_manager.can_allocate_memoized = lambda seq: 0
    return s


def capture_schedule(s, target=None):
    lines, first = inspect.getsourcelines(s._schedule)
    stop = first + next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "total_tokens_num_prefill = sum(num_scheduled_tokens)"
    )
    set_cost = first + next(
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("if self.cost_balance and delayer_allows and")
    )
    captured = {}

    class Done(Exception):
        pass

    def trace(frame, event, arg):
        if frame.f_code is s._schedule.__func__.__code__ and event == "line":
            if frame.f_lineno == set_cost and target is not None:
                s._tick_cost_limit = target
            if frame.f_lineno == stop:
                captured.update(
                    ids=list(frame.f_locals["scheduled_seqs"]),
                    widths=list(frame.f_locals["num_scheduled_tokens"]),
                    cost=s._tick_cost,
                    priority=getattr(s, "_ready_short_admitted", 0),
                    stalls=getattr(s, "_ready_short_cold_stalls", 0),
                )
                raise Done()
        return trace

    sys.settrace(trace)
    try:
        s._schedule()
    except Done:
        pass
    finally:
        sys.settrace(None)
    return captured


def test_complete_schedule_prioritizes_ready_in_multi_cold_counterexample():
    s, _ready, cold, _ = fixture()
    cold.num_tokens = 128
    other = deepcopy(cold)
    other.id = 3
    other.num_cached_tokens = 90000
    other.num_tokens = 100000
    s.running.append(other)
    s._partial_prefill_count += 1
    result = capture_schedule(schedule_fixture(s), 64_000_000)
    assert result["ids"] == [2, 1, 3]
    assert result["widths"] == [338, 128, 240]
    assert result["cost"] <= 64_000_000


def test_complete_schedule_decode_hold_and_fresh_cold_starvation_clock():
    s, ready, cold, _ = fixture()
    schedule_fixture(s)
    s.prefill_delayer = SimpleNamespace(
        protects_decode=lambda n: True,
        should_allow_prefill=lambda **kwargs: SimpleNamespace(
            allow=False, prefillable_ranks=0
        ),
        max_queue_ms=None,
    )
    s._ready_short_cold_stalls = 2
    held = capture_schedule(s)
    assert held["ids"] == [] and held["stalls"] == 2
    # Completion reconciliation happens on HOLD too, but no forward or fairness tick.
    assert ready.offload_loaded and s.finished_recving_kv_req_ids == []
    s, template, cold, _ = fixture()
    schedule_fixture(s)
    s.max_num_batched_tokens = 512
    s.running.clear()
    s._partial_prefill_count = 0
    cold.status = SequenceStatus.WAITING
    cold.replay_kv_load = False
    s.waiting = deque([cold])
    s.finished_recving_kv_req_ids.clear()
    s._num_parked_remote_kv = 0
    history = []
    for tick in range(4):
        ready = deepcopy(template)
        ready.id = 10 + tick
        ready.num_tokens = 90240  # 512 actual rows including replay.
        ready.token_ids = range(90240)
        s.waiting.append(ready)
        s.finished_recving_kv_req_ids.append(ready.id)
        s._num_parked_remote_kv += 1
        result = capture_schedule(s)
        history.append(result)
        # Model stand-in: successful ready finishes; fresh cold remains waiting
        # until the fourth allowed tick. No source scheduling method is changed.
        s.running = deque(q for q in s.running if not q.offload_loaded)
    assert [r["ids"][0] for r in history] == [10, 11, 12, 1]
    assert [r["stalls"] for r in history] == [1, 2, 3, 0]


def test_forced_cold_tick_also_disables_running_ready_priority():
    s, ready, cold, _ = fixture()
    schedule_fixture(s)
    s.max_num_batched_tokens = 512
    s._ready_short_cold_stalls = 3
    warm = deepcopy(ready)
    warm.id = 3
    warm.status = SequenceStatus.RUNNING
    warm.is_partial_prefill = warm.offload_loaded = True
    warm.num_cached_tokens = 89728
    warm.replay = Replay(89856, True, (), 89728)
    s.running.appendleft(warm)
    s._partial_prefill_count += 1
    result = capture_schedule(s)
    assert result["ids"] == [cold.id]
    assert result["priority"] == 0 and result["stalls"] == 0


def test_forced_tick_defers_running_ready_for_fresh_cold_but_not_blocked_head():
    for blocked in (False, True):
        s, ready, cold, _ = fixture()
        schedule_fixture(s)
        s.max_num_batched_tokens = 512
        s._ready_short_cold_stalls = 3
        warm = deepcopy(ready)
        warm.id = 3
        warm.status = SequenceStatus.RUNNING
        warm.is_partial_prefill = warm.offload_loaded = True
        warm.num_tokens = 90240
        warm.token_ids = range(90240)
        warm.num_cached_tokens = 89728
        warm.replay = Replay(89856, True, (), 89728)
        cold.status = SequenceStatus.WAITING
        cold.replay_kv_load = False
        s.running = deque([warm])
        s.waiting = deque([cold, ready])
        s.block_manager.can_allocate_memoized = lambda seq, blocked=blocked: (
            -1 if blocked else 0
        )
        result = capture_schedule(s)
        assert result["ids"] == ([warm.id] if blocked else [cold.id])
        assert list(s.running).count(warm) == 1


def test_active_cold_fork_retains_original_path_instead_of_overfilling_new_budget():
    s, ready, cold, _ = fixture()
    cold.state_fork_src = 99
    s.block_manager.state.min_fork_tokens = 4096
    calls = []
    s.block_manager.cancel_state_fork = lambda seq: calls.append(seq.id) or False
    # Real original finalizer illustrates why preflight sizing alone is unsafe.
    assert s._finalize_prefill_chunk(cold, 0, 128) == 4096
    calls.clear()
    selected, widths, _ = stage(s)
    assert not selected and not widths and not calls
    # Completion reconciliation happens on HOLD too, but no forward or fairness tick.
    assert ready.offload_loaded and s.finished_recving_kv_req_ids == []
    s._checkpoint_holds[cold.id] = 128
    selected, widths, _ = stage(s)
    assert list(selected) == [2] and widths == [338]
