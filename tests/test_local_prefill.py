# SPDX-License-Identifier: MIT
"""Local coalescing signals, checkpoint waits and cancellation lifetimes."""

import ast
import gc
import queue
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import MockConfig, atom_config_double
from test_state_checkpoint import DEFAULT_STATE_RUNTIME

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import DecodeScheduler, ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.utils import envs


def scheduler(hybrid=False, **kwargs):
    config = {
        "max_num_seqs": 64,
        "num_kvcache_blocks": 4096,
        "kv_cache_block_size": 4,
        "max_model_len": 65536,
        "max_num_batched_tokens": 64,
        "enable_prefix_caching": True,
        "pool_entries": {"state": 64} if hybrid else {},
        "state_checkpoint_interval_tokens": 32,
    }
    config.update(kwargs)
    sched = Scheduler(
        MockConfig(**config),
        **({"state_runtime": DEFAULT_STATE_RUNTIME} if hybrid else {}),
    )
    sched.set_prefill_delayer(
        PrefillDelayer(1, None, sched.max_num_batched_tokens, prefill_decode_interval=4)
    )
    return sched


def sequence(n=128, hybrid=False, block_size=4):
    seq = Sequence(list(range(n)), block_size, has_per_req_cache=hybrid)
    seq.arrive_time = time.time()
    return seq


def producer_waiter():
    sched = scheduler(hybrid=True, long_prefill_token_threshold=16)
    producer = sequence(hybrid=True)
    bm = sched.block_manager
    assert bm.allocate(producer, bm.can_allocate(producer))
    producer.status = SequenceStatus.RUNNING
    producer.num_cached_tokens = 16
    producer.is_partial_prefill = True
    sched.running.append(producer)
    sched._partial_prefill_count = 1
    return sched, producer, sequence(hybrid=True)


def publish(sched, seq):
    bm = sched.block_manager
    assert bm.allocate(seq, bm.can_allocate(seq))
    bm.hash_blocks(seq, seq.num_tokens)
    bm.deallocate(seq)


def test_high_hit_probe_cap_does_not_invent_fill():
    sched = scheduler(max_num_batched_tokens=64)
    publish(sched, sequence())
    sched.waiting.extend(sequence() for _ in range(8))
    assert sched._local_prefill_pending_work() == (True, 4)
    d = sched.prefill_delayer
    d._first = False
    d.stall_ticks = 2
    for _ in range(3):
        ready, pending = sched._local_prefill_pending_work()
        released = d.should_allow_prefill(ready, pending, running_decode_batch=1)
    assert released
    assert d._stat_fire_fill == 0
    assert d._stat_fire_stall == 1


def test_short_burst_can_fill_past_four_requests():
    sched = scheduler()
    sched.waiting.extend(sequence(8) for _ in range(8))
    assert sched._local_prefill_pending_work() == (True, 64)
    batch, _ = sched.schedule()
    assert batch.total_tokens_num == 64


@pytest.mark.parametrize("threshold,expected", [(0, 64), (16, 16)])
def test_new_chunk_estimate_matches_admission(threshold, expected):
    sched = scheduler(long_prefill_token_threshold=threshold)
    sched.waiting.append(sequence())
    assert sched._local_prefill_pending_work() == (True, expected)
    batch, _ = sched.schedule()
    assert batch.total_tokens_num == expected


def test_nonchunked_refusal_stops_estimate_and_admission():
    sched = scheduler(enable_chunked_prefill=False)
    first = sequence(40)
    preempted = sequence(32)
    for _ in range(8):
        preempted.append_token(500)
    tail = sequence(24)
    sched.waiting.extend([first, preempted, tail])
    assert sched._local_prefill_pending_work() == (True, 40)
    batch, admitted = sched.schedule()
    assert batch.total_tokens_num == 40
    assert list(admitted) == [first.id]
    assert list(sched.waiting) == [preempted, tail]


@pytest.mark.parametrize("cached,expected", [(900, 100), (1000, 232)])
def test_offload_resume_estimate_matches_admission(cached, expected, monkeypatch):
    sched = scheduler(kv_cache_block_size=256, max_num_batched_tokens=1024)
    seq = sequence(1000, block_size=256)
    assert sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = cached
    sched.waiting.append(seq)
    monkeypatch.setattr(sched, "_is_offload_prefill_resume", lambda value: value is seq)
    assert sched._local_prefill_pending_work() == (True, expected)
    batch, _ = sched.schedule()
    assert batch.total_tokens_num == expected
    assert seq.prefix_cache_hit_tokens == cached


def test_parked_remote_slots_do_not_report_fill(monkeypatch):
    sched = scheduler(max_num_seqs=1)
    sched._num_parked_remote_kv = 1
    sched.waiting.append(sequence(8))
    monkeypatch.setattr(sched, "_query_connector_prefill_match", lambda *a, **kw: True)
    assert sched._local_prefill_pending_work() == (True, 0)
    _, admitted = sched.schedule()
    assert not admitted


def test_partial_preview_preserves_checkpoint_cut_counter():
    sched, producer, _ = producer_waiter()
    sched.long_prefill_token_threshold = 64
    producer.checkpoint_demand_pos = 36
    assert sched._local_prefill_pending_work() == (True, 20)
    assert sched.block_manager.chunks_cut_for_demand == 0
    batch, _ = sched.schedule()
    assert batch.total_tokens_num == 20
    assert sched.block_manager.chunks_cut_for_demand == 1


def test_each_waiter_keeps_its_deadline_after_head_changes():
    sched, _, first = producer_waiter()
    second = sequence(hybrid=True)
    sched.prefill_delayer.ttft_max_ticks = 3
    assert sched._wait_for_inflight_prefix(first, 0)
    sched._schedule_tick += 1
    assert sched._wait_for_inflight_prefix(second, 0)
    deadline = sched._inflight_prefix_wait[first.id]
    sched._schedule_tick = deadline
    assert not sched._wait_for_inflight_prefix(first, 0)
    assert sched._wait_for_inflight_prefix(second, 0)
    assert sched._inflight_prefix_wait[first.id] == deadline


def test_waiting_head_allows_independent_request():
    sched, _, waiter = producer_waiter()
    independent = sequence(8, hybrid=True)
    independent.token_ids[0] = 90000
    sched.waiting.extend([waiter, independent])
    _, admitted = sched.schedule()
    assert independent.id in admitted
    assert waiter.id not in admitted
    assert not waiter.block_table
    assert list(sched.waiting) == [waiter]


def test_bypass_scan_is_bounded_and_preserves_waiters(monkeypatch):
    sched, _, waiter = producer_waiter()
    sched.waiting.extend([waiter, *[sequence(hybrid=True) for _ in range(31)]])
    original = list(sched.waiting)
    joint_skips = dict(sched.block_manager.joint_skips)
    probe = Mock(wraps=sched.block_manager.can_allocate)
    monkeypatch.setattr(sched.block_manager, "can_allocate", probe)
    sched.schedule()
    assert list(sched.waiting) == original
    assert probe.call_count == 17
    assert len(sched._inflight_prefix_wait) == 17
    assert sched.block_manager.joint_skips == joint_skips


def test_failed_allocation_keeps_expired_deadline(monkeypatch):
    sched, _, waiter = producer_waiter()
    assert sched._wait_for_inflight_prefix(waiter, 0)
    deadline = sched._inflight_prefix_wait[waiter.id]
    sched._schedule_tick = deadline
    sched.waiting.append(waiter)
    monkeypatch.setattr(sched.block_manager, "allocate", lambda *args: False)
    sched.schedule()
    assert sched._inflight_prefix_wait[waiter.id] == deadline
    assert not sched._wait_for_inflight_prefix(waiter, 0)


def test_waiter_reuses_completed_prefix():
    sched = scheduler(
        hybrid=True, max_num_batched_tokens=16, state_checkpoint_interval_tokens=64
    )
    producer = sequence(60, hybrid=True)
    waiter = sequence(76, hybrid=True)
    sched.extend([producer, waiter])
    was_deferred = False
    for _ in range(40):
        batch, admitted = sched.schedule()
        if waiter.id in admitted:
            assert was_deferred
            assert waiter.prefix_cache_hit_tokens >= 48
            assert waiter.id not in sched._inflight_prefix_wait
            return
        was_deferred |= waiter.id in sched._inflight_prefix_wait
        assert not waiter.block_table
        sched.postprocess(
            list(admitted.values()),
            ScheduledBatchOutput(
                req_ids=batch.req_ids,
                token_ids=[(501,)] * len(batch.req_ids),
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
        if batch.total_seqs_num_prefill:
            sched.prefill_delayer.notify_prefill_executed()
    pytest.fail("waiter never admitted")


def test_consumer_must_have_room_to_restore_fork():
    sched = scheduler(hybrid=True, max_num_batched_tokens=32)
    sched.block_manager = type(sched.block_manager)(
        MockConfig(
            kv_cache_block_size=64,
            num_kvcache_blocks=100,
            pool_entries={"state": 8},
            enable_prefix_caching=True,
            state_checkpoint_interval_tokens=128,
        ),
        state_runtime=StateRuntime(transfer=StateTransfer.fork(131)),
    )
    producer = sequence(1024, hybrid=True, block_size=64)
    producer.status = SequenceStatus.RUNNING
    producer.is_partial_prefill = True
    producer.checkpoint_end_pos = 832
    sched.running.append(producer)
    short = sequence(896, hybrid=True, block_size=64)
    long = sequence(1024, hybrid=True, block_size=64)
    assert not sched._wait_for_inflight_prefix(short, 0)
    assert sched._wait_for_inflight_prefix(long, 0)
    pool = sched.block_manager.state
    pool._index(13, 0)
    assert pool.resumable_hit(short, 13, list(range(1, 14))) == 0
    assert pool.resumable_hit(long, 13, list(range(1, 14))) == 13


@pytest.mark.parametrize("dcp", [1, 8])
def test_prompt_hash_reuse_rechecks_pool_and_seed(dcp, monkeypatch):
    sched = scheduler(decode_context_parallel_size=dcp)
    bm = sched.block_manager
    if dcp > 1:
        # Hash/pool logic is real; only the GPU module's shard sizing is isolated.
        monkeypatch.setattr(
            bm, "num_pool_blocks", lambda n: (n + 4 * dcp - 1) // (4 * dcp)
        )
    publish(sched, sequence())
    seq = sequence()
    hashes = Mock(wraps=bm.compute_hash)
    monkeypatch.setattr(bm, "compute_hash", hashes)
    expected = 128 // (4 * dcp) - 1
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == expected
    first_calls = hashes.call_count
    assert first_calls > 0
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == expected
    assert hashes.call_count == first_calls
    original_seed = seq.cache_seed
    seq.cache_seed = 123
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == 0
    seq.cache_seed = original_seed
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == expected
    monkeypatch.setattr(bm.kv, "lookup", lambda h: -1)
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == 0


def test_generated_suffix_is_not_memoized_and_weak_cache_releases_request():
    sched = scheduler()
    seq = sequence(10)
    for token in range(10, 24):
        seq.append_token(token)
    publish(sched, seq)
    bm = sched.block_manager
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == 5
    assert len(bm._prefill_probe_hashes[seq][1]) == 2
    seq.token_ids[12] = 10000
    assert bm.can_allocate(seq, record=False, reuse_hashes=True) == 3
    ref = weakref.ref(seq)
    del seq
    gc.collect()
    assert ref() is None
    assert not bm._prefill_probe_hashes


def test_deallocation_clears_old_joint_span():
    sched = scheduler()
    seq = sequence(8)
    bm = sched.block_manager
    assert bm.allocate(seq, 0)
    seq.offload_joint.boundary_tokens = 4
    seq.offload_joint.boundary_hash = 12
    bm.deallocate(seq)
    assert not seq.block_table
    assert seq.offload_joint.boundary_tokens == 0
    assert seq.offload_joint.boundary_hash == -1


def test_nonhead_abort_reclaims_slot_before_admission():
    sched = scheduler(hybrid=True, pool_entries={"state": 1})
    head = sequence(8, hybrid=True)
    aborted = sequence(8, hybrid=True)
    assert sched.block_manager.allocate(aborted, 0)
    sched.waiting.extend([head, aborted])
    assert sched.block_manager.can_allocate(head, record=False) == -1
    handler = EngineUtilityHandler(None, queue.Queue(), scheduler=sched)
    handler._handle_abort_request({"req_id": aborted.id})
    assert not aborted.block_table and not aborted.state_slots
    assert aborted in sched.take_rejected()
    _, admitted = sched.schedule()
    assert head.id in admitted


@pytest.mark.parametrize(
    "terminal",
    ["finished_recving", "failed_recving", "finished_loading", "failed_loading"],
)
def test_nonhead_abort_retains_inflight_resources_until_terminal(terminal):
    sched = scheduler(hybrid=True, pool_entries={"state": 1})
    head, aborted = sequence(8, hybrid=True), sequence(8, hybrid=True)
    assert sched.block_manager.allocate(aborted, 0)
    sched.waiting.extend([head, aborted])
    sched._count_inflight_load(aborted)
    aborted.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.kv_connector = SimpleNamespace(
        is_producer=False, is_offload="loading" in terminal
    )
    assert sched.abort_request(aborted.id)
    assert aborted.block_table and aborted.state_slots
    assert sched._num_parked_remote_kv == 1
    sched._update_from_kv_xfer_finished(KVConnectorOutput(**{terminal: {aborted.id}}))
    assert not aborted.block_table and not aborted.state_slots
    assert sched._num_parked_remote_kv == 0
    assert not sched.deferred_free_blocks


def test_running_abort_keeps_forward_resources():
    sched = scheduler()
    seq = sequence(8)
    sched.add(seq)
    sched.schedule()
    assert sched.abort_request(seq.id)
    assert seq.status == SequenceStatus.ABORTED
    assert seq.block_table
    assert seq in sched.running
    assert not sched.abort_request(-1)


def test_protection_skips_probes_then_resumes_partial(monkeypatch):
    sched, producer, _ = producer_waiter()
    decode = sequence(8, hybrid=True)
    assert sched.block_manager.allocate(decode, 0)
    decode.num_cached_tokens = decode.num_prompt_tokens
    decode.append_token(500)
    decode.status = SequenceStatus.RUNNING
    sched.running.appendleft(decode)
    d = sched.prefill_delayer
    d._first = False
    d.partial_max_ticks = 0
    d.notify_prefill_executed()
    with monkeypatch.context() as patch:
        patch.setattr(
            sched, "_local_prefill_pending_work", Mock(side_effect=AssertionError)
        )
        for _ in range(4):
            batch, admitted = sched.schedule()
            assert producer.id not in admitted
            assert batch.total_seqs_num_prefill == 0
    assert d._hold_ticks == 0
    batch, admitted = sched.schedule()
    assert producer.id in admitted
    assert batch.total_seqs_num_prefill == 1
    assert d._stat_fire_partial == 1


@pytest.mark.parametrize(
    "rapidserve,has_scheduler,expected",
    [(True, True, False), (False, False, False), (False, True, True)],
)
def test_delayer_init_distinguishes_rapidserve_from_connector_pd(
    rapidserve, has_scheduler, expected, monkeypatch
):
    monkeypatch.setenv("ATOM_ENABLE_PREFILL_DELAYER", "1")
    config = atom_config_double(enable_rapidserve=rapidserve, max_num_batched_tokens=64)
    config.parallel_config = SimpleNamespace(data_parallel_size=1)
    # Execute the real helper without importing AITER's worker IPC transport.
    source = Path(__file__).resolve().parents[1] / "atom/model_engine/engine_core.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_init_prefill_delayer"
    )
    namespace = {"Config": object, "envs": envs}
    exec(  # noqa: S102 — execute the repository helper, without GPU-only imports.
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    core = SimpleNamespace()
    core.scheduler = scheduler() if has_scheduler else None
    if core.scheduler:
        core.scheduler.set_prefill_delayer(None)
        core.scheduler.kv_connector = SimpleNamespace(is_producer=False)
    namespace["_init_prefill_delayer"](core, config)
    attached = core.scheduler is not None and core.scheduler.prefill_delayer is not None
    assert attached == expected


def test_rapidserve_decode_abort_retains_its_mark_only_path():
    sched = DecodeScheduler(MockConfig())
    seq = sequence(8)
    sched.waiting.append(seq)
    assert sched.abort_request(seq.id)
    assert seq.status == SequenceStatus.ABORTED
    assert seq in sched.waiting
    assert not sched._rejected


def test_nonhead_completed_offload_abort_reclaims_slot():
    sched = scheduler(hybrid=True, pool_entries={"state": 1})
    head, aborted = sequence(8, hybrid=True), sequence(8, hybrid=True)
    assert sched.block_manager.allocate(aborted, 0)
    sched.waiting.extend([head, aborted])
    sched._count_inflight_load(aborted)
    aborted.offload_loaded = True
    sched.kv_connector = SimpleNamespace(is_producer=False, is_offload=True)
    assert sched.block_manager.can_allocate(head, record=False) == -1
    assert sched.abort_request(aborted.id)
    assert not aborted.block_table and not aborted.state_slots
    assert sched._num_parked_remote_kv == 0
    assert sched.block_manager.can_allocate(head, record=False) >= 0


def test_parked_slots_still_allow_local_work_after_stall():
    sched = scheduler(max_num_seqs=1)
    sched._num_parked_remote_kv = 1
    seq = sequence(8)
    sched.waiting.append(seq)
    d = sched.prefill_delayer
    d._first = False
    d.stall_ticks = 1
    ready, pending = sched._local_prefill_pending_work()
    assert (ready, pending) == (True, 0)
    assert not d.should_allow_prefill(ready, pending, running_decode_batch=1)
    assert d.should_allow_prefill(ready, pending, running_decode_batch=1)
    assert d._stat_fire_stall == 1 and d._stat_fire_vacuous == 0
    _, admitted = sched.schedule()
    assert seq.id in admitted
