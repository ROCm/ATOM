# SPDX-License-Identifier: MIT
"""Run real scheduler admission/postprocess hooks without a GPU executor."""

# Pytest discovers these imported fixtures by their original names.
# ruff: noqa: F811

import asyncio
import pickle
from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_otel_tracing import clock, execute, spans  # noqa: F401

from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.observability import tracing
from atom.sampling_params import SamplingParams


def output(ids, tokens):
    return ScheduledBatchOutput(
        req_ids=ids,
        token_ids=[(token,) for token in tokens],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
    )


@pytest.mark.parametrize("pp_size", [1, 2])
def test_real_chunked_prefill_and_decode_hooks(spans, clock, monkeypatch, pp_size):
    async def app(scope, receive, send):
        scheduler = Scheduler(
            MockConfig(max_num_batched_tokens=4, pipeline_parallel_size=pp_size)
        )
        # An independent Prometheus early return must not erase trace timing.
        monkeypatch.setattr(scheduler.metrics, "enqueue", lambda *a, **k: None)
        monkeypatch.setattr(scheduler.metrics, "record_forward", lambda *a: None)
        seq = Sequence(
            list(range(10, 18)),
            block_size=4,
            sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
        )
        tracing.bind_sequences([seq])
        scheduler.add(seq)
        clock[0] += 1_000_000_000
        batch, seqs = scheduler.schedule()
        tracing.record_forward(batch, seqs)
        clock[0] += 1_000_000_000
        scheduler.postprocess(list(scheduler.running), output([], []), batch=batch)
        assert seq.is_partial_prefill
        assert seq._atom_trace_timing.prefill_end_ns is None
        batch, seqs = scheduler.schedule()
        tracing.record_forward(batch, seqs)
        clock[0] += 1_000_000_000
        scheduler.postprocess(
            list(scheduler.running), output([seq.id], [42]), batch=batch
        )
        assert not seq.is_partial_prefill
        assert seq._atom_trace_timing.prefill_end_ns is not None
        for _ in range(4):
            if scheduler.is_finished():
                break
            batch, seqs = scheduler.schedule()
            tracing.record_forward(batch, seqs)
            clock[0] += 1_000_000_000
            scheduler.postprocess(
                list(scheduler.running), output([seq.id], [43]), batch=batch
            )
        assert scheduler.is_finished()

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert set(result) == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
        "prefill_forward",
        "decode_forward",
    }
    assert (
        result["prefill_waiting"].end_time - result["prefill_waiting"].start_time == 1e9
    )
    assert (
        result["prefill_forward"].end_time - result["prefill_forward"].start_time == 2e9
    )
    assert result["engine.request"].attributes["gen_ai.usage.completion_tokens"] == 3


def test_real_waiting_abort_closes_queue(spans, clock):
    async def app(scope, receive, send):
        scheduler = Scheduler(MockConfig())
        seq = Sequence([10, 11, 12, 13], block_size=4)
        tracing.bind_sequences([seq])
        scheduler.add(seq)
        assert scheduler.abort_request(seq.id)
        clock[0] += 1_000_000_000
        scheduler.schedule()

    execute(app)
    assert {s.name for s in spans.get_finished_spans()} == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
    }


def test_core_manager_captures_context_after_executor_preprocess(spans):
    frames = []

    async def app(scope, receive, send):
        # Match the API: Sequence construction runs in an executor, while
        # CoreManager.add_request runs back on the HTTP event-loop thread.
        seq = await asyncio.get_running_loop().run_in_executor(
            None, lambda: Sequence([10, 11, 12, 13], block_size=4)
        )
        manager = SimpleNamespace(
            pp_size=1,
            _routable_engine_count=1,
            label="test",
            _seq_id_to_callback={},
            _send_request=lambda rank, payload: frames.append(payload),
        )
        CoreManager.add_request(manager, [seq])

    execute(app)
    _, seqs = pickle.loads(frames[0])
    http = spans.get_finished_spans()[0]
    assert (
        seqs[0]._atom_trace_carrier["traceparent"].split("-")[2]
        == f"{http.context.span_id:016x}"
    )
    assert seqs[0]._atom_trace_carrier["tracestate"] == "othervendor=x,nsdye=1"


@pytest.mark.parametrize("already_bound", [False, True])
def test_same_node_pd_marks_http_and_skips_both_engine_carriers(
    spans, clock, already_bound
):
    frames = []

    async def app(scope, receive, send):
        seq = Sequence([10, 11, 12, 13], block_size=4)
        if already_bound:
            tracing.bind_sequences([seq])
        manager = SimpleNamespace(
            label="test-pd",
            _seq_id_to_callback={},
            _send_request=lambda rank, payload: frames.append(payload),
        )
        DisaggCoreManager.add_request(manager, [seq])
        assert len(frames) == 2
        for frame in frames:
            _, seqs, *_ = pickle.loads(frame)
            for child_seq in seqs:
                assert not hasattr(child_seq, "_atom_trace_carrier")
                tracing.enqueue(child_seq)
                tracing.forward(child_seq, is_decode=False)
                tracing.finish(child_seq, reason="max_tokens")

    execute(app)
    result = spans.get_finished_spans()
    assert len(result) == 1
    assert result[0].name == "atom.request"
    assert result[0].attributes["atom.trace.unsupported"] == "pd"


@pytest.mark.parametrize("engine_kind", ["normal", "dp", "pp_head", "pp_downstream"])
@pytest.mark.parametrize("failure", ["none", "drain", "publish", "shutdown"])
def test_engine_teardown_always_flushes_traces(monkeypatch, engine_kind, failure):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.engine_core import DPEngineCoreProc, EngineCore
        from atom.model_engine.pp_engine_core import PPEngineCoreProc
    cls = {
        "normal": EngineCore,
        "dp": DPEngineCoreProc,
        "pp_head": PPEngineCoreProc,
        "pp_downstream": PPEngineCoreProc,
    }[engine_kind]
    core = cls.__new__(cls)
    calls = []

    def cleanup(name):
        calls.append(name)
        if failure == name:
            raise RuntimeError(name)

    core.label = "test"
    core.utility_queue = None
    core.utility_handler = SimpleNamespace(
        process_queue=lambda *a: None, push_metrics=lambda **k: None
    )
    core.pull_and_process_input_queue = lambda: True
    core._drain_kv_work_at_exit = lambda: cleanup("drain")
    core.scheduler = SimpleNamespace(
        heartbeat_throughput=lambda now: None,
        is_finished=lambda: True,
        publish_kv_events=lambda: cleanup("publish"),
        shutdown_kv_events=lambda: cleanup("shutdown"),
    )
    core.runner_mgr = SimpleNamespace(call_func=lambda *a, **k: None)
    core.is_head = engine_kind == "pp_head"
    core.kv_transfer_enabled = False
    core._is_rl_weights_offloaded = core._weight_sync_fenced = False
    core._sync_dp_state = lambda *a: (False, True, False)
    monkeypatch.setattr(tracing, "shutdown", lambda: calls.append("trace"))
    if failure in {"drain", "shutdown"} and not (
        engine_kind == "pp_downstream" and failure == "drain"
    ):
        with pytest.raises(RuntimeError, match=failure):
            core.busy_loop()
    else:
        core.busy_loop()
    assert calls[-1] == "trace"
    assert calls.count("trace") == 1


def test_pp_uses_frozen_chunk_boundary_with_two_batches_in_flight(spans, clock):
    async def app(scope, receive, send):
        scheduler = Scheduler(
            MockConfig(max_num_batched_tokens=4, pipeline_parallel_size=2)
        )
        seq = Sequence(
            list(range(10, 18)),
            block_size=4,
            sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
        )
        tracing.bind_sequences([seq])
        scheduler.add(seq)
        first, first_seqs = scheduler.schedule()
        tracing.record_forward(first, first_seqs)
        final, final_seqs = scheduler.schedule()
        tracing.record_forward(final, final_seqs)
        assert first.is_final_chunk == [False]
        assert final.is_final_chunk == [True]
        assert not seq.is_partial_prefill
        clock[0] += 1_000_000_000
        scheduler.postprocess(
            list(first_seqs.values()), output([seq.id], [42]), batch=first
        )
        assert seq._atom_trace_timing.prefill_end_ns is None
        clock[0] += 1_000_000_000
        scheduler.postprocess(
            list(final_seqs.values()), output([seq.id], [43]), batch=final
        )
        boundary = seq._atom_trace_timing.prefill_end_ns
        assert boundary is not None
        batch, seqs = scheduler.schedule()
        tracing.record_forward(batch, seqs)
        clock[0] += 1_000_000_000
        scheduler.postprocess(list(seqs.values()), output([seq.id], [44]), batch=batch)
        assert seq._atom_trace_timing.prefill_end_ns == boundary
        scheduler._mark_finished(seq, "max_tokens", 103)

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert result["prefill_forward"].end_time == result["decode_forward"].start_time
    assert (
        result["engine.request"].attributes["gen_ai.latency.time_in_model_prefill"] == 2
    )


@pytest.mark.parametrize("pp_size", [1, 2])
def test_real_engine_dispatch_keeps_tracing_when_metrics_returns_early(
    spans, clock, monkeypatch, pp_size
):
    import queue
    from collections import deque

    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.engine_core import EngineCore
        from atom.model_engine.pp_engine_core import PPEngineCoreProc

    async def app(scope, receive, send):
        cls = EngineCore if pp_size == 1 else PPEngineCoreProc
        core = cls.__new__(cls)
        core.scheduler = Scheduler(
            MockConfig(max_num_batched_tokens=4, pipeline_parallel_size=pp_size)
        )
        core.output_queue = queue.Queue()
        core.stream_output_queue = queue.Queue()
        core.kv_transfer_enabled = False
        core._poll_kv_transfer_progress = lambda: None
        core._poll_dynamic_chunking_calibration = lambda: None
        core._in_flight = deque()
        core._pending_prefix_hash = deque()
        core._defer_prefix_hash = False
        core.pp_size = pp_size
        completions = deque()
        core.pp_transport = SimpleNamespace(
            send_metadata=lambda batch: None,
            recv_completion=lambda **k: completions.popleft() if completions else None,
        )
        monkeypatch.setattr(core.scheduler.metrics, "enqueue", lambda *a, **k: None)
        monkeypatch.setattr(core.scheduler.metrics, "record_forward", lambda *a: None)
        seq = Sequence(
            list(range(10, 18)),
            block_size=4,
            sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
        )
        tracing.bind_sequences([seq])
        core.scheduler.add(seq)

        def run(name, batch=None, **kwargs):
            if name == "flush_pp_send":
                return None
            assert name == "forward"
            assert seq._atom_trace_timing.forward_ns is not None
            clock[0] += 1_000_000_000
            result = output([seq.id], [42])
            if batch.produces_output():
                completions.append((batch.req_ids, result))
            return result

        core.runner_mgr = SimpleNamespace(call_func=run)
        for _ in range(5):
            if core.scheduler.is_finished():
                break
            if pp_size == 1:
                assert core._process_engine_step_inner()
            else:
                core._pp_head_step()
        assert core.scheduler.is_finished()

    execute(app)
    assert {s.name for s in spans.get_finished_spans()} == {
        "atom.request",
        "engine.request",
        *tracing._PHASE_LATENCY,
    }


def test_engine_initializes_tracing_before_ready(monkeypatch):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core
    events = []
    cls = engine_core.EngineCore
    monkeypatch.setattr(
        engine_core.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None)
    )
    monkeypatch.setattr(
        engine_core,
        "init_exit_handler",
        lambda core: setattr(core, "_finalizer", lambda: None),
    )
    monkeypatch.setattr(cls, "_init_data_parallel", lambda *a: None)
    monkeypatch.setattr(cls, "_post_model_load_hook", lambda *a: None)
    monkeypatch.setattr(engine_core.StateRuntime, "from_wire", lambda wire: None)

    def call(name, *a, **k):
        events.append(name)
        return (
            {"num_kvcache_blocks": 10, "state_runtime": None}
            if name == "get_num_blocks"
            else True
        )

    monkeypatch.setattr(
        engine_core, "AsyncIOProcManager", lambda *a: SimpleNamespace(call_func=call)
    )
    monkeypatch.setattr(engine_core, "Scheduler", lambda *a, **k: None)
    monkeypatch.setattr(engine_core, "EngineUtilityHandler", lambda *a, **k: None)
    monkeypatch.setattr(tracing, "initialize", lambda: events.append("initialize"))
    monkeypatch.setattr(
        cls, "_freeze_after_startup", lambda *a: events.append("freeze")
    )
    monkeypatch.setattr(cls, "_send_ready_signal", lambda *a: events.append("ready"))
    config = MockConfig(
        parallel_config=SimpleNamespace(control_address="unused", data_parallel_size=1),
        tp_world_size=1,
        prefill_context_parallel_size=1,
        runner_qualname="unused",
        enforce_eager=True,
        disagg_is_decode=False,
        pipeline_parallel_size=1,
        kv_transfer_config=None,
    )
    cls(config, "unused", "unused")
    # Only the order around `initialize` is the contract; other startup RPCs
    # may come and go between these points.
    order = [
        events.index(name)
        for name in ("allocate_kv_cache", "initialize", "freeze", "ready")
    ]
    assert order == sorted(order)
