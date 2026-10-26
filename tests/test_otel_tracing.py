# SPDX-License-Identifier: MIT
"""CPU tests for request context, phase boundaries, and lifecycle failures."""

import asyncio
import os
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import StatusCode

from atom.observability import tracing

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_ID = "00f067aa0ba902b7"


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON, shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(
        tracing, "_cached_tracer", provider.get_tracer("atom.standalone")
    )
    monkeypatch.setattr(tracing, "_provider_pid", os.getpid())
    monkeypatch.setattr(tracing, "_init_failed_pid", None)
    monkeypatch.setattr(tracing, "_warned", False)
    yield exporter
    provider.shutdown()


@pytest.fixture
def clock(monkeypatch):
    now = [10_000_000_000]
    monkeypatch.setattr(tracing.time, "time_ns", lambda: now[0] + 100_000_000_000)
    monkeypatch.setattr(tracing.time, "perf_counter_ns", lambda: now[0])
    return now


def sequence(number=7, **kwargs):
    return SimpleNamespace(
        id=number,
        num_prompt_tokens=100,
        num_finalized_tokens=104,
        prefix_cache_hit_tokens=64,
        external_request_id=f"r-{number}",
        **kwargs,
    )


def scope(flags="01", headers=True):
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": (
            [
                (b"traceparent", f"00-{TRACE_ID}-{PARENT_ID}-{flags}".encode()),
                (b"tracestate", b"othervendor=x,nsdye=1"),
            ]
            if headers
            else []
        ),
    }


async def no_receive():
    return {"type": "http.disconnect"}


async def no_send(message):
    pass


def execute(app, **kwargs):
    asyncio.run(
        tracing.TraceMiddleware(app, model=lambda: "test-model")(
            scope(**kwargs), no_receive, no_send
        )
    )


@pytest.mark.parametrize("flags", ["00", "01"])
def test_parent_dye_pickle_and_complete_phases(spans, clock, flags):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        # This round trip is the real Sequence carrier representation, not
        # a live SDK span; engine state is created after socket receipt.
        seq = pickle.loads(pickle.dumps(seq))
        tracing.enqueue(seq)
        clock[0] += 1_000_000_000
        tracing.enqueue(seq)  # scheduler admission must not reset receipt
        tracing.forward(seq, is_decode=False)
        clock[0] += 2_000_000_000
        tracing.prefill_complete(seq)
        tracing.first_token(seq)
        clock[0] += 3_000_000_000
        tracing.finish(seq, reason="max_tokens", num_tokens=104)
        tracing.finish(seq, reason="max_tokens", num_tokens=104)
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app, flags=flags)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert len(spans.get_finished_spans()) == 5
    assert set(result) == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
        "prefill_forward",
        "decode_forward",
    }
    assert result["atom.request"].parent.span_id == int(PARENT_ID, 16)
    for span in result.values():
        assert span.context.trace_id == int(TRACE_ID, 16)
        assert span.context.trace_state.get("nsdye") == "1"
        assert span.status.status_code != StatusCode.ERROR
    root = result["engine.request"]
    assert root.parent.span_id == result["atom.request"].context.span_id
    for name, seconds in [
        ("prefill_waiting", 1),
        ("prefill_forward", 2),
        ("decode_forward", 3),
    ]:
        span = result[name]
        assert span.parent.span_id == root.context.span_id
        assert span.end_time - span.start_time == seconds * 1e9
    assert root.attributes["gen_ai.usage.cached_tokens"] == 64
    assert root.attributes["gen_ai.usage.completion_tokens"] == 4
    assert root.attributes["gen_ai.latency.time_to_first_token"] == 3
    assert tracing._request.get() is None


def test_queue_abort_exports_no_fabricated_execution(spans, clock):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        clock[0] += 1_000_000_000
        tracing.finish(seq, reason="aborted", num_tokens=100)
        await receive()

    execute(app)
    result = spans.get_finished_spans()
    assert {s.name for s in result} == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
    }
    assert all(s.status.status_code == StatusCode.ERROR for s in result)


def test_chunked_prefill_first_token_is_not_prompt_completion(spans, clock):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        tracing.forward(seq, is_decode=False)
        clock[0] += 2_000_000_000
        tracing.forward(seq, is_decode=False)  # another chunk keeps first dispatch
        clock[0] += 2_000_000_000
        tracing.prefill_complete(seq)
        clock[0] += 1_000_000_000
        tracing.first_token(seq)  # deferred output delivered one step later
        clock[0] += 1_000_000_000
        tracing.finish(seq, reason="stop_sequence", num_tokens=104)

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert (
        result["prefill_forward"].end_time - result["prefill_forward"].start_time == 4e9
    )
    assert (
        result["engine.request"].attributes["gen_ai.latency.time_to_first_token"] == 5
    )
    assert (
        result["decode_forward"].end_time - result["decode_forward"].start_time == 2e9
    )
    assert result["engine.request"].status.status_code != StatusCode.ERROR


def test_one_token_request_has_no_decode(spans, clock):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        tracing.forward(seq, is_decode=False)
        clock[0] += 1_000_000_000
        tracing.prefill_complete(seq)
        tracing.first_token(seq)
        clock[0] += 1
        tracing.finish(seq, reason="eos", num_tokens=101)

    execute(app)
    assert "decode_forward" not in {s.name for s in spans.get_finished_spans()}


def test_http_span_lives_through_last_sse_body(spans):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        assert not spans.get_finished_spans()
        await send(
            {"type": "http.response.body", "body": b"data: x\n\n", "more_body": True}
        )
        await asyncio.sleep(0)
        assert not spans.get_finished_spans()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app)
    assert len(spans.get_finished_spans()) == 1


def test_exception_keeps_only_type_and_reraises(spans):
    async def app(scope, receive, send):
        raise ValueError("PRIVATE_PROMPT_DO_NOT_EXPORT")

    with pytest.raises(ValueError, match="PRIVATE_PROMPT"):
        execute(app)
    span = spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "ValueError"
    assert not span.events
    assert "PRIVATE_PROMPT" not in str(span.attributes)


def test_concurrent_requests_and_siblings_do_not_share_parent(spans, clock):
    async def app(scope, receive, send):
        seqs = [sequence(1), sequence(2)]
        await asyncio.sleep(0)
        tracing.bind_sequences(seqs)
        for seq in seqs:
            tracing.enqueue(seq)
            tracing.forward(seq, is_decode=False)
            tracing.finish(seq, reason="max_tokens", num_tokens=101)

    async def run():
        middleware = tracing.TraceMiddleware(app)
        await asyncio.gather(
            *(middleware(scope(), no_receive, no_send) for _ in range(8))
        )

    asyncio.run(run())
    result = spans.get_finished_spans()
    http = {s.context.span_id for s in result if s.name == "atom.request"}
    engines = [s for s in result if s.name == "engine.request"]
    assert len(http) == 8
    assert len(engines) == 16
    assert all(sum(s.parent.span_id == parent for s in engines) == 2 for parent in http)


@pytest.mark.parametrize(
    "header", [b"bad", b"00-00000000000000000000000000000000-0000000000000000-01"]
)
def test_invalid_parent_creates_root(spans, header):
    async def app(scope, receive, send):
        pass

    request_scope = scope()
    request_scope["headers"] = [(b"traceparent", header)]
    asyncio.run(tracing.TraceMiddleware(app)(request_scope, no_receive, no_send))
    span = spans.get_finished_spans()[0]
    assert span.parent is None
    assert span.context.trace_id != 0


def test_disabled_path_does_not_load_sdk(monkeypatch):
    monkeypatch.setattr(tracing, "_ENABLED", False)
    monkeypatch.setattr(
        tracing, "_tracer", lambda: pytest.fail("SDK touched with tracing disabled")
    )
    seq = sequence()

    async def app(scope, receive, send):
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        tracing.forward(seq, is_decode=False)
        tracing.prefill_complete(seq)
        tracing.first_token(seq)
        tracing.finish(seq, reason="max_tokens")

    execute(app)
    assert not hasattr(seq, "_atom_trace_carrier")


@pytest.mark.parametrize("direction", ["do_remote_prefill", "do_remote_decode"])
def test_pd_is_marked_unsupported_not_misattributed(spans, direction):
    async def app(scope, receive, send):
        seq = sequence(kv_transfer_params={direction: True})
        tracing.bind_sequences([seq])
        assert not hasattr(seq, "_atom_trace_carrier")

    execute(app)
    assert spans.get_finished_spans()[0].attributes["atom.trace.unsupported"] == "pd"


def test_exporter_initialization_failure_preserves_request(monkeypatch):
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "_provider_pid", None)
    monkeypatch.setattr(tracing, "_init_failed_pid", None)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setattr(tracing, "_cached_tracer", None)
    tracing.initialize()
    tracing.initialize()  # failure is sticky until restart
    called = []

    async def app(scope, receive, send):
        called.append(True)

    execute(app)
    execute(app)
    assert called == [True, True]
    assert tracing._init_failed_pid == os.getpid()


def test_cancelled_stream_ends_http_span_and_preserves_cancellation(spans):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        execute(app)
    result = spans.get_finished_spans()
    assert len(result) == 1
    assert result[0].attributes["error.type"] == "CancelledError"
    assert result[0].status.status_code == StatusCode.ERROR
    assert tracing._request.get() is None


def test_disconnect_after_final_sse_body_is_normal_completion(spans):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send(
            {
                "type": "http.response.body",
                "body": b"data: [DONE]\n\n",
                "more_body": False,
            }
        )
        assert (await receive())["type"] == "http.disconnect"

    execute(app)
    span = spans.get_finished_spans()[0]
    assert span.status.status_code != StatusCode.ERROR
    assert "error.type" not in span.attributes


def test_public_request_id_survives_add_carrier(spans, clock):
    async def app(scope, receive, send):
        tracing.set_request_id("cmpl-public-id")
        seq = sequence()
        tracing.bind_sequences([seq])
        seq = pickle.loads(pickle.dumps(seq))
        tracing.enqueue(seq)
        clock[0] += 1
        tracing.finish(seq, reason="max_tokens", num_tokens=100)

    execute(app)
    assert all(
        s.attributes["gen_ai.request.id"] == "cmpl-public-id"
        for s in spans.get_finished_spans()
    )


@pytest.mark.parametrize("phase", ["queue", "prefill", "decode"])
def test_abort_marks_only_running_phase_and_root(spans, clock, phase):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        clock[0] += 1_000_000_000
        if phase != "queue":
            tracing.forward(seq, is_decode=False)
            clock[0] += 1_000_000_000
        if phase == "decode":
            tracing.prefill_complete(seq)
            tracing.first_token(seq)
            clock[0] += 1_000_000_000
        # Even an abort before the second output token has a decode phase.
        tracing.finish(
            seq, reason="aborted", num_tokens=101 if phase == "decode" else 100
        )

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    active = {
        "queue": "prefill_waiting",
        "prefill": "prefill_forward",
        "decode": "decode_forward",
    }[phase]
    assert {
        name
        for name, span in result.items()
        if span.status.status_code == StatusCode.ERROR
    } == {"engine.request", active}
    for name, span in result.items():
        assert ("error.type" in span.attributes) == (name in {"engine.request", active})


@pytest.mark.parametrize(
    "completion_tokens,decode_delay", [(1, 0), (1, 1), (4, 0), (4, 1)]
)
def test_latency_attributes_match_exported_phase_boundaries(
    spans, clock, completion_tokens, decode_delay
):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        clock[0] += 1_000_000_000
        tracing.forward(seq, is_decode=False)
        clock[0] += 1_000_000_000
        tracing.prefill_complete(seq)
        clock[0] += decode_delay * 1_000_000_000
        tracing.finish(seq, reason="max_tokens", num_tokens=100 + completion_tokens)

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    root = result["engine.request"]
    for name, attribute in tracing._PHASE_LATENCY.items():
        assert (name in result) == (attribute in root.attributes)
        if name in result:
            span = result[name]
            assert root.attributes[attribute] == (span.end_time - span.start_time) / 1e9
    # Collecting usage across all spans must count each sequence exactly once.
    assert (
        sum(
            s.attributes.get("gen_ai.usage.completion_tokens", 0)
            for s in result.values()
        )
        == completion_tokens
    )
    for name in tracing._PHASE_LATENCY.keys() & result.keys():
        assert set(result[name].attributes) == {
            "gen_ai.request.id",
            "atom.sequence.id",
            "gen_ai.response.model",
        }


def test_request_and_finish_never_initialize_sdk(monkeypatch, clock):
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "_provider_pid", None)
    monkeypatch.setattr(tracing, "_cached_tracer", None)
    monkeypatch.setattr(
        tracing, "initialize", lambda: pytest.fail("hot-path SDK initialization")
    )
    import builtins

    original_import = builtins.__import__
    imports = []

    def import_module(name, *a, **k):
        if name.startswith("opentelemetry"):
            imports.append(name)
            raise ImportError("unexpected hot-path SDK import")
        return original_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", import_module)

    async def app(scope, receive, send):
        seq = sequence(
            _atom_trace_carrier={"traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01"}
        )
        tracing.enqueue(seq)
        tracing.finish(seq, reason="max_tokens")

    execute(app)
    assert imports == []


def test_initialize_caches_tracer_and_resets_after_fork(monkeypatch):
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    monkeypatch.setattr(tracing, "_ENABLED", True)
    for name in ("_provider", "_cached_tracer", "_provider_pid", "_init_failed_pid"):
        monkeypatch.setattr(tracing, name, None)
    monkeypatch.setattr(tracing, "_lock", tracing.threading.Lock())
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:1/v1/traces"
    )
    exporters = []

    def exporter(**kwargs):
        exporters.append(InMemorySpanExporter())
        return exporters[-1]

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", exporter)
    tracing.initialize()
    provider = tracing._provider
    tracer = tracing._tracer()
    monkeypatch.setattr(
        provider, "get_tracer", lambda *a: pytest.fail("tracer recreated")
    )
    tracing.initialize()
    assert tracing._tracer() is tracer
    assert len(exporters) == 1
    tracing._after_fork()
    try:
        assert tracing._tracer() is None
        assert tracing._cached_tracer is None
        assert tracing._request.get() is None
        tracing.initialize()
        assert tracing._tracer() is not tracer
        assert len(exporters) == 2
    finally:
        tracing.shutdown()
        provider.shutdown()


def test_shutdown_still_closes_provider_after_flush_failure(monkeypatch):
    calls = []

    def flush(**kwargs):
        calls.append("flush")
        raise RuntimeError("flush failed")

    monkeypatch.setattr(
        tracing,
        "_provider",
        SimpleNamespace(force_flush=flush, shutdown=lambda: calls.append("shutdown")),
    )
    monkeypatch.setattr(tracing, "_provider_pid", os.getpid())
    monkeypatch.setattr(tracing, "_cached_tracer", object())
    tracing.shutdown()
    tracing.shutdown()
    assert calls == ["flush", "shutdown"]
    assert tracing._tracer() is None


def test_disabled_batch_hooks_do_not_visit_sequences(monkeypatch):
    monkeypatch.setattr(tracing, "_ENABLED", False)

    class UnusedBatch:
        def __getattribute__(self, name):
            pytest.fail("disabled tracing inspected a batch")

    class UnusedSequences:
        def __iter__(self):
            pytest.fail("disabled tracing iterated sequences")

    tracing.enqueue_many(UnusedSequences())
    tracing.record_forward(UnusedBatch(), UnusedSequences())


@pytest.mark.parametrize("status,failed", [(200, False), (429, False), (503, True)])
def test_only_server_errors_mark_http_span_failed(spans, status, failed):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app)
    span = spans.get_finished_spans()[0]
    assert span.attributes["http.response.status_code"] == status
    assert (span.status.status_code == StatusCode.ERROR) == failed


def test_repeated_tracestate_headers_are_combined(spans):
    carriers = []

    async def app(scope, receive, send):
        carriers.append(tracing._request.get()["carrier"])

    request = scope()
    request["headers"].append((b"TraceState", b"lastvendor=y"))
    asyncio.run(
        tracing.TraceMiddleware(app, model=lambda: "test-model")(
            request, no_receive, no_send
        )
    )
    span = spans.get_finished_spans()[0]
    assert span.context.trace_state.get("othervendor") == "x"
    assert span.context.trace_state.get("nsdye") == "1"
    assert span.context.trace_state.get("lastvendor") == "y"
    assert "lastvendor=y" in carriers[0]["tracestate"]


def test_hooks_fail_open_and_leave_request_untraced(spans, clock):
    class BrokenTiming:
        def __getattr__(self, name):
            raise RuntimeError("broken timing")

    async def app(scope, receive, send):
        # A truthy non-dict transfer payload cannot be inspected.
        bad = sequence(1, kv_transfer_params=["unexpected"])
        tracing.bind_sequences([bad])
        assert not hasattr(bad, "_atom_trace_carrier")
        tracing.enqueue(bad)
        assert getattr(bad, "_atom_trace_timing", None) is None

        broken = sequence(2, _atom_trace_timing=BrokenTiming())
        tracing.forward(broken, is_decode=False)
        tracing.prefill_complete(broken)
        tracing.first_token(broken)
        batch = SimpleNamespace(
            is_dummy_run=False, req_ids=[99], total_seqs_num_decode=0
        )
        tracing.record_forward(batch, {})  # missing sequence id
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app)
    assert tracing._warned
    assert [s.name for s in spans.get_finished_spans()] == ["atom.request"]


@pytest.mark.parametrize(
    "hook", ["enqueue", "forward", "prefill_complete", "first_token", "finish"]
)
def test_failed_timing_hook_drops_trace_and_later_finish_still_runs(
    spans, clock, monkeypatch, hook
):
    class BrokenTiming:
        def __getattr__(self, name):
            raise RuntimeError("broken timing")

    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        if hook == "enqueue":
            with monkeypatch.context() as patch:
                patch.setattr(
                    tracing.time,
                    "time_ns",
                    lambda: (_ for _ in ()).throw(RuntimeError("clock failure")),
                )
                tracing.enqueue(seq)
        else:
            tracing.enqueue(seq)
            seq._atom_trace_timing = BrokenTiming()
            if hook == "forward":
                tracing.forward(seq, is_decode=False)
            elif hook == "prefill_complete":
                tracing.prefill_complete(seq)
            elif hook == "first_token":
                tracing.first_token(seq)
            else:
                tracing.finish(seq, reason="max_tokens")
        assert not hasattr(seq, "_atom_trace_carrier")
        assert not hasattr(seq, "_atom_trace_timing")
        tracing.enqueue(seq)  # no re-admission of a partial trace
        tracing.finish(seq, reason="max_tokens")
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app)
    assert [s.name for s in spans.get_finished_spans()] == ["atom.request"]


def test_finish_clock_failure_preserves_request_completion(spans, clock, monkeypatch):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        with monkeypatch.context() as patch:
            patch.setattr(
                tracing.time,
                "perf_counter_ns",
                lambda: (_ for _ in ()).throw(RuntimeError("clock failure")),
            )
            tracing.finish(seq, reason="max_tokens")
        assert not hasattr(seq, "_atom_trace_carrier")
        tracing.finish(seq, reason="max_tokens")
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app)
    assert [s.name for s in spans.get_finished_spans()] == ["atom.request"]
