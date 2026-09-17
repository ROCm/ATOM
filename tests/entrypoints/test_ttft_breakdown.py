"""Preparation observations and opt-in scheduler-to-SSE delivery timing."""

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from prometheus_client.parser import text_string_to_metric_families
from starlette.responses import StreamingResponse

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
from atom.entrypoints.openai.request_timing import RequestTimingMiddleware
from atom.entrypoints.openai.streaming_dispatch import StreamBatchDispatcher
from atom.model_engine.request import RequestOutput

DELIVERY = "atom:ttft_output_delivery_seconds"


def _samples(exporter):
    return {
        sample.name: sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.endswith(("_count", "_sum"))
    }


async def _serve(source, observe):
    sent = []

    async def app(scope, receive, send):
        await StreamingResponse(
            api_server._client_stream(source, "req"),
            media_type="text/event-stream",
        )(scope, receive, send)

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    await RequestTimingMiddleware(app, observe)(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "asgi": {"spec_version": "2.4"},
        },
        receive,
        send,
    )
    return sent


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tokenized", [False, True])
@pytest.mark.parametrize("trace_enabled", [False, True])
def test_delivery_tracks_first_token_across_threads_without_changing_output(
    monkeypatch, enabled, tokenized, trace_enabled
):
    trace_events = []

    @contextmanager
    def trace_span(name):
        assert trace_enabled, "disabled trace must bypass the context manager"
        trace_events.append((name, "enter"))
        try:
            yield
        finally:
            trace_events.append((name, "exit"))

    monkeypatch.setattr(api_server, "TTFT_TRACE_ENABLED", trace_enabled)
    monkeypatch.setattr(api_server, "ttft_trace_span", trace_span)
    monkeypatch.setenv("ATOM_ENABLE_METRICS_OUTPUT_DELIVERY", str(int(enabled)))
    exporter, request_metrics, _, breakdown = create_metrics_exporter()
    monkeypatch.setattr(api_server, "_ttft_breakdown", breakdown)
    monkeypatch.setattr(api_server, "_request_logger", None)
    callbacks = []
    encoded = []
    tokenizer = SimpleNamespace(
        encode=lambda text: encoded.append(text) or [1],
        decode=lambda ids, **kw: "".join(chr(i) for i in ids),
    )
    dispatcher = StreamBatchDispatcher(tokenizer)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", dispatcher)

    def preprocess(tokens, params, *, stream_callback, **kwargs):
        assert tokens == [1]
        callbacks.append(stream_callback)
        return SimpleNamespace(id=7, num_prompt_tokens=1)

    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            tokenizer=tokenizer,
            io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
            core_mgr=SimpleNamespace(add_request=lambda seqs: None),
        ),
    )
    monkeypatch.setattr(
        api_server, "_validate_sequence_context_length", lambda seq: None
    )
    monkeypatch.setattr(api_server.time, "time", lambda: 10.5)

    class UnreadableStamp:
        output_tokens = [ord("x")]
        finished = False
        finish_reason = None

        @property
        def scheduler_output_at(self):
            raise AssertionError("disabled diagnostics must not inspect the stamp")

    def emit():
        callback = callbacks[0]
        callback(RequestOutput(7, [], False))  # Does not consume the first stamp.
        callback(RequestOutput(7, [ord("a")], False, scheduler_output_at=10.0))
        # A repeated/older-engine stamp must not replace the first one.
        callback(RequestOutput(7, [ord("b")], False, scheduler_output_at=10.4))
        if not enabled:
            callback(UnreadableStamp())
        callback(RequestOutput(7, [], True, finish_reason="stop"))
        dispatcher.flush()

    async def source():
        seq_id, collector, _ = await api_server.setup_streaming_request(
            [1] if tokenized else "prompt", object(), "req"
        )
        try:
            await asyncio.to_thread(emit)
            chunk = await collector.get()
            assert chunk["finished"]
            expected = "ab" if enabled else "abx"
            assert chunk["text"] == expected
            # A role-only frame is not the end of first-output delivery.
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": expected}}]}
            ) + "\n\n"
            yield 'data: {"choices":[{"delta":{"content":"later"}}]}\n\n'
        finally:
            api_server.cleanup_stream(seq_id)
            api_server.cleanup_request("req")

    sent = asyncio.run(_serve(source(), request_metrics.observe_time_to_first_token))
    assert any(b"later" in msg.get("body", b"") for msg in sent)
    samples = _samples(exporter)
    assert encoded == ([] if tokenized else ["prompt"])
    assert samples["atom:api_tokenize_seconds_count"] == (0 if tokenized else 1)
    assert samples["atom:ttft_api_preprocess_seconds_count"] == 1
    if enabled:
        assert samples[f"{DELIVERY}_count"] == 1
        assert samples[f"{DELIVERY}_sum"] == pytest.approx(0.5)
    else:
        assert not any(name.startswith(DELIVERY) for name in samples)
    assert "req" not in api_server._stream_loops
    assert trace_events == (
        [("ttft[api_preprocess]", "enter"), ("ttft[api_preprocess]", "exit")]
        if trace_enabled
        else []
    )


def test_unstamped_and_error_streams_do_not_create_delivery_samples(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_METRICS_OUTPUT_DELIVERY", "1")
    exporter, request_metrics, _, breakdown = create_metrics_exporter()
    monkeypatch.setattr(api_server, "_ttft_breakdown", breakdown)
    monkeypatch.setattr(api_server, "_request_logger", None)

    async def source():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    asyncio.run(_serve(source(), request_metrics.observe_time_to_first_token))
    assert _samples(exporter)[f"{DELIVERY}_count"] == 0

    async def failed():
        # Even a stamped request must not observe delivery of an error frame.
        api_server.get_request_timing().scheduler_output_at = 1.0
        yield 'data: {"error":{"message":"failed"}}\n\n'
        yield "data: [DONE]\n\n"

    asyncio.run(_serve(failed(), request_metrics.observe_time_to_first_token))
    assert _samples(exporter)[f"{DELIVERY}_count"] == 0


def test_clock_skew_and_nonfinite_delivery_are_not_observed(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_METRICS_OUTPUT_DELIVERY", "1")
    exporter, _, _, breakdown = create_metrics_exporter()
    for invalid in (-1.0, float("nan"), float("inf")):
        breakdown.observe_output_delivery(invalid)
    assert _samples(exporter)[f"{DELIVERY}_count"] == 0


def test_summary_uses_streaming_ttft_and_does_not_infer_missing_stages(
    monkeypatch, capsys
):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools/analyze_ttft_breakdown.py"
    spec = importlib.util.spec_from_file_location("ttft_analysis", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    exporter, request_metrics, _, _ = create_metrics_exporter()
    request_metrics.observe_time_to_first_token(0.1, True)
    request_metrics.observe_time_to_first_token(10.0, False)
    monkeypatch.setattr(tool, "_fetch", lambda url: exporter.render().decode())
    monkeypatch.setattr(tool.sys, "argv", [str(path), "http://unused/metrics"])
    assert tool.main() == 0
    output = capsys.readouterr().out
    ttft_row = next(row for row in output.splitlines() if "ttft_total" in row)
    assert "count=    1" in ttft_row and "mean= 100.0ms" in ttft_row
    assert "residual" not in output
    delivery_row = next(row for row in output.splitlines() if "output_delivery" in row)
    assert "n/a" in delivery_row
