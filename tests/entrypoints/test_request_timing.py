import asyncio
import json

import pytest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.entrypoints.openai.request_timing import (
    FirstOutputSSE,
    RequestTimingMiddleware,
    record_nonstream_first_token,
)


def _sse(payload, newline="\n"):
    return ("data: " + json.dumps(payload, ensure_ascii=False) + newline * 2).encode()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"name": "search"}}]}}]},
        {"choices": [{"delta": {"function_call": {"arguments": "{"}}}]},
        {"choices": [{"text": " "}]},
        {"type": "content_block_delta", "delta": {"thinking": "hmm"}},
        {"type": "content_block_delta", "delta": {"partial_json": "{"}},
        {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "search"},
        },
    ],
)
def test_first_output_handles_fragmented_events_and_ignores_metadata(payload):
    detector = FirstOutputSSE()
    prefix = b": keepalive\r\n\r\n" + _sse(
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
    )
    assert not detector.feed(prefix)
    data = _sse(payload, "\r\n")
    found = [detector.feed(bytes([byte])) for byte in data]
    assert found == [False] * (len(data) - 1) + [True]
    assert not detector.feed(data)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        None,
        {"choices": None},
        {"choices": [None]},
        {"choices": [{"delta": {"tool_calls": 4}}]},
        {"type": "content_block_delta", "delta": []},
        {"usage": {"completion_tokens": 5}},
    ],
)
def test_unrecognized_events_do_not_break_output_detection(payload):
    detector = FirstOutputSSE()
    assert not detector.feed(_sse(payload))
    assert detector.feed(_sse({"choices": [{"text": "ok"}]}))


@pytest.mark.parametrize(
    "terminal",
    [
        b"data: [DONE]\n\n",
        _sse({"error": {"message": "failed"}}),
        _sse({"type": "error", "error": "failed"}),
    ],
)
def test_no_ttft_after_error_or_empty_completion(terminal):
    detector = FirstOutputSSE()
    assert not detector.feed(terminal + _sse({"choices": [{"text": "late"}]}))


def test_incomplete_event_buffer_is_bounded():
    detector = FirstOutputSSE()
    assert not detector.feed(b"x" * (detector.MAX_PENDING_BYTES + 1))
    assert detector.done and not detector._frames.pending


@pytest.mark.parametrize("chunk_size", [1, 3, 1024, 65536])
def test_many_metadata_frames_and_large_fragmented_output(chunk_size):
    prefix = (b": keepalive\n\n" + b"data: {}\r\n\r\n") * 100
    output = _sse({"choices": [{"text": "x" * 65536}]}, "\r\n")
    payload = prefix + output
    detector = FirstOutputSSE()
    found = []
    for start in range(0, len(payload), chunk_size):
        found.append(detector.feed(payload[start : start + chunk_size]))
    assert found == [False] * (len(found) - 1) + [True]
    assert not detector._frames.pending


def test_frame_limit_applies_after_consumed_metadata():
    detector = FirstOutputSSE()
    oversized = b"data: " + b"x" * detector.MAX_PENDING_BYTES + b"\n\n"
    assert not detector.feed(b"data: {}\r\n\r\n" + oversized)
    assert detector.done


def _metrics(exporter):
    return {
        (sample.name, sample.labels.get("streaming")): sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.endswith(("_count", "_sum"))
    }


def test_streaming_ttft_includes_preprocessing_skips_role_and_records_once(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(
        "atom.entrypoints.openai.request_timing.time.perf_counter", lambda: clock[0]
    )
    exporter = AtomMetricsExporter()
    original = [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream; charset=utf-8")],
        },
        {
            "type": "http.response.body",
            "body": _sse({"choices": [{"delta": {"role": "assistant"}}]}),
            "more_body": True,
        },
        {
            "type": "http.response.body",
            "body": _sse({"choices": [{"delta": {"content": "four tokens at once"}}]}),
            "more_body": True,
        },
        {
            "type": "http.response.body",
            "body": _sse({"choices": [{"delta": {"content": "later"}}]})
            + b"data: [DONE]\n\n",
            "more_body": False,
        },
    ]

    async def app(scope, receive, send):
        for when, message in zip((2.0, 3.0, 4.0, 5.0), original):
            clock[0] = when
            await send(message)

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(
        RequestTimingMiddleware(app, exporter.observe_time_to_first_token)(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions"},
            None,
            send,
        )
    )
    assert sent == original
    samples = _metrics(exporter)
    assert samples[("atom:time_to_first_token_seconds_count", "true")] == 1
    assert samples[("atom:time_to_first_token_seconds_sum", "true")] == 3.0
    exporter.update({"enabled": True})
    assert _metrics(exporter) == samples


def test_nonstream_first_token_is_request_local_and_precedes_response(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "atom.entrypoints.openai.request_timing.time.perf_counter", lambda: clock[0]
    )
    observations = []

    async def run():
        ready = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def app(scope, receive, send):
            index = scope["index"]
            ready[index].set()
            await release[index].wait()
            record_nonstream_first_token()
            record_nonstream_first_token()  # Fanout siblings do not add samples.
            clock[0] += 100
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def send(message):
            pass

        middleware = RequestTimingMiddleware(
            app, lambda value, streaming: observations.append((value, streaming))
        )
        scope = {"type": "http", "method": "POST", "path": "/v1/completions"}
        first = asyncio.create_task(middleware({**scope, "index": 0}, None, send))
        await ready[0].wait()
        clock[0] = 1.0
        second = asyncio.create_task(middleware({**scope, "index": 1}, None, send))
        await ready[1].wait()
        clock[0] = 10.0
        release[1].set()
        await second
        clock[0] = 20.0
        release[0].set()
        await first
        record_nonstream_first_token()  # No request context survives either task.

    asyncio.run(run())
    assert observations == [(9.0, False), (20.0, False)]


@pytest.mark.parametrize(
    "status,path", [(500, "/v1/chat/completions"), (200, "/health")]
)
def test_failed_or_unrelated_responses_produce_no_sample(status, path):
    observations = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": _sse({"choices": [{"text": "not a valid generation"}]}),
            }
        )

    async def send(message):
        pass

    asyncio.run(
        RequestTimingMiddleware(app, lambda *args: observations.append(args))(
            {"type": "http", "method": "POST", "path": path}, None, send
        )
    )
    assert observations == []
