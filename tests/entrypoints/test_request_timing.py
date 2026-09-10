import asyncio
import json

import pytest
from prometheus_client.parser import text_string_to_metric_families
from starlette.responses import StreamingResponse

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
from atom.entrypoints.openai.request_timing import (
    RequestTimingMiddleware,
    record_nonstream_first_token,
)
from atom.entrypoints.openai.streaming_dispatch import longest_silence_seconds


def _sse(payload, newline="\n"):
    return "data: " + json.dumps(payload, ensure_ascii=False) + newline * 2


async def _serve_stream(
    source,
    observe,
    *,
    path="/v1/chat/completions",
    method="POST",
    status=200,
    media_type="text/event-stream",
    spec_version="2.4",
):
    sent = []

    async def app(scope, receive, send):
        response = StreamingResponse(
            api_server._client_stream(source, "req"),
            status_code=status,
            media_type=media_type,
        )
        await response(scope, receive, send)

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    await RequestTimingMiddleware(app, observe)(
        {
            "type": "http",
            "method": method,
            "path": path,
            "asgi": {"spec_version": spec_version},
        },
        receive,
        send,
    )
    return sent


def _run_stream(chunks, **kwargs):
    observations = []

    async def source():
        for chunk in chunks:
            yield chunk

    sent = asyncio.run(
        _serve_stream(source(), lambda *args: observations.append(args), **kwargs)
    )
    assert [m["body"] for m in sent if m["type"] == "http.response.body"] == [
        chunk.encode() for chunk in chunks
    ] + [b""]
    return observations


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {"content": "hello\u2028world\u2029!"}}]},
        {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
        {"choices": [{"delta": {"reasoning": "thinking"}}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"name": "search"}}]}}]},
        {"choices": [{"delta": {"function_call": {"arguments": "{"}}}]},
        {"choices": [{"text": " "}]},
        {"type": "content_block_delta", "delta": {"text": "hello"}},
        {"type": "content_block_delta", "delta": {"thinking": "hmm"}},
        {"type": "content_block_delta", "delta": {"partial_json": "{"}},
        {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "search"},
        },
        {"type": "response.output_text.delta", "delta": "hi"},
        {"type": "response.custom_tool_call_input.delta", "delta": "patch"},
        {"type": "response.function_call_arguments.delta", "delta": "{"},
    ],
)
def test_first_output_skips_metadata_and_handles_coalesced_frames(payload, newline):
    prefix = ": keepalive\r\n\r\n" + _sse(
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
    )
    output = "event: generation" + newline + _sse(payload, newline)
    observations = _run_stream([prefix + output, output, "data: [DONE]\n\n"])
    assert len(observations) == 1
    assert observations[0][1] is True


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
def test_unrecognized_events_do_not_record_or_block_output(payload):
    assert not _run_stream([_sse(payload)])
    assert len(_run_stream([_sse(payload), _sse({"choices": [{"text": "ok"}]})])) == 1


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize(
    "terminal",
    [
        "data: [DONE]\n\n",
        _sse({"error": {"message": "failed"}}),
        _sse({"type": "error", "error": "failed"}),
        _sse({"type": "error"}),
    ],
)
def test_no_ttft_after_error_or_empty_completion(terminal, coalesced):
    output = _sse({"choices": [{"text": "late"}]})
    chunks = [terminal + output] if coalesced else [terminal, output]
    assert not _run_stream(chunks)


def test_malformed_event_does_not_block_later_output():
    assert (
        len(_run_stream(["data: {not json\n\n", _sse({"choices": [{"text": "ok"}]})]))
        == 1
    )


def test_many_metadata_frames_and_large_complete_output():
    prefix = (": keepalive\n\n" + "data: {}\r\n\r\n") * 100
    output = _sse({"choices": [{"text": "你好" * 65536}]}, "\r\n")
    assert len(_run_stream([prefix + output])) == 1


def test_multiline_data_is_one_generation_event():
    assert (
        len(
            _run_stream(
                ['event: chunk\r\ndata: {"choices":\r\ndata: [{"text":"ok"}]}\r\n\r\n']
            )
        )
        == 1
    )


def _metrics(exporter):
    return {
        (sample.name, sample.labels.get("streaming")): sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.endswith(("_count", "_sum"))
    }


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/completions", "/v1/messages"]
)
def test_streaming_ttft_includes_preprocessing_skips_role_and_records_once(
    monkeypatch, path, spec_version
):
    clock = [1.0]
    monkeypatch.setattr(
        "atom.entrypoints.openai.request_timing.time.perf_counter", lambda: clock[0]
    )
    exporter, request_metrics, _ = create_metrics_exporter()
    chunks = [
        _sse({"choices": [{"delta": {"role": "assistant"}}]}),
        _sse({"choices": [{"delta": {"content": "four tokens at once"}}]}),
        _sse({"choices": [{"delta": {"content": "later"}}]}) + "data: [DONE]\n\n",
    ]

    async def source():
        for when, chunk in zip((3.0, 4.0, 5.0), chunks):
            clock[0] = when
            yield chunk

    sent = asyncio.run(
        _serve_stream(
            source(),
            request_metrics.observe_time_to_first_token,
            path=path,
            spec_version=spec_version,
        )
    )
    assert [m["body"] for m in sent if m["type"] == "http.response.body"] == [
        chunk.encode() for chunk in chunks
    ] + [b""]
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
    "kwargs",
    [
        {"status": 500},
        {"path": "/health"},
        {"method": "GET"},
        {"media_type": "application/json"},
    ],
)
def test_failed_or_unrelated_responses_produce_no_sample(kwargs):
    assert not _run_stream(
        [_sse({"choices": [{"text": "not a valid generation"}]})], **kwargs
    )


def test_logging_and_ttft_share_parsing_and_keep_all_events(monkeypatch):
    chunks = [_sse({"choices": [{"text": text}]}) for text in ("first", "later")]
    chunks.append("data: [DONE]\n\n")
    parsed = []
    written = []
    original_loads = json.loads

    def loads(data):
        parsed.append(data)
        return original_loads(data)

    class Recorder:
        @staticmethod
        def info(line):
            written.append(original_loads(line))

    monkeypatch.setattr(api_server, "_request_logger", Recorder)
    monkeypatch.setattr(api_server.json, "loads", loads)
    assert len(_run_stream(chunks)) == 1
    assert parsed == [chunk[6:-2] for chunk in chunks[:2]]
    assert [event["type"] for event in written] == [
        "stream_chunk",
        "stream_chunk",
        "stream_done",
    ]


def test_logging_disabled_stops_parsing_after_first_output(monkeypatch):
    chunks = [_sse({"choices": [{"text": text}]}) for text in ("first", "later")]
    parsed = []
    original_loads = json.loads

    def loads(data):
        parsed.append(data)
        return original_loads(data)

    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server.json, "loads", loads)
    assert len(_run_stream(["".join(chunks), *chunks])) == 1
    assert parsed == [chunks[0][6:-2]]


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
def test_concurrent_streams_keep_their_own_start_time(monkeypatch, spec_version):
    clock = [0.0]
    monkeypatch.setattr(
        "atom.entrypoints.openai.request_timing.time.perf_counter", lambda: clock[0]
    )
    observations = []

    async def run():
        ready = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def source(index):
            ready[index].set()
            await release[index].wait()
            yield _sse({"choices": [{"text": "ok"}]})

        async def serve(index):
            await _serve_stream(
                source(index),
                lambda *args: observations.append(args),
                spec_version=spec_version,
            )

        first = asyncio.create_task(serve(0))
        await ready[0].wait()
        clock[0] = 1.0
        second = asyncio.create_task(serve(1))
        await ready[1].wait()
        clock[0] = 10.0
        release[1].set()
        await second
        clock[0] = 20.0
        release[0].set()
        await first
        record_nonstream_first_token()

    asyncio.run(run())
    assert observations == [(9.0, True), (20.0, True)]


def test_cancel_before_generated_output_does_not_record_ttft():
    observations = []

    async def run():
        waiting = asyncio.Event()

        async def source():
            yield _sse({"choices": [{"delta": {"role": "assistant"}}]})
            waiting.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            _serve_stream(source(), lambda *args: observations.append(args))
        )
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record_nonstream_first_token()

    asyncio.run(run())
    assert not observations
    assert longest_silence_seconds() == 0.0
