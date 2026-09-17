import asyncio
import json

import pytest
from prometheus_client.parser import text_string_to_metric_families
from starlette.responses import StreamingResponse

from atom.entrypoints.openai import api_server, ttft_breakdown
from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
from atom.entrypoints.openai.request_timing import RequestTimingMiddleware

TO_CALLBACK = "atom:ttft_output_to_callback_seconds"
TO_SSE = "atom:ttft_callback_to_sse_seconds"


@pytest.fixture(autouse=True)
def _isolate_stamps():
    # The stamp map is module state shared by every request in the process.
    ttft_breakdown._first_callback_at.clear()
    yield
    ttft_breakdown._first_callback_at.clear()


def _stage(exporter):
    return {
        sample.name: sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.endswith(("_count", "_sum"))
    }


def _callback(breakdown, request_id, *, emit=1.0, at=1.5, perf=0.0, tokens=True):
    """Stand in for the engine output thread calling into the API process."""
    ttft_breakdown.mark_first_callback(
        request_id,
        scheduler_output_at=emit,
        callback_at=at,
        callback_perf=perf,
        metrics=breakdown,
        has_tokens=tokens,
    )


def test_only_the_first_chunk_of_a_request_is_observed():
    exporter, _, _, breakdown = create_metrics_exporter()
    for at in (1.5, 1.9, 2.4):
        _callback(breakdown, "req-a", at=at)
    samples = _stage(exporter)
    assert samples[f"{TO_CALLBACK}_count"] == 1
    assert samples[f"{TO_CALLBACK}_sum"] == pytest.approx(0.5)


def test_chunks_without_tokens_or_stamp_are_not_observed():
    exporter, _, _, breakdown = create_metrics_exporter()
    _callback(breakdown, "req-empty", at=2.0, tokens=False)
    _callback(breakdown, "req-unstamped", emit=None, at=2.0)
    assert _stage(exporter)[f"{TO_CALLBACK}_count"] == 0
    # An empty chunk must not spend the request's single observation.
    assert "req-empty" not in ttft_breakdown._first_callback_at


def test_cleanup_request_drops_the_stamp():
    exporter, _, _, breakdown = create_metrics_exporter()
    _callback(breakdown, "req-b")
    assert set(ttft_breakdown._first_callback_at) == {"req-b"}

    api_server.cleanup_request("req-b")
    assert ttft_breakdown._first_callback_at == {}

    # Proof the entry is really gone, not merely hidden: the same id observes again.
    _callback(breakdown, "req-b", emit=2.0, at=2.5)
    assert _stage(exporter)[f"{TO_CALLBACK}_count"] == 2


def test_stamps_do_not_accumulate_across_requests():
    _, _, _, breakdown = create_metrics_exporter()
    for i in range(64):
        _callback(breakdown, f"req-{i}")
        api_server.cleanup_request(f"req-{i}")
    assert ttft_breakdown._first_callback_at == {}


async def _serve(source, observe, request_id):
    sent = []

    async def app(scope, receive, send):
        await StreamingResponse(
            api_server._client_stream(source, request_id),
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


def test_callback_to_sse_spans_the_output_thread_handoff(monkeypatch):
    exporter, request_metrics, _, breakdown = create_metrics_exporter()
    monkeypatch.setattr(api_server, "_ttft_breakdown", breakdown)
    monkeypatch.setattr(api_server, "_request_logger", None)
    clock = [10.0]
    monkeypatch.setattr(
        "atom.entrypoints.openai.request_timing.time.perf_counter", lambda: clock[0]
    )

    # Production order: the output thread stamps before the frame can be yielded.
    _callback(breakdown, "req-sse", perf=10.5)

    async def source():
        clock[0] = 10.75
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "hi"}}]}
        ) + "\n\n"

    asyncio.run(
        _serve(source(), request_metrics.observe_time_to_first_token, "req-sse")
    )

    samples = _stage(exporter)
    assert samples[f"{TO_SSE}_count"] == 1
    assert samples[f"{TO_SSE}_sum"] == pytest.approx(0.25)


def test_callback_to_sse_is_skipped_when_no_callback_stamped(monkeypatch):
    exporter, request_metrics, _, breakdown = create_metrics_exporter()
    monkeypatch.setattr(api_server, "_ttft_breakdown", breakdown)
    monkeypatch.setattr(api_server, "_request_logger", None)

    async def source():
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "hi"}}]}
        ) + "\n\n"

    asyncio.run(
        _serve(source(), request_metrics.observe_time_to_first_token, "req-nostamp")
    )
    assert _stage(exporter)[f"{TO_SSE}_count"] == 0


def test_api_preprocess_subdivides_observe_zero_and_positive():
    exporter, _, _, breakdown = create_metrics_exporter()
    breakdown.observe_api_body_parse(0.01)
    breakdown.observe_api_chat_template(0.0)  # PD prompt_token_ids reuse
    breakdown.observe_api_tokenize(0.0)
    breakdown.observe_api_preprocess_wait(0.002)
    breakdown.observe_api_detokenize_chunk(0.0002)
    samples = _stage(exporter)
    assert samples["atom:api_body_parse_seconds_count"] == 1
    assert samples["atom:api_body_parse_seconds_sum"] == pytest.approx(0.01)
    assert samples["atom:api_chat_template_seconds_count"] == 1
    assert samples["atom:api_chat_template_seconds_sum"] == 0
    assert samples["atom:api_tokenize_seconds_count"] == 1
    assert samples["atom:api_tokenize_seconds_sum"] == 0
    assert samples["atom:api_preprocess_wait_seconds_sum"] == pytest.approx(0.002)
    assert samples["atom:api_detokenize_chunk_seconds_sum"] == pytest.approx(0.0002)
