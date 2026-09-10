"""Request-local first-output timing for the HTTP API."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


def _has_text(fields: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(fields, dict) and any(
        isinstance(fields.get(key), str) and bool(fields[key]) for key in keys
    )


def has_generated_output(payload: Any) -> bool:
    """Recognize text, reasoning or tool output in an API generation event."""
    if not isinstance(payload, dict) or "error" in payload:
        return False
    choices = payload.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        if not isinstance(choice, dict):
            continue
        if _has_text(choice, ("text",)):
            return True
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        if _has_text(delta, ("content", "reasoning_content", "reasoning")):
            return True
        calls = delta.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        functions = [
            call.get("function", {}) for call in calls if isinstance(call, dict)
        ]
        functions.append(delta.get("function_call") or {})
        if any(_has_text(f, ("name", "arguments")) for f in functions):
            return True
    # Anthropic messages emit generation in content blocks, not choices.
    if payload.get("type") == "content_block_delta":
        delta = payload.get("delta") or {}
        return _has_text(delta, ("text", "thinking", "partial_json"))
    if payload.get("type") == "content_block_start":
        block = payload.get("content_block") or {}
        return _has_text(block, ("text", "thinking", "name"))
    return False


@dataclass
class RequestTiming:
    started_at: float
    observe: Callable[[float, bool], None]
    recorded: bool = False
    streaming_response: bool = False

    def first_output(self, *, streaming: bool) -> None:
        if not self.recorded:
            self.recorded = True
            self.observe(time.perf_counter() - self.started_at, streaming)


_request_timing: ContextVar[RequestTiming | None] = ContextVar(
    "request_timing", default=None
)


def get_stream_timing() -> RequestTiming | None:
    """Return timing for a successful SSE response, after response headers."""
    timing = _request_timing.get()
    if timing is not None and timing.streaming_response and not timing.recorded:
        return timing
    return None


def record_nonstream_first_token() -> None:
    """Record an internal token arrival; a buffered response has no SSE event."""
    timing = _request_timing.get()
    if timing is not None:
        timing.first_output(streaming=False)


class RequestTimingMiddleware:
    """Start before preprocessing and mark successful SSE responses.

    The client stream wrapper observes generated output; this middleware only
    owns request lifetime and response eligibility, without inspecting bodies.
    """

    def __init__(self, app, observe_ttft: Callable[[float, bool], None]):
        self.app = app
        self.observe_ttft = observe_ttft

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path")
            not in {"/v1/chat/completions", "/v1/completions", "/v1/messages"}
        ):
            await self.app(scope, receive, send)
            return

        timing = RequestTiming(time.perf_counter(), self.observe_ttft)
        context_token = _request_timing.set(timing)

        async def timed_send(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                is_sse = headers.get(b"content-type", b"").startswith(
                    b"text/event-stream"
                )
                timing.streaming_response = 200 <= message["status"] < 300 and is_sse
            await send(message)

        try:
            await self.app(scope, receive, timed_send)
        finally:
            _request_timing.reset(context_token)


class RequestMetrics:
    """Request latency instruments registered once for an API instance."""

    def __init__(self, registry):
        from prometheus_client import Histogram

        self._time_to_first_token = Histogram(
            "atom:time_to_first_token_seconds",
            "Local API request arrival to first output. Streaming observes the "
            "first generated SSE payload; non-streaming observes the first "
            "internal token delivery. One sample per request.",
            labelnames=("streaming",),
            buckets=(
                0.001,
                0.005,
                0.010,
                0.025,
                0.050,
                0.100,
                0.250,
                0.500,
                1.0,
                2.5,
                5.0,
                10.0,
                15.0,
                30.0,
                45.0,
                60.0,
                90.0,
                120.0,
                180.0,
                240.0,
            ),
            registry=registry,
        )

        # Expose zero-valued children before traffic so Prometheus can establish
        # a baseline for rate(). Registering labels does not record a sample.
        for streaming in ("true", "false"):
            self._time_to_first_token.labels(streaming=streaming)

    def observe_time_to_first_token(self, interval: float, streaming: bool) -> None:
        self._time_to_first_token.labels(streaming=str(streaming).lower()).observe(
            interval
        )
