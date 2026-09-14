"""Request-local first-output timing for the HTTP API."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


def _has_text(fields: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(fields, dict) and any(
        isinstance(fields.get(key), str) and bool(fields[key]) for key in keys
    )


def _has_generated_output(payload: Any) -> bool:
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
    # OpenAI Responses / Codex events carry generation in `delta`.
    if payload.get("type") in (
        "response.output_text.delta",
        "response.custom_tool_call_input.delta",
        "response.function_call_arguments.delta",
    ):
        delta = payload.get("delta")
        return isinstance(delta, str) and bool(delta)
    return False


class SSEFrames:
    """Incrementally frame SSE bytes with amortized linear scanning and copying."""

    _ENDING = re.compile(rb"\n\n|\r\n\r\n")

    def __init__(self, max_frame_bytes: int):
        self.pending = bytearray()
        self._start = 0
        self._search = 0
        self._limit = max_frame_bytes
        self.oversized = False

    def append(self, chunk: bytes) -> None:
        # Compact only after consuming at least half the buffer. Each copied
        # byte can be charged to consumed bytes, even with a large partial frame.
        if self._start and self._start >= len(self.pending) // 2:
            del self.pending[: self._start]
            self._search -= self._start
            self._start = 0
        self.pending.extend(chunk)

    def next_frame(self) -> bytes | None:
        ending = self._ENDING.search(self.pending, self._search)
        if ending is None:
            self._search = max(self._start, len(self.pending) - 3)
            self.oversized = len(self.pending) - self._start > self._limit
            return None
        if ending.start() - self._start > self._limit:
            self.oversized = True
            return None
        frame = bytes(self.pending[self._start : ending.start()])
        self._start = self._search = ending.end()
        return frame

    def clear(self) -> None:
        self.pending.clear()
        self._start = self._search = 0


class FirstOutputSSE:
    """Inspect SSE until the first generation event, without altering the stream."""

    MAX_PENDING_BYTES = 1024 * 1024

    def __init__(self):
        self._frames = SSEFrames(self.MAX_PENDING_BYTES)
        self.done = False

    def _finish(self) -> None:
        self.done = True
        self._frames.clear()

    def feed(self, chunk: bytes) -> bool:
        if self.done:
            return False
        self._frames.append(chunk)
        while (frame := self._frames.next_frame()) is not None:
            data = b"\n".join(
                line[5:].lstrip(b" ")
                for line in frame.splitlines()
                if line.startswith(b"data:")
            )
            if data == b"[DONE]":
                self._finish()
                return False
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict) and (
                "error" in payload or payload.get("type") == "error"
            ):
                self._finish()
                return False
            if _has_generated_output(payload):
                self._finish()
                return True
        if self._frames.oversized:
            self._finish()
        return False


@dataclass
class RequestTiming:
    started_at: float
    observe: Callable[[float, bool], None]
    recorded: bool = False

    def first_output(self, *, streaming: bool) -> None:
        if not self.recorded:
            self.recorded = True
            self.observe(time.perf_counter() - self.started_at, streaming)


_request_timing: ContextVar[RequestTiming | None] = ContextVar(
    "request_timing", default=None
)


def record_nonstream_first_token() -> None:
    """Record an internal token arrival; a buffered response has no SSE event."""
    timing = _request_timing.get()
    if timing is not None:
        timing.first_output(streaming=False)


class RequestTimingMiddleware:
    """Start before preprocessing; keep timing isolated across concurrent requests."""

    def __init__(self, app, observe_ttft: Callable[[float, bool], None]):
        self.app = app
        self.observe_ttft = observe_ttft

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path")
            not in {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/messages",
                "/v1/responses",
            }
        ):
            await self.app(scope, receive, send)
            return

        timing = RequestTiming(time.perf_counter(), self.observe_ttft)
        context_token = _request_timing.set(timing)
        detector = FirstOutputSSE()
        streaming = False

        async def timed_send(message):
            nonlocal streaming
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                streaming = 200 <= message["status"] < 300 and headers.get(
                    b"content-type", b""
                ).startswith(b"text/event-stream")
            elif (
                message["type"] == "http.response.body"
                and streaming
                and not timing.recorded
            ):
                if detector.feed(message.get("body", b"")):
                    timing.first_output(streaming=True)
            await send(message)

        try:
            await self.app(scope, receive, timed_send)
        finally:
            _request_timing.reset(context_token)
