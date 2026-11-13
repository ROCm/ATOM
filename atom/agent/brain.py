# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat clients for the agent's brain: OpenAI chat completions and Anthropic
Messages, streaming, with tool calls and reasoning.

Standard library only. The OpenAI dialect works against ATOM, vLLM, SGLang
and most gateways; the Anthropic dialect against ATOM's ``/v1/messages`` and
Anthropic's API. Both produce the same :class:`Reply`, and the session keeps
its history in OpenAI shape -- the Anthropic client converts on the way out.

Failures are typed: :class:`BrainError` says whether a retry can help
(connection refused/reset, timeouts, stalled streams, HTTP 408/409/429/5xx).
:class:`Brain` retries those with backoff and then hands over to its
``fallbacks``.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504, 529}
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> dict:
        if not self.arguments.strip():
            return {}
        try:
            value = json.loads(self.arguments)
        except ValueError as e:
            raise ValueError(f"tool arguments are not JSON: {e}") from e
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")  # noqa: TRY004
        return value


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None
    ttft_s: float | None = None
    total_s: float = 0.0
    chunk_times: list[float] = field(default_factory=list)


class BrainError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _post(url: str, body: dict, headers: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:1500]
        raise BrainError(
            f"HTTP {e.code} from {url}: {detail}", retryable=e.code in RETRYABLE_HTTP
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise BrainError(f"cannot reach {url}: {e}", retryable=True) from e


def _sse_events(resp):
    """(event, data) pairs from a text/event-stream response.

    The socket timeout set at connect applies to every read, so a server
    that stops sending mid-stream raises instead of hanging forever.
    """
    event = None
    try:
        for raw in resp:
            line = raw.decode(errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                yield event, line[5:].strip()
            elif not line:
                event = None
    except (TimeoutError, socket.timeout) as e:
        raise BrainError("the stream stalled (no data within the timeout)", True) from e
    except (ConnectionError, OSError) as e:
        raise BrainError(f"the stream broke: {e}", retryable=True) from e


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    extra: dict | None = None,
    on_content: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    timeout: float = 300,
) -> Reply:
    """One streamed OpenAI-style chat completion, assembled into a Reply."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra or {})
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    reply = Reply()
    calls: dict[int, dict] = {}
    done = False
    t0 = time.perf_counter()
    url = f"{base_url.rstrip('/')}/chat/completions"
    with _post(url, body, headers, timeout) as r:
        for _, data in _sse_events(r):
            if data == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("error"):
                raise BrainError(f"server error: {chunk['error']}")
            if chunk.get("usage"):
                reply.usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                got = False
                text = delta.get("content")
                if text:
                    reply.content += text
                    got = True
                    if on_content:
                        on_content(text)
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    reply.reasoning += thought
                    got = True
                    if on_reasoning:
                        on_reasoning(thought)
                for tc in delta.get("tool_calls") or []:
                    got = True
                    slot = calls.setdefault(
                        tc.get("index", len(calls)), {"id": "", "name": "", "args": ""}
                    )
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
                if got:
                    now = time.perf_counter() - t0
                    reply.chunk_times.append(now)
                    if reply.ttft_s is None:
                        reply.ttft_s = now
                if choice.get("finish_reason"):
                    reply.finish_reason = choice["finish_reason"]
    if not done and reply.finish_reason is None:
        raise BrainError("the stream ended before the reply finished", retryable=True)
    reply.total_s = time.perf_counter() - t0
    reply.tool_calls = [
        ToolCall(c["id"] or f"call_{i}", c["name"], c["args"])
        for i, c in sorted(calls.items())
        if c["name"]
    ]
    return reply


# --------------------------------------------------------------------------
# Anthropic Messages
# --------------------------------------------------------------------------


def to_anthropic(messages: list[dict], tools: list[dict] | None):
    """OpenAI-shaped history and tools -> (system, messages, tools) for
    /v1/messages. Reasoning is not sent back: Anthropic thinking blocks need
    a signature only the original response carries."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []

    def add(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:  # strict user/assistant alternation
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            add("user", [{"type": "text", "text": m.get("content") or ""}])
        elif role == "assistant":
            blocks = []
            if (m.get("content") or "").strip():
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:
                    args = {"_unparsed": c["function"].get("arguments")}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "input": args if isinstance(args, dict) else {"value": args},
                    }
                )
            add("assistant", blocks or [{"type": "text", "text": "(no content)"}])
        elif role == "tool":
            add(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": m["tool_call_id"],
                        "content": m.get("content") or "",
                    }
                ],
            )
    a_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters") or {"type": "object"},
        }
        for t in tools or []
    ]
    return system, out, a_tools


def messages_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    api_key: str | None = None,
    max_tokens: int = 8192,
    extra: dict | None = None,
    on_content: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    timeout: float = 300,
) -> Reply:
    """One streamed Anthropic Messages call, assembled into a Reply."""
    system, a_messages, a_tools = to_anthropic(messages, tools)
    body: dict = {
        "model": model,
        "messages": a_messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if system:
        body["system"] = system
    if a_tools:
        body["tools"] = a_tools
    body.update(extra or {})
    headers = {"anthropic-version": ANTHROPIC_VERSION, "Accept": "text/event-stream"}
    if api_key:
        headers["x-api-key"] = api_key
    reply = Reply()
    blocks: dict[int, dict] = {}
    usage: dict = {}
    stopped = False
    t0 = time.perf_counter()
    url = f"{base_url.rstrip('/')}/v1/messages"
    with _post(url, body, headers, timeout) as r:
        for event, data in _sse_events(r):
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            kind = ev.get("type") or event
            if kind == "error":
                err = ev.get("error") or {}
                raise BrainError(
                    f"server error: {err}",
                    retryable=err.get("type") in ("overloaded_error", "api_error"),
                )
            if kind == "message_start":
                usage["prompt_tokens"] = (
                    (ev.get("message") or {}).get("usage") or {}
                ).get("input_tokens")
            elif kind == "content_block_start":
                cb = ev.get("content_block") or {}
                blocks[ev.get("index", len(blocks))] = {
                    "type": cb.get("type"),
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "json": "",
                }
            elif kind == "content_block_delta":
                d = ev.get("delta") or {}
                slot = blocks.setdefault(ev.get("index", 0), {"type": None, "json": ""})
                got = True
                if d.get("type") == "text_delta":
                    reply.content += d.get("text", "")
                    if on_content:
                        on_content(d.get("text", ""))
                elif d.get("type") == "thinking_delta":
                    reply.reasoning += d.get("thinking", "")
                    if on_reasoning:
                        on_reasoning(d.get("thinking", ""))
                elif d.get("type") == "input_json_delta":
                    slot["json"] += d.get("partial_json", "")
                else:
                    got = False  # signature_delta and friends
                if got:
                    now = time.perf_counter() - t0
                    reply.chunk_times.append(now)
                    if reply.ttft_s is None:
                        reply.ttft_s = now
            elif kind == "message_delta":
                stop = (ev.get("delta") or {}).get("stop_reason")
                reply.finish_reason = {
                    "tool_use": "tool_calls",
                    "end_turn": "stop",
                }.get(stop, stop)
                usage["completion_tokens"] = (ev.get("usage") or {}).get(
                    "output_tokens"
                )
            elif kind == "message_stop":
                stopped = True
                break
    if not stopped and reply.finish_reason is None:
        raise BrainError("the stream ended before the reply finished", retryable=True)
    reply.total_s = time.perf_counter() - t0
    reply.usage = usage
    reply.tool_calls = [
        ToolCall(b["id"] or f"toolu_{i}", b["name"], b["json"] or "{}")
        for i, b in sorted(blocks.items())
        if b.get("type") == "tool_use"
    ]
    return reply


# --------------------------------------------------------------------------
# The brain
# --------------------------------------------------------------------------


def list_models(
    base_url: str, api_key: str | None = None, api: str = "openai", timeout: float = 5
) -> list[str]:
    headers = {}
    if api == "anthropic":
        url = f"{base_url.rstrip('/')}/v1/models"
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if api_key:
            headers["x-api-key"] = api_key
    else:
        url = f"{base_url.rstrip('/')}/models"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


@dataclass
class Brain:
    """The LLM driving the agent, with retries and fallbacks."""

    base_url: str
    model: str
    api_key: str | None = None
    extra: dict = field(default_factory=dict)
    label: str = ""
    api: str = "openai"  # or "anthropic"
    max_retries: int = 3
    timeout: float = 300  # per read: a stream silent this long is retried
    max_tokens: int = 8192  # Anthropic requires one; OpenAI leaves it unset
    fallbacks: list[Brain] = field(default_factory=list)
    on_retry: Callable[[str], None] | None = None
    backoff_s: tuple[float, ...] = (2.0, 5.0, 10.0)

    def chat(self, messages, tools=None, on_content=None, on_reasoning=None) -> Reply:
        attempt = 0
        while True:
            try:
                if self.api == "anthropic":
                    return messages_completion(
                        self.base_url,
                        self.model,
                        messages,
                        tools=tools,
                        api_key=self.api_key,
                        max_tokens=self.max_tokens,
                        extra=self.extra,
                        on_content=on_content,
                        on_reasoning=on_reasoning,
                        timeout=self.timeout,
                    )
                return chat_completion(
                    self.base_url,
                    self.model,
                    messages,
                    tools=tools,
                    api_key=self.api_key,
                    extra=self.extra,
                    on_content=on_content,
                    on_reasoning=on_reasoning,
                    timeout=self.timeout,
                )
            except BrainError as e:
                if not e.retryable or attempt >= self.max_retries:
                    raise
                wait = self.backoff_s[min(attempt, len(self.backoff_s) - 1)]
                attempt += 1
                if self.on_retry:
                    self.on_retry(
                        f"brain error ({e}); retry {attempt}/{self.max_retries} in {wait:g}s"
                    )
                time.sleep(wait)

    def describe(self) -> str:
        api = "" if self.api == "openai" else f" [{self.api}]"
        return f"{self.model} @ {self.base_url}{api}" + (
            f" ({self.label})" if self.label else ""
        )


def connect(
    base_url: str,
    model: str | None = None,
    api_key: str | None = None,
    api: str = "openai",
    **kw,
) -> Brain | None:
    base_url = base_url.rstrip("/")
    if api == "anthropic":
        base_url = base_url.removesuffix("/v1")
    elif not base_url.endswith("/v1"):
        base_url += "/v1"
    models = list_models(base_url, api_key, api)
    if not models and not model:
        return None
    return Brain(base_url, model or models[0], api_key, api=api, **kw)


def discover(candidate_ports: list[int], api_key: str | None = None) -> Brain | None:
    """Explicit settings first, then any OpenAI-compatible server on this box."""
    url = os.environ.get("ATOM_CHAT_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if url:
        return connect(
            url,
            os.environ.get("ATOM_CHAT_MODEL"),
            api_key
            or os.environ.get("ATOM_CHAT_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            api=os.environ.get("ATOM_CHAT_API", "openai"),
            label="from environment",
        )
    for port in dict.fromkeys(candidate_ports):
        brain = connect(
            f"http://127.0.0.1:{port}", None, api_key, label=f"local :{port}"
        )
        if brain:
            return brain
    return None
