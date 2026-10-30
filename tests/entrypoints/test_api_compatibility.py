# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Protocol regressions through real adapters and ASGI routes, without a GPU."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoder_adapters import build_message_encoder_adapter
from atom.entrypoints.openai.serving_responses import (
    ResponseStreamEmitter,
    stream_failure_frames,
)
from atom.entrypoints.openai.sse import iter_sse_data

TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object"}},
}


def events(frames):
    return [
        json.loads(data)
        for frame in frames
        for data in iter_sse_data(frame)
        if data != "[DONE]"
    ]


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(
        templates=[],
        generations=0,
        chunks=[
            {
                "text": "answer",
                "token_ids": [1],
                "finished": True,
                "finish_reason": "eos",
            }
        ],
        finish_reason="eos",
        cleanups=[],
    )

    def encode(messages, **kwargs):
        state.templates.append((messages, kwargs))
        return "prompt"

    adapter = build_message_encoder_adapter(
        "encoding", encode, model_type="deepseek_v41"
    )
    for name, value in {
        "custom_message_encoder": adapter,
        "reasoning_toggle": ("thinking_mode", "chat", "thinking"),
        "default_chat_template_kwargs": {},
        "model_name": "renamed-model",
        "model_starts_in_reasoning": False,
        "reasoning_dialect": None,
        "tool_call_parser_cls": None,
        "_request_logger": None,
        "tokenizer": SimpleNamespace(
            encode=lambda *a, **kw: [1, 2], decode=lambda *a, **kw: "text"
        ),
        "engine": SimpleNamespace(
            config=SimpleNamespace(
                generation_config=SimpleNamespace(), max_model_len=8192
            )
        ),
    }.items():
        monkeypatch.setattr(api_server, name, value)

    async def generate(*args, **kwargs):
        state.generations += 1
        yield {
            "text": "answer",
            "finish_reason": state.finish_reason,
            "num_cached_tokens": 0,
            "num_tokens_input": 2,
            "num_tokens_output": 1,
            "ttft": 0.01,
            "tpot": 0.01,
            "latency": 0.02,
        }

    class Collector:
        def __init__(self):
            self.items = iter(state.chunks)

        async def get(self):
            item = next(self.items)
            if isinstance(item, Exception):
                raise item
            return item

    async def setup(*args, **kwargs):
        state.generations += 1
        return 1, Collector(), 2

    class Reasoning:
        def stream(self):
            return self

        def process(self, text):
            return (
                [("reasoning_content", text[2:])]
                if text.startswith("R:")
                else [("content", text)]
            )

        def flush(self):
            return []

        def split(self, text):
            return "", text

    monkeypatch.setattr(api_server, "generate_async", generate)
    monkeypatch.setattr(api_server, "setup_streaming_request", setup)
    monkeypatch.setattr(api_server, "reasoning_channel", lambda *a, **kw: Reasoning())
    monkeypatch.setattr(
        api_server, "cleanup_stream", lambda *a, **kw: state.cleanups.append(kw)
    )
    monkeypatch.setattr(api_server, "cleanup_request", lambda *a: None)
    return state


def post(path, body):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
@pytest.mark.parametrize(
    "effort,expected",
    [
        ("none", None),
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("minimal", "low"),
        ("medium", "high"),
        ("xhigh", "high"),
    ],
)
def test_effort_reaches_encoder(server, monkeypatch, endpoint, effort, expected):
    monkeypatch.setattr(
        api_server,
        "default_chat_template_kwargs",
        {"thinking_mode": "chat", "reasoning_effort": 20},
    )
    body = {"model": "renamed-model"}
    if endpoint == "responses":
        body.update(input="hi", reasoning={"effort": effort})
    else:
        body.update(
            messages=[{"role": "user", "content": "hi"}], reasoning_effort=effort
        )
    response = post("/v1/" + endpoint, body)
    assert response.status_code == 200, response.text
    kwargs = server.templates[-1][1]
    assert kwargs["thinking_mode"] == ("chat" if effort == "none" else "thinking")
    assert kwargs.get("reasoning_effort") == expected
    assert "thinking_effort" not in kwargs


@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
def test_invalid_effort_rejected_before_generation(server, endpoint):
    body = (
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "ultra"}
        if endpoint == "chat/completions"
        else {"input": "hi", "reasoning": {"effort": "ultra"}}
    )
    response = post("/v1/" + endpoint, body)
    assert response.status_code == 400
    assert server.generations == 0


@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "lookup"}}]
)
def test_thinking_tool_choice_uses_effective_default(server, choice):
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [TOOL],
        "tool_choice": choice,
    }
    assert post("/v1/chat/completions", body).status_code == 400
    assert server.generations == 0
    assert (
        post("/v1/chat/completions", {**body, "reasoning_effort": "none"}).status_code
        == 200
    )


def test_generic_model_keeps_existing_policy(server, monkeypatch):
    monkeypatch.setattr(
        api_server,
        "custom_message_encoder",
        build_message_encoder_adapter("other", lambda *a, **kw: "prompt"),
    )
    assert (
        post(
            "/v1/responses", {"input": "hi", "previous_response_id": "old"}
        ).status_code
        == 400
    )
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [TOOL],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("previous_response_id", "old"),
        ("conversation", {"id": "old"}),
        ("background", True),
    ],
)
def test_v41_stateless_parameters_ignored(server, field, value):
    response = post("/v1/responses", {"input": "hi", field: value})
    assert response.status_code == 200
    assert response.json()["store"] is False
    assert response.json()["previous_response_id"] is None


@pytest.mark.parametrize("stream", [False, None])
def test_stream_options_requires_stream(server, stream):
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 400
    assert server.generations == 0


@pytest.mark.parametrize("include_usage", [False, True])
def test_chat_stream_usage_is_on_final_choice(server, include_usage):
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": include_usage},
        },
    )
    assert response.status_code == 200, response.text
    assert response.text.endswith("data: [DONE]\n\n")
    chunks = events([response.text])
    assert all(c["choices"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] is not None
    assert chunks[-1]["usage"]["completion_tokens"] == 1
    for chunk in chunks[:-1]:
        assert ("usage" in chunk) == include_usage
        if include_usage:
            assert chunk["usage"] is None


def test_reasoning_is_emitted_before_completion_and_assembled_once():
    emitter = ResponseStreamEmitter("r", "model")
    frames = emitter.opening_frames()
    for text in ["think ", "more"]:
        partial = emitter.frames_for_events([("reasoning", text)])
        assert any(
            e["type"] == "response.reasoning_text.delta" for e in events(partial)
        )
        frames += partial
    frames += emitter.frames_for_events([("content", "answer")])
    frames += emitter.finish_frames()
    parsed = events(frames)
    assert "[DONE]" not in "".join(frames)
    assert [e["sequence_number"] for e in parsed] == list(range(len(parsed)))
    assert parsed[-1]["response"]["output"][0]["content"][0]["text"] == "think more"
    assert parsed[-1]["response"]["output"][1]["content"][0]["text"] == "answer"
    assert emitter.finish_frames() == []


def test_alternating_channels_have_distinct_items():
    emitter = ResponseStreamEmitter("r", "model")
    frames = emitter.frames_for_events(
        [("content", "before"), ("reasoning", "middle"), ("content", "after")]
    )
    parsed = events(frames + emitter.finish_frames())
    output = parsed[-1]["response"]["output"]
    assert [i["content"][0]["text"] for i in output] == ["before", "middle", "after"]
    assert len({i["id"] for i in output}) == 3
    assert sum(e["type"] == "response.content_part.added" for e in parsed) == 3


@pytest.mark.parametrize("stream", [False, True])
def test_length_limit_is_incomplete(server, stream):
    server.finish_reason = "max_tokens"
    server.chunks[0]["finish_reason"] = "max_tokens"
    response = post("/v1/responses", {"input": "hi", "stream": stream})
    assert response.status_code == 200
    result = events([response.text])[-1]["response"] if stream else response.json()
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["output"][-1]["status"] == "incomplete"


def test_stream_failure_closes_items_and_reports_failed(server):
    server.chunks = [
        {"text": "R:thinking", "token_ids": [1], "finished": False},
        RuntimeError("engine failed"),
    ]
    response = post("/v1/responses", {"input": "hi", "stream": True})
    parsed = events([response.text])
    assert parsed[-1]["type"] == "response.failed"
    assert parsed[-1]["response"]["error"]["message"] == "engine failed"
    assert parsed[-1]["response"]["output"][0]["status"] == "incomplete"
    assert "[DONE]" not in response.text
    assert server.cleanups[-1]["aborted"] is True


def test_failure_before_first_token_has_opening_events():
    parsed = events(
        list(
            stream_failure_frames(
                RuntimeError("bad"), ResponseStreamEmitter("r", "m"), opened=False
            )
        )
    )
    assert parsed[0]["type"] == "response.created"
    assert parsed[-1]["type"] == "response.failed"
    assert parsed[-1]["response"]["output"] == []


@pytest.mark.parametrize(
    "effort,expected", [("minimal", "low"), ("medium", "high"), ("xhigh", "high")]
)
def test_adapter_effort_aliases(effort, expected):
    adapter = build_message_encoder_adapter("encoding_dsv41", lambda messages, **kw: kw)
    for key in ("thinking_effort", "reasoning_effort"):
        assert adapter([], **{key: effort})["reasoning_effort"] == expected


def test_default_none_effort_allows_required_tool(server, monkeypatch):
    monkeypatch.setattr(
        api_server, "default_chat_template_kwargs", {"reasoning_effort": "none"}
    )
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [TOOL],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text


def test_stream_options_reject_non_boolean_usage(server):
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": "true"},
        },
    )
    assert response.status_code == 400
    assert server.generations == 0
