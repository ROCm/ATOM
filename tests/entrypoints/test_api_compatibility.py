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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("effort", ["none", "low"])
def test_flash_ignores_hosted_web_search_but_returns_answer(server, stream, effort):
    response = post(
        "/v1/responses",
        {
            "model": "renamed-model",
            "input": "Say answer",
            "tools": [{"type": "web_search"}],
            "reasoning": {"effort": effort},
            "stream": stream,
        },
    )
    assert response.status_code == 200
    assert "answer" in response.text
    assert "web_search_call" not in response.text
    if stream:
        assert "response.completed" in response.text
        assert "[DONE]" not in response.text


def test_models_metadata_is_deployment_scoped(server, monkeypatch):
    monkeypatch.setattr(api_server.engine.config, "method", "dspark", raising=False)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return await client.get("/v1/models")

    response = asyncio.run(run())
    card = response.json()["data"][0]
    assert card["id"] == "renamed-model"
    assert card["context_window"] == card["max_output_tokens"] == 8192
    assert card["input_modalities"] == ["text"]
    assert card["effort"]["default_level"] == "high"


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


@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
@pytest.mark.parametrize("default_mode", ["chat", "thinking"])
@pytest.mark.parametrize("effort", [None, 37])
def test_integer_and_absent_effort_preserve_template_default(
    server, monkeypatch, endpoint, default_mode, effort
):
    monkeypatch.setattr(
        api_server,
        "default_chat_template_kwargs",
        {"thinking_mode": default_mode, "reasoning_effort": 20},
    )
    if endpoint == "responses":
        body = {"input": "hi", "reasoning": {"effort": effort}}
    else:
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": effort,
        }
    response = post("/v1/" + endpoint, body)
    assert response.status_code == 200, response.text
    kwargs = server.templates[-1][1]
    assert kwargs["thinking_mode"] == default_mode
    assert kwargs["reasoning_effort"] == (20 if effort is None else effort)


@pytest.mark.parametrize(
    "thinking,effort,mode,expected",
    [
        ({"type": "disabled"}, "high", "chat", None),
        ({"type": "enabled"}, "none", "chat", None),
        ({"type": "enabled", "effort": "max"}, "low", "thinking", "max"),
    ],
)
def test_chat_explicit_controls_override_request_template(
    server, thinking, effort, mode, expected
):
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": thinking,
            "reasoning_effort": effort,
            "chat_template_kwargs": {
                "thinking_mode": "thinking",
                "reasoning_effort": 20,
            },
        },
    )
    assert response.status_code == 200, response.text
    kwargs = server.templates[-1][1]
    assert kwargs["thinking_mode"] == mode
    assert kwargs.get("reasoning_effort") == expected


@pytest.mark.parametrize("stream", [False, True])
def test_responses_ignored_metadata_is_not_echoed(server, stream):
    response = post(
        "/v1/responses",
        {
            "input": "hi",
            "stream": stream,
            "metadata": {"ignored": "value"},
            "parallel_tool_calls": False,
            "store": True,
        },
    )
    assert response.status_code == 200, response.text
    result = events([response.text])[-1]["response"] if stream else response.json()
    assert result["metadata"] == {}
    assert result["parallel_tool_calls"] is True
    assert result["store"] is False


@pytest.mark.parametrize(
    "field",
    [
        "include",
        "prompt",
        "truncation",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "context_management",
        "stream_options",
        "max_tool_calls",
    ],
)
def test_responses_documented_unsupported_fields_are_ignored(server, field):
    response = post("/v1/responses", {"input": "hi", field: {"ignored": True}})
    assert response.status_code == 200, response.text
    assert field not in response.json()


@pytest.mark.parametrize("endpoint", ["responses", "messages"])
def test_chat_thinking_tool_restriction_does_not_leak_to_other_endpoints(
    server, endpoint
):
    body = {"model": "renamed-model"}
    if endpoint == "responses":
        body.update(
            input="hi",
            reasoning={"effort": "low"},
            tools=[
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
            ],
            tool_choice="required",
        )
    else:
        body.update(
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled"},
            tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
            tool_choice={"type": "any"},
        )
    response = post("/v1/" + endpoint, body)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("low", "low"),
        ("minimal", "low"),
        ("medium", "high"),
        ("xhigh", "high"),
        ("max", "max"),
        (37, 37),
        ("none", None),
    ],
)
def test_messages_output_effort_reaches_encoder(server, effort, expected):
    response = post(
        "/v1/messages",
        {
            "model": "renamed-model",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1234},
            "output_config": {"effort": effort},
        },
    )
    assert response.status_code == 200, response.text
    kwargs = server.templates[-1][1]
    assert kwargs["thinking_mode"] == ("chat" if effort == "none" else "thinking")
    assert kwargs.get("reasoning_effort") == expected
    assert "budget_tokens" not in kwargs


@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
def test_messages_effort_does_not_opt_into_thinking(server, monkeypatch, thinking):
    monkeypatch.setattr(
        api_server, "default_chat_template_kwargs", {"thinking_mode": "chat"}
    )
    body = {
        "model": "renamed-model",
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "high"},
    }
    if thinking is not None:
        body["thinking"] = thinking
    response = post("/v1/messages", body)
    assert response.status_code == 200, response.text
    assert server.templates[-1][1]["thinking_mode"] == "chat"


def test_messages_invalid_output_effort_uses_anthropic_error(server):
    response = post(
        "/v1/messages",
        {
            "model": "renamed-model",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "ultra"},
        },
    )
    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert server.generations == 0


def test_generic_chat_keeps_its_stream_contract_and_template_defaults(
    server, monkeypatch
):
    original = api_server.custom_message_encoder
    monkeypatch.setattr(
        api_server,
        "custom_message_encoder",
        build_message_encoder_adapter("other", original.encode),
    )
    monkeypatch.setattr(
        api_server, "reasoning_toggle", ("enable_thinking", False, True)
    )
    monkeypatch.setattr(
        api_server, "default_chat_template_kwargs", {"enable_thinking": False}
    )
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "stream": True,
        },
    )
    assert response.status_code == 200, response.text
    assert server.templates[-1][1]["enable_thinking"] is False
    assert events([response.text])[-1]["choices"] == []


def test_tool_validation_does_not_mutate_server_defaults(server, monkeypatch):
    defaults = {"thinking_effort": "none", "reasoning_effort": 20}
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", defaults)
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [TOOL],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    assert defaults == {"thinking_effort": "none", "reasoning_effort": 20}


@pytest.mark.parametrize("endpoint", ["chat/completions", "messages"])
@pytest.mark.parametrize("key", ["thinking_effort", "reasoning_effort"])
def test_explicit_enable_overrides_inherited_none_effort(
    server, monkeypatch, endpoint, key
):
    monkeypatch.setattr(
        api_server,
        "default_chat_template_kwargs",
        {"thinking_mode": "chat", key: "none"},
    )
    response = post(
        "/v1/" + endpoint,
        {
            "model": "renamed-model",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled"},
        },
    )
    assert response.status_code == 200, response.text
    assert server.templates[-1][1]["thinking_mode"] == "thinking"
    assert server.templates[-1][1].get("reasoning_effort") != "none"


@pytest.mark.parametrize(
    "body",
    [
        {"stream_options": {"include_usage": True}},
        {"reasoning_effort": "ultra"},
        {"reasoning_effort": "low", "tools": [TOOL], "tool_choice": "required"},
        {
            "reasoning_effort": "low",
            "tools": [TOOL],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        },
    ],
)
def test_chat_validation_errors_have_openai_envelope(server, body):
    response = post(
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], **body},
    )
    assert response.status_code == 400
    assert "detail" not in response.json()
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["message"]
    assert server.generations == 0


def test_chat_model_validation_uses_same_error_envelope(server):
    response = post(
        "/v1/chat/completions",
        {
            "model": "different-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert "does not match" in response.json()["error"]["message"]
    assert server.generations == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["different-model", "", "renamed-model-extra"])
def test_responses_rejects_unknown_model_before_render_or_enqueue(
    server, stream, model
):
    response = post("/v1/responses", {"model": model, "input": "hi", "stream": stream})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "does not match" in response.json()["error"]["message"]
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert server.templates == []
    assert server.generations == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["renamed-model", "renamed-model/"])
def test_responses_accepts_served_alias_like_chat(server, stream, model):
    response = post("/v1/responses", {"model": model, "input": "hi", "stream": stream})
    assert response.status_code == 200, response.text
    assert server.generations == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["different-model", "", "renamed-model-extra"])
def test_messages_rejects_unknown_model_before_render_or_enqueue(server, stream, model):
    response = post(
        "/v1/messages",
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": stream,
        },
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "does not match" in response.json()["error"]["message"]
    assert server.templates == []
    assert server.generations == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["renamed-model", "renamed-model/"])
def test_messages_accepts_served_alias_like_chat(server, stream, model):
    response = post(
        "/v1/messages",
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    assert server.generations == 1


def test_http_error_handler_preserves_status_headers_and_other_routes():
    from fastapi import HTTPException
    from starlette.requests import Request

    async def run(path):
        request = Request(
            {"type": "http", "method": "POST", "path": path, "headers": []}
        )
        return await api_server.openai_http_error_handler(
            request,
            HTTPException(
                status_code=429, detail="slow down", headers={"Retry-After": "5"}
            ),
        )

    response = asyncio.run(run("/v1/chat/completions"))
    assert response.status_code == 429 and response.headers["Retry-After"] == "5"
    assert json.loads(response.body)["error"]["message"] == "slow down"
    assert json.loads(asyncio.run(run("/server_info")).body) == {"detail": "slow down"}


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("chat/completions", {"messages": "invalid"}),
        ("completions", {"prompt": {}, "max_tokens": "invalid"}),
        ("responses", {"input": "hi", "max_output_tokens": "invalid"}),
        ("messages", {"messages": "invalid", "max_tokens": 16}),
    ],
)
def test_pydantic_errors_use_endpoint_envelope_before_generation(
    server, endpoint, body
):
    response = post("/v1/" + endpoint, {"model": "renamed-model", **body})
    # Responses validates its rewritten body inside the endpoint, preserving
    # its existing 400; FastAPI's typed-body validation retains 422 elsewhere.
    expected_status = 400 if endpoint == "responses" else 422
    assert response.status_code == expected_status
    payload = response.json()
    assert "detail" not in payload
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"]
    if endpoint == "messages":
        assert payload["type"] == "error"
    else:
        assert payload["error"]["code"] == expected_status
    assert server.generations == 0


def test_malformed_json_uses_openai_validation_envelope(server):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                content='{"messages":',
                headers={"Content-Type": "application/json"},
            )

    response = asyncio.run(run())
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert server.generations == 0


def test_validation_handler_preserves_non_generation_routes():
    from fastapi.exceptions import RequestValidationError
    from starlette.requests import Request

    errors = [
        {
            "type": "missing",
            "loc": ("query", "value"),
            "msg": "Field required",
            "input": {},
        }
    ]
    request = Request(
        {"type": "http", "method": "GET", "path": "/server_info", "headers": []}
    )
    response = asyncio.run(
        api_server.request_validation_error_handler(
            request, RequestValidationError(errors)
        )
    )
    assert response.status_code == 422
    assert json.loads(response.body)["detail"][0]["loc"] == ["query", "value"]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("usage_on_finish", [False, True])
def test_stream_serializes_usage_once_and_preserves_every_finish(
    n, include_usage, usage_on_finish, monkeypatch
):
    from atom.entrypoints.openai import serving_chat, sse

    chunks = []
    for i in reversed(range(n)):
        chunk = {
            "text": "answer",
            "token_ids": [1, 2],
            "finished": True,
            "finish_reason": "length",
            "kv_transfer_params": {"id": "handoff"},
        }
        chunks.append((i, chunk) if n > 1 else chunk)

    class Collector:
        def __init__(self):
            self.items = iter(chunks)

        async def get(self):
            return next(self.items)

    serialized = []
    original = sse.data_frame

    def record(payload):
        serialized.append(payload)
        return original(payload)

    monkeypatch.setattr(serving_chat, "data_frame", record)
    cleanups = []

    async def run():
        fn = (
            serving_chat.stream_chat_response_fanout
            if n > 1
            else serving_chat.stream_chat_response
        )
        stream = fn(
            "req",
            "model",
            Collector(),
            list(range(n)) if n > 1 else 0,
            5,
            lambda *a, **kw: cleanups.append(kw),
            lambda *a: None,
            usage_on_finish=usage_on_finish,
            include_usage=include_usage,
        )
        return [frame async for frame in stream]

    frames = asyncio.run(run())
    payloads = events(frames)
    # Every output payload is serialized exactly once; no parse/re-encode wrapper.
    assert payloads == serialized
    finishes = [
        p for p in payloads if p["choices"] and p["choices"][0]["finish_reason"]
    ]
    assert [p["choices"][0]["index"] for p in finishes] == list(range(n))
    assert all(p["choices"][0]["finish_reason"] == "length" for p in finishes)
    last = payloads[-1]
    assert last["usage"]["completion_tokens"] == n * 2
    assert last["usage"]["prompt_tokens"] == 5
    assert last["kv_transfer_params"] == {"id": "handoff"}
    if usage_on_finish:
        assert all(p["choices"] for p in payloads)
        assert last is finishes[-1]
        assert all(
            (p.get("usage") is None and ("usage" in p) == include_usage)
            for p in payloads[:-1]
        )
    else:
        assert last["choices"] == []
        assert all("usage" not in p for p in payloads[:-1])
    assert len(cleanups) == n and all(c["aborted"] is False for c in cleanups)
    assert "".join(frames).endswith("data: [DONE]\n\n")


def test_direct_stream_cleanup_on_disconnect():
    from atom.entrypoints.openai.serving_chat import stream_chat_response

    cleanups = []

    class Collector:
        async def get(self):
            return {"text": "first", "token_ids": [1]}

    async def run():
        stream = stream_chat_response(
            "req",
            "model",
            Collector(),
            1,
            5,
            lambda *a, **kw: cleanups.append(kw),
            lambda *a: None,
            usage_on_finish=True,
            include_usage=True,
        )
        await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert cleanups == [{"aborted": True}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses", "messages"])
def test_v41_alias_without_encoder_retains_generic_template_contract(
    server, monkeypatch, stream, endpoint
):
    # A served alias/config is not an encoder adapter. Without one, retain the
    # established generic Jinja behavior rather than guessing V4.1 controls.
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "model_name", "deepseek-ai/DeepSeek-V4.1-Flash")
    api_server.engine.config.hf_config = SimpleNamespace(model_type="deepseek_v41")

    def render(messages, **kwargs):
        server.templates.append((messages, kwargs))
        return "prompt"

    monkeypatch.setattr(
        api_server.tokenizer, "apply_chat_template", render, raising=False
    )
    body = {"model": api_server.model_name, "stream": stream}
    if endpoint == "responses":
        body.update(input="hi", reasoning={"effort": "medium"})
    else:
        body.update(messages=[{"role": "user", "content": "hi"}], max_tokens=16)
        if endpoint == "chat/completions":
            body["reasoning_effort"] = "medium"
        else:
            body["thinking"] = {"type": "disabled"}
    response = post("/v1/" + endpoint, body)
    assert response.status_code == 200, response.text
    assert server.templates[-1][1]["tokenize"] is False
    if endpoint == "responses":
        assert server.templates[-1][1]["thinking_effort"] == "medium"
    elif endpoint == "chat/completions":
        # "medium" is a V4.1 alias, not a legacy Chat template effort.
        assert "thinking_effort" not in server.templates[-1][1]
    if endpoint == "chat/completions" and stream:
        assert events([response.text])[-1]["choices"] == []


@pytest.mark.parametrize("endpoint", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("key", ["reasoning_effort", "thinking_effort"])
@pytest.mark.parametrize("stream", [False, True])
def test_effort_only_preserves_inherited_none(
    server, monkeypatch, endpoint, key, stream
):
    defaults = {key: "none"}
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", defaults)
    body = {"model": "renamed-model", "stream": stream}
    if endpoint == "responses":
        body.update(input="hi", reasoning={"effort": 37})
    else:
        body["messages"] = [{"role": "user", "content": "hi"}]
        if endpoint == "messages":
            body.update(max_tokens=16, output_config={"effort": "high"})
        else:
            body["reasoning_effort"] = 37
    response = post("/v1/" + endpoint, body)
    assert response.status_code == 200, response.text
    kwargs = server.templates[-1][1]
    assert kwargs["thinking_mode"] == "chat"
    assert kwargs["reasoning_effort"] == ("high" if endpoint == "messages" else 37)
    assert defaults == {key: "none"}


@pytest.mark.parametrize(
    "reason", ["unschedulable: insufficient capacity", "rejected", "aborted"]
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_responses_engine_failure_is_not_success(
    server, monkeypatch, reason, stream, partial
):
    server.chunks = [
        {
            "text": "partial" if partial else "",
            "token_ids": [1] if partial else [],
            "finished": True,
            "finish_reason": reason,
        }
    ]

    async def generate(*args, **kwargs):
        yield {
            "text": "partial" if partial else "",
            "finish_reason": reason,
            "num_cached_tokens": 0,
        }

    monkeypatch.setattr(api_server, "generate_async", generate)
    response = post("/v1/responses", {"input": "hi", "stream": stream})
    assert response.status_code == 200
    parsed = events([response.text]) if stream else []
    result = parsed[-1]["response"] if stream else response.json()
    assert result["status"] == "failed"
    assert result["error"] == {"code": reason.split(":", 1)[0], "message": reason}
    assert result["incomplete_details"] is None
    if partial:
        assert result["output"][-1]["status"] == "incomplete"
    else:
        assert result["output"] == []
    if stream:
        assert parsed[-1]["type"] == "response.failed"
        assert not any(e["type"] == "response.completed" for e in parsed)


def test_stream_finalization_failure_preserves_terminal_and_item_state():
    from atom.entrypoints.openai.serving_responses import CodexToolContext

    emitter = ResponseStreamEmitter(
        "r", "model", context=CodexToolContext(kinds={"search": "tool_search"})
    )
    frames = emitter.opening_frames()
    frames += emitter.frames_for_events(
        [
            ("tool_call_start", {"id": "call", "function": {"name": "search"}}),
            (
                "tool_call_args",
                {"function": {"arguments": "[" * 2000 + "0" + "]" * 2000}},
            ),
            ("reasoning", "pending reasoning"),
        ]
    )
    next_sequence = emitter.sequence_number
    with pytest.raises(RecursionError) as exc:
        emitter.finish_frames(has_tool_calls=True)
    assert not emitter.terminated
    assert emitter.sequence_number == next_sequence
    assert emitter.reasoning_buffer == "pending reasoning"
    frames += list(stream_failure_frames(exc.value, emitter, opened=True))
    parsed = events(frames)
    assert [e["sequence_number"] for e in parsed] == list(range(len(parsed)))
    assert parsed[-1]["type"] == "response.failed"
    assert sum(e["type"] == "response.failed" for e in parsed) == 1
    added = [
        e["item"]["id"] for e in parsed if e["type"] == "response.output_item.added"
    ]
    done = [e["item"]["id"] for e in parsed if e["type"] == "response.output_item.done"]
    assert sorted(added) == sorted(done)
    assert all(
        item["status"] == "incomplete" for item in parsed[-1]["response"]["output"]
    )
    assert emitter.finish_frames() == []
    assert list(stream_failure_frames(exc.value, emitter, opened=True)) == []


def test_responses_route_emits_failure_when_finalization_raises(server, monkeypatch):
    from atom.entrypoints.openai import serving_responses

    original = serving_responses.ResponseStreamEmitter._terminal_frame

    def fail_success(self, usage, terminal):
        if terminal["status"] == "completed":
            raise ValueError("terminal conversion failed")
        return original(self, usage, terminal)

    monkeypatch.setattr(
        serving_responses.ResponseStreamEmitter, "_terminal_frame", fail_success
    )
    response = post("/v1/responses", {"input": "hi", "stream": True})
    parsed = events([response.text])
    assert parsed[-1]["type"] == "response.failed"
    assert parsed[-1]["response"]["error"]["message"] == "terminal conversion failed"
    assert [e["sequence_number"] for e in parsed] == list(range(len(parsed)))
    assert sum(e["type"] == "response.output_item.done" for e in parsed) == 1
    assert parsed[-1]["response"]["output"][0]["status"] == "incomplete"
    assert server.cleanups


def test_nonstream_chat_rejects_stream_options_before_generation(server):
    response = post(
        "/v1/chat/completions",
        {
            "model": "renamed-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "stream_options requires stream=true"
    assert server.generations == 0
    assert server.templates == []
