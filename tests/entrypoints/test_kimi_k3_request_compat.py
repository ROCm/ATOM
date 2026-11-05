# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 serving: request validation that turns malformed tool history into
HTTP 400, and verbatim tool-call arguments."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoders import apply_chat_template
from atom.entrypoints.openai.protocol import ChatCompletionRequest, ChatMessage
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    anthropic_to_openai_messages,
)
from atom.entrypoints.openai.serving_chat import (
    request_tools,
    validate_chat_request,
    validate_history_tool_calls,
)
from atom.entrypoints.openai.serving_responses import (
    ResponsesRequest,
    _normalize_input_items,
    responses_to_openai_messages,
)
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import (
    KIMI_K3_TOOLS_START,
    KimiK3Parser,
)
from atom.entrypoints.openai.tool_parser.registry import (
    PARSERS_BY_NAME,
    channel_for_request,
    resolve_tool_call_parser,
    template_keeps_tool_call_arguments,
)
from atom.sampling_params import RequiredChannel

TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}
USER = {"role": "user", "content": "hi"}


def _history(tool_call):
    return [
        USER,
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]


@pytest.mark.parametrize(
    "tool_call",
    [
        {"id": "c1", "type": "function", "function": {"arguments": "{}"}},
        {"id": "c1", "type": "function", "function": {"name": "", "arguments": "{}"}},
        {"id": "c1", "type": "function", "function": {"name": 3, "arguments": "{}"}},
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": 123}},
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": ["a"]}},
        {"id": "c1", "type": "function", "function": "f"},
        "not-an-object",
    ],
    ids=[
        "no-name",
        "empty-name",
        "int-name",
        "int-args",
        "list-args",
        "function-not-object",
        "call-not-object",
    ],
)
def test_malformed_history_tool_call_rejected(tool_call):
    with pytest.raises(ValueError):
        validate_chat_request(ChatCompletionRequest(messages=_history(tool_call)))


def test_malformed_history_rejected_via_prompt_alias():
    bad = {"id": "c1", "type": "function", "function": {"arguments": "{}"}}
    with pytest.raises(ValueError):
        validate_chat_request(ChatCompletionRequest(prompt=_history(bad)))


@pytest.mark.parametrize(
    "tool_call",
    [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}},
        {"id": "c1", "type": "function", "function": {"name": "f"}},
        {"id": "c1", "name": "f", "arguments": "{}"},
    ],
    ids=["string-args", "dict-args", "no-args", "flat-form"],
)
def test_wellformed_history_tool_call_accepted(tool_call):
    validate_chat_request(ChatCompletionRequest(messages=_history(tool_call)))


@pytest.mark.parametrize("tools", [None, []], ids=["absent", "empty"])
def test_required_without_tools_rejected(tools):
    with pytest.raises(ValueError):
        validate_chat_request(
            ChatCompletionRequest(messages=[USER], tools=tools, tool_choice="required")
        )


def test_required_with_tools_accepted():
    validate_chat_request(
        ChatCompletionRequest(messages=[USER], tools=[TOOL], tool_choice="required")
    )


def test_required_with_system_message_tools_accepted():
    system = {"role": "system", "content": "", "tools": [TOOL]}
    validate_chat_request(
        ChatCompletionRequest(messages=[system, USER], tool_choice="required")
    )


def _json_schema(**fields):
    return {
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object"}, **fields},
    }


def test_non_boolean_strict_rejected():
    with pytest.raises(ValueError):
        validate_chat_request(
            ChatCompletionRequest(
                messages=[USER], response_format=_json_schema(name="w", strict="true")
            )
        )


@pytest.mark.parametrize("strict", [True, False, None])
def test_json_schema_strict_bool_or_null_accepted(strict):
    validate_chat_request(
        ChatCompletionRequest(
            messages=[USER], response_format=_json_schema(name="w", strict=strict)
        )
    )


RAW_ARGUMENTS = [
    '{"precision":1e2,"threshold":1.50}',
    '{"options": { "unit" : "c" }}',
    '{"location":"北京"',
    " ",
    "[1, 2]",
]


@pytest.mark.parametrize("raw", RAW_ARGUMENTS)
def test_tool_call_arguments_kept_verbatim_when_preserved(raw):
    msg = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": raw},
            }
        ],
    )
    out = msg.to_template_dict(preserve_tool_call_arguments=True)
    assert out["tool_calls"][0]["function"]["arguments"] == raw


def test_tool_call_arguments_parsed_by_default():
    msg = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": '{"a": 1}'},
            }
        ],
    )
    assert msg.to_template_dict()["tool_calls"][0]["function"]["arguments"] == {"a": 1}


def test_anthropic_tool_use_arguments_keep_non_ascii():
    tool_input = {"todos": [{"content": "修复 bug"}]}
    msgs = [
        AnthropicMessage(role="user", content="hi"),
        AnthropicMessage(
            role="assistant",
            content=[
                {"type": "tool_use", "id": "c1", "name": "todo", "input": tool_input}
            ],
        ),
    ]
    result = anthropic_to_openai_messages(msgs)
    arguments = result[1]["tool_calls"][0]["function"]["arguments"]
    assert "修复 bug" in arguments
    assert json.loads(arguments) == tool_input


class _CapturingTokenizer:
    def __init__(self, rendered: str = ""):
        self.rendered = rendered
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return self.rendered


@pytest.mark.parametrize(
    ("preserve", "expected"), [(True, '{"a": 1e2}'), (False, {"a": 100.0})]
)
def test_apply_chat_template_forwards_preserve_flag(preserve, expected):
    tokenizer = _CapturingTokenizer()
    msg = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": '{"a": 1e2}'},
            }
        ],
    )
    apply_chat_template(tokenizer, None, [msg], preserve_tool_call_arguments=preserve)
    assert tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"] == expected


QWEN_PROBE = "<tool_call>\n<function=NAME>\n<parameter=k>v</parameter>\n</function>"


@pytest.mark.parametrize(
    ("probe", "keeps"),
    [(f"sys {KIMI_K3_TOOLS_START} x", True), (QWEN_PROBE, False), (None, False)],
    ids=["kimi-k3", "qwen", "unrendered"],
)
def test_template_keeps_tool_call_arguments(probe, keeps):
    assert template_keeps_tool_call_arguments(probe) is keeps


class _NeverRenders:
    def apply_chat_template(self, messages, **kwargs):
        raise AssertionError("probe rendered twice")


def test_resolve_uses_given_probe_without_rendering():
    probe = f"sys {KIMI_K3_TOOLS_START} x"
    assert resolve_tool_call_parser(None, _NeverRenders(), rendered_probe=probe) is (
        KimiK3Parser
    )


@pytest.mark.parametrize("override", sorted(PARSERS_BY_NAME))
def test_given_probe_is_reused_for_every_override(override):
    probe = f"sys {KIMI_K3_TOOLS_START} x"
    parser = resolve_tool_call_parser(override, _NeverRenders(), rendered_probe=probe)
    assert parser is PARSERS_BY_NAME[override]


def test_responses_custom_tool_call_arguments_keep_non_ascii():
    items = [
        {"type": "custom_tool_call", "call_id": "c1", "name": "apply", "input": "修复"}
    ]
    arguments = _normalize_input_items(items, [])[0]["arguments"]
    assert "修复" in arguments
    assert json.loads(arguments) == {"input": "修复"}


def test_required_channel_armed_for_message_declared_tools():
    channel = object()
    system = {"role": "system", "content": "", "tools": [TOOL]}
    request = ChatCompletionRequest(messages=[system, USER], tool_choice="required")
    assert request_tools(request) == [TOOL]
    assert channel_for_request(channel, "required", request_tools(request)) is channel


def _tool(name):
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_request_tools_is_top_level_plus_message_declared():
    system = {"role": "system", "content": "", "tools": [_tool("dyn")]}
    request = ChatCompletionRequest(messages=[system, USER], tools=[TOOL])
    assert request_tools(request) == [TOOL, _tool("dyn")]


def _named(name):
    return {"type": "function", "function": {"name": name}}


@pytest.mark.parametrize(
    "top_level", [None, [TOOL]], ids=["only-dynamic", "top-level-and-dynamic"]
)
def test_named_tool_choice_may_reference_a_message_declared_tool(top_level):
    system = {"role": "system", "content": "", "tools": [_tool("dyn")]}
    validate_chat_request(
        ChatCompletionRequest(
            messages=[system, USER], tools=top_level, tool_choice=_named("dyn")
        )
    )


def test_named_tool_choice_for_an_undeclared_tool_rejected():
    system = {"role": "system", "content": "", "tools": [_tool("dyn")]}
    with pytest.raises(ValueError, match="unknown tool"):
        validate_chat_request(
            ChatCompletionRequest(messages=[system, USER], tool_choice=_named("nope"))
        )


@pytest.mark.parametrize(
    ("tools", "match"),
    [
        ([{"type": "web_search"}], r"messages\[0\]\.tools\[0\]\.type"),
        ([{"type": "function"}], r"messages\[0\]\.tools\[0\]\.function"),
        ("not-a-list", r"messages\[0\]\.tools must be an array"),
        ([_tool("dyn"), _tool("dyn")], r"messages\[0\]\.tools: duplicate tool name"),
    ],
    ids=["wrong-type", "no-function", "not-a-list", "duplicate-in-message"],
)
def test_malformed_message_declared_tools_rejected(tools, match):
    system = {"role": "system", "content": "", "tools": tools}
    with pytest.raises(ValueError, match=match):
        validate_chat_request(ChatCompletionRequest(messages=[system, USER]))


def test_message_may_redeclare_a_top_level_tool():
    system = {"role": "system", "content": "", "tools": [TOOL]}
    validate_chat_request(ChatCompletionRequest(messages=[system, USER], tools=[TOOL]))


@pytest.mark.parametrize("block", [{}, {"name": ""}], ids=["no-name", "empty-name"])
def test_anthropic_tool_use_without_name_rejected(block):
    msgs = [
        AnthropicMessage(role="user", content="hi"),
        AnthropicMessage(
            role="assistant",
            content=[{"type": "tool_use", "id": "c1", "input": {}, **block}],
        ),
    ]
    with pytest.raises(ValueError):
        validate_history_tool_calls(anthropic_to_openai_messages(msgs))


def test_responses_function_call_without_name_rejected():
    request = ResponsesRequest(
        input=[{"type": "function_call", "call_id": "c1", "arguments": "{}"}]
    )
    with pytest.raises(ValueError):
        validate_history_tool_calls(responses_to_openai_messages(request))


# Endpoint level: drive the real handlers with generation stubbed out, so a
# revert at the call sites in api_server fails here, not only in the helpers.

CHANNEL = RequiredChannel(after=(10,), opening=(20, 21), opens_at=1, end_ids=(2,))


@pytest.fixture
def server(monkeypatch):
    calls = SimpleNamespace(sampling=[], template_kwargs=[])

    def record_template(*_args, **kwargs):
        calls.template_kwargs.append(kwargs)
        return "<rendered prompt>"

    async def fake_generate_async(_prompt, sampling, _request_id, **_kwargs):
        calls.sampling.append(sampling)
        yield {
            "text": "hello",
            "finish_reason": "eos",
            "num_tokens_input": 4,
            "num_tokens_output": 1,
            "num_cached_tokens": 0,
        }

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(api_server, "required_tool_channel", CHANNEL)
    monkeypatch.setattr(api_server, "preserve_tool_call_arguments", True)
    monkeypatch.setattr(
        api_server,
        "tokenizer",
        SimpleNamespace(encode=lambda _t, **_kw: [1, 2], decode=lambda *_a, **_kw: ""),
    )
    monkeypatch.setattr(api_server, "apply_chat_template", record_template)
    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            config=SimpleNamespace(
                generation_config=SimpleNamespace(), max_model_len=8192
            ),
            io_processor=SimpleNamespace(encode_prompt=lambda _text: [1, 2]),
        ),
    )
    monkeypatch.setattr(api_server, "generate_async", fake_generate_async)
    return calls


def _chat(**kwargs):
    request = ChatCompletionRequest(model="m", temperature=1.0, **kwargs)
    return asyncio.run(api_server.chat_completions(request, None))


def test_chat_arms_required_channel_for_message_declared_tools(server):
    system = {"role": "system", "content": "", "tools": [TOOL]}
    _chat(messages=[system, USER], tool_choice="required")
    assert server.sampling[-1].required_channel is CHANNEL


def test_chat_render_receives_preserve_flag(server):
    _chat(
        messages=_history(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        )
    )
    assert server.template_kwargs[-1]["preserve_tool_call_arguments"] is True


def test_messages_endpoint_rejects_nameless_tool_use(server):
    request = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[
            AnthropicMessage(role="user", content="hi"),
            AnthropicMessage(
                role="assistant",
                content=[{"type": "tool_use", "id": "c1", "input": {}}],
            ),
        ],
    )
    response = asyncio.run(api_server.anthropic_messages(request, None))
    assert response.status_code == 400


@pytest.mark.parametrize(
    "block",
    [
        {"type": "text"},
        {"type": "tool_use", "name": "f", "input": {}},
        {"type": "tool_result", "content": "ok"},
        {"type": "text", "text": None},
        {"type": "tool_use", "id": None, "name": "f", "input": {}},
        {"type": "tool_result", "tool_use_id": None, "content": "ok"},
    ],
    ids=[
        "text-without-text",
        "tool-use-without-id",
        "tool-result-without-id",
        "null-text",
        "null-tool-use-id",
        "null-tool-result-id",
    ],
)
def test_messages_endpoint_rejects_missing_required_block_field(server, block):
    role = "assistant" if block["type"] == "tool_use" else "user"
    request = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[
            AnthropicMessage(role="user", content="hi"),
            AnthropicMessage(role=role, content=[block]),
        ],
    )
    response = asyncio.run(api_server.anthropic_messages(request, None))
    assert response.status_code == 400


def test_responses_endpoint_rejects_nameless_function_call(server):
    async def post():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": [
                        {"type": "function_call", "call_id": "c1", "arguments": "{}"}
                    ],
                },
            )

    assert asyncio.run(post()).status_code == 400


def test_messages_render_receives_preserve_flag(server):
    request = AnthropicMessagesRequest(
        model="m", max_tokens=16, messages=[AnthropicMessage(role="user", content="hi")]
    )
    asyncio.run(api_server.anthropic_messages(request, None))
    assert server.template_kwargs[-1]["preserve_tool_call_arguments"] is True


def test_responses_render_receives_preserve_flag(server):
    async def post():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/responses", json={"model": "m", "input": "hi"}
            )

    assert asyncio.run(post()).status_code == 200
    assert server.template_kwargs[-1]["preserve_tool_call_arguments"] is True


def test_multimodal_chat_keeps_tool_call_arguments(server, monkeypatch):
    # Real _prepare_multimodal_inputs; only the media pipeline below it is
    # stubbed, so the flag is checked where the image path builds its messages.
    raw = '{"precision":1e2}'
    seen = []

    def parse_chat_messages(messages):
        seen.extend(messages)
        return messages, {"image": "fixture"}

    async def generate_multimodal(_ids, _media, sampling, _request_id, **_kwargs):
        server.sampling.append(sampling)
        yield {
            "text": "ok",
            "finish_reason": "stop",
            "num_tokens_input": 4,
            "num_tokens_output": 1,
        }

    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: object())
    monkeypatch.setattr(api_server, "parse_chat_messages", parse_chat_messages)
    monkeypatch.setattr(
        api_server, "prepare_multimodal_inputs", lambda *_a, **_kw: ([1, 2, 3], {})
    )
    monkeypatch.setattr(api_server, "generate_async_multimodal", generate_multimodal)
    image = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": raw}}
    _chat(messages=[*_history(call), image])
    assert seen[1]["tool_calls"][0]["function"]["arguments"] == raw
