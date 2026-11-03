# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 serving: request validation that turns malformed tool history into
HTTP 400, and verbatim tool-call arguments."""

import json

import pytest

from atom.entrypoints.openai.chat_encoders import apply_chat_template
from atom.entrypoints.openai.protocol import ChatCompletionRequest, ChatMessage
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
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


def test_request_tools_prefers_top_level_list():
    system = {"role": "system", "content": "", "tools": [{"type": "function"}]}
    request = ChatCompletionRequest(messages=[system, USER], tools=[TOOL])
    assert request_tools(request) == [TOOL]


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
