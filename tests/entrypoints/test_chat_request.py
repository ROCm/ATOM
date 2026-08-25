# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``atom.entrypoints.openai.chat_request``.

Covers the request-policy layer the MiniMax-M3 OAI conformance suite exercises:
input validation (400 instead of 200/422/500), the ``root`` role extension,
``tool_choice`` resolution and the ``thinking`` toggle. Pure python — no engine,
GPU or HTTP server required.
"""

import pytest

from atom.entrypoints.openai.chat_request import (
    ROOT_INSTRUCTION_HEADER,
    normalize_chat_messages,
    template_supported_roles,
    validate_request_messages,
)
from atom.entrypoints.openai.protocol import ChatCompletionRequest, ChatMessage

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {"name": "search", "parameters": {"properties": {}}},
}


def _request(**kwargs) -> ChatCompletionRequest:
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


# ============================================================================
# Sampling / body validation  (cases 20_01, 20_03, 06_08, 20_06)
# ============================================================================


class TestSamplingValidation:
    def test_temperature_above_two_is_rejected(self):
        with pytest.raises(ValueError, match=r"temperature must be in \[0, 2\]"):
            validate_request_messages(_request(temperature=5.0))

    def test_negative_temperature_is_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            validate_request_messages(_request(temperature=-0.5))

    @pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0])
    def test_temperature_in_range_is_accepted(self, temperature):
        validate_request_messages(_request(temperature=temperature))

    def test_top_p_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match=r"top_p must be in \(0, 1\]"):
            validate_request_messages(_request(top_p=1.5))

    def test_top_k_zero_is_rejected(self):
        with pytest.raises(ValueError, match="top_k"):
            validate_request_messages(_request(top_k=0))

    def test_negative_max_tokens_is_rejected(self):
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            validate_request_messages(_request(max_tokens=-1))

    def test_zero_max_completion_tokens_is_rejected(self):
        with pytest.raises(ValueError, match="max_completion_tokens must be >= 1"):
            validate_request_messages(_request(max_completion_tokens=0))

    def test_huge_max_tokens_is_left_to_the_context_check(self):
        # Bounded by max_model_len at request time, not by the schema.
        validate_request_messages(_request(max_tokens=524288))


class TestMessageValidation:
    def test_empty_messages_is_rejected(self):
        with pytest.raises(ValueError, match="at least one message"):
            validate_request_messages(ChatCompletionRequest(messages=[]))

    def test_missing_messages_is_rejected(self):
        with pytest.raises(ValueError, match="required"):
            validate_request_messages(ChatCompletionRequest())

    def test_unknown_role_is_rejected(self):
        with pytest.raises(ValueError, match="invalid role 'wizard'"):
            validate_request_messages(
                ChatCompletionRequest(messages=[{"role": "wizard", "content": "hi"}])
            )

    @pytest.mark.parametrize(
        "role", ["system", "developer", "user", "assistant", "tool", "root"]
    )
    def test_supported_roles_accepted(self, role):
        extra = {"tool_call_id": "call_1"} if role == "tool" else {}
        validate_request_messages(
            ChatCompletionRequest(messages=[{"role": role, "content": "x", **extra}])
        )

    def test_empty_role_is_rejected(self):
        with pytest.raises(ValueError, match="requires a 'role'"):
            validate_request_messages(
                ChatCompletionRequest(messages=[{"role": "", "content": "hi"}])
            )


# ============================================================================
# Tool-message structure  (cases 16_08, 16_09, 16_12)
# ============================================================================


def _assistant_with_calls(*calls) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def _call(
    call_id: str, name: str = "get_weather", arguments: str = '{"location":"SF"}'
):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestToolMessageValidation:
    def test_tool_message_without_tool_call_id_is_rejected(self):
        with pytest.raises(ValueError, match="requires a non-empty 'tool_call_id'"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        _assistant_with_calls(_call("call_1")),
                        {"role": "tool", "content": "72F"},
                    ]
                )
            )

    def test_mismatched_tool_call_id_is_rejected(self):
        with pytest.raises(ValueError, match="does not match any tool call"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        _assistant_with_calls(_call("call_1")),
                        {"role": "tool", "content": "72F", "tool_call_id": "call_999"},
                    ]
                )
            )

    def test_partial_tool_reply_is_rejected(self):
        with pytest.raises(ValueError, match="missing: call_2"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        _assistant_with_calls(_call("call_1"), _call("call_2")),
                        {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
                    ]
                )
            )

    def test_missing_tool_reply_before_next_user_turn_is_rejected(self):
        with pytest.raises(ValueError, match="must be followed by a 'tool' message"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        _assistant_with_calls(_call("call_1")),
                        {"role": "user", "content": "never mind"},
                    ]
                )
            )

    def test_complete_tool_round_trip_is_accepted(self):
        validate_request_messages(
            ChatCompletionRequest(
                messages=[
                    {"role": "user", "content": "weather?"},
                    _assistant_with_calls(_call("call_1"), _call("call_2")),
                    {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
                    {"role": "tool", "content": "sunny", "tool_call_id": "call_2"},
                ]
            )
        )

    def test_trailing_assistant_tool_call_is_accepted(self):
        """A conversation may end on the tool call the client wants continued."""
        validate_request_messages(
            ChatCompletionRequest(
                messages=[
                    {"role": "user", "content": "weather?"},
                    _assistant_with_calls(_call("call_1")),
                ]
            )
        )

    def test_bare_tool_result_after_trimmed_history_is_tolerated(self):
        validate_request_messages(
            ChatCompletionRequest(
                messages=[
                    {"role": "tool", "content": "72F", "tool_call_id": "call_old"},
                    {"role": "user", "content": "and tomorrow?"},
                ]
            )
        )

    def test_unparseable_tool_call_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="malformed 'function.arguments'"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        _assistant_with_calls(
                            _call("call_1", arguments="{not valid json")
                        ),
                        {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
                    ]
                )
            )

    def test_json_array_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="malformed 'function.arguments'"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        _assistant_with_calls(_call("call_1", arguments="[1, 2]")),
                        {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
                    ]
                )
            )

    def test_arguments_with_invalid_escape_are_tolerated(self):
        """Models emit ``\\k``-style escapes; the template layer repairs those."""
        validate_request_messages(
            ChatCompletionRequest(
                messages=[
                    {"role": "user", "content": "hi"},
                    _assistant_with_calls(
                        _call("call_1", arguments='{"path": "C:\\keep"}')
                    ),
                    {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
                ]
            )
        )

    def test_tool_call_without_id_is_rejected(self):
        with pytest.raises(ValueError, match="requires a non-empty 'id'"):
            validate_request_messages(
                ChatCompletionRequest(
                    messages=[
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"function": {"name": "f", "arguments": "{}"}}
                            ],
                        },
                    ]
                )
            )


# ============================================================================
# tools / tool_choice  (case 13_08)
# ============================================================================


class TestTemplateSupportedRoles:
    def test_detects_root_branch(self):
        assert "root" in template_supported_roles(
            "{% if message['role'] == 'root' %}...{% endif %}"
        )

    def test_detects_double_quoted_role(self):
        assert "root" in template_supported_roles('{% if role == "root" %}{% endif %}')

    def test_absent_role_not_reported(self):
        assert template_supported_roles("{{ messages[0]['content'] }}") == frozenset()

    def test_none_template(self):
        assert template_supported_roles(None) == frozenset()

    def test_dict_of_templates(self):
        templates = {"default": "no roles", "tool_use": "'root'"}
        assert "root" in template_supported_roles(templates)


class TestNormalizeChatMessages:
    def test_root_becomes_system_with_priority_header(self):
        messages = [
            ChatMessage(role="system", content="You are Assistant."),
            ChatMessage(role="root", content="Your name is MiniMax-M3-taoxi."),
            ChatMessage(role="user", content="who are you?"),
        ]
        out = normalize_chat_messages(messages)
        assert [m.role for m in out] == ["system", "system", "user"]
        # The root instruction lands after the competing system message, so it is
        # both flagged as higher priority and the most recent instruction read.
        assert out[0].content == "You are Assistant."
        assert out[1].content.startswith(ROOT_INSTRUCTION_HEADER)
        assert "MiniMax-M3-taoxi" in out[1].content

    def test_root_only_conversation_keeps_content_verbatim(self):
        messages = [
            ChatMessage(role="root", content="Your name is MiniMax-M3-taoxi."),
            ChatMessage(role="user", content="who are you?"),
        ]
        out = normalize_chat_messages(messages)
        assert [m.role for m in out] == ["system", "user"]
        assert out[0].content == "Your name is MiniMax-M3-taoxi."

    def test_root_is_hoisted_ahead_of_the_conversation(self):
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="root", content="Answer in French."),
        ]
        out = normalize_chat_messages(messages)
        assert [m.role for m in out] == ["system", "user"]

    def test_root_kept_when_the_template_handles_it(self):
        messages = [ChatMessage(role="root", content="rules")]
        out = normalize_chat_messages(messages, supported_roles=frozenset({"root"}))
        assert out[0].role == "root"

    def test_supported_root_is_hoisted_to_index_zero(self):
        """MiniMax-M3's template reads root only at messages[0].

        A root message anywhere else is silently dropped by the template, so it
        has to be moved to the front — ahead of the system message, which the
        template then renders as the lower-priority developer prompt.
        """
        messages = [
            ChatMessage(role="system", content="You are SystemBot."),
            ChatMessage(role="root", content="Your name is Taoxi."),
            ChatMessage(role="user", content="who are you?"),
        ]
        out = normalize_chat_messages(messages, supported_roles=frozenset({"root"}))
        assert [m.role for m in out] == ["root", "system", "user"]
        # No header text needed: the template itself ranks root above system.
        assert out[0].content == "Your name is Taoxi."

    def test_several_root_messages_are_merged(self):
        messages = [
            ChatMessage(role="root", content="Rule one."),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="root", content="Rule two."),
        ]
        out = normalize_chat_messages(messages, supported_roles=frozenset({"root"}))
        assert [m.role for m in out] == ["root", "user"]
        assert out[0].content == "Rule one.\n\nRule two."

    def test_developer_folded_into_system(self):
        out = normalize_chat_messages([ChatMessage(role="developer", content="rules")])
        assert out[0].role == "system"

    def test_developer_kept_when_supported(self):
        out = normalize_chat_messages(
            [ChatMessage(role="developer", content="rules")],
            supported_roles=frozenset({"developer"}),
        )
        assert out[0].role == "developer"

    def test_tool_fields_survive_normalization(self):
        messages = [
            ChatMessage(role="root", content="rules"),
            ChatMessage(role="tool", content="72F", tool_call_id="call_1"),
        ]
        out = normalize_chat_messages(messages)
        assert out[1].to_template_dict()["tool_call_id"] == "call_1"


# ============================================================================
# thinking  (case 04_01)
# ============================================================================
