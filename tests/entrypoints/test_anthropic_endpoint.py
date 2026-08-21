# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Anthropic Messages API endpoint adapter.

Tests the format translation layer (serving_anthropic.py) without
requiring a running GPU server — uses unit tests on the conversion
functions and response builders.
"""

import json
import pathlib

import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    build_anthropic_response,
    format_sse,
    stream_content_block_delta,
    stream_content_block_start,
    stream_content_block_stop,
    stream_message_delta,
    stream_message_start,
    stream_message_stop,
)

# ============================================================================
# Message Conversion Tests
# ============================================================================


class TestAnthropicToOpenAIMessages:
    def test_simple_user_message(self):
        msgs = [AnthropicMessage(role="user", content="Hello")]
        result = anthropic_to_openai_messages(msgs)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_system_string(self):
        msgs = [AnthropicMessage(role="user", content="Hi")]
        result = anthropic_to_openai_messages(msgs, system="You are helpful.")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1]["role"] == "user"

    def test_system_content_blocks(self):
        system = [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ]
        msgs = [AnthropicMessage(role="user", content="Hi")]
        result = anthropic_to_openai_messages(msgs, system=system)
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Be concise." in result[0]["content"]

    def test_user_content_blocks(self):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Part 1."},
                    {"type": "text", "text": "Part 2."},
                ],
            )
        ]
        result = anthropic_to_openai_messages(msgs)
        assert result[0]["content"] == "Part 1.\nPart 2."

    def test_assistant_string(self):
        msgs = [
            AnthropicMessage(role="user", content="Hi"),
            AnthropicMessage(role="assistant", content="Hello!"),
        ]
        result = anthropic_to_openai_messages(msgs)
        assert result[1] == {"role": "assistant", "content": "Hello!"}

    def test_assistant_with_tool_use(self):
        msgs = [
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                ],
            )
        ]
        result = anthropic_to_openai_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check."
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}

    def test_tool_result_in_user_message(self):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "72°F, sunny",
                    }
                ],
            )
        ]
        result = anthropic_to_openai_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "72°F, sunny"

    def test_tool_result_with_content_blocks(self):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_456",
                        "content": [
                            {"type": "text", "text": "Result line 1"},
                            {"type": "text", "text": "Result line 2"},
                        ],
                    }
                ],
            )
        ]
        result = anthropic_to_openai_messages(msgs)
        assert result[0]["role"] == "tool"
        assert "Result line 1" in result[0]["content"]
        assert "Result line 2" in result[0]["content"]

    def test_multi_turn_conversation(self):
        msgs = [
            AnthropicMessage(role="user", content="What's the weather?"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "72°F",
                    }
                ],
            ),
            AnthropicMessage(role="assistant", content="It's 72°F in NYC."),
            AnthropicMessage(role="user", content="Thanks!"),
        ]
        result = anthropic_to_openai_messages(msgs, system="Weather bot")
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"
        assert result[5]["role"] == "user"


# ============================================================================
# Tool Definition Conversion Tests
# ============================================================================


class TestAnthropicToOpenAITools:
    def test_none_tools(self):
        assert anthropic_to_openai_tools(None) is None

    def test_empty_tools(self):
        assert anthropic_to_openai_tools([]) is None

    def test_single_tool(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        result = anthropic_to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"
        assert "city" in result[0]["function"]["parameters"]["properties"]

    def test_multiple_tools(self):
        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {}},
            {"name": "tool_b", "description": "B", "input_schema": {}},
        ]
        result = anthropic_to_openai_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"


# ============================================================================
# Response Building Tests
# ============================================================================


class TestBuildAnthropicResponse:
    def test_basic_response(self):
        resp = build_anthropic_response(
            request_id="test123",
            model="test-model",
            content_text="Hello!",
            input_tokens=10,
            output_tokens=5,
        )
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["model"] == "test-model"
        assert resp["id"] == "msg_test123"
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == "Hello!"
        assert resp["usage"]["input_tokens"] == 10
        assert resp["usage"]["output_tokens"] == 5
        assert resp["stop_reason"] == "end_turn"

    def test_response_with_reasoning(self):
        resp = build_anthropic_response(
            request_id="test456",
            model="test-model",
            content_text="The answer is 42.",
            reasoning_content="Let me think about this...",
            input_tokens=20,
            output_tokens=15,
        )
        assert len(resp["content"]) == 2
        assert resp["content"][0]["type"] == "thinking"
        assert resp["content"][0]["thinking"] == "Let me think about this..."
        assert resp["content"][1]["type"] == "text"
        assert resp["content"][1]["text"] == "The answer is 42."

    def test_response_no_reasoning(self):
        resp = build_anthropic_response(
            request_id="test789",
            model="m",
            content_text="Direct answer.",
        )
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"

    def test_response_with_tool_calls(self):
        from atom.entrypoints.openai.tool_parser import ToolCall

        tc = ToolCall(
            id="call_0",
            type="function",
            function={"name": "read_file", "arguments": '{"path": "/tmp/foo.py"}'},
        )
        resp = build_anthropic_response(
            request_id="test_tc",
            model="m",
            content_text="Let me read that file.",
            tool_calls=[tc],
        )
        assert resp["stop_reason"] == "tool_use"
        types = [b["type"] for b in resp["content"]]
        assert "text" in types
        assert "tool_use" in types
        tool_block = [b for b in resp["content"] if b["type"] == "tool_use"][0]
        assert tool_block["name"] == "read_file"
        assert tool_block["input"] == {"path": "/tmp/foo.py"}
        assert tool_block["id"] == "call_0"

    def test_response_with_reasoning_and_tool_calls(self):
        from atom.entrypoints.openai.tool_parser import ToolCall

        tc = ToolCall(
            id="call_1",
            type="function",
            function={"name": "bash", "arguments": '{"command": "ls"}'},
        )
        resp = build_anthropic_response(
            request_id="test_rtc",
            model="m",
            content_text="I'll run a command.",
            reasoning_content="The user wants to list files.",
            tool_calls=[tc],
        )
        types = [b["type"] for b in resp["content"]]
        assert types == ["thinking", "text", "tool_use"]
        assert resp["stop_reason"] == "tool_use"

    def test_response_empty_content_with_tool_call(self):
        from atom.entrypoints.openai.tool_parser import ToolCall

        tc = ToolCall(
            id="call_2",
            type="function",
            function={"name": "bash", "arguments": '{"command": "pwd"}'},
        )
        resp = build_anthropic_response(
            request_id="test_empty",
            model="m",
            content_text="",
            tool_calls=[tc],
        )
        types = [b["type"] for b in resp["content"]]
        assert "tool_use" in types
        assert resp["stop_reason"] == "tool_use"


# ============================================================================
# SSE Streaming Format Tests
# ============================================================================


class TestSSEFormatting:
    def test_format_sse(self):
        result = format_sse("test_event", {"key": "value"})
        assert result.startswith("event: test_event\n")
        assert "data: " in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data["key"] == "value"

    def test_message_start(self):
        result = stream_message_start("req1", "model1", 50)
        assert "event: message_start" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "message_start"
        assert data["message"]["role"] == "assistant"
        assert data["message"]["model"] == "model1"
        assert data["message"]["usage"]["input_tokens"] == 50

    def test_content_block_start_tool_use(self):
        result = stream_content_block_start(
            2, "tool_use", tool_use_id="toolu_123", tool_name="read_file"
        )
        data = json.loads(result.split("data: ")[1].strip())
        assert data["content_block"]["type"] == "tool_use"
        assert data["content_block"]["id"] == "toolu_123"
        assert data["content_block"]["name"] == "read_file"
        assert data["index"] == 2

    def test_content_block_delta_tool_use(self):
        result = stream_content_block_delta(2, '{"path": "/foo"}', "tool_use")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["delta"]["type"] == "input_json_delta"
        assert data["delta"]["partial_json"] == '{"path": "/foo"}'

    def test_content_block_start_text(self):
        result = stream_content_block_start(0, "text")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "content_block_start"
        assert data["index"] == 0
        assert data["content_block"]["type"] == "text"

    def test_content_block_start_thinking(self):
        result = stream_content_block_start(0, "thinking")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["content_block"]["type"] == "thinking"

    def test_content_block_delta_text(self):
        result = stream_content_block_delta(0, "hello", "text")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "content_block_delta"
        assert data["delta"]["type"] == "text_delta"
        assert data["delta"]["text"] == "hello"

    def test_content_block_delta_thinking(self):
        result = stream_content_block_delta(1, "reasoning", "thinking")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["delta"]["type"] == "thinking_delta"
        assert data["delta"]["thinking"] == "reasoning"

    def test_content_block_stop(self):
        result = stream_content_block_stop(0)
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "content_block_stop"
        assert data["index"] == 0

    def test_message_delta(self):
        result = stream_message_delta("end_turn", 100)
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"
        assert data["usage"]["output_tokens"] == 100

    def test_message_stop(self):
        result = stream_message_stop()
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "message_stop"


# ============================================================================
# Request Schema Tests
# ============================================================================


class TestAnthropicMessagesRequest:
    def test_minimal_request(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        assert req.model == "test"
        assert req.max_tokens == 4096
        assert req.stream is False
        assert req.system is None

    def test_full_request(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=1000,
            system="Be helpful",
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop_sequences=["STOP"],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
        )
        assert req.max_tokens == 1000
        assert req.system == "Be helpful"
        assert req.temperature == 0.7
        assert req.stream is True
        assert req.stop_sequences == ["STOP"]
        assert len(req.tools) == 1

    def test_attribution_header_stripped(self):
        system = [
            {"type": "text", "text": "x-anthropic-billing-header: abc123"},
            {"type": "text", "text": "You are helpful."},
        ]
        msgs = [AnthropicMessage(role="user", content="Hi")]
        result = anthropic_to_openai_messages(msgs, system=system)
        assert result[0]["role"] == "system"
        assert "x-anthropic-billing-header" not in result[0]["content"]
        assert "You are helpful." in result[0]["content"]

    def test_attribution_header_only_system(self):
        system = [
            {"type": "text", "text": "x-anthropic-billing-header: xyz"},
        ]
        msgs = [AnthropicMessage(role="user", content="Hi")]
        result = anthropic_to_openai_messages(msgs, system=system)
        # No system message when all blocks are attribution headers
        assert result[0]["role"] == "user"


class TestTheStopReasonTellsTheTruth:
    """Why a message ended, not just that it did.

    `stop_reason` was the constant `end_turn` on the streaming path — the
    engine's own reason was never read. A response cut off at `max_tokens`
    therefore claimed a normal ending, and so did the one case where that
    matters most: a reasoning model asked for no `thinking` produces only
    reasoning, all of which is correctly dropped, and the client got an empty
    message that also reported nothing was wrong. It did not ask for a chain
    of thought and must not be handed one; what it can be given is the reason
    there is nothing else.
    """

    def test_the_engine_vocabulary_maps_onto_anthropic(self):
        assert api_server._ANTHROPIC_STOP_REASON == {
            "eos": "end_turn",
            "max_tokens": "max_tokens",
            "stop_sequence": "stop_sequence",
        }

    @pytest.mark.parametrize(
        "engine_reason, expected",
        [
            ("eos", "end_turn"),
            ("max_tokens", "max_tokens"),
            ("stop_sequence", "stop_sequence"),
        ],
    )
    def test_each_reason_survives_the_translation(self, engine_reason, expected):
        assert api_server._ANTHROPIC_STOP_REASON[engine_reason] == expected

    @pytest.mark.parametrize("unknown", ["aborted", None, "something_new"])
    def test_an_unmapped_reason_leaves_the_default_alone(self, unknown):
        """`aborted` has no counterpart; the client is already gone."""
        default = "end_turn"
        assert api_server._ANTHROPIC_STOP_REASON.get(unknown, default) == default


class TestThinkingOffMeansNoReasoningChannel:
    """`thinking` absent: nothing separates, so nothing can be dropped.

    The endpoint used to build a seeded filter regardless, split the output
    into channels, and then discard the reasoning half. That was invisible
    while an unseeded filter sent most output down the content channel; with
    seeding correct, a reasoning model stopped at `max_tokens` produces only
    reasoning and the client got an empty message. Putting the discarded half
    back as `text` would have handed it the one thing it declined.

    So the decision lives in two functions both entry points call, rather than
    inline in an async endpoint no test can reach -- an earlier version of
    this class re-implemented the branch to test it, and passed while the
    real one did the opposite.
    """

    RAW = "Working through it. </think>The answer is 42."

    def test_nothing_is_separated_when_thinking_is_off(self):
        reasoning, content = api_server.anthropic_reasoning_split(
            self.RAW, thinking=False, starts_thinking=True
        )
        assert reasoning is None
        assert content == self.RAW, "the raw output is the text, markers and all"

    def test_it_is_separated_when_thinking_is_on(self):
        reasoning, content = api_server.anthropic_reasoning_split(
            self.RAW, thinking=True, starts_thinking=True
        )
        assert reasoning == "Working through it."
        assert content == "The answer is 42."

    def test_no_filter_is_built_when_thinking_is_off(self):
        assert (
            api_server.anthropic_reasoning_filter(thinking=False, starts_thinking=True)
            is None
        )

    def test_a_seeded_filter_is_built_when_it_is_on(self):
        rf = api_server.anthropic_reasoning_filter(thinking=True, starts_thinking=True)
        assert rf is not None and rf.state == 1

    def test_both_halves_answer_the_same_question(self):
        """One rule, so the two paths cannot drift apart."""
        for thinking in (True, False):
            split = api_server.anthropic_reasoning_split(
                self.RAW, thinking=thinking, starts_thinking=True
            )
            filt = api_server.anthropic_reasoning_filter(
                thinking=thinking, starts_thinking=True
            )
            assert (split[0] is None) == (filt is None)

    @pytest.mark.parametrize(
        "helper", ["anthropic_reasoning_split", "anthropic_reasoning_filter"]
    )
    def test_the_endpoint_actually_calls_the_helper(self, helper):
        """Extracting a rule only helps if the caller uses it.

        The endpoint's body is an async generator inside a route handler that
        no unit test reaches, so a mutation putting `separate_reasoning` back
        inline passed every behavioural test here. Counted from the source:
        one definition plus at least one call.
        """
        src = pathlib.Path(api_server.__file__).read_text()
        assert (
            src.count(helper + "(") >= 2
        ), f"{helper} is defined but never called — the endpoint has its own copy"

    def test_no_ping_frame_survives(self):
        """Its only branch is gone; a leftover constant reads as live code."""
        src = pathlib.Path(api_server.__file__).read_text()
        assert "_ANTHROPIC_PING_FRAME" not in src
