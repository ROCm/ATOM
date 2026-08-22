# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Anthropic Messages API endpoint adapter.

Tests the format translation layer (serving_anthropic.py) without
requiring a running GPU server — uses unit tests on the conversion
functions and response builders.
"""

import ast
import json
import pathlib

import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.reasoning import separate_reasoning
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
from atom.entrypoints.openai.serving_chat import resolve_thinking
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser
from atom.entrypoints.openai.tool_parser.registry import parse_tool_calls

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
            stop_reason="end_turn",
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
            stop_reason="end_turn",
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
            stop_reason="end_turn",
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
            stop_reason="tool_use",
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
            stop_reason="tool_use",
        )
        types = [b["type"] for b in resp["content"]]
        assert types == ["thinking", "text", "tool_use"]
        assert resp["stop_reason"] == "tool_use"

    def test_the_caller_s_stop_reason_is_the_one_returned(self):
        """It used to be overwritten whenever there were tool calls, which
        discarded the answer the call site had already computed: a response
        cut off at `max_tokens` mid-call came back as an ordinary `tool_use`
        with silently truncated arguments, disagreeing with the streaming path
        for the same generation. Checked by value; the test this replaces
        grepped the source for the call and passed while the value was thrown
        away."""
        from atom.entrypoints.openai.tool_parser import ToolCall

        tc = ToolCall(
            id="call_3",
            type="function",
            function={"name": "bash", "arguments": '{"command": "l'},
        )
        for asked in ("max_tokens", "tool_use", "end_turn", "stop_sequence"):
            resp = build_anthropic_response(
                request_id="r",
                model="m",
                content_text="",
                tool_calls=[tc],
                stop_reason=asked,
            )
            assert (
                resp["stop_reason"] == asked
            ), f"asked for {asked!r}, got {resp['stop_reason']!r}"

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
            stop_reason="tool_use",
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

    @pytest.mark.parametrize(
        "engine_reason, expected",
        [
            ("eos", "end_turn"),
            ("max_tokens", "max_tokens"),
            ("stop_sequence", "stop_sequence"),
            # The scheduler's fourth ending, and the one a lookup misses: a
            # model stop *token* fired, spelled with the token's own id. That
            # is an ordinary end of turn, not the client's `stop_sequences`
            # matching -- which is what `stop_sequence` means to Anthropic,
            # and what it pairs with the matched string. Mapping it there
            # gives any model with a second declared EOS a `stop_sequence:
            # null` for a request that supplied no stop sequences.
            ("stop_163586", "end_turn"),
            ("aborted", "end_turn"),
            ("unschedulable: no free blocks", "end_turn"),
            ("", "end_turn"),
            (None, "end_turn"),
        ],
    )
    def test_every_shape_the_scheduler_emits(self, engine_reason, expected):
        """All seven, enumerated from `scheduler.py`, plus the unset default."""
        assert api_server.anthropic_stop_reason(engine_reason) == expected

    def test_the_non_streaming_path_passes_one(self):
        """It read `end_turn` from a default while the reason sat unused.

        The map was wired into the streaming generator only, so the same
        response reported two different endings depending on `stream`. Both
        bodies are unreachable from a unit test; this counts call sites.
        """
        src = pathlib.Path(api_server.__file__).read_text()
        assert (
            src.count("anthropic_stop_reason") >= 4
        ), "anthropic_stop_reason is defined but not called by both paths"
        assert (
            "stop_reason=anthropic_stop_reason_with_calls(" in src
        ), "the non-streaming response omits stop_reason"

    @pytest.mark.parametrize(
        "engine_reason, has_calls, expected",
        [
            ("max_tokens", True, "max_tokens"),
            ("max_tokens", False, "max_tokens"),
            ("eos", True, "tool_use"),
            ("eos", False, "end_turn"),
            ("stop_163586", True, "tool_use"),
        ],
    )
    def test_being_cut_short_outranks_having_made_a_call(
        self, engine_reason, has_calls, expected
    ):
        """`tool_use` says "act on this"; `max_tokens` says "this is not all of
        it". A response cut off mid-call parses to a call with a silently
        truncated argument value -- every format's unclosed-region branch
        exists to salvage exactly that -- and reporting `tool_use` for it told
        the client to run a tool with half its arguments and no sign anything
        was missing."""
        assert (
            api_server.anthropic_stop_reason_with_calls(engine_reason, has_calls)
            == expected
        )


class TestTheOffSwitchIsRecognised:
    """`{"type": "disabled"}` is how a client turns thinking off.

    It is also a non-empty dict, so `bool(request.thinking)` read it as on --
    the standard spelling of "off" was the one spelling that did not work.
    """

    class Req:
        def __init__(self, thinking):
            self.thinking = thinking

    @pytest.mark.parametrize(
        "thinking, enabled",
        [
            (None, False),
            ({}, False),
            ({"type": "disabled"}, False),
            ({"type": "enabled"}, True),
            ({"type": "enabled", "budget_tokens": 1024}, True),
        ],
    )
    def test_each_spelling(self, thinking, enabled):
        assert api_server.anthropic_thinking_enabled(self.Req(thinking)) is enabled

    def test_a_request_without_the_field_at_all(self):
        assert api_server.anthropic_thinking_enabled(object()) is False


class TestThinkingIsAnsweredInThePrompt:
    """`thinking: disabled` tells the *model* not to think.

    Three attempts to deal with an unwanted chain of thought after generating
    it each broke something else -- discarding it returned an empty message,
    relabelling it as `text` handed the client what it declined, and declining
    to separate it fed the reasoning to the tool parser, which read one model's
    musing about `<function=NAME>` as a call to a tool named `NAME`. Setting
    the template's own switch means there is no chain of thought to deal with.

    This is SGLang's design for the same field (`apply_reasoning_enabled`), and
    what vLLM gets structurally by having no such field on its Anthropic
    request at all -- its reasoning parser runs unconditionally and
    `include_reasoning` only suppresses the field after the split.
    """

    TOGGLE = ("enable_thinking", False, True)

    class Req:
        def __init__(self, thinking):
            self.thinking = thinking

    def test_disabled_sets_the_template_switch(self):
        kwargs = api_server.anthropic_template_kwargs(
            self.Req({"type": "disabled"}), self.TOGGLE
        )
        assert kwargs == {"enable_thinking": False}

    def test_enabled_writes_the_on_value(self):
        """Both directions go through the resolved name. Writing a hardcoded
        `thinking=True` here was a no-op on every template that reads another
        one, so an explicit opt-in was discarded against a server default."""
        kwargs = api_server.anthropic_template_kwargs(
            self.Req({"type": "enabled"}), self.TOGGLE
        )
        assert kwargs == {"enable_thinking": True}

    def test_an_absent_field_is_unstated_not_off(self):
        """Anthropic defaults to off, but reading absence as "switch this
        model's reasoning off" would silently change what every existing
        caller gets back. SGLang keys on the field being present too."""
        assert api_server.anthropic_template_kwargs(self.Req(None), self.TOGGLE) == {}

    def test_a_model_with_no_switch_gets_no_kwarg(self):
        assert (
            api_server.anthropic_template_kwargs(self.Req({"type": "disabled"}), None)
            == {}
        )

    def test_the_endpoint_renders_the_prompt_with_them(self):
        """The request's `thinking` used to be dropped on the floor here: the
        Anthropic path built `merged_kwargs` from server defaults only."""
        src = pathlib.Path(api_server.__file__).read_text()
        assert src.count("anthropic_template_kwargs(") >= 2
        assert "merged_kwargs.update(anthropic_template_kwargs(" in src


class TestSeparationIsUnconditional:
    """Whatever reasoning arrives is separated and reported, always.

    Not gated on `thinking`, for the same reason the chat path is not: the
    tool parser reads the same text, and a chain of thought left in it is a
    chain of thought the tool parser will try to parse.
    """

    RAW = (
        "<think>User wants weather. The syntax is <tool_call> then "
        "<function=NAME>...</think>Checking now. "
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>"
    )

    def test_no_call_is_fabricated_from_reasoning(self):
        reasoning, body = separate_reasoning(self.RAW, starts_thinking=False)
        _, calls = parse_tool_calls(body, None, parser_cls=QwenXmlParser)
        names = [c.function["name"] for c in calls]
        assert names == ["get_weather"], f"fabricated {names}"
        assert reasoning is not None and "syntax" in reasoning

    def test_tool_use_survives_a_request_that_declined_thinking(self):
        """The cost of the answer this replaces: tying the parse to `thinking`
        meant no `tool_use` block for any request that did not opt in, which
        is most of them."""
        _, body = separate_reasoning(self.RAW, starts_thinking=False)
        _, calls = parse_tool_calls(body, None, parser_cls=QwenXmlParser)
        assert len(calls) == 1

    @pytest.mark.parametrize("helper", [".split(", ".stream("])
    def test_neither_path_gates_it(self, helper):
        """Both endpoint bodies are unreachable from a unit test, so this
        counts call sites and asserts the gate that used to wrap them is
        gone.

        Spelled as the two `ReasoningChannel` accessors: separating is now one
        object with one method per delivery mode, and the endpoint reaches the
        module-level helpers through it rather than by name.
        """
        src = pathlib.Path(api_server.__file__).read_text()
        assert helper in src
        assert "anthropic_reasoning_split" not in src
        assert "anthropic_reasoning_filter" not in src
        assert "anthropic_tool_format" not in src


class TestASwitchlessModelStillHonoursDisabled:
    """The prompt is the better place to answer `thinking`, not the only one.

    `anthropic_template_kwargs` sets the template's own switch wherever there
    is one. For a model whose template has none -- gpt-oss-120b and
    DeepSeek-R1 both measure that way -- it returns `{}`, and the two
    downstream suppressions this branch removed are no longer there to catch
    it. So `thinking: {"type": "disabled"}` was honoured at neither layer:
    the client asked for no reasoning and got `thinking` blocks.

    Withheld, not left unseparated. Separation stays unconditional because
    the tool parser reads the same text -- see `TestSeparationIsUnconditional`.
    """

    TOGGLE = ("enable_thinking", False, True)

    class Req:
        def __init__(self, thinking):
            self.thinking = thinking

    @pytest.mark.parametrize(
        "thinking, expected",
        [
            ({"type": "disabled"}, True),
            # Anthropic's default is thinking-off, so a client that never sent
            # the field has no reason to expect `thinking` blocks -- and one
            # that validates block types or verifies the signature rejects
            # them. What goes in the *prompt* answers a different question and
            # still leaves an absent field alone.
            (None, True),
            ({"type": "enabled"}, False),
            ({}, True),
        ],
        ids=["off", "absent", "on", "empty"],
    )
    def test_the_client_is_shown_reasoning_only_if_it_asked(self, thinking, expected):
        assert api_server.anthropic_drop_reasoning(self.Req(thinking)) is expected

    def test_both_endpoint_paths_consult_it(self):
        """Read off the syntax tree, not grepped for: a literal check passes
        against code rewritten into a different shape with the same bug, and
        this suite has been fooled that way twice.
        """
        tree = ast.parse(pathlib.Path(api_server.__file__).read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "anthropic_messages"
        )
        reads = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Name)
            and n.id == "drop_reasoning"
            and isinstance(n.ctx, ast.Load)
        ]
        assert len(reads) >= 2, (
            "the streaming and non-streaming branches must both consult it; "
            f"found {len(reads)} read(s)"
        )


class TestWithheldReasoningStillSendsSomething:
    """Dropping the blocks must not drop the frames.

    `anthropic_drop_reasoning` suppresses `thinking` blocks for a request that
    said `disabled` on a model whose template has no switch -- and the branch
    was `continue`, with nothing in its place. The bytes are generated either
    way, so the socket went silent for the whole chain of thought: on an
    R1-shaped 5019-character trace the first client-visible frame arrived
    after 5016 of them. Long enough to trip proxy and SDK idle-read timeouts,
    and the stall watchdog this branch added can only report that, not
    prevent it.
    """

    def test_a_ping_frame_exists(self):
        assert api_server._ANTHROPIC_PING_FRAME.startswith("event: ping")

    def test_the_drop_branch_yields_it(self):
        """Read off the syntax tree: the branch must not be a bare
        `continue` again."""
        tree = ast.parse(pathlib.Path(api_server.__file__).read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "anthropic_messages"
        )
        drops = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "drop_reasoning"
        ]
        assert drops, "the suppression branch is gone; this asserts nothing"
        # The streaming branch is the one that `continue`s past a segment the
        # socket would otherwise have carried. The non-streaming sibling just
        # clears a field and has nothing to send.
        skipping = [
            n for n in drops if any(isinstance(c, ast.Continue) for c in ast.walk(n))
        ]
        assert skipping, "no branch skips a segment; matcher is stale"
        for node in skipping:
            assert [
                y for y in ast.walk(node) if isinstance(y, ast.Yield)
            ], "a reasoning segment is skipped with no frame in its place"


class TestAnEffortIsNotAnOptIn:
    """`resolve_thinking` returns `None` for "the request did not say".

    Collapsing that to `True` was harmless while the caller wrote a key no
    template read. Once it wrote the template's real switch -- merged after
    the server defaults and after the client's own `chat_template_kwargs` --
    a request carrying only `reasoning_effort` re-enabled reasoning over an
    operator's `--default-chat-template-kwargs '{"enable_thinking": false}'`.
    """

    class Req:
        def __init__(self, thinking=None, reasoning_effort=None):
            self.thinking = thinking
            self.reasoning_effort = reasoning_effort

    @pytest.mark.parametrize(
        "req, expected",
        [
            (Req(), None),
            (Req(reasoning_effort="high"), None),
            (Req(thinking={"type": "enabled"}), True),
            (Req(thinking={"type": "disabled"}), False),
            (Req(reasoning_effort="none"), False),
        ],
        ids=["nothing", "effort-only", "on", "off", "effort-none"],
    )
    def test_only_an_explicit_statement_resolves(self, req, expected):
        assert resolve_thinking(req)[0] is expected

    def test_the_toggle_is_written_only_when_stated(self):
        """Read off the syntax tree, not grepped: the guard has to be on the
        resolved value, not on the toggle merely existing."""
        tree = ast.parse(pathlib.Path(api_server.__file__).read_text())
        writes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript)
            and isinstance(n.ctx, ast.Store)
            and getattr(n.value, "id", None) == "merged_kwargs"
            and isinstance(n.slice, ast.Name)
        ]
        assert writes, "no name-keyed write to merged_kwargs; matcher is stale"
        guarded = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "_th_enabled is not None" in ast.unparse(n.test)
        ]
        assert guarded, "the toggle is written without asking whether it was stated"
