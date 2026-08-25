# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for tool call parsing."""

import json

import pytest

from atom.entrypoints.openai.tool_parser import (
    ToolCall,
    ToolCallStreamParser,
    parse_tool_calls,
    tool_call_prefill,
)

# ============================================================================
# parse_tool_calls() Tests
# ============================================================================


class TestParseToolCalls:
    """Tests for the parse_tool_calls() function."""

    def test_single_tool_call(self):
        text = (
            "I'll run that."
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.exec:0<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = parse_tool_calls(text)
        assert content == "I'll run that."
        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "exec"
        assert '"command"' in tool_calls[0].function["arguments"]
        assert tool_calls[0].type == "function"

    def test_multiple_tool_calls(self):
        text = (
            "Let me search."
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.fetch:1<|tool_call_argument_begin|>{"url": "http://example.com"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = parse_tool_calls(text)
        assert content == "Let me search."
        assert len(tool_calls) == 2
        assert tool_calls[0].function["name"] == "search"
        assert tool_calls[1].function["name"] == "fetch"

    def test_no_tool_calls(self):
        text = "Just a regular response."
        content, tool_calls = parse_tool_calls(text)
        assert content == "Just a regular response."
        assert len(tool_calls) == 0

    def test_empty_content_with_tool_call(self):
        text = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{"cmd": "echo hi"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = parse_tool_calls(text)
        assert content == ""
        assert len(tool_calls) == 1

    def test_unclosed_section(self):
        text = (
            "Here:"
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.exec:0<|tool_call_argument_begin|>{"cmd": "ls"}<|tool_call_end|>'
        )
        content, tool_calls = parse_tool_calls(text)
        assert content == "Here:"
        assert len(tool_calls) == 1

    def test_tool_call_to_dict(self):
        tc = ToolCall(
            id="call_abc",
            type="function",
            function={"name": "test", "arguments": "{}"},
        )
        d = tc.to_dict()
        assert d["id"] == "call_abc"
        assert d["type"] == "function"
        assert d["function"]["name"] == "test"

    def test_curl_tool_call(self):
        text = (
            "I'll fetch that URL for you."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.curl:0"
            '<|tool_call_argument_begin|>{"url": "https://api.example.com/data", "method": "GET", "headers": {"Authorization": "Bearer token123"}}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = parse_tool_calls(text)
        assert content == "I'll fetch that URL for you."
        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "curl"
        assert tool_calls[0].type == "function"
        args = tool_calls[0].function["arguments"]
        assert "https://api.example.com/data" in args
        assert '"method": "GET"' in args
        assert '"Authorization"' in args

    def test_tool_call_with_complex_args(self):
        args = (
            '{"messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}'
        )
        text = (
            "<|tool_calls_section_begin|>"
            f"<|tool_call_begin|>functions.chat:0<|tool_call_argument_begin|>{args}<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = parse_tool_calls(text)
        assert len(tool_calls) == 1
        assert tool_calls[0].function["arguments"] == args


# ============================================================================
# ToolCallStreamParser Tests
# ============================================================================


class TestToolCallStreamParser:
    """Tests for the ToolCallStreamParser streaming state machine."""

    def _run_parser(self, tokens):
        """Helper: run tokens through parser and return all events."""
        parser = ToolCallStreamParser()
        results = []
        for token in tokens:
            results.extend(parser.process(token))
        results.extend(parser.flush())
        return results

    def test_no_tool_calls(self):
        tokens = ["Hello", " world", "!"]
        results = self._run_parser(tokens)
        content = "".join(d for t, d in results if t == "content")
        assert "Hello" in content
        assert "world" in content
        tool_starts = [d for t, d in results if t == "tool_call_start"]
        assert len(tool_starts) == 0

    def test_single_tool_call_streaming(self):
        tokens = [
            "I'll do it.",
            "<|tool_calls_section_begin|>",
            '<|tool_call_begin|>functions.exec:0<|tool_call_argument_begin|>{"cmd": "ls"}<|tool_call_end|>',
            "<|tool_calls_section_end|>",
        ]
        results = self._run_parser(tokens)
        content = "".join(d for t, d in results if t == "content")
        assert "I'll do it." in content

        starts = [d for t, d in results if t == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["function"]["name"] == "exec"

        args = [d for t, d in results if t == "tool_call_args"]
        assert len(args) == 1
        assert '"cmd"' in args[0]["function"]["arguments"]

        ends = [d for t, d in results if t == "tool_call_end"]
        assert len(ends) == 1

    def test_content_before_tool_call(self):
        tokens = [
            "Let me ",
            "help.",
            "<|tool_calls_section_begin|>",
            '<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|>',
            "<|tool_calls_section_end|>",
        ]
        results = self._run_parser(tokens)
        content = "".join(d for t, d in results if t == "content")
        assert "Let me help." in content

    def test_curl_tool_call_streaming(self):
        tokens = [
            "I'll fetch that for you.",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.curl:0"
            '<|tool_call_argument_begin|>{"url": "https://api.example.com/data", "method": "POST", "body": "{\\"key\\": \\"value\\"}"}'
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]
        results = self._run_parser(tokens)
        content = "".join(d for t, d in results if t == "content")
        assert "I'll fetch that for you." in content

        starts = [d for t, d in results if t == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["function"]["name"] == "curl"

        args = [d for t, d in results if t == "tool_call_args"]
        assert len(args) == 1
        assert "https://api.example.com/data" in args[0]["function"]["arguments"]
        assert '"method": "POST"' in args[0]["function"]["arguments"]

        ends = [d for t, d in results if t == "tool_call_end"]
        assert len(ends) == 1

    def test_flush_with_unclosed_section(self):
        tokens = [
            "Hi",
            "<|tool_calls_section_begin|>",
            '<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>{"a": 1}<|tool_call_end|>',
        ]
        results = self._run_parser(tokens)
        starts = [d for t, d in results if t == "tool_call_start"]
        assert len(starts) == 1
        ends = [d for t, d in results if t == "tool_call_end"]
        assert len(ends) == 1  # flush should emit tool_call_end

    def test_unparsed_dialect_is_flushed_as_content(self):
        """An unrecognized <tool_call> block must not vanish from the stream."""
        results = self._run_parser(
            ["Here: ", "<tool_call>", '{"name": "f"}', "</tool_call>"]
        )
        content = "".join(d for t, d in results if t == "content")
        assert content == 'Here: <tool_call>{"name": "f"}</tool_call>'
        assert not [d for t, d in results if t == "tool_call_start"]

    def test_bare_angle_bracket_content_is_not_held_back(self):
        """A '<' that cannot start a tool marker must stream out immediately.

        Buffering it until end-of-generation would stall every SSE delta after
        the first '<' in an answer that merely talks about ``a < b``.
        """
        parser = ToolCallStreamParser()
        events = parser.process("if a ") + parser.process("< b then")
        assert [d for t, d in events if t == "content"] == ["if a ", "< b then"]


# ============================================================================
# MiniMax XML format
# ============================================================================
#
# Wire format captured from a live amd/MiniMax-M3-MXFP4 server via
# /v1/completions (the namespace token is in the vocabulary but not marked
# special, so it survives decode(skip_special_tokens=True)):
#
#   ]<]minimax[>[<tool_call>
#   ]<]minimax[>[<invoke name="get_weather">]<]minimax[>[<location>Beijing]<]minimax[>[</location>...
#   ]<]minimax[>[</tool_call>
#
# Arguments are named by the *element*, not a name= attribute, and nested values
# expand recursively (<tags><item>a</item></tags>).

NS = "]<]minimax[>["

MINIMAX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "opts": {
                        "type": "object",
                        "properties": {
                            "unit": {"type": "string"},
                            "hours": {"type": "integer"},
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "nums": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    },
]


def _param(name: str, value: str) -> str:
    """One MiniMax argument element: ``]ns[<name>value]ns[</name>``."""
    return f"{NS}<{name}>{value}{NS}</{name}>"


def _invoke(name: str, *params: str) -> str:
    return f'{NS}<invoke name="{name}">' + "".join(params) + f"{NS}</invoke>"


def _minimax_block(*invokes: str) -> str:
    return f"{NS}<tool_call>\n" + "\n".join(invokes) + f"\n{NS}</tool_call>"


# The exact bytes a live MiniMax-M3 emitted for a two-tool request.
REAL_M3_OUTPUT = (
    "I'll get the weather forecast for Beijing and search for air quality "
    "information in parallel."
    "]<]minimax[>[<tool_call>\n"
    ']<]minimax[>[<invoke name="get_weather">]<]minimax[>[<location>Beijing'
    "]<]minimax[>[</location>]<]minimax[>[<days>3]<]minimax[>[</days>"
    "]<]minimax[>[<metric>true]<]minimax[>[</metric>]<]minimax[>[</invoke>\n"
    ']<]minimax[>[<invoke name="search">]<]minimax[>[<q>beijing air quality'
    "]<]minimax[>[</q>]<]minimax[>[<tags>]<]minimax[>[<item>china"
    "]<]minimax[>[</item>]<]minimax[>[<item>env]<]minimax[>[</item>"
    "]<]minimax[>[</tags>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
)


class TestParseMiniMaxToolCalls:
    def test_real_model_output(self):
        """Regression on the exact output captured from the live model."""
        content, tool_calls = parse_tool_calls(REAL_M3_OUTPUT, MINIMAX_TOOLS)
        assert content == (
            "I'll get the weather forecast for Beijing and search for air "
            "quality information in parallel."
        )
        assert [tc.function["name"] for tc in tool_calls] == ["get_weather", "search"]
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "location": "Beijing",
            "days": 3,
            "metric": True,
        }
        assert json.loads(tool_calls[1].function["arguments"]) == {
            "q": "beijing air quality",
            "tags": ["china", "env"],
        }
        assert all(tc.type == "function" for tc in tool_calls)
        assert tool_calls[0].id != tool_calls[1].id

    def test_single_tool_call(self):
        text = "I'll check that." + _minimax_block(
            _invoke("get_weather", _param("location", "Beijing"))
        )
        content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert content == "I'll check that."
        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "get_weather"
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "location": "Beijing"
        }

    def test_scalar_values_coerced_to_schema_types(self):
        text = _minimax_block(
            _invoke(
                "get_weather",
                _param("location", "Beijing"),
                _param("days", "3"),
                _param("metric", "true"),
            )
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "location": "Beijing",
            "days": 3,
            "metric": True,
        }

    def test_nested_object_argument(self):
        text = _minimax_block(
            _invoke(
                "get_weather",
                _param("location", "Paris"),
                _param("opts", _param("unit", "c") + _param("hours", "12")),
            )
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "location": "Paris",
            "opts": {"unit": "c", "hours": 12},
        }

    def test_array_items_typed_from_the_item_schema(self):
        text = _minimax_block(
            _invoke(
                "search",
                _param("q", "x"),
                _param("nums", _param("item", "1") + _param("item", "2")),
            )
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "q": "x",
            "nums": [1, 2],
        }

    def test_values_stay_strings_without_a_schema(self):
        text = _minimax_block(_invoke("get_weather", _param("days", "3")))
        _content, tool_calls = parse_tool_calls(text)
        assert json.loads(tool_calls[0].function["arguments"]) == {"days": "3"}

    def test_multiline_parameter_value(self):
        """Inner text is the value verbatim.

        MiniMax writes a value inline -- captured from the live model, a
        multi-line body arrives as ``<content>hello\nworld\n</content>`` with no
        framing newlines. So every newline between the tags belongs to the value,
        and trimming them would hand the tool something the model did not say.
        """
        text = _minimax_block(_invoke("search", _param("q", "\nline one\nline two\n")))
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "q": "\nline one\nline two\n"
        }

    def test_trailing_newline_is_preserved(self):
        """A file body really can end in a newline; dropping it changes the file.

        The exact shape the live model emits for MiniMax conformance case 16_15.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        text = _minimax_block(
            _invoke(
                "write_file",
                _param("path", "/tmp/hello.txt"),
                _param("content", "hello\nworld\n"),
            )
        )
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "path": "/tmp/hello.txt",
            "content": "hello\nworld\n",
        }

    def test_invoke_without_the_tool_call_wrapper(self):
        """The <tool_call> wrapper is optional; the namespace token is not."""
        text = "Sure." + _invoke("search", _param("q", "hi"))
        content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert content == "Sure."
        assert tool_calls[0].function["name"] == "search"

    def test_bare_invoke_without_namespace_is_not_claimed(self):
        """``<invoke name=`` alone must not be treated as a tool call.

        It can appear in ordinary prose or code from *any* model, so claiming it
        would corrupt other models' responses. Detection requires MiniMax's
        namespace token; a bare invoke stays content.
        """
        text = 'Here is XML: <invoke name="search"><q>hi</q></invoke>'
        content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert tool_calls == []
        assert content == text

    def test_unnamespaced_tags_inside_a_claimed_block_still_parse(self):
        """Once the namespace token decides the format, plain tags are fine."""
        text = (
            f"{NS}<tool_call>\n"
            '<invoke name="search"><q>hi</q></invoke>\n'
            "</tool_call>"
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {"q": "hi"}

    def test_parameter_name_attribute_spelling_is_accepted(self):
        """``<parameter name="k">v</parameter>`` also names an argument."""
        text = (
            f'{NS}<invoke name="search">'
            f'{NS}<parameter name="q">hi{NS}</parameter>'
            f"{NS}</invoke>"
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {"q": "hi"}

    def test_truncated_invoke_is_still_parsed(self):
        text = f'{NS}<tool_call>\n{NS}<invoke name="get_weather">{NS}<location>Beij'
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "get_weather"
        assert json.loads(tool_calls[0].function["arguments"]) == {"location": "Beij"}

    def test_plain_text_untouched(self):
        content, tool_calls = parse_tool_calls("The answer is 4.", MINIMAX_TOOLS)
        assert content == "The answer is 4."
        assert tool_calls == []

    def test_qwen_dialect_still_detected(self):
        """MiniMax detection must not shadow the Qwen3 dialect."""
        qwen = (
            "<tool_call>\n<function=search>\n<parameter=q>hi</parameter>\n"
            "</function>\n</tool_call>"
        )
        _content, tool_calls = parse_tool_calls(qwen, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "search"

    def test_unsupported_dialect_is_left_in_content(self):
        """A bare <tool_call> with JSON inside is the Hermes dialect, unparsed.

        Claiming it would yield zero tool calls *and* truncate the content,
        silently losing the call; leaving the raw text visible keeps the failure
        diagnosable.
        """
        hermes = (
            '<tool_call>\n{"name": "search", "arguments": {"q": "hi"}}\n</tool_call>'
        )
        content, tool_calls = parse_tool_calls(hermes, MINIMAX_TOOLS)
        assert tool_calls == []
        assert content == hermes


class TestMiniMaxToolCallStreaming:
    def _run(self, tokens, tools=MINIMAX_TOOLS, **kwargs):
        parser = ToolCallStreamParser(tools=tools, **kwargs)
        events = []
        for token in tokens:
            events.extend(parser.process(token))
        events.extend(parser.flush())
        return events

    def test_content_then_tool_call(self):
        events = self._run(
            [
                "Let me ",
                "check.",
                "]<]minimax",  # namespace token split across chunks
                "[>[<tool_call>",
                f'{NS}<invoke name="get_weather">',
                f"{NS}<location>Bei",
                f"jing{NS}</location>",
                f"{NS}</invoke>",
                f"{NS}</tool_call>",
            ]
        )
        content = "".join(d for t, d in events if t == "content")
        assert content == "Let me check."
        starts = [d for t, d in events if t == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["function"]["name"] == "get_weather"
        assert starts[0]["index"] == 0
        assert starts[0]["id"].startswith("call_")
        args = [d for t, d in events if t == "tool_call_args"]
        assert json.loads(args[0]["function"]["arguments"]) == {"location": "Beijing"}
        assert [t for t, _ in events].count("tool_call_end") == 1

    def test_parallel_calls_emitted_incrementally(self):
        parser = ToolCallStreamParser(tools=MINIMAX_TOOLS)
        parser.process(f"{NS}<tool_call>")
        first = parser.process(_invoke("search", _param("q", "a")))
        # The first call is streamed as soon as its </invoke> arrives, without
        # waiting for the closing </tool_call>.
        assert [t for t, _ in first] == ["tool_call_start", "tool_call_args"]
        second = parser.process(_invoke("search", _param("q", "b")))
        assert [d["index"] for t, d in second if t == "tool_call_start"] == [1]
        assert parser.flush() == [("tool_call_end", None)]

    def test_truncated_stream_flushes_partial_call(self):
        events = self._run(
            [
                f"{NS}<tool_call>",
                f'{NS}<invoke name="search">',
                f"{NS}<q>hal",
            ]
        )
        starts = [d for t, d in events if t == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["function"]["name"] == "search"
        assert [t for t, _ in events].count("tool_call_end") == 1

    def test_disabled_parser_streams_everything_as_content(self):
        """``tool_choice: "none"`` must not surface tool_calls."""
        events = self._run(
            ["ok", _minimax_block(_invoke("search", _param("q", "a")))],
            enabled=False,
        )
        assert all(t == "content" for t, _ in events)
        assert "<invoke" in "".join(d for _, d in events)


# ============================================================================
# Forced tool calls (tool_choice: required / named)
# ============================================================================

# What a chat template renders into its tool instructions, which is how the
# dialect is detected.
M3_PROMPT = f"To call tools, wrap invocations in {NS}<tool_call>{NS}</tool_call>."
QWEN_PROMPT = "Emit <tool_call>\n<function=name>...</function>\n</tool_call>"
KIMI_PROMPT = "Use <|tool_calls_section_begin|> ... <|tool_calls_section_end|>"


class TestToolCallPrefill:
    """ATOM has no grammar, so a forced call is started for the model."""

    def test_minimax_required_opens_the_section(self):
        assert tool_call_prefill(M3_PROMPT) == f"{NS}<tool_call>\n"

    def test_minimax_named_also_opens_the_invoke(self):
        prefill = tool_call_prefill(M3_PROMPT, "search")
        assert prefill == f'{NS}<tool_call>\n{NS}<invoke name="search">'

    def test_qwen_dialect(self):
        assert tool_call_prefill(QWEN_PROMPT, "search") == (
            "<tool_call>\n<function=search>\n"
        )

    def test_kimi_dialect(self):
        prefill = tool_call_prefill(KIMI_PROMPT, "search")
        assert prefill.startswith("<|tool_calls_section_begin|>")
        assert "functions.search:0" in prefill

    def test_unknown_dialect_returns_none(self):
        """Falls back to prompting alone rather than inventing syntax."""
        assert tool_call_prefill("a prompt with no tool instructions") is None

    def test_prefix_plus_continuation_parses(self):
        """The prefix is not in the model output, so the parser is given it back."""
        prefill = tool_call_prefill(M3_PROMPT)
        continuation = (
            f'{NS}<invoke name="get_weather">{NS}<location>Beijing'
            f"{NS}</location>{NS}</invoke>\n{NS}</tool_call>"
        )
        _content, tool_calls = parse_tool_calls(prefill + continuation, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "get_weather"
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "location": "Beijing"
        }

    def test_named_prefix_plus_arguments_only_parses(self):
        prefill = tool_call_prefill(M3_PROMPT, "get_weather")
        continuation = (
            f"{NS}<location>Paris{NS}</location>{NS}</invoke>\n{NS}</tool_call>"
        )
        _content, tool_calls = parse_tool_calls(prefill + continuation, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "get_weather"
        assert json.loads(tool_calls[0].function["arguments"]) == {"location": "Paris"}

    def test_seeded_stream_parser_emits_the_call(self):
        parser = ToolCallStreamParser(tools=MINIMAX_TOOLS)
        assert parser.process(tool_call_prefill(M3_PROMPT)) == []
        events = parser.process(
            f'{NS}<invoke name="search">{NS}<q>hi{NS}</q>{NS}</invoke>'
        )
        assert [t for t, _ in events] == ["tool_call_start", "tool_call_args"]
        assert json.loads(events[1][1]["function"]["arguments"]) == {"q": "hi"}


# ============================================================================
# Cross-dialect protection: adding MiniMax must not change the others
# ============================================================================


class TestDialectPriority:
    """Regression guards for other models' output."""

    def test_kimi_wins_when_its_arguments_contain_a_qwen_marker(self):
        """`<function=` inside Kimi arguments must not flip the dialect."""
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>"
            '{"text": "write <function=foo> in the docs"}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "echo"
        assert "<function=foo>" in tool_calls[0].function["arguments"]

    def test_kimi_wins_when_arguments_mention_an_invoke_tag(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>"
            '{"text": "<invoke name=x>"}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "echo"

    def test_qwen_output_mentioning_invoke_still_parses_as_qwen(self):
        text = (
            "<tool_call>\n<function=search>\n"
            "<parameter=q><invoke name=x></parameter>\n</function>\n</tool_call>"
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert tool_calls[0].function["name"] == "search"

    def test_plain_prose_is_never_claimed(self):
        for text in [
            'Use <invoke name="x"> to call it.',
            "Compare a ]< b and c >[ d.",
            "The token ]<]minimax is incomplete.",
        ]:
            content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
            assert tool_calls == [], text
            assert content == text, text


class TestStreamingMatchesNonStreaming:
    """The streaming rework changed delta boundaries, never the payload.

    SSE deltas may be split anywhere, but the reassembled content and arguments
    must equal what the non-streaming parser produces — for every dialect.
    """

    def _stream(self, text, chunk=7, tools=MINIMAX_TOOLS):
        parser = ToolCallStreamParser(tools=tools)
        events = []
        for i in range(0, len(text), chunk):
            events.extend(parser.process(text[i : i + chunk]))
        events.extend(parser.flush())
        content = "".join(d for t, d in events if t == "content")
        names = [d["function"]["name"] for t, d in events if t == "tool_call_start"]
        args = [d["function"]["arguments"] for t, d in events if t == "tool_call_args"]
        return content, names, args

    KIMI = (
        "Let me check that for you."
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.exec:0<|tool_call_argument_begin|>"
        '{"cmd": "ls -la"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    QWEN = (
        "Sure, one moment.<tool_call>\n<function=search>\n"
        "<parameter=q>beijing weather</parameter>\n</function>\n</tool_call>"
    )
    PLAIN = "Compare a < b and 3 > 2; also x ]< y. Done."

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_kimi_roundtrip(self, chunk):
        content, names, args = self._stream(self.KIMI, chunk)
        ref_content, ref_calls = parse_tool_calls(self.KIMI, MINIMAX_TOOLS)
        assert content.strip() == ref_content
        assert names == [c.function["name"] for c in ref_calls]
        assert args == [c.function["arguments"] for c in ref_calls]

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_qwen_roundtrip(self, chunk):
        content, names, args = self._stream(self.QWEN, chunk)
        ref_content, ref_calls = parse_tool_calls(self.QWEN, MINIMAX_TOOLS)
        assert content.strip() == ref_content
        assert names == [c.function["name"] for c in ref_calls]
        assert args == [c.function["arguments"] for c in ref_calls]

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_trailing_newline_survives_any_split(self, chunk):
        """Case 16_15 is a streaming case: the value must not be trimmed there."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                },
            }
        ]
        text = _minimax_block(
            _invoke("write_file", _param("content", "hello\nworld\n"))
        )
        _content, _names, args = self._stream(text, chunk, tools=tools)
        assert json.loads("".join(args)) == {"content": "hello\nworld\n"}

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_composed_schema_types_survive_any_split(self, chunk):
        """Case 14_07 is a streaming case: oneOf arms must be read there too."""
        tools = TestComposedSchemas.ONEOF_TOOL
        text = _minimax_block(_invoke("ExampleFunction", _param("number", "42")))
        _content, _names, args = self._stream(text, chunk, tools=tools)
        assert json.loads("".join(args)) == {"number": 42}

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_minimax_roundtrip(self, chunk):
        content, names, args = self._stream(REAL_M3_OUTPUT, chunk)
        ref_content, ref_calls = parse_tool_calls(REAL_M3_OUTPUT, MINIMAX_TOOLS)
        assert content.strip() == ref_content
        assert names == [c.function["name"] for c in ref_calls]
        assert args == [c.function["arguments"] for c in ref_calls]

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_plain_text_is_never_altered(self, chunk):
        content, names, _args = self._stream(self.PLAIN, chunk)
        assert content == self.PLAIN
        assert names == []


# Captured from the live model: MiniMax-M3 emits <filename> as special token
# 200006, which decode(skip_special_tokens=True) erases, so the *opening* tag
# never reaches the parser while </filename> survives as ordinary text.
M3_LOST_OPENING_TAG = (
    f"{NS}<tool_call>\n"
    f'{NS}<invoke name="save_snippet">'
    f"{NS}page.html{NS}</filename>"
    f'{NS}<code><div class="hero"><h1>Hi</h1></div>{NS}</code>'
    f"{NS}</invoke>\n"
    f"{NS}</tool_call>"
)

SNIPPET_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_snippet",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["filename", "code"],
            },
        },
    }
]


class TestLostOpeningTagRecovery:
    """An argument whose opening tag was erased by the tokenizer.

    MiniMax names arguments by element, and many of MiniMax-M3's added tokens
    are tag-shaped (<filename>, <filepath>, <file_content>, ...). They are
    marked special, so the engine's decode(skip_special_tokens=True) deletes
    them. The closing tag survives and still carries the name, so the argument
    must be recovered rather than silently dropped.
    """

    def test_required_argument_is_recovered_not_dropped(self):
        _content, tool_calls = parse_tool_calls(M3_LOST_OPENING_TAG, SNIPPET_TOOLS)
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function["arguments"])
        assert args == {
            "filename": "page.html",
            "code": '<div class="hero"><h1>Hi</h1></div>',
        }

    def test_recovered_without_a_schema_too(self):
        """Text directly inside <invoke> is never a value, so recovery is safe."""
        _content, tool_calls = parse_tool_calls(M3_LOST_OPENING_TAG, None)
        assert (
            json.loads(tool_calls[0].function["arguments"])["filename"] == "page.html"
        )

    @pytest.mark.parametrize("chunk", [1, 3, 7, 13, 1000])
    def test_streaming_recovers_it_identically(self, chunk):
        parser = ToolCallStreamParser(tools=SNIPPET_TOOLS)
        events = []
        for i in range(0, len(M3_LOST_OPENING_TAG), chunk):
            events.extend(parser.process(M3_LOST_OPENING_TAG[i : i + chunk]))
        events.extend(parser.flush())
        args = "".join(
            d["function"]["arguments"] for t, d in events if t == "tool_call_args"
        )
        _content, ref = parse_tool_calls(M3_LOST_OPENING_TAG, SNIPPET_TOOLS)
        assert json.loads(args) == json.loads(ref[0].function["arguments"])

    def test_nested_property_is_recovered_when_the_schema_declares_it(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "w",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "meta": {
                                "type": "object",
                                "properties": {
                                    "filename": {"type": "string"},
                                    "n": {"type": "integer"},
                                },
                            }
                        },
                    },
                },
            }
        ]
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="w">'
            f"{NS}<meta>{NS}a.txt{NS}</filename>{NS}<n>3{NS}</n>{NS}</meta>"
            f"{NS}</invoke>"
        )
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "meta": {"filename": "a.txt", "n": 3}
        }

    def test_a_stray_closing_tag_inside_a_value_stays_literal(self):
        """Recovery must not fire on markup that is genuinely part of a value."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "say",
                    "parameters": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                    },
                },
            }
        ]
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="say">'
            f"{NS}<msg>closing </b> tag{NS}</msg>"
            f"{NS}</invoke>"
        )
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "msg": "closing </b> tag"
        }

    def test_two_consecutive_lost_openers(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "w",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "file_content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="w">'
            f"{NS}a.txt{NS}</filename>"
            f"{NS}hello{NS}</file_content>"
            f"{NS}</invoke>"
        )
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {
            "filename": "a.txt",
            "file_content": "hello",
        }


class TestComposedSchemas:
    """Arguments declared inside oneOf / anyOf / allOf still get their type.

    Guards MiniMax conformance case 14_07: a tool whose ``parameters`` puts its
    ``properties`` in ``oneOf`` arms rather than at the top level. Reading only
    the top level finds no properties at all, so every value stays the string it
    arrived as and ``number`` comes back as ``"42"``.
    """

    ONEOF_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "ExampleFunction",
                "parameters": {
                    "type": "object",
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"number": {"type": "number"}},
                        },
                        {
                            "type": "object",
                            "properties": {
                                "stringList": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "numberList": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                }
                            },
                        },
                    ],
                },
            },
        }
    ]

    def _args(self, *params: str):
        text = _minimax_block(_invoke("ExampleFunction", *params))
        _content, tool_calls = parse_tool_calls(text, self.ONEOF_TOOL)
        return json.loads(tool_calls[0].function["arguments"])

    def test_scalar_in_a_oneof_arm_is_typed(self):
        assert self._args(_param("number", "42")) == {"number": 42}

    def test_string_array_in_a_oneof_arm_stays_strings(self):
        params = _param("item", "12") + _param("item", "34")
        assert self._args(_param("stringList", params)) == {"stringList": ["12", "34"]}

    def test_number_array_in_a_oneof_arm_is_typed(self):
        params = _param("item", "12") + _param("item", "34")
        assert self._args(_param("numberList", params)) == {"numberList": [12, 34]}

    def test_anyof_arms_are_read_too(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "anyOf": [{"properties": {"n": {"type": "integer"}}}]
                    },
                },
            }
        ]
        text = _minimax_block(_invoke("f", _param("n", "7")))
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {"n": 7}

    def test_allof_arms_are_merged(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "allOf": [
                            {"properties": {"a": {"type": "integer"}}},
                            {"properties": {"b": {"type": "boolean"}}},
                        ],
                    },
                },
            }
        ]
        text = _minimax_block(_invoke("f", _param("a", "1"), _param("b", "true")))
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {"a": 1, "b": True}

    def test_a_nested_object_declared_in_an_arm_is_typed(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "meta": {
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "properties": {"n": {"type": "integer"}},
                                    }
                                ]
                            }
                        },
                    },
                },
            }
        ]
        text = _minimax_block(_invoke("f", _param("meta", _param("n", "5"))))
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {"meta": {"n": 5}}

    def test_the_outer_schema_wins_over_an_arm(self):
        """A name declared at both levels keeps the enclosing declaration."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"v": {"type": "string"}},
                        "oneOf": [{"properties": {"v": {"type": "integer"}}}],
                    },
                },
            }
        ]
        text = _minimax_block(_invoke("f", _param("v", "42")))
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {"v": "42"}


class TestSchemaDrivenValues:
    """The declared type decides an argument's shape, not its punctuation."""

    @pytest.mark.parametrize(
        "value",
        [
            '<div class="hero"><h1>Hi</h1></div>',
            "template<class T> struct Box { T v; };",
            "assert a < b and b > c",
            "if (x<y) { return a>b; }",
            "<a><b></b></a>",
        ],
    )
    def test_declared_string_survives_markup_verbatim(self, value):
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="save_snippet">'
            f"{NS}<filename>a.txt{NS}</filename>"
            f"{NS}<code>{value}{NS}</code>"
            f"{NS}</invoke>"
        )
        _content, tool_calls = parse_tool_calls(text, SNIPPET_TOOLS)
        args = json.loads(tool_calls[0].function["arguments"])
        assert args["code"] == value, "declared string must not be tree-parsed"
        assert args["filename"] == "a.txt"

    def test_declared_empty_array_is_a_list_not_a_string(self):
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="search">{NS}<tags>{NS}</tags>{NS}</invoke>'
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {"tags": []}

    def test_declared_empty_object_is_a_dict(self):
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="get_weather">{NS}<opts>{NS}</opts>{NS}</invoke>'
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {"opts": {}}

    def test_lone_scalar_for_a_declared_array_is_wrapped(self):
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="search">{NS}<tags>china{NS}</tags>{NS}</invoke>'
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {"tags": ["china"]}

    def test_union_type_with_null_resolves_to_the_real_type(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": ["integer", "null"]}},
                    },
                },
            }
        ]
        text = f"{NS}<tool_call>" f'{NS}<invoke name="f">{NS}<n>7{NS}</n>{NS}</invoke>'
        _content, tool_calls = parse_tool_calls(text, tools)
        assert json.loads(tool_calls[0].function["arguments"]) == {"n": 7}

    def test_unschemad_argument_still_uses_the_shape_heuristic(self):
        """No schema -> the XML shape is all the parser has to go on."""
        text = (
            f"{NS}<tool_call>"
            f'{NS}<invoke name="unknown">{NS}<opts>{NS}<a>1{NS}</a>{NS}</opts>'
            f"{NS}</invoke>"
        )
        _content, tool_calls = parse_tool_calls(text, MINIMAX_TOOLS)
        assert json.loads(tool_calls[0].function["arguments"]) == {"opts": {"a": "1"}}
