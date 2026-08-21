# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for tool call parsing."""

from typing import ClassVar

import pytest

from atom.entrypoints.openai.tool_parser import (
    ToolCall,
    ToolCallStreamParser,
    parse_tool_calls,
)
from atom.entrypoints.openai.tool_parser.glm_tool_parser import GlmParser
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser

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
        content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
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
        content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
        assert content == "Let me search."
        assert len(tool_calls) == 2
        assert tool_calls[0].function["name"] == "search"
        assert tool_calls[1].function["name"] == "fetch"

    def test_no_tool_calls(self):
        text = "Just a regular response."
        content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
        assert content == "Just a regular response."
        assert len(tool_calls) == 0

    def test_empty_content_with_tool_call(self):
        text = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{"cmd": "echo hi"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
        assert content == ""
        assert len(tool_calls) == 1

    def test_unclosed_section(self):
        text = (
            "Here:"
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.exec:0<|tool_call_argument_begin|>{"cmd": "ls"}<|tool_call_end|>'
        )
        content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
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
        content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
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
        _content, tool_calls = parse_tool_calls(text, parser_cls=KimiParser)
        assert len(tool_calls) == 1
        assert tool_calls[0].function["arguments"] == args


# ============================================================================
# ToolCallStreamParser Tests
# ============================================================================


class TestToolCallStreamParser:
    """Tests for the ToolCallStreamParser streaming state machine."""

    def _run_parser(self, tokens, parser_cls=KimiParser):
        """Helper: run tokens through parser and return all events.

        The format is given, as the server gives it: resolved once from the
        chat template at startup rather than guessed from the output. These
        cases are all Kimi's section syntax, so that is the default.
        """
        parser = ToolCallStreamParser(parser_cls=parser_cls)
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


class TestAParserOnItsOwnDoesNotEatText:
    """Driven directly, because the facade cannot reach these states.

    `ToolCallStreamParser` reads ahead over the format's markers itself, so a
    parser is only ever constructed once a complete marker has arrived — its
    own pre-region path is unreachable from there. The property suite runs
    through the facade and therefore cannot see this, which is exactly how a
    six-character loss in `KimiParser.flush` survived it.
    """

    def test_kimi_releases_a_partial_marker_it_was_still_holding(self):
        p = KimiParser()
        out = p.process("hello <|tool")
        out += p.flush()
        assert "".join(d for k, d in out if k == "content") == "hello <|tool"

    def test_kimi_releases_a_section_that_held_no_call(self):
        """A start marker is not a promise, for this format either."""
        text = "see <|tool_calls_section_begin|> and nothing else"
        p = KimiParser()
        out = p.process(text)
        out += p.flush()
        delivered = "".join(d for k, d in out if k == "content")
        assert "and nothing else" in delivered
        assert not [k for k, _ in out if k.startswith("tool_call_")]

    def test_kimi_k3_keeps_prose_after_a_tools_token_it_did_not_use(self):
        text = "the token <|open|>tools<|sep|> opens a section. Nothing follows."
        content, calls = KimiK3Parser.parse(text, None)
        assert calls == []
        assert "Nothing follows." in content


# ============================================================================
# What counts as a tool name
# ============================================================================


class TestOnlyAnIdentifierIsAToolName:
    """GLM's unterminated branch takes everything after `<tool_call>` as the
    name, so the name check is the only thing between prose and a fabricated
    call. It has to reject prose without rejecting names models really use.
    """

    def _call(self, name):
        text = (
            f"<tool_call>{name}"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        )
        return GlmParser.parse(text, None)[1]

    @pytest.mark.parametrize(
        "name",
        [
            "get_weather",
            "7z_extract",  # OpenAI's grammar allows a leading digit
            "天气查询",  # and nothing forbids a CJK name on a Chinese family
            "read-file",
            "fs.read",
            "x",
        ],
    )
    def test_a_legal_name_is_accepted(self, name):
        calls = self._call(name)
        assert [c.function["name"] for c in calls] == [name]

    @pytest.mark.parametrize(
        "name",
        [
            " followed by the name. Hope that helps!",
            '{"name": "get_weather", "arguments": {}}',  # Hermes-style JSON
            "two words",
        ],
    )
    def test_prose_is_not_a_name(self, name):
        assert self._call(name) == []


class TestATruncatedCallIsNotContent:
    """A call cut off by `max_tokens` parses to nothing, and "nothing parsed"
    was the test for whether a section had opened -- so the half-written
    payload was kept and shipped as the answer.
    """

    TRUNCATED = (
        "I will look it up."
        '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>Paris<|close|>argument'
    )

    def test_the_partial_payload_does_not_reach_the_client(self):
        content, calls = KimiK3Parser.parse(self.TRUNCATED, None)
        assert calls == []
        assert content == "I will look it up."

    def test_an_answer_that_only_names_the_token_still_keeps_its_tail(self):
        """The case the gate was added for, which must keep working."""
        text = "the token <|open|>tools<|sep|> opens a section. Nothing follows."
        content, calls = KimiK3Parser.parse(text, None)
        assert calls == [] and "Nothing follows." in content

    def test_a_complete_call_still_truncates_there(self):
        text = (
            "Looking._"
            '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>'
            '<|open|>argument key="city"<|sep|>Paris<|close|>argument'
            "<|close|>call"
        )
        content, calls = KimiK3Parser.parse(text, None)
        assert [c.function["name"] for c in calls] == ["get_weather"]
        assert content == "Looking._"


class TestTheAnnouncedNameIsTheOneThatParses:
    """A name sent early has to be the *first* call's, in wire order.

    GLM's peek required `<arg_key>` after the name, so it skipped a call that
    takes no arguments and announced the one after it. `parse` then returned
    them in wire order and the mismatch raised -- out of `flush`, on a live
    SSE generator with no `except` above it, from well-formed output. Zero-
    argument tools are ordinary.
    """

    TOOLS: ClassVar[list] = [
        {"type": "function", "function": {"name": "alpha"}},
        {"type": "function", "function": {"name": "beta"}},
    ]

    def _drive(self, text):
        stream = ToolCallStreamParser(parser_cls=GlmParser)
        stream.tools = self.TOOLS
        events = []
        for i in range(0, len(text), 4):
            events += stream.process(text[i : i + 4])
        return events + stream.flush()

    def test_a_zero_argument_call_before_a_real_one(self):
        events = self._drive(
            "<tool_call>alpha</tool_call>"
            "<tool_call>beta<arg_key>city</arg_key><arg_value>Q</arg_value></tool_call>"
        )
        names = [d["function"]["name"] for k, d in events if k == "tool_call_start"]
        assert names == ["alpha", "beta"]

    def test_the_peek_reads_a_zero_argument_call(self):
        assert GlmParser.peek_name("<tool_call>alpha</tool_call>") == "alpha"

    def test_the_peek_reads_the_first_of_two(self):
        assert (
            GlmParser.peek_name(
                "<tool_call>alpha</tool_call><tool_call>beta<arg_key>c</arg_key>"
            )
            == "alpha"
        )


class TestProseIsNotATruncatedCall:
    """The unclosed-region branch exists for a call cut off by `max_tokens`.

    It cannot tell that from an answer explaining how to call a tool, and used
    to accept both: an agentic client executed `get_weather({})` and the rest
    of the sentence was deleted. Prose has to fail two tests -- a name the
    request declared, and nothing after the name but this format's own next
    token -- because prose can name a real tool.
    """

    TOOLS: ClassVar[list] = [{"type": "function", "function": {"name": "get_weather"}}]

    @pytest.mark.parametrize(
        "parser, text",
        [
            (
                QwenXmlParser,
                (
                    "To call it the model writes <tool_call>"
                    "<function=get_weather> and then the parameters."
                ),
            ),
            (
                GlmParser,
                "To call it write <tool_call>get_weather and then the keys follow.",
            ),
        ],
        ids=["qwen", "glm"],
    )
    def test_prose_naming_a_declared_tool_is_not_a_call(self, parser, text):
        content, calls = parser.parse(text, self.TOOLS)
        assert calls == [], f"fabricated {[c.function['name'] for c in calls]}"
        assert content == text, "the answer was truncated at the quoted tag"

    @pytest.mark.parametrize(
        "parser, text",
        [
            (
                QwenXmlParser,
                "Sure. <tool_call><function=get_weather><parameter=city>Par",
            ),
            (QwenXmlParser, "Sure. <tool_call><function=get_weather>"),
            (
                GlmParser,
                "Sure. <tool_call>get_weather<arg_key>city</arg_key><arg_value>Pa",
            ),
        ],
        ids=["qwen-mid-param", "qwen-after-name", "glm-mid-arg"],
    )
    def test_a_genuinely_truncated_call_still_parses(self, parser, text):
        _, calls = parser.parse(text, self.TOOLS)
        assert [c.function["name"] for c in calls] == ["get_weather"]

    def test_an_undeclared_name_is_never_salvaged(self):
        text = "Sure. <tool_call><function=made_up><parameter=x>1"
        content, calls = QwenXmlParser.parse(text, self.TOOLS)
        assert calls == [] and content == text


class TestKimiKeepsWhatItDidNotParse:
    """A start marker is not a promise, for this format too.

    State 1 truncated the buffer at the section end and moved to a terminal
    state, so an answer quoting both section tokens lost its body *and*
    everything after it -- and the `flush` fallback that was meant to cover
    this could not see it, because the bytes were already gone. Measured: 26
    of 135 characters at four-character chunks, 135 in one shot.
    """

    QUOTES_BOTH = (
        "Kimi emits a tool call as <|tool_calls_section_begin|> then one entry "
        "per call and finally <|tool_calls_section_end|>. Hope that helps!"
    )

    @staticmethod
    def _stream(text, size):
        parser = ToolCallStreamParser(parser_cls=KimiParser)
        events = []
        for i in range(0, len(text), size):
            events += parser.process(text[i : i + size])
        events += parser.flush()
        return "".join(d for k, d in events if k == "content"), [
            k for k, _ in events if k.startswith("tool_call")
        ]

    @pytest.mark.parametrize("size", [1, 2, 4, 17, 999])
    def test_every_byte_survives_at_every_chunk_size(self, size):
        delivered, calls = self._stream(self.QUOTES_BOTH, size)
        assert delivered == self.QUOTES_BOTH
        assert calls == [], "a quoted section token is not a tool call"

    def test_it_agrees_with_the_non_streaming_path(self):
        delivered, _ = self._stream(self.QUOTES_BOTH, 4)
        assert delivered == parse_tool_calls(self.QUOTES_BOTH, parser_cls=KimiParser)[0]

    def test_text_after_a_real_section_still_arrives(self):
        """`state = 2` discarded the rest of the stream unconditionally."""
        text = (
            "Sure. <|tool_calls_section_begin|><|tool_call_begin|>"
            'functions.get_weather:0<|tool_call_argument_begin|>{"city":"Paris"}'
            "<|tool_call_end|><|tool_calls_section_end|> Done."
        )
        delivered, calls = self._stream(text, 4)
        assert "Sure." in delivered and "Done." in delivered
        assert "tool_call_start" in calls
