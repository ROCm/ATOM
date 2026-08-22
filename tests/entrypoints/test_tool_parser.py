# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for tool call parsing."""

from typing import ClassVar

import json

import pytest

from atom.entrypoints.openai.tool_parser.schema import ParamTypes
from atom.entrypoints.openai.tool_parser.stream import _resolved_tools

from atom.entrypoints.openai.tool_parser import (
    ToolCall,
    ToolCallStreamParser,
    parse_tool_calls,
)
from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
from atom.entrypoints.openai.tool_parser.glm_tool_parser import GlmParser
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.minimax_tool_parser import MiniMaxParser
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser


def early_name(parser, region: str, tools=None) -> str | None:
    """The name the engine would announce for `region`.

    There is no separate peek to ask: the announcement is the first call of
    `parse_region` over the bytes so far, with `at_end=False`. Naming that
    here rather than in each test keeps these tests about the formats.
    """
    calls = parser.parse_region(region, tools, at_end=False).calls
    return calls[0].function["name"] if calls else None


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
        p = ToolCallStreamParser(parser_cls=KimiParser)
        out = p.process("hello <|tool")
        out += p.flush()
        assert "".join(d for k, d in out if k == "content") == "hello <|tool"

    def test_kimi_releases_a_section_that_held_no_call(self):
        """A start marker is not a promise, for this format either."""
        text = "see <|tool_calls_section_begin|> and nothing else"
        p = ToolCallStreamParser(parser_cls=KimiParser)
        out = p.process(text)
        out += p.flush()
        delivered = "".join(d for k, d in out if k == "content")
        assert "and nothing else" in delivered
        assert not [k for k, _ in out if k.startswith("tool_call_")]

    def test_kimi_k3_keeps_prose_after_a_tools_token_it_did_not_use(self):
        text = "the token <|open|>tools<|sep|> opens a section. Nothing follows."
        content, calls = parse_tool_calls(text, None, KimiK3Parser)
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
        return parse_tool_calls(text, None, GlmParser)[1]

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


class TestATruncatedCallIsDeliveredRatherThanDeleted:
    """A call cut off by `max_tokens` parses to nothing, and a region that
    parses to nothing is released unchanged -- for this format as for every
    other, which is the change.

    K3 used to cut the answer at the tools marker instead, on a second opener
    regex that accepted shapes the call regex rejects, and the two ways of
    getting that wrong were opposite: an answer *quoting* an opener lost 62
    characters, and a truncated call kept its half-written payload with the
    dangling `<|close|>argument` still in it. Both are now the same rule.

    Kimi already behaved this way (a section with no complete entry comes back
    whole), so this is the two token formats agreeing rather than a new
    policy. The four XML-ish formats do not reach it: they salvage a truncated
    call into a real one, so there is no half-written markup to show.
    """

    TRUNCATED = (
        "I will look it up."
        '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>Paris<|close|>argument'
    )

    def test_the_partial_payload_is_delivered_not_dropped(self):
        content, calls = parse_tool_calls(self.TRUNCATED, None, KimiK3Parser)
        assert calls == []
        assert content == self.TRUNCATED, "bytes were deleted with no event"

    def test_an_answer_that_only_names_the_token_still_keeps_its_tail(self):
        """The case the gate was added for, which must keep working."""
        text = "the token <|open|>tools<|sep|> opens a section. Nothing follows."
        content, calls = parse_tool_calls(text, None, KimiK3Parser)
        assert calls == [] and "Nothing follows." in content

    def test_a_complete_call_still_truncates_there(self):
        text = (
            "Looking._"
            '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>'
            '<|open|>argument key="city"<|sep|>Paris<|close|>argument'
            "<|close|>call"
        )
        content, calls = parse_tool_calls(text, None, KimiK3Parser)
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

    def test_the_early_name_reads_a_zero_argument_call(self):
        assert early_name(GlmParser, "<tool_call>alpha</tool_call>") == "alpha"

    def test_the_early_name_reads_the_first_of_two(self):
        assert (
            early_name(
                GlmParser,
                "<tool_call>alpha</tool_call><tool_call>beta<arg_key>c</arg_key>",
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
            # `</tool_call>` closes the *outer* wrapper, so the `<function=`
            # block is still open and this is prose, not a zero-argument
            # call. Qwen's peek used to accept it as a follower while its
            # parse did not; unifying them onto one constant would have made
            # both accept it, which is worse -- a phantom dispatch rather
            # than a dangling name.
            (
                QwenXmlParser,
                (
                    "A zero-arg call is written <tool_call><function=get_weather>"
                    "</tool_call>, like that."
                ),
            ),
            (
                DsmlParser,
                (
                    'You emit <invoke name="get_weather"> and inside it a '
                    '<\uff5cDSML\uff5cparameter name="city">Paris'
                    "</\uff5cDSML\uff5cparameter> line."
                ),
            ),
            (
                MiniMaxParser,
                (
                    "To call it you emit ]<]minimax[>[<invoke "
                    'name="get_weather"> and then a '
                    "]<]minimax[>[<city>Paris</city> line."
                ),
            ),
        ],
        ids=["qwen", "glm", "qwen-outer-closer", "dsml", "minimax"],
    )
    def test_prose_naming_a_declared_tool_is_not_a_call(self, parser, text):
        content, calls = parse_tool_calls(text, self.TOOLS, parser)
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
            # No complete `<arg_key>` yet, so the name has to be cut at the
            # `<` rather than run to the end of the region.
            (GlmParser, "Sure. <tool_call>get_weather<arg_k"),
            (GlmParser, "Sure. <tool_call>\nget_weather\n"),
            (
                DsmlParser,
                (
                    'Sure. <invoke name="get_weather">'
                    '<\uff5cDSML\uff5cparameter name="city">Par'
                ),
            ),
            (DsmlParser, 'Sure. <invoke name="get_weather">'),
            (
                MiniMaxParser,
                (
                    'Sure. ]<]minimax[>[<invoke name="get_weather">'
                    "]<]minimax[>[<city>Par"
                ),
            ),
        ],
        ids=[
            "qwen-mid-param",
            "qwen-after-name",
            "glm-mid-arg",
            "glm-mid-arg-key",
            "glm-newline-before-name",
            "dsml-mid-param",
            "dsml-after-name",
            "minimax-mid-param",
        ],
    )
    def test_a_genuinely_truncated_call_still_parses(self, parser, text):
        _, calls = parse_tool_calls(text, self.TOOLS, parser)
        assert [c.function["name"] for c in calls] == ["get_weather"]

    def test_an_undeclared_name_is_never_salvaged(self):
        text = "Sure. <tool_call><function=made_up><parameter=x>1"
        content, calls = parse_tool_calls(text, self.TOOLS, QwenXmlParser)
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

    REAL_SECTION = (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        'functions.get_weather:0<|tool_call_argument_begin|>{"city":"Paris"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    @pytest.mark.parametrize("size", [1, 4, 17, 999])
    def test_text_after_a_real_section_still_arrives(self, size):
        """`state = 2` discarded the rest of the stream unconditionally.

        Parametrised over the chunk size because replacing that state fixed
        only the split case: state 0 took the remainder after the marker and
        returned without looking for the section end, so a section arriving
        whole in one chunk still lost everything after it. Under load that is
        the common case, not the rare one -- `merge_chunk` coalesces the
        backlog into exactly these large chunks.
        """
        text = "Sure. " + self.REAL_SECTION + " Done."
        delivered, calls = self._stream(text, size)
        assert "Sure." in delivered and "Done." in delivered
        assert "tool_call_start" in calls

    @pytest.mark.parametrize("size", [1, 4, 999])
    def test_a_quoted_section_after_a_real_call_is_still_not_a_call(self, size):
        """The not-a-promise branch is per section, not per stream.

        It was gated on `emitted_calls`, which is cumulative, so from the
        first real call onwards every later section read as fulfilled and its
        body was deleted -- the branch became dead code exactly when the
        format started being used.
        """
        prose = (
            " The tokens are <|tool_calls_section_begin|> and "
            "<|tool_calls_section_end|>, in case you wondered."
        )
        delivered, calls = self._stream(self.REAL_SECTION + prose, size)
        assert delivered == prose, "the second section was eaten"
        assert calls.count("tool_call_end") == 1


class TestTheNameTheModelWroteWins:
    """DSML infers a dropped tool name from the parameters. Only when it has to.

    The inference exists for the documented V4-Flash malform that drops the
    `<invoke>` wrapper entirely. It was also reached by a call the model was
    cut off inside -- whose name is written right there in the opener -- and
    scored a *different* declared tool for it, because that one happened to
    share more parameters. Two consequences: the client is handed the wrong
    tool, and `peek_name` reads the same opener, so the streaming path had
    already announced the name the parse then contradicted.
    """

    TOOLS: ClassVar[list] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "tz": {"type": "string"},
                    },
                },
            },
        },
    ]
    D = "｜DSML｜"

    def test_a_truncated_call_keeps_the_name_in_its_opener(self):
        text = (
            f'<invoke name="get_weather">'
            f'<{self.D}parameter name="city">Paris</{self.D}parameter>'
            f'<{self.D}parameter name="tz">UTC</{self.D}parameter>'
        )
        _, calls = parse_tool_calls(text, self.TOOLS, DsmlParser)
        assert [c.function["name"] for c in calls] == ["get_weather"]

    def test_the_early_name_and_the_parse_agree_on_it(self):
        text = (
            f'<invoke name="get_weather">'
            f'<{self.D}parameter name="city">Paris</{self.D}parameter>'
            f'<{self.D}parameter name="tz">UTC</{self.D}parameter>'
        )
        _, calls = parse_tool_calls(text, self.TOOLS, DsmlParser)
        assert early_name(DsmlParser, text, self.TOOLS) == calls[0].function["name"]

    def test_the_wrapper_less_malform_still_infers(self):
        """The shape the inference was written for, unchanged."""
        text = (
            f"<{self.D}tool_calls>"
            f'<{self.D}parameter name="tz">UTC</{self.D}parameter>'
            f'<{self.D}parameter name="city">Paris</{self.D}parameter>'
        )
        _, calls = parse_tool_calls(text, self.TOOLS, DsmlParser)
        assert [c.function["name"] for c in calls] == ["get_time"]


class TestKimiK3KeepsAQuotedOpener:
    """A start marker is not a promise -- the one format that had no such branch.

    `parse` cuts the answer at a call opener, and the regex it cuts on accepts
    openers the call regex rejects. An answer quoting one therefore lost
    everything from that point: 62 characters, no event, `finish_reason`
    still `stop`.
    """

    QUOTED = (
        "<|open|>response<|sep|>To call it the model writes "
        '<|open|>call tool="get_weather" index="N"<|sep|> and then the '
        "arguments. That is the whole trick.<|close|>response<|sep|>"
    )
    TRUNCATED = (
        '<|open|>tools<|sep|><|open|>call tool="get_weather" index="0"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Par'
    )

    @staticmethod
    def _stream(text, size):
        parser = ToolCallStreamParser(parser_cls=KimiK3Parser)
        events = []
        for i in range(0, len(text), size):
            events += parser.process(text[i : i + size])
        events += parser.flush()
        return "".join(d for k, d in events if k == "content"), [
            k for k, _ in events if k.startswith("tool_call")
        ]

    @pytest.mark.parametrize("size", [1, 5, 999])
    def test_the_answer_survives_the_quotation(self, size):
        delivered, _ = self._stream(self.QUOTED, size)
        assert "That is the whole trick." in delivered

    def test_both_paths_deliver_the_same_text(self):
        delivered, _ = self._stream(self.QUOTED, 5)
        assert delivered == parse_tool_calls(self.QUOTED, parser_cls=KimiK3Parser)[0]

    def test_a_real_truncated_call_is_delivered_whole(self):
        """The branch this shares with the quotation: no call parsed, so the
        bytes are released rather than cut away. See
        `TestATruncatedCallIsDeliveredRatherThanDeleted` for why that is now
        the same answer for both shapes."""
        content, calls = parse_tool_calls(self.TRUNCATED, None, KimiK3Parser)
        assert calls == []
        assert content == self.TRUNCATED

    def test_a_region_that_produced_nothing_leaves_no_announcement_behind(self):
        """An announcement is per region. Carried past the region it was made
        in, it was matched against the *next* region's parse and reported as
        a mismatch.

        Driven through the engine, which is where the announcement lives now,
        and with a shape where the region genuinely produces no call: this
        format cannot salvage a cut-off one.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        parser = ToolCallStreamParser(tools=tools, parser_cls=KimiK3Parser)
        parser.process(self.TRUNCATED)
        parser.flush()
        assert parser._announced is None


class TestAFollowerHasToHaveArrived:
    """`peek_name` requires the next token whole; `parse` accepts a prefix.

    The two run at different moments and that is the whole difference. `parse`
    runs at end of stream, where a token cut off part-way is all there will
    ever be -- a call truncated by `max_tokens`. `peek_name` runs mid-stream,
    where a prefix means "not yet".

    Sharing one prefix-accepting test made a chunk boundary decide: `<` is a
    prefix of `<parameter=`, so the same prose announced a tool at chunk sizes
    1 and 2 and stayed silent at 5. Announced, it reaches the client as a
    dispatchable zero-argument call.
    """

    PROSE = "the model writes <tool_call><function=get_weather><br> and stops"
    TOOLS: ClassVar[list] = [{"type": "function", "function": {"name": "get_weather"}}]

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 17, 999])
    def test_prose_never_announces_at_any_chunk_size(self, size):
        parser = ToolCallStreamParser(tools=self.TOOLS, parser_cls=QwenXmlParser)
        events = []
        for i in range(0, len(self.PROSE), size):
            events += parser.process(self.PROSE[i : i + size])
        events += parser.flush()
        assert [k for k, _ in events if k == "tool_call_start"] == []

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 17, 999])
    def test_a_real_call_still_announces_at_any_chunk_size(self, size):
        text = (
            "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
            "</function></tool_call>"
        )
        parser = ToolCallStreamParser(tools=self.TOOLS, parser_cls=QwenXmlParser)
        events = []
        for i in range(0, len(text), size):
            events += parser.process(text[i : i + size])
        events += parser.flush()
        starts = [d["function"]["name"] for k, d in events if k == "tool_call_start"]
        assert starts == ["get_weather"]

    def test_a_call_cut_off_inside_its_own_token_still_parses(self):
        """The other half: `parse` must keep accepting a partial follower."""
        _, calls = parse_tool_calls(
            "<tool_call><function=get_weather><par", self.TOOLS, QwenXmlParser
        )
        assert [c.function["name"] for c in calls] == ["get_weather"]


class TestGlmReadsPastAnOpenerThatCarriesNoCall:
    """A `<tool_call>` with nothing usable behind it is not the end of the region.

    The body of a call cannot contain another `<tool_call>` -- that tag is
    what opens one. Matching non-greedily from the *first* opener to the first
    close ignored that: the region below produced a "name" of everything up to
    the second opener, which is not an identifier, and `finditer` then resumed
    past the real call and found nothing at all. An answer that quotes the tag
    and then calls for real is the same shape.

    The early name reads the same enumeration, so whatever the parse finds
    here is what gets announced -- which is the property, rather than any
    particular answer to "how many calls are in this string".
    """

    TOOLS: ClassVar[list] = [{"type": "function", "function": {"name": "get_weather"}}]
    UNUSABLE_FIRST = "<tool_call><arg_key><tool_call>get_weather<arg_key>"

    def test_the_call_behind_the_unusable_opener_is_found(self):
        _, calls = parse_tool_calls(self.UNUSABLE_FIRST, self.TOOLS, GlmParser)
        assert [c.function["name"] for c in calls] == ["get_weather"]

    def test_and_the_early_name_is_that_same_call(self):
        _, calls = parse_tool_calls(self.UNUSABLE_FIRST, self.TOOLS, GlmParser)
        assert early_name(GlmParser, self.UNUSABLE_FIRST, self.TOOLS) == (
            calls[0].function["name"] if calls else None
        )

    @pytest.mark.parametrize(
        "text",
        [
            (
                "<tool_call>get_weather<arg_key>city</arg_key>"
                "<arg_value>Paris</arg_value></tool_call>"
            ),
            "<tool_call>get_weather</tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Pa",
        ],
        ids=["with-args", "zero-arg", "cut-mid-arg"],
    )
    def test_a_real_first_call_is_still_named(self, text):
        assert early_name(GlmParser, text, self.TOOLS) == "get_weather"


class TestBothPathsAgreeOnWhatFollowsACall:
    """Text after a section, and a section marker with nothing behind it."""

    TOOLS: ClassVar[list] = [{"type": "function", "function": {"name": "get_weather"}}]
    SECTION = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city":"Paris"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )

    @staticmethod
    def _stream(text, size):
        parser = ToolCallStreamParser(parser_cls=KimiParser)
        events = []
        for i in range(0, len(text), size):
            events += parser.process(text[i : i + size])
        events += parser.flush()
        return "".join(d for k, d in events if k == "content")

    @pytest.mark.parametrize("size", [1, 5, 999])
    def test_the_tail_after_a_section_survives_both_ways(self, size):
        """`parse` truncated at the section; the streaming path stopped doing
        so when its terminal state went, leaving the two disagreeing."""
        text = self.SECTION + "tail text"
        assert (
            self._stream(text, size)
            == parse_tool_calls(text, self.TOOLS, KimiParser)[0]
        )
        assert "tail text" in self._stream(text, size)

    @pytest.mark.parametrize("size", [1, 5, 999])
    def test_an_answer_ending_on_the_marker_keeps_it(self, size):
        """`elif self.buf` skipped the recovery when nothing followed the
        marker, so all 29 characters of it went missing."""
        text = "hello <|tool_calls_section_begin|>"
        assert (
            self._stream(text, size)
            == parse_tool_calls(text, self.TOOLS, KimiParser)[0]
            == text
        )


class TestMiniMaxCutsAtItsOwnCall:
    """Content is what precedes the call, and the call has two openers.

    Cutting only at `<tool_call>` -- which this format's primary ns_token
    shape does not contain -- left the entire `<invoke>` markup in `content`
    *alongside* the parsed call, so the user was shown raw XML while the
    streaming path showed nothing. And `<invoke name="` was not a scanner
    marker at all, so a bare invoke was a call one way and text the other.
    """

    TOOLS: ClassVar[list] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    NS = "]<]minimax[>["

    def _stream(self, text, size=7):
        parser = ToolCallStreamParser(tools=self.TOOLS, parser_cls=MiniMaxParser)
        events = []
        for i in range(0, len(text), size):
            events += parser.process(text[i : i + size])
        events += parser.flush()
        return (
            "".join(d for k, d in events if k == "content"),
            [d["function"]["name"] for k, d in events if k == "tool_call_start"],
        )

    def test_the_ns_token_call_leaves_no_markup_in_content(self):
        text = f'{self.NS}<invoke name="get_weather"><city>Paris</city></invoke>'
        content, calls = parse_tool_calls(text, self.TOOLS, MiniMaxParser)
        assert [c.function["name"] for c in calls] == ["get_weather"]
        assert "<invoke" not in content, f"raw markup shown to the user: {content!r}"
        assert content == self._stream(text)[0]

    @pytest.mark.parametrize("size", [1, 5, 999])
    def test_a_bare_invoke_is_a_call_on_both_paths(self, size):
        text = 'hi <invoke name="get_weather"> <city>Paris</city> bye'
        _, calls = parse_tool_calls(text, self.TOOLS, MiniMaxParser)
        _, streamed = self._stream(text, size)
        assert bool(calls) == bool(streamed) is True


class TestACallStillArrivingWhenTheStreamEnds:
    """A region whose last tag is half-written, and no more bytes are coming.

    Every format has to answer this the same way, and MiniMax was the one that
    did not: it required a *complete* tag, so a call cut off inside its first
    parameter name was delivered as text.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    NS = "]<]minimax[>["

    @pytest.mark.parametrize("tail", ["<ci", "<city>Par", ""])
    def test_a_half_written_declared_tag_is_a_call_at_end_of_region(self, tail):
        region = f'{self.NS}<invoke name="get_weather">\n{self.NS}{tail}'
        assert MiniMaxParser.parse_region(region, self.TOOLS, at_end=True).calls, (
            f"a call cut off at {tail!r} was read as prose; the client is told "
            "the model answered when it was calling a tool"
        )

    @pytest.mark.parametrize("tail", ["<c", "<city>Par", ""])
    def test_including_a_call_to_a_tool_that_declares_no_parameters(self, tail):
        """The zero-parameter tool is the one this used to drop.

        An empty schema falls back to "any tag" for a *complete* tag two lines
        up in the same function; the partial-tag branch gated on there being
        declared tags to compare against, so it had none and refused. The same
        tool declared with one property recovered the call, which is the
        asymmetry that gives it away.
        """
        zero = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        region = f'{self.NS}<invoke name="ping">\n{self.NS}{tail}'
        assert MiniMaxParser.parse_region(
            region, zero, at_end=True
        ).calls, "a truncated call to a zero-parameter tool was read as prose"

    def test_but_not_while_more_bytes_may_still_arrive(self):
        region = f'{self.NS}<invoke name="get_weather">\n{self.NS}<ci'
        assert not MiniMaxParser.parse_region(
            region, self.TOOLS, at_end=False
        ).calls, "announced a call from a prefix that has not finished arriving"


class TestTheAnswerAheadOfACallThatWasCutOff:
    """DSML anchored a truncated `<invoke>` at the region's start.

    Everything between the opening marker and the opener was therefore counted
    as this call's markup and deleted -- the one XML format of the four that
    did it, on both delivery paths.
    """

    TOOLS = TestACallStillArrivingWhenTheStreamEnds.TOOLS

    def test_prose_before_a_truncated_invoke_is_not_counted_as_markup(self):
        prose = "Let me check the weather for you right now. " * 12
        region = (
            "<｜DSML｜tool_calls>"
            + prose
            + '<｜DSML｜invoke name="get_weather">\n'
            + '<｜DSML｜parameter name="city">Paris'
        )
        parsed = DsmlParser.parse_region(region, self.TOOLS, at_end=True)
        assert parsed.calls, "the truncated call itself was lost"
        assert parsed.begins >= len(prose), (
            f"markup starts at {parsed.begins} but the answer runs to "
            f"{len(prose)}: {len(prose) - parsed.begins} characters of it are "
            "about to be deleted"
        )


class TestTheRequestsToolsAreResolvedOnce:
    """Built once per request rather than once per chunk (`_resolved_tools`).

    The substitution has to be invisible: the reader asks `not self.tools` in
    the announcement path, so a request whose tools yield nothing usable must
    still look the way the list looked, or a name is announced for a request
    that declared no names.
    """

    def test_a_real_catalogue_is_carried_in_its_built_form(self):
        resolved = _resolved_tools(TestACallStillArrivingWhenTheStreamEnds.TOOLS)
        assert isinstance(resolved, ParamTypes)
        assert set(resolved) == {"get_weather"}

    @pytest.mark.parametrize("tools", [None, [], [{"junk": 1}], ["not a dict"]])
    def test_and_anything_that_yields_no_name_keeps_its_own_truthiness(self, tools):
        assert bool(_resolved_tools(tools)) == bool(tools)

    def test_resolving_twice_is_the_same_answer(self):
        once = _resolved_tools(TestACallStillArrivingWhenTheStreamEnds.TOOLS)
        assert _resolved_tools(once) is once


class TestWhatSitsBetweenTwoCalls:
    """Prose survives; the template's own separator does not.

    Per-call markup spans made the gap between two calls answer, which is
    right for a sentence and wrong for the newline every one of these chat
    templates renders between consecutive calls. `end_of_markup` moves only on
    a closer and `begin_of_markup` only on an opener -- correct at the edge of
    a region, where the newline before the model resumes prose is the model's,
    and wrong between two calls, where nobody wrote it.
    """

    TOOLS = TestACallStillArrivingWhenTheStreamEnds.TOOLS
    CALLS = {
        "qwen": "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>Paris</parameter>\n</function>\n</tool_call>",
        "glm": "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>",
        "dsml": '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="city">Paris</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜tool_calls>",
        "minimax": ']<]minimax[>[<invoke name="get_weather">'
        "]<]minimax[>[<city>Paris</city>]<]minimax[>[</invoke>",
        "kimi_k3": '<|open|>call tool="get_weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>Paris<|close|>argument<|close|>call',
    }
    PARSERS = {
        "qwen": QwenXmlParser,
        "glm": GlmParser,
        "dsml": DsmlParser,
        "minimax": MiniMaxParser,
        "kimi_k3": KimiK3Parser,
    }

    @pytest.mark.parametrize("name", sorted(CALLS))
    @pytest.mark.parametrize("gap", ["\n", "  ", "\n\n  \t"])
    def test_whitespace_alone_between_them_is_markup(self, name, gap):
        call = self.CALLS[name]
        content, calls = parse_tool_calls(
            call + gap + call, self.TOOLS, self.PARSERS[name]
        )
        assert len(calls) == 2, "this shape proves nothing without two calls"
        assert (
            content == ""
        ), f"the template's separator reached the client as content: {content!r}"

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_but_a_sentence_between_them_is_not(self, name):
        call = self.CALLS[name]
        content, calls = parse_tool_calls(
            call + "\nNow Rome.\n" + call, self.TOOLS, self.PARSERS[name]
        )
        assert len(calls) == 2
        assert "Now Rome." in content, f"the answer was deleted: {content!r}"


class TestAZeroArgumentCallLooksTheSameInEveryFormat:
    """`arguments` is JSON on the wire, and `""` is not JSON.

    Kimi-K2 is the one format that passes the model's bytes through instead of
    building the object, so its no-argument call reached the client as an
    empty string where the other five sent `{}`. An OpenAI client calls
    `json.loads` on the accumulated arguments and raises; the Anthropic SDK
    accumulates an `input_json_delta` it cannot parse.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "now",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    ZERO_ARG = {
        "kimi": "<|tool_calls_section_begin|><|tool_call_begin|>functions.now:0"
        "<|tool_call_argument_begin|><|tool_call_end|><|tool_calls_section_end|>",
        "qwen": "<tool_call>\n<function=now>\n</function>\n</tool_call>",
        "glm": "<tool_call>now</tool_call>",
        "dsml": '<｜DSML｜tool_calls><｜DSML｜invoke name="now">'
        "</｜DSML｜invoke></｜DSML｜tool_calls>",
        "kimi_k3": '<|open|>call tool="now"<|sep|><|close|>call',
    }
    PARSERS = {
        "kimi": KimiParser,
        "qwen": QwenXmlParser,
        "glm": GlmParser,
        "dsml": DsmlParser,
        "kimi_k3": KimiK3Parser,
    }

    @pytest.mark.parametrize("name", sorted(ZERO_ARG))
    def test_the_arguments_are_parseable_json(self, name):
        _, calls = parse_tool_calls(self.ZERO_ARG[name], self.TOOLS, self.PARSERS[name])
        assert len(calls) == 1, f"{name} did not read its own zero-argument call"
        args = calls[0].function["arguments"]
        assert json.loads(args) == {}, f"{name} sent {args!r}"
