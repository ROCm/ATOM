# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for tool call parsing."""

import json
import tracemalloc
from typing import ClassVar

import pytest

from atom.entrypoints.openai.tool_parser import (
    ToolCall,
    ToolCallStreamParser,
    parse_tool_calls,
    read_whole_events,
)
from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
from atom.entrypoints.openai.tool_parser.glm_tool_parser import GlmParser
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.minimax_tool_parser import MiniMaxParser
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser
from atom.entrypoints.openai.tool_parser.registry import PARSERS_BY_NAME
from atom.entrypoints.openai.tool_parser.schema import ParamTypes
from atom.entrypoints.openai.tool_parser.stream import _resolved_tools
from entrypoints.wire_corpus import (
    PAYLOAD,
    REAL_CALLS,
    TYPED_TOOLS,
    check_corpus,
    complete,
    quoting_the_arguments,
    quoting_the_opener,
    truncated,
    truncated_after_complete,
)


def closer_reaching_the_tail_walk(parser) -> str | None:
    """The first closer `_region_tail` can actually see, or None.

    A closer the read-ahead strips as channel framing never gets there, and
    `opens_region` documents that framing is dropped -- so requiring such a
    literal to survive would be asserting against the format's own contract.
    Derived from the declarations, never a list of format names.
    """
    return next(
        (
            c
            for c in parser.CALL_CLOSERS
            if c not in parser.START_MARKERS or parser.opens_region(c)
        ),
        None,
    )


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
    """A call cut off by `max_tokens` reaches the client as a call, and what
    is not a call reaches it as text. Either way the bytes are not deleted.

    K3 used to cut the answer at the tools marker instead, on a second opener
    regex that accepted shapes the call regex rejects, and the two ways of
    getting that wrong were opposite: an answer *quoting* an opener lost 62
    characters, and a truncated call kept its half-written payload with the
    dangling `<|close|>argument` still in it. Both are now the same rule.

    This class used to assert the *other* outcome -- no call, region released
    verbatim -- and was right about the code at the time, because it passes
    `tools=None` and every format then refused every truncated call. That was
    the defect: with the same request declaring its tools, K3 salvaged this
    exact input. The salvage no longer turns on whether the client listed
    them, so what is pinned here is the salvage.
    """

    TRUNCATED = (
        "I will look it up."
        '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>Paris<|close|>argument'
    )

    def test_the_partial_payload_is_delivered_not_dropped(self):
        content, calls = parse_tool_calls(self.TRUNCATED, None, KimiK3Parser)
        assert [c.function["name"] for c in calls] == ["get_weather"]
        assert calls[0].function["arguments"] == '{"city": "Paris"}'
        assert content == "I will look it up.", "the answer around it was eaten"

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
        """The shape the quotation has to be told apart from, same request.

        `QUOTED` and this differ only in what follows the opener -- English in
        one, this format's own next token in the other -- and that is the only
        thing deciding between them here, because neither passes `tools`. It
        used to be decided by the empty tool list instead, which refused both
        and released `TRUNCATED`'s channel tokens as the answer.
        """
        content, calls = parse_tool_calls(self.TRUNCATED, None, KimiK3Parser)
        assert [c.function["name"] for c in calls] == ["get_weather"]
        assert calls[0].function["arguments"] == '{"city": "Par"}'
        assert content == "", f"markup left in the answer: {content!r}"

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

    @pytest.mark.parametrize("tool_name", ["get_weather", "get-weather", "server.tool"])
    def test_and_every_format_accepts_the_names_openai_allows(self, tool_name):
        """`^[a-zA-Z0-9_-]{1,64}$` is OpenAI's grammar, and MCP namespaces
        with a dot. Kimi-K2 matched `functions\\.(\\w+)`, so a hyphenated or
        namespaced name did not match at all -- the section was not a call and
        the whole thing, special tokens included, went out as `content` with
        `finish_reason: stop`."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        for name, template in self.ZERO_ARG.items():
            text = template.replace("now", tool_name)
            _, calls = parse_tool_calls(text, tools, self.PARSERS[name])
            assert [c.function["name"] for c in calls] == [
                tool_name
            ], f"{name} could not read a call to {tool_name!r}"

    @pytest.mark.parametrize("name", sorted(ZERO_ARG))
    def test_the_arguments_are_parseable_json(self, name):
        _, calls = parse_tool_calls(self.ZERO_ARG[name], self.TOOLS, self.PARSERS[name])
        assert len(calls) == 1, f"{name} did not read its own zero-argument call"
        args = calls[0].function["arguments"]
        assert json.loads(args) == {}, f"{name} sent {args!r}"


class TestTheWrapperClosingTagIsNeverTheAnswer:
    """Prose between the last call and the section's closing tag.

    `end_of_markup` walks forward over whitespace and closers and stops at the
    first other byte, so a model that writes a sentence before closing its
    section left the raw `</tool_call>` in `content`. The bytes are markup
    whatever sits in front of them -- but only while the region still owes a
    closing tag, which is what keeps an answer that *quotes* one after a real
    call from losing the quote.
    """

    TOOLS = TestACallStillArrivingWhenTheStreamEnds.TOOLS
    CALL = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter></function>"
    )

    def test_prose_before_the_closing_tag_keeps_the_prose(self):
        content, calls = parse_tool_calls(
            self.CALL + "\nOK, checking.\n</tool_call>", self.TOOLS, QwenXmlParser
        )
        assert len(calls) == 1
        assert "OK, checking." in content
        assert "</tool_call>" not in content, f"raw markup delivered: {content!r}"

    def test_but_a_quoted_closing_tag_after_a_closed_call_survives(self):
        """Only the region's own trailing edge is consumed."""
        text = self.CALL + "</tool_call> you end it with </tool_call>."
        content, calls = parse_tool_calls(text, self.TOOLS, QwenXmlParser)
        assert len(calls) == 1
        assert (
            content.count("</tool_call>") == 1
        ), f"the quoted tag was eaten: {content!r}"

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    def test_and_an_answer_that_ends_by_naming_the_closing_tag_keeps_it(self, name):
        """The same direction, with nothing after the tag to stop the walk.

        The two tests around this one pin it as well, but both end in a `.`
        after the quoted tag -- and a `.` is neither whitespace, a filler nor
        a closer, so the walk stops on its first step whatever rule sits in
        front of it. They are green by punctuation. An answer that ends
        `... you write </tool_call>` has no period, and every format with an
        outer wrapper deleted the literal.
        """
        parser = PARSERS_BY_NAME[name]
        closer = closer_reaching_the_tail_walk(parser)
        if closer is None:
            pytest.skip(f"{name}: no closer of its own reaches the tail walk")
        tail = f"\nTo close a call you write {closer}"
        content, calls = parse_tool_calls(complete(name) + tail, TYPED_TOOLS, parser)
        assert calls, f"{name} lost the call itself"
        assert content == tail, (
            f"{name} deleted its own closing literal out of the answer: "
            f"{content!r} != {tail!r}"
        )

    WRITE_TOOLS = [
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

    def test_and_a_markup_literal_inside_an_argument_value_changes_nothing(self):
        """The first version counted openers minus closers inside the spans
        and consumed that many closers from the tail. The count saw literals
        in argument *values*, so a call whose payload mentioned `<tool_call>`
        invented a debt and the mechanism deleted a `</tool_call>` out of the
        answer -- the very thing it was written to protect. An edge needs no
        count."""
        text = (
            "<tool_call><function=write_file><parameter=path>d.md</parameter>"
            "<parameter=content>open with <tool_call> and close.</parameter>"
            "</function></tool_call>\nDone. Pair is <tool_call> ... </tool_call>."
        )
        content, calls = parse_tool_calls(text, self.WRITE_TOOLS, QwenXmlParser)
        assert len(calls) == 1
        assert content == "\nDone. Pair is <tool_call> ... </tool_call>.", content

    def test_and_a_quoted_closer_in_the_middle_is_not_taken_for_the_real_one(self):
        """The count also discharged its debt against the earliest closer in
        document order, so DSML deleted twelve bytes of a sentence *and* left
        the real closer in."""
        d = "\uff5cDSML\uff5c"
        text = (
            f'<{d}tool_calls><{d}invoke name="get_weather">'
            f'<{d}parameter name="city">Paris</{d}parameter></{d}invoke>'
            f"\nClose with </tool_call> normally.\n</{d}tool_calls>"
        )
        content, calls = parse_tool_calls(text, self.TOOLS, DsmlParser)
        assert len(calls) == 1
        assert content == "\nClose with </tool_call> normally.\n", content

    def test_and_a_formats_filler_before_the_closer_goes_with_it(self):
        """MiniMax repeats its namespace token before every tag, so the walk
        has to step over one to reach the closer."""
        ns = "]<]minimax[>["
        text = (
            f'{ns}<tool_call>{ns}<invoke name="get_weather">{ns}<city>Paris</city>'
            f"{ns}</invoke>\nDone.\n{ns}</tool_call>"
        )
        content, calls = parse_tool_calls(text, self.TOOLS, MiniMaxParser)
        assert len(calls) == 1
        assert content == "\nDone.\n", content

    def test_two_calls_then_prose_keeps_the_prose_after_both(self):
        """Order must not depend on whether the closing tag arrived.

        The wrapper was spliced in as a span, and two adjacent calls merge
        into one -- so the wrapper span became the last, took the leftover
        call, and emitted the prose before it. Same output, two orders,
        depending on the presence of `</tool_call>`. On `/v1/messages` that
        put a text block between the two `tool_use` blocks.
        """
        a = self.CALL
        b = self.CALL.replace("get_weather", "get_time").replace("Paris", "Rome")
        tools = self.TOOLS + [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        orders = []
        for tail in ("\nBoth running.\n</tool_call>", "\nBoth running."):
            events = read_whole_events(QwenXmlParser, a + b + tail, tools)
            orders.append([k for k, _ in events if k in ("content", "tool_call_start")])
        assert (
            orders[0] == orders[1]
        ), f"the closing tag changed the order: {orders[0]} vs {orders[1]}"
        assert orders[0] == ["tool_call_start", "tool_call_start", "content"], orders[0]

    def test_and_a_quoted_opener_before_a_real_call_still_survives(self):
        """The mirror edge is deliberately not swept. A wrapper opener
        followed by prose looks exactly like an answer quoting the opener
        before calling for real, and `RegionParse.begins` already decided that
        one in favour of keeping the text. A leading scan was written here and
        deleted the quotation; the opener leaks instead, the smaller harm."""
        text = "Explaining: <tool_call> is how. Now: " + self.CALL + "</tool_call> done"
        content, calls = parse_tool_calls(text, self.TOOLS, QwenXmlParser)
        assert len(calls) == 1
        assert "Explaining: <tool_call> is how." in content, content


class TestWhatEveryFormatDoesWithACallCutOffMidWay:
    """The axis five separate review findings have each been one cell of.

    A response stopped at `max_tokens` mid-call is the single most common
    malform there is, and the six formats answer it three different ways.
    Every previous finding here was reported against one format, fixed on that
    format, and left the others unstated -- so the next cell stayed invisible
    until someone reported it too:

      MiniMax  a zero-parameter tool's truncated call was unrecoverable
      DSML     truncation was unreachable whenever a complete call existed
      Kimi-K2  a truncated entry beside a complete one is dropped
      Kimi-K3  a truncated call is never recovered at all
      DSML/MiniMax  a mid-value cut yields `{}` where Qwen/GLM keep the partial

    So the table is declared here instead of discovered. It is not an
    endorsement -- three of these cells are worse than the other three -- it
    is a statement of what is true today, in one place, so that changing a
    format shows up as a diff and adding one forces an answer.
    """

    TOOLS = TYPED_TOOLS

    def _row(self, name):
        parser = PARSERS_BY_NAME[name]
        _, cut_calls = parse_tool_calls(truncated(name), self.TOOLS, parser)
        recovers = bool(cut_calls)
        # `"Par" in ...` rather than a JSON compare: Kimi-K2 passes the
        # model's own bytes through as `arguments`, so a value cut off mid-way
        # is *invalid JSON there by construction* and any format-independent
        # assertion has to be about the bytes, not the decode.
        keeps = recovers and PAYLOAD[:-2] in cut_calls[0].function["arguments"]
        _, both = parse_tool_calls(truncated_after_complete(name), self.TOOLS, parser)
        return recovers, keeps, len(both) == 2

    def test_every_format_can_write_its_own_syntax(self):
        """The one thing a new format's author has to provide.

        `render_call` is the inverse of `parse_region`; the whole corpus is
        generated from it, so a format that has one is covered by every
        property here without a line being added to the tests.
        """
        cannot = sorted(
            name for name, cls in PARSERS_BY_NAME.items() if name not in REAL_CALLS
        )
        assert not cannot, (
            f"{cannot} cannot render their own syntax, so nothing here covers "
            "them. Implement `render_call(name, args)` on the parser."
        )

    def test_the_corpus_is_sound_and_covers_every_registered_format(self):
        """The fixture has to be real, and there has to be one per format.

        Both halves have failed here before. A format with no entry goes
        unasserted -- which is how five separate findings each came back as
        one more cell. And an entry written in another format's spelling
        parses to a call with no arguments, passing everything while touching
        none of the parameter path; MiniMax's did exactly that.
        """
        problems = check_corpus(PARSERS_BY_NAME, parse_tool_calls)
        assert not problems, "the wire corpus is unsound:\n  " + "\n  ".join(problems)

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    def test_a_truncated_call_is_recovered(self, name):
        recovers, _, _ = self._row(name)
        assert recovers, (
            f"{name} delivers a call cut off by max_tokens as raw markup "
            "instead of recovering it"
        )

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    def test_and_the_part_of_the_value_that_arrived_is_kept(self, name):
        _, keeps, _ = self._row(name)
        assert keeps, (
            f"{name} recovers the call but drops the argument value the model "
            "had already produced"
        )

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    def test_and_it_is_recovered_beside_a_complete_call(self, name):
        _, _, beside = self._row(name)
        assert beside, (
            f"{name} drops a truncated call whenever a complete one exists in "
            "the same region"
        )

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    def test_and_the_other_order_does_not_bury_the_second_call_in_the_first(self, name):
        """Cut off first, complete second -- and where the markup ended up.

        Three suites have built `truncated + complete` since the corpus
        landed, and all three stayed green while four formats mishandled it,
        because each asked something else of it: chunk invariance (which holds
        while every chunk size drops the same call), that `get_weather` is
        among the names (both calls carry it here), and announced-matches-
        parsed (both channels drop it, so they agree).

        Not `len(calls) == 2`: that is not format-independent, and a format
        whose second call lands inside the first's unterminated string literal
        would be permanently red for being honest. What every format does owe
        is that its markup reaches neither place it can leak to -- asserting
        only one lets a fix move it to the other.
        """
        parser = PARSERS_BY_NAME[name]
        content, calls = parse_tool_calls(
            truncated(name) + complete(name), self.TOOLS, parser
        )
        assert calls, f"{name} drops both calls when the cut-off one comes first"
        leaked = sorted(m for m in parser.START_MARKERS if m in content)
        assert (
            not leaked
        ), f"{name} delivers its own markup to the user as the answer: {leaked}"
        buried = sorted(
            {
                m
                for m in parser.START_MARKERS
                for c in calls
                if m in c.function["arguments"]
            }
        )
        assert not buried, (
            f"{name} runs the first call's argument value past the second "
            f"call's opener, so the tool is invoked with markup as data: "
            f"{buried} in {[c.function['arguments'] for c in calls]}"
        )

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    @pytest.mark.parametrize("order", ["complete-first", "truncated-first"])
    def test_and_the_chunk_size_does_not_change_the_answer(self, name, order):
        """The invariant the whole reader is built on, on the truncated shape.

        DSML broke exactly this: over a prefix that stopped before the
        complete call, its `else:` branch was live and named the truncated
        one; over the whole region the parse returned the other. The client
        got two `tool_call_start` deltas at small chunk sizes and one for the
        same generation in a single chunk.
        """
        whole, cut = complete(name), truncated(name)
        text = whole + cut if order == "complete-first" else cut + whole
        seen = set()
        for size in (1, 4, 13, 10**6):
            parser = ToolCallStreamParser(
                tools=self.TOOLS, parser_cls=PARSERS_BY_NAME[name]
            )
            events = []
            for i in range(0, len(text), size):
                events += parser.process(text[i : i + size])
            events += parser.flush()
            seen.add(
                tuple(
                    d["function"]["name"] for k, d in events if k == "tool_call_start"
                )
            )
        assert len(seen) == 1, f"{name} {order}: chunking changed the calls: {seen}"

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    @pytest.mark.parametrize("declares_tools", [True, False])
    def test_but_prose_that_quotes_the_opener_is_still_not_a_call(
        self, name, declares_tools
    ):
        """The other half of the pair.

        An unclosed alternative without a `_is_truncated_call` gate turns
        every sentence *about* the wire format into a tool call. Kimi-K3 was
        given the alternation alone and did exactly that.

        Both arms, because the gate has two halves and only one comes from the
        request. `declared_tools_allow` stops refusing when the request lists
        no tools, so with `tools=None` the follower test is the whole of what
        separates a cut-off call from a sentence about one -- if that ever
        stops carrying it, this is where it shows.
        """
        _, calls = parse_tool_calls(
            quoting_the_opener(name),
            self.TOOLS if declares_tools else None,
            PARSERS_BY_NAME[name],
        )
        assert calls == [], f"{name} read a sentence about calls as a call"

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    @pytest.mark.parametrize("declares_tools", [True, False])
    def test_and_neither_is_prose_that_quotes_the_arguments(self, name, declares_tools):
        """The half no shape built from the front of a call can reach.

        `quoting_the_opener` keeps everything *before* the tool name, so it
        always carries the call opener and always lands in the branch that
        reads one. A format that infers the name from the parameters instead
        has a second branch, and DSML's had no gate at all: 152 characters of
        an answer explaining the syntax collapse to 18 and a call is
        dispatched. Undeclared tools make it worse rather than safer -- the
        name cannot be inferred, so it is sent as `"unknown"`.
        """
        text = quoting_the_arguments(name)
        if text is None:
            pytest.skip(f"{name}: no self-contained region marker to quote")
        tools = TYPED_TOOLS if declares_tools else None
        content, calls = parse_tool_calls(text, tools, PARSERS_BY_NAME[name])
        assert calls == [], (
            f"{name} read a sentence about argument syntax as a call: "
            f"{[(c.function['name'], c.function['arguments']) for c in calls]}"
        )
        assert content == text, f"{name} deleted the answer: {content!r}"

    @pytest.mark.parametrize("name", sorted(REAL_CALLS))
    def test_and_the_declared_tools_are_not_the_only_gate(self, name):
        """A request that lists no tools still gets its truncated call.

        `tools` sharpens the answer -- it is how a format tells a call cut off
        by `max_tokens` from a sentence quoting an opener -- but it cannot be
        the whole of it, because a *complete* call is read without it on all
        six. Cut the same generation one byte earlier and it became raw
        special tokens in `content`, decided by whether the client had
        listed its tools rather than by anything the model wrote.
        """
        parser = PARSERS_BY_NAME[name]
        _, whole = parse_tool_calls(complete(name), None, parser)
        if not whole:
            pytest.skip(f"{name} does not read a complete call without tools either")
        _, cut = parse_tool_calls(truncated(name), None, parser)
        assert cut, (
            f"{name} reads a complete call with no tools declared and drops "
            "the truncated one, so the same generation cut a byte earlier "
            "ships raw markup as the answer"
        )


class TestTheRegionIsNotCopiedToReadACallsLeftEdge:
    """`begin_of_markup` walks back over a call's wrapper a few bytes at a time.

    It spelled each step `region[:j].rstrip()`, which copies everything to the
    left of the scan -- so a region holding several calls after a large
    argument paid O(calls x region). Measured 42.8 us against 5.8 for eight
    calls behind a 128 KB payload, and growing with the payload where the
    offset form is flat.

    Its mirror `end_of_markup` walks the other way with `startswith(s, j)` and
    never copied, and `_region_owes_a_closer` one layer up already carries
    this fix in its docstring -- it was not carried into the function that
    shape was taken from. `prompt_starts_in_reasoning` carries the twin of
    this test.
    """

    CALL: ClassVar[str] = (
        "<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "{payload}\n</parameter>\n</function>\n</tool_call>\n"
    )

    def _region(self, payload_kb: int, calls: int = 4) -> tuple[str, list[int]]:
        """`calls` calls, the first carrying a `payload_kb` argument, and every
        `<function=` offset -- which is what the formats hand `markup_begin`."""
        region = "".join(
            self.CALL.format(payload="x" * (payload_kb * 1024 if i == 0 else 8))
            for i in range(calls)
        )
        ats, pos = [], 0
        while (k := region.find("<function=", pos)) != -1:
            ats.append(k)
            pos = k + 1
        return region, ats

    @staticmethod
    def _peak_bytes(region: str, ats: list[int]) -> int:
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            for at in ats:
                QwenXmlParser.markup_begin(region, at)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_the_left_edge_is_unchanged(self):
        """The walk still steps over the wrapper and the newline before it."""
        region, ats = self._region(1)
        assert ats, "nothing to walk; this asserts nothing"
        # The first call's wrapper opens the region, so its markup begins at 0.
        assert QwenXmlParser.markup_begin(region, ats[0]) == 0
        # A later call's begins at its own `<tool_call>`, not at the prose or
        # the payload before it.
        for at in ats[1:]:
            begin = QwenXmlParser.markup_begin(region, at)
            assert region.startswith("<tool_call>", begin), (
                f"walked back to {region[begin : begin + 20]!r}, which is not "
                "this call's wrapper"
            )

    def test_prose_before_a_quoted_marker_stays_prose(self):
        region = "I would call <tool_call>\n<function=f>\n</function>\n</tool_call>\n"
        at = region.find("<function=")
        assert QwenXmlParser.markup_begin(region, at) == region.find("<tool_call>")

    def test_it_allocates_no_more_for_a_payload_128x_larger(self):
        small, small_ats = self._region(1)
        large, large_ats = self._region(128)
        assert len(small_ats) == len(large_ats), "the two shapes must be comparable"
        a = self._peak_bytes(small, small_ats)
        b = self._peak_bytes(large, large_ats)
        assert b < a + 4096, (
            f"128 KB allocated {b} bytes against {a} for 1 KB; the region is "
            "being copied on every step of the walk"
        )
