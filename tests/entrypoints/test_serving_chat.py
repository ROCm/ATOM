# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for chat completion serving logic (chunk creation, response building)."""

import ast
import asyncio
import inspect
import json
import pathlib

import pytest

from atom.entrypoints.atomesh import atom_standalone_service
from atom.entrypoints.openai import api_server, serving_chat
from atom.entrypoints.openai.protocol import (
    openai_stop_reason,
    openai_stop_reason_with_calls,
)
from atom.entrypoints.openai.reasoning import NO_REASONING, ReasoningChannel
from atom.entrypoints.openai.serving_anthropic import (
    anthropic_to_openai_tools,
    completes_a_tool_call,
)
from atom.entrypoints.openai.serving_chat import (
    _build_chat_choice,
    build_chat_response,
    build_chat_response_multi,
    create_chat_chunk,
    normalize_chat_tools,
    stream_chat_response,
    stream_chat_response_fanout,
    validate_tool_list,
)
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser, parse_tool_calls
from atom.entrypoints.openai.tool_parser.glm_tool_parser import GlmParser
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser
from atom.entrypoints.openai.tool_parser.registry import forbids_tool_calls
from atom.entrypoints.openai.tool_parser.tool_parser import usable_tool_name

# ============================================================================
# normalize_chat_tools Tests
# ============================================================================


class TestNormalizeChatTools:
    def test_converts_anthropic_tool_schema(self):
        tools = [
            {
                "name": "search",
                "description": "Search documents",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

        assert normalize_chat_tools(tools) == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search documents",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]

    def test_preserves_openai_tool_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object"},
                },
            }
        ]

        assert normalize_chat_tools(tools) == tools

    def test_leaves_malformed_tool_for_validator(self):
        tools = [{"name": "search", "input_schema": "not-an-object"}]

        assert normalize_chat_tools(tools) == tools


# ============================================================================
# create_chat_chunk Tests
# ============================================================================


class TestCreateChatChunk:
    """Tests for SSE chunk creation."""

    def test_content_chunk(self):
        chunk_str = create_chat_chunk("req-1", "test-model", delta={"content": "Hello"})
        assert chunk_str.startswith("data: ")
        assert chunk_str.endswith("\n\n")
        data = json.loads(chunk_str[6:])
        assert data["id"] == "req-1"
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_reasoning_content_chunk(self):
        chunk_str = create_chat_chunk(
            "req-1", "model", delta={"reasoning_content": "thinking..."}
        )
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["delta"]["reasoning_content"] == "thinking..."

    def test_role_chunk(self):
        chunk_str = create_chat_chunk("req-1", "model", delta={"role": "assistant"})
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_empty_delta(self):
        chunk_str = create_chat_chunk("req-1", "model")
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["delta"] == {}

    def test_role_chunk_includes_empty_content(self):
        chunk_str = create_chat_chunk(
            "req-1", "model", delta={"role": "assistant", "content": ""}
        )
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["delta"]["role"] == "assistant"
        assert data["choices"][0]["delta"]["content"] == ""

    def test_finish_reason(self):
        chunk_str = create_chat_chunk("req-1", "model", finish_reason="stop")
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_usage_chunk(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        chunk_str = create_chat_chunk("req-1", "model", usage=usage)
        data = json.loads(chunk_str[6:])
        assert data["usage"]["total_tokens"] == 15


# ============================================================================
# build_chat_response Tests
# ============================================================================


class TestBuildChatResponse:
    """Tests for non-streaming chat response building."""

    def _make_output(self, **overrides):
        defaults = {
            "text": "Hello!",
            "finish_reason": "stop",
            "num_tokens_input": 10,
            "num_tokens_output": 5,
            "ttft": 0.1,
            "tpot": 0.02,
            "latency": 0.5,
        }
        defaults.update(overrides)
        return defaults

    def test_basic_response(self):
        output = self._make_output(text="Hello!")
        resp = build_chat_response("req-1", "model", "Hello!", output)
        assert resp.id == "req-1"
        assert resp.model == "model"
        assert resp.choices[0]["message"]["content"] == "Hello!"
        assert resp.choices[0]["message"]["role"] == "assistant"
        assert resp.usage["total_tokens"] == 15

    def test_reasoning_separation(self):
        raw_text = "<think>I should say hello</think>Hello!"
        output = self._make_output(text=raw_text)
        resp = build_chat_response("req-1", "model", raw_text, output)
        assert resp.choices[0]["message"]["content"] == "Hello!"
        assert resp.choices[0]["message"]["reasoning_content"] == "I should say hello"

    def test_no_reasoning(self):
        output = self._make_output(text="No thinking here")
        resp = build_chat_response("req-1", "model", "No thinking here", output)
        assert resp.choices[0]["message"]["content"] == "No thinking here"
        assert "reasoning_content" not in resp.choices[0]["message"]

    def test_tool_call_parsed(self):
        raw = (
            "Hi"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.exec:0"
            '<|tool_call_argument_begin|>{"cmd": "ls"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        output = self._make_output(text=raw)
        # The format is the one resolved at startup, so it has to be passed.
        # Leaving it off used to reach a cascade over the *output*, which is
        # how an answer quoting these tokens got its text deleted.
        resp = build_chat_response(
            "req-1",
            "model",
            raw,
            output,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "exec", "parameters": {}},
                }
            ],
            tool_parser_cls=KimiParser,
        )
        assert resp.choices[0]["message"]["content"] == "Hi"
        assert "tool_calls" in resp.choices[0]["message"]
        tc = resp.choices[0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "exec"
        assert '"cmd"' in tc["function"]["arguments"]
        assert resp.choices[0]["finish_reason"] == "tool_calls"

    def test_timing_in_usage(self):
        output = self._make_output(ttft=0.15, tpot=0.03, latency=0.8)
        resp = build_chat_response("req-1", "model", "text", output)
        assert resp.usage["ttft_s"] == 0.15
        assert resp.usage["tpot_s"] == 0.03
        assert resp.usage["latency_s"] == 0.8


# ============================================================================
# build_chat_response_multi Tests (SamplingParams.n > 1 fan-out)
# ============================================================================


class TestBuildChatResponseMulti:
    """Tests for multi-choice (n>1) non-streaming chat response."""

    def _make_output(self, **overrides):
        defaults = {
            "text": "Hello!",
            "finish_reason": "stop",
            "num_tokens_input": 10,
            "num_tokens_output": 5,
            "ttft": 0.1,
            "tpot": 0.02,
            "latency": 0.5,
        }
        defaults.update(overrides)
        return defaults

    def test_choice_count_matches_fanout(self):
        outputs = [self._make_output(text=f"answer-{i}") for i in range(4)]
        resp = build_chat_response_multi("req-2", "model", outputs)
        assert len(resp.choices) == 4

    def test_choice_indices_are_zero_to_n_minus_one(self):
        outputs = [self._make_output(text=f"answer-{i}") for i in range(3)]
        resp = build_chat_response_multi("req-2", "model", outputs)
        assert [c["index"] for c in resp.choices] == [0, 1, 2]

    def test_per_choice_content_preserved(self):
        outputs = [
            self._make_output(text="first answer"),
            self._make_output(text="second answer"),
        ]
        resp = build_chat_response_multi("req-2", "model", outputs)
        assert resp.choices[0]["message"]["content"] == "first answer"
        assert resp.choices[1]["message"]["content"] == "second answer"

    def test_completion_tokens_summed_across_siblings(self):
        outputs = [
            self._make_output(num_tokens_output=5),
            self._make_output(num_tokens_output=7),
            self._make_output(num_tokens_output=3),
        ]
        resp = build_chat_response_multi("req-2", "model", outputs)
        assert resp.usage["completion_tokens"] == 15
        # prompt tokens come from the shared prompt and should not be multiplied
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["total_tokens"] == 25
        assert resp.usage["num_choices"] == 3

    def test_latency_is_max_across_siblings(self):
        outputs = [
            self._make_output(latency=0.3),
            self._make_output(latency=0.9),
            self._make_output(latency=0.5),
        ]
        resp = build_chat_response_multi("req-2", "model", outputs)
        assert resp.usage["latency_s"] == 0.9

    def test_reasoning_separated_per_choice(self):
        outputs = [
            self._make_output(text="<think>reasoning A</think>answer A"),
            self._make_output(text="plain answer B"),
        ]
        resp = build_chat_response_multi("req-2", "model", outputs)
        assert resp.choices[0]["message"]["content"] == "answer A"
        assert resp.choices[0]["message"]["reasoning_content"] == "reasoning A"
        assert resp.choices[1]["message"]["content"] == "plain answer B"
        assert "reasoning_content" not in resp.choices[1]["message"]


class TestCreateChatChunkWithIndex:
    """Tests for the ``index`` parameter added for fan-out streaming."""

    def test_default_index_is_zero(self):
        chunk_str = create_chat_chunk("req", "model", delta={"content": "hi"})
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["index"] == 0

    def test_explicit_index_propagated(self):
        chunk_str = create_chat_chunk("req", "model", delta={"content": "hi"}, index=3)
        data = json.loads(chunk_str[6:])
        assert data["choices"][0]["index"] == 3


# ============================================================================
# Streaming Role Chunk Content Regression Tests
# ============================================================================


class TestStreamingRoleChunkContent:
    """End-to-end regression test for the streamed role-announcement chunk.

    The unit test above (test_role_chunk_includes_empty_content) only checks
    that create_chat_chunk() can serialize a delta it's handed directly. It
    does not exercise stream_chat_response / stream_chat_response_fanout, so
    a regression that drops content="" inside those generators would not be
    caught. This drives both generators directly with a minimal queue
    payload and asserts the first emitted SSE chunk includes content="".
    """

    def test_single_stream_role_chunk_has_empty_content(self):
        async def run():
            collector = StreamOutputCollector("req-1")
            collector.put_nowait({"text": "Hi", "token_ids": [1], "finished": True})
            gen = stream_chat_response(
                request_id="req-1",
                model="model",
                stream_collector=collector,
                seq_id=0,
                num_prompt_tokens=1,
                cleanup_stream=lambda *a, **k: None,
                cleanup_request=lambda *a, **k: None,
            )
            first_chunk = await gen.__anext__()
            await gen.aclose()
            return first_chunk

        first_chunk = asyncio.run(run())
        assert first_chunk.startswith("data: ")
        data = json.loads(first_chunk[6:])
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["content"] == ""

    def test_fanout_stream_role_chunks_have_empty_content(self):
        async def run():
            collector = StreamOutputCollector("req-2")
            # Empty text + finished=False means each pending chunk triggers
            # *only* the role-announcement yield (no content/finish chunks
            # in between), so the first two yields are guaranteed to be
            # sibling 0's and sibling 1's role chunks respectively.
            collector.put_nowait((0, {"text": "", "token_ids": [], "finished": False}))
            collector.put_nowait((1, {"text": "", "token_ids": [], "finished": False}))
            gen = stream_chat_response_fanout(
                request_id="req-2",
                model="model",
                shared_collector=collector,
                seq_ids=[0, 1],
                num_prompt_tokens=1,
                cleanup_stream=lambda *a, **k: None,
                cleanup_request=lambda *a, **k: None,
            )
            chunk_0 = await gen.__anext__()
            chunk_1 = await gen.__anext__()
            await gen.aclose()
            return chunk_0, chunk_1

        chunk_0, chunk_1 = asyncio.run(run())
        for raw_chunk, expected_index in ((chunk_0, 0), (chunk_1, 1)):
            assert raw_chunk.startswith("data: ")
            data = json.loads(raw_chunk[6:])
            choice = data["choices"][0]
            assert choice["index"] == expected_index
            delta = choice["delta"]
            assert delta["role"] == "assistant"
            assert delta["content"] == ""


class TestFanoutCleanupSplit:
    """A fan-out has n streams but one request, and teardown reflects that.

    Per-sequence work (dropping the detokenizer state, aborting a seq that is
    still running) has to happen once per sibling; the per-request bookkeeping
    only once. Folding both into a single callback ran the request half n
    times, n-1 of them no-ops, and forced every caller to pass a seq id and a
    request id together when each half needs only one of them.
    """

    def _cleanup_calls(self, seq_ids):
        stream_calls, request_calls = [], []

        async def run():
            collector = StreamOutputCollector("req-3")
            for index in range(len(seq_ids)):
                collector.put_nowait(
                    (index, {"text": "", "token_ids": [], "finished": True})
                )
            gen = stream_chat_response_fanout(
                request_id="req-3",
                model="model",
                shared_collector=collector,
                seq_ids=seq_ids,
                num_prompt_tokens=1,
                cleanup_stream=lambda seq_id, **kwargs: stream_calls.append(seq_id),
                cleanup_request=request_calls.append,
            )
            async for _ in gen:
                pass

        asyncio.run(run())
        return stream_calls, request_calls

    def test_every_sibling_seq_is_torn_down(self):
        seq_ids = [70, 71, 72, 73]

        stream_calls, _ = self._cleanup_calls(seq_ids)

        assert stream_calls == seq_ids

    def test_the_request_is_torn_down_exactly_once(self):
        _, request_calls = self._cleanup_calls([70, 71, 72, 73])

        assert request_calls == ["req-3"]


class TestNoToolsMeansNoToolReadAhead:
    """Match SGLang: a request with no tools does not enter a tool region."""

    def test_an_unclosed_literal_does_not_stall_the_stream(self):
        async def run():
            collector = StreamOutputCollector("req-no-tools")
            collector.put_nowait(
                {
                    "text": "<tool_call> is syntax being discussed",
                    "token_ids": [1],
                    "finished": False,
                }
            )
            gen = stream_chat_response(
                request_id="req-no-tools",
                model="model",
                stream_collector=collector,
                seq_id=1,
                num_prompt_tokens=1,
                cleanup_stream=lambda *args, **kwargs: None,
                cleanup_request=lambda *args, **kwargs: None,
                tools=None,
                tool_parser_cls=GlmParser,
            )
            await gen.__anext__()  # role
            content = await asyncio.wait_for(gen.__anext__(), timeout=0.1)
            await gen.aclose()
            return json.loads(content[6:])["choices"][0]["delta"]["content"]

        assert asyncio.run(run()) == "<tool_call> is syntax being discussed"

    def test_non_streaming_path_matches(self):
        text = "<tool_call> is syntax being discussed"
        choice = _build_chat_choice(
            text,
            "eos",
            tools=None,
            tool_parser_cls=GlmParser,
        )

        assert choice["message"]["content"] == text
        assert "tool_calls" not in choice["message"]

    def test_channel_framing_is_still_consumed_without_tools(self):
        framed = (
            "<|open|>response<|sep|>Hello there."
            "<|close|>response<|sep|><|end_of_msg|>"
        )
        choice = _build_chat_choice(
            framed,
            "eos",
            tools=None,
            tool_parser_cls=KimiK3Parser,
        )

        assert choice["message"]["content"] == "Hello there."


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
A_CALL = (
    "Sure. <tool_call><function=get_weather><parameter=city>Paris</parameter>"
    "</function></tool_call>"
)


class TestForbiddingToolCallsDoesNotDeleteTheAnswer:
    """`tool_choice="none"` suppresses the *call*, not the model's words.

    It used to be enforced at the twelve places an event is *sent*, across two
    entrypoints, while the parser went on consuming the region -- so the text
    was eaten and nothing took its place. Measured on the answer below: 89 of
    95 characters gone, no event, `finish_reason: stop`.

    The rule lives at the one place the parser is *asked*, as a
    `suppress_calls` flag. What that flag suppresses is *dispatch*, not
    reading. It used to also skip opening the region, on the reading that
    "the request said this cannot be a call, so it is prose" -- and that
    reading turned out to be false in its own terms: the bytes released were
    the model's wire markup, `<invoke name="get_weather">` and payload, on
    every format. The region is now buffered and parsed exactly as it is for
    a permitted call, and only the calls are dropped.

    Not by dropping the parser, which is where the first fix went. Dropping
    it drops everything else a parser does -- and a format whose framing
    wraps *every* answer then leaks that framing the moment a request says
    `none`: Kimi-K3's `Hello there.` arrived as
    `<|open|>response<|sep|>Hello there.<|close|>response<|sep|><|end_of_msg|>`.
    `test_a_format_that_normalises_its_framing_still_does` is that.

    Stated behaviourally and not as a scan of the source -- the scan this
    replaces was mutation-checked and accepted `or tool_choice != "none"`,
    and it hardcoded one module while the same gap sat in three others.
    """

    @staticmethod
    def _stream(tool_choice, chunk=7, text=A_CALL, parser_cls=QwenXmlParser):
        parser = ToolCallStreamParser(
            tools=TOOLS,
            parser_cls=parser_cls,
            suppress_calls=forbids_tool_calls(tool_choice),
        )
        events = []
        for i in range(0, len(text), chunk):
            events += parser.process(text[i : i + chunk])
        events += parser.flush()
        return events

    @pytest.mark.parametrize(
        "tool_choice",
        ["none", {"type": "none"}],
        ids=["openai-string", "anthropic-object"],
    )
    def test_the_answer_is_delivered_and_the_markup_is_not(self, tool_choice):
        """Both protocols' spellings, because both endpoints ask.

        The answer around the call survives. What does not survive is the
        call's own bytes: a client that asked for no tool calls is not asking
        to be shown the wire format.
        """
        events = self._stream(tool_choice)
        content = "".join(d for k, d in events if k == "content")
        assert "Sure." in content, "the answer was eaten again"
        for token in ("<tool_call>", "<function=", "<parameter=", "</function>"):
            assert token not in content, f"raw markup shown to the user: {token}"

    @pytest.mark.parametrize(
        "tool_choice",
        ["none", {"type": "none"}],
        ids=["openai-string", "anthropic-object"],
    )
    def test_and_no_call_is_reported(self, tool_choice):
        events = self._stream(tool_choice)
        assert [k for k, _ in events if k.startswith("tool_call")] == []
        assert not completes_a_tool_call(events)

    @pytest.mark.parametrize(
        "tool_choice", [None, "auto", "required", {"type": "auto"}]
    )
    def test_anything_else_still_calls_the_tool(self, tool_choice):
        """The prohibition is `none` and nothing else -- reading `required`
        or a named tool as one would silently stop every such request from
        ever producing a call."""
        events = self._stream(tool_choice)
        assert completes_a_tool_call(events)

    def test_the_two_paths_agree(self):
        streamed = "".join(d for k, d in self._stream("none") if k == "content")
        non_streaming, calls = parse_tool_calls(
            A_CALL, TOOLS, parser_cls=QwenXmlParser, suppress_calls=True
        )
        assert calls == [] and non_streaming == streamed

    def test_forbidding_calls_withholds_no_more_than_allowing_them(self):
        """Parity, not zero.

        Reading the format costs a region's worth of buffering, and that cost
        is the same whatever the request said about dispatch -- it is the
        latency `_PEEK_WINDOW` and `REGION_END_MARKERS` exist to bound, not
        something `tool_choice` should change. Zero was the old promise, and
        it was bought by not reading the format at all, which is what put the
        raw markup in the answer.
        """
        forbidden = sum(len(d) for k, d in self._held_back("none") if k == "content")
        allowed = sum(len(d) for k, d in self._held_back("auto") if k == "content")
        assert forbidden <= allowed, (
            f"forbidding calls held {forbidden} characters to EOS where "
            f"allowing them held {allowed}"
        )

    @staticmethod
    def _held_back(tool_choice, chunk=7):
        """Only what `flush` produced -- what streaming failed to deliver."""
        parser = ToolCallStreamParser(
            tools=TOOLS,
            parser_cls=QwenXmlParser,
            suppress_calls=forbids_tool_calls(tool_choice),
        )
        for i in range(0, len(A_CALL), chunk):
            parser.process(A_CALL[i : i + chunk])
        return parser.flush()

    def test_a_format_that_normalises_its_framing_still_does(self):
        """Suppressing a call must not also switch off reading the format.

        Kimi-K3 wraps every answer, tool call or not, in channel tokens that
        the reader removes. Answering `tool_choice: "none"` by using no
        parser removed nothing, so the client was handed the wire.
        """
        framed = (
            "<|open|>response<|sep|>Hello there."
            "<|close|>response<|sep|><|end_of_msg|>"
        )
        forbidden = "".join(
            d
            for k, d in self._stream("none", text=framed, parser_cls=KimiK3Parser)
            if k == "content"
        )
        allowed = "".join(
            d
            for k, d in self._stream(None, text=framed, parser_cls=KimiK3Parser)
            if k == "content"
        )
        assert forbidden == allowed == "Hello there."

    def test_every_construction_site_goes_through_the_helper(self):
        """The gap this had was a site nobody listed, so count them rather
        than list them: a parser built straight from the resolved class again
        is one endpoint that stopped honouring the field."""
        roots = [
            pathlib.Path(serving_chat.__file__),
            pathlib.Path(api_server.__file__),
            pathlib.Path(atom_standalone_service.__file__),
        ]
        built = unguarded = 0
        for path in roots:
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name not in ("ToolCallStreamParser", "parse_tool_calls"):
                    continue
                built += 1
                parser_arg = next(
                    (k.value for k in node.keywords if k.arg == "parser_cls"), None
                )
                suppress = next(
                    (k.value for k in node.keywords if k.arg == "suppress_calls"), None
                )
                via_helper = (
                    isinstance(suppress, ast.Call)
                    and getattr(suppress.func, "id", None) == "forbids_tool_calls"
                )
                if not (via_helper or _is_none(parser_arg)):
                    unguarded += 1
        assert built >= 4, f"only {built} construction sites found; matcher is stale"
        assert unguarded == 0, f"{unguarded} of {built} sites bypass the helper"


def _is_none(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


class TestTheStreamReportsWhyItStopped:
    """`stop` was hardcoded on every streaming path.

    So a response the engine cut off at `max_tokens` reported completion
    while `stream=false` reported `length` for the same generation, and an
    agentic client kept a truncated answer as final. A `stop_<token_id>`
    stop -- a model EOS token other than the primary one, which both
    endpoints report as an ordinary ending -- collapsed
    the same way. All three lines sat next to edits this branch made.
    """

    @pytest.mark.parametrize(
        "engine_reason, expected",
        [
            ("max_tokens", "length"),
            ("length", "length"),
            ("stop", "stop"),
            ("stop_163586", "stop"),
        ],
    )
    def test_the_two_paths_agree_on_the_reason(self, engine_reason, expected):
        assert _build_chat_choice("hi", engine_reason)["finish_reason"] == expected
        assert (openai_stop_reason(engine_reason) or "stop") == expected

    def test_no_reason_at_all_still_ends_the_stream(self):
        """The engine always gives one for a finished generation; if it did
        not, a stream still has to close on something."""
        assert (openai_stop_reason(None) or "stop") == "stop"

    def test_no_streaming_path_hardcodes_stop(self):
        """Counted rather than listed: the gap was a site nobody listed."""
        hardcoded = 0
        for path in (
            pathlib.Path(serving_chat.__file__),
            pathlib.Path(atom_standalone_service.__file__),
        ):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.IfExp):
                    continue
                orelse = node.orelse
                if isinstance(orelse, ast.Constant) and orelse.value == "stop":
                    hardcoded += 1
        assert hardcoded == 0, (
            f"{hardcoded} streaming path(s) still report `stop` regardless of "
            "why the engine stopped"
        )


class TestBeingCutShortOutranksHavingMadeACall:
    """`tool_calls` says "act on this"; `length` says "this is not all of it".

    A response the engine cut off mid-call still parses to a call -- every
    format's unclosed-region branch exists to salvage exactly that -- but its
    last argument value is silently truncated. Reporting `tool_calls` told the
    client to run the tool with half its arguments and no sign anything was
    missing. OpenAI reports `length` for a truncated response whatever else is
    in it.
    """

    @pytest.mark.parametrize(
        "engine_reason, has_calls, expected",
        [
            ("max_tokens", True, "length"),
            ("max_tokens", False, "length"),
            ("length", True, "length"),
            ("eos", True, "tool_calls"),
            ("stop_163586", True, "tool_calls"),
            ("eos", False, "stop"),
            (None, True, "tool_calls"),
            (None, False, "stop"),
        ],
    )
    def test_the_rule(self, engine_reason, has_calls, expected):
        assert openai_stop_reason_with_calls(engine_reason, has_calls) == expected

    def test_end_to_end_on_a_call_the_engine_cut_off(self):
        truncated = "<tool_call><function=get_weather><parameter=city>Par"
        choice = _build_chat_choice(
            truncated,
            "max_tokens",
            tools=TOOLS,
            tool_parser_cls=QwenXmlParser,
        )
        assert choice["message"]["tool_calls"], "the salvaged call is the premise"
        assert choice["finish_reason"] == "length"


class TestTheReasonAResponseEnded:
    """`""` is a reason the engine really forwards, and it is not absence.

    `Sequence.leave_reason` starts as the empty string, and a response with no
    recorded reason arrives carrying it. A truthiness test put
    `finish_reason: null` in the body, which the OpenAI schema reserves for a
    choice that is still being generated -- clients wait on it, and some SDKs
    reject the response outright.
    """

    def _choice(self, finish_reason):
        return _build_chat_choice(
            "the answer", finish_reason, 0, None, None, NO_REASONING, None
        )

    def test_an_empty_engine_reason_still_terminates_the_choice(self):
        assert self._choice("")["finish_reason"] == "stop"

    def test_and_a_genuinely_absent_one_is_still_null(self):
        assert self._choice(None)["finish_reason"] is None


class TestBothEndpointsAskOneQuestionAboutAToolName:
    """What a request may declare is what a parser can dispatch.

    There were two grammars. `protocol.TOOL_NAME_RE`
    (`^[A-Za-z_][A-Za-z0-9_-]*$`) guarded the chat entrance and was the
    stricter: it rejected a leading digit that OpenAI's own
    `^[a-zA-Z0-9_-]{1,64}$` allows, every non-ASCII name, and the
    `server.tool` spelling MCP namespaces with -- so declaring an MCP tool was
    a 400 before the model ever ran. `usable_tool_name` guarded the response
    side and allowed all three. And `/v1/messages` asked neither: it converted
    the tools and validated nothing, so the two endpoints disagreed about
    which tools were legal at all.
    """

    #: Names a client may legitimately declare, and the shape of the mistake
    #: each one used to be taken for.
    LEGAL = ("get_weather", "server.tool", "7z_extract", "查天气", "a-b", "_x")
    ILLEGAL = ("", "   ", "two words", ".hidden", "-lead", "a b.c")

    @staticmethod
    def _openai(name):
        return [{"type": "function", "function": {"name": name, "parameters": {}}}]

    @pytest.mark.parametrize("name", LEGAL)
    def test_a_dispatchable_name_is_accepted_by_the_chat_entrance(self, name):
        validate_tool_list(self._openai(name))

    @pytest.mark.parametrize("name", ILLEGAL)
    def test_an_undispatchable_name_is_refused_by_the_chat_entrance(self, name):
        with pytest.raises(ValueError):
            validate_tool_list(self._openai(name))

    @pytest.mark.parametrize("name", LEGAL + ILLEGAL)
    def test_the_entrance_and_the_parsers_agree(self, name):
        """The property, of which the two tables above are the illustration.

        Neither side may be the stricter: a name the entrance refuses is a
        tool the client cannot declare, and a name the parsers refuse is a
        call the model can never be seen to make.
        """
        try:
            validate_tool_list(self._openai(name))
            accepted = True
        except ValueError:
            accepted = False
        assert accepted == usable_tool_name(name), (
            f"the entrance and `usable_tool_name` disagree about {name!r}: "
            f"entrance={accepted}, parsers={usable_tool_name(name)}"
        )

    @pytest.mark.parametrize("name", LEGAL + ILLEGAL)
    def test_the_anthropic_entrance_asks_the_same_question(self, name):
        """`/v1/messages` validates the conversion, so there is one rule and
        not a second written in Anthropic's spelling."""
        converted = anthropic_to_openai_tools([{"name": name, "input_schema": {}}])
        try:
            validate_tool_list(converted)
            accepted = True
        except ValueError:
            accepted = False
        assert accepted == usable_tool_name(
            name
        ), f"/v1/messages and /v1/chat/completions disagree about {name!r}"


class TestFanoutSeedsImplicitReasoning:
    """A template-injected opener must reach every fan-out sibling's filter.

    PR #1961's class, translated: it passed `starts_thinking: bool` and this
    branch passes a `ReasoningChannel`, because the dialect and the
    starts-open flag were being handed to different readers and had to travel
    together. The assertions are the same three.

    Kept as behaviour beside `TestEverySeedingSiteIsSeeded`, which walks the
    AST of `atom/entrypoints/` and refuses an unseeded construction site. That
    scan is structural and would pass a site that is seeded with the wrong
    thing; this one drives the generator and reads the deltas. #1961 makes the
    point that decides it: a direct `ReasoningFilter(starts_thinking=True)`
    unit test cannot catch this, because the defect is in the wiring.
    """

    # Long enough to pass any grace period a buffering filter might have, and
    # carrying a '<' -- the two conditions that decided whether unseeded
    # reasoning leaked.
    REASONING = "Checking the bound: if (a < b) we return a. " + "Considering. " * 12

    def _deltas(self, starts_open):
        """Every delta the fan-out yields for a two-sibling reasoning stream."""

        async def run():
            collector = StreamOutputCollector("req-rs")
            for index in (0, 1):
                collector.put_nowait(
                    (
                        index,
                        {"text": self.REASONING, "token_ids": [1], "finished": False},
                    )
                )
            gen = stream_chat_response_fanout(
                request_id="req-rs",
                model="model",
                shared_collector=collector,
                seq_ids=[0, 1],
                num_prompt_tokens=1,
                cleanup_stream=lambda *a, **k: None,
                cleanup_request=lambda *a, **k: None,
                reasoning=ReasoningChannel(starts_open=starts_open),
            )
            out = []
            for _ in range(6):
                try:
                    raw = await asyncio.wait_for(gen.__anext__(), timeout=2)
                except (StopAsyncIteration, TimeoutError):
                    break
                if raw.startswith("data: ") and not raw.startswith("data: [DONE]"):
                    out.append(json.loads(raw[6:])["choices"][0]["delta"])
            await gen.aclose()
            return out

        return asyncio.run(run())

    def test_pre_close_text_is_reasoning_not_content(self):
        deltas = self._deltas(starts_open=True)

        leaked = [d for d in deltas if d.get("content")]
        assert not leaked, f"reasoning leaked to content: {leaked}"
        assert any(
            d.get("reasoning_content") for d in deltas
        ), f"no reasoning_content was emitted at all: {deltas}"

    def test_without_the_seed_the_same_text_is_not_reasoning(self):
        # The contrast that shows the seed is what classified it, not the text.
        # Asserting the text arrives as *content* here would not work: content
        # segments pass through the tool-call parser, which holds them until
        # the stream finishes, while reasoning_content is forwarded at once.
        assert not any(
            d.get("reasoning_content") for d in self._deltas(starts_open=False)
        )

    def test_both_streaming_entry_points_accept_the_seed(self):
        # #1961's bug was a missing parameter, so pin the shape of both.
        for fn in (stream_chat_response, stream_chat_response_fanout):
            assert (
                "reasoning" in inspect.signature(fn).parameters
            ), f"{fn.__name__} cannot be told the prompt opened reasoning"
