# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for reasoning/thinking content separation."""

from typing import ClassVar

import pytest

from atom.entrypoints.openai.reasoning import (
    ReasoningChannel,
    ReasoningFilter,
    prompt_starts_in_reasoning,
    separate_reasoning,
)
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser, parse_tool_calls
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

# ============================================================================
# separate_reasoning() Tests
# ============================================================================


class TestSeparateReasoning:
    """Tests for the separate_reasoning() function."""

    def test_with_thinking_block(self):
        text = "<think>Let me think about this.</think>The answer is 42."
        reasoning, content = separate_reasoning(text)
        assert reasoning == "Let me think about this."
        assert content == "The answer is 42."

    def test_without_thinking(self):
        text = "The answer is 42."
        reasoning, content = separate_reasoning(text)
        assert reasoning is None
        assert content == "The answer is 42."

    def test_empty_thinking(self):
        text = "<think></think>Just the answer."
        reasoning, content = separate_reasoning(text)
        assert reasoning is None
        assert content == "Just the answer."

    def test_unclosed_thinking(self):
        """Truncated response where </think> was never generated."""
        text = "<think>I'm still thinking about this and the response got truncated"
        reasoning, content = separate_reasoning(text)
        assert reasoning is not None
        assert "still thinking" in reasoning
        assert content == ""

    def test_multiline_thinking(self):
        text = (
            "<think>Step 1: analyze\nStep 2: compute\nStep 3: answer</think>Result: 42"
        )
        reasoning, content = separate_reasoning(text)
        assert "Step 1" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 42"

    def test_tool_calls_preserved(self):
        """Tool calls are NOT stripped by separate_reasoning (handled by tool_parser)."""
        text = "Hello<|tool_calls_section_begin|>function call here<|tool_calls_section_end|>"
        reasoning, content = separate_reasoning(text)
        assert reasoning is None
        # Tool call tokens remain — tool_parser.parse_tool_calls() handles them
        assert "Hello" in content
        assert "<|tool_calls_section_begin|>" in content

    def test_thinking_with_tool_call(self):
        text = (
            "<think>thinking</think>Answer"
            "<|tool_calls_section_begin|>call<|tool_calls_section_end|>"
        )
        reasoning, content = separate_reasoning(text)
        assert reasoning == "thinking"
        # Content includes tool call tokens (parsed separately by tool_parser)
        assert "Answer" in content

    def test_only_thinking_no_answer(self):
        """Model generated only thinking content then stopped."""
        text = "<think>thinking only</think>"
        reasoning, content = separate_reasoning(text)
        assert reasoning == "thinking only"
        assert content == ""

    def test_whitespace_after_thinking_survives(self):
        """It used to be trimmed here and delivered by the streaming path.

        The newline a model writes before its answer is not a marker this
        dialect declares, and only markers may be removed -- the same rule
        `ToolCallParser.parse` states one stage later, after a trailing
        `.strip()` there cost a code-block answer its final newline.
        """
        text = "<think>thought</think>\n\nThe answer."
        reasoning, content = separate_reasoning(text)
        assert content == "\n\nThe answer."
        assert reasoning == "thought"

    def test_no_think_start_tag(self):
        """MiniMax M2.7 pattern: model doesn't generate <think>, only </think>.

        The chat template injects <think> into the prompt, which is what
        `starts_thinking` carries -- the text itself cannot say so, and this
        used to be inferred from the bare `</think>`. Inferring it is what the
        streaming path cannot do without waiting for an end marker that may
        never come, so the two disagreed; the caller answers it once now.
        """
        text = "The user wants hello world...\n</think>\n\nprint('Hello')"
        reasoning, content = separate_reasoning(text, starts_thinking=True)
        assert reasoning == "The user wants hello world...\n"
        assert content == "\n\nprint('Hello')"

    def test_no_think_start_tag_empty_content(self):
        text = "Reasoning only\n</think>"
        reasoning, content = separate_reasoning(text, starts_thinking=True)
        assert reasoning == "Reasoning only\n"
        assert content == ""

    def test_an_unopened_end_tag_is_text_when_the_prompt_did_not_open_one(self):
        """The other side of the same switch, and the reason it is a switch.

        Unseeded, nothing opened a reasoning channel, so a stray `</think>`
        is a string the model wrote. Claiming it as a delimiter is a guess,
        and the guess is what made an ordinary answer wait for an end marker
        that was never coming.
        """
        text = "The user wants hello world...\n</think>\n\nprint('Hello')"
        reasoning, content = separate_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_streaming_and_non_streaming_agree_on_a_truncated_trace(self):
        """Same prompt, same output, same split -- whether streamed or not.

        A reasoning model stopped at `max_tokens` emits no end marker. Seeded,
        both paths must call the whole thing reasoning; the streaming path did
        while this one called it content, so a client reading `content` saw
        the trace and a client reading `delta.content` saw nothing.
        """
        raw = "Let me consider whether the user wants a GLM parser. " * 4
        reasoning, content = separate_reasoning(raw, starts_thinking=True)

        rf = ReasoningFilter(starts_thinking=True)
        segments = rf.process(raw) + rf.flush()
        streamed_reasoning = "".join(s for f, s in segments if f == "reasoning_content")
        streamed_content = "".join(s for f, s in segments if f == "content")

        assert (reasoning or "") == streamed_reasoning
        assert content == streamed_content


# ============================================================================
# ReasoningFilter (Streaming) Tests
# ============================================================================


class TestReasoningFilter:
    """Tests for the ReasoningFilter streaming state machine."""

    def _run_filter(self, tokens):
        """Helper: run tokens through filter and return all segments."""
        rf = ReasoningFilter()
        results = []
        for token in tokens:
            results.extend(rf.process(token))
        results.extend(rf.flush())
        return results

    def test_simple_thinking_and_content(self):
        tokens = ["<think>", "thinking", "</think>", "answer"]
        results = self._run_filter(tokens)
        reasoning = "".join(t for f, t in results if f == "reasoning_content")
        content = "".join(t for f, t in results if f == "content")
        assert "thinking" in reasoning
        assert "answer" in content

    def test_no_thinking(self):
        tokens = ["Hello", " world", "!"]
        results = self._run_filter(tokens)
        content = "".join(t for f, t in results if f == "content")
        assert "Hello" in content
        assert "world" in content
        # No reasoning
        reasoning = [t for f, t in results if f == "reasoning_content"]
        assert len(reasoning) == 0

    def test_think_tag_in_single_token(self):
        tokens = ["<think>all thinking</think>the answer"]
        results = self._run_filter(tokens)
        reasoning = "".join(t for f, t in results if f == "reasoning_content")
        content = "".join(t for f, t in results if f == "content")
        assert "all thinking" in reasoning
        assert "the answer" in content

    def test_multiple_tokens_in_thinking(self):
        tokens = ["<think>", "step", " 1", " step", " 2", "</think>", "done"]
        results = self._run_filter(tokens)
        reasoning = "".join(t for f, t in results if f == "reasoning_content")
        content = "".join(t for f, t in results if f == "content")
        assert "step 1" in reasoning
        assert "step 2" in reasoning
        assert content == "done"

    def test_tool_calls_passed_through(self):
        """ReasoningFilter passes tool call tokens through as content
        (ToolCallStreamParser handles them in serving_chat)."""
        tokens = [
            "<think>",
            "think",
            "</think>",
            "Hi",
            "<|tool_calls_section_begin|>",
            "call",
            "<|tool_calls_section_end|>",
        ]
        results = self._run_filter(tokens)
        content = "".join(t for f, t in results if f == "content")
        assert "Hi" in content
        # Tool call tokens are preserved (handled by ToolCallStreamParser)
        assert "<|tool_calls_section_begin|>" in content

    def test_content_before_think(self):
        """Content before <think> should be emitted as content."""
        tokens = ["prefix", "<think>", "thought", "</think>", "suffix"]
        results = self._run_filter(tokens)
        content = "".join(t for f, t in results if f == "content")
        reasoning = "".join(t for f, t in results if f == "reasoning_content")
        assert "prefix" in content
        assert "suffix" in content
        assert "thought" in reasoning

    def test_flush_remaining_buffer(self):
        """Flush should emit any remaining buffered content."""
        rf = ReasoningFilter()
        # Short text that doesn't trigger immediate emit (buffered for tag detection)
        results = rf.process("Hi")
        results.extend(rf.flush())
        content = "".join(t for f, t in results if f == "content")
        assert "Hi" in content


class TestTheChannelEndsAtWhicheverCloserComesFirst:
    """One rule, where there were three branches and one of them lost data.

    Driven through a `ReasoningChannel` naming the channel dialect, because
    that is now what decides these markers mean anything: a model resolves to
    one dialect at startup and both delivery modes read that one. Asked of the
    module-level helpers instead, this would be asserting the old behaviour --
    every dialect tried in turn on one path and the union of their markers on
    the other, which is the divergence the channel exists to remove.

    A channel format can leave the think channel by *opening* another one, so
    `<|close|>think<|sep|>` and `<|open|>response<|sep|>` both end it. The
    branch for the second returned `reasoning=None` and discarded everything
    ahead of the marker: a single byte between the two was enough to reach
    it, and the chain of thought then appeared in neither field.

    And all three were ungated, so a model that merely *quotes* one of these
    tokens had the text before it deleted -- the inference `parse_tool_calls`
    was changed to stop making, in the half still making it.
    """

    THINK_END = "<|close|>think<|sep|>"
    RESPONSE = "<|open|>response<|sep|>"
    K3 = ReasoningChannel(dialect=resolve_dialect(THINK_END)[0], starts_open=True)
    K3_CLOSED = ReasoningChannel(dialect=K3.dialect, starts_open=False)

    def test_a_byte_between_the_markers_keeps_the_reasoning(self):
        text = f"my reasoning{self.THINK_END}\n{self.RESPONSE}the answer"
        reasoning, content = self.K3.split(text)
        assert reasoning == "my reasoning"
        assert "the answer" in content

    def test_opening_the_response_channel_ends_the_reasoning(self):
        text = f"my reasoning{self.RESPONSE}the answer"
        reasoning, content = self.K3.split(text)
        assert reasoning == "my reasoning"
        assert content == "the answer"

    @pytest.mark.parametrize("marker_attr", ["THINK_END", "RESPONSE"])
    def test_a_quoted_token_does_not_open_a_channel_that_was_closed(self, marker_attr):
        """It stays content. Whether the *token* survives is a separate
        question -- `<|open|>response<|sep|>` is framing this format wraps
        every answer in, so it is removed on both delivery paths, while
        `<|close|>think<|sep|>` is a channel delimiter and only means anything
        once a channel is open."""
        marker = getattr(self, marker_attr)
        text = f"The K3 format uses {marker} to open the answer."
        reasoning, content = self.K3_CLOSED.split(text)
        assert reasoning is None
        assert content == self.K3_CLOSED.dialect.strip_framing(text)
        assert "The K3 format uses" in content and "to open the answer." in content

    @pytest.mark.parametrize("chunk", [1, 3, 999])
    def test_the_streaming_filter_closes_on_the_same_markers(self, chunk):
        """It knew only the explicit close, so a K3 answer that goes straight
        to the response channel -- its own docs call that the common path --
        streamed entirely as `reasoning_content` with an empty `content`."""
        text = f"{self.RESPONSE}Hello world"
        f = self.K3.stream()
        segs = []
        for i in range(0, len(text), chunk):
            segs += f.process(text[i : i + chunk])
        segs += f.flush()
        assert "".join(s for k, s in segs if k == "content") == "Hello world"
        assert "".join(s for k, s in segs if k == "reasoning_content") == ""


class TestTheDialectIsReadFromWhicheverEvidenceExists:
    """A rendered prompt and a template source are both evidence, and only one
    of them is always available.

    Kimi-K3 ships neither a Jinja `chat_template` nor an encoder under the path
    the loader searches, so `chat_template_source` returns `""` for it. Reading
    only the source therefore fell back to the inline-`<think>` dialect and a
    K3 answer came back as `reasoning_content` with `content` empty -- while
    the *tool-call* format resolved correctly, because that one reads the
    rendered probe. One model, two answers to "what does this speak", from two
    different pieces of evidence.
    """

    K3_PROMPT = "<|im_system|>you are helpful<|im_end|><|open|>think<|sep|>"
    QWEN_SOURCE = "{% if enable_thinking %}<think>{% endif %}...</think>"

    def test_the_render_answers_when_the_source_cannot(self):
        dialect, stated = resolve_dialect("", self.K3_PROMPT)
        assert dialect.think_end_marker == "<|close|>think<|sep|>"
        assert stated, "a named dialect must not be reported as a fallback"

    def test_the_source_still_answers_when_there_is_one(self):
        dialect, stated = resolve_dialect(self.QWEN_SOURCE, "...<think>")
        assert dialect.think_end_marker == "</think>" and stated

    def test_neither_falls_back_and_says_so(self):
        dialect, stated = resolve_dialect("", "")
        assert dialect.think_end_marker == "</think>"
        assert not stated, "a fallback must be reported as one"

    ANSWER = (
        "The user wants the capital.<|close|>think<|sep|>"
        "<|open|>response<|sep|>Paris.<|close|>response<|sep|><|end_of_msg|>"
    )

    def _channel(self):
        dialect, _ = resolve_dialect("", self.K3_PROMPT)
        return ReasoningChannel(dialect=dialect, starts_open=True)

    def test_a_k3_answer_is_split_by_the_dialect_it_resolved_to(self):
        """The consequence, end to end: the fallback returned `content: ''`."""
        reasoning, content = self._channel().split(self.ANSWER)
        assert reasoning == "The user wants the capital."
        assert "Paris." in content

    @pytest.mark.parametrize("chunk", [1, 5, 999])
    def test_and_both_delivery_modes_agree_through_the_whole_pipeline(self, chunk):
        """Reasoning *and* the tool parser, because the channel framing is the
        latter's to remove.

        The reasoning split used to strip it as well, which made the two paths
        agree only by way of a third stage and only when a K3 parser had been
        resolved -- and it truncated at `<|end_of_msg|>` rather than removing
        it, deleting anything after that token on one path alone.
        """
        channel = self._channel()
        reasoning, rest = channel.split(self.ANSWER)
        content, _ = parse_tool_calls(rest, None, KimiK3Parser)

        filt, parser = channel.stream(), ToolCallStreamParser(parser_cls=KimiK3Parser)
        streamed_reasoning, streamed_content = [], []
        segments = []
        for i in range(0, len(self.ANSWER), chunk):
            segments += filt.process(self.ANSWER[i : i + chunk])
        segments += filt.flush()
        for field, seg in segments:
            if field == "reasoning_content":
                streamed_reasoning.append(seg)
            else:
                streamed_content += [
                    d for k, d in parser.process(seg) if k == "content"
                ]
        streamed_content += [d for k, d in parser.flush() if k == "content"]

        assert "".join(streamed_reasoning) == reasoning
        assert "".join(streamed_content) == content == "Paris."


class TestTheReasoningStageAgreesWithItselfWithoutHelp:
    """`.split()` and `.stream()` must produce the same bytes on their own.

    Not "the same bytes once the tool parser has had them". The reasoning
    split used to remove Kimi-K3's channel framing itself, which the streaming
    filter has no step for -- so the two agreed only because a *third* stage
    removed it on the streamed path too, and only when a K3 parser had been
    resolved. Asserted here without a tool parser, which is the only way to
    see it: with one, both paths come out identical either way.
    """

    SHAPES: ClassVar[list[str]] = [
        "",
        "reasoning only",
        "reason</think>answer",
        "reason<|close|>think<|sep|>answer",
        "reason<|close|>think<|sep|><|open|>response<|sep|>Paris.",
        "reason<|close|>think<|sep|><|open|>response<|sep|>Paris."
        "<|close|>response<|sep|><|end_of_msg|>",
        "reason</think>answer<|end_of_msg|>and more",
        "</think>",
        "<|close|>think<|sep|>",
    ]

    @pytest.mark.parametrize("source", ["<think></think>", "<|open|>think<|sep|>"])
    @pytest.mark.parametrize("starts_open", [True, False])
    @pytest.mark.parametrize("chunk", [1, 4, 999])
    def test_the_two_modes_agree_byte_for_byte(self, source, starts_open, chunk):
        dialect, _ = resolve_dialect(source)
        channel = ReasoningChannel(dialect=dialect, starts_open=starts_open)
        for text in self.SHAPES:
            filt = channel.stream()
            segments = []
            for i in range(0, len(text), chunk):
                segments += filt.process(text[i : i + chunk])
            segments += filt.flush()
            streamed = (
                "".join(x for k, x in segments if k == "reasoning_content") or None,
                "".join(x for k, x in segments if k == "content"),
            )
            assert streamed == channel.split(text), (
                f"{dialect.think_end_marker!r} starts_open={starts_open} "
                f"chunk={chunk} text={text!r}: {streamed} != {channel.split(text)}"
            )

    @pytest.mark.parametrize("source", ["<think></think>", "<|open|>think<|sep|>"])
    @pytest.mark.parametrize("starts_open", [True, False])
    def test_and_neither_loses_a_byte_it_did_not_declare(self, source, starts_open):
        """Everything the model wrote comes back, minus this dialect's own
        markers and nothing else."""
        dialect, _ = resolve_dialect(source)
        channel = ReasoningChannel(dialect=dialect, starts_open=starts_open)
        declared = (
            dialect.output_open_marker,
            *dialect.end_markers,
            *dialect.content_framing,
        )
        for text in self.SHAPES:
            reasoning, content = channel.split(text)
            rebuilt = (reasoning or "") + content
            remainder = text
            for marker in declared:
                if marker:
                    remainder = remainder.replace(marker, "", 1)
            assert len(rebuilt) >= len(remainder) - 1, (
                f"{text!r} -> {rebuilt!r}: bytes went missing that no marker "
                "accounts for"
            )


class TestTheDialectsAsRegistered:
    """The properties above are checked on dialects; these check the registry.

    A property test that builds its own dialect proves the mechanism and
    nothing about what any model is actually served with. The channel format's
    answer framing was declared in exactly one place, and blanking it there
    left every mechanism test green while a Kimi-K3 client got
    `<|open|>response<|sep|>` in its content.
    """

    def test_the_channel_dialect_declares_the_framing_it_wraps_answers_in(self):
        dialect, _ = resolve_dialect("<|open|>think<|sep|>")
        framed = "<|open|>response<|sep|>Hi there.<|close|>response<|sep|>"
        assert ReasoningChannel(dialect=dialect).split(framed) == (None, "Hi there.")

    def test_minimax_m3_resolves_to_its_own_tags(self):
        """`<mm:think>`, not `<think>`.

        MiniMax-M3's template sets `think_begin_token = '<mm:think>'`, and no
        dialect declared either literal -- so it fell through to the inline
        `<think>` entry, matched nothing, and delivered the whole chain of
        thought as `content` with `reasoning_content: null` on both delivery
        paths. On `/v1/messages`, where reasoning is dropped by default, that
        put the trace inside the client's text block.
        """
        source = (
            "{%- set think_begin_token = '<mm:think>' -%}"
            "{%- set think_end_token = '</mm:think>' -%}"
        )
        dialect, stated = resolve_dialect(source)
        assert (dialect.think_end_marker, stated) == ("</mm:think>", True)
        channel = ReasoningChannel(dialect=dialect, starts_open=True)
        assert channel.split("Reasoning.</mm:think>The answer.") == (
            "Reasoning.",
            "The answer.",
        )
        # And the prompt marker, which is what seeds `starts_open` in the
        # first place: the template ends the generation prompt with the
        # opener under `thinking_mode == "enabled"`.
        assert prompt_starts_in_reasoning("]~b]ai\n<mm:think>")
        assert dialect.output_open_marker == "<mm:think>"

    def test_and_it_does_not_steal_the_generic_think_template(self):
        """The two marker sets must not shadow each other."""
        dialect, _ = resolve_dialect("<think></think>")
        assert dialect.think_end_marker == "</think>"

    def test_and_the_inline_think_dialect_declares_none(self):
        dialect, _ = resolve_dialect("<think></think>")
        assert dialect.content_framing == (), (
            "a dialect that wraps nothing must declare nothing, or it deletes "
            "literals out of ordinary answers"
        )
