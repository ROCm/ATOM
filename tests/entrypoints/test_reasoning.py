# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for reasoning/thinking content separation."""

from atom.entrypoints.openai.reasoning import (
    ReasoningFilter,
    separate_reasoning,
)

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

    def test_whitespace_after_thinking(self):
        text = "<think>thought</think>\n\nThe answer."
        reasoning, content = separate_reasoning(text)
        assert content == "The answer."

    def test_no_think_start_tag(self):
        """MiniMax M2.7 pattern: model doesn't generate <think>, only </think>.
        The chat template injects <think> as part of the prompt."""
        text = "The user wants hello world...\n</think>\n\nprint('Hello')"
        reasoning, content = separate_reasoning(text)
        assert reasoning == "The user wants hello world..."
        assert content == "print('Hello')"

    def test_no_think_start_tag_empty_content(self):
        text = "Reasoning only\n</think>"
        reasoning, content = separate_reasoning(text)
        assert reasoning == "Reasoning only"
        assert content == ""


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


class TestNonReasoningOutputIsNotWithheld:
    """A stream that never opens a reasoning channel must still reach the client.

    Regression for an observed production stall: 1.5-2.4% of requests on a
    Claude-Code agent corpus returned HTTP 200 and then **zero body bytes** until
    the client's 600s read timeout, while the engine kept decoding to
    max_tokens. The cause was state 0 refusing to release the buffer whenever a
    '<' appeared anywhere in it -- and code-bearing answers are full of '<'.
    Holding back a partial *trailing* marker is correct; holding the whole
    buffer because it contains a '<' somewhere is not.
    """

    def _emit(self, chunks):
        f = ReasoningFilter()
        out = []
        for c in chunks:
            out.extend(f.process(c))
        return out

    def test_angle_bracket_in_plain_output_does_not_withhold_forever(self):
        # A '<' that is not a reasoning marker: the classic case is code.
        text = "Here is the fix:\n\nif (a < b) { return a; }\n" + "x" * 200
        emitted = self._emit([text])
        assert emitted, "nothing was emitted: the stream would stall"
        assert "".join(t for f, t in emitted if f == "content").startswith(
            "Here is the fix:"
        )

    def test_html_like_output_does_not_withhold_forever(self):
        text = "<div class='x'>hello</div>\n" + "y" * 200
        emitted = self._emit([text])
        assert emitted, "nothing was emitted: the stream would stall"

    def test_token_by_token_code_output_streams(self):
        # The real shape: many small chunks, one of which contains '<'.
        chunks = ["Sure. ", "Use ", "a < b ", "to compare. "] + [
            "word " for _ in range(60)
        ]
        emitted = self._emit(chunks)
        assert emitted, "nothing was emitted: the stream would stall"

    def test_a_real_think_block_still_separates(self):
        # The fix must not cost the feature it guards.
        f = ReasoningFilter()
        out = []
        for c in ["<think>", "pondering", "</think>", "answer"]:
            out.extend(f.process(c))
        assert ("reasoning_content", "pondering") in out
        assert ("content", "answer") in out

    def test_partial_end_marker_is_still_held_back(self):
        # A marker split across chunks must not leak as content.
        f = ReasoningFilter()
        out = []
        for c in ["<think>", "abc", "</thi"]:
            out.extend(f.process(c))
        assert not any("</thi" in t for _, t in out), "leaked a partial marker"

    def test_template_injected_opener_still_reaches_end_marker(self):
        # starts_thinking=True => state 1; unchanged by this fix.
        f = ReasoningFilter(starts_thinking=True)
        out = []
        for c in ["reasoning here", "</think>", "final"]:
            out.extend(f.process(c))
        assert ("reasoning_content", "reasoning here") in out
        assert ("content", "final") in out
