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
