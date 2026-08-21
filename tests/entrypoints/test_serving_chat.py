# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for chat completion serving logic (chunk creation, response building)."""

import ast
import asyncio
import json
import pathlib

from atom.entrypoints.openai import serving_chat
from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    create_chat_chunk,
    normalize_chat_tools,
    stream_chat_response,
    stream_chat_response_fanout,
)
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser

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
            "req-1", "model", raw, output, tool_parser_cls=KimiParser
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


def _forbids_none(node) -> bool:
    """Is this the expression `tool_choice != "none"`?"""
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "tool_choice"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotEq)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "none"
    )


class TestToolChoiceIsHonouredEverywhereItIsEmitted:
    """`tool_choice="none"` forbids tool calls on every path that can emit one.

    The single-choice stream gated on it and the fan-out did not, so the same
    request with `n=2, stream=true` returned `tool_calls` deltas and
    `finish_reason="tool_calls"` that `n=1` suppressed. Checked by walking the
    source rather than by listing the four call sites, because the gap was a
    site nobody listed.
    """

    SOURCE = pathlib.Path(serving_chat.__file__)

    def test_no_tool_event_is_emitted_without_checking_tool_choice(self):
        tree = ast.parse(self.SOURCE.read_text())
        unguarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            left = node.left
            if not (isinstance(left, ast.Name) and left.id == "event_type"):
                continue
            const = node.comparators[0]
            if not (
                isinstance(const, ast.Constant)
                and isinstance(const.value, str)
                and const.value.startswith("tool_call_")
                and const.value != "tool_call_end"
            ):
                continue
            # The comparison is the left half of `event_type == X and <gate>`;
            # walk up is not available, so re-find the enclosing BoolOp. The
            # gate must be `tool_choice != "none"` specifically -- an earlier
            # version asked only whether the name appeared anywhere, and
            # passed when every gate was inverted to `== "none"`.
            guarded = any(
                isinstance(b, ast.BoolOp)
                and node in b.values
                and any(_forbids_none(v) for v in b.values)
                for b in ast.walk(tree)
                if isinstance(b, ast.BoolOp)
            )
            if not guarded:
                unguarded.append(f"line {node.lineno}: {const.value}")
        assert not unguarded, "tool events emitted regardless of tool_choice:\n  " + (
            "\n  ".join(unguarded)
        )

    def test_the_scan_matched_something(self):
        """A matcher that finds no nodes passes forever.

        `event_type` is one rename away from `etype`, which api_server already
        uses for the same variable -- and this scan would then be vacuously
        green.
        """
        tree = ast.parse(self.SOURCE.read_text())
        matched = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Compare)
            and isinstance(n.left, ast.Name)
            and n.left.id == "event_type"
        ]
        assert len(matched) >= 4, f"only {len(matched)} event_type comparisons found"

    def test_an_inverted_gate_is_not_accepted(self):
        """`tool_choice == "none"` is the opposite rule and must not pass."""
        assert _forbids_none(ast.parse('tool_choice != "none"', mode="eval").body)
        assert not _forbids_none(ast.parse('tool_choice == "none"', mode="eval").body)
        assert not _forbids_none(ast.parse("tool_choice is not None", mode="eval").body)
