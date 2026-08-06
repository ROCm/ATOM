# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for chat completion serving logic (chunk creation, response building)."""

import asyncio
import json

from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    create_chat_chunk,
    normalize_chat_tools,
    stream_chat_response,
    stream_chat_response_fanout,
)

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
        resp = build_chat_response("req-1", "model", raw, output)
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
            queue = asyncio.Queue()
            await queue.put({"text": "Hi", "token_ids": [1], "finished": True})
            gen = stream_chat_response(
                request_id="req-1",
                model="model",
                stream_queue=queue,
                seq_id=0,
                num_prompt_tokens=1,
                cleanup_fn=lambda *a, **k: None,
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
            queue = asyncio.Queue()
            # Empty text + finished=False means each queue item triggers
            # *only* the role-announcement yield (no content/finish chunks
            # in between), so the first two yields are guaranteed to be
            # sibling 0's and sibling 1's role chunks respectively.
            await queue.put((0, {"text": "", "token_ids": [], "finished": False}))
            await queue.put((1, {"text": "", "token_ids": [], "finished": False}))
            gen = stream_chat_response_fanout(
                request_id="req-2",
                model="model",
                shared_queue=queue,
                seq_ids=[0, 1],
                num_prompt_tokens=1,
                cleanup_fn=lambda *a, **k: None,
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
