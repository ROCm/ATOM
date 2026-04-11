# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for chat completion serving logic (chunk creation, response building)."""

import json


from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    create_chat_chunk,
)

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
