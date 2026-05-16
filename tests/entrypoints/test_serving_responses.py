# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Responses API serving helpers."""

from __future__ import annotations

import asyncio
import json

from atom.entrypoints.openai.serving_responses import (
    build_responses_response,
    stream_responses_response,
)


class _FakeTokenizer:
    def encode(self, text: str):
        return text.split()


def _decode_sse_event(chunk: str):
    assert chunk.startswith("data: ")
    payload = chunk.removeprefix("data: ").strip()
    if payload == "[DONE]":
        return payload
    return json.loads(payload)


def test_build_responses_response_splits_reasoning_and_text():
    response = build_responses_response(
        "resp-test",
        "test-model",
        "<think>use arithmetic</think>2 + 3 = 5",
        {"num_tokens_input": 4, "num_tokens_output": 6},
    )

    assert response.object == "response"
    assert response.status == "completed"
    assert response.output_text == "2 + 3 = 5"
    assert response.usage == {
        "input_tokens": 4,
        "output_tokens": 6,
        "total_tokens": 10,
    }
    assert [item["type"] for item in response.output] == ["reasoning", "message"]
    assert response.output[0]["content"][0]["text"] == "use arithmetic"
    assert response.output[1]["content"][0]["text"] == "2 + 3 = 5"


def test_stream_responses_response_emits_text_events_and_done_marker():
    async def collect_events():
        queue = asyncio.Queue()
        await queue.put({"text": "Hello", "token_ids": [1], "finished": False})
        await queue.put(
            {
                "text": " world",
                "token_ids": [2],
                "finished": True,
                "finish_reason": "stop",
            }
        )
        cleaned = []

        def cleanup(request_id, seq_id):
            cleaned.append((request_id, seq_id))

        return [
            _decode_sse_event(chunk)
            async for chunk in stream_responses_response(
                "resp-stream",
                "test-model",
                "hello prompt",
                queue,
                42,
                _FakeTokenizer(),
                cleanup,
            )
        ], cleaned

    events, cleaned = asyncio.run(collect_events())
    event_types = [event if event == "[DONE]" else event["type"] for event in events]

    assert event_types[:2] == ["response.created", "response.in_progress"]
    assert "response.output_item.added" in event_types
    assert event_types.count("response.output_text.delta") >= 1
    assert "response.output_text.done" in event_types
    assert "response.completed" in event_types
    assert event_types[-1] == "[DONE]"
    assert cleaned == [("resp-stream", 42)]

    completed = next(
        event
        for event in events
        if event != "[DONE]" and event["type"] == "response.completed"
    )
    assert completed["response"]["output_text"] == "Hello world"
    assert completed["response"]["usage"] == {
        "input_tokens": 2,
        "output_tokens": 2,
        "total_tokens": 4,
    }


def test_stream_responses_response_keeps_reasoning_and_text_indices_stable():
    async def collect_events():
        queue = asyncio.Queue()
        await queue.put(
            {
                "text": "<think>reason</think>answer",
                "token_ids": [1, 2],
                "finished": True,
            }
        )
        return [
            _decode_sse_event(chunk)
            async for chunk in stream_responses_response(
                "resp-indices",
                "test-model",
                "prompt",
                queue,
                7,
                _FakeTokenizer(),
                lambda _request_id, _seq_id: None,
            )
        ]

    events = [event for event in asyncio.run(collect_events()) if event != "[DONE]"]
    reasoning_delta = next(
        event for event in events if event["type"] == "response.reasoning_text.delta"
    )
    reasoning_done = next(
        event for event in events if event["type"] == "response.reasoning_text.done"
    )
    text_delta = next(
        event for event in events if event["type"] == "response.output_text.delta"
    )
    text_done = next(
        event for event in events if event["type"] == "response.output_text.done"
    )

    assert reasoning_delta["output_index"] == reasoning_done["output_index"]
    assert text_delta["output_index"] == text_done["output_index"]
    assert reasoning_delta["output_index"] != text_delta["output_index"]
