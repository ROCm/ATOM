# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the ATOM standalone service's streaming chat state."""

import json
import queue
from typing import ClassVar

import pytest
from import_guard import skip_if_dependency_missing

try:
    from atom.entrypoints.atomesh.atom_standalone_service import (
        ChatCompletionStreamState,
    )
except Exception as exc:  # noqa: BLE001 pragma: no cover
    skip_if_dependency_missing(exc, "atomesh service import unavailable")
    ChatCompletionStreamState = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None

pytestmark = pytest.mark.skipif(
    ChatCompletionStreamState is None,
    reason=f"atom_standalone_service import unavailable: {_import_error!r}",
)


class _StubTokenizer:
    """Minimal tokenizer stub: only .encode() is used by ChatCompletionStreamState.__init__."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return ""


def _make_state(n: int) -> ChatCompletionStreamState:
    return ChatCompletionStreamState(
        request_id="chatcmpl-test",
        model_name="model",
        prompt="hello",
        tokenizer=_StubTokenizer(),
        stream_queue=queue.Queue(),
        n=n,
    )


class TestChatCompletionStreamStateRoleChunkContent:
    """Regression test: AtomStandaloneRouter::route_chat()'s streaming path
    must emit content="" with role="assistant", matching the fix in serving_chat.py's
    stream_chat_response/_fanout."""

    def test_single_sequence_role_chunk_has_empty_content(self):
        state = _make_state(n=1)

        chunks = state.drain(max_items=16)

        assert len(chunks) == 1
        assert chunks[0].startswith("data: ")
        data = json.loads(chunks[0][6:])
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["content"] == ""

    def test_fanout_role_chunks_have_empty_content(self):
        state = _make_state(n=3)

        chunks = state.drain(max_items=16)

        assert len(chunks) == 3
        for expected_index, raw_chunk in enumerate(chunks):
            assert raw_chunk.startswith("data: ")
            data = json.loads(raw_chunk[6:])
            choice = data["choices"][0]
            assert choice["index"] == expected_index
            delta = choice["delta"]
            assert delta["role"] == "assistant"
            assert delta["content"] == ""

    def test_role_chunks_not_resent_on_subsequent_drain(self):
        state = _make_state(n=1)

        first = state.drain(max_items=16)
        assert len(first) == 1

        second = state.drain(max_items=16, timeout=0.01)
        assert second == []


class TestTheDrainKeepsWhatItCannotHandOutYet:
    """`max_items` is a batch size, not a licence to discard.

    The build loop `break`ed when it reached that count -- after the parser
    had already yielded the events -- so everything past the first was gone
    permanently. At `max_items=1` a tool call lost its arguments and, once
    `has_tool_calls` moved onto the argument event, reported
    `finish_reason: stop` for a call it had just announced.
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
    CALL = (
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>"
    )

    def _state(self):
        from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser

        return ChatCompletionStreamState(
            request_id="chatcmpl-test",
            model_name="model",
            prompt="hello",
            tokenizer=_StubTokenizer(),
            stream_queue=queue.Queue(),
            n=1,
            tool_parser_cls=QwenXmlParser,
            tools=self.TOOLS,
        )

    @staticmethod
    def _drain_all(state, event, max_items):
        out = state._event_to_chunks(event, max_items)
        idle = {"index": 0, "text": "", "token_ids": [], "finished": True}
        for _ in range(20):
            more = state._event_to_chunks(idle, max_items)
            if not more:
                break
            out += more
        return out

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_the_arguments_survive_any_batch_size(self, max_items):
        state = self._state()
        event = {"index": 0, "text": self.CALL, "token_ids": [1], "finished": True}
        payloads = [
            json.loads(c.split("data: ", 1)[1])
            for c in self._drain_all(state, event, max_items)
            if c.split("data: ", 1)[1].strip() != "[DONE]"
        ]
        arguments = "".join(
            tc.get("function", {}).get("arguments", "")
            for p in payloads
            for tc in (
                (p.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []
            )
        )
        assert '"city"' in arguments, f"arguments lost at max_items={max_items}"

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_and_the_finish_reason_says_a_tool_was_called(self, max_items):
        state = self._state()
        event = {"index": 0, "text": self.CALL, "token_ids": [1], "finished": True}
        self._drain_all(state, event, max_items)
        assert state.has_tool_calls == [True]

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_the_stream_still_closes_at_any_batch_size(self, max_items):
        """Queueing the overflow made the early return the common path, and
        it returned before the final chunks -- a fully drained stream with no
        `finish_reason`, no usage and no `[DONE]`."""
        state = self._state()
        event = {"index": 0, "text": self.CALL, "token_ids": [1], "finished": True}
        chunks = self._drain_all(state, event, max_items)
        assert chunks[-1].split("data: ", 1)[1].strip() == "[DONE]"
        reasons = [
            c["finish_reason"]
            for raw in chunks
            if raw.split("data: ", 1)[1].strip() != "[DONE]"
            for c in json.loads(raw.split("data: ", 1)[1]).get("choices", [])
            if c.get("finish_reason")
        ]
        assert reasons == ["tool_calls"], reasons
