# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the ATOM standalone service's streaming chat state."""

import json
import queue

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
