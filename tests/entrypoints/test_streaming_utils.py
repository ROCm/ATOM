# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the streaming SSE hot-path helpers."""

import asyncio

from atom.entrypoints.openai.streaming_utils import (
    coalesce_ready_chunks,
    get_chunk_text,
)


class _FakeTokenizer:
    """Decodes ints to '<id>' joined by '|' so token→text mapping is visible."""

    def decode(self, token_ids, skip_special_tokens=True):
        return "|".join(str(t) for t in token_ids)


def test_get_chunk_text_prefers_existing_text():
    tok = _FakeTokenizer()
    assert get_chunk_text({"text": "hello", "token_ids": [1, 2]}, tok) == "hello"


def test_get_chunk_text_decodes_token_ids():
    tok = _FakeTokenizer()
    assert get_chunk_text({"token_ids": [7, 8, 9]}, tok) == "7|8|9"


def test_get_chunk_text_empty():
    tok = _FakeTokenizer()
    assert get_chunk_text({"token_ids": []}, tok) == ""
    assert get_chunk_text({}, tok) == ""


def test_coalesce_single_chunk_empty_queue():
    q: asyncio.Queue = asyncio.Queue()
    merged = coalesce_ready_chunks({"token_ids": [1, 2], "finished": False}, q)
    assert merged["token_ids"] == [1, 2]
    assert merged["finished"] is False
    assert "text" not in merged  # stale text is dropped


def test_coalesce_merges_backlog_and_stops_on_finished():
    q: asyncio.Queue = asyncio.Queue()
    # Two more steps already queued; the last one finishes the request.
    q.put_nowait({"token_ids": [3], "finished": False})
    q.put_nowait(
        {
            "token_ids": [4, 5],
            "finished": True,
            "finish_reason": "stop",
            "num_cached_tokens": 111,
            "kv_transfer_params": {"k": "v"},
        }
    )
    # An extra message after 'finished' must NOT be consumed.
    q.put_nowait({"token_ids": [6], "finished": False})

    merged = coalesce_ready_chunks({"token_ids": [1, 2], "finished": False}, q)

    assert merged["token_ids"] == [1, 2, 3, 4, 5]  # delta order preserved
    assert merged["finished"] is True
    assert merged["finish_reason"] == "stop"
    assert merged["num_cached_tokens"] == 111
    assert merged["kv_transfer_params"] == {"k": "v"}
    # The post-finish message is left in the queue untouched.
    assert q.get_nowait()["token_ids"] == [6]


def test_coalesce_stops_when_queue_drains():
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait({"token_ids": [3], "finished": False})
    merged = coalesce_ready_chunks({"token_ids": [1, 2], "finished": False}, q)
    # Drained both ready chunks, still not finished.
    assert merged["token_ids"] == [1, 2, 3]
    assert merged["finished"] is False


def test_coalesce_then_decode_matches_per_chunk_decode():
    """Decoding the coalesced batch equals decoding each delta and concatenating
    (true for this fake tokenizer; documents the intended equivalence)."""
    tok = _FakeTokenizer()
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait({"token_ids": [3, 4], "finished": True, "finish_reason": "stop"})
    merged = coalesce_ready_chunks({"token_ids": [1, 2], "finished": False}, q)
    assert get_chunk_text(merged, tok) == "1|2|3|4"
