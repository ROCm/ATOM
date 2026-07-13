# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers for the streaming SSE hot path.

Why this module exists
----------------------
Streamed tokens travel:

    EngineCore (separate proc)
      --ZMQ--> engine_core_mgr output thread   <-- ONE thread, drains the socket
        --callback--> stream_queue (asyncio)
          --> per-request async consumer (serving_chat / serving_completion)
            --> SSE bytes to the client

The engine_core_mgr output thread is the *only* thread draining the IPC
socket. Historically the per-request callback ran ``tokenizer.decode(...)``
right there, so detokenization for every concurrent stream was serialized on
that one thread (holding the GIL). When it fell behind, the ZMQ socket stopped
draining, back-pressuring the EngineCore's send and delaying delivery for all
streams — visible as high TTFT-to-second-token / inter-chunk latency even
though backend token generation was healthy.

The two helpers below keep the callback trivial (enqueue token ids only) and
move decode + SSE assembly into the per-request asyncio consumer, batched over
whatever is already queued.
"""

import asyncio


def get_chunk_text(chunk_data, tokenizer):
    """Return the delta text for one (possibly coalesced) stream chunk.

    If a pre-decoded ``text`` is present it is used as-is (back-compat with any
    producer that still decodes upstream); otherwise the delta ``token_ids``
    are decoded here, in the asyncio consumer — never on the engine output
    callback thread.
    """
    if "text" in chunk_data:
        return chunk_data["text"]
    token_ids = chunk_data.get("token_ids") or []
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def coalesce_ready_chunks(first_chunk, stream_queue):
    """Merge ``first_chunk`` with every chunk already sitting in ``stream_queue``.

    The engine emits one stream message per decode step (one MTP step yields a
    burst of accepted tokens). By the time the consumer wakes up, several
    steps' worth of messages may already be queued. Draining them
    non-blockingly and decoding/parsing/formatting the batch ONCE — instead of
    once per engine message — collapses the per-message Python overhead
    (tokenizer.decode + reasoning/tool parsing + json.dumps) on the SSE
    delivery path.

    Merge semantics:
      * ``token_ids``      concatenated in arrival order (delta order kept)
      * ``finished`` /
        ``finish_reason``  taken from the terminal message (nothing follows it)
      * ``num_cached_tokens`` last non-zero value seen
      * ``kv_transfer_params`` carried through if present
      * ``started_at``     kept from ``first_chunk``

    Stops as soon as a finished chunk is drained, or the queue is momentarily
    empty. Only single-stream (n == 1) consumers use this; the n>1 fan-out
    consumers interleave siblings on one queue and decode per message instead.
    """
    merged = dict(first_chunk)
    token_ids = list(first_chunk.get("token_ids") or [])

    while not merged.get("finished", False):
        try:
            nxt = stream_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        token_ids.extend(nxt.get("token_ids") or [])
        merged["finished"] = nxt.get("finished", False)
        merged["finish_reason"] = nxt.get("finish_reason")
        _ct = nxt.get("num_cached_tokens", 0)
        if _ct:
            merged["num_cached_tokens"] = _ct
        if "kv_transfer_params" in nxt:
            merged["kv_transfer_params"] = nxt["kv_transfer_params"]

    merged["token_ids"] = token_ids
    # Drop any stale pre-decoded text; the consumer decodes the merged ids.
    merged.pop("text", None)
    return merged
