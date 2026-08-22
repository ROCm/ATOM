# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SSE frame encoding for the streaming endpoints.

One frame is encoded per streamed chunk per request, so at high concurrency
this is one of the fixed per-token costs that caps the API server before the
GPU does. ``msgspec`` with a reusable encoder measures 5.8x faster than
``json.dumps`` on a chat delta (0.25 vs 1.47 us/chunk; see
``/app/logs_claude/bench_sse_encode.py``).

Encoding straight to ``bytes`` and letting the ASGI layer write them is another
10% on top, but the frame builders are shared with the atomesh service, which
assembles ``list[str]`` batches -- not worth splitting them for 1.6% of the
original cost.

Unlike ``json.dumps``, msgspec rejects NaN and Infinity rather than emitting
the non-standard literals. Any float that can reach a frame (logprobs, for
instance) has to be sanitised before it gets here.
"""

from typing import Any

import msgspec

_encoder = msgspec.json.Encoder()


def data_frame(payload: Any) -> str:
    """Encode one anonymous ``data:`` SSE frame."""
    return f"data: {_encoder.encode(payload).decode()}\n\n"


def event_frame(event: str, payload: Any) -> str:
    """Encode one named-event SSE frame."""
    return f"event: {event}\ndata: {_encoder.encode(payload).decode()}\n\n"
