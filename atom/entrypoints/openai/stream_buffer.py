# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Look-ahead helper shared by the streaming reasoning and tool-call parsers.

Both parsers watch the token stream for multi-character markers (``<think>``,
``<minimax:tool_call>``, ...) that can straddle a chunk boundary, so they must
hold back a tail that might still grow into a marker while releasing everything
before it.
"""

from typing import Sequence


def releasable_prefix_len(buf: str, markers: Sequence[str]) -> int:
    """How many leading characters of ``buf`` cannot be part of a marker.

    Returns the offset of the earliest position that either starts a complete
    marker or could still grow into one once more text arrives; text before it
    is safe to emit now. Without this look-ahead a parser has to choose between
    emitting a partial marker and stalling the whole stream on the first ``<``.
    """
    starts = [i for i in (buf.find(marker) for marker in markers) if i != -1]
    if starts:
        return min(starts)
    longest = max((len(marker) for marker in markers), default=0)
    for start in range(max(0, len(buf) - longest + 1), len(buf)):
        tail = buf[start:]
        if any(marker.startswith(tail) for marker in markers):
            return start
    return len(buf)
