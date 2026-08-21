# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How much of a stream can be released without splitting a marker.

One question, one answer. Everything on the streaming path that has to notice
a literal in the model's output -- the reasoning channel's delimiters, the
tool-call formats' opening tags -- asks it, and asking it in more than one
place is how the four incompatible answers this replaces came about.

The rule: release everything except the longest *suffix* of the buffer that is
a prefix of some marker. Not "hold everything once a marker's first character
appears anywhere", which is the shape being retired: it withholds on a '<' in
the middle of an answer that will never become a tag, and the buffer it holds
grows without bound, so the scan is O(n) per chunk and O(n^2) over a response.
Measured at 64 KB of answer that cost 515 ms of pure host CPU against 6 ms
here, and the first byte of a '<'-bearing answer never reached the client
until the stream ended.

The `'<'` test is still the fast path -- it was the right idea at the wrong
scope. Applied to a buffer that can never exceed the longest marker it is a
constant-time reject, and the common chunk carries no marker character at all.
"""

from __future__ import annotations

from dataclasses import dataclass


def partial_suffix_len(text: str, marker: str) -> int:
    """Length of the longest proper prefix of `marker` that ends `text`.

    `("if (a < b", "<tool_call>")` is 0 -- the '<' is not at the end, so
    nothing here can still grow into the marker. `("... <tool_", ...)` is 6.
    Bounded by `len(marker) - 1`: a whole marker is not a partial one, and its
    caller has already looked for complete ones.
    """
    for k in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:k]):
            return k
    return 0


@dataclass(frozen=True)
class Scan:
    """What one chunk produced.

    `released` is safe to send now. `hit` is the marker that completed, if one
    did, and `rest` is everything after it -- handed back rather than kept,
    because who owns the text after a marker is the caller's decision and not
    this class's.
    """

    released: str
    hit: str | None = None
    rest: str = ""


class MarkerScanner:
    """Incremental reader over a stream that must not split a marker.

    Stateful across chunks and cheap to hold: the buffer is bounded by the
    longest marker, which is what makes the withhold bounded and the cost per
    chunk independent of how long the response runs.
    """

    def __init__(self, markers: tuple[str, ...]):
        if not markers or any(not m for m in markers):
            raise ValueError("a scanner needs at least one non-empty marker")
        # Longest first, so a marker that is a prefix of another never wins the
        # tie at the same position and truncate the longer one.
        self._markers = tuple(sorted(set(markers), key=len, reverse=True))
        self._longest = len(self._markers[0])
        self._firsts = frozenset(m[0] for m in self._markers)
        self._buf = ""

    @property
    def held(self) -> str:
        """What is being withheld right now. Bounded by the longest marker."""
        return self._buf

    def feed(self, text: str) -> Scan:
        buf = self._buf + text
        if not self._firsts.intersection(buf):
            # Nothing here can begin a marker. The buffer is already bounded,
            # so this scan is over at most one marker's worth of tail plus the
            # chunk -- it is the fast path, not a shortcut past correctness.
            self._buf = ""
            return Scan(buf)

        at, hit = self._earliest_complete(buf)
        if hit is not None:
            self._buf = ""
            return Scan(buf[:at], hit, buf[at + len(hit) :])

        hold = max(partial_suffix_len(buf, m) for m in self._markers)
        cut = len(buf) - hold
        self._buf = buf[cut:]
        # The invariant the whole class exists for: a stall is not something
        # to test for here, it is something that cannot be represented.
        assert len(self._buf) < self._longest, "withheld more than a marker could be"
        return Scan(buf[:cut])

    def flush(self) -> str:
        """Release the held tail at end of stream; it never became a marker."""
        out, self._buf = self._buf, ""
        return out

    def _earliest_complete(self, buf: str) -> tuple[int, str | None]:
        """Where the first complete marker starts, and which one it is.

        Earliest wins, and at the same position the longest does -- `<think>`
        must not be reported where `<thinking>` was meant when both are
        registered.
        """
        best_at, best = len(buf), None
        for m in self._markers:  # already longest-first
            at = buf.find(m)
            if 0 <= at < best_at:
                best_at, best = at, m
        return best_at, best
