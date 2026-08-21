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
scope. It rejects the common chunk, which carries no marker character at all,
before any per-marker work happens.

Both halves run once per token per stream, so how they are spelled shows up in
event-loop time rather than in a profile of the model. Two spellings that read
as equivalent are not: `frozenset.intersection(buf)` hashes every character of
`buf` (2.9 us on 900 chars) where `any(c in buf ...)` is a C substring search
per marker character (0.12 us, 25x), and the buffer is not bounded by a marker
on the way in -- callers concatenate a backlog into `text`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass


def partial_suffix_len(text: str, marker: str) -> int:
    """Length of the longest proper prefix of `marker` that ends `text`.

    `("if (a < b", "<tool_call>")` is 0 -- the '<' is not at the end, so
    nothing here can still grow into the marker. `("... <tool_", ...)` is 6.
    Bounded by `len(marker) - 1`: a whole marker is not a partial one, and its
    caller has already looked for complete ones.

    Kept for a single marker asked about in isolation. :class:`MarkerScanner`
    does not call it: over a set of markers it re-slices the same suffixes once
    per marker, which is the work `_prefixes_by_len` precomputes away.
    """
    for k in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:k]):
            return k
    return 0


def _plan_for(markers) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """:func:`_plan`, keyed on the marker *set* rather than how it was spelled.

    The cache key is the argument, so without this the same markers in a
    different order are a second entry holding an equal copy -- and the two
    declaration sites for one format (its own `START_MARKERS` and a dialect's
    tuple) do not have to agree on order for that to happen.
    """
    return _plan(tuple(sorted(set(markers))))


@functools.lru_cache(maxsize=64)
def _plan(markers: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """Everything about a marker set that does not depend on the stream.

    Cached on the set, not computed per scanner: a scanner is built per
    request and the sets are class constants -- one per tool-call format plus
    the reasoning dialects' -- so this runs a handful of times per process.
    Building it per scanner cost 8.7 us of request setup to save 4.5 us per
    chunk, which is the right trade only for streams longer than two chunks.

    Returns the markers longest-first (so a marker that is a prefix of another
    never wins a tie at the same position and truncates the longer one), their
    distinct first characters, and every proper prefix grouped by length.
    """
    ordered = tuple(sorted(set(markers), key=len, reverse=True))
    firsts = tuple(sorted({m[0] for m in ordered}))
    by_len: dict[int, set[str]] = {}
    for m in ordered:
        for k in range(1, len(m)):
            by_len.setdefault(k, set()).add(m[:k])
    return ordered, firsts, {k: frozenset(v) for k, v in by_len.items()}


def held_suffix_len(text: str, markers: tuple[str, ...]) -> int:
    """Longest suffix of `text` that is a proper prefix of any of `markers`.

    The stateless form of what :class:`MarkerScanner` withholds, for callers
    that own their own buffer. Same cached plan, so they get the same answer
    and the same first-character reject rather than a second implementation.
    """
    if not markers:
        return 0
    ordered, firsts, by_len = _plan_for(markers)
    for k in range(min(len(ordered[0]) - 1, len(text)), 0, -1):
        if text[-k] in firsts and text[-k:] in by_len[k]:
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

    Stateful across chunks and cheap to hold: what it *withholds* is bounded by
    the longest marker, which is what makes the withhold bounded and the cost
    per chunk independent of how long the response runs. What it *scans* is
    that tail plus the incoming chunk, which the caller sizes.
    """

    def __init__(self, markers: tuple[str, ...]):
        if not markers or any(not m for m in markers):
            raise ValueError("a scanner needs at least one non-empty marker")
        self._markers, self._firsts, self._prefixes_by_len = _plan_for(markers)
        self._longest = len(self._markers[0])
        self._buf = ""

    @property
    def held(self) -> str:
        """What is being withheld right now. Bounded by the longest marker."""
        return self._buf

    def feed(self, text: str) -> Scan:
        buf = self._buf + text
        if not any(c in buf for c in self._firsts):
            # Nothing here can begin a marker, so nothing needs holding: the
            # fast path, not a shortcut past correctness. `in` on a str is a
            # C substring search; a set intersection would hash every
            # character of `buf` instead, which is 25x more for the same
            # answer and is paid once per token per stream.
            self._buf = ""
            return Scan(buf)

        at, hit = self._earliest_complete(buf)
        if hit is not None:
            self._buf = ""
            return Scan(buf[:at], hit, buf[at + len(hit) :])

        cut = len(buf) - self._held_suffix_len(buf)
        self._buf = buf[cut:]
        # The invariant the whole class exists for: a stall is not something
        # to test for here, it is something that cannot be represented.
        assert len(self._buf) < self._longest, "withheld more than a marker could be"
        return Scan(buf[:cut])

    def flush(self) -> str:
        """Release the held tail at end of stream; it never became a marker."""
        out, self._buf = self._buf, ""
        return out

    def _held_suffix_len(self, buf: str) -> int:
        """Longest suffix of `buf` that is a proper prefix of some marker.

        Longest first, so the first hit is the answer. `buf[-k]` is that
        suffix's first character, and a marker's first characters are few, so
        the check rejects most lengths before the slice is even taken.
        """
        for k in range(min(self._longest - 1, len(buf)), 0, -1):
            if buf[-k] in self._firsts and buf[-k:] in self._prefixes_by_len[k]:
                return k
        return 0

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
