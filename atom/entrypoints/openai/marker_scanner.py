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

Judge a change here on a whole response through the real pipeline -- reasoning
filter into tool parser, four-character tokens -- and not on one function.
Alternating arms, three rounds each: 1579 -> 1383 ns/token on plain prose and
1652 -> 1537 on an answer ending in a tool call. A single cross-process pair
said the opposite before the arms were alternated.
"""

from __future__ import annotations

import functools
import re
from typing import NamedTuple


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


@functools.lru_cache(maxsize=64)
def _plan_for(markers) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """:func:`_plan`, keyed on the marker *set* rather than how it was spelled.

    Two caches, and both earn their place. `_plan` is keyed on the normalised
    set, so two spellings of one set share a single plan *object*. This one is
    keyed on the spelling, so the callers that ask once per token -- always
    with the same module-level constant -- skip the normalisation as well:
    `sorted(set(...))` on every call was 148 ns of the 193 it took to answer.
    """
    return _plan(tuple(sorted(set(markers))))


def _suffix_len(text: str, ordered: tuple, firsts: tuple, by_len: dict) -> int:
    """Longest suffix of `text` that is a proper prefix of one of the markers.

    The one loop. It was written twice -- once here for callers holding their
    own buffer and once as a method on the scanner -- in a module whose whole
    thesis is that asking this question in more than one place is how the
    answers drift apart.

    Nothing can be held unless the tail *starts* a marker, so a marker's first
    character has to appear within the last `longest - 1` bytes. One substring
    search over that window answers that, and it is a bet rather than a free
    win: 2.4x when it rejects (737 ns -> 307 ns) against 1.3x slower when it
    does not (847 ns -> 1108 ns), because the loop then repeats the search it
    just paid for. A chunk whose last twenty bytes contain a marker's opening
    character is the rare one, and this runs once per token on every stream
    including models whose markers never appear at all.
    """
    longest = len(ordered[0])
    if longest == 1:
        return 0  # a one-character marker has no proper prefix
    if not any(c in text[-(longest - 1) :] for c in firsts):
        return 0
    for k in range(min(longest - 1, len(text)), 0, -1):
        if text[-k] in firsts and text[-k:] in by_len[k]:
            return k
    return 0


@functools.lru_cache(maxsize=64)
def _plan(markers: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """Everything about a marker set that does not depend on the stream.

    Cached on the set, not computed per scanner: a scanner is built per
    request and the sets are class constants -- one per tool-call format plus
    the reasoning dialects' -- so this runs a handful of times per process.
    Building it per scanner cost 8.7 us of request setup to save 4.5 us per
    chunk, which is the right trade only for streams longer than two chunks.

    Returns the markers longest-first, their distinct first characters, and
    every proper prefix grouped by length.

    Longest-first settles a tie only between markers that are *both already
    complete in the buffer*. A chunk ending exactly at the shorter of a
    prefix pair reports the shorter one, because the longer one has not
    arrived to be preferred -- so which of the two fires is a function of
    where the chunk boundary landed, not of the text. `MarkerScanner.feed`
    says the same thing from the other side.

    Nothing here withholds a complete match on the chance it could still
    grow; that would be the stronger rule, and it would cost every marker
    that is a prefix of another a wait of the length difference. It is not
    needed while no format has a prefix pair whose halves disagree about
    handing the stream over -- which is what
    `TestAPrefixPairCannotChangeTheHandover` holds them to. The moment one
    does, this becomes "the region opened, or did not, depending on
    chunking", and the stronger rule has to be written.
    """
    ordered = tuple(sorted(set(markers), key=len, reverse=True))
    firsts = tuple(sorted({m[0] for m in ordered}))
    by_len: dict[int, set[str]] = {}
    for m in ordered:
        for k in range(1, len(m)):
            by_len.setdefault(k, set()).add(m[:k])
    return ordered, firsts, {k: frozenset(v) for k, v in by_len.items()}


@functools.lru_cache(maxsize=64)
def _pattern_for(ordered: tuple[str, ...]) -> re.Pattern:
    """One alternation of the markers, longest first.

    A search finds the leftmost position where any marker starts, and at
    that position tries the alternatives in order, so the longest one wins:
    the same answer as a `find` per marker, in one C call per hit.
    """
    return re.compile("|".join(re.escape(m) for m in ordered))


def held_suffix_len(text: str, markers: tuple[str, ...]) -> int:
    """Longest suffix of `text` that is a proper prefix of any of `markers`.

    The stateless form of what :class:`MarkerScanner` withholds, for callers
    that own their own buffer. Same cached plan and the same loop, so they get
    the same answer rather than a second implementation of the rule.
    """
    if not markers:
        return 0
    return _suffix_len(text, *_plan_for(markers))


class Scan(NamedTuple):
    """What one step of the scan produced.

    `released` is safe to send now. `hit` is the marker that completed, if one
    did. The text after it stays in the scanner: `feed("")` goes on reading
    it, and `take_rest` hands it to a caller that routes it elsewhere.
    """

    released: str
    hit: str | None = None


class MarkerScanner:
    """Incremental reader over a stream that must not split a marker.

    Stateful across chunks and cheap to hold: what it *withholds* is bounded by
    the longest marker, which is what makes the withhold bounded and the cost
    per chunk independent of how long the response runs.

    Linear in what it is fed, however many markers that carries. It keeps one
    buffer and a cursor, and each search runs from the cursor to the next
    marker, so a hit neither copies the text after it nor searches past the
    next one. Handing the remainder back on every hit, and searching all of
    it for every marker, made one large feed quadratic in its marker count:
    31.7 s for an 819K-character output with 32,768 markers.
    """

    def __init__(self, markers: tuple[str, ...]):
        if not markers or any(not m for m in markers):
            raise ValueError("a scanner needs at least one non-empty marker")
        self._plan = _plan_for(markers)
        self._markers, self._firsts, self._prefixes_by_len = self._plan
        self._longest = len(self._markers[0])
        self._pattern = _pattern_for(self._markers)
        self._buf = ""
        self._pos = 0  # where the unread text starts in `_buf`

    @property
    def held(self) -> str:
        """What is in hand and not yet released. After a scan that found no
        marker, this is bounded by the longest marker."""
        return self._buf[self._pos :]

    def feed(self, text: str) -> Scan:
        """Add `text` and read up to the first complete marker, if any."""
        if text:
            if self._pos >= len(self._buf) and not any(c in text for c in self._firsts):
                # Nothing in hand and nothing here can begin a marker, so
                # nothing needs holding: the fast path, not a shortcut past
                # correctness. `in` on a str is a C substring search; a set
                # intersection would hash every character instead, which is
                # 25x more for the same answer and is paid once per token per
                # stream.
                self._reset("")
                return Scan(text)
            self._reset(self._buf[self._pos :] + text)

        buf, pos = self._buf, self._pos
        # Earliest wins, and at the same position the longest does -- `<think>`
        # must not be reported where `<thinking>` was meant when both are
        # registered. Only among those already complete in `buf`, though: see
        # `_plan` for why that is weaker than it sounds, and what holds the
        # registered formats inside the gap.
        found = self._pattern.search(buf, pos)
        if found is not None:
            self._pos = found.end()
            return Scan(buf[pos : found.start()], buf[found.start() : found.end()])

        start = max(pos, len(buf) - self._longest + 1)
        cut = len(buf) - _suffix_len(buf[start:], *self._plan)
        released = buf[pos:cut]
        self._reset(buf[cut:])
        # The invariant the whole class exists for: a stall is not something
        # to test for here, it is something that cannot be represented.
        assert len(self._buf) < self._longest, "withheld more than a marker could be"
        return Scan(released)

    def take_rest(self) -> str:
        """Everything in hand and not yet released, handed over and forgotten:
        for a caller that routes the text after a marker somewhere else."""
        rest = self._buf[self._pos :]
        self._reset("")
        return rest

    def flush(self) -> str:
        """Release the held tail at end of stream; it never became a marker."""
        return self.take_rest()

    def _reset(self, buf: str) -> None:
        self._buf, self._pos = buf, 0
