# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The one rule for how much of a stream can be released.

Everything here is about the boundary between "this could still become a
marker" and "this never will". The properties the callers rely on are stated
as properties, because the failures they replace were all of the form "correct
on the inputs someone thought of".
"""

from __future__ import annotations

import random

import pytest

from atom.entrypoints.openai.marker_scanner import (
    MarkerScanner,
    Scan,
    partial_suffix_len,
)

TOOL = ("<tool_call>", "<function=")
THINK = ("<think>", "</think>")


def run(markers, chunks) -> tuple[str, tuple[str, ...]]:
    """Drive a scanner and return (everything released, the markers hit).

    Deliberately not `Scan.rest`. That is how much text happened to be in hand
    when the marker completed -- everything after it in the same chunk -- so it
    is larger for coarse chunkings and empty for one-character ones. It is a
    buffer handoff, not something a client can observe, and putting it in the
    key would make chunk-invariance fail on a scanner that is behaving. What
    must not vary is the text released and which markers fired.
    """
    s = MarkerScanner(markers)
    out, hits = "", []
    for c in chunks:
        pending = c
        while True:
            scan = s.feed(pending)
            out += scan.released
            if not scan.hit:
                break
            # `feed` stops at the first marker and hands the rest back: a hit
            # is a state change, and where the text after it goes is the
            # caller's call, not the scanner's. This caller keeps scanning, so
            # it feeds the rest back -- without which a second marker in the
            # same chunk is never seen, and the result depends on whether the
            # chunking happened to put the two markers together.
            hits.append(scan.hit)
            pending = scan.rest
    return out + s.flush(), tuple(hits)


class TestPartialSuffixLen:
    def test_a_marker_character_that_is_not_at_the_end_holds_nothing(self):
        """The bug this replaces, stated as a test.

        `if (a < b)` contains a '<' and can never become `<tool_call>`,
        because the '<' is not where the next character would extend it.
        """
        assert partial_suffix_len("if (a < b) { return a; }", "<tool_call>") == 0

    @pytest.mark.parametrize(
        "tail, want", [("<", 1), ("<t", 2), ("<tool_", 6), ("<tool_call", 10)]
    )
    def test_a_growing_prefix_is_measured(self, tail, want):
        assert partial_suffix_len("some answer " + tail, "<tool_call>") == want

    def test_a_complete_marker_is_not_a_partial_one(self):
        """Capped at len-1: completeness is the caller's separate question."""
        assert partial_suffix_len("x<tool_call>", "<tool_call>") == 0

    def test_the_longest_overlap_wins(self):
        """`<<think>` ends with both '<' and '<<'; the short answer leaks."""
        assert partial_suffix_len("x<<", "<<think>") == 2


class TestBoundedWithhold:
    def test_a_stray_marker_character_does_not_hold_the_answer(self):
        text = "Here is the fix: if (a < b) { return a; } and that is all."
        out, hits = run(TOOL, [text[i : i + 4] for i in range(0, len(text), 4)])
        assert out == text and not hits

    def test_the_held_buffer_never_exceeds_the_longest_marker(self):
        """The invariant that makes a stall unrepresentable rather than tested."""
        s = MarkerScanner(TOOL)
        longest = max(len(m) for m in TOOL)
        text = ("< <t <to <tool <tool_c prose " * 40) + "<tool_"
        for i in range(0, len(text), 3):
            s.feed(text[i : i + 3])
            assert len(s.held) < longest

    def test_text_is_released_before_the_marker_that_follows_it(self):
        out, hits = run(TOOL, ["Sure. ", "<tool", "_call>", "{}"])
        assert out == "Sure. {}"
        assert hits == ("<tool_call>",)


# The shapes every property below is checked against: no marker, a stray
# marker character, a marker in the middle, a marker pair, a dangling partial,
# two markers running together, and nothing at all.
CASES = [
    "plain text with no markers at all",
    "a < b and c < d, all prose",
    "before <tool_call> after",
    "<think>reasoning about a < b</think>the answer",
    "trailing partial <func",
    "<tool_call><tool_call> twice",
    "",
]


class TestChunkInvariance:
    @pytest.mark.parametrize("text", CASES)
    @pytest.mark.parametrize("markers", [TOOL, THINK, TOOL + THINK])
    def test_where_the_boundaries_fall_changes_nothing(self, text, markers):
        rng = random.Random(7)
        ways = [[text]] + [
            [text[i : i + n] for i in range(0, len(text), n)] for n in (1, 2, 3, 5)
        ]
        for _ in range(4):
            parts, i = [], 0
            while i < len(text):
                n = rng.randint(1, 9)
                parts.append(text[i : i + n])
                i += n
            ways.append(parts)
        results = {run(markers, w) for w in ways}
        assert len(results) == 1, f"{len(results)} different results: {results}"


class TestConservation:
    @pytest.mark.parametrize("text", CASES)
    def test_every_byte_comes_back(self, text):
        """Released text plus the markers hit must reconstitute the input."""
        s = MarkerScanner(TOOL + THINK)
        rebuilt = ""
        for i in range(0, len(text), 2):
            scan = s.feed(text[i : i + 2])
            rebuilt += scan.released + (scan.hit or "") + scan.rest
        assert rebuilt + s.flush() == text


class TestMarkerPrecedence:
    def test_the_rest_is_handed_back_not_kept(self):
        """A hit is a state change, so what follows is the caller's to route."""
        scan = MarkerScanner(TOOL).feed('Sure. <tool_call>{"a": 1}')
        assert scan == Scan("Sure. ", "<tool_call>", '{"a": 1}')

    def test_the_earliest_marker_wins(self):
        scan = MarkerScanner(TOOL).feed("a <function=x b <tool_call>")
        assert scan.hit == "<function=" and scan.released == "a "

    def test_at_one_position_the_longest_wins(self):
        """Otherwise a marker that prefixes another truncates it."""
        scan = MarkerScanner(("<think>", "<thinking>")).feed("x<thinking>y")
        assert scan.hit == "<thinking>" and scan.rest == "y"


class TestConstruction:
    @pytest.mark.parametrize("bad", [(), ("",), ("<a>", "")])
    def test_an_empty_marker_is_refused(self, bad):
        """It would match everywhere and hold nothing; fail where it is set."""
        with pytest.raises(ValueError):
            MarkerScanner(bad)

    def test_duplicates_do_not_change_the_answer(self):
        a = MarkerScanner(TOOL).feed("x<tool_call>y")
        b = MarkerScanner(TOOL + ("<tool_call>",)).feed("x<tool_call>y")
        assert a == b == Scan("x", "<tool_call>", "y")
