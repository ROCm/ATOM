# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Properties the streaming text pipeline must hold for every format it knows.

Not a list of cases. The corpus is *generated* from the two production
registries -- ``reasoning_dialects.DIALECTS`` and the tool-parser
``_DETECT_ORDER`` -- crossed with a handful of text shapes built out of each
entry's own declared markers. Registering a new model family or a new
tool-call format therefore adds coverage by itself, and
``test_every_registered_parser_declares_its_markers`` is what stops a new
entry from joining the registry without the declaration the generation needs.

Two properties, and they are not nested:

`chunk-invariance` -- the same text split differently must produce the same
    reasoning, the same content, the same tool events and the same finish
    reason. Catches markers split across boundaries, buffers carried between
    states, and a sniffer latching onto whatever it happened to see first.

`bounded withhold` -- text must not be held back longer than a marker could
    justify. Judged against the same text with its trigger characters
    neutralised, because an absolute budget cannot tell a stall from the
    reasoning channel legitimately withholding until its end marker arrives.

The stall is *chunk-invariant* -- everything comes out at flush no matter how
the input was split -- so the first property cannot see it and the second is
not redundant. Both are needed.

The pipeline modelled is `serving_chat.py:280-333`: reasoning filter first,
its content segments into the tool parser, both flushed on the last chunk.
"""

from __future__ import annotations

import ast
import itertools
import json
import pathlib
import random
import re
import sys
import time
from typing import ClassVar

import pytest

from atom.entrypoints.openai.reasoning import ReasoningFilter, separate_reasoning
from atom.entrypoints.openai.reasoning_dialects import DIALECTS, resolve_dialect
from atom.entrypoints.openai.serving_anthropic import completes_a_tool_call
from atom.entrypoints.openai.tool_parser import RegionParse, ToolCall, parse_tool_calls
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser
from atom.entrypoints.openai.tool_parser.registry import _DETECT_ORDER
from atom.entrypoints.openai.tool_parser.stream import (
    _PEEK_WINDOW,
    ToolCallStreamParser,
)

# Kimi is the terminal fallback and so is not in the detect order, but it is a
# registered format with markers of its own.
ALL_PARSERS = (*_DETECT_ORDER, KimiParser)

# Slack allowed on top of the neutralised control before a hold counts as a
# stall. One marker's worth, since that is the most a correct rule can need.
SLACK = 40

PROSE = "The comparison was inverted, so the branch never ran. "


def dialect_markers(dialect) -> tuple[str, ...]:
    return tuple(
        m
        for m in (
            dialect.prompt_open_marker,
            dialect.output_open_marker,
            dialect.think_end_marker,
        )
        if m
    )


class Seen:
    """Only what a client could observe."""

    def __init__(self):
        self.reasoning = ""
        self.content = ""
        self.events: list[str] = []
        self.first_content_at: int | None = None

    @property
    def key(self):
        finish = "tool_calls" if "tool_call_start" in self.events else "stop"
        return (self.reasoning, self.content, tuple(self.events), finish)


def drive(
    text: str, chunks: list[str], parser=None, *, suppress_calls: bool = False
) -> Seen:
    """Replay the serving loop over one chunking.

    `parser` is what the server resolved from the chat template at startup, so
    each case reads its own format explicitly rather than relying on the shape
    of the text to select one. `suppress_calls` is `tool_choice: "none"`,
    which is a property of the request and not of the text, so it has to be
    passed in the same way.
    """
    rf, tp = ReasoningFilter(), ToolCallStreamParser(
        parser_cls=parser, suppress_calls=suppress_calls
    )
    seen = Seen()
    consumed = 0

    def take(kind: str, payload: str) -> None:
        if kind == "reasoning_content":
            seen.reasoning += payload
        elif kind == "content":
            if payload and seen.first_content_at is None:
                seen.first_content_at = consumed
            seen.content += payload
        else:
            seen.events.append(kind)

    for i, chunk in enumerate(chunks):
        consumed += len(chunk)
        last = i == len(chunks) - 1
        segments = rf.process(chunk)
        if last:
            segments.extend(rf.flush())
        for field, seg in segments:
            if field == "reasoning_content":
                take(field, seg)
            else:
                for kind, data in tp.process(seg):
                    take(kind, data)
        if last:
            for kind, data in tp.flush():
                take(kind, data)
    return seen


def split_every_way(text: str) -> dict[str, list[str]]:
    """One-shot, several fixed strides, and seeded random splits."""
    ways = {"one-shot": [text]}
    for n in (1, 2, 3, 7, 64):
        ways[f"fixed-{n}"] = [text[i : i + n] for i in range(0, len(text), n)] or [""]
    rng = random.Random(1234)
    for r in range(2):
        parts, i = [], 0
        while i < len(text):
            n = rng.randint(1, 17)
            parts.append(text[i : i + n])
            i += n
        ways[f"random-{r}"] = parts or [""]
    return ways


def trigger_chars(*marker_groups) -> set[str]:
    """The characters that can open any of these markers."""
    return {m[0] for group in marker_groups for m in group if m}


def defuse(text: str, triggers: set[str]) -> str:
    """The same text with nothing in it that could begin a marker."""
    for ch in triggers:
        text = text.replace(ch, "‹")
    return text


def shapes(dialect, parser) -> dict[str, str]:
    """Text shapes built from this pair's own markers.

    Each is a sentence a model could plausibly emit and none of them is a tool
    call: the point is text that merely *looks like* it might start one.
    """
    marks = parser.START_MARKERS
    end = dialect.think_end_marker
    # Every trigger character, dropped into ordinary prose without ever
    # completing a marker -- `if (a < b)` and its equivalent for every format.
    seeded = "".join(f"a {ch} b holds. " for ch in sorted(trigger_chars(marks)))
    out = {
        "trigger chars in ordinary prose": f"Here is the fix: {seeded}" + PROSE * 6,
        # A whole marker, mid-sentence, quoted rather than used.
        "a marker quoted inside the answer": (
            f"The model writes {marks[0]} to open a call. " + PROSE * 4
        ),
        # The shape where reasoning mentions a tool and then declines to use it.
        "a tool marker inside reasoning": (
            f"I could call {marks[0]} but I will answer directly. "
            + "Hmm. " * 8
            + end
            + "It is sunny in Paris."
        ),
    }

    # Every marker the format has, not just the first: Kimi-K3's parse keys on
    # its tools token while its first marker is the call prefix, so quoting
    # only `marks[0]` left a whole branch unreached.
    for i, extra in enumerate(marks[1:], start=1):
        out[f"marker {i} quoted inside the answer"] = (
            f"The model writes {extra} to open a call. " + PROSE * 4
        )

    # Ends mid-marker: the read-ahead is still holding a partial when the
    # stream ends, and flush has to release it. Half a marker cannot complete.
    out["an answer that ends mid-marker"] = (
        PROSE * 3 + marks[0][: max(1, len(marks[0]) // 2)]
    )

    if dialect.output_open_marker:
        out["reasoning the model opens itself"] = (
            dialect.output_open_marker + PROSE * 3 + end + "The answer is 42."
        )
    return out


# Shapes that close the reasoning channel without ever having opened it. Where
# the reasoning/content boundary falls is chunk-dependent here and cannot be
# otherwise: knowing a `</think>` is still to come means waiting for it, and
# waiting for it is the stall. Bounded first-byte latency, honouring an
# unopened end marker, and chunk-invariance are three properties of which an
# implementation gets two. vLLM drops the second -- no start token in the
# vocabulary means content, emitted at once -- and so does SGLang, whose test
# for it is named `test_text_before_think_token_is_chunk_dependent`.
#
# `starts_thinking` is what makes dropping it safe: a prompt whose template
# opened the channel says so, and such a stream never reaches that state.
#
# Weakened here, not excluded. The text as a whole and the tool events must
# still be invariant -- which is what caught the fabricated tool call in
# #1961, where the variants disagreed on both.
CLOSE_WITHOUT_OPEN = ("a tool marker inside reasoning",)


def _pairs():
    for dialect in DIALECTS:
        for parser in ALL_PARSERS:
            for shape_name, text in shapes(dialect, parser).items():
                yield pytest.param(
                    dialect,
                    parser,
                    text,
                    shape_name in CLOSE_WITHOUT_OPEN,
                    id=f"{dialect.think_end_marker.strip('<>/|')}-{parser.NAME}-"
                    f"{shape_name.replace(' ', '_')}",
                )


PAIRS = list(_pairs())


# Shapes with no marker the pipeline is entitled to honour. Withholding in
# these is never correct, which is what makes them the bounded-withhold
# corpus; the reasoning-bearing shapes hold until their end marker by design.
def carries_no_promise(shape_name: str) -> bool:
    """Shapes with no marker the pipeline is entitled to act on.

    Withholding or discarding anything in these is never correct, which is
    what makes them the bounded-withhold and conservation corpus. Matched by
    prefix rather than listed, because the per-marker shapes are generated
    -- a named list silently excluded them and left Kimi-K3's truncation
    uncovered.
    """
    return shape_name.startswith(
        ("trigger chars in ordinary prose", "a marker", "marker ")
    )


def _hold_pairs():
    for dialect in DIALECTS:
        for parser in ALL_PARSERS:
            for shape_name, text in shapes(dialect, parser).items():
                if not carries_no_promise(shape_name):
                    continue
                yield pytest.param(
                    dialect,
                    parser,
                    text,
                    id=f"{dialect.think_end_marker.strip('<>/|')}-{parser.NAME}-"
                    f"{shape_name.replace(' ', '_')}",
                )


HOLD_PAIRS = list(_hold_pairs())


def _partial_pairs():
    for dialect in DIALECTS:
        for parser in ALL_PARSERS:
            yield pytest.param(
                dialect,
                parser,
                shapes(dialect, parser)["an answer that ends mid-marker"],
                id=f"{dialect.think_end_marker.strip('<>/|')}-{parser.NAME}",
            )


PARTIAL_PAIRS = list(_partial_pairs())


class TestEveryFormatIsCovered:
    """The generation's own preconditions. These are what make it extensible."""

    def test_every_registered_parser_declares_its_markers(self):
        """A new format joins the corpus by declaring, not by being written up.

        Without this the generation below would silently produce nothing for a
        parser that forgot `START_MARKERS`, and the format would look covered.
        """
        # Asked of the class's own `__dict__`, not of the attribute: a
        # parser that subclasses another inherits its markers, so a missing
        # declaration reads as present and the new format silently gets
        # covered against the *parent's* markers. A registered format is a
        # distinct thing on the wire and has to say so itself, even when the
        # tuple would repeat.
        undeclared = [
            p.NAME
            for p in ALL_PARSERS
            if "START_MARKERS" not in vars(p) or not p.START_MARKERS
        ]
        assert not undeclared, f"registered parsers with no START_MARKERS: {undeclared}"

    def test_every_dialect_declares_an_end_marker(self):
        missing = [d for d in DIALECTS if not d.think_end_marker]
        assert not missing, "a dialect with no end marker cannot be streamed"

    def test_the_corpus_grows_with_the_registries(self):
        """Guards against the generator quietly degenerating to nothing."""
        assert len(PAIRS) >= 3 * len(ALL_PARSERS), (
            f"{len(PAIRS)} cases for {len(ALL_PARSERS)} parsers and "
            f"{len(DIALECTS)} dialects -- the generator lost a dimension"
        )


class TestEverySeedingSiteIsSeeded:
    """Nothing may build a reasoning filter without saying where it starts.

    An output that begins inside the reasoning channel carries no opening
    marker, so the text cannot say so and the prompt has to. Since state 0
    stopped inferring it from a bare end marker -- inferring it meant waiting
    for one, and waiting was the stall -- an unseeded site does not degrade
    gracefully: the model's whole chain of thought is delivered as the answer.

    Checked by walking the source rather than by listing the sites, so an
    endpoint added later is covered the moment it exists. Three of the four
    sites that exist today were unseeded before this change, including the one
    the original bug was reported against.
    """

    ROOT = pathlib.Path(__file__).resolve().parents[2] / "atom" / "entrypoints"
    SEEDED = ("ReasoningFilter", "separate_reasoning")

    def _unseeded(self) -> list[str]:
        found = []
        for path in sorted(self.ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if name not in self.SEEDED:
                    continue
                seed = next(
                    (kw.value for kw in node.keywords if kw.arg == "starts_thinking"),
                    None,
                )
                # Positional counts: `separate_reasoning(text, seeded)` is a
                # correct call and an earlier version of this scan rejected it.
                if seed is None and name == "separate_reasoning":
                    seed = node.args[1] if len(node.args) > 1 else None
                rel = path.relative_to(self.ROOT.parents[1])
                if seed is None:
                    found.append(f"{rel}:{node.lineno} {name}(...) — not answered")
                elif isinstance(seed, ast.Constant):
                    # A literal is not an answer. `starts_thinking=False`
                    # spells the keyword and reintroduces the bug: the earlier
                    # scan accepted it, and a test in this very change was
                    # "repaired" by hardcoding the other literal.
                    found.append(
                        f"{rel}:{node.lineno} {name}(starts_thinking={seed.value!r})"
                        " — a literal, not the prompt"
                    )
        return found

    def test_no_entry_point_builds_one_without_the_seed(self):
        unseeded = self._unseeded()
        assert not unseeded, "sites that never say where reasoning starts:\n  " + (
            "\n  ".join(unseeded)
        )

    @pytest.mark.parametrize(
        "source, ok",
        [
            ("ReasoningFilter(starts_thinking=prompt_starts_in_reasoning(p))", True),
            ("separate_reasoning(t, starts_thinking=seeded or flag)", True),
            ("separate_reasoning(t, seeded)", True),
            ("ReasoningFilter()", False),
            ("ReasoningFilter(starts_thinking=False)", False),
            ("separate_reasoning(t)", False),
        ],
    )
    def test_the_scan_accepts_answers_and_rejects_literals(self, source, ok, tmp_path):
        """The scan's own two-sided check.

        Without it the rule drifts: the first version spelled "is the keyword
        present", which passes `starts_thinking=False` -- exactly the bug --
        and fails a correct positional call.
        """
        f = tmp_path / "atom" / "entrypoints" / "probe.py"
        f.parent.mkdir(parents=True)
        f.write_text(source + "\n")
        scan = TestEverySeedingSiteIsSeeded()
        scan.ROOT = f.parent
        assert (scan._unseeded() == []) is ok, scan._unseeded()


class TestChunkInvariance:
    """Where the token boundaries fall must not change what the client sees."""

    @pytest.mark.parametrize("dialect, parser, text, split_may_move", PAIRS)
    def test_the_same_text_split_differently_reads_the_same(
        self, dialect, parser, text, split_may_move
    ):
        by_split = {
            label: drive(text, chunks, parser)
            for label, chunks in split_every_way(text).items()
        }
        variants: dict = {}
        for label, seen in by_split.items():
            key = seen.key
            if split_may_move:
                key = (seen.reasoning + seen.content, key[2], key[3])
            variants.setdefault(key, []).append(label)
        if len(variants) > 1:
            report = "\n".join(
                f"  {'/'.join(labels)}: "
                + "  ".join(
                    (
                        f"{part!r}"
                        if isinstance(part, str) and len(part) < 40
                        else (f"{len(part)}ch" if isinstance(part, str) else f"{part}")
                    )
                    for part in k
                )
                for k, labels in variants.items()
            )
            pytest.fail(f"{len(variants)} different results for one text:\n{report}")


class TestConservation:
    """Text handed to the pipeline comes back, or is part of a tool call.

    Chunk-invariance cannot see deletion -- text dropped the same way under
    every chunking is perfectly invariant -- and bounded withhold cannot
    either, because the bytes before the loss are released on time. A quoted
    `<tool_call>` in an ordinary answer opened a region that never closed,
    and everything after it was discarded at flush: no event, no error,
    `finish_reason` still `stop`. Fifty of eighty-two characters, silently.
    """

    @pytest.mark.parametrize("dialect, parser, text", HOLD_PAIRS)
    def test_an_answer_that_calls_nothing_keeps_what_follows_the_marker(
        self, dialect, parser, text
    ):
        """Judged on the tail, not byte-for-byte.

        A format may legitimately consume its own markers -- Kimi-K3 strips
        channel framing from plain answers by design -- so equality would fail
        on correct behaviour. What no format may do is swallow the prose that
        came after one.
        """
        marker = next((m for m in parser.START_MARKERS if m in text), None)
        if marker is None:
            pytest.skip("this shape carries no marker for this format")
        # Stripped: a format may trim the edges of what it hands back, and
        # whether it should is a separate question from whether it kept the
        # prose at all. This property is only about the prose.
        tail = text.split(marker, 1)[1].strip()
        for label, chunks in split_every_way(text).items():
            seen = drive(text, chunks, parser)
            if seen.events:
                continue  # a real tool call consumes its own bytes
            got = seen.reasoning + seen.content
            assert tail in got, (
                f"{label}: everything after {marker!r} was dropped\n"
                f"  missing: {tail[:60]!r}\n"
                f"  delivered {len(got)} of {len(text)} characters"
            )


class TestNothingIsHeldPastTheEnd:
    """Whatever the read-ahead still holds is released at end of stream.

    A partial marker at the end of a response never completes, so it is text.
    Losing it is invisible to every other property here: the stream is
    chunk-invariant, the withhold stayed bounded, and the loss is a handful of
    characters at the very end. `KimiParser.flush` dropped six of them.
    """

    @pytest.mark.parametrize("dialect, parser, text", PARTIAL_PAIRS)
    def test_a_dangling_partial_marker_still_arrives(self, dialect, parser, text):
        for label, chunks in split_every_way(text).items():
            seen = drive(text, chunks, parser)
            got = seen.reasoning + seen.content
            assert got.strip().endswith(text.strip()[-4:]), (
                f"{label}: the held tail was never released\n"
                f"  expected to end with {text.strip()[-12:]!r}\n"
                f"  got {got.strip()[-12:]!r}"
            )


# Text built from a dialect's own markers, every way they can sit around
# them. Generated rather than listed because hand-picked cases found three
# divergences in a row here, each after the previous one was declared fixed:
# a lost prefix, then a space, then a newline.
_REASONING_GLUE = ["", " ", "\n", "\n\n", "x", " x ", "\nx\n"]


def reasoning_shapes(dialect) -> list[str]:
    open_m = dialect.output_open_marker or dialect.prompt_open_marker
    end_m = dialect.think_end_marker
    out = set()
    for a, b, c in itertools.product(_REASONING_GLUE, repeat=3):
        out.add(a + open_m + b + end_m + c)  # closed block
        out.add(a + open_m + b)  # truncated mid-block
        out.add(a + end_m + c)  # end marker with no opener
        out.add(a + open_m + b + end_m + c + open_m + b + end_m + c)  # two blocks
    return sorted(out)


REASONING_DIALECTS = [
    pytest.param(d, sth, id=f"d{i}-{'prompt-opened' if sth else 'self-opened'}")
    for i, d in enumerate(DIALECTS)
    for sth in (False, True)
]


def split_reasoning_streaming(text: str, chunk: int, starts_thinking: bool):
    f = ReasoningFilter(starts_thinking=starts_thinking)
    segs = []
    for i in range(0, len(text), chunk):
        segs += f.process(text[i : i + chunk])
    segs += f.flush()
    return (
        "".join(t for k, t in segs if k == "reasoning_content"),
        "".join(t for k, t in segs if k == "content"),
    )


class TestTheReasoningSplitAgreesWithItself:
    """The same rule as the class below, one stage earlier.

    That class holds the *tool parser* to stream/non-stream agreement. Nothing
    held the *reasoning* split to it, and it diverged: a model that answers,
    opens a `<think>` block and answers again had the block extracted when
    streamed, and handed to the client as literal tags with the chain of
    thought inside `content` when not.

    One test per dialect rather than per shape. The corpus is a few thousand
    strings and pytest ids for each would outnumber the rest of this suite ten
    to one; the loop reports every divergence at once, which is also what you
    want when a change breaks a whole class of them.

    Judged byte-for-byte. It could not be, until the two things stopping it
    were removed: this path stripped, and the filter's own `lstrip("\n")`
    after the end marker saw only what happened to be buffered, so the same
    answer kept its newlines at one chunk size and lost them at another.
    Neither survived the question of what it was for -- the newline a model
    writes before its answer is not a marker, and only markers may be
    removed. Across this corpus that took content agreement from 50% to 100%.
    """

    CHUNKS = (1, 3, 11, 10_000)

    def _divergences(self, dialect, starts_thinking, field: int):
        out = []
        for text in reasoning_shapes(dialect):
            non = separate_reasoning(text, starts_thinking=starts_thinking)[field]
            for chunk in self.CHUNKS:
                got = split_reasoning_streaming(text, chunk, starts_thinking)[field]
                if (non or "") != got:
                    out.append((text, chunk, non, got))
        return out

    @pytest.mark.parametrize("dialect, starts_thinking", REASONING_DIALECTS)
    def test_the_answer_is_the_same_however_it_is_delivered(
        self, dialect, starts_thinking
    ):
        bad = self._divergences(dialect, starts_thinking, 1)
        assert not bad, self._report("content", bad)

    @pytest.mark.parametrize("dialect, starts_thinking", REASONING_DIALECTS)
    def test_the_chain_of_thought_is_the_same_too(self, dialect, starts_thinking):
        """Agreement on the answer is not enough: a split that put the
        reasoning in the wrong field would still pass the test above if the
        words happened to land in `content` either way."""
        bad = self._divergences(dialect, starts_thinking, 0)
        assert not bad, self._report("reasoning", bad)

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("a<think>b</think>\nc", "a\nc"),
            ("a<think>b</think>\n\nc", "a\n\nc"),
            ("a<think>b</think>c", "ac"),
            ("a<think>b</think> c", "a c"),
            ("<think>b</think>\n\nThe answer.", "\n\nThe answer."),
            ("<think>b</think>```\nx\n```\n", "```\nx\n```\n"),
        ],
        ids=["one-newline", "two-newlines", "none", "a-space", "an-answer", "a-block"],
    )
    def test_the_newline_a_model_puts_before_its_answer_survives(self, text, expected):
        """Spelled out, because the sweep above says only that the two *agree*.

        Two paths that both dropped it would satisfy that and still lose a
        code block's final newline -- which is the symptom the byte-for-byte
        rule was written for one stage later, and the one measured here
        before the strips came out.
        """
        assert separate_reasoning(text, starts_thinking=False)[1] == expected
        f = ReasoningFilter(starts_thinking=False)
        segs = f.process(text) + f.flush()
        assert "".join(t for k, t in segs if k == "content") == expected

    @staticmethod
    def _report(field, bad):
        lines = [f"{len(bad)} shapes split {field} two ways:"]
        for text, chunk, non, got in bad[:5]:
            lines.append(
                f"  chunk={chunk} text={text!r}\n"
                f"    stream=false {non!r}\n"
                f"    stream=true  {got!r}"
            )
        return "\n".join(lines)


class TestNonStreamingAgreesWithStreaming:
    """An answer with no tool call comes back the same on both paths.

    The non-streaming path runs the format's `parse`; the streaming path
    releases bytes as they arrive and owns nothing to tidy them with. So any
    tidying `parse` does to an answer it found no call in is a difference the
    client sees between `stream=true` and `stream=false` -- and every format
    did some, `.strip()`, which cost a code-block answer its trailing newline.

    Generated from the registry rather than listed, so the rule binds a format
    added later without anyone remembering to add a case. It is stated on
    `ToolCallParser.parse`, and this is what holds formats to it.
    """

    @pytest.mark.parametrize("dialect, parser, text", HOLD_PAIRS)
    def test_the_two_paths_deliver_the_same_text(self, dialect, parser, text):
        non_streaming, calls = parse_tool_calls(text, None, parser)
        if calls:
            pytest.skip("this shape parsed a call; the rule binds the no-call case")
        streamed = drive(text, split_every_way(text)["fixed-3"], parser)
        assert streamed.content == non_streaming, (
            f"{parser.NAME} answers the same request two ways\n"
            f"  stream=false {non_streaming[-60:]!r}\n"
            f"  stream=true  {streamed.content[-60:]!r}"
        )

    @pytest.mark.parametrize("dialect, parser, text", HOLD_PAIRS)
    def test_no_call_means_nothing_but_this_format_s_own_framing_goes(
        self, dialect, parser, text
    ):
        """Agreement is necessary but not sufficient: both could delete it.

        What may be removed is a marker this format declares. Everything else
        -- in particular whitespace, which is what every format's trailing
        `.strip()` took -- has to survive.
        """
        content, calls = parse_tool_calls(text, None, parser)
        if calls:
            pytest.skip("this shape parsed a call; the rule binds the no-call case")
        rebuilt = content
        for marker in parser.START_MARKERS:
            rebuilt = rebuilt.replace(marker, "")
        expected = text
        for marker in parser.START_MARKERS:
            expected = expected.replace(marker, "")
        assert rebuilt == expected, (
            f"{parser.NAME} removed something that was not one of its markers\n"
            f"  in  {expected[-60:]!r}\n"
            f"  out {rebuilt[-60:]!r}"
        )


class TestBoundedWithhold:
    """Text must not be held back longer than a marker could justify."""

    @pytest.mark.parametrize("dialect, parser, text", HOLD_PAIRS)
    def test_a_trigger_character_does_not_hold_the_rest_of_the_answer(
        self, dialect, parser, text
    ):
        triggers = trigger_chars(parser.START_MARKERS, dialect_markers(dialect))
        control = defuse(text, triggers)

        fine = split_every_way(text)["fixed-1"]
        seen = drive(text, fine, parser)
        if not seen.content:
            pytest.skip("no content channel in this shape; nothing to hold back")

        ctl = drive(control, split_every_way(control)["fixed-1"], parser)
        held = seen.first_content_at or len(text)
        baseline = ctl.first_content_at or len(control)
        assert held - baseline <= SLACK, (
            f"first content byte at input offset {held}/{len(text)}, against "
            f"{baseline} for the same text with {sorted(triggers)} neutralised"
        )


# One real tool call per registered format, in that format's own syntax. Not
# generated: a call's payload is the one thing each format spells differently,
# so the table is written out and `test_every_format_has_a_call` is what stops
# a new format from joining the registry without one.
_NS = "]<]minimax[>["
_D = "｜DSML｜"
REAL_CALLS: dict[str, str] = {
    "kimi": (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    ),
    "glm": (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    ),
    "qwen": (
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>"
    ),
    "kimi_k3": (
        '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>Paris<|close|>argument<|close|>call'
    ),
    "dsml": (
        f'<{_D}tool_calls><{_D}invoke name="get_weather">'
        f'<{_D}parameter name="city" string="true">Paris</{_D}parameter>'
        f"</{_D}invoke></{_D}tool_calls>"
    ),
    # Parameters named by the tag, which is what tells this format from DSML.
    # This entry was written in DSML's `<parameter name="...">` spelling and
    # so parsed to `get_weather({})` -- a call with no arguments, passing
    # every test here while exercising none of the parameter path.
    "minimax": (
        f'{_NS}<tool_call>{_NS}<invoke name="get_weather">'
        f"{_NS}<city>Paris{_NS}</city>"
        f"{_NS}</invoke>{_NS}</tool_call>"
    ),
}


class TestARealCallSurvivesTheStream:
    """The corpus above is all *non*-calls -- text that merely looks like one.

    That is deliberate and it left a hole: nothing drove an actual tool call
    through the streaming facade, so the wiring between the facade, each
    format's read-ahead and its parser was unasserted. Pointing every parser's
    scanner at another format's marker broke tool calls on four formats and
    the whole suite stayed green.
    """

    def test_every_format_has_a_call(self):
        assert set(REAL_CALLS) == {p.NAME for p in ALL_PARSERS}, (
            "a registered format has no real call here, so nothing checks that "
            "its streaming path produces one"
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_non_streaming_path_reads_the_call(self, parser):
        """The fixture is a real call in this format, and this says so."""
        _, calls = parse_tool_calls(REAL_CALLS[parser.NAME], DECLARED_TOOLS, parser)
        assert [c.function["name"] for c in calls] == ["get_weather"]

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_call_carries_its_argument(self, parser):
        """And the payload is in *this* format's spelling, not another's.

        Only the name was asserted, so an entry written in the wrong format's
        parameter syntax parsed to `get_weather({})` and passed everything
        here -- minimax's was, for as long as the table existed.
        """
        _, calls = parse_tool_calls(REAL_CALLS[parser.NAME], DECLARED_TOOLS, parser)
        assert json.loads(calls[0].function["arguments"]) == {"city": "Paris"}

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_call_arrives_however_it_is_chunked(self, parser):
        text = "Let me look. " + REAL_CALLS[parser.NAME]
        for label, chunks in split_every_way(text).items():
            seen = drive(text, chunks, parser)
            starts = seen.events.count("tool_call_start")
            assert starts == 1, f"{label}: {starts} calls, events {seen.events}"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_text_before_it_still_arrives(self, parser):
        text = "Let me look. " + REAL_CALLS[parser.NAME]
        seen = drive(text, split_every_way(text)["fixed-3"], parser)
        assert "Let me look." in seen.content

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_and_so_does_the_text_written_between_two_calls(self, parser):
        """The sentence a model writes while making a second call.

        `TestConservation` cannot see this: it `continue`s the moment a shape
        produces events, on the grounds that "a real tool call consumes its
        own bytes" -- true of the call, not of the prose beside it. And the
        two chunk-invariance properties cannot see it either, because both
        delivery paths are the same engine and agreed on deleting it. Five of
        the six formats did, silently, with `finish_reason: tool_calls`.
        """
        call = REAL_CALLS[parser.NAME]
        middle = "Now let me also check Rome."
        text = f"Before. {call} {middle} {call.replace('Paris', 'Rome')} After."
        content, calls = parse_tool_calls(text, DECLARED_TOOLS, parser)
        assert len(calls) == 2, f"{len(calls)} calls, so this shape proves nothing"
        assert (
            middle in content
        ), f"the sentence between two calls was deleted: {content!r}"
        for label, chunks in split_every_way(text).items():
            seen = drive(text, chunks, parser)
            assert middle in seen.content, f"{label}: {seen.content!r}"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_forbidding_calls_delivers_the_answer_and_not_the_wire_format(self, parser):
        """`tool_choice: "none"` on every format, not just the one tested.

        Suppression used to skip opening the region, so the call's own bytes
        streamed out as `content`: the whole `<invoke name=...>` payload, on
        all six. The single-format test that covered this asserted the raw
        text came back *verbatim* and so encoded the leak as the contract --
        and on Kimi-K3 even that was untrue, because the read-ahead drops its
        framing mid-region and the client got a mangled fragment.
        """
        text = "Sure. " + REAL_CALLS[parser.NAME] + " Done."
        content, calls = parse_tool_calls(
            text, DECLARED_TOOLS, parser, suppress_calls=True
        )
        assert calls == [], "a forbidden call was dispatched"
        assert (
            "Sure." in content and "Done." in content
        ), f"the answer around the call was eaten: {content!r}"
        for marker in parser.START_MARKERS:
            assert (
                marker not in content
            ), f"raw wire markup shown to the user: {marker!r} in {content!r}"
        for label, chunks in split_every_way(text).items():
            seen = drive(text, chunks, parser, suppress_calls=True)
            assert seen.content == content, f"{label}: {seen.content!r}"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_and_it_arrives_between_them_rather_than_ahead_of_both(self, parser):
        """Order is part of the answer.

        Emitting both calls and then the prose would put the same characters
        on the wire in the wrong place: a client rendering deltas in arrival
        order shows "let me also check Rome" after the Rome call it introduces.
        """
        call = REAL_CALLS[parser.NAME]
        middle = "Now let me also check Rome."
        text = f"Before. {call} {middle} {call.replace('Paris', 'Rome')} After."
        parser_obj = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
        events = parser_obj.process(text) + parser_obj.flush()
        order, said = [], ""
        for kind, data in events:
            if kind == "content":
                said += data
                if middle in said and "middle" not in order:
                    order.append("middle")
            elif kind == "tool_call_start":
                order.append(data["function"]["name"])
        assert order.count("middle") == 1
        assert order.index("middle") > order.index(
            "get_weather"
        ), f"the sentence introducing the second call arrived first: {order}"


# Markers a format declares so the read-ahead will not split them, but which
# do not hand the stream over: channel framing that wraps every answer. Only
# Kimi-K3 has any; a format absent here declares none, which is the default.
FRAMING_NOT_A_REGION: dict[str, set[str]] = {
    # Every channel token K3 wraps an answer in. Only the call prefix and the
    # tools wrapper mean a call; the rest are removed on both paths, so the
    # read-ahead has to know them to keep the two in step. Written out rather
    # than read off `_K3_CONTENT_FRAMING`, for the reason in the test below:
    # a copy that agrees with the code by construction agrees with a broken
    # code too. It went from eleven to fourteen when the last three -- which
    # `parse` stripped and the scanner had never heard of -- were declared;
    # they leaked verbatim into streamed content and vanished when not.
    "kimi_k3": {
        "<|open|>response<|sep|>",
        "<|close|>response<|sep|>",
        "<|end_of_msg|>",
        "<|open|>think<|sep|>",
        "<|close|>think<|sep|>",
        "<|open|>message<|sep|>",
        "<|close|>message<|sep|>",
        "<|close|>response",
        "<|close|>think",
        "<|close|>message",
        "<|close|>tools",
        "<|close|>argument",
        "<|close|>call",
        "<|sep|>",
    },
}


def prefix_pairs(parser) -> list[tuple[str, str]]:
    """Every (short, long) pair of this format's markers where short opens long."""
    ms = parser.START_MARKERS
    return [(a, b) for a in ms for b in ms if a != b and b.startswith(a)]


def _drive_parser(parser, text, size):
    stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
    events = []
    for i in range(0, len(text), size):
        events += stream.process(text[i : i + size])
    return events + stream.flush()


def channel_tokens(parser) -> list[str]:
    """Framing-token-shaped strings this format's own module names.

    Harvested from the module rather than from `START_MARKERS`, and that is
    the whole point: a corpus built from the declared list cannot contain a
    token the format strips but never declared, which is exactly the drift
    this looks for. Kimi-K3 stripped `<|close|>tools`, `<|close|>call` and
    `<|close|>argument` and declared none of them.
    """
    module = sys.modules[parser.__module__]
    found: set[str] = set()
    for value in vars(module).values():
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, tuple):
            found.update(v for v in value if isinstance(v, str))
        elif isinstance(value, re.Pattern):
            # `<|close|>tools` and its siblings exist only inside an
            # alternation, so the pattern is where they have to be read from
            # -- unescaped, since that is how they arrive on the wire.
            found.update(
                re.findall(r"(?:<\\\|[^|]*\\\|>)+\w*", value.pattern),
            )
    return sorted(
        {
            t.replace("\\", "")
            for t in found
            if t.startswith(("<", "]<")) and "(" not in t
        }
    )


class TestAFormatDeclaresEveryTokenItStrips:
    """What `parse` removes from content, the read-ahead has to know.

    They answer the same question -- which bytes are framing rather than
    answer -- and the streaming path can only hold back a literal it was told
    about. Kimi-K3 kept two lists and they drifted: three tokens were
    stripped and undeclared, so they reached the client verbatim when
    streamed and vanished when not, and a quoted
    `<|open|>argument key="city"<|sep|>` came out with only its separator
    removed -- text matching neither path.
    """

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_framing_comes_out_the_same_whether_or_not_it_is_streamed(self, parser):
        tokens = channel_tokens(parser)
        assert tokens, f"{parser.NAME}: no tokens harvested, this asserts nothing"
        bad = []
        for a, b in itertools.product(tokens, repeat=2):
            text = f"A {a} B {b} C"
            non = parse_tool_calls(text, DECLARED_TOOLS, parser)[0]
            if parse_tool_calls(text, DECLARED_TOOLS, parser)[1]:
                continue  # a real call; the no-call rule is what binds here
            for size in (1, 3, 999):
                got = "".join(
                    d for k, d in _drive_parser(parser, text, size) if k == "content"
                )
                if got != non:
                    bad.append((text, non, got))
                    break
        assert not bad, (
            f"{len(bad)} of {len(tokens) ** 2} token pairs split two ways, "
            f"first: {bad[0]}"
        )


class TestAPrefixPairCannotChangeTheHandover:
    """Longest-first only settles a tie the buffer can already see.

    `MarkerScanner` reports the longest marker at a position -- among the ones
    already complete in its buffer. A chunk ending exactly at the shorter of a
    prefix pair reports the shorter one, because the longer has not arrived to
    be preferred. So which of the two fires is a function of where the
    boundary landed.

    Harmless while both halves agree about handing the stream over, and today
    every pair does: K3's three are all channel framing that opens no region,
    so either way the marker is dropped and the remainder is caught as its own
    marker. Measured -- a K3 answer comes out identical at seven chunk sizes.

    It stops being harmless the moment a pair disagrees. With a synthetic
    `("<|end|>", "<|end|>call")` where only the long one opens a region, one
    text produced two different answers across six chunk sizes: the marker was
    deleted as framing at chunk 1, 2 and 9, and handed over as a region at 7,
    8 and 999.

    So this is the cheap half of the fix. The expensive half -- withholding a
    complete match that could still grow -- is the rule `_plan` would need to
    actually keep its promise, and is worth writing when a format needs it,
    not before.
    """

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_both_halves_agree_about_opening_a_region(self, parser):
        disagreeing = [
            (short, long)
            for short, long in prefix_pairs(parser)
            if parser.opens_region(short) != parser.opens_region(long)
        ]
        assert not disagreeing, (
            f"{parser.NAME} declares a marker that is a prefix of another and "
            f"they disagree about handing the stream over, so which happens "
            f"depends on the chunk boundary: {disagreeing}"
        )

    def test_the_registry_still_has_pairs_to_check(self):
        """Otherwise the test above is green because it examined nothing.

        K3's `<|close|>response` / `<|close|>response<|sep|>` and its two
        siblings are the only pairs there are; drop them and this suite would
        keep passing while the rule went unenforced.
        """
        found = {p.NAME: prefix_pairs(p) for p in ALL_PARSERS}
        total = sum(len(v) for v in found.values())
        assert total >= 3, f"no prefix pairs left to check: {found}"

    @pytest.mark.parametrize(
        "source", ["<think></think>", "<|open|>think<|sep|>"], ids=["think", "channel"]
    )
    def test_a_dialect_declares_no_pair_that_changes_the_channel(self, source):
        """The same rule on the reasoning side, where it was not being asked.

        `ReasoningFilter` watches this dialect's framing *and* its end markers
        in one scanner, and the two do opposite things: framing is dropped and
        the state is unchanged, an end marker closes the channel. So a bare
        closer that is a proper prefix of a paired end marker is precisely the
        disagreeing pair `_plan` warns about -- the short one fires, the
        channel never closes, and at four bytes per chunk the whole answer
        comes back as `reasoning_content` while `stream=false` splits it
        correctly. Adding `<|close|>think` to Kimi-K3's channel framing did
        exactly that; the tool-parser version of this test could not see it,
        because a dialect is not a parser.
        """
        dialect, _ = resolve_dialect(source)
        framing, ends = set(dialect.content_framing), set(dialect.end_markers)
        watched = framing | ends
        disagreeing = [
            (short, long)
            for short in watched
            for long in watched
            if short != long
            and long.startswith(short)
            and ((short in ends) != (long in ends))
        ]
        assert not disagreeing, (
            f"{source!r} watches a marker that is a prefix of another and only "
            f"one of them closes the reasoning channel, so which happens "
            f"depends on the chunk boundary: {disagreeing}"
        )

    def test_a_disagreeing_pair_is_rejected(self):
        """And that the check can fail at all -- built rather than waited for."""

        class Synthetic(QwenXmlParser):
            NAME = "synthetic"
            START_MARKERS = ("<|end|>", "<|end|>call")

            @classmethod
            def opens_region(cls, marker):
                return marker == "<|end|>call"

        with pytest.raises(AssertionError, match="chunk boundary"):
            self.test_both_halves_agree_about_opening_a_region(Synthetic)


class TestAPlainAnswerDoesNotWaitForTheEnd:
    """A format's own framing is not a reason to stop streaming.

    `START_MARKERS` answers "which literals must not be split"; `opens_region`
    answers "which of them hand the rest of the stream to this format". For
    most formats those are the same set. Kimi-K3 is where they are not: three
    of its five wrap every answer it gives, including `<|open|>response<|sep|>`
    at the very start, so reading any marker as a handover meant a K3 response
    delivered nothing until EOS -- 324 of 324 characters in one frame.

    Chunk-invariance cannot see this and neither can the agreement property:
    delivering everything at flush is perfectly invariant and the text is
    identical. What separates them is *when*.
    """

    @staticmethod
    def _split_by_arrival(parser, text) -> tuple[int, int]:
        stream = ToolCallStreamParser(parser_cls=parser)
        during = 0
        for i in range(0, len(text), 4):
            during += sum(
                len(d) for k, d in stream.process(text[i : i + 4]) if k == "content"
            )
        at_flush = sum(len(d) for k, d in stream.flush() if k == "content")
        return during, at_flush

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_partition_is_the_declared_one(self, parser):
        """Which markers open a region, stated here and not read off the code.

        Asking `opens_region` for the shape *and* the expectation makes the
        test agree with whatever the code says: flipping every answer to True
        emptied the framing list below and the behavioural test skipped itself
        clean through the mutation.
        """
        declared = {m for m in parser.START_MARKERS if not parser.opens_region(m)}
        assert declared == FRAMING_NOT_A_REGION.get(parser.NAME, set())

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_framing_that_opens_no_region_does_not_stop_delivery(self, parser):
        framing = sorted(FRAMING_NOT_A_REGION.get(parser.NAME, ()))
        if not framing:
            pytest.skip("every marker this format declares opens a region")
        text = framing[0] + PROSE * 6 + framing[-1]
        during, at_flush = self._split_by_arrival(parser, text)
        assert during > at_flush, (
            f"{parser.NAME} held {at_flush} characters to EOS and streamed "
            f"{during}; its framing is being read as a tool region"
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_an_answer_with_no_marker_at_all_streams_whole(self, parser):
        """The floor: nothing to hold means nothing held."""
        during, at_flush = self._split_by_arrival(parser, PROSE * 6)
        assert at_flush == 0 and during == len(PROSE * 6)

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_region_marker_still_hands_over(self, parser):
        """The other half: what does open a region must still be buffered,
        or a half-written call would be emitted as text."""
        opener = next(m for m in parser.START_MARKERS if parser.opens_region(m))
        during, _ = self._split_by_arrival(parser, "Before. " + opener + "junk")
        assert during == len("Before. "), f"{parser.NAME} leaked past its opener"


DECLARED_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
]


def early_name(parser, region: str, tools) -> str | None:
    """The name the engine would announce for `region`.

    There is no separate peek to ask any more, and that is the point: the
    announcement is the first call of `parse_region` over the bytes so far,
    with `at_end=False`. The declared-name filter lives in the engine, so a
    test that wants it drives the engine instead.
    """
    calls = parser.parse_region(region, tools, at_end=False).calls
    return calls[0].function["name"] if calls else None


def sees_a_call_in_progress(parser) -> bool:
    """Can this format's `parse_region` read a call whose arguments are still
    arriving?

    Measured, not declared. There used to be a class attribute saying this;
    it outlived its only reader and the docs built on it went false without
    anything failing. The property is directly observable, so observe it:
    take the format's own call, make its argument long, and cut in the middle
    of the value.
    """
    long_value = "x" * 4000
    call = REAL_CALLS[parser.NAME].replace("Paris", long_value)
    partial = call[: call.index(long_value) + 2000]
    return bool(parser.parse_region(partial, DECLARED_TOOLS, at_end=False).calls)


# The two token formats: a Kimi entry is invisible until `<|tool_call_end|>`
# and a K3 call until `<|close|>call`, so neither can name a call whose
# arguments are still arriving.
NO_EARLY_NAME = {
    p.NAME: "a call of this format is invisible until its own end token"
    for p in ALL_PARSERS
    if not sees_a_call_in_progress(p)
}


class TestTheNameArrivesBeforeTheArguments:
    """Which tool is being called, sent as soon as the region reveals it.

    A region is buffered until it closes, so the client learned the tool only
    after the whole payload: measured on a 20 KB file write, 5030 of 5040
    tokens of nothing. Every format carries the name in its opener or close
    behind it, so it can go out first.

    Arguments stay buffered. SGLang streams those too, in JSON fragments, and
    a stream cut short then leaves the client holding an unterminated object.
    The name is the part worth the risk and the only part taken here.

    Judged on *when* each event lands and not on where it sits in the event
    list: the payload between the name and the arguments produces no events at
    all, so the two are adjacent either way. An earlier version of these
    checked adjacency and passed on every arm.
    """

    @staticmethod
    def _drive(parser, text, tools):
        """(chunks, chunk the name landed on, chunk the arguments landed on,
        every event kind in order)."""
        stream = ToolCallStreamParser(parser_cls=parser)
        stream.tools = tools
        events, at = [], {}
        chunks = [text[i : i + 4] for i in range(0, len(text), 4)]
        for n, chunk in enumerate(chunks, 1):
            for kind, _ in stream.process(chunk):
                events.append(kind)
                at.setdefault(kind, n)
        for kind, _ in stream.flush():
            events.append(kind)
            at.setdefault(kind, len(chunks))
        return len(chunks), at.get("tool_call_start"), at.get("tool_call_args"), events

    @staticmethod
    def _big_call(parser) -> str:
        """The registry's own call for this format, with a large payload."""
        return "Let me look. " + REAL_CALLS[parser.NAME].replace(
            "Paris", "Paris" + "x" * 800
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_reading_a_call_in_progress_is_what_moves_the_name(self, parser):
        """Two independently observable things, asserted to agree.

        One is whether `parse_region` can read a call whose argument is still
        arriving; the other is whether, on a real stream of a call with an
        800-byte payload, the name lands before the arguments. Neither is
        derived from the other and neither is a declaration -- the attribute
        that used to stand in for both outlived its reader, and the docs built
        on it said Kimi-K3 names nothing early when it does so for any call
        that fits inside the peek window.

        800 bytes because that is the shape announcing early exists for. A
        call short enough to fit in `Region.head` is named early by every
        format, which is a different claim.
        """
        total, at, args_at, _ = self._drive(
            parser, self._big_call(parser), DECLARED_TOOLS
        )
        assert at is not None and args_at is not None
        early = at < args_at
        assert early is sees_a_call_in_progress(parser), (
            f"{parser.NAME}: parse_region {'can' if not early else 'cannot'} read "
            f"a call in progress, but the name landed on chunk {at} of {total} "
            f"and the arguments on {args_at}"
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_call_that_fits_the_window_is_named_early_by_every_format(self, parser):
        """Including the two that cannot read one in progress -- for a short
        call the whole thing is inside `Region.head`, so `parse_region` sees a
        finished call there. The docs claimed otherwise for K3."""
        text = "Sure. " + REAL_CALLS[parser.NAME]
        assert len(REAL_CALLS[parser.NAME]) < _PEEK_WINDOW, "premise"
        _, at, args_at, _ = self._drive(parser, text, DECLARED_TOOLS)
        assert at is not None and args_at is not None and at <= args_at

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_declared_tool_is_named_early(self, parser):
        if parser.NAME in NO_EARLY_NAME:
            pytest.skip(NO_EARLY_NAME[parser.NAME])
        total, at, args_at, _ = self._drive(
            parser, self._big_call(parser), DECLARED_TOOLS
        )
        assert at is not None and args_at is not None
        assert at < args_at, f"{parser.NAME} sent the name with its arguments"
        assert at < total // 4, (
            f"{parser.NAME} announced at chunk {at} of {total}; the name is in "
            "the opener and should not wait for the payload"
        )

    @pytest.mark.parametrize(
        "tools, label",
        [
            ([{"type": "function", "function": {"name": "something_else"}}], "other"),
            (None, "none"),
            ([], "empty"),
        ],
    )
    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_an_undeclared_tool_is_not_announced(self, parser, tools, label):
        """The check that makes an early name safe: it cannot be retracted,
        and prose quoting a tool tag opens a region too."""
        _, at, args_at, _ = self._drive(parser, self._big_call(parser), tools)
        assert at == args_at, (
            f"{parser.NAME} sent the name of an undeclared tool ({label}) at "
            f"chunk {at}, ahead of its arguments at {args_at}"
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_declaring_it_is_what_moves_the_name_earlier(self, parser):
        """Same input, one variable: the arms differ only in `tools`."""
        if parser.NAME in NO_EARLY_NAME:
            pytest.skip(NO_EARLY_NAME[parser.NAME])
        text = self._big_call(parser)
        _, early, _, _ = self._drive(parser, text, DECLARED_TOOLS)
        _, late, _, _ = self._drive(parser, text, None)
        assert early < late, f"{parser.NAME}: declared {early}, undeclared {late}"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_client_still_sees_exactly_one_call(self, parser):
        """The announcement replaces the parse's own start, never doubles it.

        Kimi builds its start event inline rather than through `_emit_call`,
        so the deduplication had to be shared rather than written twice -- it
        was not, and the name went out twice for one call.
        """
        for tools in (DECLARED_TOOLS, None):
            _, _, _, events = self._drive(parser, self._big_call(parser), tools)
            assert events.count("tool_call_start") == 1, events
            assert events.count("tool_call_args") == 1, events

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_announcing_changes_nothing_but_the_timing(self, parser):
        """Same events, same order, whether or not the name went early."""
        text = self._big_call(parser)
        _, _, _, announced = self._drive(parser, text, DECLARED_TOOLS)
        _, _, _, plain = self._drive(parser, text, None)
        assert announced == plain

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_an_answer_that_only_quotes_a_marker_announces_nothing(self, parser):
        """No name to peek, so nothing is promised and the text still lands."""
        opener = next(m for m in parser.START_MARKERS if parser.opens_region(m))
        text = f"The model writes {opener} to open one. " + PROSE * 3
        _, _, _, events = self._drive(parser, text, DECLARED_TOOLS)
        assert "tool_call_start" not in events


# Each format's call opener, up to and including the tool name -- written
# out, like REAL_CALLS, because it is data about the format. Derived by
# re-peeking instead, the corpus moved whenever `peek_name` did and the
# property below went quietly vacuous.
CALL_OPENERS: dict[str, str] = {
    "kimi": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"),
    "glm": "<tool_call>get_weather",
    "qwen": "<tool_call><function=get_weather>",
    "kimi_k3": '<|open|>tools<|sep|><|open|>call tool="get_weather"<|sep|>',
    "dsml": f'<{_D}tool_calls><{_D}invoke name="get_weather">',
    "minimax": f'{_NS}<tool_call>{_NS}<invoke name="get_weather">',
}

# What legitimately comes next in each format -- the one tail that must make
# the name go out. Without a positive row the property below is satisfied by
# a peek that never announces anything.
CALL_CONTINUATIONS: dict[str, str] = {
    "kimi": '<|tool_call_argument_begin|>{"city":"Paris"}<|tool_call_end|>',
    "glm": "<arg_key>city</arg_key>",
    "qwen": "<parameter=city>Paris</parameter>",
    "kimi_k3": (
        '<|open|>argument key="city"<|sep|>Paris<|close|>argument<|close|>call'
    ),
    "dsml": f'<{_D}parameter name="city">Paris',
    "minimax": f"{_NS}<city>Paris",
}

# A closer that is NOT this format's own. Per format, because one format's
# foreign closer is another's legitimate one: `</tool_call>` leaves Qwen's
# `<function=` block open and so is prose there, while for GLM it closes the
# very block the name opened and `<tool_call>get_weather</tool_call>` is a
# real zero-argument call.
FOREIGN_CLOSERS: dict[str, str] = {
    "kimi": "</tool_call>",
    "glm": "</function>",
    "qwen": "</tool_call>",
    "kimi_k3": "<|close|>response<|sep|>",
    "dsml": "</tool_call>",
    "minimax": "</function>",
}

# And what prose looks like next. None of these may make the name go out.
PROSE_TAILS = [
    ("", "nothing yet"),
    (" and then the parameters.", "English"),
    ("<br>, like that.", "a tag, but not this format's"),
]

# A schema with a parameter in it. MiniMax names parameters by the tag, so
# with an empty schema it can only fall back to accepting any tag and the
# `<br>` row above passes for the wrong reason.
PEEK_TOOLS = [
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


class TestThePeekNeverNamesWhatTheParseCallsProse:
    """The early name and the parse must agree about what a call looks like.

    Every format is held to this, including the two whose name cannot go out
    before their arguments: "does not name prose" is a different claim from
    "names a real call early", and only the second is format-specific. Gating
    both on the same table skipped ten cases that pass.

    A name cannot be retracted. If the peek names a region and the parse then
    reads that same region as prose, the client has been told about a call
    that does not exist -- on `/v1/chat/completions` as a `tool_calls` delta
    whose `arguments` is `""`, which every agent loop feeds to `json.loads`,
    and on `/v1/messages` as a syntactically complete `tool_use` block with
    `input: {}` that a client cannot tell from a real zero-argument call.

    Four of the five formats that announce had the two disagree, because each
    wrote the rule twice: a follower set in a peek regex, a truncation test in
    `parse`. Qwen's peek accepted `</tool_call>` -- which closes the *outer*
    wrapper and leaves the `<function=` block unterminated, so `parse` read it
    as prose. Each format now answers the question once, from one constant,
    and both callers ask it.

    Which is also this property's limit, and worth being plain about: with one
    constant per format the two *cannot* disagree, so no mutation of that
    constant will fail this. What it guards is the next format, or the next
    rewrite, that goes back to writing the rule twice --
    `test_a_looser_peek_is_caught` is what shows it still can.
    """

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_opener_matches_this_format(self, parser):
        """Otherwise the regions below are not this format's syntax at all."""
        opener = CALL_OPENERS[parser.NAME]
        assert REAL_CALLS[parser.NAME].startswith(
            opener
        ), f"{parser.NAME}'s opener is not a prefix of its own real call"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    @pytest.mark.parametrize("tail, why", PROSE_TAILS, ids=lambda x: x)
    def test_prose_after_the_opener_names_nothing(self, parser, tail, why):
        self._check(parser, CALL_OPENERS[parser.NAME] + tail, why, expected=None)

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_closer_that_is_not_this_format_s_names_nothing(self, parser):
        """The shape Qwen got wrong: a closer that ends some *other* block."""
        region = (
            CALL_OPENERS[parser.NAME] + FOREIGN_CLOSERS[parser.NAME] + ", like that."
        )
        self._check(parser, region, "a closer from another block", None)

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_this_format_s_own_next_token_does_name_it(self, parser):
        region = CALL_OPENERS[parser.NAME] + CALL_CONTINUATIONS[parser.NAME]
        self._check(parser, region, "this format's own next token", "get_weather")

    @staticmethod
    def _check(parser, region, why, expected):
        """Asked of `peek_name` directly, not of a consequence.

        The obvious consequence -- "did `parse` hand the region back
        unchanged" -- is unsound for Kimi-K3, whose `parse` rewrites the
        content of *every* answer by stripping channel framing. A version
        keyed on that passed while K3 announced a tool for a sentence merely
        quoting a call opener.
        """
        got = early_name(parser, region, PEEK_TOOLS)
        assert got == expected, (
            f"{parser.NAME} with {why} after its opener: peek said {got!r}, "
            f"expected {expected!r} -- region {region!r}"
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_bare_opener_alone_names_nothing(self, parser):
        """A follower has to have arrived, not merely be possible.

        The opener on its own is the shape a quotation and a cut-off call
        share; only what comes next tells them apart. Waiting for it costs a
        few characters, and nothing at all when the call completes -- `parse`
        still produces it at flush.

        Stated for every format because it is the difference between the two
        kinds of dangling name. K3 announced here, and its `parse` never
        salvages a truncated call, so the client was left holding a name for
        a call that produced no arguments and no event.
        """
        opener = CALL_OPENERS[parser.NAME]
        assert (
            early_name(parser, opener, DECLARED_TOOLS) is None
        ), f"{parser.NAME} named a tool off its opener alone: {opener!r}"

    def test_a_looser_peek_is_caught(self):
        """The check can fail -- built rather than waited for."""

        class Loose(QwenXmlParser):
            NAME = "loose"

            @classmethod
            def parse_region(cls, region, tools, *, at_end):
                m = re.search(r"<function=([^>\n]+)>", region)
                if m is None:
                    return RegionParse()
                call = ToolCall(
                    id="loose",
                    type="function",
                    function={"name": m.group(1), "arguments": "{}"},
                )
                return RegionParse((call,), ((m.start(), len(region)),))

        with pytest.raises(AssertionError, match="peek said"):
            self._check(Loose, CALL_OPENERS["qwen"] + " and then...", "English", None)

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_real_call_is_still_named_early(self, parser):
        """And the rule above is not satisfied by never announcing anything."""
        assert (
            early_name(parser, REAL_CALLS[parser.NAME], DECLARED_TOOLS) == "get_weather"
        )


class TestAPromiseCannotBeTakenBack:
    """The cost of announcing early, stated rather than discovered.

    A name goes out before the call is known to close, so a response cut off
    at `max_tokens` mid-call has sent a name and may never send arguments.
    Nothing can retract it. What can be arranged is that it is not mistaken
    for a call the client should run: `completes_a_tool_call` keys on the
    arguments, so `stop_reason` / `finish_reason` stay ordinary and the text
    is still delivered.
    """

    @staticmethod
    def _drive(parser, text):
        stream = ToolCallStreamParser(parser_cls=parser)
        stream.tools = DECLARED_TOOLS
        events = []
        for i in range(0, len(text), 4):
            events += stream.process(text[i : i + 4])
        return events + stream.flush()

    @staticmethod
    def _drive_without_tools(parser, text):
        stream = ToolCallStreamParser(parser_cls=parser)
        events = []
        for i in range(0, len(text), 4):
            events += stream.process(text[i : i + 4])
        return events + stream.flush()

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_call_cut_off_before_it_parses_is_not_a_usable_call(self, parser):
        """Stated as an invariant, not as "this input parses nothing".

        Whether a format salvages a call from a given prefix is its own
        business and now depends on the tool being declared: GLM reads
        `<tool_call>get_weather` as a cut-off call to a declared tool, which
        it is. What must hold either way is that a name with no arguments
        behind it is never reported as something the client can run.
        """
        head = REAL_CALLS[parser.NAME].split("get_weather")[0] + "get_weather"
        events = self._drive(parser, "Sure. " + head)
        if "tool_call_args" not in [k for k, _ in events]:
            assert not completes_a_tool_call(
                events
            ), "a name with no arguments must not report as a usable call"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_answer_still_arrives_when_the_call_does_not(self, parser):
        head = REAL_CALLS[parser.NAME].split("get_weather")[0] + "get_weather"
        events = self._drive(parser, "Sure. " + head)
        delivered = "".join(d for k, d in events if k == "content")
        assert "Sure." in delivered, "the text before the call was dropped"

    @staticmethod
    def _liar():
        """A format whose `peek_name` and `parse` read the same bytes
        differently -- a bug in that format, and one two registered parsers
        were found to have."""

        class Liar(QwenXmlParser):
            @classmethod
            def parse_region(cls, region, tools, *, at_end):
                parsed = QwenXmlParser.parse_region(region, tools, at_end=at_end)
                if at_end:
                    for c in parsed.calls:
                        c.function["name"] = "something_else"
                return parsed

        return Liar

    def test_peeking_a_different_name_than_the_parse_does_not_kill_the_stream(self):
        """This used to raise. The caller is `flush`, on a live SSE stream
        that has already sent its 200, so the exception reached the client as
        a cut connection with no `[DONE]` -- and on `n>1` took the other
        choices with it."""
        events = self._drive(self._liar(), "Sure. " + REAL_CALLS["qwen"])
        assert [k for k, _ in events], "the stream produced nothing"

    def test_the_call_that_parsed_goes_out_whole(self):
        events = self._drive(self._liar(), "Sure. " + REAL_CALLS["qwen"])
        args = [d for k, d in events if k == "tool_call_args"]
        starts = {
            d["function"]["name"]: d["index"]
            for k, d in events
            if k == "tool_call_start"
        }
        assert "something_else" in starts, f"the parsed call never went out: {starts}"
        assert (
            len(args) == 1 and args[0]["index"] == starts["something_else"]
        ), "the arguments landed on an index the client bound to another name"

    def test_the_announced_name_is_left_without_arguments(self):
        """It cannot be retracted, but it can be left unusable -- which is
        what `completes_a_tool_call` and `finish_reason` both read."""
        events = self._drive(self._liar(), "Sure. " + REAL_CALLS["qwen"])
        announced = [
            d["index"]
            for k, d in events
            if k == "tool_call_start" and d["function"]["name"] == "get_weather"
        ]
        assert announced, "the announcement is the premise of this test"
        assert not [
            d for k, d in events if k == "tool_call_args" and d["index"] in announced
        ], "the wrong name was given arguments to run with"


def _big_kimi_call(payload_bytes: int) -> str:
    """The same, in Kimi's self-delimiting token format."""
    return (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        f'<|tool_call_argument_begin|>{{"note": "{"x" * payload_bytes}"}}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )


def _big_call(payload_bytes: int) -> str:
    """One Qwen tool call whose argument is `payload_bytes` long."""
    return (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter>"
        f"<parameter=note>{'x' * payload_bytes}</parameter>"
        "</function></tool_call>"
    )


class TestTheRegionIsNotCopiedPerChunk:
    """A buffered region costs what it is, not what it is squared.

    `self.buf += text` on an *attribute* is quadratic in CPython: the
    instance dict holds a reference, so the in-place fast path never applies
    and every chunk copies the whole buffer. Measured on a 128 KB tool call
    at four characters a chunk, 23 ms of event-loop CPU in `process` alone,
    growing 17x for an 8x payload. The same loop over a *local* string is
    linear, which is why a microbenchmark of `s += x` finds nothing and why
    this is asserted on the parser rather than on the idiom.
    """

    SMALL_KB = 32
    LARGE_KB = 128

    @staticmethod
    def _stream_ms(payload_bytes: int, parser=None, build=None) -> float:
        text = (build or _big_call)(payload_bytes)
        best = None
        for _ in range(3):
            stream = ToolCallStreamParser(
                tools=DECLARED_TOOLS, parser_cls=parser or QwenXmlParser
            )
            start = time.perf_counter()
            for i in range(0, len(text), 4):
                stream.process(text[i : i + 4])
            stream.flush()
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        return best * 1000

    @staticmethod
    def _control_ms(payload_bytes: int) -> float:
        """The same loop over a local string, which is linear by construction."""
        best = None
        for _ in range(3):
            start = time.perf_counter()
            buf = ""
            for _i in range(payload_bytes // 4):
                buf += "xxxx"
            best = (
                time.perf_counter() - start
                if best is None
                else min(best, time.perf_counter() - start)
            )
        return best * 1000

    @staticmethod
    def _scans_of_an_open_region(text, parser_cls=QwenXmlParser):
        """Every length the format was asked about while the region was open."""
        seen: list[int] = []

        class Watching(parser_cls):
            @classmethod
            def parse_region(cls, region, tools, *, at_end):
                if not at_end:
                    seen.append(len(region))
                return parser_cls.parse_region(region, tools, at_end=at_end)

        parser = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=Watching)
        for i in range(0, len(text), 4):
            parser.process(text[i : i + 4])
        parser.flush()
        return seen

    def test_the_open_region_is_never_scanned_beyond_the_window(self):
        """Nothing on the per-chunk path may be handed the whole region.

        Running a format's regex over the growing buffer once per chunk is
        quadratic in the response, and a probe that did it on a doubling
        schedule was tried and reverted -- see
        `TestTheDeclarationAboutACallInProgressIsMeasured`. So the bound is
        flat again: `Region.head`, and nothing larger, ever.
        """
        text = _big_call(8 * 1024)
        seen = self._scans_of_an_open_region(text)
        assert seen, "nothing asked; this asserts nothing"
        assert max(seen) <= _PEEK_WINDOW, (
            f"a scan of an open region was handed {max(seen)} characters; the "
            f"window is {_PEEK_WINDOW}"
        )

    @pytest.mark.parametrize(
        "parser, build",
        [
            (QwenXmlParser, _big_call),
            (KimiParser, _big_kimi_call),
        ],
        ids=["buffered-region", "kimi-incremental"],
    )
    def test_no_format_pays_more_per_byte_as_the_payload_grows(self, parser, build):
        """Kimi is the format that is not a `BufferedMarkerParser`, so the
        sweep that put the others on a list accumulator missed it -- and it
        carried a second factor of its own, re-scanning the whole buffer for
        an entry end on every chunk. 128 KB cost 428 ms of event-loop CPU
        against Qwen's 10 for the same payload.
        """
        control = self._control_ms(self.LARGE_KB * 1024) / (
            self._control_ms(self.SMALL_KB * 1024) * (self.LARGE_KB / self.SMALL_KB)
        )
        if not 0.6 < control < 1.6:
            pytest.skip(f"machine too noisy to measure: control ratio {control:.2f}")
        small = self._stream_ms(self.SMALL_KB * 1024, parser, build) / self.SMALL_KB
        large = self._stream_ms(self.LARGE_KB * 1024, parser, build) / self.LARGE_KB
        assert large / small < 1.5, (
            f"{parser.NAME}: cost per KB grew {large / small:.2f}x from "
            f"{self.SMALL_KB} to {self.LARGE_KB} KB ({small:.3f} -> {large:.3f})"
        )

    def test_the_cost_per_byte_does_not_grow(self):
        """The timed half, with a control arm.

        Two numbers being equal proves nothing on a shared machine unless
        something in the same run is known to move -- so the control is the
        linear loop, and its own per-byte cost has to come out flat before
        this measurement is allowed to mean anything.
        """
        control = self._control_ms(self.LARGE_KB * 1024) / (
            self._control_ms(self.SMALL_KB * 1024) * (self.LARGE_KB / self.SMALL_KB)
        )
        if not 0.6 < control < 1.6:
            pytest.skip(f"machine too noisy to measure: control ratio {control:.2f}")

        small = self._stream_ms(self.SMALL_KB * 1024) / self.SMALL_KB
        large = self._stream_ms(self.LARGE_KB * 1024) / self.LARGE_KB
        # Quadratic measured 1.75 across this pair; linear measures ~1.0.
        assert large / small < 1.5, (
            f"cost per KB grew {large / small:.2f}x from {self.SMALL_KB} KB to "
            f"{self.LARGE_KB} KB ({small:.3f} -> {large:.3f} ms/KB); the region "
            "is being copied per chunk again"
        )


class TestThePeekIsBounded:
    """Asking per token over a growing region is the shape this branch retired.

    The first version ran the format's regex over the whole buffer on every
    chunk: 3.0 -> 9.8 -> 36 -> 137 ms across 2k/4k/8k/16k tokens, quadratic,
    against a 1383 ns/token budget for the entire pipeline. Bounded to a
    prefix, and stopped once that prefix has gone by without a name.
    """

    @staticmethod
    def _count_peeks(parser_cls, text, tools):
        calls = []

        class Counting(parser_cls):
            @classmethod
            def parse_region(cls, region, tools, *, at_end):
                if not at_end:
                    calls.append(len(region))
                return parser_cls.parse_region(region, tools, at_end=at_end)

        stream = ToolCallStreamParser(parser_cls=Counting)
        stream.tools = tools
        for i in range(0, len(text), 4):
            stream.process(text[i : i + 4])
        stream.flush()
        return calls

    def test_the_announcement_never_sees_more_than_the_window(self):
        text = "The model writes <tool_call> to open one. " + "x" * 4000
        sizes = self._count_peeks(QwenXmlParser, text, DECLARED_TOOLS)
        assert sizes, "the peek never ran"
        assert (
            max(sizes) <= _PEEK_WINDOW
        ), f"the announcement was handed {max(sizes)} characters"

    def test_it_stops_once_the_window_has_gone_by_without_a_name(self):
        text = "The model writes <tool_call> to open one. " + "x" * 4000
        sizes = self._count_peeks(QwenXmlParser, text, DECLARED_TOOLS)
        # One per chunk until the region passes the window, then never again.
        assert len(sizes) <= _PEEK_WINDOW // 4 + 2, (
            f"{len(sizes)} peeks over a {len(text)}-character answer; the "
            "latch is not holding and the cost is quadratic again"
        )

    def test_an_undeclared_name_also_stops_it(self):
        """A name the request never offered is prose to every format's own
        truncated-call test, so nothing parses and the window runs out --
        the same latch, reached one step later than when the peek had its
        own name check."""
        other = [{"type": "function", "function": {"name": "something_else"}}]
        text = "Sure. " + REAL_CALLS["qwen"].replace("Paris", "x" * 4000)
        sizes = self._count_peeks(QwenXmlParser, text, other)
        assert len(sizes) <= _PEEK_WINDOW // 4 + 2, (
            f"{len(sizes)} peeks over a {len(text)}-character answer; the "
            "latch is not holding and the cost is quadratic again"
        )


class TestOneReaderPerFormat:
    """A format may not grow a second way of reading the stream.

    Everything this branch fixed twice over came from the same shape: a
    format read once by ``parse`` and again by a ``process``/``flush`` state
    machine of its own, with four rules -- where content ends, whether an
    unclosed tag is a call, what a region that parses to nothing means, which
    bytes are framing -- written out in both. The engine owns all four now,
    and a format that defines any of these names has taken one back.
    """

    FORBIDDEN = ("parse", "process", "flush", "peek_name")

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_it_defines_no_reader_of_its_own(self, parser):
        own = [name for name in self.FORBIDDEN if name in vars(parser)]
        assert not own, f"{parser.NAME} defines {own}; the engine owns those"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_it_holds_no_per_request_state(self, parser):
        """Class-side only, so one request cannot leak into the next and one
        region cannot leak into the one after it."""
        instance_attrs = [
            name
            for name, value in vars(parser).items()
            if not name.startswith("_")
            and not isinstance(value, (classmethod, staticmethod, property))
            and not name.isupper()
        ]
        assert not instance_attrs, f"{parser.NAME} carries {instance_attrs}"


class TestARegionIsAccountedForByteByByte:
    """What a region's bytes become, stated as a conservation law.

    `parse_region` reports two offsets, and everything outside them is the
    answer. Getting either wrong is silent: too small a `begins` leaves markup
    in the answer, too large swallows a sentence, and a `consumed` that does
    not advance hands the same region back forever.
    """

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_region_that_is_exactly_one_call_leaves_no_answer(self, parser):
        """The whole of this format's own call is markup, opener to closer."""
        call = REAL_CALLS[parser.NAME]
        content, calls = parse_tool_calls(call, DECLARED_TOOLS, parser)
        assert len(calls) == 1, f"{parser.NAME} did not parse its own call"
        assert content == "", f"{parser.NAME} left {content!r} in the answer"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_offsets_bracket_something(self, parser):
        """`begins < consumed <= len(region)` whenever a call was found.

        The lower bound is what stops the engine looping: it hands
        `region[consumed:]` back to be read again, so a `consumed` that did
        not advance would find the same marker and parse it again forever.
        """
        region = REAL_CALLS[parser.NAME]
        parsed = parser.parse_region(region, DECLARED_TOOLS, at_end=True)
        assert parsed.calls
        assert 0 <= parsed.begins < parsed.consumed <= len(region), (
            f"{parser.NAME}: begins={parsed.begins} consumed={parsed.consumed} "
            f"len={len(region)}"
        )

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_sentence_between_a_quotation_and_a_real_call_survives(self, parser):
        """A region opens at the *first* marker in the text, which an answer
        that quotes one before calling for real puts in the wrong place: the
        sentence in between is inside the region and is not markup.

        This is what `RegionParse.begins` is for. Reported as the first call's
        own match and widened back over the wrapper enclosing it, so it stops
        where the prose ends -- and not at the region's start, which would
        swallow the sentence, nor at the call's match, which would leave the
        wrapper in the answer.
        """
        marker = next(m for m in parser.START_MARKERS if parser.opens_region(m))
        prefix = f"Explaining: {marker} is how. Now: "
        text = prefix + REAL_CALLS[parser.NAME] + " done"
        content, calls = parse_tool_calls(text, DECLARED_TOOLS, parser)
        assert len(calls) == 1, f"{parser.NAME} did not find the real call"
        assert content == prefix + " done", f"{parser.NAME}: {content!r}"
        for size in (1, 5, len(text)):
            stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
            events = []
            for i in range(0, len(text), size):
                events += stream.process(text[i : i + size])
            events += stream.flush()
            streamed = "".join(d for k, d in events if k == "content")
            assert streamed == content, f"{parser.NAME} at chunk {size}: {streamed!r}"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_prose_around_a_call_survives_on_both_paths(self, parser):
        """Before and after, and the same either way it is delivered."""
        text = "Sure. " + REAL_CALLS[parser.NAME] + "\nAnything else?"
        content, calls = parse_tool_calls(text, DECLARED_TOOLS, parser)
        assert len(calls) == 1
        assert content == "Sure. \nAnything else?", content
        for size in (1, 3, 17, len(text)):
            stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
            events = []
            for i in range(0, len(text), size):
                events += stream.process(text[i : i + size])
            events += stream.flush()
            streamed = "".join(d for k, d in events if k == "content")
            assert streamed == content, f"{parser.NAME} at chunk {size}: {streamed!r}"


class TestNoSizeAtWhichACallStopsBeingOne:
    """A "give up on a region producing nothing after N bytes" probe was added
    here and reverted.

    It rests on acceptance being monotone in how many bytes have arrived, and
    that is false for three of the six formats -- MiniMax gates its
    in-progress test on the first tag being in the declared schema, and DSML's
    wrapper-less and direct-JSON branches match no prefix at all -- so real
    calls over the probe size were delivered as raw markup with
    `finish_reason: stop` while `stream=false` returned the call. It was also
    quadratic. These rows are what catches it coming back.
    """

    @staticmethod
    def _cut_inside_an_argument(parser) -> str:
        """This format's own call, cut off in the middle of a value.

        "One byte short of complete" is the wrong probe: a Kimi entry cut
        there is still whole and it is the *section* that is unterminated,
        while the case the flag is about is a single call whose argument is
        still arriving.
        """
        long_value = "x" * 4000
        call = REAL_CALLS[parser.NAME].replace("Paris", long_value)
        return call[: call.index(long_value) + 2000]

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_no_size_at_which_a_call_stops_being_one(self, parser):
        call = REAL_CALLS[parser.NAME].replace("Paris", "x" * 8192)
        stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
        events = []
        for i in range(0, len(call), 4):
            events += stream.process(call[i : i + 4])
        events += stream.flush()
        args = [d for k, d in events if k == "tool_call_args"]
        assert len(args) == 1, f"{parser.NAME} lost a {len(call)}-character call"
        assert "x" * 32 in args[0]["function"]["arguments"]
        assert (
            "".join(d for k, d in events if k == "content") == ""
        ), f"{parser.NAME} delivered part of a call as text"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_a_long_call_costs_what_it_is_and_not_its_square(self, parser):
        """The other half of why the give-up came out.

        Giving up re-fed bytes that immediately reopened a region with a fresh
        budget, and the probe re-ran a failing match over the whole growing
        region: 1.19 ms to 18.2 s on a 250 KB answer, synchronously inside the
        request coroutine, stalling every other in-flight request.
        """

        def ms(payload):
            text = REAL_CALLS[parser.NAME].replace("Paris", "x" * payload)
            best = None
            for _ in range(3):
                stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
                start = time.perf_counter()
                for i in range(0, len(text), 64):
                    stream.process(text[i : i + 64])
                stream.flush()
                took = time.perf_counter() - start
                best = took if best is None else min(best, took)
            return best * 1000

        small, large = ms(8 * 1024), ms(64 * 1024)
        assert large / max(small, 1e-6) < 16, (
            f"{parser.NAME}: 8x the payload cost {large / small:.1f}x the time "
            f"({small:.2f} -> {large:.2f} ms)"
        )


class TestAQuotedCallOpenerIsNotTheCallTheModelMade:
    """A format's call opener written in prose, before a real call.

    The non-greedy body ran from the quoted opener all the way to the real
    call's closer, so the client got one call named after the placeholder,
    carrying the real call's arguments, with the explanatory sentence deleted
    and `finish_reason: tool_calls`. `finditer` then resumed past the real
    call, so the call the model actually made never went out at all.

    A call's body cannot contain another opener -- that literal is what opens
    one -- and every format says so now. GLM was given the guard first, for a
    different shape; this is the sweep, and the row that keeps it swept.
    """

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_placeholder_never_becomes_the_call(self, parser):
        quoted = CALL_OPENERS[parser.NAME].replace("get_weather", "NAME")
        text = f"To call it you write {quoted} and then the arguments. Now: "
        text += REAL_CALLS[parser.NAME]
        content, calls = parse_tool_calls(text, DECLARED_TOOLS, parser)
        assert [c.function["name"] for c in calls] == ["get_weather"], (
            f"{parser.NAME} produced {[c.function['name'] for c in calls]} from an "
            "answer that quotes an opener before making one real call"
        )
        assert "and then the arguments" in content, f"{parser.NAME}: {content!r}"
        for size in (1, 6, len(text)):
            stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
            events = []
            for i in range(0, len(text), size):
                events += stream.process(text[i : i + size])
            events += stream.flush()
            assert [
                d["function"]["name"] for k, d in events if k == "tool_call_start"
            ] == ["get_weather"], f"{parser.NAME} at chunk {size}"
            assert (
                "".join(d for k, d in events if k == "content") == content
            ), f"{parser.NAME} at chunk {size}"


class TestAnUnclosedCallDoesNotSwallowTheNextOne:
    """A call the model forgot to close ends where the next one opens.

    Anchoring the unclosed alternative at end of *stream* meant a call
    followed by a second call matched neither alternative at the first opener,
    so the first call was skipped entirely and the client got the second one
    where the model had made two -- and a different one from the name already
    announced.
    """

    CLOSERS: ClassVar[dict[str, str]] = {
        "qwen": "</function></tool_call>",
        "glm": "</tool_call>",
        "dsml": f"</{_D}invoke></{_D}tool_calls>",
        "minimax": f"{_NS}</invoke>{_NS}</tool_call>",
        "kimi": "<|tool_calls_section_end|>",
        "kimi_k3": "<|close|>call",
    }

    def _unclosed_then_real(self, parser) -> str:
        first = REAL_CALLS[parser.NAME]
        closer = self.CLOSERS[parser.NAME]
        assert first.endswith(
            closer
        ), f"{parser.NAME}: corpus does not end on {closer!r}"
        return first[: -len(closer)] + first

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_complete_call_is_never_lost(self, parser):
        """The invariant every format holds. Whether the *unclosed* one is
        also recovered is per format -- Qwen, GLM and MiniMax read it, DSML
        and the two token formats do not, and never did -- but the complete
        one must always go out, and nothing may be invented.
        """
        text = self._unclosed_then_real(parser)
        _, calls = parse_tool_calls(text, DECLARED_TOOLS, parser)
        names = [c.function["name"] for c in calls]
        assert (
            "get_weather" in names
        ), f"{parser.NAME} lost the complete call behind an unclosed one: {names}"
        declared = {t["function"]["name"] for t in DECLARED_TOOLS}
        assert set(names) <= declared, f"{parser.NAME} invented {set(names) - declared}"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_and_the_two_delivery_modes_agree_about_how_many(self, parser):
        """However many that is, it is the same number either way. The
        anchoring bug made `stream=false` and `stream=true` disagree here."""
        text = self._unclosed_then_real(parser)
        _, calls = parse_tool_calls(text, DECLARED_TOOLS, parser)
        for size in (1, 9, len(text)):
            stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=parser)
            events = []
            for i in range(0, len(text), size):
                events += stream.process(text[i : i + size])
            events += stream.flush()
            streamed = [d for k, d in events if k == "tool_call_args"]
            assert len(streamed) == len(calls), (
                f"{parser.NAME} at chunk {size}: {len(streamed)} streamed vs "
                f"{len(calls)} not"
            )


class TestTheEarlyNameCannotDisagreeWithTheParse:
    """The property the whole announcement design rests on.

    The name is read out of `parse_region` over the region so far; the call
    that goes out is `parse_region` over the whole region. They agree iff
    acceptance is *monotone* in how many bytes have arrived -- if a prefix
    yields a first call named N, so does every longer prefix, and so does the
    finished region.

    Asserted by fuzzing rather than by argument, because the argument has been
    wrong before: a probe added last round rested on the same claim and it is
    false for three formats when the question is asked at `at_end=False` on a
    *region* rather than a prefix.
    """

    SHAPES: ClassVar[list[str]] = [
        "{call}",
        PROSE + "{call}",
        "{call}" + PROSE,
        "{call}{call}",
        PROSE + "{half}" + PROSE + "{call}",
        "{half}{call}",
    ]

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_no_prefix_of_a_head_ever_names_a_different_tool(self, parser):
        call = REAL_CALLS[parser.NAME]
        texts = [
            shape.format(call=call, half=call[: len(call) // 2])
            for shape in self.SHAPES
        ]
        texts.append(call.replace("Paris", "x" * 300))
        texts.append(call.replace("get_weather", "undeclared_tool"))
        for text in texts:
            head = text[:_PEEK_WINDOW]
            first = None
            for n in range(1, len(head) + 1):
                calls = parser.parse_region(
                    head[:n], DECLARED_TOOLS, at_end=False
                ).calls
                if not calls:
                    continue
                name = calls[0].function["name"]
                if first is None:
                    first = (n, name)
                else:
                    assert name == first[1], (
                        f"{parser.NAME}: {head[:first[0]]!r} names {first[1]!r} but "
                        f"{head[:n]!r} names {name!r}"
                    )
            if first is None:
                continue
            whole = parser.parse_region(text, DECLARED_TOOLS, at_end=True).calls
            if whole:
                assert whole[0].function["name"] == first[1], (
                    f"{parser.NAME}: announced {first[1]!r}, parsed "
                    f"{whole[0].function['name']!r} from {text[:80]!r}"
                )

    def test_and_a_format_that_breaks_it_leaves_no_index_behind(self):
        """The recovery, driven by a format built to break the property.

        A name that goes out and never gets arguments has still used its
        index. A later real call landing on the same one is merged into it by
        any accumulator that keys on index -- the OpenAI streaming contract --
        so the client ends up with a single call whose name is two names glued
        together and which no tool answers to.
        """

        class Fickle(QwenXmlParser):
            NAME = "fickle"

            @classmethod
            def parse_region(cls, region, tools, *, at_end):
                if not at_end and "<parameter=" in region:
                    call = ToolCall(
                        id="fickle",
                        type="function",
                        function={"name": "get_weather", "arguments": "{}"},
                    )
                    return RegionParse((call,), ((0, len(region)),))
                return RegionParse()

        text = "<tool_call><function=get_weather><parameter=city>Paris"
        stream = ToolCallStreamParser(tools=DECLARED_TOOLS, parser_cls=Fickle)
        events = []
        for i in range(0, len(text), 4):
            events += stream.process(text[i : i + 4])
        events += stream.flush()
        announced = [d["index"] for k, d in events if k == "tool_call_start"]
        assert announced, "the announcement is the premise of this test"
        assert stream._index > announced[-1], (
            "the index of a name that produced no call was not stepped past; a "
            "later call would land on it"
        )
