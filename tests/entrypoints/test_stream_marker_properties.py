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
import pathlib
import random

import pytest

from atom.entrypoints.openai.reasoning import ReasoningFilter
from atom.entrypoints.openai.reasoning_dialects import DIALECTS
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.registry import _DETECT_ORDER, _SNIFF_ONLY
from atom.entrypoints.openai.tool_parser.stream import ToolCallStreamParser

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


def drive(text: str, chunks: list[str]) -> Seen:
    """Replay the serving loop over one chunking."""
    rf, tp = ReasoningFilter(), ToolCallStreamParser()
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
    # A literal the sniffer picks the format by that is *not* a region
    # opener: the parser is chosen and then handed text its own
    # `START_MARKERS` do not match, so it reads on in content mode. That is
    # the path whose holdback used to stop emitting entirely once its buffer
    # began with a marker character. Generated from the difference between
    # the two declarations, so a future format with the same split is covered.
    if _SNIFF_ONLY:
        out["detected by a literal that opens no region"] = (
            _SNIFF_ONLY[0] + "city</arg_key> then a < b and " + PROSE * 4
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
NOTHING_TO_HONOUR = (
    "trigger chars in ordinary prose",
    "a marker quoted inside the answer",
)


def _hold_pairs():
    for dialect in DIALECTS:
        for parser in ALL_PARSERS:
            for shape_name, text in shapes(dialect, parser).items():
                if shape_name not in NOTHING_TO_HONOUR:
                    continue
                yield pytest.param(
                    dialect,
                    parser,
                    text,
                    id=f"{dialect.think_end_marker.strip('<>/|')}-{parser.NAME}-"
                    f"{shape_name.replace(' ', '_')}",
                )


HOLD_PAIRS = list(_hold_pairs())


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
                if not any(kw.arg == "starts_thinking" for kw in node.keywords):
                    rel = path.relative_to(self.ROOT.parents[1])
                    found.append(f"{rel}:{node.lineno} {name}(...)")
        return found

    def test_no_entry_point_builds_one_without_the_seed(self):
        unseeded = self._unseeded()
        assert not unseeded, "sites that never say where reasoning starts:\n  " + (
            "\n  ".join(unseeded)
        )

    def test_the_scan_can_actually_fail(self):
        """A source scan that matches nothing would pass forever."""
        tree = ast.parse("ReasoningFilter()\nseparate_reasoning(x)\n")
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert len(calls) == 2
        assert all(
            not any(kw.arg == "starts_thinking" for kw in c.keywords) for c in calls
        )


class TestChunkInvariance:
    """Where the token boundaries fall must not change what the client sees."""

    @pytest.mark.parametrize("dialect, parser, text, split_may_move", PAIRS)
    def test_the_same_text_split_differently_reads_the_same(
        self, dialect, parser, text, split_may_move
    ):
        by_split = {
            label: drive(text, chunks)
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


class TestBoundedWithhold:
    """Text must not be held back longer than a marker could justify."""

    @pytest.mark.parametrize("dialect, parser, text", HOLD_PAIRS)
    def test_a_trigger_character_does_not_hold_the_rest_of_the_answer(
        self, dialect, parser, text
    ):
        triggers = trigger_chars(parser.START_MARKERS, dialect_markers(dialect))
        control = defuse(text, triggers)

        fine = split_every_way(text)["fixed-1"]
        seen = drive(text, fine)
        if not seen.content:
            pytest.skip("no content channel in this shape; nothing to hold back")

        ctl = drive(control, split_every_way(control)["fixed-1"])
        held = seen.first_content_at or len(text)
        baseline = ctl.first_content_at or len(control)
        assert held - baseline <= SLACK, (
            f"first content byte at input offset {held}/{len(text)}, against "
            f"{baseline} for the same text with {sorted(triggers)} neutralised"
        )
