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
from atom.entrypoints.openai.serving_anthropic import completes_a_tool_call
from atom.entrypoints.openai.tool_parser.kimi_tool_parser import KimiParser
from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser
from atom.entrypoints.openai.tool_parser.registry import _DETECT_ORDER
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


def drive(text: str, chunks: list[str], parser=None) -> Seen:
    """Replay the serving loop over one chunking.

    `parser` is what the server resolved from the chat template at startup, so
    each case reads its own format explicitly rather than relying on the shape
    of the text to select one.
    """
    rf, tp = ReasoningFilter(), ToolCallStreamParser(parser_cls=parser)
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
        non_streaming, calls = parser.parse(text, None)
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
        content, calls = parser.parse(text, None)
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
    "minimax": (
        f'{_NS}<tool_call>{_NS}<invoke name="get_weather">'
        f'{_NS}<parameter name="city">Paris</{_NS}parameter>'
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
        _, calls = parser.parse(REAL_CALLS[parser.NAME], None)
        assert [c.function["name"] for c in calls] == ["get_weather"]

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


# Markers a format declares so the read-ahead will not split them, but which
# do not hand the stream over: channel framing that wraps every answer. Only
# Kimi-K3 has any; a format absent here declares none, which is the default.
FRAMING_NOT_A_REGION: dict[str, set[str]] = {
    "kimi_k3": {
        "<|open|>response<|sep|>",
        "<|close|>response<|sep|>",
        "<|end_of_msg|>",
    },
}


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
    def test_a_declared_tool_is_named_early(self, parser):
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
        """Truncated mid-call, against the same input with nothing declared.

        Compared arm to arm rather than asserted outright: GLM's unterminated
        branch salvages `<tool_call>get_weather` into a call with empty
        arguments all by itself, which predates announcing and is that
        format's own business. What must hold is that announcing adds no call
        the parse did not already find.
        """
        head = REAL_CALLS[parser.NAME].split("get_weather")[0] + "get_weather"
        text = "Sure. " + head
        kinds = [k for k, _ in self._drive(parser, text)]
        baseline = [k for k, _ in self._drive_without_tools(parser, text)]
        if "tool_call_args" in baseline:
            pytest.skip("this format salvages a call from this much on its own")
        assert (
            "tool_call_args" not in kinds
        ), "arguments were invented for a call that never parsed"
        assert not completes_a_tool_call(
            self._drive(parser, text)
        ), "a name with no arguments must not report as a usable tool call"

    @pytest.mark.parametrize("parser", ALL_PARSERS, ids=lambda p: p.NAME)
    def test_the_answer_still_arrives_when_the_call_does_not(self, parser):
        head = REAL_CALLS[parser.NAME].split("get_weather")[0] + "get_weather"
        events = self._drive(parser, "Sure. " + head)
        delivered = "".join(d for k, d in events if k == "content")
        assert "Sure." in delivered, "the text before the call was dropped"

    def test_peeking_a_different_name_than_the_parse_is_raised(self):
        """`peek_name` and `parse` disagreeing about the same bytes is a bug
        in that format, and the name is already on the wire. Loud, not
        smoothed over -- there is nothing to smooth it into."""

        class Liar(QwenXmlParser):
            @classmethod
            def peek_name(cls, region):
                return "get_weather" if "<function=" in region else None

            @classmethod
            def parse(cls, text, tools):
                content, calls = QwenXmlParser.parse(text, tools)
                for c in calls:
                    c.function["name"] = "something_else"
                return content, calls

        with pytest.raises(AssertionError, match="announced"):
            self._drive(Liar, "Sure. " + REAL_CALLS["qwen"])
