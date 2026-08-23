# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One reading of a model's tool-call syntax, driven by one engine.

A format used to be read twice: ``parse`` took a complete output and the
``process``/``flush`` state machine took it a chunk at a time. Both had to
decide where content ends, whether an unclosed tag is a call or a quotation,
what to do with a region that parses to nothing, and which bytes are framing --
so every one of those rules existed six times over, once per format, in two
copies. Three rounds of review found the copies disagreeing about all four,
and every disagreement is a response a client gets one way with
``stream=false`` and another with ``stream=true``.

So a format now declares only what is particular to it:

``START_MARKERS``     literals that must not be split across a chunk boundary
``opens_region``      which of those hand the stream over (the rest are framing)
``parse_region``      what one region's bytes mean -- the calls, and the two
                      offsets that bracket the format's own markup

and :class:`ToolCallStreamParser` -- the engine -- owns everything else:
reading ahead of the region, releasing content, dropping framing, the rule
that a start marker is not a promise, stamping call indices, and handing back
whatever followed the markup. ``stream=false`` is the engine run over a single
chunk, so the two modes cannot disagree: there is nothing left for them to
disagree with.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


def continues_a_call(rest: str, tokens: tuple[str, ...], *, arrived: bool) -> bool:
    """Is what follows a call's name this format's own next token?

    One implementation, because four formats had a copy and the copies were
    the point of failure: each format supplies the tokens, this decides.

    `arrived` is `parse_region`'s `at_end`, inverted, and it is the whole of
    the difference between the two moments that function is asked about. At
    end of region a token cut off part-way through is all there will ever be,
    so a *prefix* counts -- that is what a call truncated by `max_tokens`
    looks like. Mid-region, where the early name is read, a prefix means "not
    yet" and more is coming; accepting one there let a chunk boundary landing
    one character into `<br>` announce a tool for prose, since `<` is a prefix
    of `<parameter=`. Same text, same parser: announced at one chunk size,
    silent at another.
    """
    if arrived:
        return any(rest.startswith(tok) for tok in tokens)
    return any(tok.startswith(rest[: len(tok)]) for tok in tokens)


def declared_tools_allow(name: str, param_types: dict) -> bool:
    """Can the request's declared tools rule this name out?

    Only when it declared any. Every format's truncation gate asks whether the
    name is one the request listed, because prose quoting an opener usually
    invents one -- but a request with no ``tools`` field declares nothing, and
    six formats read that as "no name is real". A call cut off by
    `max_tokens` then went out as raw special tokens in ``content`` while the
    *same* generation, one byte longer, was read as a call: the answer turned
    on whether the client had listed its tools rather than on what the model
    wrote. Agent harnesses that describe their tools in the system prompt hit
    it on every truncation.

    So: a declared list refutes a name it does not contain, and an absent list
    refutes nothing. Whatever follower test the format applies still runs --
    that is the evidence which does not come from the request.
    """
    return not param_types or name in param_types


def unique_tool_call_id() -> str:
    # OpenAI tool_call ids must be unique across the whole conversation, not just
    # within one response. A per-response index (call_0, call_1, ...) collides
    # across turns -> clients (e.g. qwen-code) dedupe by id and silently ignore
    # every repeat, causing an infinite tool-call retry loop. Use a random id.
    return f"call_{uuid.uuid4().hex}"


def begin_of_markup(
    region: str,
    at: int,
    openers: tuple[str, ...] = (),
    fillers: tuple[str, ...] = (),
) -> int:
    """Pull `at` back over the wrapper a format opens its calls with.

    The mirror of :func:`end_of_markup`, and needed for the same reason: a
    call's own match starts at `<function=`, and the `<tool_call>` before it is
    markup too. Without it every real call left its wrapper in the answer.

    It also draws the line an answer that *quotes* a marker before making a
    real call depends on. A region opens at the first marker in the text, so
    the sentence between a quotation and the call that follows it is inside
    the region -- and it stops here, because prose is not one of the literals
    this walks over.
    """
    if not openers:
        return at
    begin = j = at
    while j > 0:
        stripped = region[:j].rstrip()
        if len(stripped) < j:
            # Whitespace moves the scan but not the answer, exactly as in
            # `end_of_markup`: a newline between two markers is markup, a
            # newline between prose and a marker belongs to the prose.
            j = len(stripped)
            continue
        token = next(
            (t for t in (*fillers, *openers) if stripped.endswith(t)),
            None,
        )
        if token is None:
            break
        j -= len(token)
        begin = j
    return begin


def end_of_markup(
    region: str,
    at: int,
    closers: tuple[str, ...] = (),
    fillers: tuple[str, ...] = (),
) -> int:
    """Extend `at` over the wrapper a format closes its calls with.

    A call's own match ends at `</function>`; the `</tool_call>` after it is
    markup too, and so is the newline between them. Without this the wrapper
    was either left in the answer or swallowed along with everything after it,
    depending on which format's ``parse`` you read.

    Whitespace and `fillers` (MiniMax repeats its namespace token before every
    tag) advance the *scan* but never the answer: only a closer moves `end`,
    so the newline a model writes before resuming prose survives.
    """
    if not closers:
        return at
    end = j = at
    n = len(region)
    while j < n:
        if region[j].isspace():
            j += 1
            continue
        skipped = next((s for s in fillers if region.startswith(s, j)), None)
        if skipped is not None:
            j += len(skipped)
            continue
        closer = next((c for c in closers if region.startswith(c, j)), None)
        if closer is None:
            break
        j += len(closer)
        end = j
    return end


@dataclass
class ToolCall:
    """Parsed tool call in OpenAI format."""

    id: str
    type: str
    function: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function}


@dataclass(frozen=True)
class RegionParse:
    """What one format made of a region's bytes.

    ``spans`` is every stretch of this region that is the format's own markup,
    in ascending order and not overlapping: everything inside a span is
    markup, everything outside every span is answer. The region is accounted
    for byte by byte and no format has to remember to return the leftovers.

    Several spans and not one outer pair, because a region can hold more than
    one call and a model writes between them -- "let me also check Rome"
    between two `<tool_call>` blocks is the ordinary shape of a parallel-call
    answer. A single outer pair calls that sentence markup and deletes it. Kimi-K2 is
    the exception, and only because it is the one format whose
    `REGION_END_MARKERS` close a region per section, so the question never
    reached here -- which is also why it still returns a single span covering
    its whole section, where the others return one per call.

    Each span is a call's own match widened by :meth:`ToolCallParser.
    markup_begin` / :meth:`~ToolCallParser.markup_end`, which walk over that
    format's wrappers and fillers and stop at prose -- so the wrapper opening
    the first call and the one closing the last belong to those calls' spans,
    and whitespace between two markers is markup while whitespace between
    prose and a marker is not.

    ``calls`` empty means the region was a quotation rather than a call, and
    the engine releases its bytes unchanged; ``spans`` is then ignored.
    """

    calls: tuple[ToolCall, ...] = ()
    spans: tuple[tuple[int, int], ...] = ()

    @property
    def begins(self) -> int:
        """Where the first call's markup starts. Derived, so that the answer
        cannot drift from ``spans``."""
        return self.spans[0][0] if self.spans else 0

    @property
    def consumed(self) -> int:
        """Where the last call's markup ends."""
        return self.spans[-1][1] if self.spans else 0


NO_CALLS = RegionParse()


class ToolCallParser(ABC):
    """One on-the-wire tool-call format.

    Entirely class-side. A format holds no per-request state -- the engine
    does -- so there is nothing here for two requests to share or for one
    region to leak into the next.
    """

    NAME: ClassVar[str]
    # Every literal that opens this format's tool-call region, plus every
    # literal it treats as framing. Declared once, here, rather than spelled
    # out again in each parser's own logic: the read-ahead needs them to know
    # how much of its buffer could still be the start of one, and the property
    # tests enumerate them so a newly registered format is covered the moment
    # it exists rather than when someone writes a case.
    START_MARKERS: ClassVar[tuple[str, ...]] = ()
    # Literals that can close a region mid-stream. Declared so the engine can
    # tell, from the incoming chunk alone, whether it is worth materialising
    # the buffer to ask `region_end` -- the check runs once per chunk on every
    # stream, and materialising to answer it is the quadratic this class's
    # `Region` exists to avoid. Empty means "this format's regions close only
    # at end of stream", which is what four of the six do.
    REGION_END_MARKERS: ClassVar[tuple[str, ...]] = ()
    # The wrapper literals this format writes around its calls -- before the
    # first and after the last -- and any token it repeats between tags. All
    # markup; see `begin_of_markup` and `end_of_markup`.
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ()
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ()
    CALL_FILLERS: ClassVar[tuple[str, ...]] = ()
    # The literals that *identify* this format, where that is narrower than
    # "what must not be split". Empty means the two coincide, as they do for
    # five of the six.
    #
    # They came apart because DSML matches its `｜DSML｜` marker optionally --
    # the V4-Flash malform drops it -- so `<invoke name=` is a start marker,
    # and that is also how MiniMax opens a call. DSML claimed MiniMax's
    # templates and only `_DETECT_ORDER` kept it from winning them.
    #
    # Leniency about what to *parse* stays: once identified, a format should
    # read the malforms its own model emits. Only the claim must be single.
    DETECT_MARKERS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def opens_region(cls, marker: str) -> bool:
        """Does this marker hand the rest of the stream to this format?

        `START_MARKERS` answers "which literals must not be split across a
        chunk boundary", and for most formats every one of them also opens a
        tool-call region, so the two questions have the same answer and this
        is that default. Kimi-K3 is where they come apart: three of its five
        are channel framing that wraps *every* answer, tool call or not, and
        treating those as a handover stopped it streaming at all.

        A marker this says no to is framing: the reader drops it and carries
        on. That is the only place framing is removed, on either delivery
        path -- Kimi-K3 used to strip its own list a second time inside
        ``parse``, and the two lists disagreed about four tokens.
        """
        return True

    @classmethod
    def markup_begin(cls, region: str, at: int) -> int:
        """Where this format's markup really starts, given its first call
        started at ``at``. See :func:`begin_of_markup`."""
        return begin_of_markup(region, at, cls.CALL_OPENERS, cls.CALL_FILLERS)

    @classmethod
    def markup_end(cls, region: str, at: int) -> int:
        """Where this format's markup really ends, given its last call ended
        at ``at``. See :func:`end_of_markup`."""
        return end_of_markup(region, at, cls.CALL_CLOSERS, cls.CALL_FILLERS)

    @classmethod
    def find_start(cls, text: str) -> int:
        """Index of the earliest start marker, or -1."""
        positions = [i for i in (text.find(m) for m in cls.START_MARKERS) if i != -1]
        return min(positions) if positions else -1

    @classmethod
    def detect(cls, text: str) -> bool:
        """Is this text this format's, as opposed to merely readable by it?

        Keyed on `DETECT_MARKERS` where a format declares them: the literals
        that must not be split are not always the literals that identify.
        """
        return any(m in text for m in (cls.DETECT_MARKERS or cls.START_MARKERS))

    @classmethod
    def region_end(cls, region: str) -> int:
        """How many bytes of ``region`` form a closed unit, or 0 for "not yet".

        Asked only after one of ``REGION_END_MARKERS`` has arrived. A format
        that answers lets the engine emit its calls and go back to reading the
        answer without waiting for end of stream; one that does not is
        buffered to EOS, which is correct but silent for as long as the region
        runs.

        Answer only on a literal that cannot appear inside an argument value.
        Kimi's section end is a special token and qualifies; the XML formats'
        `</tool_call>` does not, because a model writing *about* tool calls
        puts one inside a parameter.
        """
        return 0

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        """One call, in this format's own syntax. The inverse of `parse_region`.

        Nothing in the engine renders; this exists so the test corpus can be
        generated from the registry rather than hand-written. A hand-written
        sample can be wrong in a way that makes every assertion about it
        vacuous -- MiniMax's was written in DSML's spelling and parsed to
        `get_weather({})`, exercising none of the parameter path.

        Not `@abstractmethod`: nothing here is ever instantiated, so ABC would
        not enforce it. `test_every_format_can_write_its_own_syntax` does.
        """
        raise NotImplementedError(f"{cls.NAME} cannot render its own syntax")

    @classmethod
    @abstractmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        """What this region's bytes mean.

        It sees only the region -- never the content before it, which the
        engine has already released, and never the answer after it, which it
        reports the end of rather than consuming.

        Returning no calls says the region was a quotation: an answer
        explaining that a model "writes ``<tool_call>`` to call something"
        opens one and never closes it. The engine then releases the bytes
        unchanged, so a start marker is not a promise -- a rule that used to
        be written out in each format, and was missing from two of them.

        Where a call's own pattern *ends* is part of that meaning, and it has
        five terminators, not four: its closing tag, the next parameter, the
        end of the call, end of input, and **the next call's opener**. Four of
        the six formats were missing the last, so a call cut off mid-value ran
        on into the sibling behind it and the tool was invoked with markup as
        data -- ``city="Par<tool_call>"``. The call count stayed right either
        way, which is what hid it.

        ``at_end`` is whether more bytes can still arrive for this region, and
        it is the *only* difference between the two questions this used to be
        asked twice. With ``at_end`` a token cut off part-way through is all
        there will ever be, so a prefix of one counts -- that is what a call
        truncated by ``max_tokens`` looks like. Without it a prefix means "not
        yet", and accepting one let a chunk boundary landing one character
        into ``<br>`` name a tool for prose.

        The engine calls this with ``at_end=False`` to find the name to
        announce early. That is what makes the early name and the parsed call
        agree by construction: they are the same enumeration of the same
        function over a prefix and then the whole. There used to be a separate
        ``peek_name`` per format, and its disagreements with this were
        the single most repeated defect in this module -- most recently a
        DeepSeek-V4 response whose peek skipped a self-closing
        ``<invoke name="x"/>`` that the parse returned first, putting three
        tool calls on the wire where the model had written two.
        """
