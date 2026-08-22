# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The engine: one reader of a model's output, for both delivery modes.

A format (:class:`~.tool_parser.ToolCallParser`) says what its own bytes mean.
Everything a reader has to decide that is *not* particular to a format is
decided here, once: how far ahead of a region it is safe to release, which
markers are framing and get dropped, what a region that parses to no call
means, which index a call is stamped with, and what happens to the answer that
follows the markup.

`stream=false` is :func:`read_whole`, which is this engine over a single chunk.
That is the whole of why the two delivery modes agree -- not a test that
compares them, but no second implementation to compare against.
"""

import logging

from ..marker_scanner import MarkerScanner
from .schema import ParamTypes, build_param_types
from .tool_parser import RegionParse, ToolCall, ToolCallParser

logger = logging.getLogger("atom")

# How far into a region a name may be before the peek gives up. Every
# format on this box puts it in the first 30-70 characters; the margin is
# for a long name or leading whitespace, and the bound is what keeps the
# peek from re-scanning a growing buffer once per token.
_PEEK_WINDOW = 256

# Do not add a "give up on a region that has produced nothing after N bytes"
# probe. It was tried and reverted: it rests on acceptance being monotone in
# how many bytes have arrived, and that is false -- MiniMax gates its in-progress test on the
# first tag being in the declared schema, and DSML's wrapper-less and
# direct-JSON branches cannot match any prefix at all -- so real calls over
# N bytes were delivered as text, differently from `stream=false`, on three of
# the six formats. It was also quadratic, because giving up re-feeds bytes that
# immediately reopen a region with a fresh budget: 1.19 ms -> 18.2 s on a
# 250 KB answer, in the request coroutine, stalling the event loop for
# everything else.
#
# Fixing the latency needs the *format* to say "this can no longer become a
# call", which is a different question from "does not parse yet" and one no
# format answers today.


class Region:
    """The bytes buffered since a tool-call region opened.

    A list and a join, not `self.buf += text`. Appending to a *string
    attribute* is quadratic in CPython: the instance dict holds a reference,
    so the in-place fast path never applies and every chunk copies the whole
    buffer. Measured on a 128 KB tool call, streamed four characters at a
    time: 23 ms of event-loop CPU in `process` alone, growing 17x for an 8x
    payload while `flush` stayed linear. The same loop over a *local* string
    is linear, which is why a microbenchmark of `s += x` finds nothing.

    `head` is the first `_PEEK_WINDOW` bytes, kept separately so the early
    name announcement never needs the region materialised -- it only ever
    looks that far in, and handing it the whole buffer is how the quadratic
    scan it already fixed once could come back.
    """

    __slots__ = ("_len", "_parts", "head")

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0
        self.head = ""

    def append(self, text: str) -> None:
        if not text:
            return
        self._parts.append(text)
        self._len += len(text)
        if len(self.head) < _PEEK_WINDOW:
            self.head = (self.head + text)[:_PEEK_WINDOW]

    def text(self) -> str:
        """Everything buffered, without giving it up.

        Collapses the parts as a side effect, so asking twice costs one join
        rather than two. Only the region-close probe asks, and only once a
        literal that could close the region has actually arrived.
        """
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]
        return self._parts[0] if self._parts else ""

    def take(self) -> str:
        """Everything buffered, and start again."""
        out = self.text()
        self._parts.clear()
        self._len = 0
        self.head = ""
        return out

    def __len__(self) -> int:
        return self._len

    def __bool__(self) -> bool:
        return self._len > 0


def _peek_declared(tools: list | None) -> frozenset[str]:
    return frozenset(build_param_types(tools))


def _markup_spans(
    spans: tuple[tuple[int, int], ...], body: str
) -> list[tuple[int, int]]:
    """A format's markup intervals, clamped and merged into a partition.

    The engine walks these in order and treats the gaps as answer, so they
    have to be ascending and non-overlapping whatever a format reports.

    Two spans separated by nothing but whitespace are joined. That whitespace
    is the template's, not the model's: `end_of_markup` moves only on a closer
    and `begin_of_markup` only on an opener, which is right at the edge of a
    region -- the newline before the model resumes prose belongs to the prose
    -- and wrong between two calls, where every one of these chat templates
    renders a separator. Left in the gap it became a `content: "\n"` delta
    sitting between two `tool_calls` deltas, and on `/v1/messages` a whole
    text block containing one newline.

    Every span kept here is non-empty, so the engine always consumes at least
    one byte and cannot hand a region back, find the same marker and parse it
    forever. A format that reports only degenerate spans would break that, so
    it is refused rather than papered over -- the caller falls back to
    releasing the region unchanged.
    """
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(spans):
        start, stop = max(0, min(start, len(body))), max(0, min(stop, len(body)))
        if stop <= start:
            continue
        if merged and not body[merged[-1][1] : start].strip():
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _resolved_tools(tools: list | None):
    """The request's tools as the parsers want them: built once, not per chunk.

    Falls back to whatever was passed when nothing usable comes out, so a
    caller's truthiness test sees the same answer either way.
    """
    if tools is None or isinstance(tools, ParamTypes):
        return tools
    return build_param_types(tools) or tools


class ToolCallStreamParser:
    """Reads one request's output, in chunks or all at once.

    Emits ``(event_type, data)`` tuples:

    - ``("content", text)`` -- answer text, in the order the model wrote it
    - ``("tool_call_start", {"index", "id", "type", "function": {"name", ...}})``
    - ``("tool_call_args", {"index", "function": {"arguments": ...}})``
    - ``("tool_call_end", None)`` -- a region's calls are complete

    ``parser_cls`` is the format this model was resolved to at startup, from
    its chat template (`registry.resolve_from_prompt`). ``None`` means no
    registered format recognised it: everything is emitted as content and
    nothing is parsed, which the server announced at startup. It is
    deliberately not a fallback to guessing -- the guess this replaces
    mis-read a Hermes `<tool_call>{...}` as GLM and delivered the whole JSON
    blob as the tool's *name*.

    ``suppress_calls`` is ``tool_choice: "none"``. The format is still read --
    that is the difference from ``parser_cls=None``, and it matters, because a
    format whose framing wraps *every* answer would otherwise leak that
    framing to the client the moment a request forbade tool calls. What is
    suppressed is dispatch: a region's bytes are released as the prose the
    request said they must be.

    ``tools`` enables JSON-Schema type coercion of parameter values. It may be
    assigned after construction (several call sites do) and is read when a
    region closes, so it takes effect as long as it is set before then. It is
    resolved to :class:`~.schema.ParamTypes` on the way in -- the catalogue is
    walked once per request rather than once per chunk. A request whose tools
    declare no usable name keeps the list it was given, so that ``not
    self.tools`` still asks what it always asked.
    """

    def __init__(
        self,
        tools: list | None = None,
        parser_cls: type[ToolCallParser] | None = None,
        *,
        suppress_calls: bool = False,
    ) -> None:
        self.tools = tools  # type: ignore[assignment]
        self.parser_cls = parser_cls
        self.suppress_calls = suppress_calls
        # Reads ahead of the region: releases everything that cannot begin one
        # of `parser_cls`'s own markers, and drops the ones that are framing.
        # One scanner, not one here and another on the format -- two readers
        # of the same marker set is how framing came to be dropped before a
        # region and kept inside one.
        self._scanner = (
            MarkerScanner(parser_cls.START_MARKERS) if parser_cls is not None else None
        )
        self._region: Region | None = None

        # The tail a region-closing literal split across a chunk boundary
        # would need, and nothing more.
        self._end_markers = parser_cls.REGION_END_MARKERS if parser_cls else ()
        self._end_tail_len = max((len(m) for m in self._end_markers), default=1) - 1
        self._end_tail = ""
        # Stamped by the engine, so no format has to. Kimi took its index off
        # the wire, where every section restarts at 0, and a client
        # accumulating arguments by index merged two calls into one.
        self._index = 0
        self._emitted = 0
        # The name already sent for the region being buffered, if any.
        self._announced: str | None = None
        # Set once no name can still turn up, so the peek stops running per
        # token over a region that will never yield one.
        self._peek_exhausted = False
        # The request's declared names, built once rather than per token.
        self._declared: frozenset[str] | None = None

    @property
    def tools(self):
        """The request's tools, resolved once (see :func:`_resolved_tools`).

        A property and not a line in ``__init__`` because several call sites
        assign it afterwards, and the resolution has to happen for those too
        or they are the slow path again.
        """
        return self._tools

    @tools.setter
    def tools(self, value: list | None) -> None:
        self._tools = _resolved_tools(value)
        self._declared = None

    @property
    def fmt(self) -> str | None:
        """The format this stream is being read as, or None."""
        return self.parser_cls.NAME if self.parser_cls is not None else None

    # -- public ------------------------------------------------------------
    def process(self, text: str) -> list:
        """Consume one chunk; return the events it completed."""
        if self.parser_cls is None:
            return [("content", text)] if text else []
        return self._pump(text, final=False)

    def flush(self) -> list:
        """End of stream: nothing more is coming, so nothing is held back."""
        if self.parser_cls is None:
            return []
        return self._pump("", final=True)

    # -- the loop ----------------------------------------------------------
    def _pump(self, text: str, *, final: bool) -> list:
        """Read `text`, and whatever a closing region hands back, to a stop.

        The one loop. Content before a region, framing, the region itself and
        the answer after it are all reached from here, so "what happens to
        these bytes" has a single answer per byte rather than one per format
        per delivery mode.
        """
        out: list = []
        while True:
            if self._region is None:
                if text:
                    scan = self._scanner.feed(text)
                    text = ""
                    if scan.released:
                        out.append(("content", scan.released))
                    if scan.hit is not None:
                        if not self.parser_cls.opens_region(scan.hit):
                            # Framing this format wraps every answer in.
                            # Dropping it and carrying on is what lets such a
                            # format stream at all: treating it as a handover
                            # meant a Kimi-K3 answer, which opens with
                            # `<|open|>response<|sep|>`, buffered its entire
                            # body to EOS -- 324 of 324 characters in one
                            # frame at the end.
                            text = scan.rest
                        else:
                            # The region has opened. Opened, but not read
                            # here: the marker goes back through the region
                            # branch below, so a chunk carrying a whole call
                            # is closed on this pass rather than one late.
                            #
                            # Not skipped for `suppress_calls`: releasing the
                            # marker as content there let the rest of the
                            # region stream out as content too, putting raw
                            # wire tokens in the answer. A region is buffered
                            # the same way whatever the request said about
                            # dispatch; only `_close_region` differs.
                            self._region = Region()
                            self._end_tail = ""
                            text = scan.hit + scan.rest
                        continue
                if not final:
                    return out
                held = self._scanner.flush()
                if held:
                    out.append(("content", held))
                return out

            probe = ""
            if text:
                if self._end_markers:
                    probe = self._end_tail + text
                    self._end_tail = (
                        probe[-self._end_tail_len :] if self._end_tail_len else ""
                    )
                self._region.append(text)
                text = ""
                out.extend(self._announce())
            end = len(self._region) if final else self._closed_len(probe)
            if end <= 0:
                return out
            body = self._region.take()
            self._region = None
            self._end_tail = ""
            events, rest = self._close_region(body[:end])
            out.extend(events)
            text = rest + body[end:]

    def _closed_len(self, probe: str) -> int:
        """Whether the open region has closed, without materialising it.

        The probe is the literal itself: nothing in a chunk can close a region
        unless one of `REGION_END_MARKERS` lands in it, so until one does
        there is nothing to look at. Asking the format instead would mean
        joining the buffer once per chunk, which is quadratic in the region --
        measured at 55 ms of event-loop CPU on a 50 KB argument against 4 ms
        for the same payload in a format that did not do it.

        `probe` is the incoming chunk with the tail a marker split across the
        boundary would need, so a section end arriving one character at a time
        is still seen.
        """
        if not probe or not any(m in probe for m in self._end_markers):
            return 0
        return self.parser_cls.region_end(self._region.text())

    def _close_region(self, body: str) -> tuple[list, str]:
        """One region's events, and the bytes that were not its markup.

        The announcement is per region and is settled here either way: bound
        to the region's first call, or dropped. Carrying it past the region it
        was made in is how one region's peek came to be matched against the
        next region's parse, and reported as a mismatch.
        """
        parsed = self.parser_cls.parse_region(body, self.tools, at_end=True)
        if self.suppress_calls and parsed.calls:
            # `tool_choice: "none"`. The format is read either way -- that is
            # what separates this from `parser_cls=None` -- and what is
            # suppressed is dispatch, not reading. Skipping the parse instead
            # put the model's raw wire tokens in `content`, on every format
            # and both delivery paths. The markup is now located exactly as it
            # is for a dispatched call and only the answer around it goes out,
            # so a response that was nothing but a call has empty content.
            logger.debug(
                "tool_choice=none: dropped %d %s call(s) from the response",
                len(parsed.calls),
                self.parser_cls.NAME,
            )
            parsed = RegionParse((), parsed.spans)
        events: list = []
        rest = ""
        # Everything outside the format's markup is answer, including whatever
        # the model wrote *between* two calls. Reading only the outer pair
        # deleted that -- "let me also check Rome" between two `<tool_call>`
        # blocks, on five of the six formats, silently.
        #
        # `spans` empty with calls present would mean a format reported
        # nothing this engine can consume, and consuming nothing loops
        # forever. No registered format can do it -- 1.2M fuzzed regions
        # produced zero -- so this is the boundary check for the seventh, and
        # it fails towards releasing the region rather than towards eating a
        # byte of it.
        spans = _markup_spans(parsed.spans, body)
        if parsed.calls and not spans:
            logger.warning(
                "%s reported %d call(s) with no usable markup span in %d bytes; "
                "releasing the region as text",
                self.parser_cls.NAME,
                len(parsed.calls),
                len(body),
            )
        if spans and (parsed.calls or self.suppress_calls):
            pending = list(parsed.calls)
            at = 0
            for i, (start, stop) in enumerate(spans):
                if start > at:
                    events.append(("content", body[at:start]))
                at = stop
                # In the order the model wrote them, so a client sees the
                # sentence between two calls between them and not hoisted in
                # front of both. One call per span, and the last span takes
                # whatever is left -- which is how Kimi-K2's single
                # section-wide span carries all of its entries.
                take = len(pending) if i == len(spans) - 1 else 1
                for call in pending[:take]:
                    events.extend(self._emit_call(call))
                del pending[:take]
            for call in pending:
                events.extend(self._emit_call(call))
            if parsed.calls:
                # Not for a suppressed region: there is no call to end, and a
                # bare `tool_call_end` is read downstream as "a call just
                # completed" -- `completes_a_tool_call` says so, and the
                # Anthropic path closes a block on it.
                events.append(("tool_call_end", None))
            rest = body[at:]
        elif body:
            # A start marker is not a promise. An answer explaining that a
            # model "writes <tool_call> to call something" opens the region
            # and never closes it, and this used to drop everything from the
            # marker on -- no event, no error, `finish_reason` still `stop`.
            # Fifty characters of eighty-two, on the shapes measured.
            #
            # Released verbatim, and released here rather than handed back:
            # handing it back would find the same marker again.
            #
            # A name may already have gone out, and there is no retracting it.
            # What there is: no arguments follow, so nothing downstream counts
            # this as a usable call and `finish_reason` stays `stop`. The text
            # is still delivered -- the promise costs a dangling name, not the
            # answer.
            events.append(("content", body))
        if self._announced is not None and not parsed.calls:
            # The name went out and nothing will follow it. Step past the index
            # it was sent at, exactly as `_start_event` does when it drops one:
            # a later call landing on an index the client has already bound to
            # a name is merged into it by any accumulator that keys on index,
            # which is the OpenAI streaming contract.
            #
            # Not reachable while every format's acceptance is monotone in how
            # many bytes have arrived -- an announcement is the first call of
            # `parse_region` over a prefix of these same bytes. That property
            # is fuzzed rather than assumed
            # (`TestTheEarlyNameCannotDisagreeWithTheParse`), and this is what
            # the wire looks like if it ever stops holding.
            self._index += 1
        self._announced = None
        self._peek_exhausted = False
        return events, rest

    # -- events ------------------------------------------------------------
    def _announce(self) -> list:
        """Send the tool's name as soon as the region reveals it, once.

        The name is read out of ``parse_region`` itself, over the region so
        far and with ``at_end=False``. That is what makes it agree with the
        call that eventually goes out: same function, same enumeration, the
        second run seeing a superset of the first's bytes. Every format used
        to answer this with a regex of its own, and four of the five had it
        disagree with their parse -- most recently over a self-closing
        ``<invoke name="x"/>`` the peek skipped and the parse returned first,
        which put three tool calls on the wire for a response containing two.

        Only for a name the request actually declared. That check is what
        makes an early name safe to send: it cannot be taken back, and an
        answer merely quoting `<tool_call><function=NAME>` opens a region too.
        A name the client never offered is overwhelmingly likelier to be prose
        than a call, so it waits for the region to close like everything else.
        SGLang's cursor parsers announce with no such check and will emit a
        call named after whatever follows the tag.

        Asked of `Region.head` -- the first `_PEEK_WINDOW` bytes and no more
        -- and asked at most once more after it fills. Running a format's
        regex over the whole region on every chunk is quadratic in the
        response and measured 3.0 -> 9.8 -> 36 -> 137 ms across 2k/4k/8k/16k
        tokens: the shape `marker_scanner` exists to retire, put back one
        layer up.
        """
        if (
            self._announced is not None
            or self._peek_exhausted
            or self.suppress_calls
            or not self.tools
        ):
            return []
        head = self._region.head
        parsed = self.parser_cls.parse_region(head, self.tools, at_end=False)
        if not parsed.calls:
            self._peek_exhausted = len(head) >= _PEEK_WINDOW
            return []
        if parsed.begins > 0:
            # There is answer text ahead of the call *inside* this region --
            # an answer that quotes a marker and then calls for real. That
            # text is only released when the region closes, so announcing now
            # would put the call in front of prose that `stream=false` puts
            # behind it, and a client that closes its text pane on a tool call
            # renders the explanation after the call. The name still goes out,
            # with the arguments, in the right order.
            self._peek_exhausted = True
            return []
        name = parsed.calls[0].function["name"]
        if self._declared is None:
            self._declared = _peek_declared(self.tools)
        if name not in self._declared:
            self._peek_exhausted = True
            return []
        self._announced = name
        return [
            (
                "tool_call_start",
                {
                    "index": self._index,
                    # The parsed call's own id, not a fresh one. Announcing
                    # skips the start event the call would otherwise send, so
                    # whatever goes out here is the only id the client sees --
                    # and Kimi's is the model's `functions.NAME:INDEX`, which
                    # survived only for requests that declared no tools.
                    "id": parsed.calls[0].id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                },
            )
        ]

    def _start_event(self, call: ToolCall) -> list:
        """The `tool_call_start` for a call, unless its name already went out.

        A mismatch is not reachable: the announcement is the first call of
        ``parse_region`` over a prefix of these same bytes, and acceptance is
        monotone in what has arrived. It is checked rather than asserted
        because the caller is on a live SSE stream that has already sent its
        200 -- an exception here reached the client as a connection cut
        mid-frame with no `[DONE]` and, on the `n>1` path, took the other
        choices with it.

        So it is logged and recovered from. The announced name is already on
        the wire and cannot be retracted, but it can be left with no arguments
        -- the shape every other unfulfilled announcement takes, and one that
        `completes_a_tool_call` and `finish_reason` both read as "not a call".
        The real call goes out at the next index, whole.
        """
        name = call.function["name"]
        if self._announced is not None and self._announced != name:
            logger.warning(
                "%s announced %r and then parsed %r from the same region; "
                "sending %r as a new call and leaving %r without arguments",
                self.parser_cls.__name__,
                self._announced,
                name,
                name,
                self._announced,
            )
            self._announced = None
            # Past the dangling announcement, so the real call does not land
            # on an index the client has already bound to the wrong name.
            self._index += 1
            return self._start_event(call)
        if self._announced is not None:
            self._announced = None  # binds to the first call of the region only
            return []
        return [
            (
                "tool_call_start",
                {
                    "index": self._index,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                },
            )
        ]

    def _emit_call(self, call: ToolCall) -> list:
        """One parsed call as start+args events, at the engine's own index.

        The arguments go out unconditionally, empty ones included: a
        zero-parameter tool is still a call the client should run, and
        `finish_reason` keys on this event. Gating it reported `stop` for a
        response that had already sent a `tool_calls` delta.
        """
        events = self._start_event(call)
        events.append(
            (
                "tool_call_args",
                {
                    "index": self._index,
                    "function": {"arguments": call.function["arguments"]},
                },
            )
        )
        self._index += 1
        self._emitted += 1
        return events


def read_whole(
    parser_cls: type[ToolCallParser] | None,
    text: str,
    tools: list | None = None,
    *,
    suppress_calls: bool = False,
) -> tuple[str, list[ToolCall]]:
    """A complete output as ``(content, calls)`` -- the engine, in one chunk.

    Not a second implementation of anything. `stream=false` and `stream=true`
    read the model's words with the same code, so the four rules they used to
    answer separately -- where content ends, whether an unclosed tag is a call,
    what a region that parses to nothing means, which bytes are framing -- have
    one answer each and no way to drift.

    Content comes back byte-for-byte except for markers the format declares:
    the region a call occupied, and the framing `opens_region` says no to.
    Nothing is trimmed. A trailing `.strip()` here cost a code-block answer its
    final newline, which streaming had no way to reproduce.
    """
    return flatten_tool_events(
        read_whole_events(parser_cls, text, tools, suppress_calls=suppress_calls)
    )


def read_whole_events(
    parser_cls: type[ToolCallParser] | None,
    text: str,
    tools: list | None = None,
    *,
    suppress_calls: bool = False,
) -> list:
    """The same single chunk, as the engine's own ordered events.

    `read_whole` flattens these into `(content, calls)`, which loses where the
    content sat relative to the calls. A caller that renders blocks in order
    -- the Anthropic non-streaming path -- needs that order, and building it
    from a joined string put a response's whole answer in one block ahead of
    every `tool_use`, so `stream=false` disagreed with `stream=true` about the
    order of the blocks in the same generation.
    """
    engine = ToolCallStreamParser(
        tools=tools, parser_cls=parser_cls, suppress_calls=suppress_calls
    )
    events = engine.process(text)
    events.extend(engine.flush())
    return events


def flatten_tool_events(events: list) -> tuple[str, list[ToolCall]]:
    """Ordered events as `(joined content, calls)`."""
    content: list[str] = []
    calls: list[ToolCall] = []
    pending: dict | None = None
    for etype, data in events:
        if etype == "content":
            content.append(data)
        elif etype == "tool_call_start":
            fn = data["function"]
            pending = {"id": data["id"], "name": fn["name"]}
        elif etype == "tool_call_args":
            if pending is None:
                # An announcement is what normally supplies the name, and it
                # cannot go missing between the two events of one call -- but
                # a format whose peek and parse disagree leaves a start event
                # behind without one. Dropping the arguments is what the
                # streaming clients do with the same pair.
                continue
            calls.append(
                ToolCall(
                    id=pending["id"],
                    type="function",
                    function={
                        "name": pending["name"],
                        "arguments": data["function"]["arguments"],
                    },
                )
            )
            pending = None
    return "".join(content), calls
