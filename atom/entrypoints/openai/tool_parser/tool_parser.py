# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parser interface and the shared buffered-marker streaming strategy.

Every wire format implements :class:`ToolCallParser`. Four of the five
(Qwen / DSML / GLM / MiniMax) stream identically — buffer from the first start
marker, parse the whole block at flush — so that strategy lives once in
:class:`BufferedMarkerParser` and each format only declares its markers and its
``parse``. Kimi is the exception: its token format is self-delimiting, so it
emits tool calls incrementally and implements ``process``/``flush`` itself.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..marker_scanner import MarkerScanner
from .schema import build_param_types


def unique_tool_call_id() -> str:
    # OpenAI tool_call ids must be unique across the whole conversation, not just
    # within one response. A per-response index (call_0, call_1, ...) collides
    # across turns -> clients (e.g. qwen-code) dedupe by id and silently ignore
    # every repeat, causing an infinite tool-call retry loop. Use a random id.
    return f"call_{uuid.uuid4().hex}"


@dataclass
class ToolCall:
    """Parsed tool call in OpenAI format."""

    id: str
    type: str
    function: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function}


class ToolCallParser(ABC):
    """One on-the-wire tool-call format.

    Class side is the stateless non-streaming path (``detect`` + ``parse``);
    instance side is the stateful streaming path (``process`` + ``flush``).
    """

    NAME: ClassVar[str]
    # Every literal that opens this format's tool-call region. Declared once,
    # here, rather than spelled out again in each parser's own logic: a reader
    # ahead of detection needs them to know how much of its buffer could still
    # be the start of one, `BufferedMarkerParser` locates the region with them,
    # and the property tests enumerate them so a newly registered format is
    # covered the moment it exists rather than when someone writes a case.
    START_MARKERS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def peek_name(cls, region: str) -> str | None:
        """The tool being called, from a region that has not closed yet.

        ``None`` while the name is not yet legible, and ``None`` by default:
        a format that does not override this simply announces nothing early,
        which is what every format did before.

        The point is latency, and it is not small. A region is buffered until
        it closes, so on a 20 KB file write the client learned *which tool*
        only after 5030 of 5040 tokens; every one of these locates the name
        inside the first 30-70 characters. SGLang streams the whole call this
        way; only the name is taken here, because arguments arriving in
        fragments cannot be made coherent when the stream is cut short.
        """
        return None

    @classmethod
    def opens_region(cls, marker: str) -> bool:
        """Does this marker hand the rest of the stream to this format?

        `START_MARKERS` answers "which literals must not be split across a
        chunk boundary", and for most formats every one of them also opens a
        tool-call region, so the two questions have the same answer and this
        is that default. Kimi-K3 is where they come apart: three of its five
        are channel framing that wraps *every* answer, tool call or not, and
        treating those as a handover stopped it streaming at all.
        """
        return True

    def __init__(self, tools: list | None = None):
        self.tools = tools
        self.buf = ""
        # 0 = still in plain content, 1 = inside a tool-call region. Kimi adds
        # 2 = section closed; see KimiParser.
        self.state = 0
        self.current_index = 0
        self.emitted_calls = 0
        self._scanner_cache: MarkerScanner | None = None
        # The name already sent for the call being buffered, if any; cleared
        # when that call's arguments go out. See `announce`.
        self._announced: str | None = None

    @property
    def _scanner(self) -> MarkerScanner:
        """Reads the text before the region; built on first use, per instance.

        On the base rather than on `BufferedMarkerParser`, because Kimi is not
        one of those and had grown its own copy -- lazy build and all -- with
        the marker written out a second time instead of read from
        `START_MARKERS`. A format's markers are declared once; a reader that
        spells them again is the shape this module removed everywhere else.
        """
        if self._scanner_cache is None:
            self._scanner_cache = MarkerScanner(self.START_MARKERS)
        return self._scanner_cache

    # -- non-streaming ------------------------------------------------------
    @classmethod
    @abstractmethod
    def detect(cls, text: str) -> bool:
        """Whether a complete model output is in this format."""

    @classmethod
    @abstractmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse a complete output into ``(leading_content, tool_calls)``.

        **With no tool calls, the content must come back byte-for-byte.** This
        is the one rule that keeps `stream=false` answering what `stream=true`
        answers: the streaming path releases bytes as they arrive and has
        nothing to trim them with, so any tidying done only here is a
        divergence a client sees. A trailing `.strip()` cost a code-block
        answer its final newline in exactly that way.

        It binds the *no-call* case only. Trimming the content that precedes a
        real call is a choice each format may keep, and normalising framing a
        format wraps every answer in (Kimi-K3's channel tokens) is required
        rather than forbidden -- the streaming path strips those too, so
        leaving them would be the divergence. The test that enforces this
        enumerates the registry, so a format added later is bound by it
        without anyone remembering to add a case.
        """

    # -- streaming ----------------------------------------------------------
    @abstractmethod
    def process(self, text: str) -> list:
        """Consume one chunk; return ``(event_type, data)`` tuples."""

    @abstractmethod
    def flush(self) -> list:
        """Drain whatever is buffered at end of stream."""

    def announce(self, region: str) -> list:
        """Send the tool's name as soon as the region reveals it, once.

        Only for a name the request actually declared. That check is what
        makes an early name safe to send: it cannot be taken back, and an
        answer merely quoting `<tool_call><function=NAME>` opens a region too.
        A name the client never offered is overwhelmingly likelier to be prose
        than a call, so it waits for the region to close like everything else.

        SGLang's cursor parsers announce with no such check and will emit a
        call named after whatever follows the tag.
        """
        if self._announced is not None or not self.tools:
            return []
        name = self.peek_name(region)
        if name is None or name not in build_param_types(self.tools):
            return []
        self._announced = name
        return [
            (
                "tool_call_start",
                {
                    "index": self.current_index,
                    "id": unique_tool_call_id(),
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                },
            )
        ]

    def _start_event(self, index: int, call_id: str, name: str) -> list:
        """The `tool_call_start` for a call, unless its name already went out.

        Both places that emit a call go through here: `_emit_call` for the
        formats that parse a whole region, and Kimi's own drain loop, which
        builds the event inline because its id and index come off the wire.
        Two copies of "have we announced this already" is how the announcement
        was sent twice for one call.

        A mismatch is raised, not smoothed over: `peek_name` and the parse
        disagreeing about the same bytes is a bug in that format, and the name
        is already on the wire where it cannot be corrected.
        """
        if self._announced is None:
            return [
                (
                    "tool_call_start",
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    },
                )
            ]
        if self._announced != name:
            raise AssertionError(
                f"{type(self).__name__} announced {self._announced!r} and then "
                f"parsed {name!r} from the same region"
            )
        self._announced = None  # binds to the first call only
        return []

    def _emit_call(self, tc: ToolCall) -> list:
        """Render one parsed ToolCall as start+args stream events.

        The name is skipped when it has already gone out, and its id is
        reused so the client sees one call rather than two. A mismatch means
        `peek_name` and `parse` disagree about the same bytes, which is a bug
        in that format and is raised rather than papered over -- the name is
        already on the wire and cannot be corrected.
        """
        events = self._start_event(self.current_index, tc.id, tc.function["name"])
        events.append(
            (
                "tool_call_args",
                {
                    "index": self.current_index,
                    "function": {"arguments": tc.function["arguments"]},
                },
            )
        )
        self.current_index += 1
        self.emitted_calls += 1
        return events


class BufferedMarkerParser(ToolCallParser):
    """Formats that buffer from a start marker and parse the block at flush.

    The block is only parsed once complete because partial XML streams badly
    (a half-written ``<parameter=`` would emit garbage). Content before the
    first marker still streams normally.

    Subclasses declare ``START_MARKERS`` and implement ``parse``.
    """

    @classmethod
    def find_start(cls, text: str) -> int:
        """Index of the earliest start marker, or -1."""
        positions = [i for i in (text.find(m) for m in cls.START_MARKERS) if i != -1]
        return min(positions) if positions else -1

    @classmethod
    def detect(cls, text: str) -> bool:
        return cls.find_start(text) != -1

    def process(self, text: str) -> list:
        results: list = []
        if self.state == 0:
            # Held back: only a suffix that could still grow into a start
            # marker. This used to hold from the *last* holdback character
            # anywhere in the buffer, which had no branch for that character
            # landing at index 0 -- and after one emission the buffer always
            # began with one, so the parser stopped emitting for the rest of
            # the stream and delivered the remainder in a single frame at EOS.
            # `HOLDBACK_CHARS` went with it: it was a hand-kept copy of the
            # first characters of `START_MARKERS`, which the scanner derives.
            scan = self._scanner.feed(text)
            if scan.released:
                results.append(("content", scan.released))
            if scan.hit is None:
                return results
            self.buf = scan.hit + scan.rest
            self.state = 1
            # Also here: a coarse chunk can carry the opener and the name
            # together, and waiting for the next one would give that back.
            return results + self.announce(self.buf)
        self.buf += text
        # The name is legible long before the region closes, and the
        # region is what the wait is for: on a 20 KB file write the
        # client learned the tool only after 5030 of 5040 tokens.
        return results + self.announce(self.buf)

    def flush(self) -> list:
        results: list = []
        if self.state == 0:
            rest = self._scanner.flush() + self.buf
            self.buf = ""
            if rest:
                results.append(("content", rest))
            return results
        # state 1: parse the complete (or trailing) tool-call block.
        region, self.buf = self.buf, ""
        _content, tool_calls = self.parse(region, self.tools)
        if not tool_calls:
            # A start marker is not a promise. An answer explaining that a
            # model "writes <tool_call> to call something" opens the region
            # and never closes it, and this used to drop everything from the
            # marker on -- no event, no error, `finish_reason` still `stop`.
            # Fifty characters of eighty-two, on the shapes measured.
            #
            # Released verbatim rather than as `parse`'s content, which strips.
            #
            # A name may already have gone out, and there is no retracting it.
            # What there is: no arguments follow, so nothing downstream counts
            # this as a usable call and `finish_reason` stays `stop`. The text
            # is still delivered -- the promise costs a dangling name, not the
            # answer.
            self._announced = None
            return [("content", region)] if region else []
        for tc in tool_calls:
            results.extend(self._emit_call(tc))
        if self.emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results
