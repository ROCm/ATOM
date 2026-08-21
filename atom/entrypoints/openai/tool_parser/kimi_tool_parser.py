# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K2 special-token tool-call format::

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>ARGS_JSON<|tool_call_end|>
    <|tool_calls_section_end|>

Unlike the XML-ish formats this one is self-delimiting, so entries can be
emitted as soon as their ``<|tool_call_end|>`` arrives rather than buffering the
whole block. It therefore implements streaming itself instead of inheriting
:class:`~.tool_parser.BufferedMarkerParser`.

Arguments are already JSON on the wire, so no schema coercion is applied and
``tools`` is unused. The call id is the model's own ``functions.NAME:INDEX``
rather than a random one, and ``index`` comes from the wire too.
"""

import re
from typing import ClassVar

from .tool_parser import ToolCall, ToolCallParser

KIMI_SECTION_BEGIN = "<|tool_calls_section_begin|>"
KIMI_SECTION_END = "<|tool_calls_section_end|>"

_SECTION_RE = re.compile(
    re.escape(KIMI_SECTION_BEGIN) + r"(.*?)" + re.escape(KIMI_SECTION_END),
    re.DOTALL,
)
_UNCLOSED_RE = re.compile(re.escape(KIMI_SECTION_BEGIN) + r"(.*?)$", re.DOTALL)
_ENTRY_RE = re.compile(
    r"<\|tool_call_begin\|>"
    r"functions\.(\w+):(\d+)"
    r"<\|tool_call_argument_begin\|>"
    r"(.*?)"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)


def _parse_entries(section_text: str) -> list[ToolCall]:
    """Parse individual tool call entries from the section content."""
    tool_calls = []
    for match in _ENTRY_RE.finditer(section_text):
        name = match.group(1)
        index = match.group(2)
        arguments = match.group(3).strip()
        tool_id = f"functions.{name}:{index}"
        tool_calls.append(
            ToolCall(
                id=tool_id,
                type="function",
                function={"name": name, "arguments": arguments},
            )
        )
    return tool_calls


class KimiParser(ToolCallParser):
    """States: 0 = plain content, 1 = inside section, 2 = section closed."""

    NAME: ClassVar[str] = "kimi"
    # The section opener, and the only literal detection keys on. The entry
    # markers inside it are `_drain_entries`' business, never a reader's.
    START_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_BEGIN,)

    # No `peek_name`, deliberately. This format carries the call's index and
    # id on the wire (`functions.NAME:INDEX`), and an announcement has to be
    # stamped with both before the entry that supplies them has arrived --
    # every announced call went out at index 0, so a client accumulating by
    # index overwrote the first call with the second. It also drains per
    # completed entry rather than at flush, so it has the least to gain.

    @classmethod
    def detect(cls, text: str) -> bool:
        return KIMI_SECTION_BEGIN in text

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        section_match = _SECTION_RE.search(text)
        if not section_match:
            # Unclosed section: the model was cut off mid-block; salvage whatever
            # complete entries it managed to emit.
            unclosed = _UNCLOSED_RE.search(text)
            if unclosed:
                entries = _parse_entries(unclosed.group(1))
                content = text[: unclosed.start()]
                # No call -> verbatim; see ToolCallParser.parse.
                return (content.strip() if entries else text), entries
            return text, []
        entries = _parse_entries(section_match.group(1))
        content = text[: section_match.start()]
        return (content.strip() if entries else text), entries

    def process(self, text: str) -> list:
        results: list = []

        if self.state == 0:
            # Held back: only a suffix that could still grow into the section
            # marker. It used to be `"<|tool" not in self.buf` over a buffer
            # that was never cleared while that held, so a single `<|tool` --
            # or an answer merely discussing one -- withheld everything after
            # it until the stream ended. The 30-character floor went with it;
            # the scanner's bound is the marker's own length.
            scan = self._scanner.feed(text)
            if scan.released:
                results.append(("content", scan.released))
            if scan.hit is not None:
                self.state = 1
                self.buf = scan.rest
                results.extend(self._drain_entries())

        elif self.state == 1:
            self.buf += text
            if KIMI_SECTION_END in self.buf:
                section, _, after = self.buf.partition(KIMI_SECTION_END)
                self.buf = section
                results.extend(self._drain_entries())
                if self.emitted_calls:
                    results.append(("tool_call_end", None))
                else:
                    # A start marker is not a promise, for this format too.
                    # The section body was dropped here and `state = 2` then
                    # discarded the rest of the stream: an answer quoting both
                    # section tokens delivered 26 of its 135 characters when
                    # fed four at a time, and all 135 in one shot. `flush`'s
                    # fallback could not see it -- the bytes were already gone.
                    kept = KIMI_SECTION_BEGIN + section + KIMI_SECTION_END
                    results.append(("content", kept))
                # Back to plain content, not a terminal state: text after the
                # section is still the answer.
                self.state = 0
                self.buf = ""
                if after:
                    results.extend(self.process(after))
            else:
                results.extend(self._drain_entries())

        return results

    def _drain_entries(self) -> list:
        """Emit every complete tool-call entry sitting in the buffer."""
        results: list = []
        while "<|tool_call_begin|>" in self.buf and "<|tool_call_end|>" in self.buf:
            match = _ENTRY_RE.search(self.buf)
            if not match:
                break

            name = match.group(1)
            index = int(match.group(2))
            arguments = match.group(3).strip()

            results.extend(self._start_event(index, f"functions.{name}:{index}", name))
            # Unconditional, empty arguments included: a zero-parameter tool
            # is a call the client should run, and `finish_reason` keys on
            # this event. Gating it here reported `stop` for a response that
            # had already sent a `tool_calls` delta.
            results.append(
                (
                    "tool_call_args",
                    {"index": index, "function": {"arguments": arguments}},
                )
            )

            self.buf = self.buf[match.end() :]
            self.emitted_calls += 1

        return results

    def flush(self) -> list:
        results: list = []
        if self.state == 0:
            # The scanner owns the held tail now, and `self.buf` is never
            # written in this state -- draining only `self.buf` here dropped
            # whatever was still being read ahead. Six characters on
            # `process("hello <|tool")`.
            held = self._scanner_cache.flush() if self._scanner_cache else ""
            rest = held + self.buf
            self.buf = ""
            if rest:
                results.append(("content", rest))
        elif self.state == 1:
            results.extend(self._drain_entries())
            if self.emitted_calls > 0:
                results.append(("tool_call_end", None))
            elif self.buf:
                # The section opened and closed nothing. Same rule as every
                # other format: a start marker is not a promise.
                results.append(("content", KIMI_SECTION_BEGIN + self.buf))
                self.buf = ""
        return results
