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
                content = text[: unclosed.start()]
                return content.strip(), _parse_entries(unclosed.group(1))
            return text, []
        content = text[: section_match.start()]
        return content.strip(), _parse_entries(section_match.group(1))

    def process(self, text: str) -> list:
        results: list = []

        if self.state == 0:
            self.buf += text
            if KIMI_SECTION_BEGIN in self.buf:
                before = self.buf.split(KIMI_SECTION_BEGIN)[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split(KIMI_SECTION_BEGIN, 1)[1]
                results.extend(self._drain_entries())
            elif "<|tool" not in self.buf and len(self.buf) > 30:
                # No partial special token possible yet -> safe to release.
                results.append(("content", self.buf))
                self.buf = ""

        elif self.state == 1:
            self.buf += text
            if KIMI_SECTION_END in self.buf:
                self.buf = self.buf.split(KIMI_SECTION_END)[0]
                results.extend(self._drain_entries())
                results.append(("tool_call_end", None))
                self.state = 2
                self.buf = ""
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

            results.append(
                (
                    "tool_call_start",
                    {
                        "index": index,
                        "id": f"functions.{name}:{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    },
                )
            )
            if arguments:
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
        if self.state == 0 and self.buf:
            results.append(("content", self.buf))
            self.buf = ""
        elif self.state == 1:
            results.extend(self._drain_entries())
            if self.emitted_calls > 0:
                results.append(("tool_call_end", None))
        return results
