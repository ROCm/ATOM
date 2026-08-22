# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K2 special-token tool-call format::

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>ARGS_JSON<|tool_call_end|>
    <|tool_calls_section_end|>

Arguments are already JSON on the wire, so no schema coercion is applied and
``tools`` is unused for parsing. The call id is the model's own
``functions.NAME:INDEX``.

The section end is a special token that cannot occur inside an argument value,
so this is the one format that can say where a region closes without waiting
for end of stream -- see ``REGION_END_MARKERS``. That is what lets a second
section, and the answer after the last one, be read at all: both used to be
swallowed, differently, by the two readers this format used to have.
"""

import re
from typing import ClassVar

from .tool_parser import RegionParse, ToolCall, ToolCallParser

KIMI_SECTION_BEGIN = "<|tool_calls_section_begin|>"
KIMI_SECTION_END = "<|tool_calls_section_end|>"
KIMI_ENTRY_END = "<|tool_call_end|>"

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
    NAME: ClassVar[str] = "kimi"
    # The section opener, and the only literal detection keys on. The entry
    # markers inside it are `parse_region`'s business, never a reader's.
    START_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_BEGIN,)
    # A special token, so it cannot appear inside a JSON argument value. That
    # is the whole licence for closing a region on it: the XML formats' own
    # closers fail this test, because a model writing about tool calls puts
    # one inside a parameter.
    REGION_END_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_END,)

    @classmethod
    def detect(cls, text: str) -> bool:
        return KIMI_SECTION_BEGIN in text

    @classmethod
    def region_end(cls, region: str) -> int:
        at = region.find(KIMI_SECTION_END)
        return at + len(KIMI_SECTION_END) if at != -1 else 0

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        """One section: everything between its two markers, or what arrived.

        `region_end` hands this exactly one section at a time, so a response
        with two of them is two regions and the text between and after them
        reaches the client. Reading the *first* section out of the whole
        output -- which is what `_SECTION_RE.search` did -- lost the second
        call entirely and delivered the raw wire tokens of both as content.
        """
        entries = _parse_entries(region)
        if not entries:
            return RegionParse()
        # The region opens at the *first* section marker in the text, which an
        # answer that quotes one before making a real call puts in the wrong
        # place. The section that matters is the last one opened before the
        # first entry; everything before it is the answer.
        first = _ENTRY_RE.search(region)
        opened = region.rfind(KIMI_SECTION_BEGIN, 0, first.start())
        # One span for the whole section, not one per entry: `region_end`
        # already hands this exactly one section, and between two entries
        # there is only `<|tool_call_end|><|tool_call_begin|>` -- special
        # tokens a model cannot write prose between.
        return RegionParse(tuple(entries), ((max(opened, 0), len(region)),))
