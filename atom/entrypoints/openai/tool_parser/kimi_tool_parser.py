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

import json
import re
from typing import ClassVar

from .schema import build_param_types
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    usable_tool_name,
)

KIMI_SECTION_BEGIN = "<|tool_calls_section_begin|>"
KIMI_SECTION_END = "<|tool_calls_section_end|>"
KIMI_ENTRY_END = "<|tool_call_end|>"

_ENTRY_RE = re.compile(
    r"<\|tool_call_begin\|>"
    # `[\w.\-]`, not `\w`: function names are `^[a-zA-Z0-9_-]{1,64}$` and MCP
    # servers namespace theirs `server.tool`. With `\w` the entry did not
    # match, so the whole section -- special tokens included -- went out as
    # `content` with `finish_reason: stop`.
    r"functions\.([\w.\-]+):(\d+)" r"<\|tool_call_argument_begin\|>"
    # Ends at `<|tool_call_end|>`, the next entry, the section close, the next
    # *section*, or end of input. Without end-of-input a `max_tokens`
    # truncation matched nothing and went out as raw special tokens -- and
    # beside a complete entry, was dropped entirely. The section opener is the
    # fifth terminator `parse_region` names; here the next call opens a new
    # section rather than a sibling entry.
    r"(.*?)"
    r"(?:(?P<closed><\|tool_call_end\|>)"
    r"|(?=<\|tool_call_begin\|>)"
    r"|(?=<\|tool_calls_section_end\|>)"
    r"|(?=<\|tool_calls_section_begin\|>)"
    r"|\Z)",
    re.DOTALL,
)


def _is_truncated_call(
    name: str, args: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed entry a cut-off call, or prose about the format?

    The gate that travels with an unclosed alternative. Without it a sentence
    quoting `<|tool_call_begin|>functions.x:0<|tool_call_argument_begin|>`
    parses as a call -- the same pair-error K3 had, and it was reintroduced
    here in the very commit that fixed it there.

    The follower is `{`: this format's arguments are JSON on the wire, so a
    real call continues with an object or, at end of region, with nothing.
    """
    if not declared_tools_allow(name, param_types):
        return False
    rest = args.lstrip()
    return (not rest and at_end) or continues_a_call(rest, ("{",), arrived=not at_end)


def _parse_entries(
    section_text: str, param_types: dict, *, at_end: bool
) -> tuple[list[ToolCall], int]:
    """The entries this section really made, and where the first one starts.

    The offset is returned rather than recomputed by the caller, because the
    caller cannot: entries that fail the truncation gate are skipped here, and
    asking `_ENTRY_RE.search` for the first *match* answers a different
    question than "the first entry we accepted". They differ exactly when an
    answer quotes the wire format before making a real call, which is the case
    the section anchor exists for -- so the recomputation was wrong precisely
    where it mattered.
    """
    tool_calls = []
    first_start = -1
    for match in _ENTRY_RE.finditer(section_text):
        name = match.group(1)
        index = match.group(2)
        # `[\w.\-]+` is not the same set: it admits a leading `.` or `-`, so
        # `functions..hidden:0` shipped a call named `.hidden`. "Safe by
        # construction" is why this parser was left out of the sweep.
        if not usable_tool_name(name):
            continue
        if match.group("closed") is None and not _is_truncated_call(
            name, match.group(3), param_types, at_end=at_end
        ):
            continue
        # `"{}"` and not `""` for a zero-argument call. This format passes
        # the wire bytes through where the other five build a JSON object, so
        # it was the one format whose no-argument call reached the client as
        # an empty string -- `json.loads("")` raises, and on `/v1/messages` it
        # becomes an `input_json_delta` the Anthropic SDK cannot accumulate.
        arguments = match.group(3).strip() or "{}"
        tool_id = f"functions.{name}:{index}"
        if first_start < 0:
            first_start = match.start()
        tool_calls.append(
            ToolCall(
                id=tool_id,
                type="function",
                function={"name": name, "arguments": arguments},
            )
        )
    return tool_calls, first_start


class KimiParser(ToolCallParser):
    NAME: ClassVar[str] = "kimi"
    # The section opener, and the only literal detection keys on. The entry
    # markers inside it are `parse_region`'s business, never a reader's.
    START_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_BEGIN,)
    # An entry's own end. `CALL_CLOSERS` stays empty: the section end is a
    # `REGION_END_MARKER`, which is a stronger statement than "markup at a
    # call's right edge" and is what `markup_end` must not walk over here.
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = (KIMI_ENTRY_END,)
    # A special token, so it cannot appear inside a JSON argument value. That
    # is the whole licence for closing a region on it: the XML formats' own
    # closers fail this test, because a model writing about tool calls puts
    # one inside a parameter.
    REGION_END_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_END,)

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        # One call, so the entry index is 0. Do not read a multi-call shape
        # off two of these concatenated: a real section increments the index
        # across its entries, and taking the doubled `:0` for model output
        # produced a duplicate-id "defect" that does not exist.
        return (
            f"{KIMI_SECTION_BEGIN}<|tool_call_begin|>functions.{name}:0"
            f"<|tool_call_argument_begin|>{json.dumps(args)}"
            f"{KIMI_ENTRY_END}{KIMI_SECTION_END}"
        )

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
        entries, first_start = _parse_entries(
            region, build_param_types(tools), at_end=at_end
        )
        if not entries:
            return RegionParse()
        # The region opens at the *first* section marker in the text, which an
        # answer that quotes one before making a real call puts in the wrong
        # place. The section that matters is the last one opened before the
        # first entry; everything before it is the answer.
        #
        # The first entry *accepted*, not the first thing the entry pattern
        # matched. `_ENTRY_RE.search` was the latter, so a quotation the
        # truncation gate had already rejected still moved the anchor back to
        # its section -- and the sentence the model wrote between the
        # quotation and its real call was deleted as markup. The other five
        # formats keep it; this was the only one that did not.
        opened = region.rfind(KIMI_SECTION_BEGIN, 0, first_start)
        # One span for the whole section, not one per entry: `region_end`
        # already hands this exactly one section, and between two entries
        # there is only `<|tool_call_end|><|tool_call_begin|>` -- special
        # tokens a model cannot write prose between.
        return RegionParse(tuple(entries), ((max(opened, 0), len(region)),))
