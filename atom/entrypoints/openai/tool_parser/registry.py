# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Format detection.

The wire format is sniffed from the model's own output rather than configured,
so detection order is load-bearing: several formats share tags and are only
told apart by a discriminator that a later entry would also match. The order
below is the single place that ordering is expressed — do not reorder without
re-reading the notes on each entry.
"""

from .deepseekv4_tool_parser import DsmlParser
from .glm_tool_parser import GlmParser
from .kimi_tool_parser import KIMI_SECTION_BEGIN, KimiParser
from .minimax_tool_parser import MINIMAX_NS, MiniMaxParser
from .qwen3_tool_parser import QWEN_TOOL_PREFIX, QwenXmlParser
from .tool_parser import ToolCall, ToolCallParser

# Checked in order on a COMPLETE output. Kimi is not listed: it is the terminal
# fallback, because its parse() also defines the "no tool calls at all" result.
#
#   MiniMax before DSML — both use `<invoke name=..>`; MiniMax additionally
#                         prefixes every tag with the ns_token.
#   GLM before Qwen     — both use `<tool_call>`; GLM never emits `<function=`,
#                         which GlmParser.detect checks for explicitly.
_DETECT_ORDER: tuple[type[ToolCallParser], ...] = (
    MiniMaxParser,
    DsmlParser,
    GlmParser,
    QwenXmlParser,
)


def parse_tool_calls(
    text: str, tools: list | None = None
) -> tuple[str, list[ToolCall]]:
    """Parse tool calls from a complete model output.

    Args:
        text: Raw model output that may contain tool calls.
        tools: Optional request tool definitions; used to type-coerce parameter
            values to their declared JSON-Schema types.

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call sections removed.
    """
    for parser in _DETECT_ORDER:
        if parser.detect(text):
            return parser.parse(text, tools)
    # Kimi is terminal: when it finds no section either, it returns the text
    # unchanged. Note that path does NOT strip, unlike every format that did
    # match — preserved as-is, callers rely on plain content surviving verbatim.
    return KimiParser.parse(text, tools)


# -- streaming sniff --------------------------------------------------------
#
# Deciding on a PREFIX is strictly harder than on a complete output: a format's
# discriminator may not have arrived yet. These two sentinels say "cannot decide
# from what I have":

# Enough plain text to be sure no marker is starting -> release it as content
# and stay undecided.
EMIT_CONTENT = object()
# Might still become a tool call -> keep buffering, emit nothing.
WAIT = object()


def sniff_stream(buf: str):
    """Pick a parser from a partial stream, or return EMIT_CONTENT / WAIT.

    Deliberately NOT the same rules as the per-parser ``detect``: on a prefix,
    GLM is only accepted on the unambiguous ``<arg_key>`` (a bare ``<tool_call>``
    could still turn out to be Qwen once ``<function=`` arrives).
    """
    if MINIMAX_NS in buf:
        return MiniMaxParser
    if DsmlParser.detect(buf):
        return DsmlParser
    if "<arg_key>" in buf:
        return GlmParser
    if QWEN_TOOL_PREFIX in buf:
        return QwenXmlParser
    if "<tool_call>" in buf:
        # '<tool_call>' seen but neither '<function=' (Qwen) nor '<arg_key>'
        # (GLM) yet. A no-arg GLM call is complete once the closing tag arrives;
        # otherwise wait for the sub-marker.
        if "</tool_call>" in buf:
            return GlmParser
        return WAIT
    if KIMI_SECTION_BEGIN in buf:
        return KimiParser
    if "<" not in buf and len(buf) > 8:
        # No '<' anywhere, so no tag has started: release the text. The length
        # floor is an unexplained heuristic carried over verbatim from the
        # original parser — it trades a little first-token latency for not
        # committing on a 1-2 char buffer. Note it does not actually rule out a
        # partial MiniMax ns_token, whose first char is ']'.
        return EMIT_CONTENT
    return WAIT
