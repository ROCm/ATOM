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
from .kimi_k3_tool_parser import KimiK3Parser, is_kimi_k3
from .kimi_tool_parser import KIMI_SECTION_BEGIN, KimiParser
from .minimax_tool_parser import MINIMAX_NS, MiniMaxParser
from .qwen3_tool_parser import QWEN_TOOL_PREFIX, QwenXmlParser
from .tool_parser import ToolCall, ToolCallParser

# Checked in order on a COMPLETE output. Kimi (K2) is not listed: it is the
# terminal fallback, because its parse() also defines the "no tool calls at all"
# result.
#
#   K3 first             — its `<|open|>...<|sep|>` channel tokens are disjoint
#                          from every other format's tags, so it never collides;
#                          it also strips its own channel framing from plain
#                          answers (which the terminal K2 fallback would not).
#   MiniMax before DSML — both use `<invoke name=..>`; MiniMax additionally
#                         prefixes every tag with the ns_token.
#   GLM before Qwen     — both use `<tool_call>`; GLM never emits `<function=`,
#                         which GlmParser.detect checks for explicitly.
_DETECT_ORDER: tuple[type[ToolCallParser], ...] = (
    KimiK3Parser,
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

# Might still become a tool call -> the caller keeps this text.
#
# `WAIT` is only about the *format*, never about whether text may be released.
# It used to be both, alongside an `EMIT_CONTENT` that meant "no '<' anywhere,
# so nothing can be starting". That conflation was the stall: one '<' in an
# ordinary answer -- `if (a < b)` -- made every branch here miss forever, and
# `ToolCallStreamParser` never cleared the buffer it was accumulating, so the
# answer arrived in a single frame at end of stream. Deciding how much text is
# safe to send is now `MarkerScanner`'s, over the union of every registered
# format's `MARKERS`, and it is asked before this function is.
WAIT = object()


# Literals the cascade below discriminates by that open no region of their
# own, so no parser declares them and a reader would not otherwise hold them
# back. `<arg_key>` is what tells GLM from Qwen -- both open with
# `<tool_call>` -- and it appears *inside* that region, which is why it is not
# one of GLM's `START_MARKERS`: `find_start` would take it for the region's
# beginning and parse from the wrong offset.
_SNIFF_ONLY: tuple[str, ...] = ("<arg_key>",)


def all_markers() -> tuple[str, ...]:
    """Every literal that could be starting before the format is known.

    A suffix that could still grow into one of these is not safe to send. Taken
    from the parsers' own `START_MARKERS` plus the cascade's own discriminators,
    so a format added to `_DETECT_ORDER` is covered without a second edit here.
    """
    formats = {m for p in (*_DETECT_ORDER, KimiParser) for m in p.START_MARKERS}
    return tuple(formats | set(_SNIFF_ONLY))


def sniff_stream(buf: str):
    """Pick a parser from a partial stream, or return EMIT_CONTENT / WAIT.

    Deliberately NOT the same rules as the per-parser ``detect``: on a prefix,
    GLM is only accepted on the unambiguous ``<arg_key>`` (a bare ``<tool_call>``
    could still turn out to be Qwen once ``<function=`` arrives).
    """
    # K3's channel tokens (``<|open|>response<|sep|>`` / ``<|open|>call tool=``)
    # are disjoint from every other format, so a complete one decides immediately.
    # K3 buffers to EOS and strips its own framing, so plain answers route here
    # too rather than leaking channel tokens through the undecided-content path.
    if is_kimi_k3(buf):
        return KimiK3Parser
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
    return WAIT
