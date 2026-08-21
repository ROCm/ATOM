# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Format detection.

Several formats share tags and are only told apart by a discriminator that a
later entry would also match, so detection order is load-bearing. `_DETECT_ORDER`
is the single place that ordering is expressed — do not reorder without
re-reading the notes on each entry — and both callers use it: `parse_tool_calls`
on a complete output, and `resolve_from_prompt` on a rendered chat template at
startup.
"""

import logging
from typing import Any

from ..chat_encoders import apply_chat_template
from .deepseekv4_tool_parser import DsmlParser
from .glm_tool_parser import GlmParser
from .kimi_k3_tool_parser import KimiK3Parser
from .kimi_tool_parser import KimiParser
from .minimax_tool_parser import MiniMaxParser
from .qwen3_tool_parser import QwenXmlParser
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
    text: str,
    tools: list | None = None,
    parser_cls: "type[ToolCallParser] | None" = None,
) -> tuple[str, list[ToolCall]]:
    """Parse tool calls from a complete model output.

    Args:
        text: Raw model output that may contain tool calls.
        tools: Optional request tool definitions; used to type-coerce parameter
            values to their declared JSON-Schema types.
        parser_cls: The format resolved for this model at startup. Given, it is
            used and the cascade below is not consulted — the streaming path
            reads the same output as that same format, and the two answering
            differently is a divergence a client sees as a tool call appearing
            only when it does not stream.

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call sections removed.
    """
    if parser_cls is not None:
        content, tool_calls = parser_cls.parse(text, tools)
        if not tool_calls:
            # No call, so nothing was a tool-call section and nothing should
            # have been rewritten. Formats strip their content, which for an
            # ordinary answer means a code block comes back without its
            # trailing newline -- and only when `stream=false`, since the
            # streaming path releases bytes as they arrive.
            return text, []
        return content, tool_calls
    for parser in _DETECT_ORDER:
        if parser.detect(text):
            return parser.parse(text, tools)
    # Kimi is terminal: when it finds no section either, it returns the text
    # unchanged. Note that path does NOT strip, unlike every format that did
    # match — preserved as-is, callers rely on plain content surviving verbatim.
    return KimiParser.parse(text, tools)


# -- format resolution -----------------------------------------------------

# Every format by the name `--tool-call-parser` takes. Derived from the same
# order, so a newly registered format is selectable without a second edit.
PARSERS_BY_NAME: dict[str, type[ToolCallParser]] = {
    p.NAME: p for p in (*_DETECT_ORDER, KimiParser)
}


def resolve_from_prompt(rendered_prompt: str) -> type[ToolCallParser] | None:
    """Which format this model will emit, decided before it emits anything.

    A chat template rendered with a tools payload *is* the model's instructions
    for how to call one, so the same cascade `parse_tool_calls` runs on a
    complete output answers the question on the prompt instead -- earlier, and
    without depending on what the model happens to produce first.

    Asked once at startup. `None` means no registered format recognised the
    prompt, which is the honest answer for a model with no tool syntax ATOM
    knows; the caller says so out loud rather than falling back to guessing,
    because a guess here is a tool call fabricated out of ordinary text.

    This replaces `sniff_stream`, which decided from a *prefix* of the output.
    That was strictly harder -- a format's discriminator may not have arrived
    yet, so the answer needed a "cannot tell yet" state, and that state was
    read as "and therefore send nothing", which is how one '<' in an answer
    withheld the rest of the stream.
    """
    # Kimi is included here though `_DETECT_ORDER` omits it. There it is the
    # terminal fallback, because its `parse` also defines "no tool calls at
    # all"; that makes it wrong to try *last* on an output and right to try at
    # all on a prompt. Omitting it meant a K2 deployment resolved to nothing
    # and its tool calls arrived as raw section tokens in `delta.content`,
    # while the non-streaming path still fell through to `KimiParser.parse`
    # and returned them -- the same request answered two ways.
    for parser in (*_DETECT_ORDER, KimiParser):
        if parser.detect(rendered_prompt):
            return parser
    return None


logger = logging.getLogger("atom")

AUTO = "auto"

# The smallest request that makes a template render its tool instructions. The
# tool is never called; only the framing the template wraps it in matters.
_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def resolve_tool_call_parser(
    override: str | None,
    tokenizer: Any,
    custom_encoder: Any = None,
    *,
    model: str = "",
) -> type[ToolCallParser] | None:
    """The format this model's tool calls will arrive in, or ``None``.

    ``override`` is ``--tool-call-parser``: a format name, or ``"auto"``/``None``
    to read it off the chat template. An unknown name raises rather than
    falling back, because a typo that silently disables tool parsing is the
    failure this whole path exists to stop.

    ``None`` is a real answer, not an error: gpt-oss and DeepSeek-R1 render no
    tool syntax ATOM knows, and for them parsing nothing is correct. It is
    logged either way.

    Rendering is best-effort — a template can raise on a tools payload it does
    not accept — and a failure to render is reported and treated as
    unrecognised, never as a reason to fall back to reading the output.
    """
    if override and override != AUTO:
        parser = PARSERS_BY_NAME.get(override)
        if parser is None:
            raise ValueError(
                f"--tool-call-parser={override!r} is not a known format; "
                f"choose one of {sorted(PARSERS_BY_NAME)} or {AUTO!r}"
            )
        logger.info(f"Tool-call format: {parser.NAME} (from --tool-call-parser)")
        return parser

    try:
        rendered = apply_chat_template(
            tokenizer, custom_encoder, _PROBE_MESSAGES, tools=_PROBE_TOOLS
        )
    except Exception as e:  # noqa: BLE001 - any template failure means "cannot tell"
        logger.warning(
            f"Could not render {model or 'the model'}'s chat template with a tools "
            f"payload, so its tool-call format is unknown and tool calls will be "
            f"delivered as plain text: {e}. Pass --tool-call-parser to set it."
        )
        return None

    parser = resolve_from_prompt(rendered)
    if parser is None:
        logger.info(
            f"No known tool-call format in {model or 'the model'}'s chat template; "
            f"tool calls will be delivered as plain text. Pass --tool-call-parser "
            f"if this model does emit one."
        )
    else:
        logger.info(f"Tool-call format: {parser.NAME} (from the chat template)")
    return parser
