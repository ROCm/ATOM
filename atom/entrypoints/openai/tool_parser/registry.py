# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Format detection.

Several formats share tags and are only told apart by a discriminator that a
later entry would also match, so detection order is load-bearing. `_DETECT_ORDER`
is the single place that ordering is expressed — do not reorder without
re-reading the notes on each entry.

It is consulted on a *rendered chat template*, at startup, and nowhere else.
Running it on a model's output as well was the same question asked of worse
evidence: a template states the format it taught the model, while an output
only exhibits whatever it happens to contain, so an ordinary answer quoting
`<|tool_calls_section_begin|>` was read as a Kimi section and everything after
it deleted — on the non-streaming path only, because streaming had already
been given the startup answer. One question, one place to ask it.
"""

import logging
from typing import Any

from ..chat_encoders import render_probe_prompt
from .deepseekv4_tool_parser import DsmlParser
from .glm_tool_parser import GlmParser
from .kimi_k3_tool_parser import KimiK3Parser
from .kimi_tool_parser import KimiParser
from .minimax_tool_parser import MiniMaxParser
from .qwen3_tool_parser import QwenXmlParser
from .stream import read_whole
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
    *,
    suppress_calls: bool = False,
) -> tuple[str, list[ToolCall]]:
    """Parse tool calls from a complete model output.

    Args:
        text: Raw model output that may contain tool calls.
        tools: Optional request tool definitions; used to type-coerce parameter
            values to their declared JSON-Schema types.
        parser_cls: The format resolved for this model at startup, or ``None``
            for "do not parse". ``None`` is not "work it out from the text":
            the streaming path is handed the same ``None`` and emits
            everything as content, so guessing here would answer the same
            request two ways — measured, an answer describing
            `<|tool_calls_section_begin|>` lost 30 characters when
            `stream=false` and arrived whole when `stream=true`.
        suppress_calls: ``tool_choice: "none"``. See the note above
            :func:`forbids_tool_calls` for why this is a flag and not
            "use no parser".

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call regions and this format's declared framing removed, and is
        ``text`` byte-for-byte when there is neither.

    This is :func:`~.stream.read_whole` — the streaming engine over one chunk —
    and deliberately not a second implementation of it.
    """
    return read_whole(parser_cls, text, tools, suppress_calls=suppress_calls)


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
    # Kimi is included here though `_DETECT_ORDER` omits it. Its `parse` also
    # defines "no tool calls at all", which is why it is not an entry in an
    # ordering meant to tell formats apart -- but a K2 prompt does carry K2's
    # section tokens, so on a prompt it is a real answer. Omitting it meant a
    # K2 deployment resolved to nothing and delivered its tool calls as raw
    # section tokens in `delta.content`.
    for parser in (*_DETECT_ORDER, KimiParser):
        if parser.detect(rendered_prompt):
            return parser
    return None


logger = logging.getLogger("atom")

AUTO = "auto"

# Written once and used by both entrypoints, which is also the shape of the
# bug it replaces: the flag existed on the OpenAI server and not on atomesh,
# so the documented escape hatch did not exist there.
TOOL_CALL_PARSER_HELP = (
    "Tool-call wire format. 'auto' (default) reads it off the model's chat "
    "template at startup; an explicit name overrides that. When neither "
    "resolves, tool calls are delivered as plain text and the startup log "
    "says so \u2014 the format is never guessed from output."
)


def forbids_tool_calls(tool_choice: Any) -> bool:
    """Did this request say the model may not call a tool?

    Both protocols' spellings, because both endpoints ask. OpenAI sends the
    bare string ``"none"``; Anthropic sends ``{"type": "none"}``. Anything
    else -- absent, ``auto``, ``required``, ``any``, a named tool -- is not a
    prohibition, and in particular is not answered here.
    """
    if isinstance(tool_choice, str):
        return tool_choice == "none"
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "none"
    return False


"""Why ``tool_choice: "none"`` is a flag and not "use no parser".

`tool_choice: "none"` used to be enforced where the events were *sent*, twelve
places across two endpoints, while the parser went on consuming the region. So
the model's own words were deleted rather than its call suppressed: a
95-character answer reached the client as its first six, with no event and
`finish_reason: stop`.

The fix for that dropped the parser instead -- and dropping the parser also
drops everything else a parser does. A format whose framing wraps *every*
answer then leaks that framing: Kimi-K3's `Hello there.` arrived as
`<|open|>response<|sep|>Hello there.<|close|>response<|sep|><|end_of_msg|>`
the moment a request said `none`. The format still has to be read; what is
suppressed is dispatch. `ToolCallStreamParser(suppress_calls=True)` and
:func:`parse_tool_calls(..., suppress_calls=True)` are that, and both leave
the region's bytes in the answer -- which is the correct reading, since the
request said they could not be a call.
"""


def validate_tool_call_parser(override: str | None) -> type[ToolCallParser] | None:
    """The format `override` names, or ``None`` for "read the template".

    Separate from :func:`resolve_tool_call_parser` because the name can be
    checked before a tokenizer exists, and therefore before the weights load:
    atomesh resolved inside the service constructor, so a typo was reported
    only after a full model load.
    """
    if not override or override == AUTO:
        return None
    parser = PARSERS_BY_NAME.get(override)
    if parser is None:
        raise ValueError(
            f"--tool-call-parser={override!r} is not a known format; "
            f"choose one of {sorted(PARSERS_BY_NAME)} or {AUTO!r}"
        )
    return parser


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
    parser = validate_tool_call_parser(override)
    if parser is not None:
        logger.info(f"Tool-call format: {parser.NAME} (from --tool-call-parser)")
        return parser

    rendered = render_probe_prompt(tokenizer, custom_encoder, tools=True)
    if rendered is None:
        logger.warning(
            f"Could not render {model or 'the model'}'s chat template with a tools "
            f"payload, so its tool-call format is unknown and tool calls will be "
            f"delivered as plain text. Pass --tool-call-parser to set it."
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
