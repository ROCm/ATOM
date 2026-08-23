# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3 (qwen3_coder / qwen3_xml) XML tool-call format::

    <tool_call>
    <function=NAME>
    <parameter=PNAME>VALUE</parameter>
    ...
    </function>
    </tool_call>

The XML carries no value types, so parameters are coerced against the request's
``tools`` schema when supplied. Mirrors the qwen3_coder/qwen3_xml parsers in
vLLM and SGLang.
"""

import json
import re
from typing import Any, ClassVar

from .kimi_tool_parser import KIMI_SECTION_BEGIN
from .schema import build_param_types, coerce_param_value
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
)

# Also read by GlmParser.detect: '<function=' is what tells Qwen's <tool_call>
# apart from GLM's identically-named tag.
QWEN_TOOL_PREFIX = "<function="

# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <function=NAME>" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
_NOT_NESTED = r"(?:(?!<function=).)"
_FUNCTION_RE = re.compile(
    r"<function=(" + _NOT_NESTED + r"*?)</function>|<function=(" + _NOT_NESTED + r"*)",
    re.DOTALL,
)
_PARAM_OPENER = "<parameter="
_PARAM_RE = re.compile(
    # `\Z`, not `$`: `$` also matches before a trailing newline, so a value
    # ending in one was cut a byte short. `(?=<tool_call>)` is the fifth
    # terminator `parse_region` names -- a value ends where the next call
    # opens, or it swallows that call's wrapper as data.
    r"<parameter=(.*?)"
    r"(?:</parameter>|(?=<parameter=)|(?=</function>)|(?=<tool_call>)|\Z)",
    re.DOTALL,
)


# What may follow the name inside a call: another parameter, or the close of
# the very block the name opened. NOT `</tool_call>`, which closes the
# *outer* wrapper -- `<function=get_weather></tool_call>` leaves the function
# block unterminated, so `parse` reads it as prose. The peek used to accept
# it and `parse` did not, which is the whole of the mismatch this shared
# tuple exists to prevent: one spelling, both readers.
_CALL_CONTINUES = (_PARAM_OPENER, "</function>")


def _name_and_rest(fn_text: str) -> tuple[str, str] | None:
    """Split `NAME>whatever` at the tag close, or ``None`` if it has not come."""
    gt = fn_text.find(">")
    if gt == -1:
        return None
    return fn_text[:gt].strip(), fn_text[gt + 1 :].lstrip()


def _is_truncated_call(fn_text: str, param_types: dict, *, at_end: bool) -> bool:
    """Is this unclosed `<function=...` a cut-off call, or prose quoting a tag?

    The unclosed branch exists for a call the model was cut off mid-way
    through. It cannot tell that from an answer explaining how to call a tool,
    and used to accept both: "the model writes <tool_call><function=get_weather>
    and then the parameters" produced `get_weather({})`, deleted the rest of
    the sentence and reported `finish_reason: tool_calls`, so an agentic
    client ran a tool nobody asked for.

    Two things separate them, and prose has to fail both. The name is one the
    request declared -- prose can name a real tool, so this alone is not
    enough. And what follows the name is this format's own next token: a
    cut-off call stops inside its own syntax, while prose continues in
    English. The early announcement runs this same function over the region
    so far, so the two cannot answer differently.
    """
    split = _name_and_rest(fn_text)
    if split is None:
        return False
    name, rest = split
    return declared_tools_allow(name, param_types) and (
        (not rest and at_end)
        or continues_a_call(rest, _CALL_CONTINUES, arrived=not at_end)
    )


def _parse_function(
    fn_text: str, param_types: dict[str, dict[str, Any]]
) -> ToolCall | None:
    """Parse the inside of one ``<function=NAME>...`` block into a ToolCall."""
    gt = fn_text.find(">")
    if gt == -1:
        return None
    name = fn_text[:gt].strip()
    if not name:
        return None
    body = fn_text[gt + 1 :]
    types = param_types.get(name, {})
    args: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(body):
        seg = pm.group(1)
        if seg is None:
            continue
        pgt = seg.find(">")
        if pgt == -1:
            continue
        pname = seg[:pgt].strip()
        pval = seg[pgt + 1 :]
        if pname:
            args[pname] = coerce_param_value(pval, types.get(pname))
    return ToolCall(
        id=unique_tool_call_id(),
        type="function",
        function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    )


class QwenXmlParser(ToolCallParser):
    NAME: ClassVar[str] = "qwen"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>", QWEN_TOOL_PREFIX)
    # `</tool_call>` closes the wrapper the calls sit in, so it is markup too
    # -- and so is the newline between it and `</function>`. Not a
    # `REGION_END_MARKER`: a model writing *about* tool calls puts the literal
    # inside a `<parameter=` value, where closing the region on it would cut a
    # real call in half.
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in args.items())
        return f"<tool_call><function={name}>{body}</function></tool_call>"

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the Qwen3 XML format (and not the Kimi token format)."""
        return QWEN_TOOL_PREFIX in text and KIMI_SECTION_BEGIN not in text

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for fm in _FUNCTION_RE.finditer(region):
            closed = fm.group(1) is not None
            fn_text = fm.group(1) if closed else fm.group(2)
            if not fn_text:
                continue
            tc = _parse_function(fn_text, param_types)
            if tc is None:
                continue
            if not closed and not _is_truncated_call(
                fn_text, param_types, at_end=at_end
            ):
                continue
            calls.append(tc)
            spans.append(
                (
                    cls.markup_begin(region, fm.start()),
                    cls.markup_end(region, fm.end()),
                )
            )
        return RegionParse(tuple(calls), tuple(spans))
