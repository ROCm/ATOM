# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 tool-call format.

Every tag is prefixed by the ns_token ``]<]minimax[>[``::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">
    ]<]minimax[>[<pname>value]<]minimax[>[</pname>
    ...
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Unlike DSML, parameters are named by the TAG itself (``<city>Paris</city>``),
not a ``name="..."`` attribute. Values: schema type wins, else JSON, else raw
string.

The ns_token is *not* stripped first, though this said so long after the code
stopped doing it: parsing a copy with it deleted made every offset meaningless
against the bytes that arrived.
"""

import json
import re
from typing import Any, ClassVar

from .schema import (
    build_param_schemas,
    build_param_types,
    coerce_json_or_raw,
    coerce_param_value,
    declared_properties,
    item_schema,
    schema_type,
)
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

MINIMAX_NS = "]<]minimax[>["

# The ns_token is matched optionally wherever a tag can appear, as DSML does
# with its own marker, rather than deleted from a copy of the text first.
# Deleting it made every offset in the parsed copy meaningless against the
# bytes that actually arrived, which is why `parse` could only ever report
# "content precedes the call" and never "and this follows it".
_NS = r"(?:" + re.escape(MINIMAX_NS) + r")?"

# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <invoke name="NAME">" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
_NOT_NESTED = r"(?:(?!" + _NS + r"<invoke\s).)"
# No leading ns_token: `CALL_FILLERS` declares it and `markup_begin` walks
# back over it, so matching it here too is a second reader of one token -- and
# the expensive kind. A pattern opening with an optional group has no fixed
# first byte, so the scan tries every position rather than skipping to the
# next `<`: on 18 KB, 524 us against 3.8, and 262 -> 3.8 for `_PARAM_RE`. The
# other five formats open on a literal and never paid it.
_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*?)"
    + _NS
    + r"</invoke>|"
    + r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*)",
    re.DOTALL,
)
# --- Value reading -------------------------------------------------------
#
# An object or array argument is written as child elements -- the chat template
# spells this out and shows it. A flat scan for `<tag>value</tag>` reads
# `<tags><item>a</item></tags>` as an empty `tags` plus a stray `item`, so the
# schema has to decide where descending stops.

_XML_TAG_RE = re.compile(r"<(/?)\s*([^\s/>]+)([^>]*?)/?>", re.DOTALL)
_NAME_ATTR_RE = re.compile(
    r"""name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.DOTALL
)

# The wrapper tags, which are the only markup that can follow a value the model
# was cut off inside: `_INVOKE_RE` already ends the body before the next
# `<invoke ` or `</invoke>`. Without this the value ran on into the wrapper and
# the tool was called with the format's own markup as data.
_CALL_MARKUP = ("<tool_call>", "</tool_call>")

_STRING_SCHEMA_TYPES = ("string", "str", "text", "varchar", "char", "enum")
_ARRAY_SCHEMA_TYPES = ("array", "list", "tuple", "set")
_OBJECT_SCHEMA_TYPES = ("object", "dict", "map", "struct")


def _attr_name(attrs: str) -> str:
    m = _NAME_ATTR_RE.search(attrs or "")
    if m is None:
        return ""
    return next((g for g in m.groups() if g is not None), "").strip()


def _element_key(tag: str, attrs: str) -> str:
    # M3 names an argument with the element itself; older MiniMax builds write
    # `<parameter name="location">`.
    if tag == "parameter":
        return _attr_name(attrs) or tag
    return tag


def _element_body(text: str, match: "re.Match") -> tuple[str, int]:
    """Body of the element opened by ``match``, plus where to resume scanning.

    Depth-tracked so ``<opts><opts>..</opts></opts>`` closes on the right tag.
    An element a truncated call left open yields what arrived, which keeps a
    partial tool call usable.
    """
    tag = re.escape(match.group(2))
    opener = re.compile(r"<" + tag + r"(?:\s[^>]*)?>", re.DOTALL)
    closer = re.compile(r"</\s*" + tag + r"\s*>")
    depth, pos = 1, match.end()
    while True:
        nxt_close = closer.search(text, pos)
        if nxt_close is None:
            return _up_to_call_markup(text[match.end() :]), len(text)
        nxt_open = opener.search(text, pos)
        if nxt_open is not None and nxt_open.start() < nxt_close.start():
            depth += 1
            pos = nxt_open.end()
            continue
        depth -= 1
        if depth == 0:
            return text[match.end() : nxt_close.start()], nxt_close.end()
        pos = nxt_close.end()


def _up_to_call_markup(rest: str) -> str:
    cuts = [at for at in (rest.find(m) for m in _CALL_MARKUP) if at != -1]
    return rest[: min(cuts)] if cuts else rest


def _xml_children(
    body: str,
    expected: tuple[str, ...] = (),
    recover_lost_openers: bool = False,
) -> list[tuple[str, str]]:
    """Top-level ``(name, raw inner text)`` child elements; ``[]`` for a leaf.

    Also recovers an element whose opening tag never arrived: many of M3's added
    tokens are tag-shaped and marked special, so
    ``decode(skip_special_tokens=True)`` erases the opener and leaves the closing
    tag, and the argument would be dropped without a word.

    Recovery is narrow, because a stray ``</x>`` inside a value is usually just
    text: always directly inside ``<invoke>``, where nothing else can be outside
    an element, and deeper only for a name ``expected`` declares.
    """
    out: list[tuple[str, str]] = []
    pos = 0
    text_start = 0
    while True:
        m = _XML_TAG_RE.search(body, pos)
        if m is None:
            return out
        if m.group(1):  # closing tag with no opening tag before it
            name = m.group(2)
            leading = body[text_start : m.start()]
            if leading.strip() and (recover_lost_openers or name in expected):
                out.append((name, leading.strip("\n")))
            pos = text_start = m.end()
            continue
        inner, pos = _element_body(body, m)
        out.append((_element_key(m.group(2), m.group(3)), inner))
        text_start = pos


def _leaf_value(raw: str, kind: str | None) -> Any:
    # `strip_framing=False`: this format writes the value between its tags, so a
    # declared `string` can legitimately end in a newline.
    if kind is not None:
        return coerce_param_value(raw, kind, strip_framing=False)
    return coerce_json_or_raw(raw, None, strip_framing=False)


def _value_from_schema(raw: str, schema: Any) -> Any:
    """Read one element's inner text against its JSON-Schema fragment.

    A declared ``string`` stays text even when it contains markup: tool
    arguments really do carry HTML and ``a < b``. With no schema at all, the
    XML's own shape decides.
    """
    kind = schema_type(schema)

    if kind in _STRING_SCHEMA_TYPES:
        return _leaf_value(raw, kind)

    if kind in _ARRAY_SCHEMA_TYPES:
        items = item_schema(schema)
        children = _xml_children(raw)
        if children:
            return [_value_from_schema(text, items) for _, text in children]
        if not raw.strip():
            return []
        # A lone scalar where an array was declared: wrap it, so the client
        # still gets the type it asked for.
        value = _leaf_value(raw, kind)
        return value if isinstance(value, list) else [value]

    if kind in _OBJECT_SCHEMA_TYPES:
        props = declared_properties(schema)
        children = _xml_children(raw, expected=tuple(props))
        if children:
            return {
                name: _value_from_schema(text, props.get(name))
                for name, text in children
            }
        return {} if not raw.strip() else _leaf_value(raw, kind)

    children = _xml_children(raw)
    if not children:
        return _leaf_value(raw, kind)
    if all(name == "item" for name, _ in children):
        return [_value_from_schema(text, None) for _, text in children]
    return {name: _value_from_schema(text, None) for name, text in children}


_FIRST_TAG_RE = re.compile(r"^" + _NS + r"<([\w-]+)>")


def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`. The sweep that added that
    test to Qwen and GLM missed this format and DSML both, and "you emit
    `<invoke name="get_weather">` and then a `<city>Paris</city>` line" was
    still dispatched as a real call with the arguments filled in.

    The follower is a tag, because this format names parameters by the tag
    itself and so has no keyword to look for. Which tag is checked too when
    the request declared a schema for this tool -- otherwise `<br>` reads as
    the start of a call body -- but a tool declared without one still has to
    be callable, so an empty schema falls back to "any tag".
    """
    if not declared_tools_allow(name, param_types):
        return False
    # An undeclared schema falls back to "any tag", the same way an
    # empty one does below.
    types = param_types.get(name) or {}
    rest = body.lstrip()
    if not rest:
        return at_end
    tag = _FIRST_TAG_RE.match(rest)
    if tag is not None:
        return not types or tag.group(1) in types
    # A tag that has not finished arriving. At end of region that is all
    # there will ever be, so a prefix counts -- the rule `continues_a_call`
    # states and the other three formats have applied since it was written.
    # The followers are the declared tags, built from the request, because
    # this format names a parameter by the tag itself.
    #
    # An empty schema falls back to "any tag" here exactly as it does for a
    # complete tag four lines up. Gating on `bool(openers)` instead made a
    # zero-parameter tool the one tool whose truncated call could not be
    # recovered -- the same tool declared with one property recovered it --
    # and a truncation is precisely what this function exists to catch.
    followers = tuple(f"{MINIMAX_NS}<{t}>" for t in types) + tuple(
        f"<{t}>" for t in types
    ) or (f"{MINIMAX_NS}<", "<")
    return continues_a_call(rest, followers, arrived=not at_end)


class MiniMaxParser(ToolCallParser):
    NAME: ClassVar[str] = "minimax"
    # `<invoke name="` too: `_INVOKE_RE` matches it anywhere, so an invoke the
    # model wrote without the ns_token was a call when parsed whole and plain
    # text when streamed -- the read-ahead never opened a region for it. DSML
    # lists the same marker-less malform for the same reason.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        MINIMAX_NS,
        "<tool_call>",
        '<invoke name="',
    )
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)
    CALL_FILLERS: ClassVar[tuple[str, ...]] = (MINIMAX_NS,)
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = ("</invoke>",)

    # The ns_token opens a region like the other two, rather than being
    # framing the reader drops. It prefixes every tag including ones inside a
    # call, so calling it framing would delete it from an answer that merely
    # *mentions* it while an answer that mentions the tag it prefixes keeps
    # both -- the same token surviving or not depending on what followed it.
    # `parse` had exactly that split (every occurrence deleted from a copy of
    # the text when a call was found, all of them left when one was not); a
    # region that parses to nothing is released whole, so this way both
    # answers keep their bytes.

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(
            f"{MINIMAX_NS}<{k}>{v}{MINIMAX_NS}</{k}>" for k, v in args.items()
        )
        return (
            f'{MINIMAX_NS}<tool_call>{MINIMAX_NS}<invoke name="{name}">'
            f"{body}{MINIMAX_NS}</invoke>{MINIMAX_NS}</tool_call>"
        )

    @classmethod
    def call_prefill(cls, function_name: str | None = None) -> str:
        """See :meth:`ToolCallParser.call_prefill`. Checked against M3."""
        prefix = f"{MINIMAX_NS}<tool_call>\n"
        if function_name:
            # The name reaches the prompt as markup, so a quote or angle bracket
            # in it would open a tag of its own.
            if any(ch in function_name for ch in '"<>'):
                return ""
            prefix += f'{MINIMAX_NS}<invoke name="{function_name}">'
        return prefix

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the MiniMax-M3 ns_token tool-call format."""
        return MINIMAX_NS in text

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        param_schemas = build_param_schemas(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _INVOKE_RE.finditer(region):
            closed = m.group(1) is not None
            name = m.group(1) if closed else m.group(3)
            body = m.group(2) if closed else (m.group(4) or "")
            # Strip first, then judge. The other order -- `if not name` before
            # `name.strip()` -- let `name="   "` through the guard and out as
            # the empty string.
            name = (name or "").strip()
            if not usable_tool_name(name):
                continue
            if not closed and not _is_truncated_call(
                name, body, param_types, at_end=at_end
            ):
                continue
            properties = param_schemas.get(name, {})
            # The body is walked with the ns_token removed, which leaves plain
            # XML. Only the body: the span below is measured on the raw region,
            # because that is the text the caller cuts the markup out of.
            args = {
                key: _value_from_schema(raw, properties.get(key))
                for key, raw in _xml_children(
                    body.replace(MINIMAX_NS, ""),
                    expected=tuple(properties),
                    recover_lost_openers=True,
                )
            }
            spans.append(
                (cls.markup_begin(region, m.start()), cls.markup_end(region, m.end()))
            )
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
        return RegionParse(tuple(tool_calls), tuple(spans))
