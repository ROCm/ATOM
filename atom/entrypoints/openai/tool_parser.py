# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tool call parser for models that output tool calls.

Three on-the-wire formats are auto-detected and normalized into the OpenAI
``tool_calls`` structure:

1. Kimi-K2 special-token format::

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>ARGS_JSON<|tool_call_end|>
    <|tool_calls_section_end|>

2. Qwen3 (qwen3_coder / qwen3_xml) XML format::

    <tool_call>
    <function=NAME>
    <parameter=PNAME>VALUE</parameter>
    ...
    </function>
    </tool_call>

3. MiniMax-M3 XML format, in which every tag is prefixed with the
   ``]<]minimax[>[`` namespace token and each argument is named by its
   *element* rather than by a ``name=`` attribute::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">]<]minimax[>[<PNAME>VALUE]<]minimax[>[</PNAME>
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

   Arrays repeat an ``<item>`` child; objects nest named children. The
   ``<parameter name="PNAME">`` spelling is accepted as an alternative way to
   name an argument. Detection keys on the namespace token alone -- see
   :func:`_detect_format`.

Neither XML dialect carries value types, so when the request's ``tools`` schema
is supplied each parameter is coerced to the declared JSON-Schema type (int,
float, bool, null, object, array); otherwise it is left as a string. This
mirrors the qwen3_coder/qwen3_xml and minimax_m2 parsers in vLLM and SGLang.

OpenAI format:
    {"tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "NAME", "arguments": "ARGS_JSON"}}]}
"""

import ast
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .stream_buffer import releasable_prefix_len


def _unique_tool_call_id() -> str:
    # OpenAI tool_call ids must be unique across the whole conversation, not just
    # within one response. A per-response index (call_0, call_1, ...) collides
    # across turns -> clients (e.g. qwen-code) dedupe by id and silently ignore
    # every repeat, causing an infinite tool-call retry loop. Use a random id.
    return f"call_{uuid.uuid4().hex}"


@dataclass
class ToolCall:
    """Parsed tool call in OpenAI format."""

    id: str
    type: str
    function: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function}


# ---------------------------------------------------------------------------
# Format markers
# ---------------------------------------------------------------------------

_KIMI_SECTION_OPEN = "<|tool_calls_section_begin|>"

_QWEN_TOOL_PREFIX = "<function="
_QWEN_SECTION_OPEN = "<tool_call>"

# MiniMax-M3 prefixes every tool-call tag with this namespace token (which the
# tokenizer keeps under skip_special_tokens=True, so the parser really does see
# it). Older MiniMax builds emit the same XML without the prefix, so the parser
# strips the token and works on the plain XML underneath, accepting both.
_MINIMAX_NS = "]<]minimax[>["
_MINIMAX_SECTION_OPEN = "<tool_call>"
_MINIMAX_INVOKE_PREFIX = "<invoke name="
_MINIMAX_INVOKE_CLOSE = "</invoke>"
# Ordered so the namespaced spellings (which start earlier in the raw text) are
# found before their bare equivalents; used for the streaming content cut.
_MINIMAX_STREAM_MARKERS = (
    _MINIMAX_NS + _MINIMAX_SECTION_OPEN,
    _MINIMAX_NS + _MINIMAX_INVOKE_PREFIX,
    _MINIMAX_INVOKE_PREFIX,
)

# Every marker that can start a tool-call span, for streaming look-ahead.
# Deliberately excludes the bare ``<invoke name=`` — see _detect_format.
_STREAM_MARKERS = (
    _KIMI_SECTION_OPEN,
    _QWEN_SECTION_OPEN,
    _QWEN_TOOL_PREFIX,
    _MINIMAX_NS,
)


def _detect_format(text: str) -> Optional[str]:
    """Return ``"kimi"`` / ``"qwen"`` / ``"minimax"``, or None if undecided.

    Probe order preserves what the pre-existing dialects already claimed, so
    adding MiniMax cannot change how another model's output is parsed:

    * **Kimi first.** Its special tokens are unambiguous, and ``<function=`` can
      legitimately appear *inside* Kimi arguments — which is why the original
      non-streaming detector required the Kimi marker to be absent before
      accepting Qwen.
    * **Qwen keyed on ``<function=``**, not on its ``<tool_call>`` wrapper: a bare
      ``<tool_call>`` also introduces the Hermes JSON dialect, which this parser
      cannot read, and claiming it would silently drop the call instead of
      leaving the raw text in ``content`` where the client can see it.
    * **MiniMax last, and only on its namespace token.** ``<invoke name=`` alone
      is too weak a signal — it can occur in ordinary prose or code from any
      model — so it is deliberately *not* a trigger. Once the namespace token has
      decided the format, un-namespaced tags inside the block still parse.
    """
    if _KIMI_SECTION_OPEN in text:
        return "kimi"
    if _QWEN_TOOL_PREFIX in text:
        return "qwen"
    if _MINIMAX_NS in text:
        return "minimax"
    return None


def _releasable_content_len(buf: str) -> int:
    """How much of ``buf`` can be streamed out as plain content right now.

    Without the partial-marker look-ahead a stream containing a bare ``<``
    (``"a < b"``) would stall in the buffer until end of generation.
    """
    return releasable_prefix_len(buf, _STREAM_MARKERS)


# ---------------------------------------------------------------------------
# Shared XML parameter handling (Qwen3 + MiniMax)
# ---------------------------------------------------------------------------


def _build_param_types(tools: Optional[list]) -> Dict[str, Dict[str, Any]]:
    """Map ``function_name -> {param_name: json_schema_type}`` from request tools.

    Accepts OpenAI (``{"type": "function", "function": {...}}``) and bare
    (``{"name": ..., "parameters"/"input_schema": {...}}``) tool entries.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        props = schema.get("properties") if isinstance(schema, dict) else None
        out[name] = {
            k: (v.get("type") if isinstance(v, dict) else None)
            for k, v in (props or {}).items()
        }
    return out


def _build_param_schemas(tools: Optional[list]) -> Dict[str, Dict[str, Any]]:
    """Map ``function_name -> {param_name: json_schema_fragment}`` from tools.

    Like :func:`_build_param_types` but keeps the whole sub-schema, which the
    MiniMax parser needs to type nested objects and array items.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        out[name] = {
            k: (v if isinstance(v, dict) else {})
            for k, v in _property_schemas(schema).items()
        }
    return out


def _coerce_param_value(value: str, ptype: Any) -> Any:
    """Coerce a string parameter value to its declared JSON-Schema type.

    The value is taken verbatim: trimming belongs to whichever dialect framed it,
    since MiniMax writes values inline and a declared ``string`` really can end in
    a newline (a file body bound for ``write_file``). The conversions below
    tolerate surrounding whitespace on their own.

    No schema type (string/unknown) -> returned unchanged. Conversion failures
    fall back to the original string rather than raising.
    """
    if ptype is None:
        return value
    t = str(ptype).lower()
    try:
        if t in ("string", "str", "text", "varchar", "char", "enum"):
            return value
        if t in ("null", "none"):
            return None
        if t.startswith(("int", "uint", "long", "short", "unsigned")):
            return int(value)
        if t.startswith(("num", "float", "double", "decimal")):
            f = float(value)
            return int(f) if f.is_integer() else f
        if t.startswith(("bool", "binary")):
            return value.strip().lower() == "true"
        if t.startswith(("object", "dict", "map", "array", "list", "tuple")):
            try:
                return json.loads(value)
            except Exception:
                # safer for single-quoted Python literals
                return ast.literal_eval(value)
    except Exception:
        return value
    return value


def _make_tool_call(name: str, args: Dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=_unique_tool_call_id(),
        type="function",
        function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    )


def _leading_content(text: str, markers: Tuple[str, ...]) -> str:
    """Text before the first of ``markers`` (all of ``text`` if none present)."""
    starts = [i for i in (text.find(m) for m in markers) if i != -1]
    return text[: min(starts)] if starts else text


# ---------------------------------------------------------------------------
# Qwen3 XML tool-call format (qwen3_coder / qwen3_xml)
# ---------------------------------------------------------------------------

_QWEN_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_QWEN_PARAM_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


def _parse_qwen_function(
    fn_text: str, param_types: Dict[str, Dict[str, Any]]
) -> Optional[ToolCall]:
    """Parse the inside of one ``<function=NAME>...`` block into a ToolCall."""
    gt = fn_text.find(">")
    if gt == -1:
        return None
    name = fn_text[:gt].strip()
    if not name:
        return None
    body = fn_text[gt + 1 :]
    types = param_types.get(name, {})
    args: Dict[str, Any] = {}
    for pm in _QWEN_PARAM_RE.finditer(body):
        seg = pm.group(1)
        if seg is None:
            continue
        pgt = seg.find(">")
        if pgt == -1:
            continue
        pname = seg[:pgt].strip()
        pval = seg[pgt + 1 :]
        if pname:
            # Qwen frames the value on its own line, and an omitted closing
            # </parameter> runs the match on into the next tag.
            args[pname] = _coerce_param_value(pval.strip("\n"), types.get(pname))
    return _make_tool_call(name, args)


def _parse_qwen_xml(text: str, tools: Optional[list]) -> Tuple[str, List[ToolCall]]:
    """Parse Qwen3 XML tool calls; return (leading_content, tool_calls)."""
    param_types = _build_param_types(tools)
    content = _leading_content(text, (_QWEN_SECTION_OPEN, _QWEN_TOOL_PREFIX))
    tool_calls: List[ToolCall] = []
    for fm in _QWEN_FUNCTION_RE.finditer(text):
        fn_text = fm.group(1) if fm.group(1) is not None else fm.group(2)
        if not fn_text:
            continue
        tc = _parse_qwen_function(fn_text, param_types)
        if tc is not None:
            tool_calls.append(tc)
    return content.strip(), tool_calls


# ---------------------------------------------------------------------------
# MiniMax XML tool-call format
# ---------------------------------------------------------------------------
#
# Unlike Qwen3, MiniMax names each argument with the *element* rather than a
# ``name=`` attribute, and expands nested values recursively:
#
#     <invoke name="search">
#       <q>beijing air quality</q>
#       <tags><item>china</item><item>env</item></tags>
#       <opts><unit>c</unit></opts>
#     </invoke>
#
# Arguments are therefore parsed as a tree. The request's JSON Schema drives that
# walk wherever it is available -- a declared ``string`` stays a string even when
# its value contains markup -- and the shape of the XML itself is only the
# fallback for arguments no schema describes.

_MINIMAX_INVOKE_RE = re.compile(r"<invoke\s+([^>]*?)>(.*?)(?:</invoke>|$)", re.DOTALL)
_XML_TAG_RE = re.compile(r"<(/?)\s*([^\s/>]+)([^>]*?)/?>", re.DOTALL)
_NAME_ATTR_RE = re.compile(
    r"""name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.DOTALL
)


def _attr_name(attrs: str) -> str:
    """Value of the ``name=`` attribute (quoted or bare); "" when absent."""
    m = _NAME_ATTR_RE.search(attrs or "")
    if m is None:
        return ""
    return next((g for g in m.groups() if g is not None), "").strip()


def _element_key(tag: str, attrs: str) -> str:
    """Argument name an element contributes.

    MiniMax-M3 uses the element name (``<location>``). Older MiniMax builds and
    the Qwen-style spelling use ``<parameter name="location">``, so a
    ``parameter`` element defers to its ``name=`` attribute.
    """
    if tag == "parameter":
        return _attr_name(attrs) or tag
    return tag


def _element_body(text: str, match: "re.Match") -> Tuple[str, int]:
    """Body of the element opened by ``match`` plus where to resume scanning.

    Tracks same-name nesting so ``<opts><opts>..</opts></opts>`` closes on the
    right tag. An unclosed element (output truncated mid-call) yields the rest
    of the text, which keeps a partial tool call usable.
    """
    tag = re.escape(match.group(2))
    opener = re.compile(r"<" + tag + r"(?:\s[^>]*)?>", re.DOTALL)
    closer = re.compile(r"</\s*" + tag + r"\s*>")
    depth, pos = 1, match.end()
    while True:
        nxt_close = closer.search(text, pos)
        if nxt_close is None:
            return text[match.end() :], len(text)
        nxt_open = opener.search(text, pos)
        if nxt_open is not None and nxt_open.start() < nxt_close.start():
            depth += 1
            pos = nxt_open.end()
            continue
        depth -= 1
        if depth == 0:
            return text[match.end() : nxt_close.start()], nxt_close.end()
        pos = nxt_close.end()


_STRING_SCHEMA_TYPES = ("string", "str", "text", "varchar", "char", "enum")
_ARRAY_SCHEMA_TYPES = ("array", "list", "tuple", "set")
_OBJECT_SCHEMA_TYPES = ("object", "dict", "map", "struct")


_COMPOSITION_KEYWORDS = ("oneOf", "anyOf", "allOf")
# Bounded because this runs per parsed element: an absurdly nested request
# schema would otherwise be re-walked in full every time.
_MAX_COMPOSITION_DEPTH = 4


def _schema_branches(schema: Any) -> List[Dict[str, Any]]:
    """``schema`` itself followed by its ``oneOf`` / ``anyOf`` / ``allOf`` arms.

    A tool may declare its arguments inside a composition rather than directly
    (``{"type": "object", "oneOf": [{"properties": {...}}, ...]}``). Reading only
    the top level then finds no ``properties`` at all and every argument loses
    its declared type, so a number comes back as the string ``"42"``. Flattening
    the arms lets the rest of the schema handling stay composition-unaware.

    Outermost-first, so the enclosing schema outranks an arm that respells a name.
    """
    out: List[Dict[str, Any]] = []
    pending: List[Tuple[Any, int]] = [(schema, 0)]
    while pending:
        node, depth = pending.pop(0)
        if not isinstance(node, dict):
            continue
        out.append(node)
        if depth >= _MAX_COMPOSITION_DEPTH:
            continue
        for keyword in _COMPOSITION_KEYWORDS:
            arms = node.get(keyword)
            if isinstance(arms, (list, tuple)):
                pending.extend((arm, depth + 1) for arm in arms)
    return out


def _schema_type(schema: Any) -> Optional[str]:
    """Lower-cased ``type`` declared by a JSON-Schema fragment, if any.

    A union spelling (``["string", "null"]``, how an optional value is written)
    resolves to its first non-null member. A composition that declares its type
    only on the arms (``{"oneOf": [{"type": "object"}, ...]}``) resolves to the
    first arm that names one.
    """
    for node in _schema_branches(schema):
        declared = node.get("type")
        if isinstance(declared, (list, tuple)):
            declared = next(
                (d for d in declared if str(d).lower() not in ("null", "none")), None
            )
        if declared is not None:
            return str(declared).lower()
    return None


def _property_schemas(schema: Any) -> Dict[str, Any]:
    """``properties`` of an object-schema fragment ({} when there are none).

    Merged across composition arms (see :func:`_schema_branches`); the outermost
    declaration of a name wins.
    """
    out: Dict[str, Any] = {}
    for node in _schema_branches(schema):
        props = node.get("properties")
        if isinstance(props, dict):
            for name, sub_schema in props.items():
                out.setdefault(name, sub_schema)
    return out


def _item_schema(schema: Any) -> Any:
    """``items`` of an array-schema fragment, following composition arms."""
    for node in _schema_branches(schema):
        items = node.get("items")
        if items is not None:
            return items
    return None


def _xml_children(
    body: str,
    expected: Tuple[str, ...] = (),
    recover_lost_openers: bool = False,
) -> List[Tuple[str, str]]:
    """Top-level ``(name, raw inner text)`` child elements of ``body``.

    Returns ``[]`` for a leaf. Values stay raw here; interpreting them is
    :func:`_value_from_schema`'s job, so an argument's shape is decided by its
    schema rather than by whatever punctuation its text happens to contain.

    This also recovers an element whose *opening* tag never reached the parser.
    MiniMax names each argument with its element, and many of MiniMax-M3's added
    tokens are tag-shaped (``<filename>``, ``<filepath>``, ``<file_content>``,
    ``<commit_message>``, ...). Those are marked special, so the engine's
    ``decode(skip_special_tokens=True)`` erases them, while the closing tag --
    ordinary text to the tokenizer -- survives. Without recovery the argument
    and its value are dropped silently, required ones included, and the client
    sees a tool call that simply lacks a parameter.

    Recovery is deliberately narrow, because a stray ``</x>`` inside a leaf
    value is ordinarily literal text:

    * ``recover_lost_openers`` (set only directly inside ``<invoke>``): always
      recovered. Text outside an element carries no meaning there, so a stray
      closing tag can only be a lost opening tag.
    * nested: recovered only when ``expected`` -- the enclosing object's declared
      property names -- contains the closing tag's name.
    """
    out: List[Tuple[str, str]] = []
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
                # With no opening tag the value's start is unknown, so the
                # span also caught whatever layout preceded it.
                out.append((name, leading.strip("\n")))
            pos = text_start = m.end()
            continue
        inner, pos = _element_body(body, m)
        out.append((_element_key(m.group(2), m.group(3)), inner))
        text_start = pos


def _value_from_schema(raw: str, schema: Any) -> Any:
    """Interpret one element's raw inner text against its JSON-Schema fragment.

    The declared type decides the shape:

    * ``string`` stays text, even when the value contains markup. Tool arguments
      really do carry HTML, C++ templates and ``a < b``; reading those as child
      elements shreds the value into a nested object *and* silently drops
      whatever preceded the first tag.
    * ``array`` / ``object`` recurse, typing each element or property.
    * anything else, or no schema at all, falls back to the shape of the XML:
      all-``item`` children are a list, named children an object, and no
      children a leaf coerced to the declared scalar type.
    """
    kind = _schema_type(schema)

    if kind in _STRING_SCHEMA_TYPES:
        return _coerce_param_value(raw, kind)

    if kind in _ARRAY_SCHEMA_TYPES:
        items = _item_schema(schema)
        children = _xml_children(raw)
        if children:
            return [_value_from_schema(text, items) for _, text in children]
        if not raw.strip():
            return []
        # A lone scalar where an array was declared: keep a JSON spelling as
        # parsed, otherwise wrap it so the client still gets the declared type.
        value = _coerce_param_value(raw, kind)
        return value if isinstance(value, list) else [value]

    if kind in _OBJECT_SCHEMA_TYPES:
        props = _property_schemas(schema)
        children = _xml_children(raw, expected=tuple(props))
        if children:
            return {
                name: _value_from_schema(text, props.get(name))
                for name, text in children
            }
        return {} if not raw.strip() else _coerce_param_value(raw, kind)

    children = _xml_children(raw)
    if not children:
        return _coerce_param_value(raw, kind)
    if all(name == "item" for name, _ in children):
        return [_value_from_schema(text, None) for _, text in children]
    return {name: _value_from_schema(text, None) for name, text in children}


def _parse_minimax_invoke(
    attrs: str, body: str, schemas: Dict[str, Dict[str, Any]]
) -> Optional[ToolCall]:
    """Parse one ``<invoke name="NAME">...</invoke>`` block into a ToolCall."""
    name = _attr_name(attrs)
    if not name:
        return None
    properties = schemas.get(name, {})
    args = {
        key: _value_from_schema(raw, properties.get(key))
        for key, raw in _xml_children(
            body, expected=tuple(properties), recover_lost_openers=True
        )
    }
    return _make_tool_call(name, args)


def _parse_minimax_xml(text: str, tools: Optional[list]) -> Tuple[str, List[ToolCall]]:
    """Parse MiniMax XML tool calls; return (leading_content, tool_calls)."""
    schemas = _build_param_schemas(tools)
    # Dropping the namespace token leaves plain XML, which is also exactly what
    # the un-namespaced MiniMax spelling looks like — one parser serves both.
    plain = text.replace(_MINIMAX_NS, "")
    content = _leading_content(plain, (_MINIMAX_SECTION_OPEN, _MINIMAX_INVOKE_PREFIX))
    tool_calls: List[ToolCall] = []
    for im in _MINIMAX_INVOKE_RE.finditer(plain):
        tc = _parse_minimax_invoke(im.group(1), im.group(2), schemas)
        if tc is not None:
            tool_calls.append(tc)
    return content.strip(), tool_calls


# ---------------------------------------------------------------------------
# Forced tool calls
# ---------------------------------------------------------------------------


def tool_call_prefill(
    prompt: str, function_name: Optional[str] = None
) -> Optional[str]:
    """Assistant-turn prefix that forces the model into a tool call.

    ATOM has no constrained decoding, so ``tool_choice: "required"`` cannot be
    enforced by a grammar. It can be enforced by *starting* the call: append the
    opening tokens of the model's own tool-call dialect to the generation prompt
    and the only continuation left is the call itself. Passing
    ``function_name`` also opens the named ``invoke``, which pins the choice.

    The dialect is read off the rendered prompt, because a chat template that
    supports tools spells its tool-call syntax out in the instructions it
    renders. Returns ``None`` when no known dialect is advertised, in which case
    the caller falls back to prompting alone.

    The returned prefix is not part of the model's output, so it must be handed
    to the parser as well (``tool_call_prefix`` in :mod:`.serving_chat`).
    """
    if _MINIMAX_NS + _MINIMAX_SECTION_OPEN in prompt:
        prefix = _MINIMAX_NS + _MINIMAX_SECTION_OPEN + "\n"
        if function_name:
            prefix += f'{_MINIMAX_NS}<invoke name="{function_name}">'
        return prefix
    if _QWEN_TOOL_PREFIX in prompt or _QWEN_SECTION_OPEN in prompt:
        prefix = _QWEN_SECTION_OPEN + "\n"
        if function_name:
            prefix += f"{_QWEN_TOOL_PREFIX}{function_name}>\n"
        return prefix
    if _KIMI_SECTION_OPEN in prompt:
        prefix = _KIMI_SECTION_OPEN
        if function_name:
            prefix += (
                f"<|tool_call_begin|>functions.{function_name}:0"
                "<|tool_call_argument_begin|>"
            )
        return prefix
    return None


# ---------------------------------------------------------------------------
# Non-streaming entry point
# ---------------------------------------------------------------------------


def parse_tool_calls(
    text: str, tools: Optional[list] = None
) -> Tuple[str, List[ToolCall]]:
    """Parse tool calls from model output text.

    Args:
        text: Raw model output that may contain tool calls (Kimi token format,
            Qwen3 XML format, or MiniMax XML format).
        tools: Optional request tool definitions; used to type-coerce XML
            parameter values to their declared JSON-Schema types.

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call sections removed.
    """
    fmt = _detect_format(text)

    if fmt == "minimax":
        return _parse_minimax_xml(text, tools)

    if fmt == "qwen":
        return _parse_qwen_xml(text, tools)

    if fmt != "kimi":
        return text, []

    # Kimi-K2 special-token format
    section_match = re.search(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
        text,
        flags=re.DOTALL,
    )
    if not section_match:
        # Unclosed section (truncated output)
        unclosed = re.search(
            r"<\|tool_calls_section_begin\|>(.*?)$", text, flags=re.DOTALL
        )
        content = text[: unclosed.start()]
        tool_calls = _parse_tool_call_entries(unclosed.group(1))
        return content.strip(), tool_calls

    content = text[: section_match.start()]
    tool_calls = _parse_tool_call_entries(section_match.group(1))

    return content.strip(), tool_calls


def _parse_tool_call_entries(section_text: str) -> List[ToolCall]:
    """Parse individual tool call entries from the section content."""
    tool_calls = []
    pattern = re.compile(
        r"<\|tool_call_begin\|>"
        r"functions\.(\w+):(\d+)"
        r"<\|tool_call_argument_begin\|>"
        r"(.*?)"
        r"<\|tool_call_end\|>",
        re.DOTALL,
    )
    for match in pattern.finditer(section_text):
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


# ---------------------------------------------------------------------------
# Streaming entry point
# ---------------------------------------------------------------------------


@dataclass
class ToolCallStreamParser:
    """Stateful streaming parser for tool calls (Kimi tokens, Qwen3 or MiniMax XML).

    Processes text chunks and emits structured events:
    - ("content", text) — regular content before tool calls
    - ("tool_call_start", {"index": N, "id": ..., "function": {"name": ..., "arguments": ""}})
    - ("tool_call_args", {"index": N, "function": {"arguments": chunk}})
    - ("tool_call_end", None) — all tool calls complete

    The wire format is auto-detected from the first chunks. For both XML formats
    content is streamed normally and the tool-call block is buffered, then parsed
    once complete (robust against partial-XML streaming edge cases); ``tools``
    enables JSON-Schema type coercion of parameter values. MiniMax emits each
    ``<invoke>`` as soon as it closes, so parallel calls arrive incrementally.

    ``enabled=False`` disables tool-call recognition entirely and streams
    everything as content — used for ``tool_choice: "none"``.

    Kimi states:
        0 = normal content (no tool call tokens seen)
        1 = inside tool_calls_section (buffering)
        2 = done (after tool_calls_section_end)
    """

    state: int = 0
    buf: str = ""
    current_index: int = 0
    _emitted_calls: int = 0
    tools: Optional[list] = None
    fmt: Optional[str] = None  # None (undecided) | "kimi" | "qwen" | "minimax"
    enabled: bool = True

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples."""
        if not self.enabled:
            return [("content", text)] if text else []

        if self.fmt is None:
            self.buf += text
            fmt = _detect_format(self.buf)
            if fmt is None:
                # Release everything that cannot still grow into a marker.
                cut = _releasable_content_len(self.buf)
                if cut <= 0:
                    return []
                out = [("content", self.buf[:cut])]
                self.buf = self.buf[cut:]
                return out
            self.fmt = fmt
            # Format decided: replay the accumulated buffer through the handler.
            text, self.buf = self.buf, ""

        if self.fmt == "minimax":
            return self._process_minimax(text)
        if self.fmt == "qwen":
            return self._process_qwen(text)
        return self._process_kimi(text)

    # -- shared XML helpers -------------------------------------------------
    def _consume_leading_content(self, markers: Tuple[str, ...]) -> list:
        """State 0 for the XML formats: emit content, enter the tool span."""
        results: list = []
        starts = [i for i in (self.buf.find(m) for m in markers) if i != -1]
        if starts:
            m = min(starts)
            before = self.buf[:m]
            if before:
                results.append(("content", before))
            self.buf = self.buf[m:]
            self.state = 1
        else:
            cut = _releasable_content_len(self.buf)
            if cut > 0:
                results.append(("content", self.buf[:cut]))
                self.buf = self.buf[cut:]
        return results

    def _emit_tool_call(self, tc: ToolCall) -> list:
        results = [
            (
                "tool_call_start",
                {
                    "index": self.current_index,
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function["name"], "arguments": ""},
                },
            ),
            (
                "tool_call_args",
                {
                    "index": self.current_index,
                    "function": {"arguments": tc.function["arguments"]},
                },
            ),
        ]
        self.current_index += 1
        self._emitted_calls += 1
        return results

    # -- Qwen3 XML ----------------------------------------------------------
    def _process_qwen(self, text: str) -> list:
        self.buf += text
        if self.state == 0:
            return self._consume_leading_content(
                (_QWEN_SECTION_OPEN, _QWEN_TOOL_PREFIX)
            )
        return []

    def _flush_qwen(self) -> list:
        results: list = []
        if self.state == 0:
            if self.buf:
                results.append(("content", self.buf))
                self.buf = ""
            return results
        # state 1: parse the complete (or trailing) tool-call block.
        _content, tool_calls = _parse_qwen_xml(self.buf, self.tools)
        self.buf = ""
        for tc in tool_calls:
            results.extend(self._emit_tool_call(tc))
        if self._emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results

    # -- MiniMax XML --------------------------------------------------------
    def _process_minimax(self, text: str) -> list:
        results: list = []
        self.buf += text
        if self.state == 0:
            results.extend(self._consume_leading_content(_MINIMAX_STREAM_MARKERS))
        if self.state == 1:
            results.extend(self._drain_minimax_invokes())
        return results

    def _drain_minimax_invokes(self) -> list:
        """Emit every ``<invoke>...</invoke>`` block that has closed so far."""
        results: list = []
        while True:
            end = self.buf.find(_MINIMAX_INVOKE_CLOSE)
            if end == -1:
                break
            block, self.buf = (
                self.buf[: end + len(_MINIMAX_INVOKE_CLOSE)],
                self.buf[end + len(_MINIMAX_INVOKE_CLOSE) :],
            )
            _content, tool_calls = _parse_minimax_xml(block, self.tools)
            for tc in tool_calls:
                results.extend(self._emit_tool_call(tc))
        return results

    def _flush_minimax(self) -> list:
        results: list = []
        if self.state == 0:
            if self.buf:
                results.append(("content", self.buf))
                self.buf = ""
            return results
        results.extend(self._drain_minimax_invokes())
        # A truncated final <invoke> (max_tokens hit mid-call) still carries a
        # usable name + arguments; parse what arrived.
        if _MINIMAX_INVOKE_PREFIX in self.buf:
            _content, tool_calls = _parse_minimax_xml(self.buf, self.tools)
            for tc in tool_calls:
                results.extend(self._emit_tool_call(tc))
        self.buf = ""
        if self._emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results

    # -- Kimi tokens --------------------------------------------------------
    def _process_kimi(self, text: str) -> list:
        results = []

        if self.state == 0:
            self.buf += text
            if _KIMI_SECTION_OPEN in self.buf:
                before = self.buf.split(_KIMI_SECTION_OPEN)[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split(_KIMI_SECTION_OPEN, 1)[1]
                results.extend(self._process_buffer())

        elif self.state == 1:
            self.buf += text
            if "<|tool_calls_section_end|>" in self.buf:
                remaining = self.buf.split("<|tool_calls_section_end|>")[0]
                self.buf = remaining
                results.extend(self._process_buffer())
                results.append(("tool_call_end", None))
                self.state = 2
                self.buf = ""
            else:
                results.extend(self._process_buffer())

        return results

    def _process_buffer(self) -> list:
        """Extract complete tool call entries from the buffer."""
        results = []
        while "<|tool_call_begin|>" in self.buf and "<|tool_call_end|>" in self.buf:
            match = re.search(
                r"<\|tool_call_begin\|>"
                r"functions\.(\w+):(\d+)"
                r"<\|tool_call_argument_begin\|>"
                r"(.*?)"
                r"<\|tool_call_end\|>",
                self.buf,
                re.DOTALL,
            )
            if not match:
                break

            name = match.group(1)
            index = int(match.group(2))
            arguments = match.group(3).strip()

            tool_id = f"functions.{name}:{index}"
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": index,
                        "id": tool_id,
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
            self._emitted_calls += 1

        return results

    def flush(self) -> list:
        """Flush remaining buffer content."""
        if not self.enabled:
            return []
        if self.fmt == "minimax":
            return self._flush_minimax()
        if self.fmt == "qwen":
            return self._flush_qwen()
        results = []
        if self.state == 0 and self.buf:
            results.append(("content", self.buf))
            self.buf = ""
        elif self.state == 1:
            results.extend(self._process_buffer())
            if self._emitted_calls > 0:
                results.append(("tool_call_end", None))
        elif self.fmt is None and self.buf:
            # Undecided at EOS: no tool markers ever appeared -> plain content.
            results.append(("content", self.buf))
            self.buf = ""
        return results
