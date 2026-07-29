# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tool call parser for models that output tool calls.

Two on-the-wire formats are auto-detected and normalized into the OpenAI
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

The Qwen XML carries no value types, so when the request's ``tools`` schema is
supplied each parameter is coerced to the declared JSON-Schema type (int, float,
bool, null, object, array); otherwise it is left as a string. This mirrors the
qwen3_coder/qwen3_xml parsers in vLLM and SGLang.

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
# Qwen3 XML tool-call format (qwen3_coder / qwen3_xml)
# ---------------------------------------------------------------------------

_QWEN_TOOL_PREFIX = "<function="
_QWEN_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_QWEN_PARAM_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


def _is_qwen_xml(text: str) -> bool:
    """Detect the Qwen3 XML tool-call format (and not the Kimi token format)."""
    return _QWEN_TOOL_PREFIX in text and "<|tool_calls_section_begin|>" not in text


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


def _coerce_param_value(value: str, ptype: Any) -> Any:
    """Coerce a string parameter value to its declared JSON-Schema type.

    No schema type (string/unknown) -> returned unchanged. Conversion failures
    fall back to the original string rather than raising.
    """
    v = value.strip("\n")
    if ptype is None:
        return v
    t = str(ptype).lower()
    try:
        if t in ("string", "str", "text", "varchar", "char", "enum"):
            return v
        if t in ("null", "none"):
            return None
        if t.startswith(("int", "uint", "long", "short", "unsigned")):
            return int(v)
        if t.startswith(("num", "float", "double", "decimal")):
            f = float(v)
            return int(f) if f.is_integer() else f
        if t.startswith(("bool", "binary")):
            return v.strip().lower() == "true"
        if t.startswith(("object", "dict", "map", "array", "list", "tuple")):
            try:
                return json.loads(v)
            except Exception:
                return ast.literal_eval(v)  # safer for single-quoted Python literals
    except Exception:
        return v
    return v


def _parse_qwen_function(
    fn_text: str, param_types: Dict[str, Dict[str, Any]], index: int
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
            args[pname] = _coerce_param_value(pval, types.get(pname))
    return ToolCall(
        id=_unique_tool_call_id(),
        type="function",
        function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    )


def _parse_qwen_xml(text: str, tools: Optional[list]) -> Tuple[str, List[ToolCall]]:
    """Parse Qwen3 XML tool calls; return (leading_content, tool_calls)."""
    param_types = _build_param_types(tools)
    # Content precedes the first tool marker.
    markers = [
        i for i in (text.find("<tool_call>"), text.find(_QWEN_TOOL_PREFIX)) if i != -1
    ]
    content = text[: min(markers)] if markers else text
    tool_calls: List[ToolCall] = []
    for fm in _QWEN_FUNCTION_RE.finditer(text):
        fn_text = fm.group(1) if fm.group(1) is not None else fm.group(2)
        if not fn_text:
            continue
        tc = _parse_qwen_function(fn_text, param_types, len(tool_calls))
        if tc is not None:
            tool_calls.append(tc)
    return content.strip(), tool_calls


# ---------------------------------------------------------------------------
# DeepSeek-V4 DSML tool-call format
# ---------------------------------------------------------------------------
#
#     <｜DSML｜tool_calls>
#     <｜DSML｜invoke name="NAME">
#     <｜DSML｜parameter name="PNAME" string="true|false">VALUE</｜DSML｜parameter>
#     ...
#     </｜DSML｜invoke>
#     </｜DSML｜tool_calls>
#
# string="true"  -> value is a raw string; string="false" -> value is JSON.
# DeepSeek-V4-Flash occasionally malforms this (singular ``tool_call``, a missing
# ``invoke`` wrapper, or params without ``string=``); the parser recovers those
# best-effort: it infers a dropped tool name from the parameter signature vs the
# request's ``tools`` and infers a missing value type from the schema / JSON.

_DSML = "｜DSML｜"
# The model often DROPS the ``｜DSML｜`` marker and emits bare
# ``<invoke name=...>``/``<parameter ...>``/``<tool_calls>`` tags, so the marker
# is matched OPTIONALLY everywhere.
_OPT = r"(?:" + re.escape(_DSML) + r")?"      # optional ｜DSML｜ prefix
_DSML_PARAM_RE = re.compile(
    r"<" + _OPT + r'parameter\s+name="(.*?)"(?:\s+string="(true|false)")?\s*>'
    r"(.*?)</" + _OPT + r"parameter>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<" + _OPT + r'invoke\s+name="(.*?)"\s*>(.*?)</' + _OPT + r"invoke>",
    re.DOTALL,
)
# Region-start markers, both marked and marker-less variants.
_DSML_STARTS = (
    "<" + _DSML + "tool_call",   # marked (covers tool_call / tool_calls)
    "<" + _DSML + "invoke",      # marked invoke
    "<invoke name=",             # marker-less invoke (common malform)
    "<tool_calls>",              # marker-less section open
)


def _dsml_start(text: str) -> int:
    """Index of the earliest DSML tool-call marker (marked or marker-less), or -1."""
    positions = [i for i in (text.find(m) for m in _DSML_STARTS) if i != -1]
    return min(positions) if positions else -1


def _is_dsml(text: str) -> bool:
    return _dsml_start(text) != -1


def _dsml_coerce(value: str, string_attr: Optional[str], ptype: Any) -> Any:
    if string_attr == "true":
        return value
    if string_attr == "false":
        try:
            return json.loads(value)
        except Exception:
            return value
    # attr absent -> use declared schema type if known, else infer via JSON.
    if ptype is not None:
        return _coerce_param_value(value, ptype)
    v = value.strip()
    try:
        return json.loads(v)
    except Exception:
        return v


def _infer_dsml_name(arg_names: set, param_types: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Pick the request tool whose parameter set best matches ``arg_names``."""
    best, best_score = None, -1e9
    for name, props in param_types.items():
        p = set(props)
        if not p:
            continue
        score = len(p & arg_names) - 0.1 * len(p ^ arg_names)
        if score > best_score:
            best_score, best = score, name
    return best


def _parse_dsml(text: str, tools: Optional[list]) -> Tuple[str, List[ToolCall]]:
    """Parse DeepSeek-V4 DSML tool calls; return (leading_content, tool_calls)."""
    param_types = _build_param_types(tools)
    start = _dsml_start(text)
    if start == -1:
        return text.strip(), []
    content = text[:start]
    region = text[start:]

    calls: List[Tuple[str, Dict[str, Any]]] = []
    invokes = list(_DSML_INVOKE_RE.finditer(region))
    if invokes:
        for m in invokes:
            name = m.group(1)
            types = param_types.get(name, {})
            args = {
                pm.group(1): _dsml_coerce(pm.group(3), pm.group(2), types.get(pm.group(1)))
                for pm in _DSML_PARAM_RE.finditer(m.group(2))
            }
            calls.append((name, args))
    else:
        # malformed: no complete invoke wrapper -> collect params, infer tool name
        raw = {pm.group(1): (pm.group(3), pm.group(2)) for pm in _DSML_PARAM_RE.finditer(region)}
        if raw:
            name = _infer_dsml_name(set(raw), param_types) or "unknown"
            types = param_types.get(name, {})
            args = {k: _dsml_coerce(v, s, types.get(k)) for k, (v, s) in raw.items()}
            calls.append((name, args))

    tool_calls = [
        ToolCall(
            id=_unique_tool_call_id(),
            type="function",
            function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        )
        for name, args in calls
    ]
    if _DSML in content:  # scrub any stray marker fragment
        content = content.split("<" + _DSML, 1)[0]
    return content.strip(), tool_calls


# ---------------------------------------------------------------------------
# GLM-4.5 / 4.6 / 5.x tool-call format
# ---------------------------------------------------------------------------
#
#     <tool_call>NAME
#     <arg_key>K1</arg_key><arg_value>V1</arg_value>
#     <arg_key>K2</arg_key><arg_value>V2</arg_value>
#     ...</tool_call>
#
# The function name follows the opening tag directly (no ``<function=`` wrapper,
# which is how this is told apart from the Qwen3 XML format). GLM's chat template
# renders non-string argument values with ``tojson`` and string values raw, so a
# value is JSON-decoded when the request schema declares a non-string type (or
# when it parses as JSON) and otherwise kept as a raw string.

_GLM_TOOLCALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)$", re.DOTALL
)
_GLM_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>"
    r"(.*?)(?:</arg_value>|(?=<arg_key>)|(?=</tool_call>)|$)",
    re.DOTALL,
)


def _is_glm(text: str) -> bool:
    """Detect the GLM ``<tool_call>...<arg_key>`` format (never Qwen/DSML)."""
    if _QWEN_TOOL_PREFIX in text:  # '<function=' -> Qwen, not GLM
        return False
    return "<arg_key>" in text or "<tool_call>" in text


def _glm_coerce(value: str, ptype: Any) -> Any:
    """Decode one GLM ``<arg_value>``: schema type wins, else JSON, else raw."""
    v = value.strip("\n")
    if ptype is not None:
        return _coerce_param_value(v, ptype)
    s = v.strip()
    try:
        return json.loads(s)
    except Exception:
        return v


def _parse_glm(text: str, tools: Optional[list] = None) -> Tuple[str, List[ToolCall]]:
    """Parse GLM tool calls; return (leading_content, tool_calls)."""
    param_types = _build_param_types(tools)
    start = text.find("<tool_call>")
    if start == -1:
        return text.strip(), []
    content = text[:start]
    tool_calls: List[ToolCall] = []
    for m in _GLM_TOOLCALL_RE.finditer(text):
        body = m.group(1) if m.group(1) is not None else m.group(2)
        if not body:
            continue
        ak = body.find("<arg_key>")
        name = (body if ak == -1 else body[:ak]).strip()
        if not name:
            continue
        types = param_types.get(name, {})
        args: Dict[str, Any] = {}
        for pm in _GLM_ARG_RE.finditer(body):
            k = pm.group(1).strip()
            if k:
                args[k] = _glm_coerce(pm.group(2), types.get(k))
        tool_calls.append(
            ToolCall(
                id=_unique_tool_call_id(),
                type="function",
                function={
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            )
        )
    return content.strip(), tool_calls


# ---------------------------------------------------------------------------
# MiniMax-M3 tool-call format
# ---------------------------------------------------------------------------
#
# Every tag is prefixed by the ns_token ``]<]minimax[>[``:
#
#     ]<]minimax[>[<tool_call>
#     ]<]minimax[>[<invoke name="NAME">
#     ]<]minimax[>[<pname>value]<]minimax[>[</pname>
#     ...
#     ]<]minimax[>[</invoke>
#     ]<]minimax[>[</tool_call>
#
# Unlike DSML, parameters are named by the TAG itself (``<city>Paris</city>``),
# not a ``name="..."`` attribute. Strip the ns_token first, then parse
# <invoke>/<tag> pairs. Values: schema type wins, else JSON, else raw string.

_MINIMAX_NS = "]<]minimax[>["
_MINIMAX_INVOKE_RE = re.compile(
    r'<invoke\s+name="(.*?)"\s*>(.*?)</invoke>|<invoke\s+name="(.*?)"\s*>(.*)$',
    re.DOTALL,
)
_MINIMAX_PARAM_RE = re.compile(r"<([\w-]+)>(.*?)</\1>", re.DOTALL)


def _is_minimax(text: str) -> bool:
    """Detect the MiniMax-M3 ns_token tool-call format."""
    return _MINIMAX_NS in text


def _minimax_coerce(value: str, ptype: Any) -> Any:
    v = value.strip("\n")
    if ptype is not None:
        return _coerce_param_value(v, ptype)
    s = v.strip()
    try:
        return json.loads(s)
    except Exception:
        return v


def _parse_minimax(text: str, tools: Optional[list] = None) -> Tuple[str, List[ToolCall]]:
    """Parse MiniMax-M3 tool calls; return (leading_content, tool_calls)."""
    param_types = _build_param_types(tools)
    clean = text.replace(_MINIMAX_NS, "")
    tc = clean.find("<tool_call>")
    content = clean[:tc] if tc > 0 else ("" if tc == 0 else clean)
    tool_calls: List[ToolCall] = []
    for m in _MINIMAX_INVOKE_RE.finditer(clean):
        name = m.group(1) if m.group(1) is not None else m.group(3)
        body = m.group(2) if m.group(2) is not None else (m.group(4) or "")
        if not name:
            continue
        name = name.strip()
        types = param_types.get(name, {})
        args: Dict[str, Any] = {}
        for pm in _MINIMAX_PARAM_RE.finditer(body):
            k = pm.group(1).strip()
            if k:
                args[k] = _minimax_coerce(pm.group(2), types.get(k))
        tool_calls.append(
            ToolCall(
                id=_unique_tool_call_id(),
                type="function",
                function={
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            )
        )
    for mk in ("<tool_call>", "</tool_call>"):
        content = content.replace(mk, "")
    return content.strip(), tool_calls


def parse_tool_calls(
    text: str, tools: Optional[list] = None
) -> Tuple[str, List[ToolCall]]:
    """Parse tool calls from model output text.

    Args:
        text: Raw model output that may contain tool calls (DeepSeek-V4 DSML,
            Kimi token format, or Qwen3 XML format).
        tools: Optional request tool definitions; used to type-coerce parameter
            values to their declared JSON-Schema types.

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call sections removed.
    """
    # MiniMax-M3 ns_token format (checked before DSML: both use <invoke name=..>,
    # but MiniMax names params by tag and prefixes every tag with ]<]minimax[>[)
    if _is_minimax(text):
        return _parse_minimax(text, tools)

    # DeepSeek-V4 DSML format
    if _is_dsml(text):
        return _parse_dsml(text, tools)

    # GLM <tool_call>/<arg_key> format (checked before Qwen: both use
    # <tool_call>, but GLM never emits the Qwen '<function=' sub-tag)
    if _is_glm(text):
        return _parse_glm(text, tools)

    # Qwen3 XML format
    if _is_qwen_xml(text):
        return _parse_qwen_xml(text, tools)

    # Kimi-K2 special-token format
    section_match = re.search(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
        text,
        flags=re.DOTALL,
    )
    if not section_match:
        # Check for unclosed section
        unclosed = re.search(
            r"<\|tool_calls_section_begin\|>(.*?)$", text, flags=re.DOTALL
        )
        if unclosed:
            content = text[: unclosed.start()]
            tool_calls = _parse_tool_call_entries(unclosed.group(1))
            return content.strip(), tool_calls
        return text, []

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


@dataclass
class ToolCallStreamParser:
    """Stateful streaming parser for tool calls (Kimi tokens or Qwen3 XML).

    Processes text chunks and emits structured events:
    - ("content", text) — regular content before tool calls
    - ("tool_call_start", {"index": N, "id": ..., "function": {"name": ..., "arguments": ""}})
    - ("tool_call_args", {"index": N, "function": {"arguments": chunk}})
    - ("tool_call_end", None) — all tool calls complete

    The wire format is auto-detected from the first chunks. For the Qwen3 XML
    format content is streamed normally and the ``<tool_call>`` block is buffered
    and parsed when complete (robust against partial-XML streaming edge cases);
    ``tools`` enables JSON-Schema type coercion of parameter values.

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
    fmt: Optional[str] = None  # None|kimi|qwen|dsml|glm|minimax

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples."""
        if self.fmt is None:
            self.buf += text
            if _MINIMAX_NS in self.buf:
                self.fmt = "minimax"
            elif _is_dsml(self.buf):
                self.fmt = "dsml"
            elif "<arg_key>" in self.buf:
                self.fmt = "glm"
            elif _QWEN_TOOL_PREFIX in self.buf:
                self.fmt = "qwen"
            elif "<tool_call>" in self.buf:
                # '<tool_call>' seen but neither '<function=' (Qwen) nor
                # '<arg_key>' (GLM) yet. A no-arg GLM call is complete once the
                # closing tag arrives; otherwise wait for the sub-marker.
                if "</tool_call>" in self.buf:
                    self.fmt = "glm"
                else:
                    return []
            elif "<|tool_calls_section_begin|>" in self.buf:
                self.fmt = "kimi"
            elif "<" not in self.buf and len(self.buf) > 8:
                # No markup possible yet; emit accumulated content and stay undecided.
                out = [("content", self.buf)]
                self.buf = ""
                return out
            else:
                return []
            # Format decided: replay the accumulated buffer through the handler.
            text, self.buf = self.buf, ""

        if self.fmt == "minimax":
            return self._process_minimax(text)
        if self.fmt == "dsml":
            return self._process_dsml(text)
        if self.fmt == "glm":
            return self._process_glm(text)
        if self.fmt == "qwen":
            return self._process_qwen(text)
        return self._process_kimi(text)

    # -- MiniMax-M3 ns_token ------------------------------------------------
    def _process_minimax(self, text: str) -> list:
        results: list = []
        self.buf += text
        if self.state == 0:
            markers = [
                i
                for i in (self.buf.find(_MINIMAX_NS), self.buf.find("<tool_call>"))
                if i != -1
            ]
            if markers:
                m = min(markers)
                before = self.buf[:m]
                if before:
                    results.append(("content", before))
                self.buf = self.buf[m:]
                self.state = 1
            else:
                cut = self.buf.rfind("<")
                cut = max(cut, self.buf.rfind("]"))  # ns_token starts with ']'
                if cut == -1:
                    if self.buf:
                        results.append(("content", self.buf))
                        self.buf = ""
                elif cut > 0:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]
        return results

    def _flush_minimax(self) -> list:
        results: list = []
        if self.state == 0:
            if self.buf:
                results.append(("content", self.buf))
                self.buf = ""
            return results
        _content, tool_calls = _parse_minimax(self.buf, self.tools)
        self.buf = ""
        for tc in tool_calls:
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": self.current_index,
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function["name"], "arguments": ""},
                    },
                )
            )
            results.append(
                (
                    "tool_call_args",
                    {
                        "index": self.current_index,
                        "function": {"arguments": tc.function["arguments"]},
                    },
                )
            )
            self.current_index += 1
            self._emitted_calls += 1
        if self._emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results

    # -- DeepSeek-V4 DSML ---------------------------------------------------
    def _process_dsml(self, text: str) -> list:
        results: list = []
        self.buf += text
        if self.state == 0:
            m = _dsml_start(self.buf)
            if m != -1:
                before = self.buf[:m]
                if before:
                    results.append(("content", before))
                self.buf = self.buf[m:]
                self.state = 1
            else:
                # Emit content but hold back a possible partial '<...' marker tail.
                cut = self.buf.rfind("<")
                if cut == -1:
                    if self.buf:
                        results.append(("content", self.buf))
                        self.buf = ""
                elif cut > 0:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]
        return results

    def _flush_dsml(self) -> list:
        results: list = []
        if self.state == 0:
            if self.buf:
                results.append(("content", self.buf))
                self.buf = ""
            return results
        _content, tool_calls = _parse_dsml(self.buf, self.tools)
        self.buf = ""
        for tc in tool_calls:
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": self.current_index,
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function["name"], "arguments": ""},
                    },
                )
            )
            results.append(
                (
                    "tool_call_args",
                    {
                        "index": self.current_index,
                        "function": {"arguments": tc.function["arguments"]},
                    },
                )
            )
            self.current_index += 1
            self._emitted_calls += 1
        if self._emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results

    # -- Qwen3 XML ----------------------------------------------------------
    # -- GLM <tool_call>/<arg_key> -----------------------------------------
    def _process_glm(self, text: str) -> list:
        results: list = []
        self.buf += text
        if self.state == 0:
            m = self.buf.find("<tool_call>")
            if m != -1:
                before = self.buf[:m]
                if before:
                    results.append(("content", before))
                self.buf = self.buf[m:]
                self.state = 1
            else:
                # Emit content but hold back a possible partial '<...' marker tail.
                cut = self.buf.rfind("<")
                if cut == -1:
                    if self.buf:
                        results.append(("content", self.buf))
                        self.buf = ""
                elif cut > 0:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]
        return results

    def _flush_glm(self) -> list:
        results: list = []
        if self.state == 0:
            if self.buf:
                results.append(("content", self.buf))
                self.buf = ""
            return results
        _content, tool_calls = _parse_glm(self.buf, self.tools)
        self.buf = ""
        for tc in tool_calls:
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": self.current_index,
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function["name"], "arguments": ""},
                    },
                )
            )
            results.append(
                (
                    "tool_call_args",
                    {
                        "index": self.current_index,
                        "function": {"arguments": tc.function["arguments"]},
                    },
                )
            )
            self.current_index += 1
            self._emitted_calls += 1
        if self._emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results

    # -- Qwen3 XML ----------------------------------------------------------
    def _process_qwen(self, text: str) -> list:
        results: list = []
        self.buf += text
        if self.state == 0:
            markers = [
                i
                for i in (
                    self.buf.find("<tool_call>"),
                    self.buf.find(_QWEN_TOOL_PREFIX),
                )
                if i != -1
            ]
            if markers:
                m = min(markers)
                before = self.buf[:m]
                if before:
                    results.append(("content", before))
                self.buf = self.buf[m:]
                self.state = 1
            else:
                # Emit content but hold back a possible partial '<...' marker tail.
                cut = self.buf.rfind("<")
                if cut == -1:
                    if self.buf:
                        results.append(("content", self.buf))
                        self.buf = ""
                elif cut > 0:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]
        return results

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
            tc.id = _unique_tool_call_id()
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": self.current_index,
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function["name"], "arguments": ""},
                    },
                )
            )
            results.append(
                (
                    "tool_call_args",
                    {
                        "index": self.current_index,
                        "function": {"arguments": tc.function["arguments"]},
                    },
                )
            )
            self.current_index += 1
            self._emitted_calls += 1
        if self._emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results

    # -- Kimi tokens --------------------------------------------------------
    def _process_kimi(self, text: str) -> list:
        results = []

        if self.state == 0:
            self.buf += text
            if "<|tool_calls_section_begin|>" in self.buf:
                before = self.buf.split("<|tool_calls_section_begin|>")[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split("<|tool_calls_section_begin|>", 1)[1]
                results.extend(self._process_buffer())
            elif "<|tool" not in self.buf and len(self.buf) > 30:
                results.append(("content", self.buf))
                self.buf = ""

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
        if self.fmt == "minimax":
            return self._flush_minimax()
        if self.fmt == "dsml":
            return self._flush_dsml()
        if self.fmt == "glm":
            return self._flush_glm()
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
