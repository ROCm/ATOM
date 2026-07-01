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


def parse_tool_calls(
    text: str, tools: Optional[list] = None
) -> Tuple[str, List[ToolCall]]:
    """Parse tool calls from model output text.

    Args:
        text: Raw model output that may contain tool calls (Kimi token format
            or Qwen3 XML format).
        tools: Optional request tool definitions; used to type-coerce Qwen XML
            parameter values to their declared JSON-Schema types.

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call sections removed.
    """
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
    fmt: Optional[str] = None  # None (undecided) | "kimi" | "qwen"

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples."""
        if self.fmt is None:
            self.buf += text
            if _QWEN_TOOL_PREFIX in self.buf or "<tool_call>" in self.buf:
                self.fmt = "qwen"
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

        if self.fmt == "qwen":
            return self._process_qwen(text)
        return self._process_kimi(text)

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
