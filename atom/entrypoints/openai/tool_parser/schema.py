# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request-schema lookup and value coercion shared by every wire format.

The XML-ish tool-call formats (Qwen, DSML, GLM, MiniMax) carry no value types on
the wire, so parameter values arrive as strings. When the request supplies a
``tools`` schema each value is coerced to its declared JSON-Schema type;
otherwise it is left alone.
"""

import ast
import json
from typing import Any


def build_param_types(tools: list | None) -> dict[str, dict[str, Any]]:
    """Map ``function_name -> {param_name: json_schema_type}`` from request tools.

    Accepts OpenAI (``{"type": "function", "function": {...}}``) and bare
    (``{"name": ..., "parameters"/"input_schema": {...}}``) tool entries.
    """
    out: dict[str, dict[str, Any]] = {}
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


def coerce_param_value(value: str, ptype: Any) -> Any:
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
            except (ValueError, TypeError):
                return ast.literal_eval(v)  # safer for single-quoted Python literals
    except Exception:  # noqa: BLE001
        return v
    return v


def coerce_json_or_raw(value: str, ptype: Any) -> Any:
    """Decode one untyped value: schema type wins, else JSON, else raw string.

    Shared by GLM (``<arg_value>``) and MiniMax (``<tag>``), whose templates both
    render non-string values with ``tojson`` and string values raw.
    """
    v = value.strip("\n")
    if ptype is not None:
        return coerce_param_value(v, ptype)
    s = v.strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return v
