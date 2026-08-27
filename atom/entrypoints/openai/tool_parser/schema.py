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
from typing import Any, ClassVar


class ParamTypes(dict):
    """A mapping :func:`build_param_types` has already produced.

    Every ``parse_region`` begins by asking for this mapping, and the streaming
    reader asks once per chunk while it is still deciding whether the region
    reveals a call. Rebuilding it each time walks the whole request catalogue:
    measured at 90 us per call with 200 declared tools, which was the entire
    difference between a 0.12 ms and a 1.28 ms streamed call. Carrying the
    built mapping in place of the list it came from makes the rebuild a type
    check. The subclass is the tag -- a plain dict could be a caller's tools.

    ``schemas`` carries the property fragments the type names were read off, for
    formats that need more than the type name.
    """

    schemas: ClassVar[dict[str, dict[str, Any]]] = {}


# Bounded because this runs per parsed element.
_MAX_COMPOSITION_DEPTH = 4


def _schema_branches(schema: Any) -> list[dict[str, Any]]:
    """``schema`` itself followed by its ``oneOf`` / ``anyOf`` / ``allOf`` arms.

    A tool may declare its arguments inside a composition rather than at the top
    level, where reading only ``properties`` finds nothing and every argument
    loses its type. Outermost-first, so the enclosing schema wins a name clash.
    """
    out: list[dict[str, Any]] = []
    pending: list[tuple[Any, int]] = [(schema, 0)]
    while pending:
        node, depth = pending.pop(0)
        if not isinstance(node, dict):
            continue
        out.append(node)
        if depth >= _MAX_COMPOSITION_DEPTH:
            continue
        for keyword in ("oneOf", "anyOf", "allOf"):
            arms = node.get(keyword)
            if isinstance(arms, (list, tuple)):
                pending.extend((arm, depth + 1) for arm in arms)
    return out


def declared_properties(schema: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in _schema_branches(schema):
        props = node.get("properties")
        if isinstance(props, dict):
            for name, sub_schema in props.items():
                out.setdefault(name, sub_schema)
    return out


def schema_type(schema: Any) -> str | None:
    # A union -- `["string", "null"]` -- resolves to its first non-null member.
    for node in _schema_branches(schema):
        declared = node.get("type")
        if isinstance(declared, (list, tuple)):
            declared = next(
                (d for d in declared if str(d).lower() not in ("null", "none")), None
            )
        if declared is not None:
            return str(declared).lower()
    return None


def item_schema(schema: Any) -> Any:
    for node in _schema_branches(schema):
        items = node.get("items")
        if items is not None:
            return items
    return None


def build_param_types(tools: list | None) -> dict[str, dict[str, Any]]:
    """Map ``function_name -> {param_name: json_schema_type}`` from request tools.

    Accepts OpenAI (``{"type": "function", "function": {...}}``) and bare
    (``{"name": ..., "parameters"/"input_schema": {...}}``) tool entries, and
    its own output, which it returns unchanged (see :class:`ParamTypes`).

    The schema fragments are kept on the result for :func:`build_param_schemas`.
    Both answers come out of one walk, and the streaming reader resolves the
    catalogue once per request -- after that the original tools list is gone.
    """
    if isinstance(tools, ParamTypes):
        return tools
    out: dict[str, dict[str, Any]] = {}
    # ClassVar: the shared empty mapping is the answer for a catalogue built
    # from no tools, and every instance that has any assigns its own.
    schemas: ClassVar[dict[str, dict[str, Any]]] = {}
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
        props = declared_properties(schema)
        schemas[name] = props
        out[name] = {
            k: (v.get("type") if isinstance(v, dict) else None)
            for k, v in props.items()
        }
    built = ParamTypes(out)
    built.schemas = schemas
    return built


def build_param_schemas(tools: list | None) -> dict[str, dict[str, Any]]:
    """Map ``function_name -> {param_name: json_schema_fragment}``.

    :func:`build_param_types` keeps only the type name, which is all a flat
    format can use. MiniMax has to keep descending, so it needs the fragment.
    """
    if isinstance(tools, ParamTypes):
        return tools.schemas
    return build_param_types(tools).schemas


def coerce_param_value(value: str, ptype: Any, *, strip_framing: bool = True) -> Any:
    """Coerce a string parameter value to its declared JSON-Schema type.

    No schema type (string/unknown) -> returned unchanged. Conversion failures
    fall back to the original string rather than raising.

    ``strip_framing=False`` takes the value verbatim, for a format that writes it
    between its tags, where a declared ``string`` can legitimately end in a
    newline. The numeric and JSON conversions tolerate whitespace either way.
    """
    v = value.strip("\n") if strip_framing else value
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


def coerce_json_or_raw(value: str, ptype: Any, *, strip_framing: bool = True) -> Any:
    """Decode one untyped value: schema type wins, else JSON, else raw string.

    Shared by GLM (``<arg_value>``) and MiniMax (``<tag>``), whose templates both
    render non-string values with ``tojson`` and string values raw.

    ``strip_framing=False`` keeps leading and trailing newlines, for a format
    that writes the value inline: there every newline between the tags belongs
    to the value, and trimming one silently truncates a file being written.
    """
    v = value.strip("\n") if strip_framing else value
    if ptype is not None:
        # Passed on: `coerce_param_value` strips again on its own, which undid
        # `strip_framing=False` for every value whose type was declared.
        return coerce_param_value(v, ptype, strip_framing=strip_framing)
    s = v.strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return v
