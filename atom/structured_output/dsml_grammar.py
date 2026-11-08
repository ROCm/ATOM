# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lark grammar for a reply that answers or calls tools in DSML.

The native envelope, as the DeepSeek encoders render it::

    <{p}{section}>
    <{p}invoke name="NAME">
    <{p}parameter name="KEY" string="true|false">VALUE</{p}parameter>
    </{p}invoke>
    </{p}{section}>

``string="true"`` carries a raw string and ``string="false"`` a JSON value.
A strict tool's parameters are constrained in declared order by its JSON
Schema; any other tool's body is free. Inside a call, whitespace is allowed
only where the encoder writes line breaks, so a raw string value is never
padded. Around the answer and the calls it is free, as it always was.
"""

import json
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import unquote

from .schema import canonical_defs

q = json.dumps

# A raw string the client can compare exactly: annotations do not constrain.
# JSON Schema's meta-data and content keywords (pydantic writes `deprecated`
# beside a deprecated field's type), and OpenAPI's `example`.
_RAW_STRING_KEYS = {
    "type",
    "title",
    "description",
    "$comment",
    "default",
    "examples",
    "example",
    "deprecated",
    "readOnly",
    "writeOnly",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
}
# Keys that only describe a value.
_ANNOTATIONS = _RAW_STRING_KEYS - {"type"}
# Whitespace as the tool parser's `\s` reads it: Python's `re` counts the
# separators \x1c-\x1f, llguidance's regex does not.
_PARSER_SPACE = "[\\s\\x1c-\\x1f]"
_STRICT_JSON = {"lenient": False, "coerce_one_of": False}
_DEF_CONTAINERS = ("$defs", "definitions")
# Literal values: a "$ref" key inside one is data, not a reference.
_LITERALS = {"const", "enum", "default", "examples"}
# Schema maps: their keys are names, so a property called `examples` is a
# schema like any other.
_NAME_MAPS = (
    "properties",
    "patternProperties",
    "$defs",
    "$def",
    "definitions",
    "dependentSchemas",
)
# Each JSON-valued parameter is its own lexeme and carries the definitions it
# references; what parameters take alike is one shared rule. Every rule the
# tools add counts: the grammar may grow past the tool schemas only so far.
_MIN_GRAMMAR_CHARS = 1 << 20


def _same(schema: Any) -> Any:
    return schema


def json_lexeme(schema: Any) -> str:
    """Embed ``schema`` as a strict llguidance JSON lexeme."""
    if isinstance(schema, dict):
        schema = dict(schema, **{"x-guidance": _STRICT_JSON})
    return f"%json {q(schema)}"


def dsml_grammar(
    spec: dict, answer_schema: Any = None, normalize: Callable[[Any], Any] = _same
) -> str:
    """The reply grammar: ``answer_schema``'s JSON, free text, or DSML calls.

    Without ``answer_schema`` the reply may be text followed by calls; the
    text cannot contain anything the tool parser would read as the start of
    a call (``spec["markers"]``), so no unconstrained call escapes the
    grammar inside it. ``normalize`` lowers each parameter's JSON Schema the
    same way the answer's was, from one budget for the whole reply.
    """
    p, section = spec["prefix"], spec["section"]
    calls_open, calls_close = f"<{p}{section}>", f"</{p}{section}>"
    markers = [*spec.get("markers", ()), "<" + p]
    invoke_opens = [
        _regex_literal(f"<{t}") + _PARSER_SPACE for t in _spellings(p, "invoke")
    ]
    # What the parser reads as the end of a parameter value (`PARAM_RE`) or
    # of a call (`INVOKE_RE`), with or without the marker: a raw value that
    # contained one would be read back cut short, or as another call.
    value_patterns = (
        [_regex_literal(m) for m in markers]
        + invoke_opens
        + [_regex_literal(f"<{t}") + _PARSER_SPACE for t in _spellings(p, "parameter")]
        + [
            _regex_literal(f"</{t}>")
            for tag in ("parameter", "invoke", section)
            for t in _spellings(p, tag)
            if t != f"{p}parameter"
        ]
        + [_regex_literal(f"<{t}>") for t in _spellings(p, section)]
    )
    value_stops = _excluding(value_patterns)
    # A free body holds its own parameter tags, but not another call: one
    # opened inside it would be parsed as a call outside this tool's grammar.
    # Nor the call's end in the spelling the grammar does not close it with:
    # the parser ends the call at `</invoke>` with or without the marker.
    body_stops = _excluding(
        [_regex_literal(m) for m in spec.get("markers", ())]
        + invoke_opens
        + [
            _regex_literal(f"</{t}>")
            for t in _spellings(p, "invoke")
            if t != f"{p}invoke"
        ]
    )
    rules = [
        "ws: /[ \\t\\r\\n]+/",
        f"calls: {q(calls_open)} calls_rest",
        f"calls_rest: ws? invoke (ws? invoke)* ws? {q(calls_close)} ws?",
        "invoke: " + " | ".join(f"invoke_{i}" for i in range(len(spec["tools"]))),
    ]
    param_close, invoke_close = f"</{p}parameter>", f"</{p}invoke>"
    tool_rules = _ToolRules(
        max(_MIN_GRAMMAR_CHARS, 2 * len(q(spec))),
        # One for every raw string value, one for every free body.
        {
            "raw_value": f"[suffix={q(param_close)}]: /(?s:.*)/ & {value_stops}",
            "free_body": f"[suffix={q(invoke_close)}]: /(?s:.*)/ & {body_stops}",
        },
    )
    # A literal holding a value stop, or the parameter's own closer (where a
    # lazy raw value already stops), would be read back cut short.
    literal_stops = [*value_patterns, _regex_literal(param_close)]
    unsafe = re.compile("|".join(literal_stops))
    for i, tool in enumerate(spec["tools"]):
        _invoke_rules(i, tool, p, normalize, tool_rules, unsafe)
    rules += tool_rules.rules

    if spec["required"]:
        start = "ws? calls"
    elif answer_schema is not None:
        start = "answer | ws? calls"
        rules.append(f"answer: ws? {json_lexeme(answer_schema)} ws?")
    else:
        start = "text | text_before_calls calls_rest"
        no_call = _excluding([_regex_literal(m) for m in markers])
        rules += [
            "text: TEXT",
            f"TEXT: /(?s:.+)/ & {no_call}",
            f"text_before_calls[suffix={q(calls_open)}]: /(?s:.*)/ & {no_call}",
        ]
    return "\n".join([f"start: {start}", *rules]) + "\n"


def _spellings(p: str, tag: str) -> list[str]:
    """``tag`` as the parser reads it: with the DSML marker or without."""
    return sorted({p + tag, tag})


def _excluding(patterns: list[str]) -> str:
    """A lark regex matching no text that contains any of ``patterns``."""
    return "~/(?s:.*(" + "|".join(sorted(set(patterns))) + ").*)/"


class _ToolRules:
    """The rules the tools add, within what the grammar may still take."""

    def __init__(self, chars: int, shared: dict[str, str]):
        self.chars = chars
        self.rules: list[str] = []
        # Rules every parameter or tool may share, added once one does.
        self._shared = shared
        self._choices: dict[tuple[str, ...], str] = {}

    def add(self, *rules: str) -> None:
        self.chars -= sum(map(len, rules))
        if self.chars < 0:
            raise ValueError(
                "strict tool schemas expand to too large a grammar: each "
                "parameter carries every definition it references"
            )
        self.rules += rules

    def shared(self, name: str) -> str:
        if name in self._shared:
            self.add(name + self._shared.pop(name))
        return name

    def choice(self, values: list[str]) -> str:
        """One rule for ``values``, however many parameters take them (an
        enum behind a reference)."""
        key = tuple(values)
        if key not in self._choices:
            self._choices[key] = f"choice_{len(self._choices)}"
            self.add(f"{self._choices[key]}: {' | '.join(q(v) for v in values)}")
        return self._choices[key]


def _invoke_rules(
    i: int,
    tool: dict,
    p: str,
    normalize: Callable,
    rules: _ToolRules,
    unsafe: re.Pattern,
) -> None:
    invoke_open = q(f'<{p}invoke name="{tool["name"]}">')
    parameters = tool.get("parameters")
    if parameters is None:
        rules.add(f"invoke_{i}: {invoke_open} {rules.shared('free_body')}")
        return
    parameters = canonical_defs(parameters)
    defs = {
        k: parameters[k] for k in _DEF_CONTAINERS if isinstance(parameters.get(k), dict)
    }
    params = []
    for j, (key, schema) in enumerate(parameters.get("properties", {}).items()):
        params.append(f"param_{i}_{j}")
        rules.add(
            f"{params[-1]}: "
            + _parameter_value(key, schema, defs, p, normalize, rules, unsafe)
        )
    body = " ".join(f"ws? {name}" for name in params)
    rules.add(f"invoke_{i}: {invoke_open} {body} ws? {q(f'</{p}invoke>')}")


def _parameter_value(
    key: str,
    schema: Any,
    defs: dict,
    p: str,
    normalize: Callable,
    rules: _ToolRules,
    unsafe: re.Pattern,
) -> str:
    """The rule body for parameter ``key``: its tags around its value."""

    def opener(is_string: bool) -> str:
        return q(f'<{p}parameter name="{key}" string="{str(is_string).lower()}">')

    close = q(f"</{p}parameter>")

    def literals(values: list[str]) -> str | None:
        # A literal holding a value stop would be read back cut short: it
        # goes as JSON instead.
        if any(unsafe.search(v) for v in values):
            return None
        return f"{opener(True)} {rules.choice(values)} {close}"

    # Strings are written raw, as the encoder writes them, including an enum
    # behind a reference (how pydantic spells one).
    resolved = _resolve(schema, defs)
    if _plain_string(resolved):
        return f"{opener(True)} {rules.shared('raw_value')}"
    values = _string_values(resolved)
    if values is not None and (literal := literals(values)):
        return literal
    # Everything else -- patterned strings included -- is a JSON value, so the
    # schema's own JSON grammar applies; the parser decodes it the same way.
    # Its lexeme carries the definitions it references, not every one.
    raw = _raw_alternative(resolved, defs)
    if isinstance(schema, dict):
        merged = dict(schema)
        for container, entries in _referenced_defs(schema, defs).items():
            merged[container] = {**entries, **(schema.get(container) or {})}
        schema = merged
    json_value = f"{opener(False)} {json_lexeme(normalize(schema))} {close}"
    # A string this schema takes beside other values (`null`) is written raw
    # too; the others as JSON.
    if raw == "any":
        return f"{opener(True)} {rules.shared('raw_value')} | {json_value}"
    if raw and (literal := literals(raw)):
        return f"{literal} | {json_value}"
    return json_value


def _resolve(schema: Any, defs: dict) -> Any:
    """``schema``, or the one schema it only names: the definition it refers
    to, or the single member of an ``allOf`` (how pydantic v1 annotates a
    reference)."""
    for _ in range(8):  # a chain of references, not a cycle
        if not isinstance(schema, dict):
            break
        extra = set(schema) - _ANNOTATIONS
        if extra == {"$ref"} and isinstance(schema["$ref"], str):
            path = _def_path(schema["$ref"])
            # A path into a definition names only part of it.
            if path is None or len(path) != 2 or path[1] not in defs.get(path[0], {}):
                break
            schema = defs[path[0]][path[1]]
        elif extra == {"allOf"} and isinstance(schema["allOf"], list):
            if len(schema["allOf"]) != 1:
                break
            schema = schema["allOf"][0]
        else:
            break
    return schema


def _plain_string(schema: Any) -> bool:
    return (
        isinstance(schema, dict)
        and schema.get("type") == "string"
        and set(schema) <= _RAW_STRING_KEYS
    )


def _raw_alternative(schema: Any, defs: dict) -> str | list[str] | None:
    """The raw form of a schema that takes other JSON values too (a nullable
    string or enum): ``"any"``, the strings it allows, or ``None``."""
    if not isinstance(schema, dict):
        return None
    types = schema.get("type")
    if isinstance(types, list):
        # `{"type": ["string", "null"], "enum": [...]}`, as OpenAI spells it.
        if "string" not in types or set(schema) - _RAW_STRING_KEYS - {"enum"}:
            return None
        if "enum" not in schema:
            return "any"
        values = [v for v in schema["enum"] if isinstance(v, str)]
        return list(dict.fromkeys(values)) or None
    branches = schema.get("anyOf")
    if not isinstance(branches, list) or set(schema) - {"anyOf"} - _ANNOTATIONS:
        return None
    values: list[str] = []
    for branch in branches:
        branch = _resolve(branch, defs)
        if _plain_string(branch):
            return "any"
        values += _string_values(branch) or []
    return list(dict.fromkeys(values)) or None


def _referenced_defs(schema: Any, defs: dict) -> dict:
    """The definitions ``schema`` references, directly or through others."""
    needed: dict[str, dict] = {}
    pending = [schema]
    while pending:
        for ref in _refs(pending.pop()):
            path = _def_path(ref)
            if path is None:
                continue
            container, def_name = path[:2]
            found = defs.get(container, {})
            if def_name in found and def_name not in needed.get(container, {}):
                needed.setdefault(container, {})[def_name] = found[def_name]
                pending.append(found[def_name])
    return needed


def _refs(node: Any):
    """Every ``$ref`` in ``node``'s schema positions."""
    if isinstance(node, list):
        for child in node:
            yield from _refs(child)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            elif key in _NAME_MAPS and isinstance(value, dict):
                for child in value.values():
                    yield from _refs(child)
            elif key not in _LITERALS:
                yield from _refs(value)


def _def_path(ref: str) -> list[str] | None:
    """``[container, name, ...]``: the JSON Pointer of a reference into the
    root's definitions, percent-decoded before it is split, as a URI fragment
    is (`Scalar%2FanyOf` is a path into `Scalar`)."""
    if not ref.startswith("#/"):
        return None
    path = [
        part.replace("~1", "/").replace("~0", "~")
        for part in unquote(ref[2:]).split("/")
    ]
    if len(path) < 2 or path[0] not in _DEF_CONTAINERS:
        return None
    return path


def _string_values(schema: Any) -> list[str] | None:
    """The finite raw values of a string enum/const, if that is all it says."""
    if not isinstance(schema, dict) or schema.get("type", "string") != "string":
        return None
    if set(schema) - _RAW_STRING_KEYS - {"enum", "const"}:
        return None
    values = schema.get("enum")
    if "const" in schema:  # within the enum, if both are given
        values = (
            [schema["const"]] if values is None or schema["const"] in values else []
        )
    if not values or not all(isinstance(v, str) for v in values):
        return None
    return list(dict.fromkeys(values))


def _regex_literal(text: str) -> str:
    """``text`` as a literal inside a lark ``/regex/``."""
    special = set("\\.^$|?*+()[]{}/")
    return "".join("\\" + c if c in special else c for c in text)
