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
Schema; any other tool's body is free. Whitespace is allowed only where the
encoder writes line breaks, so a raw string value is never padded.
"""

import json
from collections.abc import Callable
from typing import Any

q = json.dumps

# A raw string the client can compare exactly: annotations do not constrain.
_RAW_STRING_KEYS = {"type", "title", "description", "$comment"}
_STRICT_JSON = {"lenient": False, "coerce_one_of": False}


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
    text cannot contain the DSML prefix, so no unconstrained call escapes the
    grammar inside it. ``normalize`` lowers each parameter's JSON Schema the
    same way the answer's was.
    """
    p, section = spec["prefix"], spec["section"]
    calls_open, calls_close = f"<{p}{section}>", f"</{p}{section}>"
    rules = [
        "ws: /[ \\t\\r\\n]+/",
        f"calls: {q(calls_open)} calls_rest",
        f"calls_rest: ws? invoke (ws? invoke)* ws? {q(calls_close)}",
        "invoke: " + " | ".join(f"invoke_{i}" for i in range(len(spec["tools"]))),
    ]
    for i, tool in enumerate(spec["tools"]):
        rules += _invoke_rules(i, tool, p, normalize)

    if spec["required"]:
        start = "calls"
    elif answer_schema is not None:
        start = "answer | calls"
        rules.append(f"answer: ws? {json_lexeme(answer_schema)}")
    else:
        start = "text | text_before_calls calls_rest"
        no_call = f"~/(?s:.*{_regex_literal('<' + p)}.*)/"
        rules += [
            "text: TEXT",
            f"TEXT: /(?s:.+)/ & {no_call}",
            f"text_before_calls[suffix={q(calls_open)}]: /(?s:.*)/ & {no_call}",
        ]
    return "\n".join([f"start: {start}", *rules]) + "\n"


def _invoke_rules(i: int, tool: dict, p: str, normalize: Callable) -> list[str]:
    invoke_open = q(f'<{p}invoke name="{tool["name"]}">')
    invoke_close = f"</{p}invoke>"
    parameters = tool.get("parameters")
    if parameters is None:
        return [
            f"invoke_{i}: {invoke_open} body_{i}",
            f"body_{i}[suffix={q(invoke_close)}]: /(?s:.*)/",
        ]
    # DeepSeek's strict mode spells the container `$def`.
    defs = {
        k: parameters[k] for k in ("$defs", "$def", "definitions") if k in parameters
    }
    params, rules = [], []
    for j, (key, schema) in enumerate(parameters.get("properties", {}).items()):
        name = f"param_{i}_{j}"
        params.append(name)
        rules += _parameter_rules(name, key, schema, defs, p, normalize)
    body = " ".join(f"ws? {name}" for name in params)
    return [f"invoke_{i}: {invoke_open} {body} ws? {q(invoke_close)}", *rules]


def _parameter_rules(
    name: str, key: str, schema: Any, defs: dict, p: str, normalize: Callable
) -> list[str]:
    def opener(is_string: bool) -> str:
        return q(f'<{p}parameter name="{key}" string="{str(is_string).lower()}">')

    close = f"</{p}parameter>"
    if (
        isinstance(schema, dict)
        and schema.get("type") == "string"
        and set(schema) <= _RAW_STRING_KEYS
    ):
        return [
            f"{name}: {opener(True)} {name}_value",
            f"{name}_value[suffix={q(close)}]: /(?s:.*)/",
        ]
    values = _string_values(schema)
    if values is not None:
        choices = " | ".join(q(v) for v in values)
        return [f"{name}: {opener(True)} ({choices}) {q(close)}"]
    # Everything else -- patterned strings included -- is a JSON value, so the
    # schema's own JSON grammar applies; the parser decodes it the same way.
    if isinstance(schema, dict) and defs:
        schema = {**defs, **schema}
    return [f"{name}: {opener(False)} {json_lexeme(normalize(schema))} {q(close)}"]


def _string_values(schema: Any) -> list[str] | None:
    """The finite raw values of a string enum/const, if that is all it says."""
    if not isinstance(schema, dict) or schema.get("type", "string") != "string":
        return None
    if set(schema) - _RAW_STRING_KEYS - {"enum", "const"}:
        return None
    values = [schema["const"]] if "const" in schema else schema.get("enum")
    if not values or not all(isinstance(v, str) for v in values):
        return None
    return list(dict.fromkeys(values))


def _regex_literal(text: str) -> str:
    """``text`` as a literal inside a lark ``/regex/``."""
    special = set("\\.^$|?*+()[]{}/")
    return "".join("\\" + c if c in special else c for c in text)
