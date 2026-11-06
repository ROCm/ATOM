# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Lark grammar of one Kimi-K3 assistant message, tool calls held to their schemas.

A K3 message is ``think? response? tools? <|close|>message<|sep|>`` in XTML
(`encoding_k3.py` in the checkpoint). Reasoning and response text stay free;
inside the tools section every call must name a declared tool and carry each
of its required arguments, so ``<|close|>call`` cannot be sampled before them.
Non-string arguments are held to the parameter schema, and undeclared keys
are refused when the schema closes ``additionalProperties``.

Without it the model can copy an empty call it sees in its own history: on the
Kimi Vendor Verifier ToolCall bench, prompts whose history repeats
``write {}`` get ``write {}`` back. SGLang constrains the same section for
K3 even with ``tool_choice="auto"`` (`kimik3_structural_tag.py`).

The control tokens ``<|open|>``, ``<|close|>`` and ``<|sep|>`` are written as
special tokens, so free text can never contain them and the structure cannot
be bypassed through a text rule.
"""

import itertools
import json

# The XTML `type` attribute values (`_xtml_type` in encoding_k3.py).
XTML_TYPES = ("string", "number", "boolean", "null", "array", "object")
# Required arguments may come in any order. Up to this many, every order is
# enumerated; beyond it they are taken in declaration order.
MAX_PERMUTED_REQUIRED = 4
Q = '"'


def _lit(text: str) -> str:
    return json.dumps(text)


def _open(text: str) -> str:
    return f"<|open|> {_lit(text)} <|sep|>"


def _close(tag: str) -> str:
    return f"<|close|> {_lit(tag)} <|sep|>"


def _attr(value) -> str:
    # `_escape_attr_value` in encoding_k3.py.
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _xtml_type(schema) -> str | None:
    """The single XTML type a parameter schema implies, or None (any type)."""
    kind = schema.get("type") if isinstance(schema, dict) else None
    if kind == "integer":
        return "number"
    return kind if kind in XTML_TYPES else None


def _call_rules(index: int, function: dict) -> tuple[str, list[str]]:
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    required = [k for k in params.get("required") or () if k in props]
    closed = params.get("additionalProperties") is False
    any_type = "(" + " | ".join(_lit(t) for t in XTML_TYPES) + ")"
    rule = f"call_{index}"
    rules = []

    # <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|>
    names = {}
    for i, (key, schema) in enumerate(props.items()):
        names[key] = f"{rule}_arg_{i}"
        head = f'argument key="{_attr(key)}" type="'
        kind = _xtml_type(schema)
        if kind == "string":
            body = f"{_open(head + 'string' + Q)} TEXT?"
        elif kind is not None:
            body = f"{_open(head + kind + Q)} %json {json.dumps(schema)}"
        else:  # no single type: any XTML type, free value
            body = f"<|open|> {_lit(head)} {any_type} {_lit(Q)} <|sep|> TEXT?"
        rules.append(f"{names[key]}: {body} {_close('argument')}")
    optional = [names[k] for k in props if k not in required]
    if not closed:  # undeclared keys: any key, any type, free value
        extra = f"{rule}_arg_extra"
        rules.append(
            f"{extra}: <|open|> {_lit('argument key=' + Q)} /[^\"<>&]+/ "
            f"{_lit(Q + ' type=' + Q)} {any_type} {_lit(Q)} <|sep|> TEXT? "
            f"{_close('argument')}"
        )
        optional.append(extra)
    any_optional = f"({' | '.join(optional)})*" if optional else ""

    def ordered(keys) -> str:
        parts = [any_optional]
        for key in keys:
            parts += [names[key], any_optional]
        return " ".join(p for p in parts if p)

    if not required:
        arguments = any_optional
    elif len(required) <= MAX_PERMUTED_REQUIRED:
        orders = itertools.permutations(required)
        arguments = "(" + " | ".join(ordered(o) for o in orders) + ")"
    else:
        arguments = ordered(required)
    # The encoder's other argument form: one JSON object for the whole call.
    json_block = (
        f"{_open('json type=' + Q + 'object' + Q)} "
        f"%json {json.dumps(params or {'type': 'object'})} {_close('json')}"
    )
    body = f"({arguments} | {json_block})" if arguments else f"({json_block})?"
    # <|open|>call tool="NAME" index="N"<|sep|>ARGUMENTS<|close|>call<|sep|>
    head = f'call tool="{_attr(function["name"])}" index="'
    rules.append(
        f"{rule}: <|open|> {_lit(head)} /[0-9]+/ {_lit(Q)} <|sep|> {body} "
        f"{_close('call')}"
    )
    return rule, rules


def build_tool_call_grammar(
    functions: list[dict], *, required: bool, starts_in_reasoning: bool
) -> str:
    """Lark (llguidance) grammar for the rest of a K3 assistant message.

    ``functions``: the callable function objects (``name``/``parameters``).
    ``required``: a tools section must be written (``tool_choice`` required or
    a named function). ``starts_in_reasoning``: the prompt opened the think
    channel; otherwise K3's generation prompt opened the response channel.
    """
    calls, rules = [], []
    for i, function in enumerate(functions):
        rule, call_rules = _call_rules(i, function)
        calls.append(rule)
        rules += call_rules
    tools = "tools" if required else "tools?"
    if starts_in_reasoning:
        head = [
            f"start: reasoning? response? {tools} end?",
            f"reasoning: TEXT? {_close('think')}",
            f"response: {_open('response')} TEXT? {_close('response')}",
        ]
    else:
        head = [
            f"start: response? {tools} end?",
            f"response: TEXT? {_close('response')}",
        ]
    head += [
        f"tools: {_open('tools')} ({' | '.join(calls)})+ {_close('tools')}",
        f"end: {_close('message')}",
        r"TEXT: /(.|\n)+/",
    ]
    return "\n".join(["%llguidance {}", *head, *rules]) + "\n"
