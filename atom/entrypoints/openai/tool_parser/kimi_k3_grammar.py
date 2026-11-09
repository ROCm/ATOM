# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Lark grammar of one Kimi-K3 assistant message, tool calls held to their schemas.

A K3 message is ``think? response? tools? <|close|>message<|sep|>`` in XTML
(`encoding_k3.py` in the checkpoint). Reasoning and response text stay free;
inside the tools section every call must name a declared tool and carry each
of its required arguments, so ``<|close|>call`` cannot be sampled before them.
Typed values are held to their schema, string ``enum`` / ``const`` values to
their literals, and declared keys to their own rules; other keys are refused
when the schema closes ``additionalProperties``.

The grammar may admit a value the schema forbids -- a value of a union type,
or a string past ``enum`` / ``const`` (``pattern``, ``maxLength``) is free --
but it never refuses an argument the schema allows, with one exception: past
``MAX_PERMUTED_REQUIRED`` required arguments they must come in declaration
order, because enumerating every order grows factorially.

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
import re

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


def _pointer(ref, root: dict):
    """The target of a local ``$ref`` (``#`` or ``#/a/b``) in ``root``, or None."""
    if ref == "#":
        return root
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _resolve(schema, root: dict):
    """Follow local ``$ref``s against the parameters schema ``root``."""
    seen = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in seen or not isinstance(_pointer(ref, root), dict):
            return {}
        seen.add(ref)
        schema = _pointer(ref, root)
    return schema


def _localize(schema: dict, root: dict) -> dict:
    """Make ``schema``'s refs into the parameters' own structure resolve where
    it is compiled on its own, where ``#`` would name the value instead.

    Each such target is copied under ``$defs`` (its refs rewritten the same
    way) and the ref pointed at the copy; refs into ``$defs`` /
    ``definitions`` already resolve, as those are carried. Unresolvable refs
    are left for the compiler to report. As SGLang's ``_localize_root_refs``.
    """
    names: dict = {}
    pending = []
    taken = set(schema.get("$defs") or ())

    def rewrite(node):
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {key: rewrite(value) for key, value in node.items()}
        ref = node.get("$ref")
        if (
            isinstance(ref, str)
            and not ref.startswith(("#/$defs/", "#/definitions/"))
            and _pointer(ref, root) is not None
        ):
            if ref not in names:
                i = len(names)
                while f"root_ref_{i}" in taken:
                    i += 1
                names[ref] = f"root_ref_{i}"
                taken.add(names[ref])
                pending.append(ref)
            out["$ref"] = f"#/$defs/{names[ref]}"
        return out

    out = rewrite(schema)
    hoisted = {}
    while pending:
        ref = pending.pop()
        hoisted[names[ref]] = rewrite(_pointer(ref, root))
    if hoisted:
        out["$defs"] = {**(out.get("$defs") or {}), **hoisted}
    return out


def _value(schema, root: dict, carried: dict) -> str:
    """The rule for ``type="T"<|sep|>VALUE`` of one argument. ``root`` is the
    parameters schema and ``carried`` its own ``$defs`` / ``definitions``, so
    a ``$ref`` inside a typed value still resolves where it is compiled on
    its own."""
    target = _resolve(schema, root)
    kind = _xtml_type(target)
    if kind == "string":
        allowed = target.get("enum", [target["const"]] if "const" in target else None)
        if allowed and all(isinstance(v, str) for v in allowed):
            literals = " | ".join(_lit(v) for v in allowed if v)
            optional = "?" if "" in allowed else ""
            return f"{_lit('string' + Q)} <|sep|> ({literals}){optional}"
        return f"{_lit('string' + Q)} <|sep|> TEXT?"
    if kind is not None:
        json_schema = _localize({**target, **carried}, root)
        return f"{_lit(kind + Q)} <|sep|> %json {json.dumps(json_schema)}"
    types = " | ".join(_lit(t) for t in XTML_TYPES)
    return f"({types}) {_lit(Q)} <|sep|> TEXT?"


def _key_except(words: list[str]) -> str:
    """A regex for an attribute value (no ``"<>&``) that is none of ``words``.

    Walks the words' trie: at each node a character no word continues with
    leaves the set, a character some word continues with descends, and the
    string may stop at a node no word ends at.
    """
    trie: dict = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node[None] = True

    def esc(text: str) -> str:
        return re.escape(text).replace("/", "\\/")

    def walk(node: dict, root: bool) -> str:
        nexts = [ch for ch in node if ch is not None]
        branches = [f'[^"<>&{"".join(esc(c) for c in nexts)}][^"<>&]*']
        branches += [esc(ch) + walk(node[ch], False) for ch in nexts]
        body = "(" + "|".join(branches) + ")"
        return body if root or None in node else body + "?"

    return walk(trie, True)


def _call_rules(index: int, function: dict) -> tuple[str, list[str]]:
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    carried = {k: params[k] for k in ("$defs", "definitions") if k in params}
    extra = params.get("additionalProperties", True)
    required = list(
        dict.fromkeys(k for k in params.get("required") or () if isinstance(k, str))
    )
    rule = f"call_{index}"
    rules = []

    # <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|>
    names = {}
    for i, key in enumerate([*props, *(k for k in required if k not in props)]):
        names[key] = f"{rule}_arg_{i}"
        schema = props.get(key, extra if isinstance(extra, dict) else {})
        head = f'argument key="{_attr(key)}" type="'
        rules.append(
            f"{names[key]}: <|open|> {_lit(head)} {_value(schema, params, carried)} "
            f"{_close('argument')}"
        )
    optional = [names[k] for k in props if k not in required]
    if extra is not False:  # undeclared keys, never a declared one
        undeclared = _value(extra if isinstance(extra, dict) else {}, params, carried)
        rules.append(
            f"{rule}_arg_extra: <|open|> {_lit('argument key=' + Q)} "
            f"/{_key_except([_attr(k) for k in names])}/ {_lit(Q + ' type=' + Q)} "
            f"{undeclared} {_close('argument')}"
        )
        optional.append(f"{rule}_arg_extra")
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
    # <|open|>call tool="NAME" index="N"<|sep|>ARGUMENTS<|close|>call<|sep|>
    head = f'call tool="{_attr(function["name"])}" index="'
    rules.append(
        f"{rule}: <|open|> {_lit(head)} /[0-9]+/ {_lit(Q)} <|sep|> {arguments} "
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
