# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 DSML tool-call format::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="NAME">
    <｜DSML｜parameter name="PNAME" string="true|false">VALUE</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

``string="true"`` -> value is a raw string; ``string="false"`` -> value is JSON.
DeepSeek-V4-Flash occasionally malforms this (singular ``tool_call``, a missing
``invoke`` wrapper, or params without ``string=``); the parser recovers those
best-effort: it infers a dropped tool name from the parameter signature vs the
request's ``tools`` and infers a missing value type from the schema / JSON.
"""

import json
import re
from typing import Any, ClassVar

from .schema import build_param_types, coerce_param_value
from .tool_parser import BufferedMarkerParser, ToolCall, unique_tool_call_id

_DSML = "｜DSML｜"
# The model often DROPS the ``｜DSML｜`` marker and emits bare
# ``<invoke name=...>``/``<parameter ...>``/``<tool_calls>`` tags, so the marker
# is matched OPTIONALLY everywhere.
_OPT = r"(?:" + re.escape(_DSML) + r")?"  # optional ｜DSML｜ prefix
_PARAM_RE = re.compile(
    r"<" + _OPT + r'parameter\s+name="(.*?)"(?:\s+string="(true|false)")?\s*>'
    r"(.*?)</" + _OPT + r"parameter>",
    re.DOTALL,
)
# Long-form `<invoke name="x">...</invoke>` OR self-closing `<invoke name="x"/>`
# (the zero-arg shape; group(2) is None for self-closing). Matches SGLang's V4
# detector, which accepts both.
_INVOKE_RE = re.compile(
    r"<" + _OPT + r'invoke\s+name="(.*?)"\s*(?:/>|>(.*?)</' + _OPT + r"invoke>)",
    re.DOTALL,
)


def _unwrap_wrapper_args(args: Any, allowed: set) -> Any:
    """Strip spurious ``{"arguments": {...}}`` / ``{"input": {...}}`` envelopes.

    Non-tuned models (DeepSeek-V4-Pro) frequently wrap the real args in an extra
    ``arguments``/``input`` object — sometimes nested 2-3 deep, or stringified —
    so a call meant as ``{"cmd": "ls"}`` arrives as ``{"arguments": {"cmd":
    "ls"}}``. Recursively unwrap while the sole key is a wrapper that is NOT
    itself a declared param of the tool. Mirrors vLLM's ``_unwrap_wrapper_args``
    (deepseek_v4.py)."""
    for _ in range(4):  # bounded against pathological nesting
        if not (isinstance(args, dict) and len(args) == 1):
            break
        ((k, v),) = args.items()
        if k not in ("arguments", "input"):
            break
        if allowed and k in allowed:
            break  # this tool really has a param named arguments/input
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (ValueError, TypeError):
                break
        if not isinstance(v, dict):
            break
        args = v
    return args


def _coerce(value: str, string_attr: str | None, ptype: Any) -> Any:
    """Decode one ``<parameter>`` body.

    Deliberately not :func:`~.schema.coerce_json_or_raw`: on a JSON-decode miss
    this falls back to ``value.strip()`` where that one falls back to
    ``value.strip("\\n")``, which differs for values with surrounding spaces.
    """
    if string_attr == "true":
        return value
    if string_attr == "false":
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    # attr absent -> use declared schema type if known, else infer via JSON.
    if ptype is not None:
        return coerce_param_value(value, ptype)
    v = value.strip()
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def _infer_name(arg_names: set, param_types: dict[str, dict[str, Any]]) -> str | None:
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


# Name plus whatever follows it; `_continues_a_call` judges that part, so the
# rule lives in one place. See QwenXmlParser for why a follower is required.
_PEEK_NAME_RE = re.compile(r'invoke\s+name="([^"]+)"\s*>(.*)', re.DOTALL)

# The same opener with no closing tag: a call the model was cut off inside.
_TRUNCATED_INVOKE_RE = re.compile(
    r"<" + _OPT + r'invoke\s+name="([^"]*)"\s*>(.*)$', re.DOTALL
)
# What may follow that opener in a real call: another parameter, or the
# close of the invoke the name opened. Either spelling of the marker, which
# the model drops about as often as it writes. One tuple, read by both
# `_is_truncated_call` and `peek_name` -- they used to encode this
# separately and the peek's version accepted any tag with a slash in it,
# `<br/>` included.
_CALL_CONTINUES = (
    "<" + _DSML + "parameter",
    "<parameter",
    "</" + _DSML + "invoke>",
    "</invoke>",
)


def _continues_a_call(rest: str) -> bool:
    """Is `rest` the start of this format's own next token?"""
    return any(tok.startswith(rest[: len(tok)]) for tok in _CALL_CONTINUES)


def _is_truncated_call(name: str, body: str, param_types: dict) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`, which this is the DSML
    spelling of; the sweep that added it there and to GLM missed this format,
    and the same sentence -- "you emit `<invoke name="get_weather">` and
    inside it a `<parameter>` line" -- was still being dispatched as a call.
    """
    if name not in param_types:
        return False
    rest = body.lstrip()
    return not rest or _continues_a_call(rest)


class DsmlParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "dsml"
    # Region-start markers, both marked and marker-less variants.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        "<" + _DSML + "tool_call",  # marked (covers tool_call / tool_calls)
        "<" + _DSML + "invoke",  # marked invoke
        "<invoke name=",  # marker-less invoke (common malform)
        "<tool_calls>",  # marker-less section open
    )

    @classmethod
    def peek_name(cls, region: str, tools: list | None = None) -> str | None:
        """`<｜DSML｜invoke name="NAME">`, and the bare `<invoke ...>` the
        model often emits instead -- both carry the name in the opener.

        The self-closing `<invoke name="X"/>` shape has no `>` after the
        quote and so is not matched: it is a complete zero-argument call that
        `parse` reads on its own, and there is nothing to announce early
        about a region that is already finished.
        """
        m = _PEEK_NAME_RE.search(region)
        if m is None:
            return None
        rest = m.group(2).lstrip()
        return m.group(1) if rest and _continues_a_call(rest) else None

    # detect() is inherited: any start marker present means DSML.

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse DeepSeek-V4 DSML tool calls; return (leading_content, tool_calls)."""
        param_types = build_param_types(tools)
        start = cls.find_start(text)
        if start == -1:
            return text, []  # no call -> verbatim; see ToolCallParser.parse
        content = text[:start]
        region = text[start:]

        calls: list[tuple[str, dict[str, Any]]] = []
        invokes = list(_INVOKE_RE.finditer(region))
        if invokes:
            for m in invokes:
                name = m.group(1)
                body = m.group(2) or ""  # None for self-closing <invoke .../>
                types = param_types.get(name, {})
                args: dict[str, Any] = {
                    pm.group(1): _coerce(
                        pm.group(3), pm.group(2), types.get(pm.group(1))
                    )
                    for pm in _PARAM_RE.finditer(body)
                }
                # Direct-JSON parameter body (DSML "Format 2", also accepted by
                # vLLM/SGLang): `<invoke name="x"> { "k": "v" } </invoke>` with no
                # <parameter> tags. Falls through here with empty args; recover them.
                if not args:
                    stripped = body.strip()
                    if stripped.startswith("{"):
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                args = parsed
                        except (ValueError, TypeError):
                            pass
                args = _unwrap_wrapper_args(args, set(types))
                calls.append((name, args))
        else:
            # No complete invoke wrapper. Two shapes reach here and they want
            # opposite things: a call cut off mid-way, whose name is written
            # in the opener, and the documented V4-Flash malform that drops
            # the wrapper entirely, whose name has to be inferred from the
            # parameters. Inferring in both cases scored `get_time` for a
            # truncated `<invoke name="get_weather">` because it happened to
            # share more parameters -- a different tool than the one the model
            # named, and than the one the streaming path had already
            # announced off the same opener.
            opener = _TRUNCATED_INVOKE_RE.search(region)
            body = opener.group(2) if opener else region
            raw = {
                pm.group(1): (pm.group(3), pm.group(2))
                for pm in _PARAM_RE.finditer(body)
            }
            if opener is not None:
                name = opener.group(1).strip()
                keep = _is_truncated_call(name, body, param_types)
            else:
                name = _infer_name(set(raw), param_types) or "unknown"
                keep = bool(raw)
            if keep:
                types = param_types.get(name, {})
                args = {k: _coerce(v, s, types.get(k)) for k, (v, s) in raw.items()}
                args = _unwrap_wrapper_args(args, set(types))
                calls.append((name, args))

        tool_calls = [
            ToolCall(
                id=unique_tool_call_id(),
                type="function",
                function={
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            )
            for name, args in calls
        ]
        if _DSML in content:  # scrub any stray marker fragment
            content = content.split("<" + _DSML, 1)[0]
        return (content.strip() if tool_calls else text), tool_calls
