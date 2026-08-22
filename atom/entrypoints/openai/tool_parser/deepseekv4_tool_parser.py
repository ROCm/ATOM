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
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    unique_tool_call_id,
)

_DSML = "｜DSML｜"
# The model often DROPS the ``｜DSML｜`` marker and emits bare
# ``<invoke name=...>``/``<parameter ...>``/``<tool_calls>`` tags, so the marker
# is matched OPTIONALLY everywhere.
_OPT = r"(?:" + re.escape(_DSML) + r")?"  # optional ｜DSML｜ prefix
# The end-of-input alternative is what keeps the part of a value that
# arrived before `max_tokens`; requiring `</parameter>` yielded `{}`.
_PARAM_RE = re.compile(
    r"<" + _OPT + r'parameter\s+name="(.*?)"(?:\s+string="(true|false)")?\s*>'
    r"(.*?)(?:</" + _OPT + r"parameter>"
    r"|(?=<" + _OPT + r"parameter\s)"
    r"|(?=</" + _OPT + r"invoke>)"
    r"|\Z)",
    re.DOTALL,
)
# Long-form `<invoke name="x">...</invoke>` OR self-closing `<invoke name="x"/>`
# (the zero-arg shape; group(2) is None for self-closing). Matches SGLang's V4
# detector, which accepts both.
# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <invoke name="NAME">" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
_NOT_NESTED = r"(?:(?!<" + _OPT + r"invoke\s).)"
# `closed | self-closing | unclosed`, in one pattern. Recovering truncation
# in an `else:` under `if invokes:` instead meant no cut-off call could be
# recovered once any complete invoke existed in the region -- the ordinary
# `max_tokens` shape -- and it was not monotone in arrived bytes, which the
# early announcement (`parse_region` over a prefix) requires.
_INVOKE_RE = re.compile(
    r"<"
    + _OPT
    + r'invoke\s+name="([^"]*)"\s*(?:/>|>('
    + _NOT_NESTED
    + r"*?)</"
    + _OPT
    + r"invoke>)"
    + r"|<"
    + _OPT
    + r'invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*)",
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


# What may follow an unclosed opener in a real call: another parameter, or the
# close of the invoke the name opened. Either spelling of the marker, which
# the model drops about as often as it writes. One tuple, read by both
# `_is_truncated_call` at both ends of a region's life -- the peek used to
# encode this separately and accepted any tag with a slash in it, `<br/>`
# included.
_CALL_CONTINUES = (
    "<" + _DSML + "parameter",
    "<parameter",
    "</" + _DSML + "invoke>",
    "</invoke>",
)


def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`, which this is the DSML
    spelling of; the sweep that added it there and to GLM missed this format,
    and the same sentence -- "you emit `<invoke name="get_weather">` and
    inside it a `<parameter>` line" -- was still being dispatched as a call.
    """
    if name not in param_types:
        return False
    rest = body.lstrip()
    return (not rest and at_end) or continues_a_call(
        rest, _CALL_CONTINUES, arrived=not at_end
    )


class DsmlParser(ToolCallParser):
    NAME: ClassVar[str] = "dsml"
    # Region-start markers, both marked and marker-less variants.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        "<" + _DSML + "tool_call",  # marked (covers tool_call / tool_calls)
        "<" + _DSML + "invoke",  # marked invoke
        "<invoke name=",  # marker-less invoke (common malform)
        "<tool_calls>",  # marker-less section open
        # ...and its singular spelling, which `CALL_OPENERS` already claims as
        # markup. Declaring it here was impossible while this tuple was also
        # the fingerprint -- `<tool_call>` is the Hermes and Qwen opener too,
        # so DSML would have claimed their templates. `DETECT_MARKERS` is that
        # separation, and this is the first thing it buys: the wrapper stops
        # being released as content before the region opens.
        "<tool_call>",
    )
    # What makes a text DSML's, which is narrower than what must not be split:
    # the marker-bearing spellings only. The marker-less ones above are
    # malform recovery, and `<invoke name=` is also how MiniMax opens a call
    # -- keying identification on them had DSML claiming MiniMax's templates
    # with nothing but `_DETECT_ORDER` in the way.
    DETECT_MARKERS: ClassVar[tuple[str, ...]] = (
        "<" + _DSML + "tool_call",
        "<" + _DSML + "invoke",
        "<" + _DSML + "parameter",
    )
    # The section wrapper closing after the last invoke -- markup, not answer.
    # Both spellings, since the model drops the marker about as often as it
    # writes it, and both arities, since it writes the singular too.
    CALL_OPENERS: ClassVar[tuple[str, ...]] = (
        "<" + _DSML + "tool_calls>",
        "<" + _DSML + "tool_call>",
        "<tool_calls>",
        "<tool_call>",
    )
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = (
        "</" + _DSML + "tool_calls>",
        "</" + _DSML + "tool_call>",
        "</tool_calls>",
        "</tool_call>",
    )

    # detect() is inherited: any start marker present means DSML.

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(
            f'<{_DSML}parameter name="{k}" string="true">{v}</{_DSML}parameter>'
            for k, v in args.items()
        )
        return (
            f'<{_DSML}tool_calls><{_DSML}invoke name="{name}">'
            f"{body}</{_DSML}invoke></{_DSML}tool_calls>"
        )

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        calls: list[tuple[str, dict[str, Any]]] = []
        # A truncated or wrapper-less call runs to the end of what arrived;
        # complete invokes start and end where their own matches do, one span
        # each.
        spans: list[tuple[int, int]] = []
        invokes = list(_INVOKE_RE.finditer(region))
        if invokes:
            for m in invokes:
                closed = m.group(1) is not None
                name = m.group(1) if closed else m.group(3)
                # `None` for a self-closing `<invoke .../>`, which is closed
                # and carries no body.
                body = (m.group(2) if closed else m.group(4)) or ""
                if not closed and not _is_truncated_call(
                    name, body, param_types, at_end=at_end
                ):
                    continue
                spans.append(
                    (
                        cls.markup_begin(region, m.start()),
                        cls.markup_end(region, m.end()),
                    )
                )
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
            # No `<invoke>` at all: the documented V4-Flash malform that drops
            # the wrapper and writes bare `<parameter>` lines. The name has to
            # be inferred from the parameter signature, which is why this is
            # its own branch -- a *truncated* call has its name in the opener
            # and is read by the loop above. Inferring for both scored
            # `get_time` for a cut-off `<invoke name="get_weather">` because
            # it happened to share more parameters, so the parse named a
            # different tool than the announcement had.
            raw = {
                pm.group(1): (pm.group(3), pm.group(2))
                for pm in _PARAM_RE.finditer(region)
            }
            if raw:
                name = _infer_name(set(raw), param_types) or "unknown"
                # Nothing to anchor to, so the parameters run from wherever
                # the region began to the end of what arrived.
                spans.append((0, len(region)))
                types = param_types.get(name, {})
                args = {k: _coerce(v, s, types.get(k)) for k, (v, s) in raw.items()}
                args = _unwrap_wrapper_args(args, set(types))
                calls.append((name, args))

        tool_calls = tuple(
            ToolCall(
                id=unique_tool_call_id(),
                type="function",
                function={
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            )
            for name, args in calls
        )
        return RegionParse(tool_calls, tuple(spans))
