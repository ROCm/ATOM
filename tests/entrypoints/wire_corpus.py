# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One call per wire format, asked of the formats themselves.

Nothing here is written by hand: `REAL_CALLS` is built by asking every
registered parser to `render_call`, and every shape the suites need is derived
from it. A format registered tomorrow is covered by every property the moment
it exists, with nothing added here.

Hand-written samples were the alternative and one of them was wrong in the way
that matters: MiniMax's was written in DSML's spelling and parsed to
`get_weather({})`, exercising none of the parameter path.

Generating does not by itself fix that -- a renderer and a parser written from
the same misunderstanding round-trip perfectly. :func:`check_corpus`'s last
check does: a format's call must be *identified* by its own parser and no
other.
"""

from __future__ import annotations

#: The value every call carries. Derivations cut here, so it must appear
#: exactly once in each rendered call.
PAYLOAD = "Paris"

#: The tool every call names, and the parameter it passes.
TOOL = "get_weather"
PARAM = "city"

#: The second declared tool. Never called; it exists so that "the early name
#: disagreed with the parse" is expressible at all -- with one declared tool a
#: property comparing names compares `get_weather` to `get_weather`.
OTHER_TOOL = "get_time"

DECLARED_TOOLS: list[dict] = [
    {"type": "function", "function": {"name": TOOL, "parameters": {}}},
    {"type": "function", "function": {"name": OTHER_TOOL, "parameters": {}}},
]

#: Typed variant, for the parsers that coerce argument values.
TYPED_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {PARAM: {"type": "string"}},
            },
        },
    }
    for name in (TOOL, OTHER_TOOL)
]


def _render(parser_cls) -> str:
    return parser_cls.render_call(TOOL, {PARAM: PAYLOAD})


class _Corpus(dict):
    """The rendered calls, with a format that cannot render explained."""

    def __missing__(self, name):
        raise AssertionError(
            f"the registered format {name!r} cannot render its own syntax. "
            "Implement `render_call(name, args)` on its parser -- the inverse "
            "of `parse_region` for one call, wrapper included. The whole test "
            "corpus is generated from it, so every property covers the format "
            "as soon as it exists, and nothing needs adding to the tests."
        )


def build(parsers: dict) -> dict[str, str]:
    """One rendered call per format that can render one."""
    out = _Corpus()
    for name, cls in parsers.items():
        try:
            out[name] = _render(cls)
        except NotImplementedError:
            continue
    return out


def _registry() -> dict:
    from atom.entrypoints.openai.tool_parser.registry import PARSERS_BY_NAME

    return PARSERS_BY_NAME


#: Built at import, from the registry. THE corpus; nobody writes an entry.
REAL_CALLS: dict[str, str] = build(_registry())


# --- derived shapes ---------------------------------------------------------


def complete(name: str) -> str:
    """This format's call, whole."""
    return REAL_CALLS[name]


def cut_at_payload(call: str) -> str:
    """`call` as it looked when generation stopped inside its argument value.

    The one place the cut rule lives, because where a call is cut decides what
    the check is asking. Both other rules that have existed here asked
    something else on half the registry: a fixed twelve characters left a
    *fully closed* call for three of six formats, and the midpoint
    (``call[:len(call)//2]``, which the announcement suite derived for itself)
    leaves no recoverable call at all for three -- Kimi-K2's does not parse.
    That second rule is why a real announce-vs-parse divergence stayed green.
    """
    return call[: call.index(PAYLOAD) + len(PAYLOAD) - 2]


def truncated(name: str) -> str:
    """The same call, cut off partway through its argument value."""
    return cut_at_payload(REAL_CALLS[name])


def truncated_after_complete(name: str) -> str:
    """A cut-off call after a finished one, in one region.

    The ordinary `max_tokens` shape, and the one where a format that recovers
    truncation only in an `else:` branch drops the second call.
    """
    return REAL_CALLS[name] + truncated(name)


def naming_another_tool(name: str) -> str:
    """The same call, for the *other* declared tool."""
    return REAL_CALLS[name].replace(TOOL, OTHER_TOOL)


def truncated_naming_another_tool(name: str) -> str:
    """A cut-off call for the *other* tool.

    Needed because "the name announced from a prefix is not the name the
    finished region parses to" cannot be stated with one tool name: comparing
    `get_weather` to `get_weather` is true however badly the parse went.
    """
    return cut_at_payload(naming_another_tool(name))


def quoting_the_opener(name: str) -> str:
    """A sentence that mentions this format's opener and calls nothing.

    An unclosed alternative without a truncation gate turns every sentence
    *about* the wire format into a tool call; Kimi-K3 did exactly that when it
    was given the alternation alone.
    """
    call = REAL_CALLS[name]
    return (
        f"You write {call[: call.index(TOOL)]}undeclared_thing "
        "and then the parameters."
    )


# --- the corpus has to be real ----------------------------------------------


def check_corpus(parsers: dict, parse) -> list[str]:
    """Complaints about the corpus itself, empty when it is sound.

    ``parsers`` is the registry and ``parse`` is `parse_tool_calls`; injected
    so this module imports nothing from the package under test at definition
    time.
    """
    import json

    problems = []

    cannot = sorted(set(parsers) - set(REAL_CALLS))
    if cannot:
        problems.append(
            f"registered but cannot render their own syntax: {cannot} -- "
            "implement `render_call` and every property covers them"
        )

    for name in sorted(set(REAL_CALLS) & set(parsers)):
        call = REAL_CALLS[name]
        if call.count(PAYLOAD) != 1:
            problems.append(
                f"{name}: {PAYLOAD!r} appears {call.count(PAYLOAD)} times; the "
                "derivations cut there and need exactly one"
            )
        # Round trip: what this format writes, it must read back.
        content, calls = parse(call, TYPED_TOOLS, parsers[name])
        got = [c.function["name"] for c in calls]
        if got != [TOOL]:
            problems.append(f"{name}: renders a call its own parser reads as {got}")
            continue
        try:
            decoded = json.loads(calls[0].function["arguments"])
        except ValueError:
            problems.append(f"{name}: arguments are not JSON")
            continue
        if decoded != {PARAM: PAYLOAD}:
            problems.append(f"{name}: round-trips to {decoded!r}")
        if content:
            problems.append(f"{name}: left {content!r} outside the call")

        # The ground truth a round trip cannot give: a renderer and a parser
        # written from the same misunderstanding agree with each other, but a
        # sample in another format's spelling is claimed by that format.
        #
        # `detect`, not `parse`. Two formats may both be able to *read* a
        # string -- DSML matches its marker optionally, so it reads MiniMax's
        # calls -- but only one may *claim* it. No exception list on purpose:
        # one was written for the MiniMax/DSML pair, and that was the smell
        # that led to splitting identification off `START_MARKERS`.
        claimers = sorted(n for n in parsers if parsers[n].detect(call))
        if claimers != [name]:
            problems.append(
                f"{name}'s own call is identified as {claimers} -- either the "
                "renderer is written in another format's spelling, or two "
                "formats are confusable by construction and one needs a "
                "narrower `DETECT_MARKERS`"
            )
    return problems
