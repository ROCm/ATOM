# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 tool-call format.

Every tag is prefixed by the ns_token ``]<]minimax[>[``::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">
    ]<]minimax[>[<pname>value]<]minimax[>[</pname>
    ...
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Unlike DSML, parameters are named by the TAG itself (``<city>Paris</city>``),
not a ``name="..."`` attribute. Strip the ns_token first, then parse
<invoke>/<tag> pairs. Values: schema type wins, else JSON, else raw string.
"""

import json
import re
from typing import Any, ClassVar

from .schema import build_param_types, coerce_json_or_raw
from .tool_parser import BufferedMarkerParser, ToolCall, unique_tool_call_id

MINIMAX_NS = "]<]minimax[>["

_INVOKE_RE = re.compile(
    r'<invoke\s+name="(.*?)"\s*>(.*?)</invoke>|<invoke\s+name="(.*?)"\s*>(.*)$',
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<([\w-]+)>(.*?)</\1>", re.DOTALL)
_FIRST_TAG_RE = re.compile(r"^<([\w-]+)>")


def _is_truncated_call(name: str, body: str, param_types: dict) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`. The sweep that added that
    test to Qwen and GLM missed this format and DSML both, and "you emit
    `<invoke name="get_weather">` and then a `<city>Paris</city>` line" was
    still dispatched as a real call with the arguments filled in.

    The follower is a tag, because this format names parameters by the tag
    itself and so has no keyword to look for. Which tag is checked too when
    the request declared a schema for this tool -- otherwise `<br>` reads as
    the start of a call body -- but a tool declared without one still has to
    be callable, so an empty schema falls back to "any tag".
    """
    types = param_types.get(name)
    if types is None:
        return False
    rest = body.lstrip()
    if not rest:
        return True
    tag = _FIRST_TAG_RE.match(rest)
    return bool(tag) and (not types or tag.group(1) in types)


# Name plus whatever follows it; `_is_truncated_call` judges that part, so
# the rule lives in one place. Not DSML's `<...parameter`, which this was
# copied from: MiniMax names a parameter by the tag itself, so the DSML
# spelling matched no call that took arguments -- the feature was dead for
# every real call, and `search` then slid forward to announce the name of
# some *later* zero-argument one.
_PEEK_NAME_RE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*)', re.DOTALL)


class MiniMaxParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "minimax"
    # `<invoke name="` too: `_INVOKE_RE` matches it anywhere, so an invoke the
    # model wrote without the ns_token was a call when parsed whole and plain
    # text when streamed -- the read-ahead never opened a region for it. DSML
    # lists the same marker-less malform for the same reason.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        MINIMAX_NS,
        "<tool_call>",
        '<invoke name="',
    )

    @classmethod
    def peek_name(cls, region: str, tools: list | None = None) -> str | None:
        """`<invoke name="NAME">`, ns_token stripped first as `parse` does.

        Judged by the same `_is_truncated_call` the unclosed branch of `parse`
        uses -- including its schema lookup, which is why this takes `tools`:
        a parameter here is named by its own tag, so `<city>` and `<br>` are
        the same shape and only the request can tell them apart.
        """
        m = _PEEK_NAME_RE.search(region.replace(MINIMAX_NS, ""))
        if m is None:
            return None
        name, rest = m.group(1).strip(), m.group(2)
        if not rest.lstrip():
            return None  # nothing has followed yet; not "cut off", just early
        return (
            name if _is_truncated_call(name, rest, build_param_types(tools)) else None
        )

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the MiniMax-M3 ns_token tool-call format."""
        return MINIMAX_NS in text

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse MiniMax-M3 tool calls; return (leading_content, tool_calls)."""
        param_types = build_param_types(tools)
        clean = text.replace(MINIMAX_NS, "")
        # Content is what precedes the *call*, and the call may open with
        # either token. Cutting only at `<tool_call>` -- which this format's
        # primary ns_token shape does not contain -- left `content` holding
        # the entire `<invoke>` markup alongside the parsed call, so the user
        # was shown the raw XML while the streaming path showed nothing.
        starts = [
            i for i in (clean.find("<tool_call>"), clean.find("<invoke")) if i != -1
        ]
        cut = min(starts) if starts else -1
        content = clean[:cut] if cut != -1 else clean
        tool_calls: list[ToolCall] = []
        for m in _INVOKE_RE.finditer(clean):
            closed = m.group(1) is not None
            name = m.group(1) if closed else m.group(3)
            body = m.group(2) if closed else (m.group(4) or "")
            if not name:
                continue
            name = name.strip()
            if not closed and not _is_truncated_call(name, body, param_types):
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _PARAM_RE.finditer(body):
                k = pm.group(1).strip()
                if k:
                    args[k] = coerce_json_or_raw(pm.group(2), types.get(k))
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
        for mk in ("<tool_call>", "</tool_call>"):
            content = content.replace(mk, "")
        # No call -> verbatim; see ToolCallParser.parse. `text` and not
        # `clean`: the ns_token is a start marker, so an answer that merely
        # mentions it opens a region that parses to nothing, and the streaming
        # path then releases that region unchanged -- measured, ns_token and
        # all. Returning `clean` here would delete it on one path only.
        return (content.strip() if tool_calls else text), tool_calls
