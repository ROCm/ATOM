# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-4.5 / 4.6 / 5.x tool-call format::

    <tool_call>NAME
    <arg_key>K1</arg_key><arg_value>V1</arg_value>
    <arg_key>K2</arg_key><arg_value>V2</arg_value>
    ...</tool_call>

The function name follows the opening tag directly (no ``<function=`` wrapper,
which is how this is told apart from the Qwen3 XML format). GLM's chat template
renders non-string argument values with ``tojson`` and string values raw, so a
value is JSON-decoded when the request schema declares a non-string type (or
when it parses as JSON) and otherwise kept as a raw string.
"""

import json
import re
from typing import Any, ClassVar

from .qwen3_tool_parser import QWEN_TOOL_PREFIX
from .schema import build_param_types, coerce_json_or_raw
from .tool_parser import BufferedMarkerParser, ToolCall, unique_tool_call_id

TOOL_CALL_OPEN = "<tool_call>"
_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)$", re.DOTALL)
# A tool name is an identifier, which is what the model was given. Without
# this the unterminated branch above turns any prose after a `<tool_call>`
# into a call: an answer explaining that "the model writes <tool_call>
# followed by the name" produced a call named " followed by the name. Hope
# that helps!", and a Hermes-style `<tool_call>{"name": ...}` produced one
# named after the whole JSON object. Both reached clients as `tool_calls`.
#
# `\w` and not `[A-Za-z_]`: OpenAI's own grammar is `^[a-zA-Z0-9_-]{1,64}$`,
# so a leading digit is legal (`7z_extract`), and `\w` is Unicode-aware, so a
# CJK name is too -- which matters rather a lot on a Chinese model family.
# Rejecting one is silent, and prose is still rejected because a space cannot
# appear in the tail. `\Z` and not `$`, which also matches before a trailing
# newline and would admit a name with one.
_TOOL_NAME_RE = re.compile(r"^\w[\w.\-]*\Z")
_ARG_OPENER = "<arg_key>"
_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>"
    r"(.*?)(?:</arg_value>|(?=<arg_key>)|(?=</tool_call>)|$)",
    re.DOTALL,
)


# Both call shapes, so the *first* call is what matches rather than the first
# one that happens to take arguments. Requiring `<arg_key>` skipped a
# zero-argument call and announced the name of the one after it.
#
# Anchored at the region's own first `<tool_call>`, not searched for: `parse`
# reads the unclosed case from the first one to end-of-string, so when that
# one carries no usable name it produces nothing at all -- while a `search`
# here slid forward to a later opener and announced its name. Measured on
# `<tool_call><arg_key><tool_call>get_weather<arg_key>`, at every chunk size.
_PEEK_NAME_RE = re.compile(r"<tool_call>\s*(\w[\w.\-]*)\s*(?:<arg_key>|</tool_call>)")


class GlmParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "glm"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)

    @classmethod
    def peek_name(cls, region: str, tools: list | None = None) -> str | None:
        """`<tool_call>NAME<arg_key>` -- the name is what precedes the first
        argument key, so it is legible only once that key arrives.

        The one format whose peek and `parse` already agreed. Both followers
        it accepts are real: `<arg_key>` starts the arguments and
        `</tool_call>` closes the very block the name opened -- unlike Qwen's,
        where `</tool_call>` closes an *outer* wrapper and left the inner
        block unterminated.
        """
        start = region.find(TOOL_CALL_OPEN)
        if start == -1:
            return None
        m = _PEEK_NAME_RE.match(region, start)
        return m.group(1) if m else None

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the GLM ``<tool_call>...<arg_key>`` format (never Qwen/DSML)."""
        if QWEN_TOOL_PREFIX in text:  # '<function=' -> Qwen, not GLM
            return False
        # `<arg_key>` and not a bare `<tool_call>`: this runs on a rendered
        # chat template, where a Hermes-JSON model shows the same tag and has
        # no `<arg_key>` anywhere. Accepting the tag alone bound every such
        # model to this parser for the process lifetime and logged it as a
        # success -- /mnt/Qwen3-8B resolved to `glm` while /data/Qwen3.5-27B
        # resolved to `qwen`, two members of one family disagreeing.
        return "<arg_key>" in text

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse GLM tool calls; return (leading_content, tool_calls)."""
        param_types = build_param_types(tools)
        start = text.find("<tool_call>")
        if start == -1:
            return text, []  # no call -> verbatim; see ToolCallParser.parse
        content = text[:start]
        tool_calls: list[ToolCall] = []
        for m in _TOOLCALL_RE.finditer(text):
            closed = m.group(1) is not None
            body = m.group(1) if closed else m.group(2)
            if not body:
                continue
            ak = body.find("<arg_key>")
            if closed or ak != -1:
                name, rest = body[:ak].strip() if ak != -1 else body.strip(), ""
            else:
                # Unclosed and no complete `<arg_key>` yet. The name runs to
                # the first `<`, and whatever follows is the test below.
                #
                # Cut on the character and not on `len(name)`: the name is
                # stripped, so a single newline after `<tool_call>` shifted
                # that slice by one and left the name's own last character in
                # `rest`. A `read_file` call truncated after a leading newline
                # failed the test and was shipped to the user as raw markup.
                lt = body.find("<")
                name = (body if lt == -1 else body[:lt]).strip()
                rest = "" if lt == -1 else body[lt:]
            if not _TOOL_NAME_RE.match(name):
                continue
            # See `QwenXmlParser._is_truncated_call`: an unclosed region is a
            # cut-off call or prose quoting the tag, and prose has to fail
            # both tests -- a declared name, and nothing after it but this
            # format's own next token. `<tool_call>get_weather<br>` fails the
            # second; a call cut off inside its own `<arg_key>` passes it, and
            # used to be rejected along with the prose.
            if not closed and not (
                name in param_types
                and (not rest or _ARG_OPENER.startswith(rest[: len(_ARG_OPENER)]))
            ):
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _ARG_RE.finditer(body):
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
        return (content.strip() if tool_calls else text), tool_calls
