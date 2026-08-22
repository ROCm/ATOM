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
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    unique_tool_call_id,
)

TOOL_CALL_OPEN = "<tool_call>"
# A call's body may not contain another `<tool_call>`. Without that the
# non-greedy match ran from the *first* opener in the region to the first
# close, so an answer that quotes the tag and then makes a real call produced
# a "name" of the whole sentence -- rejected by `_TOOL_NAME_RE`, after which
# `finditer` resumed past the real call and found nothing at all. The tag is
# what this format opens a call with, so it cannot appear inside one.
_NOT_NESTED = r"(?:(?!<tool_call>).)"
_TOOLCALL_RE = re.compile(
    r"<tool_call>("
    + _NOT_NESTED
    + r"*?)</tool_call>|<tool_call>("
    + _NOT_NESTED
    + r"*)",
    re.DOTALL,
)
# No `$` on the unclosed alternative. An unclosed call ends where the next one
# opens, not only at end of stream: with the anchor, a call the model forgot
# to close followed by a second call matched neither alternative at the first
# opener, so `finditer` skipped the first call entirely and returned the
# second -- one call where the model made two, and a different one from the
# one the early name had already announced.
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


class GlmParser(ToolCallParser):
    NAME: ClassVar[str] = "glm"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    # No `CALL_CLOSERS`: this format's `</tool_call>` closes the call itself
    # rather than a wrapper around it, so `_TOOLCALL_RE` already spans it.

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
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _TOOLCALL_RE.finditer(region):
            closed = m.group(1) is not None
            body = m.group(1) if closed else m.group(2)
            if not body:
                continue
            ak = body.find("<arg_key>")
            # The argument opener having arrived *is* the follower evidence,
            # and it is consumed into the name/body split rather than left in
            # `rest` -- so an empty `rest` here means "the follower came and
            # went", not "nothing came". Reading the two the same way made a
            # call whose arguments were still arriving unnameable, which is
            # the one shape announcing early exists for.
            arg_key_arrived = ak != -1
            if closed or arg_key_arrived:
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
                and (
                    arg_key_arrived
                    or (not rest and at_end)
                    or continues_a_call(rest, (_ARG_OPENER,), arrived=not at_end)
                )
            ):
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _ARG_RE.finditer(body):
                k = pm.group(1).strip()
                if k:
                    args[k] = coerce_json_or_raw(pm.group(2), types.get(k))
            spans.append((m.start(), cls.markup_end(region, m.end())))
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
        return RegionParse(tuple(tool_calls), tuple(spans))
