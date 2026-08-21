# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deciding a model's tool-call format before it emits anything.

The format used to be sniffed from the output, which meant deciding from a
prefix: a discriminator might not have arrived yet, so the answer needed a
"cannot tell" state, and a wrong guess was silent. A chat template rendered
with a tools payload is the model's own instructions for calling one, and it
exists before the first token.

The rule under test is deliberately not a new table: it is the shipped
`_DETECT_ORDER` cascade, asked of the prompt instead of the output.
"""

from __future__ import annotations

import pytest

from atom.entrypoints.openai.tool_parser.registry import (
    PARSERS_BY_NAME,
    resolve_tool_call_parser,
)


class _Tokenizer:
    """Renders whatever it was told to, ignoring the messages."""

    def __init__(self, rendered: str | Exception):
        self._rendered = rendered

    def apply_chat_template(self, messages, **kwargs):
        if isinstance(self._rendered, Exception):
            raise self._rendered
        return self._rendered


QWEN_TEMPLATE = (
    "You may call tools. Emit:\n<tool_call>\n<function=NAME>\n"
    "<parameter=key>value</parameter>\n</function>\n</tool_call>"
)
NO_TOOLS_TEMPLATE = "You are a helpful assistant. Answer the user's question."


class TestExplicitOverride:
    @pytest.mark.parametrize("name", sorted(PARSERS_BY_NAME))
    def test_every_registered_format_is_selectable_by_name(self, name):
        """`--tool-call-parser` reaches every format, including new ones.

        The map is derived from the same registry the cascade walks, so this
        fails for a format that joins without a usable name rather than
        leaving it unreachable from the command line.
        """
        chosen = resolve_tool_call_parser(name, _Tokenizer(NO_TOOLS_TEMPLATE))
        assert chosen is PARSERS_BY_NAME[name]

    def test_an_override_beats_the_template(self):
        chosen = resolve_tool_call_parser("kimi", _Tokenizer(QWEN_TEMPLATE))
        assert chosen.NAME == "kimi"

    def test_an_unknown_name_is_refused_not_ignored(self):
        """A typo must not read as "no tool parsing" and disappear.

        Silently disabling tool calls is the failure this whole path exists to
        stop, so the name is checked where it is set.
        """
        with pytest.raises(ValueError, match="not a known format"):
            resolve_tool_call_parser("qwen3", _Tokenizer(QWEN_TEMPLATE))


class TestFromTheTemplate:
    @pytest.mark.parametrize("override", [None, "auto"])
    def test_a_template_that_teaches_a_format_resolves_to_it(self, override):
        chosen = resolve_tool_call_parser(override, _Tokenizer(QWEN_TEMPLATE))
        assert chosen is not None and chosen.NAME == "qwen"

    def test_a_template_with_no_tool_syntax_resolves_to_nothing(self):
        """`None` is an answer, not a failure.

        gpt-oss and DeepSeek-R1 render no tool syntax ATOM knows, and parsing
        nothing is right for them. What matters is that it is decided and
        logged here rather than discovered mid-stream.
        """
        assert resolve_tool_call_parser(None, _Tokenizer(NO_TOOLS_TEMPLATE)) is None

    def test_a_template_that_cannot_render_does_not_take_the_server_down(self):
        """A template may reject a tools payload; that is not fatal.

        It is also not a reason to fall back to reading the output — the
        answer is "unknown", which the caller reports.
        """
        broken = _Tokenizer(TypeError("this template takes no tools="))
        assert resolve_tool_call_parser(None, broken) is None

    def test_the_probe_carries_a_tool_so_the_template_renders_its_instructions(self):
        """A template that only mentions tools when given some must still work.

        Verified by rendering with the probe and asserting the tool's name
        reached the template -- without that, a conditional template would
        render its plain-chat branch and every model would resolve to None.
        """
        seen = {}

        class _Recording:
            def apply_chat_template(self, messages, **kwargs):
                seen.update(kwargs)
                return NO_TOOLS_TEMPLATE

        resolve_tool_call_parser(None, _Recording())
        assert seen.get("tools"), "the probe rendered without any tools"
        assert seen["tools"][0]["function"]["name"] == "get_weather"
