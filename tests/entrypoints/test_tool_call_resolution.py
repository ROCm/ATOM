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

import ast
import pathlib
from typing import ClassVar

import pytest

from atom.entrypoints.atomesh import server as atomesh_server
from atom.entrypoints.openai.tool_parser import registry
from atom.entrypoints.openai.tool_parser.registry import (
    PARSERS_BY_NAME,
    parse_tool_calls,
    resolve_from_prompt,
    resolve_tool_call_parser,
    validate_tool_call_parser,
)
from atom.entrypoints.openai.tool_parser.stream import ToolCallStreamParser


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


class TestUnresolvedMeansUnparsed:
    """`None` is "do not parse", on both paths, and not "work it out".

    The streaming facade already honoured it -- `ToolCallStreamParser` with no
    format emits everything as content -- while the non-streaming path ran a
    cascade over the *output*. So the same request was answered two ways, and
    the way that guessed deleted text: gpt-oss and DeepSeek-R1 both resolve to
    `None` on this box, and an answer of theirs quoting another format's
    section token lost everything from the token onward.
    """

    QUOTES_A_MARKER = "The model emits <|tool_calls_section_begin|> then the calls."
    QUOTES_ANOTHER = 'To call a tool you write <invoke name="get_weather"> and so on.'

    @pytest.mark.parametrize("text", [QUOTES_A_MARKER, QUOTES_ANOTHER])
    def test_nothing_is_parsed_and_nothing_is_lost(self, text):
        assert parse_tool_calls(text, None, parser_cls=None) == (text, [])

    @pytest.mark.parametrize("text", [QUOTES_A_MARKER, QUOTES_ANOTHER])
    def test_the_streaming_path_agrees(self, text):
        stream = ToolCallStreamParser(parser_cls=None)
        events = []
        for i in range(0, len(text), 5):
            events += stream.process(text[i : i + 5])
        events += stream.flush()
        assert "".join(d for k, d in events if k == "content") == text
        assert not [k for k, _ in events if k.startswith("tool_call_")]

    def test_the_output_cascade_is_gone(self):
        """A second place to ask is a second answer; there is one place now.

        By AST, not by slicing the source: `PARSERS_BY_NAME` is built from
        `_DETECT_ORDER` on the lines just below, and a text search for the
        function "body" swept it up and failed on the wrong thing.
        """
        tree = ast.parse(pathlib.Path(registry.__file__).read_text())
        walked = [
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "_DETECT_ORDER"
        ]
        assert walked == ["resolve_from_prompt"], (
            f"_DETECT_ORDER is consulted by {walked}; the prompt is the only "
            "evidence it should be asked about"
        )


class TestABadNameIsCaughtBeforeTheWeightsLoad:
    """`AtomStandaloneService` raises on an unknown name, and it is built once
    the model is in memory -- so a typo cost a full load before saying so.
    """

    def test_a_known_name_resolves(self):
        assert validate_tool_call_parser("glm") is PARSERS_BY_NAME["glm"]

    @pytest.mark.parametrize("unset", [None, "", "auto"])
    def test_unset_defers_to_the_template(self, unset):
        assert validate_tool_call_parser(unset) is None

    def test_an_unknown_name_raises_and_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="not a known format"):
            validate_tool_call_parser("deepseekv4")

    def test_atomesh_validates_before_it_creates_the_engine(self):
        src = pathlib.Path(atomesh_server.__file__).read_text()
        assert "validate_tool_call_parser(" in src
        tree = ast.parse(src)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "launch_atom_standalone"
        )
        calls = [
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert calls.index("parse_standalone_args") < calls.index("initialize_engine")


class TestTheMeshRouterStillGetsItsFlag:
    """Only parser names implemented by the Rust router may be forwarded."""

    BASE: ClassVar[list] = ["--model", "/x", "--port", "9000"]

    def test_atom_only_parser_is_rejected(self):
        with pytest.raises(SystemExit):
            atomesh_server.parse_standalone_args(
                [*self.BASE, "--tool-call-parser", "dsml"]
            )

    def test_unset_forwards_nothing(self):
        """Not "auto" either: the router has its own default to apply."""
        args = atomesh_server.parse_standalone_args(self.BASE)
        assert args.engine_args.tool_call_parser is None
        assert "--tool-call-parser" not in args.mesh_args

    def test_the_network_args_still_get_through(self):
        args = atomesh_server.parse_standalone_args(self.BASE)
        assert args.mesh_args[:2] == ["--port", "9000"]


class TestGlmDoesNotClaimEveryTemplate:
    """`<tool_call>` is not GLM's discriminator; `<arg_key>` is.

    Detection now runs on a rendered chat template, where a Hermes-JSON model
    shows the same `<tool_call>` tag and has no `<arg_key>` anywhere.
    Accepting the tag alone bound every such model to GlmParser for the
    process lifetime and logged it as a success -- /mnt/Qwen3-8B resolved to
    `glm` while /data/Qwen3.5-27B resolved to `qwen`, one family, two answers.
    GLM then produced no call (its name check rejects the JSON body) and the
    whole region, tool call and following answer alike, arrived in one frame.
    """

    HERMES = (
        "You may call tools. Emit:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json>}\n</tool_call>'
    )
    GLM = (
        "You may call tools. Emit:\n<tool_call>NAME"
        "<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
    )

    def test_a_hermes_template_is_not_glm(self):
        assert resolve_from_prompt(self.HERMES) is not PARSERS_BY_NAME["glm"]

    def test_a_hermes_template_resolves_to_nothing(self):
        """`None` is the honest answer: ATOM has no Hermes parser, so its
        calls are delivered as text rather than swallowed by the wrong one."""
        assert resolve_from_prompt(self.HERMES) is None

    def test_a_real_glm_template_still_resolves(self):
        assert resolve_from_prompt(self.GLM) is PARSERS_BY_NAME["glm"]


class TestTheTwoVocabulariesCoexist:
    """Rust parser names work; legacy Python-only names fail explicitly."""

    BASE: ClassVar[list] = ["--model", "/x", "--port", "9000"]

    def _parse(self, name):
        return atomesh_server.parse_standalone_args(
            [*self.BASE, "--tool-call-parser", name]
        )

    @pytest.mark.parametrize("name", ["json", "python", "xml", "hermes"])
    def test_a_router_name_starts_and_reaches_the_router(self, name):
        args = self._parse(name)
        assert args.mesh_args[-2:] == ["--tool-call-parser", name]
        assert (
            args.engine_args.tool_call_parser is None
        ), "a name ATOM does not know must leave ATOM reading the template"

    @pytest.mark.parametrize("name", sorted(PARSERS_BY_NAME))
    def test_an_atom_only_name_is_rejected(self, name):
        with pytest.raises(SystemExit):
            self._parse(name)

    def test_unset_forwards_nothing_and_binds_nothing(self):
        args = atomesh_server.parse_standalone_args(self.BASE)
        assert args.engine_args.tool_call_parser is None
        assert "--tool-call-parser" not in args.mesh_args
