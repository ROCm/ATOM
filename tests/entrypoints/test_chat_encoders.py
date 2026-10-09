# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for model-scoped custom chat encoder dispatch."""

import pathlib

import pytest
from jinja2 import TemplateError

from atom.entrypoints.atomesh import atom_standalone_service
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoder_adapters import (
    build_message_encoder_adapter,
)
from atom.entrypoints.openai.chat_encoders import (
    _PROBE_REFUSALS,
    REASONING_TOGGLES,
    _load_encoder_from_dir,
    apply_chat_template,
    chat_template_source,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.tool_parser.registry import resolve_tool_call_parser


def test_loader_selects_dsv4_adapter_and_preserves_encoder_defaults(tmp_path):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, **kwargs):\n"
        "    return repr((messages, kwargs))\n",
        encoding="utf-8",
    )

    adapter = _load_encoder_from_dir(str(tmp_path))

    assert adapter is not None
    assert adapter.name == "encoding_dsv4"
    assert adapter.supports_tools is True
    rendered = apply_chat_template(
        tokenizer=None,
        custom_encoder=adapter,
        messages=[{"role": "user", "content": "hello"}],
    )
    assert "'thinking_mode': 'thinking'" in rendered


def test_dsv4_adapter_prepends_tools_without_reordering_messages():
    captured = {}

    def raw_encoder(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "rendered"

    adapter = build_message_encoder_adapter("encoding_dsv4", raw_encoder)
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "trailing context"},
    ]
    original = [dict(message) for message in messages]
    tools = [{"type": "function", "function": {"name": "search"}}]

    result = apply_chat_template(
        tokenizer=None,
        custom_encoder=adapter,
        messages=messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        thinking_mode="chat",
    )

    assert result == "rendered"
    assert captured["messages"] == [
        {"role": "system", "tools": tools},
        *original,
    ]
    assert captured["kwargs"] == {"thinking_mode": "chat"}
    assert messages == original
    assert captured["messages"][1:] is not messages
    assert all(
        prepared is not source
        for prepared, source in zip(captured["messages"][1:], messages)
    )


def test_unknown_custom_encoder_does_not_receive_dsv4_fields(caplog):
    captured = {}

    def raw_encoder(messages, **kwargs):
        captured["messages"] = messages
        return "rendered"

    adapter = build_message_encoder_adapter("encoding_other", raw_encoder)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "search"}}]

    result = apply_chat_template(
        tokenizer=None,
        custom_encoder=adapter,
        messages=messages,
        tools=tools,
    )

    assert result == "rendered"
    assert captured["messages"] == messages
    assert captured["messages"] is not messages
    assert captured["messages"][0] is not messages[0]
    assert "tools" not in captured["messages"][0]
    assert "tools= is not supported" in caplog.text


def test_jinja_path_forwards_tools_and_generation_kwargs():
    class Tokenizer:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "jinja-rendered"

    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "search"}}]

    result = apply_chat_template(
        tokenizer=tokenizer,
        custom_encoder=None,
        messages=messages,
        tools=tools,
        enable_thinking=True,
    )

    assert result == "jinja-rendered"
    assert tokenizer.messages is messages
    assert tokenizer.kwargs == {
        "enable_thinking": True,
        "tokenize": False,
        "add_generation_prompt": True,
        "tools": tools,
    }


class TestChatTemplateSource:
    """The template's own text, for the question a rendered prompt cannot answer.

    Whether a model begins inside the reasoning channel shows only in what the
    template does with a *reply*, so it never reaches a fresh prompt. Measured
    on this box: Qwen3.5's source carries `<think>` and `</think>`, its
    rendered prompt carries only the opener, and Qwen3-8B's carries neither.
    Asking a render would answer False for every model alive.

    Two shapes made the raw attribute answer False by accident, and both are
    silent, which is why this is a function and not a `getattr`.
    """

    class Tok:
        def __init__(self, template):
            self.chat_template = template

    def test_a_plain_jinja_template_is_itself(self):
        assert chat_template_source(self.Tok("hello {{ x }}")) == "hello {{ x }}"

    def test_a_multi_template_dict_is_searched_by_value(self):
        """`"</think>" in <dict>` tests the keys, and quietly says no."""
        tok = self.Tok({"default": "plain", "tool_use": "closes </think> here"})
        src = chat_template_source(tok)
        assert "</think>" in src and "plain" in src

    def test_a_tokenizer_with_no_template_is_empty_not_an_error(self):
        assert chat_template_source(self.Tok(None)) == ""
        assert chat_template_source(object()) == ""

    def test_a_python_encoder_contributes_its_source(self, tmp_path):
        """`chat_template` is None for every model shipping one of these, so
        the literals live in the module instead."""

        def encode(messages, **kwargs):
            return "unused"

        # The path, not the function: `encode` is a closure defined in
        # `chat_encoders`, so asking `inspect.getmodule` for it returned
        # ATOM's own source -- 11 KB of this repo instead of the model's
        # 27 KB encoder, and the answer keyed on ATOM's own comments.
        model_file = tmp_path / "encoding_probe.py"
        model_file.write_text("MARKER = '<|open|>think<|sep|>'\n")
        adapter = build_message_encoder_adapter(
            "encoding_probe", encode, str(model_file)
        )
        src = chat_template_source(self.Tok(None), adapter)
        assert "<|open|>think<|sep|>" in src
        assert "_load_encoder_from_dir" not in src, "it read ATOM's own source"

    def test_an_encoder_with_no_path_contributes_nothing(self):
        adapter = build_message_encoder_adapter("x", lambda m, **k: "")
        assert chat_template_source(self.Tok(None), adapter) == ""

    def test_the_startup_callers_use_it(self):
        """Both entry points asked `getattr(tokenizer, "chat_template", None)`
        directly, which is the shape that answers False for a whole class of
        model. Neither body is reachable from a unit test."""
        for module in (api_server, atom_standalone_service):
            src = pathlib.Path(module.__file__).read_text()
            assert "template_opens_reasoning_implicitly(" in src
            assert "chat_template_source(" in src, f"{module.__name__} reads it raw"
            assert 'getattr(tokenizer, "chat_template", None)' not in src


class TestResolveReasoningToggle:
    """Which kwarg switches this model's reasoning off, asked rather than listed.

    A Jinja template silently ignores a kwarg it does not read, so a hardcoded
    name is a no-op that looks like a feature. Measured on this box:
    `thinking=False` leaves Qwen3.5's `<think>` prefill exactly where it was,
    while `enable_thinking=False` replaces it with a closed empty block. The
    chat path had the hardcoded name, correct for Kimi-K3 alone.

    SGLang answers the same question with ~200 lines of regex over the template
    source (`template_detection.py`); rendering twice and comparing needs no
    table and cannot go stale against a template it has never seen.
    """

    class Tok:
        """Reads exactly one kwarg, like a real template."""

        def __init__(self, reads: str | None, off_value=False):
            self.reads = reads
            self.off_value = off_value
            self.chat_template = "..."

        def apply_chat_template(self, messages, **kwargs):
            if self.reads is not None and kwargs.get(self.reads) == self.off_value:
                return "PROMPT<think>\n\n</think>"
            return "PROMPT<think>"

    @pytest.mark.parametrize(
        "name, off", [("enable_thinking", False), ("thinking", False)]
    )
    def test_it_finds_the_kwarg_the_template_reads(self, name, off):
        expected_on = next(
            on for k, o, on in REASONING_TOGGLES if (k, o) == (name, off)
        )
        assert resolve_reasoning_toggle(self.Tok(name, off)) == (name, off, expected_on)

    def test_it_finds_a_non_boolean_switch(self):
        tok = self.Tok("thinking_mode", "disabled")
        assert resolve_reasoning_toggle(tok) == ("thinking_mode", "disabled", "enabled")

    def test_a_template_with_no_switch_answers_none(self):
        """gpt-oss and DeepSeek-R1 on this box; saying so is the point."""
        assert resolve_reasoning_toggle(self.Tok(None)) is None

    def test_a_value_the_encoder_rejects_does_not_end_the_search(self):
        """DeepSeek-V4's encoder asserts on any `thinking_mode` outside
        {"chat", "thinking"}, and MiniMax-M3's wants "disabled" -- one kwarg,
        two disjoint vocabularies. A refusal has to mean "try the next pair",
        or whichever family is tried second never resolves."""

        class Picky:
            chat_template = "..."

            def apply_chat_template(self, messages, **kwargs):
                mode = kwargs.get("thinking_mode")
                if mode is None:
                    return "PROMPT<think>"
                assert mode in ("chat", "thinking"), f"bad mode {mode}"
                return "PROMPT" if mode == "chat" else "PROMPT<think>"

        assert resolve_reasoning_toggle(Picky()) == (
            "thinking_mode",
            "chat",
            "thinking",
        )

    def test_the_candidates_cover_both_thinking_mode_vocabularies(self):
        modes = [off for k, off, _ in REASONING_TOGGLES if k == "thinking_mode"]
        assert modes == ["disabled", "chat"], (
            "order matters: MiniMax must match before DeepSeek-V4's rejection "
            "of 'disabled' sends the probe on to 'chat'"
        )

    def test_an_unrenderable_template_answers_none(self):
        class Broken:
            chat_template = "..."

            def apply_chat_template(self, messages, **kwargs):
                raise TemplateError("nope")

        assert resolve_reasoning_toggle(Broken()) is None

    def test_every_candidate_switches_reasoning_off_not_on(self):
        """The values are the *off* values; a typo turning one on would be
        invisible in the probe, which only checks that the render changed."""
        assert {off for _, off, _ in REASONING_TOGGLES} == {False, "disabled", "chat"}
        assert {on for _, _, on in REASONING_TOGGLES} == {True, "enabled", "thinking"}
        for name, off, on in REASONING_TOGGLES:
            assert off != on, name

    @pytest.mark.parametrize("enabled", [True, False])
    def test_both_directions_use_the_resolved_name(self, enabled):
        """Asserted on the kwargs produced, not on a source literal.

        The previous version checked that the string
        `merged_kwargs["thinking"] = _th_enabled` was absent. Splitting that
        line into an `if`/`elif` and writing `= True` in the enable branch
        satisfied it while leaving the enable direction hardcoded -- a no-op
        on every template that reads another name, which is all of Qwen.
        """
        toggle = ("enable_thinking", False, True)

        class Req:
            thinking = {"type": "enabled"} if enabled else {"type": "disabled"}

        kwargs = api_server.anthropic_template_kwargs(Req(), toggle)
        assert kwargs == {"enable_thinking": enabled}
        assert "thinking" not in kwargs, "a hardcoded kwarg name came back"

    def test_the_chat_path_writes_no_hardcoded_reasoning_kwarg(self):
        """That body is inside a route handler no unit test reaches, so this
        reads the source -- but for the shape of the answer, not a literal:
        no `merged_kwargs["<anything about thinking>"] = ...` at all."""
        import ast
        import inspect
        import textwrap

        # Only the on/off names. `thinking_effort` is a separate control --
        # an effort level, validated against the dialects' own vocabulary --
        # and is not what the resolved toggle answers.
        _TOGGLE_NAMES = {name for name, _, _ in REASONING_TOGGLES}
        src = inspect.getsource(api_server.chat_completions)
        tree = ast.parse(textwrap.dedent(src))
        hardcoded = [
            n.slice.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript)
            and isinstance(n.ctx, ast.Store)
            and getattr(n.value, "id", "") == "merged_kwargs"
            and isinstance(n.slice, ast.Constant)
            and str(n.slice.value) in _TOGGLE_NAMES
        ]
        assert not hardcoded, f"hardcoded reasoning kwarg name(s): {hardcoded}"
        assert "reasoning_toggle" in src, "it never consults the resolved name"


class TestAModelWithNoChatTemplateStillBoots:
    """Both startup probes run before the engine is created.

    So a template that refuses one does not degrade a feature, it stops the
    server. `transformers` raises `ValueError` for a checkpoint that ships no
    chat template at all, `render_probe_prompt` caught only `TemplateError`
    and `TypeError`, and a base model that used to serve `/v1/completions`
    perfectly well died at startup instead.
    """

    class NoTemplate:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError(
                "Cannot use chat template functions because "
                "tokenizer.chat_template is not set"
            )

    def test_the_two_probes_agree_on_what_a_refusal_is(self):
        """They ask the same template the same kind of question. Disagreeing
        about which exceptions mean "no" is what let one of them through."""
        source = pathlib.Path("atom/entrypoints/openai/chat_encoders.py").read_text()
        assert (
            "except _PROBE_REFUSALS" in source
        ), "render_probe_prompt has its own refusal list again"

    def test_valueerror_is_a_refusal(self):
        assert ValueError in _PROBE_REFUSALS

    def test_the_probe_returns_none_rather_than_raising(self):
        assert render_probe_prompt(self.NoTemplate(), None, tools=True) is None

    @pytest.mark.parametrize(
        "probe",
        [
            lambda t: resolve_reasoning_toggle(t, None),
            lambda t: resolve_tool_call_parser(None, t, None),
        ],
        ids=["reasoning", "tool-parser"],
    )
    def test_neither_startup_probe_raises(self, probe):
        assert probe(self.NoTemplate()) is None
