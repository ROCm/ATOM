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
    REASONING_OFF_KWARGS,
    _load_encoder_from_dir,
    apply_chat_template,
    chat_template_source,
    resolve_reasoning_toggle,
)


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

    def test_a_python_encoder_contributes_its_source(self):
        """`chat_template` is None for every model shipping one of these, so
        the literals live in the module instead."""

        def encode(messages, **kwargs):
            return "<|open|>think<|sep|>"

        adapter = build_message_encoder_adapter("encoding_probe", encode)
        src = chat_template_source(self.Tok(None), adapter)
        assert "<|open|>think<|sep|>" in src

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
        assert resolve_reasoning_toggle(self.Tok(name, off)) == (name, off)

    def test_it_finds_a_non_boolean_switch(self):
        tok = self.Tok("thinking_mode", "disabled")
        assert resolve_reasoning_toggle(tok) == ("thinking_mode", "disabled")

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

        assert resolve_reasoning_toggle(Picky()) == ("thinking_mode", "chat")

    def test_the_candidates_cover_both_thinking_mode_vocabularies(self):
        modes = [v for k, v in REASONING_OFF_KWARGS if k == "thinking_mode"]
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
        assert {v for _, v in REASONING_OFF_KWARGS} == {False, "disabled", "chat"}

    def test_both_endpoints_use_the_resolved_name(self):
        """A hardcoded kwarg is a no-op on any template that reads another,
        and a no-op here is invisible -- the model reasons anyway. Neither
        endpoint body is reachable from a unit test, so this reads the source.
        """
        src = pathlib.Path(api_server.__file__).read_text()
        assert "resolve_reasoning_toggle(" in src, "the toggle is never resolved"
        assert (
            src.count("reasoning_toggle") >= 4
        ), "resolved but not used by both the chat and Anthropic paths"
        assert (
            'merged_kwargs["thinking"] = _th_enabled' not in src
        ), "the chat path still writes a hardcoded kwarg name for every model"
