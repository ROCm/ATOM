# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for model-scoped custom chat encoder dispatch."""

from atom.entrypoints.openai.chat_encoder_adapters import (
    build_message_encoder_adapter,
)
from atom.entrypoints.openai.chat_encoders import (
    _load_encoder_from_dir,
    apply_chat_template,
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
