# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""System messages inside the conversation, for templates that cannot take them.

Claude Code sends reminders as ``role: "system"`` entries in ``messages``,
typically right after a user turn. DeepSeek-R1's template moves them to the
head of the prompt, gpt-oss's drops them, and DeepSeek-V4's encoder renders one
after a user turn with no role markers, merging the user turn, the reminder and
the reply. Such a template is found at startup, and /v1/messages then moves
those messages into user turns.
"""

import asyncio
import json

import pytest
from jinja2 import TemplateError

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoders import template_takes_mid_system
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    anthropic_to_openai_messages,
    fold_mid_system,
)


class _Template:
    """A stub tokenizer whose chat template renders turns as `<role>text`,
    with one way of handling a system message inside the conversation."""

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def apply_chat_template(self, messages, **_kwargs):
        b = self.behaviour
        top = [m for m in messages[:1] if m["role"] == "system"]
        rest = messages[len(top) :]
        mid = [m for m in rest if m["role"] == "system"]
        if b == "raises" and mid:
            raise NotImplementedError("Unknown role: system")
        if b == "refuses" and mid:
            raise ValueError("System message must be at the beginning.")
        if b == "moves_to_head":
            top, rest = top + mid, [m for m in rest if m["role"] != "system"]
        if b == "drops" or (b == "drops_trailing" and rest[-1]["role"] == "system"):
            rest = [m for m in rest if m not in mid] if b == "drops" else rest[:-1]
        if b == "moves_to_tail":
            rest = [m for m in rest if m["role"] != "system"] + mid
        roles = [m["role"] for m in rest]
        if b == "defers_past_reply" and "system" in roles:
            # A reminder after a user turn rendered after the reply instead.
            i = roles.index("system")
            if (
                0 < i < len(rest) - 1
                and roles[i - 1] == "user"
                and roles[i + 1] == "assistant"
            ):
                rest = rest[:i] + [rest[i + 1], rest[i]] + rest[i + 2 :]
        if b == "moves_to_head_after_assistant":
            keep = []
            for i, m in enumerate(rest):
                if m["role"] == "system" and i and rest[i - 1]["role"] == "assistant":
                    top = top + [m]
                else:
                    keep.append(m)
            rest = keep
        out = "".join(f"<system>{m['content']}" for m in top)
        if b == "unmarked_after_user":  # DeepSeek-V4's encoder
            for i, m in enumerate(rest):
                nxt = rest[i + 1]["role"] if i + 1 < len(rest) else None
                if m["role"] == "user":
                    out += f"<user>{m['content']}"
                    out += "<assistant>" if nxt in ("assistant", None) else ""
                elif m["role"] == "assistant":
                    out += f"{m['content']}<eos>"
                else:
                    out += m["content"]
            return out
        for i, m in enumerate(rest):
            unmarked = (
                b == "unmarked_before_reply"
                and m["role"] == "assistant"
                and i
                and rest[i - 1]["role"] == "system"
            )
            out += m["content"] if unmarked else f"<{m['role']}>{m['content']}"
        if b == "no_header_after_system" and rest[-1]["role"] == "system":
            return out
        return out + "<assistant>"


@pytest.mark.parametrize(
    "behaviour, takes",
    [
        ("renders_in_place", True),
        ("moves_to_head", False),  # DeepSeek-R1
        ("drops", False),  # gpt-oss
        ("unmarked_after_user", False),  # DeepSeek-V4
        ("no_header_after_system", False),
        ("drops_trailing", False),
        ("defers_past_reply", False),
        ("unmarked_before_reply", False),
        ("moves_to_head_after_assistant", False),
        ("moves_to_tail", False),
        ("refuses", False),
        ("raises", False),
    ],
)
def test_the_probe_tells_which_templates_take_it(behaviour, takes):
    assert template_takes_mid_system(_Template(behaviour), None) is takes


def _sys(text):
    return {"role": "system", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text, **kw):
    return {"role": "assistant", "content": text, **kw}


def _tool(text):
    return {"role": "tool", "tool_call_id": "t1", "content": text}


CALL = {"tool_calls": [{"id": "t1", "type": "function", "function": {}}]}


class TestFold:
    def test_claude_codes_shape(self):
        # A reminder after each user turn, before the reply.
        assert fold_mid_system(
            [
                _sys("top"),
                _user("a"),
                _sys("r1"),
                _assistant("b"),
                _user("c"),
                _sys("r2"),
            ]
        ) == [
            _sys("top"),
            _user("a\n\nr1"),
            _assistant("b"),
            _user("c\n\nr2"),
        ]

    def test_before_a_user_turn_it_joins_that_turn(self):
        assert fold_mid_system(
            [_user("a"), _assistant("b"), _sys("r"), _user("c")]
        ) == [
            _user("a"),
            _assistant("b"),
            _user("r\n\nc"),
        ]

    def test_it_never_separates_a_tool_call_from_its_result(self):
        assert fold_mid_system(
            [_user("a"), _assistant("", **CALL), _sys("r"), _tool("out"), _user("c")]
        ) == [_user("a"), _assistant("", **CALL), _tool("out"), _user("r\n\nc")]
        assert fold_mid_system(
            [_user("a"), _assistant("", **CALL), _sys("r"), _tool("out")]
        ) == [_user("a"), _assistant("", **CALL), _tool("out"), _user("r")]

    def test_the_leading_system_prompt_stays(self):
        messages = [_sys("top"), _sys("more"), _user("a")]
        assert fold_mid_system(messages) == messages

    def test_consecutive_ones_join_together(self):
        assert fold_mid_system(
            [_user("a"), _assistant("b"), _sys("x"), _sys("y"), _user("c")]
        ) == [
            _user("a"),
            _assistant("b"),
            _user("x\n\ny\n\nc"),
        ]

    def test_image_content_gets_a_text_part(self):
        image = {"type": "image_url", "image_url": {"url": "data:,"}}
        out = fold_mid_system([_assistant("b"), _sys("r"), _user([image])])
        assert out[1]["content"] == [{"type": "text", "text": "r"}, image]


def test_tool_results_come_before_the_text_beside_them():
    # Claude Code puts reminder text after the tool_result in one user message;
    # text first left a user turn between the call and its result.
    messages = anthropic_to_openai_messages(
        [
            AnthropicMessage(role="user", content="a"),
            AnthropicMessage(
                role="assistant",
                content=[{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "tool_result", "tool_use_id": "t1", "content": "out"},
                    {"type": "text", "text": "reminder"},
                ],
            ),
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[-1]["content"] == "reminder"


@pytest.fixture
def rendered(monkeypatch):
    """The messages /v1/messages hands the chat template."""
    seen = []

    def render(_tok, _enc, messages, **_kw):
        seen.append([(m.role, m.content) for m in messages])
        raise RuntimeError("stop after rendering")

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "apply_chat_template", render)
    monkeypatch.setattr(
        api_server,
        "engine",
        type("E", (), {"config": type("C", (), {"generation_config": None})()})(),
    )
    return seen


@pytest.mark.parametrize("supported", [True, False])
def test_messages_folds_only_for_a_template_that_cannot_take_it(
    rendered, monkeypatch, supported
):
    monkeypatch.setattr(api_server, "mid_system_supported", supported)
    request = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        system="top",
        messages=[
            AnthropicMessage(role="user", content="a"),
            AnthropicMessage(role="system", content="note"),
        ],
    )
    asyncio.run(api_server.anthropic_messages(request, None))
    expected = (
        [("system", "top"), ("user", "a"), ("system", "note")]
        if supported
        else [("system", "top"), ("user", "a\n\nnote")]
    )
    assert rendered[0] == expected


def _recording(seen, refuse):
    def render(_tok, _enc, messages, **_kw):
        seen.append([(m.role, m.content) for m in messages])
        if refuse([m.role for m in messages]):
            raise TemplateError("System message must be at the beginning.")
        return "<rendered prompt>"

    return render


def _system_after_the_start(roles):
    start = next((i for i, r in enumerate(roles) if r != "system"), len(roles))
    return "system" in roles[start:]


def _messages(*messages, system="top"):
    request = AnthropicMessagesRequest(
        model="m", max_tokens=16, system=system, messages=list(messages)
    )
    return asyncio.run(api_server.anthropic_messages(request, None))


def _error(reply):
    return json.loads(reply.body)["error"]["message"]


LEAD = [("system", "top"), ("system", "lead")]


def test_a_leading_run_is_kept_when_a_reminder_is_refused(rendered, monkeypatch):
    monkeypatch.setattr(api_server, "mid_system_supported", True)
    monkeypatch.setattr(
        api_server, "apply_chat_template", _recording(rendered, _system_after_the_start)
    )
    reply = _messages(
        AnthropicMessage(role="system", content="lead"),
        AnthropicMessage(role="user", content="a"),
        AnthropicMessage(role="system", content="r"),
    )
    # The 400 Claude Code answers by folding the reminder.
    assert reply.status_code == 400
    assert _error(reply).startswith('Unexpected role "system"')
    assert rendered[-1] == LEAD + [("user", "a")]


def test_a_leading_run_is_not_folded(rendered, monkeypatch):
    monkeypatch.setattr(api_server, "mid_system_supported", False)
    _messages(
        AnthropicMessage(role="system", content="lead"),
        AnthropicMessage(role="user", content="a"),
        AnthropicMessage(role="system", content="r"),
    )
    assert rendered[0] == LEAD + [("user", "a\n\nr")]


def test_a_template_that_takes_one_system_message_is_not_blamed_on_a_mid_one(
    rendered, monkeypatch
):
    # Nothing follows the user turn, so folding could not help: the original
    # error stands rather than the 400 Claude Code would resend on.
    monkeypatch.setattr(api_server, "mid_system_supported", True)
    monkeypatch.setattr(
        api_server,
        "apply_chat_template",
        _recording(rendered, lambda roles: "system" in roles[1:]),
    )
    reply = _messages(
        AnthropicMessage(role="system", content="lead"),
        AnthropicMessage(role="user", content="a"),
    )
    assert reply.status_code == 500
    assert "Unexpected role" not in _error(reply)


def test_only_system_messages_are_not_blamed_on_a_mid_one(rendered, monkeypatch):
    # Nothing is mid-conversation, so the original error stands even for an
    # encoder that renders an empty conversation.
    monkeypatch.setattr(api_server, "mid_system_supported", True)
    monkeypatch.setattr(
        api_server,
        "apply_chat_template",
        _recording(rendered, lambda roles: bool(roles)),
    )
    reply = _messages(AnthropicMessage(role="system", content="x"))
    assert reply.status_code == 500
    assert "Unexpected role" not in _error(reply)
