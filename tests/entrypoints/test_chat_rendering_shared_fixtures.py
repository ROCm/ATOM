# SPDX-License-Identifier: MIT
"""One fixture source for the template kwargs Python and mesh Rust give a chat.

atomesh tokenizes a chat to place it by the KV cache it will reuse, so it
renders the chat with the kwargs this server would: it reads the server's
`chat_rendering` from /v1/models and applies the rules and kwargs hooks it
names, which are ATOM's code, from its own copy. The fixture holds what servers
report and the kwargs their chat endpoint renders requests with; mesh's tests
read the same file.
"""

import json
import os
from pathlib import Path

import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoder_adapters import _PREPARERS, _identity_kwargs
from atom.entrypoints.openai.chat_encoders import (
    _load_encoder_from_dir,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.model_capabilities import chat_rendering
from atom.entrypoints.openai.protocol import ChatCompletionRequest
from atom.entrypoints.openai.reasoning_controls import CHAT_REASONING_RULES
from atom.entrypoints.openai.serving_chat import resolve_chat_template_kwargs

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/chat_rendering.json").read_text()
)

# DeepSeek-V4.1's `encode_messages` signature, which is what the kwargs read of
# its encoder; `test_v41_reports_the_pinned_encoders_rendering` checks it.
V41_ENCODER = """\
def encode_messages(
    messages,
    thinking_mode,
    context=None,
    drop_thinking=True,
    add_default_bos_token=True,
    reasoning_effort=None,
    return_multi_modal_data=False,
):
    return f"{thinking_mode} {reasoning_effort}"
"""


class Template:
    """A chat template switched by one kwarg's off value, or by none."""

    def __init__(self, reads=None, off=None):
        self.reads, self.off = reads, off

    def apply_chat_template(self, messages, **kwargs):
        switched = self.reads is not None and kwargs.get(self.reads) == self.off
        return "PROMPT" if switched else "PROMPT<think>"


V41 = "deepseek_v41"

# Each rendering's server: its --default-chat-template-kwargs, and its model's
# template or DeepSeek-V4.1's encoder.
SERVERS = {
    "template": ({}, Template("enable_thinking", False)),
    "template_thinking_off": (
        {"enable_thinking": False},
        Template("enable_thinking", False),
    ),
    "template_effort_low": (
        {"thinking_effort": "low"},
        Template("enable_thinking", False),
    ),
    "template_thinking_mode": ({}, Template("thinking_mode", "disabled")),
    "template_without_switch": ({}, Template()),
    "deepseek_v41": ({}, V41),
    "deepseek_v41_chat_20": ({"thinking_mode": "chat", "reasoning_effort": 20}, V41),
    "deepseek_v41_effort_none": ({"reasoning_effort": "none"}, V41),
}


@pytest.fixture(scope="module")
def v41_encoder(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deepseek_v41")
    (directory / "encoding").mkdir()
    (directory / "encoding/encoding.py").write_text(V41_ENCODER)
    (directory / "config.json").write_text(json.dumps({"model_type": V41}))
    return _load_encoder_from_dir(str(directory))


@pytest.fixture
def servers(v41_encoder, monkeypatch):
    """Each rendering's server as it starts: defaults, the toggle its startup
    probe finds, the encoder it loads, and the rules its chat endpoint uses."""
    started = {}
    for name, (defaults, model) in SERVERS.items():
        encoder = v41_encoder if model == V41 else None
        toggle = resolve_reasoning_toggle(None if encoder else model, encoder)
        monkeypatch.setattr(api_server, "custom_message_encoder", encoder)
        started[name] = (defaults, toggle, encoder, api_server._model_reasoning_rules())
    return started


def test_the_fixture_names_atoms_rules_and_kwargs_hooks():
    rules = {CHAT_REASONING_RULES.name} | {
        p.reasoning_rules.name for p in _PREPARERS.values() if p.reasoning_rules
    }
    assert FIXTURE["reasoning_rules"] == sorted(rules)
    assert FIXTURE["kwargs_hooks"] == sorted(
        name for name, p in _PREPARERS.items() if p.kwargs is not _identity_kwargs
    )


def test_servers_report_the_fixture_renderings(servers):
    assert sorted(FIXTURE["renderings"]) == sorted(servers)
    for name, server in servers.items():
        assert chat_rendering(*server) == FIXTURE["renderings"][name], name


def test_chats_render_with_the_fixture_kwargs(servers):
    for case in FIXTURE["cases"]:
        defaults, toggle, encoder, rules = servers[case["rendering"]]
        try:
            request = ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                **case["request"],
            )
            kwargs = resolve_chat_template_kwargs(
                request, defaults=defaults, toggle=toggle, rules=rules
            )
            if encoder is not None:
                kwargs = encoder.render_kwargs(kwargs)
        except ValueError:
            kwargs = None
        assert kwargs == case["kwargs"], case


def test_v41_reports_the_pinned_encoders_rendering():
    directory = Path(os.environ.get("ATOM_DSV41_MODEL", "/mnt/DeepSeek-V4.1-Flash"))
    if not (directory / "encoding/encoding.py").is_file():
        pytest.skip("set ATOM_DSV41_MODEL to check the official encoder's rendering")
    encoder = _load_encoder_from_dir(str(directory))
    rendering = chat_rendering(
        {}, resolve_reasoning_toggle(None, encoder), encoder, encoder.reasoning_rules
    )
    assert rendering == FIXTURE["renderings"]["deepseek_v41"]
