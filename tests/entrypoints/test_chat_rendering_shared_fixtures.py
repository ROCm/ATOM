# SPDX-License-Identifier: MIT
"""One fixture source for the template kwargs Python and mesh Rust give a chat.

atomesh tokenizes a chat to place it by the KV cache it will reuse, so it
renders the chat with the kwargs this server would: it reads the server's
`chat_rendering` from /v1/models and applies the rules and kwargs hooks it
names, which are ATOM's code, from its own copy. The fixture holds what servers
report and the kwargs their chat endpoint renders requests with; mesh's tests
read the same file.

To regenerate the renderings and each case's kwargs from ATOM's code, after a
change to them or to add a case, run
`python -m tests.entrypoints.test_chat_rendering_shared_fixtures`.
"""

import json
import os
import tempfile
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

FIXTURE_PATH = Path(__file__).parent / "fixtures/chat_rendering.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())

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


def load_v41_encoder(directory):
    """DeepSeek-V4.1's encoder, as ATOM loads it from a checkpoint in ``directory``."""
    (directory / "encoding").mkdir()
    (directory / "encoding/encoding.py").write_text(V41_ENCODER)
    (directory / "config.json").write_text(json.dumps({"model_type": V41}))
    return _load_encoder_from_dir(str(directory))


def start_servers(v41_encoder, set_encoder):
    """Each rendering's server as it starts: its defaults, the toggle its startup
    probe finds, the encoder it loads (``set_encoder`` installs it), and the
    rules its chat endpoint uses."""
    started = {}
    for name, (defaults, model) in SERVERS.items():
        encoder = v41_encoder if model == V41 else None
        toggle = resolve_reasoning_toggle(None if encoder else model, encoder)
        set_encoder(encoder)
        started[name] = (defaults, toggle, encoder, api_server._model_reasoning_rules())
    return started


def chat_kwargs(server, fields):
    """The template kwargs ``server``'s chat endpoint renders a chat with
    ``fields`` with."""
    defaults, toggle, encoder, rules = server
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **fields
    )
    kwargs = resolve_chat_template_kwargs(
        request, defaults=defaults, toggle=toggle, rules=rules
    )
    return kwargs if encoder is None else encoder.render_kwargs(kwargs)


def atoms_rules():
    return sorted(
        {CHAT_REASONING_RULES.name}
        | {p.reasoning_rules.name for p in _PREPARERS.values() if p.reasoning_rules}
    )


def atoms_kwargs_hooks():
    return sorted(
        name for name, p in _PREPARERS.items() if p.kwargs is not _identity_kwargs
    )


def dump(fixture):
    """``fixture`` as the file keeps it: a rendering or case per line."""

    def line(value):
        return json.dumps(value, ensure_ascii=False)

    renderings = ",\n".join(
        f"    {line(name)}: {line(rendering)}"
        for name, rendering in fixture["renderings"].items()
    )
    cases = ",\n".join(f"    {line(case)}" for case in fixture["cases"])
    return (
        "{\n"
        f'  "reasoning_rules": {line(fixture["reasoning_rules"])},\n'
        f'  "kwargs_hooks": {line(fixture["kwargs_hooks"])},\n'
        f'  "renderings": {{\n{renderings}\n  }},\n'
        f'  "cases": [\n{cases}\n  ]\n'
        "}\n"
    )


@pytest.fixture(scope="module")
def v41_encoder(tmp_path_factory):
    return load_v41_encoder(tmp_path_factory.mktemp("deepseek_v41"))


@pytest.fixture
def servers(v41_encoder, monkeypatch):
    return start_servers(
        v41_encoder,
        lambda encoder: monkeypatch.setattr(
            api_server, "custom_message_encoder", encoder
        ),
    )


def test_the_fixture_is_kept_a_record_per_line():
    assert FIXTURE_PATH.read_text() == dump(FIXTURE)


def test_the_fixture_names_atoms_rules_and_kwargs_hooks():
    assert FIXTURE["reasoning_rules"] == atoms_rules()
    assert FIXTURE["kwargs_hooks"] == atoms_kwargs_hooks()


def test_servers_report_the_fixture_renderings(servers):
    assert sorted(FIXTURE["renderings"]) == sorted(servers)
    for name, server in servers.items():
        assert chat_rendering(*server) == FIXTURE["renderings"][name], name


@pytest.mark.parametrize(
    "case",
    FIXTURE["cases"],
    ids=lambda case: f"{case['rendering']} {json.dumps(case['request'])}",
)
def test_chats_render_with_the_fixture_kwargs(servers, case):
    server = servers[case["rendering"]]
    if case["kwargs"] is None:
        with pytest.raises(ValueError):
            chat_kwargs(server, case["request"])
    else:
        assert chat_kwargs(server, case["request"]) == case["kwargs"]


def test_v41_reports_the_pinned_encoders_rendering():
    model = os.environ.get("ATOM_DSV41_MODEL")
    if not model:
        pytest.skip("set ATOM_DSV41_MODEL to check the official encoder's rendering")
    encoder = _load_encoder_from_dir(model)
    rendering = chat_rendering(
        {}, resolve_reasoning_toggle(None, encoder), encoder, encoder.reasoning_rules
    )
    assert rendering == FIXTURE["renderings"]["deepseek_v41"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        servers = start_servers(
            load_v41_encoder(Path(directory)),
            lambda encoder: setattr(api_server, "custom_message_encoder", encoder),
        )

        def kwargs(case):
            try:
                return chat_kwargs(servers[case["rendering"]], case["request"])
            except ValueError:
                return None

        fixture = {
            "reasoning_rules": atoms_rules(),
            "kwargs_hooks": atoms_kwargs_hooks(),
            "renderings": {
                name: chat_rendering(*server) for name, server in servers.items()
            },
            "cases": [{**case, "kwargs": kwargs(case)} for case in FIXTURE["cases"]],
        }
    FIXTURE_PATH.write_text(dump(fixture))
