# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `EngineSpec` the atomesh Rust runtime starts from, and the launch that
builds it: what Python works out once so Rust serves requests as the Python
server does."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.entrypoints.atomesh import engine_spec
from atom.entrypoints.atomesh import server as atomesh_server
from atom.entrypoints.openai.tool_parser.registry import PARSERS_BY_NAME
from atom.sampling_params import RequiredChannel

MESH_NAMES = {
    "qwen",
    "qwen_xml",
    "json",
    "kimi",
    "kimi_k3",
    "glm",
    "minimax",
    "dsml",
    "dsml_v41",
}


def fake_engine(speculative=None):
    config = SimpleNamespace(
        model="/models/m",
        max_model_len=4096,
        hf_config=SimpleNamespace(max_position_embeddings=2048),
        end_token_ids=(9, 3),
        dp_load_balance="least_requests",
        speculative_config=speculative,
    )
    return SimpleNamespace(
        config=config,
        core_mgr=SimpleNamespace(engine_endpoints=[{"dp_rank": 0}]),
    )


def build(monkeypatch, *, toggle=None, channel=None, speculative=None):
    monkeypatch.setattr(engine_spec, "resolve_reasoning_toggle", lambda tok: toggle)
    monkeypatch.setattr(engine_spec, "required_call_channel", lambda cls, tok: channel)
    spec = engine_spec.build_engine_spec(
        fake_engine(speculative), object(), "served", {"k": 1}, None, "/t/chat.jinja"
    )
    return json.loads(spec)["model"]


def test_the_output_cap_and_card_are_the_deployments(monkeypatch):
    model = build(monkeypatch)
    # min(max_model_len, max_position_embeddings), as /v1/models reports it.
    assert model["max_output_tokens"] == 2048
    assert model["model_card"]["context_window"] == 2048
    assert model["served_name"] == "served"
    assert model["chat_template"] == "/t/chat.jinja"
    assert model["required_channel"] is None
    assert model["reasoning_toggle"] is None
    assert model["synthetic_acceptance"] is False


def test_only_forced_acceptance_withholds_logprobs(monkeypatch):
    mtp = SimpleNamespace(num_speculative_tokens=3, synthetic_acceptance_rates=None)
    assert build(monkeypatch, speculative=mtp)["synthetic_acceptance"] is False
    forced = SimpleNamespace(num_speculative_tokens=3, synthetic_acceptance_rates=[0.7])
    assert build(monkeypatch, speculative=forced)["synthetic_acceptance"] is True


def test_the_reasoning_switch_is_handed_over(monkeypatch):
    model = build(monkeypatch, toggle=("thinking_mode", "disabled", "enabled"))
    assert model["reasoning_toggle"] == {
        "name": "thinking_mode",
        "off": "disabled",
        "on": "enabled",
    }


def test_the_required_channel_also_ends_on_the_engines_end_ids(monkeypatch):
    channel = RequiredChannel(after=(1,), opening=(2, 5), opens_at=1, end_ids=(7,))
    model = build(monkeypatch, channel=channel)
    assert model["required_channel"] == {
        "after": [1],
        "opening": [2, 5],
        "opens_at": 1,
        "end_ids": [3, 7, 9],
    }


def test_a_dialect_mesh_cannot_split_is_refused(monkeypatch):
    template = "{{ '<mm:think>' }}...{{ '</mm:think>' }}"
    monkeypatch.setattr(engine_spec, "chat_template_source", lambda tok: template)
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "")
    with pytest.raises(ValueError, match="USE_ATOMESH_ENTRYPOINTS=0"):
        engine_spec.default_reasoning_parser(object())


def test_plain_think_maps_to_mesh_parsers(monkeypatch):
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        engine_spec, "chat_template_source", lambda tok: "<think>{{ x }}</think>"
    )
    assert engine_spec.default_reasoning_parser(object()) == "qwen3"
    # A template that closes a block it never opens starts in reasoning.
    monkeypatch.setattr(
        engine_spec,
        "template_opens_reasoning_implicitly",
        lambda source: True,
    )
    assert engine_spec.default_reasoning_parser(object()) == "deepseek_r1"


def test_every_atom_tool_format_has_a_mesh_name():
    assert set(engine_spec.MESH_TOOL_PARSERS) == set(PARSERS_BY_NAME)


def tool_format(monkeypatch, requested, resolved=None):
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        engine_spec,
        "resolve_tool_call_parser",
        lambda name, *a, **k: PARSERS_BY_NAME.get(name) or resolved,
    )
    return engine_spec.tool_call_format(requested, object(), "/m", MESH_NAMES)


def test_atoms_qwen_is_meshs_xml_parser(monkeypatch):
    fmt = tool_format(monkeypatch, "qwen")
    assert fmt == (PARSERS_BY_NAME["qwen"], "qwen_xml")


def test_a_mesh_only_name_binds_mesh_alone(monkeypatch):
    assert tool_format(monkeypatch, "json") == (None, "json")


def test_an_unknown_name_is_refused_before_the_weights_load(monkeypatch):
    with pytest.raises(ValueError, match="unknown"):
        tool_format(monkeypatch, "qwenn")


def test_unset_reads_the_template_and_falls_back_to_mesh(monkeypatch):
    glm = PARSERS_BY_NAME["glm"]
    assert tool_format(monkeypatch, None, resolved=glm) == (glm, "glm")
    # The template names no format: mesh keeps its guess by model name.
    assert tool_format(monkeypatch, None) == (None, None)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
def test_a_pd_worker_is_refused(monkeypatch, role):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    args = SimpleNamespace(model="/m", kv_transfer_config=json.dumps({"kv_role": role}))
    with pytest.raises(ValueError, match="P/D"):
        engine_spec.check_standalone_model(args)


def test_kv_offload_is_not_a_pd_worker(monkeypatch):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    args = SimpleNamespace(model="/m", kv_transfer_config='{"kv_role": "offload"}')
    engine_spec.check_standalone_model(args)


def test_the_template_rust_renders_is_the_one_probed(tmp_path):
    tokenizer = SimpleNamespace(chat_template="{{ hf }}")
    path = engine_spec.apply_chat_template(tokenizer, None, str(tmp_path))
    assert Path(path).read_text() == "{{ hf }}"
    override = tmp_path / "mine.jinja"
    override.write_text("{{ mine }}")
    path = engine_spec.apply_chat_template(tokenizer, str(override), str(tmp_path))
    assert tokenizer.chat_template == "{{ mine }}"
    assert Path(path).read_text() == "{{ mine }}"
    engine_spec.apply_chat_template(tokenizer, "{{ inline }}", str(tmp_path))
    assert tokenizer.chat_template == "{{ inline }}"
    assert (
        engine_spec.apply_chat_template(
            SimpleNamespace(chat_template=None), None, str(tmp_path)
        )
        is None
    )


def test_launch_decides_everything_before_weights_and_reaps_after(monkeypatch):
    """The refusals and the template, reasoning and tool decisions run before
    the engine (and its weights) load; the Cores are reaped whatever happens."""
    calls = []
    monkeypatch.setattr(
        atomesh_server, "check_standalone_model", lambda a: calls.append("check")
    )
    monkeypatch.setattr(
        atomesh_server,
        "load_tokenizer",
        lambda a: calls.append("tokenizer") or object(),
    )
    monkeypatch.setattr(
        atomesh_server, "apply_chat_template", lambda *a: calls.append("template")
    )
    monkeypatch.setattr(
        atomesh_server,
        "default_reasoning_parser",
        lambda t: calls.append("reasoning") or "qwen3",
    )
    monkeypatch.setattr(
        atomesh_server,
        "tool_call_format",
        lambda *a: calls.append("tools")
        or engine_spec.ToolCallFormat(None, "qwen_xml"),
    )
    engine = SimpleNamespace(close=lambda: calls.append("close"))

    def initialize(args):
        calls.append("engine")
        atomesh_server.engine = engine

    monkeypatch.setattr(atomesh_server, "initialize_engine", initialize)
    monkeypatch.setattr(atomesh_server, "build_engine_spec", lambda *a: "{}")
    for name in ("tune_gc", "maybe_attach_gc_debug_callback", "freeze_gc_heap"):
        monkeypatch.setattr(atomesh_server, name, lambda *a: None)

    runner = SimpleNamespace(
        tool_parser_names=lambda: ["qwen_xml"],
        parse_from=lambda args: calls.append(("mesh_args", args))
        or {"cli_args": {"host": "h", "port": 1}, "server_config": None},
        launch_mesh=lambda **kw: (_ for _ in ()).throw(RuntimeError("core lost")),
    )
    with pytest.raises(RuntimeError, match="core lost"):
        atomesh_server.launch_atom_standalone(
            runner, ["--model", "/m", "--tool-call-parser", "qwen", "--port", "9"]
        )
    order = [c for c in calls if isinstance(c, str)]
    assert order == [
        "check",
        "tokenizer",
        "template",
        "reasoning",
        "tools",
        "engine",
        "close",
    ]
    [(_, mesh_args)] = [c for c in calls if isinstance(c, tuple)]
    assert mesh_args == [
        "--port",
        "9",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen_xml",
    ]
