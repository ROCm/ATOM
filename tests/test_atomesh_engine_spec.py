# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `EngineSpec` the atomesh Rust runtime starts from: what Python works
out once at launch so Rust serves requests as the Python server does."""

import json
from types import SimpleNamespace

import pytest

from atom.entrypoints.atomesh import engine_spec
from atom.sampling_params import RequiredChannel


def fake_engine():
    config = SimpleNamespace(
        model="/models/m",
        max_model_len=4096,
        hf_config=SimpleNamespace(max_position_embeddings=2048),
        end_token_ids=(9, 3),
        dp_load_balance="least_requests",
    )
    return SimpleNamespace(
        config=config,
        io_processor=SimpleNamespace(num_speculative_tokens=0),
        core_mgr=SimpleNamespace(engine_endpoints=[{"dp_rank": 0}]),
    )


def build(monkeypatch, *, toggle=None, channel=None):
    monkeypatch.setattr(engine_spec, "resolve_reasoning_toggle", lambda tok: toggle)
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "")
    monkeypatch.setattr(engine_spec, "resolve_tool_call_parser", lambda *a, **k: None)
    monkeypatch.setattr(engine_spec, "required_call_channel", lambda cls, tok: channel)
    spec = engine_spec.build_engine_spec(fake_engine(), object(), "m", {"k": 1})
    return json.loads(spec)["model"]


def test_the_output_cap_is_the_deployments(monkeypatch):
    model = build(monkeypatch)
    # min(max_model_len, max_position_embeddings), as /v1/models reports it.
    assert model["max_output_tokens"] == 2048
    assert model["required_channel"] is None
    assert model["reasoning_toggle"] is None


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
