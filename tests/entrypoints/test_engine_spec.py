# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`--engine-dispatch direct`: what it refuses before the weights load, the
`EngineSpec` it hands Rust, and that the launcher reaps the EngineCores
however Rust exits."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from atom.entrypoints.atomesh import engine_spec
from atom.entrypoints.atomesh import server as atomesh_server
from atom.sampling_params import RequiredChannel

FIXTURE = (
    pathlib.Path(__file__).parents[1] / "fixtures" / "atom_engine" / "engine_spec.json"
)
BASE = ["--model", "/x", "--port", "9000"]


def _args(**overrides):
    fields = {"kv_transfer_config": None, "dp_load_balance": "least_requests"}
    return SimpleNamespace(**{**fields, **overrides})


@pytest.mark.parametrize(
    "transfer",
    [
        '{"kv_connector": "moriio", "kv_role": "kv_consumer"}',
        # A Python literal, as some recipes write it, and a nested P/D entry.
        "{'kv_connector': 'multi', 'connectors': [{'kv_connector': 'mooncake'}]}",
    ],
)
def test_a_pd_worker_is_refused(transfer):
    with pytest.raises(ValueError, match="P/D worker"):
        engine_spec.check_direct_args(_args(kv_transfer_config=transfer))


@pytest.mark.parametrize("strategy", ["share_cache", "prefix_spread"])
def test_a_prefix_routed_placement_is_refused(strategy):
    with pytest.raises(ValueError, match="--engine-dispatch bridge"):
        engine_spec.check_direct_args(_args(dp_load_balance=strategy))


@pytest.mark.parametrize("strategy", engine_spec.DIRECT_DP_STRATEGIES)
def test_what_rust_places_by_is_served(strategy):
    engine_spec.check_direct_args(_args(dp_load_balance=strategy))


def test_a_chat_python_renders_is_refused():
    with pytest.raises(ValueError, match="rendered in Python"):
        engine_spec.check_direct_tokenizer(SimpleNamespace(needed=True), "/m")
    engine_spec.check_direct_tokenizer(SimpleNamespace(needed=False), "/m")


def test_the_spec_is_what_rust_reads(monkeypatch):
    """Every field is one `spec.rs` declares (the cross-language fixture
    lists them), and the values are the deployment's."""
    channel = RequiredChannel(after=(1,), opening=(2,), opens_at=1, end_ids=(3,))
    monkeypatch.setattr(
        engine_spec, "required_call_channel", lambda parser, tokenizer: channel
    )
    endpoints = [{"dp_rank": 0, "pid": 7, "input": "i", "control": "c", "output": "o"}]
    engine = SimpleNamespace(
        config=SimpleNamespace(
            model="/models/m",
            max_model_len=4096,
            end_token_ids=(3, 9),
            dp_load_balance="least_tokens",
            hf_config=None,
        ),
        core_mgr=SimpleNamespace(engine_endpoints=endpoints),
    )
    renderer = SimpleNamespace(encoder=None, tool_parser_cls=object)
    spec = json.loads(
        engine_spec.build_engine_spec(engine, None, renderer, "served", {"k": 1})
    )
    fixture = json.loads(FIXTURE.read_text())
    assert spec.keys() == fixture.keys()
    assert spec["model"].keys() <= {*fixture["model"], "end_token_ids"}
    assert spec["balance"].keys() == fixture["balance"].keys()
    model = spec["model"]
    assert (model["served_name"], model["tokenizer_path"]) == ("served", "/models/m")
    assert model["default_chat_template_kwargs"] == {"k": 1}
    assert model["end_token_ids"] == [3, 9]
    assert model["required_channel"] == {
        "after": [1],
        "opening": [2],
        "opens_at": 1,
        "end_ids": [3],
    }
    assert spec["balance"]["strategy"] == "least_tokens"
    assert spec["engines"] == endpoints


def test_engine_dispatch_is_the_launchers_not_the_routers():
    args = atomesh_server.parse_standalone_args([*BASE, "--engine-dispatch", "direct"])
    assert args.engine_args.engine_dispatch == "direct"
    assert "--engine-dispatch" not in args.mesh_args
    default = atomesh_server.parse_standalone_args(BASE)
    assert default.engine_args.engine_dispatch == "bridge"


@pytest.mark.parametrize("fails", [False, True])
def test_the_cores_are_reaped_however_rust_exits(monkeypatch, fails):
    closed = []
    launched = []
    engine = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(atomesh_server, "engine", engine)
    monkeypatch.setattr(atomesh_server, "tokenizer", None)
    monkeypatch.setattr(engine_spec, "build_engine_spec", lambda *a: "{}")

    def launch_mesh(**kwargs):
        launched.append(kwargs)
        if fails:
            raise RuntimeError("Atomesh exited: a Core was lost")

    runner = SimpleNamespace(launch_mesh=launch_mesh)
    parsed = {"server_config": "config"}
    if fails:
        with pytest.raises(RuntimeError, match="Core was lost"):
            atomesh_server.serve_direct(runner, parsed, None, "m", {})
    else:
        atomesh_server.serve_direct(runner, parsed, None, "m", {})
    assert launched == [{"server_config": "config", "engine_spec": "{}"}]
    assert closed == [True]
