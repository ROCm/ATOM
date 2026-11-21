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
    "glm47_moe",
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


def build(
    monkeypatch,
    *,
    toggle=None,
    channel=None,
    speculative=None,
    preserve=False,
    synthetic_text=None,
):
    monkeypatch.setattr(engine_spec, "resolve_reasoning_toggle", lambda tok: toggle)
    monkeypatch.setattr(engine_spec, "required_call_channel", lambda cls, tok: channel)
    monkeypatch.setattr(
        engine_spec, "preserve_tool_call_arguments", lambda tok: preserve
    )
    spec = engine_spec.build_engine_spec(
        fake_engine(speculative),
        object(),
        "served",
        {"k": 1},
        None,
        "/t/chat.jinja",
        synthetic_text=synthetic_text,
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
    assert model["synthetic_token_text"] is None
    assert model["preserve_tool_call_arguments"] is False


def test_the_python_servers_text_decisions_are_handed_over(monkeypatch):
    model = build(monkeypatch, preserve=True, synthetic_text="synthetic ")
    assert model["preserve_tool_call_arguments"] is True
    assert model["synthetic_token_text"] == "synthetic "


@pytest.mark.parametrize(
    "length, rate, text",
    [(None, None, None), (2.5, None, "synthetic "), (None, 0.7, "synthetic ")],
)
def test_synthetic_text_is_decided_off_the_python_servers_flags(length, rate, text):
    args = SimpleNamespace(
        spec_decode_acceptance_length=length, spec_decode_acceptance_rate=rate
    )
    assert engine_spec.synthetic_token_text(args) == text


@pytest.mark.parametrize(
    "keeps, needs, preserve",
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_tool_call_arguments_are_preserved_as_the_python_server_decides(
    monkeypatch, keeps, needs, preserve
):
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "probe")
    monkeypatch.setattr(
        engine_spec, "template_keeps_tool_call_arguments", lambda probe: keeps
    )
    monkeypatch.setattr(
        engine_spec,
        "template_needs_tool_call_argument_strings",
        lambda tok, encoder: needs,
    )
    assert engine_spec.preserve_tool_call_arguments(object()) is preserve


def write_checkpoint(directory, files, model_type="llama"):
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    for name in files:
        (directory / name).write_text("")
    return str(directory)


def test_gpt_oss_is_refused(tmp_path):
    model = write_checkpoint(tmp_path, ["tokenizer.json"], model_type="gpt_oss")
    with pytest.raises(ValueError, match="harmony"):
        engine_spec.check_standalone_checkpoint(model)


@pytest.mark.parametrize("tiktoken", ["tiktoken.model", "o200k.tiktoken"])
def test_a_tiktoken_only_tokenizer_is_refused(tmp_path, tiktoken):
    model = write_checkpoint(tmp_path, [tiktoken, "tokenizer_config.json"])
    with pytest.raises(ValueError, match="tiktoken"):
        engine_spec.check_standalone_checkpoint(model)


def test_a_tokenizer_json_beside_tiktoken_is_served(tmp_path):
    # Mesh loads tokenizer.json first, as a tokenizer that renders templates.
    engine_spec.check_standalone_checkpoint(
        write_checkpoint(tmp_path, ["tiktoken.model", "tokenizer.json"])
    )


def test_only_forced_acceptance_withholds_logprobs(monkeypatch):
    mtp = SimpleNamespace(num_speculative_tokens=3, synthetic_acceptance_rates=None)
    assert build(monkeypatch, speculative=mtp)["synthetic_acceptance"] is False
    forced = SimpleNamespace(num_speculative_tokens=3, synthetic_acceptance_rates=[0.7])
    assert build(monkeypatch, speculative=forced)["synthetic_acceptance"] is True


def test_the_named_efforts_are_the_python_servers(monkeypatch):
    from atom.entrypoints.openai.reasoning import VALID_TEMPLATE_EFFORTS

    efforts = build(monkeypatch)["template_efforts"]
    assert efforts == sorted(VALID_TEMPLATE_EFFORTS)


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


SPEC_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "atom_engine" / "engine_spec.json"
)


def test_the_spec_matches_what_rust_parses(monkeypatch):
    """The spec as Python writes it, kept for `atom_engine::tests` to parse
    with `EngineSpec::from_json`; `ATOM_UPDATE_FIXTURES=1` rewrites it."""
    import os

    monkeypatch.setattr(
        engine_spec,
        "resolve_reasoning_toggle",
        lambda tok: ("enable_thinking", False, True),
    )
    monkeypatch.setattr(
        engine_spec,
        "required_call_channel",
        lambda cls, tok: RequiredChannel(
            after=(1,), opening=(2,), opens_at=1, end_ids=(7,)
        ),
    )
    engine = fake_engine()
    engine.core_mgr.engine_endpoints = [
        {
            "dp_rank": rank,
            "pid": 100 + rank,
            "input": f"ipc:///tmp/core-{rank}-input",
            "control": f"ipc:///tmp/core-{rank}-control",
            "output": f"ipc:///tmp/core-{rank}-output",
        }
        for rank in range(2)
    ]
    monkeypatch.setattr(engine_spec, "preserve_tool_call_arguments", lambda tok: True)
    spec = json.loads(
        engine_spec.build_engine_spec(
            engine,
            object(),
            "served",
            {"k": 1},
            None,
            "/t/chat.jinja",
            synthetic_text="synthetic ",
        )
    )
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if os.environ.get("ATOM_UPDATE_FIXTURES"):
        SPEC_FIXTURE.write_text(text)
        pytest.fail(f"rewrote {SPEC_FIXTURE.name}; rerun without ATOM_UPDATE_FIXTURES")
    assert SPEC_FIXTURE.read_text() == text, (
        f"{SPEC_FIXTURE.name} is stale: rerun with ATOM_UPDATE_FIXTURES=1, then "
        "the mesh tests (cargo test --lib atom_engine) against it"
    )


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
    assert tool_format(monkeypatch, None, resolved=glm) == (glm, "glm47_moe")
    # The template names no format: mesh keeps its guess by model name.
    assert tool_format(monkeypatch, None) == (None, None)


def standalone_args(kv_transfer_config="{}", dp_load_balance="least_requests"):
    """The EngineArgs fields check_standalone_model reads, as the CLI sets them."""
    return SimpleNamespace(
        model="/m",
        kv_transfer_config=kv_transfer_config,
        dp_load_balance=dp_load_balance,
    )


@pytest.mark.parametrize(
    "transfer",
    [
        {"kv_role": "kv_producer"},
        {"kv_role": "kv_consumer"},
        {"kv_role": "kv_both"},
        # moriio is the connector a config that names none builds, as a
        # producer when it names no role either.
        {"kv_connector": "moriio"},
        {"proxy_ip": "10.0.0.1"},
        {"kv_connector": "Mooncake", "kv_role": "kv_consumer"},
        # The Agentic-GLM-5.2 recipe's prefill node: the role is the leaf's.
        {
            "kv_connector": "multi",
            "connectors": [
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                {"kv_connector": "lmcache_offload", "kv_role": "offload"},
            ],
        },
    ],
)
def test_a_pd_worker_is_refused(monkeypatch, transfer):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    with pytest.raises(ValueError, match="P/D"):
        engine_spec.check_standalone_model(standalone_args(json.dumps(transfer)))


def test_a_python_literal_transfer_config_is_read_as_the_engine_reads_it(
    monkeypatch,
):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    literal = "{'kv_connector': 'mooncake', 'kv_role': 'kv_producer', 'x': True}"
    with pytest.raises(ValueError, match="P/D"):
        engine_spec.check_standalone_model(standalone_args(literal))
    offload = "{'kv_connector': 'lmcache_offload', 'kv_role': 'kv_both'}"
    engine_spec.check_standalone_model(standalone_args(offload))


@pytest.mark.parametrize("strategy", ["share_cache", "prefix_spread"])
def test_a_dp_placement_rust_does_not_have_is_refused(monkeypatch, strategy):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    with pytest.raises(ValueError, match="dp-load-balance"):
        engine_spec.check_standalone_model(standalone_args(dp_load_balance=strategy))


@pytest.mark.parametrize("role", ["offload", "kv_both", "kv_producer", "kv_consumer"])
def test_kv_offload_is_not_a_pd_worker(monkeypatch, role):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    transfer = {"kv_connector": "lmcache_offload", "kv_role": role}
    engine_spec.check_standalone_model(standalone_args(json.dumps(transfer)))


def test_no_kv_transfer_is_not_a_pd_worker(monkeypatch):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    engine_spec.check_standalone_model(standalone_args())


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
    """The refusals run before the Rust runtime is imported, and they and the
    template, reasoning and tool decisions before the engine (and its weights)
    load; the Cores are reaped whatever happens."""
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
        atomesh_server,
        "check_standalone_checkpoint",
        lambda m: calls.append("checkpoint"),
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
    monkeypatch.setattr(atomesh_server, "build_engine_spec", lambda *a, **k: "{}")
    for name in ("tune_gc", "maybe_attach_gc_debug_callback", "freeze_gc_heap"):
        monkeypatch.setattr(atomesh_server, name, lambda *a: None)

    runner = SimpleNamespace(
        tool_parser_names=lambda: ["qwen_xml"],
        parse_from=lambda args: calls.append(("mesh_args", args))
        or {"cli_args": {"host": "h", "port": 1}, "server_config": None},
        launch_mesh=lambda **kw: (_ for _ in ()).throw(RuntimeError("core lost")),
    )
    monkeypatch.setattr(
        atomesh_server,
        "import_atomesh_runner",
        lambda: calls.append("runtime") or runner,
    )
    with pytest.raises(RuntimeError, match="core lost"):
        atomesh_server.launch_atom_standalone(
            ["--model", "/m", "--tool-call-parser", "qwen", "--port", "9"]
        )
    order = [c for c in calls if isinstance(c, str)]
    assert order == [
        "check",
        "runtime",
        "tokenizer",
        "checkpoint",
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


def test_the_decided_flags_take_both_spellings_and_need_a_value():
    parsed = atomesh_server.parse_standalone_args(
        [
            "--model",
            "/m",
            "--tool_call_parser",
            "hermes",
            "--chat_template",
            "t.jinja",
            "--reasoning_parser",
            "step3",
            "--server_port",
            "9",
        ]
    )
    args = parsed.engine_args
    assert (args.tool_call_parser, args.chat_template, args.reasoning_parser) == (
        "hermes",
        "t.jinja",
        "step3",
    )
    assert parsed.mesh_args == ["--port", "9"]
    for flag in ("--tool-call-parser", "--chat-template", "--reasoning-parser"):
        with pytest.raises(SystemExit):
            atomesh_server.parse_standalone_args(["--model", "/m", flag])


def test_a_named_reasoning_parser_is_handed_over_once(monkeypatch):
    mesh_args = []
    monkeypatch.setattr(atomesh_server, "check_standalone_model", lambda a: None)
    monkeypatch.setattr(atomesh_server, "load_tokenizer", lambda a: object())
    monkeypatch.setattr(atomesh_server, "check_standalone_checkpoint", lambda m: None)
    monkeypatch.setattr(atomesh_server, "apply_chat_template", lambda *a: None)
    monkeypatch.setattr(
        atomesh_server,
        "default_reasoning_parser",
        lambda t: pytest.fail("the named parser is not read off the template"),
    )
    monkeypatch.setattr(
        atomesh_server,
        "tool_call_format",
        lambda *a: engine_spec.ToolCallFormat(None, None),
    )
    monkeypatch.setattr(
        atomesh_server,
        "serve_standalone",
        lambda runner, standalone, tok, args, *rest: mesh_args.extend(args),
    )
    monkeypatch.setattr(
        atomesh_server,
        "import_atomesh_runner",
        lambda: SimpleNamespace(tool_parser_names=list),
    )
    atomesh_server.launch_atom_standalone(
        ["--model", "/m", "--reasoning_parser", "step3"]
    )
    assert mesh_args == ["--reasoning-parser", "step3"]
