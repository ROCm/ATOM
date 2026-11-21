# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `EngineSpec` the atomesh Rust runtime starts from, and the launch that
builds it: what Python works out once so Rust serves requests as the Python
server does.

`launch_mesh` is a PyO3 function (`atom/mesh/src/python.rs`) that takes only
keyword arguments; a keyword it does not declare is a TypeError at startup,
which no Rust test sees, so the fake runners here hold the entrypoint to the
keywords the Rust signature declares.
"""

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
SPEC_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "atom_engine" / "engine_spec.json"
)
_PYTHON_RS = Path(__file__).resolve().parents[2] / "atom/mesh/src/python.rs"


def build(monkeypatch, *, speculative=None, endpoints=1):
    """The spec for a fake engine whose template has a reasoning switch, a
    required channel and string tool-call arguments."""
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
    monkeypatch.setattr(engine_spec, "preserve_tool_call_arguments", lambda tok: True)
    config = SimpleNamespace(
        model="/models/m",
        max_model_len=4096,
        hf_config=SimpleNamespace(max_position_embeddings=2048),
        end_token_ids=(9, 3),
        dp_load_balance="least_requests",
        speculative_config=speculative,
    )
    cores = [
        {
            "dp_rank": rank,
            "pid": 100 + rank,
            "input": f"ipc:///tmp/core-{rank}-input",
            "control": f"ipc:///tmp/core-{rank}-control",
            "output": f"ipc:///tmp/core-{rank}-output",
        }
        for rank in range(endpoints)
    ]
    engine = SimpleNamespace(
        config=config, core_mgr=SimpleNamespace(engine_endpoints=cores)
    )
    spec = engine_spec.build_engine_spec(
        engine,
        object(),
        "served",
        {"k": 1},
        None,
        "/t/chat.jinja",
        synthetic_text="synthetic ",
    )
    return json.loads(spec)


def test_the_spec_matches_what_rust_parses(monkeypatch):
    """The spec as Python writes it, kept for `atom_engine::tests` to parse
    with `EngineSpec::from_json`; `ATOM_UPDATE_FIXTURES=1` rewrites it. It
    pins every field the decisions hand over: the output cap and card, the
    reasoning switch, the required channel ending on the engine's end ids too,
    the named efforts, preserved arguments and the synthetic text."""
    text = json.dumps(build(monkeypatch, endpoints=2), indent=2, sort_keys=True)
    text += "\n"
    if os.environ.get("ATOM_UPDATE_FIXTURES"):
        SPEC_FIXTURE.write_text(text)
        pytest.fail(f"rewrote {SPEC_FIXTURE.name}; rerun without ATOM_UPDATE_FIXTURES")
    assert SPEC_FIXTURE.read_text() == text, (
        f"{SPEC_FIXTURE.name} is stale: rerun with ATOM_UPDATE_FIXTURES=1, then "
        "the mesh tests (cargo test --lib atom_engine) against it"
    )


def test_only_forced_acceptance_withholds_logprobs(monkeypatch):
    mtp = SimpleNamespace(num_speculative_tokens=3, synthetic_acceptance_rates=None)
    assert build(monkeypatch, speculative=mtp)["model"]["synthetic_acceptance"] is False
    forced = SimpleNamespace(num_speculative_tokens=3, synthetic_acceptance_rates=[0.7])
    assert build(monkeypatch, speculative=forced)["model"]["synthetic_acceptance"]


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


@pytest.mark.parametrize(
    "model_type, files, refused",
    [
        ("gpt_oss", ["tokenizer.json"], "harmony"),
        ("llama", ["tiktoken.model", "tokenizer_config.json"], "tiktoken"),
        ("llama", ["o200k.tiktoken", "tokenizer_config.json"], "tiktoken"),
        # Mesh loads tokenizer.json first, as a tokenizer that renders templates.
        ("llama", ["tiktoken.model", "tokenizer.json"], None),
    ],
)
def test_a_checkpoint_mesh_cannot_serve_is_refused(
    tmp_path, model_type, files, refused
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    for name in files:
        (tmp_path / name).write_text("")
    if refused is None:
        engine_spec.check_standalone_checkpoint(str(tmp_path))
        return
    with pytest.raises(ValueError, match=refused):
        engine_spec.check_standalone_checkpoint(str(tmp_path))


@pytest.mark.parametrize(
    "template, implicit, parser",
    [
        ("<think>{{ x }}</think>", False, "qwen3"),
        # A template that closes a block it never opens starts in reasoning.
        ("<think>{{ x }}</think>", True, "deepseek_r1"),
        ("{{ '<mm:think>' }}...{{ '</mm:think>' }}", False, None),
    ],
)
def test_a_reasoning_dialect_maps_to_a_mesh_parser_or_is_refused(
    monkeypatch, template, implicit, parser
):
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "")
    monkeypatch.setattr(engine_spec, "chat_template_source", lambda tok: template)
    monkeypatch.setattr(
        engine_spec, "template_opens_reasoning_implicitly", lambda source: implicit
    )
    if parser is None:
        with pytest.raises(ValueError, match="USE_ATOMESH_ENTRYPOINTS=0"):
            engine_spec.default_reasoning_parser(object())
        return
    assert engine_spec.default_reasoning_parser(object()) == parser


def test_every_atom_tool_format_has_a_mesh_name():
    assert set(engine_spec.MESH_TOOL_PARSERS) == set(PARSERS_BY_NAME)


@pytest.mark.parametrize(
    "requested, resolved, expected",
    [
        ("qwen", None, ("qwen", "qwen_xml")),  # ATOM's qwen is mesh's XML parser
        ("json", None, (None, "json")),  # a mesh-only name binds mesh alone
        (None, "glm", ("glm", "glm47_moe")),  # unset: the template's format
        (None, None, (None, None)),  # no format: mesh guesses by model name
    ],
)
def test_a_tool_call_format_binds_both_halves(
    monkeypatch, requested, resolved, expected
):
    monkeypatch.setattr(engine_spec, "render_probe_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        engine_spec,
        "resolve_tool_call_parser",
        lambda name, *a, **k: PARSERS_BY_NAME.get(name)
        or PARSERS_BY_NAME.get(resolved),
    )
    fmt = engine_spec.tool_call_format(requested, object(), "/m", MESH_NAMES)
    parser, mesh_name = expected
    assert fmt == (PARSERS_BY_NAME.get(parser), mesh_name)
    # An unknown name is refused before the weights load.
    with pytest.raises(ValueError, match="unknown"):
        engine_spec.tool_call_format("qwenn", object(), "/m", MESH_NAMES)


@pytest.mark.parametrize(
    "transfer, dp_load_balance, refused",
    [
        ({"kv_role": "kv_producer"}, "least_requests", "P/D"),
        ({"kv_role": "kv_consumer"}, "least_requests", "P/D"),
        ({"kv_role": "kv_both"}, "least_requests", "P/D"),
        # moriio is the connector a config that names none builds, as a
        # producer when it names no role either.
        ({"kv_connector": "moriio"}, "least_requests", "P/D"),
        ({"proxy_ip": "10.0.0.1"}, "least_requests", "P/D"),
        (
            {"kv_connector": "Mooncake", "kv_role": "kv_consumer"},
            "least_requests",
            "P/D",
        ),
        # The Agentic-GLM-5.2 recipe's prefill node: the role is the leaf's.
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            "least_requests",
            "P/D",
        ),
        # A Python-literal config, read as the engine reads it.
        (
            "{'kv_connector': 'mooncake', 'kv_role': 'kv_producer'}",
            "least_requests",
            "P/D",
        ),
        (
            "{'kv_connector': 'lmcache_offload', 'kv_role': 'kv_both'}",
            "least_requests",
            None,
        ),
        # An offload tier is no P/D worker, whatever its role.
        *(
            (
                {"kv_connector": "lmcache_offload", "kv_role": role},
                "least_requests",
                None,
            )
            for role in ("offload", "kv_both", "kv_producer", "kv_consumer")
        ),
        ({}, "least_requests", None),
        ({}, "share_cache", "dp-load-balance"),
        ({}, "prefix_spread", "dp-load-balance"),
    ],
)
def test_standalone_refuses_what_rust_cannot_serve(
    monkeypatch, transfer, dp_load_balance, refused
):
    monkeypatch.setattr(engine_spec, "load_custom_message_encoder", lambda m: None)
    args = SimpleNamespace(
        model="/m",
        kv_transfer_config=(
            transfer if isinstance(transfer, str) else json.dumps(transfer)
        ),
        dp_load_balance=dp_load_balance,
    )
    if refused is None:
        engine_spec.check_standalone_model(args)
        return
    with pytest.raises(ValueError, match=refused):
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


def launch_mesh_keywords() -> set[str]:
    """The keywords `#[pyo3(signature = (*, ...))]` gives `launch_mesh`."""
    src = _PYTHON_RS.read_text()
    match = re.search(
        r"#\[pyo3\(signature = \(\*, ([^)]*)\)\)\]\s*pub fn launch_mesh\(", src
    )
    assert match, "launch_mesh's signature is not where this test reads it"
    return {param.split("=")[0].strip() for param in match.group(1).split(",")}


@pytest.fixture
def launch(monkeypatch):
    """`launch_atom_standalone` with each decision stubbed and recorded, in
    the order it runs, in the list this returns. The runner's `launch_mesh`
    takes only the keywords the Rust signature declares, then loses a Core."""
    calls = []

    def record(name, result=None):
        return lambda *a, **k: calls.append(name) or result

    for name, result in (
        ("check_standalone_model", None),
        ("load_tokenizer", object()),
        ("check_standalone_checkpoint", None),
        ("apply_chat_template", None),
        ("default_reasoning_parser", "qwen3"),
        ("tool_call_format", engine_spec.ToolCallFormat(None, "qwen_xml")),
        ("build_engine_spec", "spec"),
    ):
        monkeypatch.setattr(atomesh_server, name, record(name, result))
    for name in ("tune_gc", "maybe_attach_gc_debug_callback", "freeze_gc_heap"):
        monkeypatch.setattr(atomesh_server, name, lambda *a: None)

    def initialize_engine(args):
        calls.append("initialize_engine")
        atomesh_server.engine = SimpleNamespace(close=lambda: calls.append("close"))

    def launch_mesh(**kwargs):
        unknown = set(kwargs) - launch_mesh_keywords()
        assert not unknown, f"launch_mesh() takes no {unknown}"
        calls.append(("launch_mesh", kwargs))
        raise RuntimeError("core lost")

    runner = SimpleNamespace(
        tool_parser_names=lambda: ["qwen_xml"],
        parse_from=lambda args: calls.append(("mesh_args", args))
        or {"cli_args": {"host": "h", "port": 1}, "server_config": "config"},
        launch_mesh=launch_mesh,
    )
    monkeypatch.setattr(atomesh_server, "initialize_engine", initialize_engine)
    monkeypatch.setattr(
        atomesh_server,
        "import_atomesh_runner",
        lambda: calls.append("import_atomesh_runner") or runner,
    )
    return calls


def test_launch_decides_everything_before_weights_and_reaps_after(launch):
    """The refusals run before the Rust runtime is imported, and they and the
    template, reasoning and tool decisions before the engine (and its weights)
    load; the spec reaches `launch_mesh`, and the Cores are reaped whatever
    happens."""
    with pytest.raises(RuntimeError, match="core lost"):
        atomesh_server.launch_atom_standalone(
            ["--model", "/m", "--tool-call-parser", "qwen", "--port", "9"]
        )
    assert [c for c in launch if isinstance(c, str)] == [
        "check_standalone_model",
        "import_atomesh_runner",
        "load_tokenizer",
        "check_standalone_checkpoint",
        "apply_chat_template",
        "default_reasoning_parser",
        "tool_call_format",
        "initialize_engine",
        "build_engine_spec",
        "close",
    ]
    calls = dict(c for c in launch if isinstance(c, tuple))
    assert calls["mesh_args"] == [
        "--port",
        "9",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen_xml",
    ]
    assert calls["launch_mesh"] == {"server_config": "config", "engine_spec": "spec"}


def test_a_named_reasoning_parser_is_handed_over_once(launch, monkeypatch):
    monkeypatch.setattr(
        atomesh_server,
        "default_reasoning_parser",
        lambda t: pytest.fail("the named parser is not read off the template"),
    )
    with pytest.raises(RuntimeError, match="core lost"):
        atomesh_server.launch_atom_standalone(
            ["--model", "/m", "--reasoning_parser", "step3"]
        )
    mesh_args = dict(c for c in launch if isinstance(c, tuple))["mesh_args"]
    assert mesh_args.count("--reasoning-parser") == 1
    assert mesh_args[mesh_args.index("--reasoning-parser") + 1] == "step3"


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


def test_mesh_only_launches_with_keywords_launch_mesh_takes():
    launched = []

    def launch_mesh(**kwargs):
        assert set(kwargs) <= launch_mesh_keywords()
        launched.append(kwargs)

    runner = SimpleNamespace(
        parse_from=lambda args: {
            "cli_args": {
                "host": "127.0.0.1",
                "port": 8000,
                "pd_disaggregation": False,
                "backend": "sglang",
                "policy": "round_robin",
            },
            "server_config": object(),
            "prefill_urls": [],
            "decode_urls": [],
        },
        launch_mesh=launch_mesh,
    )
    with (
        mock.patch.object(sys, "argv", ["atomesh", "mesh-only", "--port", "8000"]),
        mock.patch.object(atomesh_server, "import_atomesh_runner", return_value=runner),
    ):
        atomesh_server.main()
    assert len(launched) == 1


def test_standalone_refuses_before_it_imports_the_rust_runtime():
    """A model standalone cannot serve is refused even where libmesh is not
    installed, instead of failing on the import first."""
    pd = json.dumps({"kv_role": "kv_producer"})
    with (
        mock.patch.object(
            engine_spec, "load_custom_message_encoder", return_value=None
        ),
        mock.patch.object(
            sys, "argv", ["atomesh", "--model", "/m", "--kv-transfer-config", pd]
        ),
        mock.patch.object(
            atomesh_server,
            "import_atomesh_runner",
            side_effect=ModuleNotFoundError("atomesh_runner"),
        ),
        pytest.raises(ValueError, match="P/D"),
    ):
        atomesh_server.main()
