"""The atomesh entrypoint against `atomesh_runner.launch_mesh`'s signature.

`launch_mesh` is a PyO3 function (`atom/mesh/src/python.rs`) that takes only
keyword arguments; a keyword it does not declare is a TypeError at startup,
which no Rust test sees. The runner here is a fake that holds the entrypoint
to the keywords the Rust signature declares.
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from atom.entrypoints.atomesh import engine_spec
from atom.entrypoints.atomesh import server as atomesh_server

_PYTHON_RS = Path(__file__).resolve().parents[2] / "atom/mesh/src/python.rs"


def _launch_mesh_keywords() -> set[str]:
    """The keywords `#[pyo3(signature = (*, ...))]` gives `launch_mesh`."""
    src = _PYTHON_RS.read_text()
    match = re.search(
        r"#\[pyo3\(signature = \(\*, ([^)]*)\)\)\]\s*pub fn launch_mesh\(", src
    )
    assert match, "launch_mesh's signature is not where this test reads it"
    return {param.split("=")[0].strip() for param in match.group(1).split(",")}


class _Runner:
    def __init__(self):
        self.launches = []

    def parse_from(self, args):
        return {
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
        }

    def tool_parser_names(self):
        return []

    def launch_mesh(self, **kwargs):
        unknown = set(kwargs) - _launch_mesh_keywords()
        if unknown:
            raise TypeError(f"launch_mesh() got unexpected keywords {unknown}")
        self.launches.append(kwargs)


def main(argv, runner):
    """`server.main()` over `argv`, with `runner` as the Rust runtime."""
    with (
        mock.patch.object(sys, "argv", ["atomesh", *argv]),
        mock.patch.object(atomesh_server, "import_atomesh_runner", return_value=runner),
    ):
        atomesh_server.main()


def test_mesh_only_launches_with_keywords_launch_mesh_takes():
    runner = _Runner()
    main(["mesh-only", "--port", "8000"], runner)
    assert len(runner.launches) == 1


def test_standalone_launches_with_keywords_launch_mesh_takes():
    """The engine spec reaches `launch_mesh`, and the Cores are reaped after."""
    runner = _Runner()
    engine = mock.Mock()

    def initialize_engine(_args):
        atomesh_server.engine = engine

    tool_format = SimpleNamespace(parser=None, mesh_name=None)
    with (
        mock.patch.object(atomesh_server, "check_standalone_model"),
        mock.patch.object(atomesh_server, "load_tokenizer"),
        mock.patch.object(atomesh_server, "apply_chat_template", return_value=None),
        mock.patch.object(
            atomesh_server, "default_reasoning_parser", return_value="qwen3"
        ),
        mock.patch.object(atomesh_server, "tool_call_format", return_value=tool_format),
        mock.patch.object(atomesh_server, "initialize_engine", initialize_engine),
        mock.patch.object(atomesh_server, "build_engine_spec", return_value="spec"),
        mock.patch.object(atomesh_server, "freeze_gc_heap"),
    ):
        main(["--model", "/m"], runner)
    [launch] = runner.launches
    assert set(launch) == {"server_config", "engine_spec"}
    assert launch["engine_spec"] == "spec"
    engine.close.assert_called_once_with()


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
