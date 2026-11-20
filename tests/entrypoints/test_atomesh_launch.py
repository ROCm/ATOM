"""The atomesh entrypoint against `atomesh_runner.launch_mesh`'s signature.

`launch_mesh` is a PyO3 function (`atom/mesh/src/python.rs`) that takes only
keyword arguments; a keyword it does not declare is a TypeError at startup,
which no Rust test sees. The runner here is a fake that holds the entrypoint
to the keywords the Rust signature declares.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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

    def launch_mesh(self, **kwargs):
        unknown = set(kwargs) - _launch_mesh_keywords()
        if unknown:
            raise TypeError(f"launch_mesh() got unexpected keywords {unknown}")
        self.launches.append(kwargs)


def test_mesh_only_launches_with_keywords_launch_mesh_takes():
    runner = _Runner()
    atomesh_server.launch_atomesh(runner, ["mesh-only", "--port", "8000"])
    assert len(runner.launches) == 1


def test_standalone_launches_with_keywords_launch_mesh_takes():
    """The engine spec reaches `launch_mesh`, and the Cores are reaped after."""
    runner = _Runner()
    engine = mock.Mock()
    args = atomesh_server.StandaloneArgs(
        engine_args=SimpleNamespace(model="m", served_model_name=None),
        mesh_args=[],
        default_chat_template_kwargs={},
    )

    def initialize_engine(_args):
        atomesh_server.engine = engine

    with (
        mock.patch.object(atomesh_server, "initialize_engine", initialize_engine),
        mock.patch.object(atomesh_server, "build_engine_spec", return_value="spec"),
        mock.patch.object(atomesh_server, "freeze_gc_heap"),
    ):
        atomesh_server.serve_standalone(runner, args, None, [], None, None)
    assert runner.launches == [
        {"server_config": runner.launches[0]["server_config"], "engine_spec": "spec"}
    ]
    engine.close.assert_called_once_with()
