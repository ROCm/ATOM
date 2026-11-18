"""The atomesh entrypoint against `atomesh_runner.launch_mesh`'s signature.

`launch_mesh` is a PyO3 function (`atom/mesh/src/python.rs`) that takes only
keyword arguments; a keyword it does not declare is a TypeError at startup,
which no Rust test sees. The runner here is a fake that holds the entrypoint
to the keywords the Rust signature declares.
"""

import re
from pathlib import Path
from unittest import mock

import pytest

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


def test_standalone_is_refused_before_it_loads_a_model():
    runner = _Runner()
    with (
        mock.patch.object(atomesh_server, "initialize_engine") as initialize,
        pytest.raises(SystemExit, match="mesh-only"),
    ):
        atomesh_server.launch_atom_standalone(runner, ["--model", "m"])
    initialize.assert_not_called()
    assert runner.launches == []
