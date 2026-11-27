# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The router argv `mesh-only` hands the Rust parser."""

from __future__ import annotations

import pytest

from atom.entrypoints.atomesh import server as atomesh_server


class _Runner:
    """Records the argv `parse_from` gets; launching does nothing."""

    def __init__(self):
        self.argv: list[str] | None = None

    def parse_from(self, argv):
        self.argv = list(argv)
        return {
            "cli_args": {
                "host": "0.0.0.0",
                "port": 30000,
                "pd_disaggregation": False,
                "backend": "atom",
                "policy": "cache_aware",
            },
            "prefill_urls": [],
            "decode_urls": [],
            "server_config": object(),
        }

    def launch_mesh(self, server_config, **_kwargs):
        pass


def _router_argv(raw_args: list[str]) -> list[str]:
    runner = _Runner()
    atomesh_server.launch_atomesh(runner, raw_args)
    return runner.argv


WORKERS = ["--worker-urls", "http://w1:8000", "http://w2:8000"]


def test_mesh_only_routes_to_atom_workers_by_default():
    argv = _router_argv(["mesh-only", *WORKERS, "--policy", "cache_aware"])
    assert argv == [*WORKERS, "--policy", "cache_aware", "--backend", "atom"]


def test_the_backend_lands_inside_a_launch_subcommand():
    argv = _router_argv(["mesh-only", "launch", *WORKERS])
    assert argv == ["launch", *WORKERS, "--backend", "atom"]


@pytest.mark.parametrize(
    "backend",
    [
        ["--backend", "sglang"],
        ["--backend=vllm"],
        ["--runtime", "sglang"],
        ["--runtime=atom"],
    ],
)
def test_a_backend_on_the_command_line_is_kept_alone(backend):
    """The command line's backend wins and is not given twice."""
    argv = _router_argv(["mesh-only", *WORKERS, *backend])
    assert argv == [*WORKERS, *backend]
