# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_rl_dependencies_belong_to_atomrl_distribution() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        atom_dependencies = tomllib.load(stream)["project"]["dependencies"]
    with (REPO_ROOT / "atomrl" / "pyproject.toml").open("rb") as stream:
        atomrl_dependencies = tomllib.load(stream)["project"]["dependencies"]

    assert not any(dep.startswith(("ray", "omegaconf")) for dep in atom_dependencies)
    assert "ray[default]>=2.9" in atomrl_dependencies
    assert "omegaconf>=2.3,<2.4" in atomrl_dependencies
