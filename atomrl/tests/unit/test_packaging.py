# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from pathlib import Path

from setuptools import find_packages

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_miles_adapter_is_not_in_source_or_discovered_packages() -> None:
    assert not list((PROJECT_ROOT / "atomrl" / "plugin" / "miles").glob("*.py"))
    assert "atomrl.plugin.miles" not in find_packages(where=PROJECT_ROOT)
