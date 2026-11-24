# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import pathlib
import re

PYPROJECT = pathlib.Path(__file__).parents[1] / "pyproject.toml"


def _list(key):
    # tomllib is 3.11+; the project supports 3.10.
    match = re.search(rf"^{key} = (\[.*\])$", PYPROJECT.read_text(), re.MULTILINE)
    return None if match is None else json.loads(match[1])


def test_base_install_includes_the_structured_output_backend():
    base = _list("dependencies")
    assert base is not None, "dependencies is not a one-line list; update the parser"
    assert "llguidance==1.9.1" in base
    assert any(dep.startswith("google-re2") for dep in base)
    extra = _list("structured-output")
    assert extra is None or set(extra) <= set(base)
