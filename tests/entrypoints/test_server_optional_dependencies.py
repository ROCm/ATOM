# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Server startup and tool parsing do not require structured-output extras."""

import subprocess
import sys
from pathlib import Path


def test_server_import_without_structured_output_extras():
    # A fresh process also catches imports hidden by another test's module cache.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class MissingExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jsonschema', 'referencing', 'llguidance', 're2'}:
            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)

sys.meta_path.insert(0, MissingExtras())

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

grammar = KimiK3Parser.tool_call_grammar(
    [{'name': 'ping', 'parameters': {'type': 'object', 'properties': {}}}],
    required=True,
    starts_in_reasoning=False,
)
assert grammar and 'ping' in grammar
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
