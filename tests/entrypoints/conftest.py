# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Make ``atom.entrypoints.openai.api_server`` importable for the tests here.

The repo-level ``tests/conftest.py`` replaces the top-level ``atom`` package with
a stub so importing any submodule never boots an engine. ``api_server`` reaches
through that stub twice — ``from atom import SamplingParams`` directly, and
``from atom import LLMEngine`` transitively via
``atom.model_engine.arg_utils`` — so the stub needs those two names before the
import can succeed.

Importing it here (once, as a conftest side effect) means every test module in
this directory can just ``from atom.entrypoints.openai import api_server``:
the module is already in ``sys.modules`` by then.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock

from atom.sampling_params import SamplingParams


def _install_atom_stub_attrs() -> None:
    """Add the attributes ``api_server``'s import chain reads off ``atom``."""
    atom_pkg = sys.modules.get("atom")
    if atom_pkg is None:
        return
    if not hasattr(atom_pkg, "SamplingParams"):
        atom_pkg.SamplingParams = SamplingParams  # real, dependency-free
    if not hasattr(atom_pkg, "LLMEngine"):
        # Only needed so atom.model_engine.arg_utils imports; never instantiated.
        atom_pkg.LLMEngine = MagicMock()


def _stub_unimportable(mod_name: str, attr_name: str, stub_cls) -> bool:
    """Stub ``mod_name`` when the real module cannot be imported here.

    Returns True when a stub was injected, so the caller can remove it again —
    leaving it in ``sys.modules`` would shadow the real module for tests
    collected later (notably ``tests/test_arg_utils_spec.py``).
    """
    if mod_name in sys.modules:
        return False
    try:
        importlib.import_module(mod_name)
    except Exception:
        stub = types.ModuleType(mod_name)
        setattr(stub, attr_name, stub_cls)
        sys.modules[mod_name] = stub
        return True
    return False


class _StubCoreManager:
    def __init__(self, *args, **kwargs):
        pass

    def add_request(self, requests):
        return None


class _StubEngineArgs:
    @classmethod
    def add_cli_args(cls, parser):
        return parser

    @classmethod
    def from_cli_args(cls, args):
        return cls()

    def create_engine(self, tokenizer=None):
        return None


_install_atom_stub_attrs()
_injected = [
    name
    for name, injected in (
        (
            "atom.model_engine.engine_core_mgr",
            _stub_unimportable(
                "atom.model_engine.engine_core_mgr", "CoreManager", _StubCoreManager
            ),
        ),
        (
            "atom.model_engine.arg_utils",
            _stub_unimportable(
                "atom.model_engine.arg_utils", "EngineArgs", _StubEngineArgs
            ),
        ),
    )
    if injected
]
try:
    importlib.import_module("atom.entrypoints.openai.api_server")
except Exception:
    # Left to each test module to skip on; the endpoint tests are optional
    # coverage on environments missing PIL/transformers/uvicorn.
    pass
finally:
    for _name in _injected:
        sys.modules.pop(_name, None)
