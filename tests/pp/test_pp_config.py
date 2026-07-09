# SPDX-License-Identifier: MIT
# CPP A·P0 — EngineArgs pipeline_parallel_size plumbing (GPU-free).

import argparse
import sys
from unittest.mock import MagicMock

_atom_stub = sys.modules.get("atom")
if _atom_stub is not None and not hasattr(_atom_stub, "LLMEngine"):
    _atom_stub.LLMEngine = MagicMock()

_atom_config_stub = sys.modules.get("atom.config")
if _atom_config_stub is not None:
    if not hasattr(_atom_config_stub, "CompilationConfig"):
        _atom_config_stub.CompilationConfig = MagicMock(
            side_effect=lambda **kw: MagicMock(**kw)
        )
    if not hasattr(_atom_config_stub, "SpeculativeConfig"):
        _atom_config_stub.SpeculativeConfig = MagicMock(
            side_effect=lambda **kw: MagicMock(**kw)
        )

from atom.model_engine.arg_utils import EngineArgs  # noqa: E402


def test_default_pp_is_one():
    assert EngineArgs().pipeline_parallel_size == 1


def test_pp_passthrough_to_engine_kwargs():
    kwargs = EngineArgs(pipeline_parallel_size=4)._get_engine_kwargs()
    assert kwargs["pipeline_parallel_size"] == 4


def test_pp_default_in_engine_kwargs():
    kwargs = EngineArgs()._get_engine_kwargs()
    assert kwargs["pipeline_parallel_size"] == 1


def test_cli_parses_pp_short_and_long():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)

    ns = parser.parse_args(["-pp", "4"])
    assert ns.pipeline_parallel_size == 4

    ns = parser.parse_args(["--pipeline-parallel-size", "2"])
    assert ns.pipeline_parallel_size == 2

    ns = parser.parse_args([])
    assert ns.pipeline_parallel_size == 1


def test_from_cli_args_roundtrip():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    ns = parser.parse_args(["-pp", "8", "-tp", "8"])
    args = EngineArgs.from_cli_args(ns)
    assert args.pipeline_parallel_size == 8
    assert args.tensor_parallel_size == 8
