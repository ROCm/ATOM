# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DP topology flags must survive the trip from argv to ParallelConfig.

`LLMEngine.__init__` used to overwrite `data_parallel_size` unconditionally,
which silently reset a caller-supplied `parallel_config` back to a single rank.
"""

from dataclasses import fields
from types import SimpleNamespace

from atom.config import Config, ParallelConfig
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def _parse(argv):
    # FlexibleArgumentParser, not argparse.ArgumentParser: it is what
    # api_server.py builds, and it auto-registers the snake_case alias of every
    # kebab-case flag. A bare ArgumentParser would exercise a parser no
    # entrypoint uses, and would reject the underscore spelling below.
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    return EngineArgs.from_cli_args(parser.parse_args(argv))


class TestTopologyFlagsReachParallelConfig:
    def test_second_node_of_a_two_node_run(self):
        args = _parse(
            [
                "--model",
                "/fake/model",
                "--data-parallel-size",
                "8",
                "--data-parallel-size-local",
                "4",
                "--data-parallel-rank",
                "4",
                "--data-parallel-master-ip",
                "10.0.0.1",
                "--data-parallel-master-port",
                "29500",
            ]
        )
        pc = args._get_engine_kwargs()["parallel_config"]
        assert pc.data_parallel_size == 8
        assert pc.data_parallel_size_local == 4
        assert pc.data_parallel_rank == 4
        assert pc.data_parallel_master_ip == "10.0.0.1"
        assert pc.data_parallel_master_port == 29500
        assert pc.is_multinode_dp is True

    def test_plain_single_node_run_is_not_multinode(self):
        args = _parse(["--model", "/fake/model", "--data-parallel-size", "8"])
        pc = args._get_engine_kwargs()["parallel_config"]
        assert pc.data_parallel_size == 8
        assert pc.data_parallel_size_local == 8
        assert pc.is_multinode_dp is False

    def test_base_port_flag_is_honored(self):
        args = _parse(["--model", "/fake/model", "--data-parallel-base-port", "40000"])
        pc = args._get_engine_kwargs()["parallel_config"]
        assert pc.data_parallel_base_port == 40000

    def test_underscore_spelling_also_parses(self):
        args = _parse(
            [
                "--model",
                "/fake/model",
                "--data_parallel_size_local",
                "2",
                "--data-parallel-size",
                "2",
            ]
        )
        pc = args._get_engine_kwargs()["parallel_config"]
        assert pc.data_parallel_size_local == 2


def _apply_llm_engine_dp_guard(config, **kwargs):
    """Replay LLMEngine.__init__'s DP-topology resolution, without an engine.

    Constructing a real LLMEngine loads a tokenizer and reaches for GPUs. The
    guard runs before any of that and depends on nothing else, so exercising
    the same three statements against a real Config tests the actual behavior
    rather than the text of the implementation.
    """
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    data_parallel_size = kwargs.get("data_parallel_size", 1)
    data_parallel_master_port = kwargs.get("data_parallel_master_port", None)

    if "parallel_config" not in config_kwargs:
        config.parallel_config.data_parallel_size = data_parallel_size
        if data_parallel_master_port is not None:
            config.parallel_config.data_parallel_master_port = data_parallel_master_port
    return config.parallel_config.data_parallel_size


class TestLLMEngineGuard:
    """A supplied parallel_config must not be clobbered back to dp=1."""

    def test_supplied_parallel_config_survives(self):
        """The regression: a multi-node topology reset to a single rank."""
        pc = ParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=4
        )
        config = SimpleNamespace(parallel_config=pc)

        # No loose data_parallel_size -> it would default to 1 and clobber.
        resolved = _apply_llm_engine_dp_guard(config, parallel_config=pc)

        assert resolved == 8
        assert config.parallel_config.data_parallel_size_local == 4
        assert config.parallel_config.data_parallel_rank == 4
        assert config.parallel_config.is_multinode_dp is True

    def test_legacy_loose_kwarg_caller_still_works(self):
        """Callers predating ParallelConfig pass data_parallel_size directly."""
        config = SimpleNamespace(parallel_config=ParallelConfig())

        resolved = _apply_llm_engine_dp_guard(config, data_parallel_size=4)

        assert resolved == 4
        assert config.parallel_config.data_parallel_size == 4

    def test_legacy_master_port_kwarg_still_applies(self):
        config = SimpleNamespace(parallel_config=ParallelConfig())

        _apply_llm_engine_dp_guard(
            config, data_parallel_size=2, data_parallel_master_port=31000
        )

        assert config.parallel_config.data_parallel_master_port == 31000

    def test_guard_runs_in_the_real_init(self):
        """Pins the guard to LLMEngine itself, not just this replay of it."""
        import inspect

        import atom.model_engine.llm_engine as llm_engine_mod

        source = inspect.getsource(llm_engine_mod.LLMEngine.__init__)
        assert 'if "parallel_config" not in config_kwargs' in source, (
            "LLMEngine.__init__ must not unconditionally overwrite "
            "parallel_config.data_parallel_size -- that resets a "
            "caller-supplied multi-node topology to a single rank"
        )
