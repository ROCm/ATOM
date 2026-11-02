# SPDX-License-Identifier: MIT

import argparse
import os
from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

import atom.config as config_module
import atom.model_engine.llm_engine as engine_module
from atom.model_engine.arg_utils import EngineArgs

UNSET = "<unset>"


@pytest.fixture(autouse=True)
def _cpu_handoff(monkeypatch):
    """Drive the real CLI -> EngineArgs -> LLMEngine -> Config hand-off on CPU.

    Only the parts that need a checkpoint, a HuggingFace reach or GPU workers
    are replaced: Config's checkpoint and quantization lookups, and the
    engine's tokenizer and CoreManager. Argument parsing,
    ``_get_engine_kwargs``, ``LLMEngine.__init__`` and Config construction run
    for real.
    """
    monkeypatch.delenv("ATOM_TORCH_PROFILER_DIR", raising=False)
    monkeypatch.delenv("ATOM_KV_OFFLOAD", raising=False)
    monkeypatch.delenv("ATOM_KV_OFFLOAD_EXTRA_CONFIG", raising=False)
    hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"], max_position_embeddings=8192
    )
    monkeypatch.setattr(
        config_module,
        "get_hf_config",
        lambda _model, trust_remote_code=False: hf_config,
    )
    monkeypatch.setattr(config_module, "get_generation_config", lambda _model: None)
    monkeypatch.setattr(
        config_module,
        "QuantizationConfig",
        lambda *_args, **_kwargs: SimpleNamespace(exclude_layers=[]),
    )
    monkeypatch.setattr(
        engine_module,
        "load_tokenizer",
        lambda *_args, **_kwargs: SimpleNamespace(bos_token_id=0, eos_token_id=1),
    )
    monkeypatch.setattr(
        engine_module,
        "CoreManager",
        lambda _config: SimpleNamespace(close=lambda: None),
    )


def _create_engine(argv):
    """The hand-off every server and example uses: CLI -> EngineArgs -> LLMEngine."""
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(argv)).create_engine()


@pytest.mark.parametrize(
    "env_dir, cli_dir, expected",
    [
        (UNSET, None, None),
        (UNSET, "cli", "cli"),
        ("env", None, "env"),
        ("env", "cli", "cli"),
        ("", None, None),
        ("", "cli", "cli"),
    ],
)
def test_torch_profiler_dir_handoff(monkeypatch, tmp_path, env_dir, cli_dir, expected):
    if env_dir != UNSET:
        monkeypatch.setenv(
            "ATOM_TORCH_PROFILER_DIR", "" if env_dir == "" else str(tmp_path / env_dir)
        )
    argv = ["--model", "test-model"]
    if cli_dir is not None:
        argv += ["--torch-profiler-dir", str(tmp_path / cli_dir)]

    engine = _create_engine(argv)

    expected_dir = None if expected is None else str(tmp_path / expected)
    assert engine.config.torch_profiler_dir == expected_dir
    # Config creates and validates the directory it was handed.
    assert expected_dir is None or os.path.isdir(expected_dir)
    engine.close()


def test_empty_cli_directory_is_not_treated_as_none(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", str(tmp_path / "env"))
    # An explicitly empty CLI value still reaches Config's validation; it does
    # not silently select the environment default.
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(["--torch-profiler-dir", ""]))
    assert args._get_engine_kwargs()["torch_profiler_dir"] == ""
