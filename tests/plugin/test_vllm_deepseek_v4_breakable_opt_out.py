# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's DeepSeek-V4 opts out of vLLM's breakable cudagraph auto-enable.

vLLM sets VLLM_USE_BREAKABLE_CUDAGRAPH=1 in VllmConfig.__post_init__ for the V4
architectures when the variable is unset. The plugin's wrapper must set it to 0
first, only for a V4 architecture the registry resolves to ATOM, and leave every
other model and any explicit user setting alone.
"""

import logging
import os
import sys
from types import SimpleNamespace as NS
from typing import ClassVar

import pytest

from atom.plugin.vllm import deepseek_v4_cudagraph_patch as cg

ENV = cg.BREAKABLE_ENV
ATOM_WRAPPER = NS(module_name="atom.plugin.vllm.model_wrapper", class_name="X")


class _Registry:
    models: ClassVar[dict] = {}


class _FakeVllmConfig:
    """Stands in for VllmConfig: __post_init__ records what vLLM would do."""

    def __init__(self, architectures):
        self.model_config = (
            None if architectures is None else NS(architectures=architectures)
        )
        self.seen_by_vllm = None
        self.__post_init__()

    def __post_init__(self):
        # vLLM's own auto-enable (config/vllm.py) runs after the wrapper.
        self.seen_by_vllm = os.environ.get(ENV)
        if ENV not in os.environ:
            os.environ[ENV] = "1"


@pytest.fixture
def fake_vllm(monkeypatch):
    class Config(_FakeVllmConfig):
        pass

    registry = _Registry()
    registry.models = {
        "DeepseekV4ForCausalLM": ATOM_WRAPPER,
        "DeepSeekV4MTPModel": ATOM_WRAPPER,
        "KimiK3ForConditionalGeneration": NS(
            module_name="atom.plugin.vllm.models.kimi_k3", class_name="K3"
        ),
        "MiniMaxM3SparseForCausalLM": ATOM_WRAPPER,
        "LlamaForCausalLM": ATOM_WRAPPER,
    }
    monkeypatch.setitem(sys.modules, "vllm.config", NS(VllmConfig=Config))
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.registry",
        NS(ModelRegistry=registry),
    )
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("ATOM_DISABLE_VLLM_PLUGIN", raising=False)
    assert cg.apply_vllm_v4_breakable_cudagraph_opt_out()
    yield Config, registry
    os.environ.pop(ENV, None)


@pytest.mark.parametrize("arch", ["DeepseekV4ForCausalLM", "DeepSeekV4MTPModel"])
def test_atom_v4_sets_zero_before_vllm_decides(fake_vllm, arch):
    Config, _ = fake_vllm
    cfg = Config([arch])
    assert cfg.seen_by_vllm == "0"
    assert os.environ[ENV] == "0"


@pytest.mark.parametrize(
    "arch",
    [
        "KimiK3ForConditionalGeneration",
        "MiniMaxM3SparseForCausalLM",
        "LlamaForCausalLM",
        "SomethingVllmOnly",
    ],
)
def test_other_architectures_are_untouched(fake_vllm, arch):
    Config, _ = fake_vllm
    cfg = Config([arch])
    assert cfg.seen_by_vllm is None
    assert os.environ[ENV] == "1"  # vLLM's own auto-enable still applies


def test_no_model_config_is_untouched(fake_vllm):
    Config, _ = fake_vllm
    cfg = Config(None)
    assert cfg.seen_by_vllm is None


def test_v4_not_served_by_atom_is_untouched(fake_vllm):
    Config, registry = fake_vllm
    registry.models["DeepseekV4ForCausalLM"] = NS(
        module_name="vllm.model_executor.models.deepseek_v4", class_name="V4"
    )
    cfg = Config(["DeepseekV4ForCausalLM"])
    assert cfg.seen_by_vllm is None
    assert os.environ[ENV] == "1"


def test_eagerly_registered_atom_class_counts(fake_vllm):
    Config, registry = fake_vllm
    registry.models["DeepseekV4ForCausalLM"] = NS(
        model_cls=type("V4", (), {"__module__": "atom.plugin.vllm.model_wrapper"})
    )
    assert Config(["DeepseekV4ForCausalLM"]).seen_by_vllm == "0"


def test_plugin_disabled_is_untouched(fake_vllm, monkeypatch):
    Config, _ = fake_vllm
    monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "1")
    cfg = Config(["DeepseekV4ForCausalLM"])
    assert cfg.seen_by_vllm is None


def test_explicit_one_is_kept_with_a_warning(fake_vllm, monkeypatch, caplog):
    Config, _ = fake_vllm
    monkeypatch.setenv(ENV, "1")
    with caplog.at_level(logging.WARNING, logger="atom"):
        cfg = Config(["DeepseekV4ForCausalLM"])
    assert cfg.seen_by_vllm == "1"
    assert any(
        "wrong output" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_explicit_zero_is_kept_quietly(fake_vllm, monkeypatch, caplog):
    Config, _ = fake_vllm
    monkeypatch.setenv(ENV, "0")
    with caplog.at_level(logging.INFO, logger="atom"):
        cfg = Config(["DeepseekV4ForCausalLM"])
    assert cfg.seen_by_vllm == "0"
    assert not [r for r in caplog.records if "[atom-v4-cg]" in r.getMessage()]


def test_apply_is_idempotent(fake_vllm):
    Config, _ = fake_vllm
    wrapped = Config.__post_init__
    assert not cg.apply_vllm_v4_breakable_cudagraph_opt_out()
    assert Config.__post_init__ is wrapped


def test_real_vllm_config_honours_the_class_level_wrap(monkeypatch):
    """VllmConfig is a pydantic dataclass; its __post_init__ must still go
    through the wrapper installed on the class."""
    vllm_config_mod = pytest.importorskip("vllm.config")
    VllmConfig = vllm_config_mod.VllmConfig
    seen = []
    monkeypatch.setattr(
        cg, "_opt_out_of_breakable_cudagraph", lambda cfg: seen.append(type(cfg))
    )
    monkeypatch.setattr(VllmConfig, "__post_init__", VllmConfig.__post_init__)
    cg.apply_vllm_v4_breakable_cudagraph_opt_out()
    VllmConfig()
    assert seen == [VllmConfig]
