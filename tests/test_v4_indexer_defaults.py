# SPDX-License-Identifier: MIT

import sys
from types import ModuleType, SimpleNamespace

import pytest
from transformers import PretrainedConfig

import atom.config as config_module
from atom.config import CompilationConfig, Config


class _GfxProbe:
    def __init__(self, gfx: str):
        self.gfx = gfx
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.gfx


def _install_chip_info_stub(monkeypatch, get_gfx) -> None:
    aiter = ModuleType("aiter")
    aiter.__path__ = []
    jit = ModuleType("aiter.jit")
    jit.__path__ = []
    utils = ModuleType("aiter.jit.utils")
    utils.__path__ = []
    chip_info = ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = get_gfx

    aiter.jit = jit
    jit.utils = utils
    utils.chip_info = chip_info
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.jit", jit)
    monkeypatch.setitem(sys.modules, "aiter.jit.utils", utils)
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", chip_info)


def _make_config(
    monkeypatch,
    *,
    gfx: str = "gfx950",
    architectures: list[str] | None = None,
    **overrides,
):
    hf_config = PretrainedConfig(
        architectures=(
            ["DeepseekV4ForCausalLM"] if architectures is None else architectures
        ),
        max_position_embeddings=8192,
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
        config_module,
        "is_plugin_mode",
        lambda: overrides.get("plugin_config") is not None,
    )

    probe = _GfxProbe(gfx)
    _install_chip_info_stub(monkeypatch, probe)
    config = Config(
        model="test-model",
        kv_cache_dtype="fp8",
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **overrides,
    )
    return config, probe


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250", "future_gfx"])
def test_native_v4_defaults_indexer_to_fp4(monkeypatch, gfx):
    config, probe = _make_config(monkeypatch, gfx=gfx)

    assert config.index_cache_dtype == "fp4"
    assert config.kv_cache_block_size == 256
    assert probe.calls == 1


def test_native_v4_defaults_indexer_to_fp8_on_gfx942(monkeypatch):
    config, probe = _make_config(monkeypatch, gfx="gfx942")

    assert config.index_cache_dtype == "fp8"
    assert probe.calls == 1


@pytest.mark.parametrize("integration", ["plugin", "transfer"])
def test_v4_integrations_without_fp4_layout_default_to_fp8(monkeypatch, integration):
    overrides = {}
    if integration == "plugin":
        overrides["plugin_config"] = SimpleNamespace(
            is_vllm=False,
            vllm_config=None,
        )
    else:
        overrides["kv_transfer_config"] = {"kv_role": "producer"}

    config, probe = _make_config(monkeypatch, **overrides)

    assert config.index_cache_dtype == "fp8"
    assert probe.calls == 0


@pytest.mark.parametrize("requested", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("gfx", ["gfx942", "gfx950"])
def test_v4_preserves_explicit_indexer_dtype(monkeypatch, requested, gfx):
    config, probe = _make_config(
        monkeypatch,
        gfx=gfx,
        index_cache_dtype=requested,
    )

    assert config.index_cache_dtype == requested
    assert probe.calls == 0


def test_non_v4_indexer_defaults_to_kv_cache_dtype(monkeypatch):
    config, probe = _make_config(
        monkeypatch,
        architectures=["LlamaForCausalLM"],
    )

    assert config.index_cache_dtype == config.kv_cache_dtype == "fp8"
    assert probe.calls == 0


@pytest.mark.parametrize(
    ("gfx", "expected"),
    [("gfx942", False), ("gfx950", True), ("gfx1250", True)],
)
def test_fp4_indexer_runtime_predicate(monkeypatch, caplog, gfx, expected):
    v4_kernels = pytest.importorskip("atom.model_ops.v4_kernels")
    monkeypatch.setattr(v4_kernels, "get_gfx", lambda: gfx)

    with caplog.at_level("WARNING", logger="atom"):
        enabled = v4_kernels.fp4_indexer_enabled("fp4", warn=True)

    assert enabled is expected
    assert ("unsupported" in caplog.text) is (not expected)


def test_non_fp4_indexer_predicate_does_not_query_arch(monkeypatch):
    v4_kernels = pytest.importorskip("atom.model_ops.v4_kernels")

    def fail_if_called():
        raise AssertionError("non-FP4 indexers must not query the GPU architecture")

    monkeypatch.setattr(v4_kernels, "get_gfx", fail_if_called)

    assert not v4_kernels.fp4_indexer_enabled("fp8")
