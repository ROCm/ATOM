# SPDX-License-Identifier: MIT

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from transformers import PretrainedConfig

import atom.config as config_module
from atom.config import CompilationConfig, Config

_V4_KERNELS_INIT = (
    Path(__file__).resolve().parents[1] / "atom/model_ops/v4_kernels/__init__.py"
)


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
    chip_info.get_gfx_runtime = get_gfx

    aiter.jit = jit
    jit.utils = utils
    utils.chip_info = chip_info
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.jit", jit)
    monkeypatch.setitem(sys.modules, "aiter.jit.utils", utils)
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", chip_info)


def _load_v4_kernels(monkeypatch, gfx: str):
    import atom.model_ops as model_ops_package

    probe = _GfxProbe(gfx)
    _install_chip_info_stub(monkeypatch, probe)

    tree = ast.parse(
        _V4_KERNELS_INIT.read_text(encoding="utf-8"),
        filename=str(_V4_KERNELS_INIT),
    )
    package_prefix = "atom.model_ops.v4_kernels."
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = node.module or ""
        if not module_name.startswith(package_prefix):
            continue
        stub = ModuleType(module_name)
        for imported in node.names:
            setattr(stub, imported.name, object())
        monkeypatch.setitem(sys.modules, module_name, stub)

    module_name = "atom.model_ops.v4_kernels"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _V4_KERNELS_INIT,
        submodule_search_locations=[str(_V4_KERNELS_INIT.parent)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(model_ops_package, "v4_kernels", module, raising=False)
    spec.loader.exec_module(module)
    return module, probe


def _make_config(
    monkeypatch,
    *,
    gfx: str = "gfx950",
    architectures: list[str] | None = None,
    hf: dict | None = None,
    **overrides,
):
    hf_config = PretrainedConfig(
        architectures=(
            ["DeepseekV4ForCausalLM"] if architectures is None else architectures
        ),
        max_position_embeddings=8192,
        sparse_attention_config={"sparse_local_block": 1, "sparse_block_size": 128},
        **(hf or {}),
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
        lambda *_args, **_kwargs: SimpleNamespace(
            exclude_layers=[], compute_hash=lambda: ""
        ),
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
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **{"kv_cache_dtype": "fp8", **overrides},
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
    v4_kernels, probe = _load_v4_kernels(monkeypatch, gfx)

    with caplog.at_level("WARNING", logger="atom"):
        enabled = v4_kernels.fp4_indexer_enabled("fp4", warn=True)

    assert enabled is expected
    assert probe.calls == 1
    assert ("unsupported" in caplog.text) is (not expected)


def test_non_fp4_indexer_predicate_does_not_query_arch(monkeypatch):
    v4_kernels, probe = _load_v4_kernels(monkeypatch, "gfx950")

    assert not v4_kernels.fp4_indexer_enabled("fp8")
    assert probe.calls == 0


M3 = ["MiniMaxM3SparseForConditionalGeneration"]


def _m3(monkeypatch, **overrides):
    return _make_config(
        monkeypatch, architectures=M3, kv_cache_block_size=128, **overrides
    )[0]


@pytest.mark.parametrize(
    ("kv", "index", "want"),
    [
        # KV-only: the index default follows the fp8 storage dtype.
        ("fp4", None, ("fp8", True, "fp8")),
        # index-only: KV stays as asked.
        ("bf16", "fp4", ("bf16", False, "fp4x2")),
        ("fp4", "fp4", ("fp8", True, "fp4x2")),
        ("fp4", "fp8", ("fp8", True, "fp8")),
        ("fp8", None, ("fp8", False, "fp8")),
    ],
)
def test_m3_public_fp4_is_the_internal_fp4x2(monkeypatch, kv, index, want):
    config = _m3(monkeypatch, kv_cache_dtype=kv, index_cache_dtype=index)

    assert (
        config.kv_cache_dtype,
        config.kv_mxfp4,
        config.index_cache_dtype,
    ) == want


def test_m3_fp4_caches_are_normalised_before_the_cp_fallback(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="atom"):
        config = _m3(
            monkeypatch,
            kv_cache_dtype="fp4",
            index_cache_dtype="fp4",
            tensor_parallel_size=4,
            hf={"num_key_value_heads": 4},
            dcp_config=config_module.DCPConfig(indexer_dcp_only=True),
        )

    assert config.dcp_config.indexer_dcp_only is False
    assert "MXFP4 caches are not wired to the CP scorer" in caplog.text


def test_m3_fp4_keeps_the_m3_startup_refusals(monkeypatch):
    with pytest.raises(ValueError, match="MXFP4 pages are 128 tokens"):
        _make_config(
            monkeypatch, architectures=M3, kv_cache_dtype="fp4", kv_cache_block_size=16
        )


def test_m3_fp4_hashes_apart_from_fp8(monkeypatch):
    fp8 = _m3(monkeypatch, kv_cache_dtype="fp8", index_cache_dtype="fp8")
    fp4 = _m3(monkeypatch, kv_cache_dtype="fp4", index_cache_dtype="fp4")
    kv_only = _m3(monkeypatch, kv_cache_dtype="fp4", index_cache_dtype="fp8")

    assert len({fp8.compute_hash(), fp4.compute_hash(), kv_only.compute_hash()}) == 3


def test_v4_keeps_its_fp4_indexer_and_refuses_a_fp4_kv(monkeypatch):
    config, _ = _make_config(monkeypatch, index_cache_dtype="fp4")
    assert (config.kv_mxfp4, config.index_cache_dtype) == (False, "fp4")

    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        _make_config(monkeypatch, kv_cache_dtype="fp4")


def test_other_models_refuse_a_fp4_kv(monkeypatch):
    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        _make_config(
            monkeypatch, architectures=["Glm5NextForCausalLM"], kv_cache_dtype="fp4"
        )
