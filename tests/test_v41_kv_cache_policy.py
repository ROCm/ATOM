# SPDX-License-Identifier: MIT
"""Serving dtype, pool sizing and worker identity agree before allocation."""

import ast
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from atom import config as config_module
from atom.config import CompilationConfig, Config
from atom.models.deepseek_v41.pool_geometry import build_v41_pool_geometry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def chip(monkeypatch):
    chip = ModuleType("aiter.jit.utils.chip_info")
    chip.get_gfx = Mock(side_effect=AssertionError("build target queried"))
    chip.get_gfx_runtime = Mock(return_value="gfx950")
    monkeypatch.setitem(sys.modules, chip.__name__, chip)
    monkeypatch.setenv("GPU_ARCHS", "gfx950;gfx1250")
    monkeypatch.setattr(config_module, "get_generation_config", lambda _: None)
    monkeypatch.setattr(config_module, "QuantizationConfig", lambda *a, **kw: None)
    return chip


def config(heads, tp, dp, dtype, **kwargs):
    return Config(
        model=str(ROOT / "tests/models/deepseek_v41/fixtures"),
        hf_overrides={"num_attention_heads": heads},
        tensor_parallel_size=tp,
        enable_dp_attention=dp,
        enable_expert_parallel=True,
        kv_cache_dtype=dtype,
        index_cache_dtype="fp8",
        enforce_eager=True,
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **kwargs,
    )


def geometry(cfg):
    return build_v41_pool_geometry(
        cfg.hf_config,
        cfg.kv_cache_block_size,
        kv_dtype=cfg.kv_cache_dtype,
        speculative_tokens=0,
        index_fp4=cfg.index_cache_dtype == "fp4",
        index_layout=cfg.index_cache_layout,
    )


@pytest.mark.parametrize(
    "arch,heads,tp,dp,warns",
    [
        ("gfx950", 64, 4, False, False),
        ("gfx950", 64, 2, False, False),
        ("gfx950", 64, 1, False, True),
        ("gfx950", 64, 4, True, True),
        ("gfx950", 128, 1, False, True),
        ("gfx950", 128, 2, False, True),
        ("gfx950", 128, 4, False, False),
        ("gfx950", 128, 4, True, True),
        ("gfx950", 512, 4, False, True),
        ("gfx1250", 64, 1, False, False),
        ("gfx1250", 128, 4, True, False),
        ("gfx942", 64, 4, True, False),
    ],
)
@pytest.mark.parametrize("a8w4_mode", ["auto", "off"])
def test_config_preserves_fp4_sizing_and_reports_performance(
    monkeypatch, chip, caplog, arch, heads, tp, dp, warns, a8w4_mode
):
    chip.get_gfx_runtime.return_value = arch
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", a8w4_mode)
    cfg = config(heads, tp, dp, "fp4")
    assert cfg.kv_cache_dtype == "fp4"
    assert cfg.index_cache_dtype == "fp8"
    assert cfg.dspark.draft_kv_dtype == "auto"
    fp8 = config(heads, tp, dp, "fp8")
    assert geometry(cfg).paged_bytes < geometry(fp8).paged_bytes
    assert cfg.compute_hash() != fp8.compute_hash()
    message = "Keeping the requested FP4 KV cache and FP4 cache sizing."
    assert (message in caplog.text) == warns
    if warns:
        assert "--kv_cache_dtype fp8" in caplog.text
    chip.get_gfx.assert_not_called()
    # Worker startup must not reinterpret the already sized allocation.
    worker = pickle.loads(pickle.dumps(cfg))
    worker._check_kv_cache_dtype()
    assert worker.kv_cache_dtype == "fp4"
    assert geometry(worker) == geometry(cfg)
    assert worker.compute_hash() == cfg.compute_hash()


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_explicit_non_fp4_does_not_probe_architecture(chip, dtype):
    chip.get_gfx_runtime.side_effect = AssertionError("unneeded device probe")
    cfg = config(128, 4, True, dtype)
    assert cfg.kv_cache_dtype == dtype


def bridge_functions():
    path = ROOT / "atom/plugin/vllm/deepseek_v41_bridge.py"
    names = {"v41_kv_cache_dtype", "v41_proxy_geometry", "_build_v41_pool_geometry"}
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    scope = {"ATOM_DEEPSEEK_V41_BLOCK_SIZE": 256, "v41_speculative_tokens": lambda _: 0}
    # Execute only the selected repository functions without importing vLLM.
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    exec(code, scope)  # noqa: S102
    return scope


@pytest.mark.parametrize(
    "heads,tp",
    [(64, 1), (128, 1), (128, 2), (64, 4)],
)
def test_vllm_proxy_and_native_preserve_the_same_fp4_pool(chip, heads, tp):
    native = config(heads, tp, False, "fp4")
    api = bridge_functions()
    api["v41_text_config"] = lambda _: native.hf_config
    plugin = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="nvfp4"),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=4),
    )
    chip.get_gfx_runtime.side_effect = AssertionError("proxy dtype probed the GPU")
    assert api["v41_kv_cache_dtype"](plugin) == "fp4"
    # Compare real layouts to detect a wider allocation than the user requested.
    native.kv_cache_block_size = 256
    actual = api["v41_proxy_geometry"](plugin)
    assert actual == geometry(native)
    assert api["v41_kv_cache_dtype"](plugin) == "fp4"
    assert api["v41_proxy_geometry"](plugin) is actual


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("auto", "bf16"),
        ("bfloat16", "bf16"),
        ("float16", "bf16"),
        ("nvfp4", "fp4"),
        ("fp8", "fp8"),
        ("fp8_e4m3", "fp8"),
        ("fp8_e5m2", None),
    ],
)
def test_vllm_dtype_spellings(spelling, expected):
    resolve = bridge_functions()["v41_kv_cache_dtype"]
    cfg = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=spelling))
    if expected is None:
        with pytest.raises(ValueError, match="kv-cache-dtype"):
            resolve(cfg)
    else:
        assert resolve(cfg) == expected
