# SPDX-License-Identifier: MIT
"""The plugin's V4.1 index plane is the one the runtime resolves.

Both sides build the pool from `build_v41_pool_geometry`, but each passes its
own `index_fp4`: core from `Config.index_cache_dtype`, the plugin a constant,
because its config translation forwards no `--index-cache-dtype` and the ATOM
Config then takes V4.1's default. If that default changes, or the plugin starts
forwarding the flag, vLLM allocates the proxy pool for one plane format while
the kernels address the other: pool corruption, with nothing raising.

Reads the plugin side from source and builds the core side from a real Config,
so it imports no vLLM and CPU CI collects it.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "atom/plugin/vllm/deepseek_v41_bridge.py"
PLUGIN_CONFIG = ROOT / "atom/plugin/config.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(fn: ast.FunctionDef, callee: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == callee
    ]


def _plugin_index_fp4() -> bool:
    (call,) = _calls(
        _function(BRIDGE, "v41_proxy_geometry"), "_build_v41_pool_geometry"
    )
    value = next(kw.value for kw in call.keywords if kw.arg == "index_fp4")
    assert isinstance(value, ast.Constant), (
        "the plugin no longer passes a constant index_fp4; compare it with "
        "Config.index_cache_dtype here instead"
    )
    return bool(value.value)


def test_plugin_forwards_no_index_cache_dtype():
    (call,) = _calls(
        _function(PLUGIN_CONFIG, "_generate_atom_config_from_vllm_config"), "Config"
    )
    assert "index_cache_dtype" not in {kw.arg for kw in call.keywords}, (
        "the plugin forwards --index-cache-dtype now; the bridge's index_fp4 "
        "must follow it"
    )


def test_plugin_index_plane_matches_the_resolved_default(monkeypatch):
    from atom import config as config_module
    from atom.config import Config

    fixture = ROOT / "tests/models/deepseek_v41/fixtures"
    # The quantization parser resolves AITER enums; only that GPU dependency
    # is bypassed, and the real default resolution runs.
    monkeypatch.setattr(
        config_module, "QuantizationConfig", lambda *args, **kwargs: None
    )
    # As the plugin builds it for V4.1: no index_cache_dtype, no transport.
    cfg = Config(
        model=str(fixture),
        kv_cache_dtype="bf16",
        kv_transfer_config={},
        enforce_eager=True,
    )
    assert (cfg.index_cache_dtype == "fp4") == _plugin_index_fp4()
