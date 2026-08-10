import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from atom.plugin.vllm.cudagraph_memory_profiler_patch import (
    apply_vllm_cudagraph_memory_profiler_patch,
)
from atom.plugin.vllm.deepseek_v4_prefix_patch import (
    _drop_swa_warmup_blocks,
    _kv_cache_config_has_v4_proxy,
    _kv_cache_config_needs_non_immediate_reuse,
    _mark_v4_proxy_cache_mode,
)
from atom.plugin.vllm.spec_decode_patch import _make_atom_compatible_eagle3_type

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeKVCacheManager:
    @staticmethod
    def create_kv_cache_blocks(groups):
        return SimpleNamespace(blocks=groups)


def test_v4_prefix_cache_drop_preserves_vllm_026_boundary():
    manager = _FakeKVCacheManager()
    computed_blocks = SimpleNamespace(
        blocks=(
            (0, 1, 2, 3),
            (4, 5, 6, 7),
        )
    )

    new_blocks, num_tokens, shared_prefix_boundary = _drop_swa_warmup_blocks(
        manager,
        computed_blocks,
        1024,
        384,
        warmup_blocks=2,
        block_size=128,
    )

    assert new_blocks.blocks == ([0, 1], [4, 5])
    assert num_tokens == 768
    assert shared_prefix_boundary == 384


def test_v4_prefix_cache_empty_hit_keeps_vllm_026_result():
    manager = _FakeKVCacheManager()
    computed_blocks = SimpleNamespace(blocks=((),))

    result = _drop_swa_warmup_blocks(
        manager,
        computed_blocks,
        0,
        0,
        warmup_blocks=2,
        block_size=128,
    )

    assert result == (computed_blocks, 0, 0)


def test_v4_profile_cache_mode_marks_only_proxy_layers():
    proxy = SimpleNamespace(
        _atom_v4_proxy_layer=True,
        _atom_v4_profiling_kv_cache=False,
    )
    unrelated = SimpleNamespace(_atom_v4_profiling_kv_cache=False)
    context = {"proxy": proxy, "unrelated": unrelated}

    _mark_v4_proxy_cache_mode(context, True)

    assert proxy._atom_v4_profiling_kv_cache is True
    assert unrelated._atom_v4_profiling_kv_cache is False

    _mark_v4_proxy_cache_mode(context, False)
    assert proxy._atom_v4_profiling_kv_cache is False


def test_v4_proxy_kv_cache_detection_ignores_unrelated_layers():
    unrelated = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["model.layers.0.attn"])],
        has_mamba_layers=False,
    )
    proxy = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.atom_deepseek_v4_proxy"]
            )
        ],
        has_mamba_layers=False,
    )
    assert not _kv_cache_config_has_v4_proxy(unrelated)
    assert _kv_cache_config_has_v4_proxy(proxy)
    assert not _kv_cache_config_needs_non_immediate_reuse(unrelated)
    assert _kv_cache_config_needs_non_immediate_reuse(proxy)

    mamba = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["model.layers.0.linear_attn"])],
        has_mamba_layers=True,
    )
    assert _kv_cache_config_needs_non_immediate_reuse(mamba)


def test_eagle3_type_proxy_is_valid_isinstance_guard():
    class NativeEagle3:
        pass

    class AtomModelBase:
        pass

    proxy = _make_atom_compatible_eagle3_type(NativeEagle3, AtomModelBase)
    guard = (proxy, str)

    assert isinstance(proxy, type)
    assert isinstance(NativeEagle3(), guard)
    assert isinstance(AtomModelBase(), guard)
    assert not isinstance(object(), guard)


def test_disabled_cudagraph_memory_profiler_skips_temporary_capture(monkeypatch):
    vllm_envs = importlib.import_module("vllm.envs")

    calls = []

    class FakeGPUModelRunner:
        def profile_cudagraph_memory(self, marker=None):
            calls.append(marker)
            return 17

    fake_module = SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        fake_module,
    )
    monkeypatch.setattr(
        vllm_envs,
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        False,
    )

    apply_vllm_cudagraph_memory_profiler_patch()
    patched = FakeGPUModelRunner.profile_cudagraph_memory
    apply_vllm_cudagraph_memory_profiler_patch()

    runner = FakeGPUModelRunner()
    assert FakeGPUModelRunner.profile_cudagraph_memory is patched
    assert runner.profile_cudagraph_memory("disabled") == 0
    assert calls == []

    monkeypatch.setattr(
        vllm_envs,
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        True,
    )
    assert runner.profile_cudagraph_memory("enabled") == 17
    assert calls == ["enabled"]


def test_stateful_nightly_cases_use_root_cudagraph_fix():
    catalog = json.loads(
        (REPO_ROOT / ".github/benchmark/oot_models_accuracy.json").read_text()
    )
    target_models = {
        "MiniMax-M2.5 TP2",
        "MiniMax-M2.5 TP4",
        "GLM-4.7-FP8 MTP TP4",
        "GLM-4.7-FP8 MTP TP8",
    }
    entries = {entry["model_name"]: entry for entry in catalog}

    for model_name in target_models:
        entry = entries[model_name]
        assert "client_command" not in entry
        assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in entry["env_vars"]
        assert "PIECEWISE" not in entry["extra_args"]
