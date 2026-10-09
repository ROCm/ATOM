import json
import sys
from pathlib import Path
from types import SimpleNamespace

from atom.plugin.vllm.cudagraph_memory_profiler_patch import (
    apply_vllm_cudagraph_memory_profiler_patch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rocm_skips_temporary_cudagraph_capture(monkeypatch):
    calls = []

    class FakePlatform:
        is_rocm_platform = True

        def is_rocm(self):
            return self.is_rocm_platform

    class FakeGPUModelRunner:
        def profile_cudagraph_memory(self, marker=None):
            calls.append(marker)
            return 17

    platform = FakePlatform()
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=platform),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        SimpleNamespace(GPUModelRunner=FakeGPUModelRunner),
    )

    apply_vllm_cudagraph_memory_profiler_patch()
    patched = FakeGPUModelRunner.profile_cudagraph_memory
    apply_vllm_cudagraph_memory_profiler_patch()

    runner = FakeGPUModelRunner()
    assert FakeGPUModelRunner.profile_cudagraph_memory is patched
    assert runner.profile_cudagraph_memory("rocm") == 0
    assert calls == []

    platform.is_rocm_platform = False
    assert runner.profile_cudagraph_memory("cuda") == 17
    assert calls == ["cuda"]


def test_affected_nightly_cases_restore_default_concurrency():
    catalog = json.loads(
        (REPO_ROOT / ".github/benchmark/oot_models_accuracy.json").read_text()
    )
    target_models = {
        "Qwen3-Next-80B-A3B-Instruct-FP8-MTP TP4",
        "MiniMax-M2.5 TP2",
        "MiniMax-M2.5 TP4",
        "GLM-4.7-FP8 MTP TP4",
        "GLM-4.7-FP8 MTP TP8",
    }
    entries = {entry["model_name"]: entry for entry in catalog}

    for model_name in target_models:
        entry = entries[model_name]
        assert "client_command" not in entry
        assert "PIECEWISE" not in entry["extra_args"]
