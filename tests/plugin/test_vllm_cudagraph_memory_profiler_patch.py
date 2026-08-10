import sys
from types import SimpleNamespace

from atom.plugin.vllm.cudagraph_memory_profiler_patch import (
    apply_vllm_cudagraph_memory_profiler_patch,
)


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
