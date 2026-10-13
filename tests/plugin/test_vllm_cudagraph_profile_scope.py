"""Who the ROCm cudagraph-profiling skip applies to, and who it must not.

The V1 skip is global on ROCm. Extending that to V2 wholesale is not free:
the wrapper returns 0, which also removes the cudagraph reservation from
vLLM's KV budget, and the M3 / K3 / Qwen3.5 accuracy cells were tuned with
that reservation in place. DeepSeek-V4.1 is the one model that cannot run
without the skip -- its V2 profiling capture happens before the KV cache
exists -- so the V2 side is scoped to it by model, and these tests are what
keep the scope from widening by accident.
"""

import pytest

from atom.plugin.vllm import cudagraph_memory_profiler_patch as patch


class _Runner:
    """Only what the wrapper reads."""

    def __init__(self, arches):
        self.model_config = type("MC", (), {"architectures": tuple(arches)})()


def _unpatched_profile(self, *args, **kwargs):
    """What a real profile reports: a non-zero cudagraph reservation."""
    return 4096


@pytest.fixture
def v2(monkeypatch):
    """A stand-in V2 runner class the patch can be pointed at."""
    v2_mod = pytest.importorskip("vllm.v1.worker.gpu.model_runner")
    original = v2_mod.GPUModelRunner.profile_cudagraph_memory
    yield v2_mod
    v2_mod.GPUModelRunner.profile_cudagraph_memory = original


def test_v41_is_skipped_on_v2(v2, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        v2.GPUModelRunner, "profile_cudagraph_memory", _unpatched_profile
    )
    patch._skip_v2_for_deepseek_v41()
    runner = _Runner(["DeepseekV41ForCausalLM"])
    assert v2.GPUModelRunner.profile_cudagraph_memory(runner) == 0


def test_every_other_v2_model_keeps_its_reservation(v2, monkeypatch):
    """The regression this scope exists to avoid.

    A wrapper that returned 0 for these would silently change the KV budget
    the accuracy cells were tuned against.
    """
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        v2.GPUModelRunner, "profile_cudagraph_memory", _unpatched_profile
    )
    patch._skip_v2_for_deepseek_v41()
    for arch in ("MiniMaxM3ForCausalLM", "KimiK3ForCausalLM", "Qwen3MoeForCausalLM"):
        runner = _Runner([arch])
        assert v2.GPUModelRunner.profile_cudagraph_memory(runner) == 4096, arch


def test_v41_on_cuda_is_not_skipped(v2, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        v2.GPUModelRunner, "profile_cudagraph_memory", _unpatched_profile
    )
    patch._skip_v2_for_deepseek_v41()
    runner = _Runner(["DeepseekV41ForCausalLM"])
    assert v2.GPUModelRunner.profile_cudagraph_memory(runner) == 4096


def test_v41_really_is_outside_the_v1_default_list():
    """Why the V2 side is needed at all.

    If upstream ever adds V4.1 to `ROCM_DEFAULT_MRV1_ARCHITECTURES` it runs V1,
    the global skip covers it, and this whole V2 branch can be deleted. This
    test is what would say so.
    """
    from vllm.config.vllm import ROCM_DEFAULT_MRV1_ARCHITECTURES

    assert "DeepseekV41ForCausalLM" not in ROCM_DEFAULT_MRV1_ARCHITECTURES
    assert "DeepseekV4ForCausalLM" in ROCM_DEFAULT_MRV1_ARCHITECTURES
