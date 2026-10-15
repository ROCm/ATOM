import functools
import logging

logger = logging.getLogger("atom")


def apply_vllm_cudagraph_memory_profiler_patch() -> None:
    """Skip vLLM's temporary CUDA graph capture on ROCm.

    vLLM 0.26 expanded CUDA graph memory profiling to ROCm. The profiling pass
    captures and destroys a temporary copy of every graph before the real
    capture, leaving AITER with stale graph-owned state.

    **This covers model runner V1 only.** vLLM 0.31 makes V2 the ROCm default
    for every architecture outside `ROCM_DEFAULT_MRV1_ARCHITECTURES`
    (`DeepseekV32ForCausalLM`, `DeepseekV4ForCausalLM`, `GlmMoeDsaForCausalLM`),
    and V2's runner (`vllm/v1/worker/gpu/model_runner.py`) is a separate class
    with its own `profile_cudagraph_memory`, not a subclass of the one patched
    here. On 0.31 the plugin's MiniMax-M3, Kimi-K3 and Qwen3.5 accuracy cells
    all run V2, so they do the temporary capture this patch exists to avoid --
    and all three pass, which is why the patch is not simply extended to V2
    here. Doing that is a behaviour change needing its own measured round:
    `wrapped` returns 0, so extending it also removes the cudagraph reservation
    from vLLM's KV budget, and the K3 cell's `--gpu-memory-utilization 0.91`
    was derived with that reservation in place.

    Until then the gap is logged rather than left silent -- except for
    DeepSeek-V4.1, which cannot run at all with it. V4.1 is outside
    `ROCM_DEFAULT_MRV1_ARCHITECTURES`, so it runs V2, and V2's profiling
    capture happens inside `determine_available_memory`, before the KV cache
    exists. The V4.1 proxy bind then meets the profiling stub and refuses
    ("pool is too small: 47,349,760 bytes ... short by 337,707,008"), so the
    engine never starts. `_skip_v2_for_deepseek_v41` covers exactly that model
    and leaves every other V2 model's budget where the accuracy cells measured
    it.
    """
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _warn_if_v2_runner_unpatched()

    original = GPUModelRunner.profile_cudagraph_memory
    if getattr(original, "_atom_skip_rocm_profile", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if current_platform.is_rocm():
            logger.info(
                "ATOM plugin: skipping unsafe temporary CUDA graph memory "
                "capture on ROCm"
            )
            return 0
        return original(self, *args, **kwargs)

    wrapped._atom_skip_rocm_profile = True
    GPUModelRunner.profile_cudagraph_memory = wrapped
    _skip_v2_for_deepseek_v41()


_V41_ARCHES = ("DeepseekV41ForCausalLM",)


def _skip_v2_for_deepseek_v41() -> None:
    """Skip the V2 profiling capture for DeepSeek-V4.1 and nothing else.

    Narrow on purpose. Extending the V1 behaviour to every V2 model would also
    drop the cudagraph reservation from vLLM's KV budget, which the M3, K3 and
    Qwen3.5 accuracy cells were tuned against -- that is the measured round
    the docstring above says is still owed. V4.1 is not a tuning question: the
    capture runs before its KV cache exists and the bind cannot succeed.

    The model is read from the runner at call time rather than from a config
    captured at install time, because this patch is installed once per process
    from `register_model` and the runner is the only thing that knows which
    model it is serving.
    """
    from vllm.platforms import current_platform

    try:
        from vllm.v1.worker.gpu.model_runner import (
            GPUModelRunner as V2GPUModelRunner,
        )
    except ImportError:
        return  # pre-0.31; the V1 patch is the whole story there

    original = V2GPUModelRunner.profile_cudagraph_memory
    if getattr(original, "_atom_skip_rocm_profile_v41", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        arches = getattr(getattr(self, "model_config", None), "architectures", ()) or ()
        if current_platform.is_rocm() and any(a in _V41_ARCHES for a in arches):
            logger.info(
                "ATOM plugin: skipping the V2 CUDA graph memory profiling "
                "capture for %s; it runs before the KV cache exists and the "
                "V4.1 proxy bind cannot meet the profiling stub",
                arches,
            )
            return 0
        return original(self, *args, **kwargs)

    wrapped._atom_skip_rocm_profile_v41 = True
    V2GPUModelRunner.profile_cudagraph_memory = wrapped


def _warn_if_v2_runner_unpatched() -> None:
    """Say so when the runner that will actually run is not the patched one.

    A monkeypatch that lands on a class nobody instantiates is the quietest
    kind of no-op: everything imports, nothing raises, and the behaviour the
    patch was written to prevent simply happens.
    """
    try:
        from vllm.v1.worker.gpu.model_runner import (
            GPUModelRunner as V2GPUModelRunner,
        )
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1GPUModelRunner
    except ImportError:
        return
    if issubclass(V2GPUModelRunner, V1GPUModelRunner):
        return
    logger.info(
        "ATOM plugin: model runner V2 is a separate class from the one this "
        "patch wraps, so its profile_cudagraph_memory is NOT skipped. On vLLM "
        "0.31 V2 is the ROCm default outside ROCM_DEFAULT_MRV1_ARCHITECTURES; "
        "a V2 run therefore still does the temporary CUDA graph capture."
    )
