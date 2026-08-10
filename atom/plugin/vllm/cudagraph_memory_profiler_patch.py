import functools
import logging

logger = logging.getLogger("atom")


def apply_vllm_cudagraph_memory_profiler_patch() -> None:
    """Skip temporary graph capture when vLLM's profiler is disabled.

    vLLM 0.26 expanded CUDA graph memory profiling to ROCm. The profiling pass
    captures and destroys a temporary copy of every graph before the real
    capture, leaving AITER with stale graph-owned state. The documented
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS switch otherwise only controls
    whether the estimate is applied to KV-cache sizing.
    """
    import vllm.envs as vllm_envs
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.profile_cudagraph_memory
    if getattr(original, "_atom_skip_disabled_profile", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if not vllm_envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
            logger.info(
                "ATOM plugin: skipping temporary CUDA graph memory capture "
                "because VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
            )
            return 0
        return original(self, *args, **kwargs)

    wrapped._atom_skip_disabled_profile = True
    GPUModelRunner.profile_cudagraph_memory = wrapped
