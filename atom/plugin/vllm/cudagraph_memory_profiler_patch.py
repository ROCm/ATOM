import functools
import logging

logger = logging.getLogger("atom")


def apply_vllm_cudagraph_memory_profiler_patch() -> None:
    """Skip vLLM's temporary CUDA graph capture on ROCm.

    vLLM 0.26 expanded CUDA graph memory profiling to ROCm. The profiling pass
    captures and destroys a temporary copy of every graph before the real
    capture, leaving AITER with stale graph-owned state.
    """
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

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
