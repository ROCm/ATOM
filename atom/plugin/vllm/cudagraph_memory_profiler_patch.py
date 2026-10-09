import functools
import logging

logger = logging.getLogger("atom")

# vLLM has two GPUModelRunner classes that do not inherit from each other, and
# `GPUWorker` picks between them on `use_v2_model_runner`. 0.31 runs the V2 one
# by default. A patch that names only the V1 module installs cleanly, logs that
# it installed, and covers nothing -- "installed" and "covered zero classes"
# produce the same line. Both are named here, and the count of classes actually
# patched is logged, so the inert case is visible rather than inferred.
_RUNNER_MODULES = (
    "vllm.v1.worker.gpu_model_runner",  # V1
    "vllm.v1.worker.gpu.model_runner",  # V2, the default from 0.31
)


def apply_vllm_cudagraph_memory_profiler_patch() -> None:
    """Skip vLLM's temporary CUDA graph capture on ROCm.

    vLLM 0.26 expanded CUDA graph memory profiling to ROCm. The profiling pass
    captures and destroys a temporary copy of every graph before the real
    capture, leaving AITER with stale graph-owned state. For DeepSeek-V4.1 it
    does something worse: the capture runs during `determine_available_memory`,
    before the KV cache exists, so the proxy pool it binds against is the
    profiling stub -- the bind fails with a pool far too small for the geometry
    and the engine never starts.
    """
    import importlib

    from vllm.platforms import current_platform

    patched = []
    for name in _RUNNER_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            # A version difference, not a fault: V2 does not exist before 0.31
            # and V1 may eventually go. Covering neither is what gets reported.
            continue
        runner = getattr(module, "GPUModelRunner", None)
        original = getattr(runner, "profile_cudagraph_memory", None)
        if original is None or getattr(original, "_atom_skip_rocm_profile", False):
            continue

        @functools.wraps(original)
        def wrapped(self, *args, __original=original, **kwargs):
            if current_platform.is_rocm():
                logger.info(
                    "ATOM plugin: skipping unsafe temporary CUDA graph memory "
                    "capture on ROCm"
                )
                return 0
            return __original(self, *args, **kwargs)

        wrapped._atom_skip_rocm_profile = True
        runner.profile_cudagraph_memory = wrapped
        patched.append(name)

    if not patched:
        logger.warning(
            "ATOM plugin: found no GPUModelRunner.profile_cudagraph_memory to "
            "patch in %s. On ROCm the temporary capture will run, which leaves "
            "AITER holding stale graph state and fails the DeepSeek-V4.1 proxy "
            "bind against the profiling stub.",
            ", ".join(_RUNNER_MODULES),
        )
    else:
        logger.info(
            "ATOM plugin: ROCm cudagraph memory profiling disabled on %d runner "
            "class(es): %s",
            len(patched),
            ", ".join(patched),
        )
