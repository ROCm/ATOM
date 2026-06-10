"""Track vLLM CUDA-graph warmup/capture phase for ATOM model forwards.

vLLM runs eager warmups immediately before CUDA/HIP graph capture on the same
graph stream, but labels those warmups as ``CUDAGraphMode.NONE`` in its forward
context. Native ATOM sets ``in_hipgraph=True`` for both that eager warmup and
the subsequent capture, so V4 side-stream work and AITER workspaces are prepared
before capture starts.

This patch keeps that phase signal inside ATOM without modifying vLLM sources.
"""

from __future__ import annotations

import functools
import logging
import threading

logger = logging.getLogger("atom")
_phase_local = threading.local()


def is_vllm_graph_capture_phase() -> bool:
    return getattr(_phase_local, "depth", 0) > 0


def apply_vllm_cudagraph_phase_patch() -> bool:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:
        logger.debug(
            "ATOM vLLM cudagraph phase patch: GPUModelRunner unavailable (%s), skip",
            e,
        )
        return False

    original = getattr(GPUModelRunner, "_warmup_and_capture", None)
    if original is None or getattr(original, "_atom_cudagraph_phase_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        _phase_local.depth = getattr(_phase_local, "depth", 0) + 1
        try:
            return original(self, *args, **kwargs)
        finally:
            _phase_local.depth -= 1

    wrapped._atom_cudagraph_phase_patched = True  # type: ignore[attr-defined]
    GPUModelRunner._warmup_and_capture = wrapped
    logger.info(
        "ATOM plugin: patched vLLM GPUModelRunner._warmup_and_capture "
        "to expose CUDA-graph warmup/capture phase to ATOM"
    )
    return True
