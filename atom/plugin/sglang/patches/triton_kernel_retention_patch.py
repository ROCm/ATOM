"""Keep Triton modules alive across SGLang graph and compile-only execution."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any
from unittest.mock import patch

_CAPTURED_GRAPH_TRITON_KERNELS: list[Any] = []
_PATCH_APPLIED = False
FreezeGC = Callable[[bool], AbstractContextManager[None]]


@contextmanager
def retain_triton_kernels() -> Iterator[None]:
    """Retain Triton kernels whose destructors run inside this execution scope."""

    try:
        from triton.compiler.compiler import CompiledKernel
    except ImportError:
        yield
        return

    def retain_kernel(kernel: Any) -> None:
        _CAPTURED_GRAPH_TRITON_KERNELS.append(kernel)

    with patch.object(CompiledKernel, "__del__", retain_kernel):
        yield


def _wrap_freeze_gc(original_freeze_gc: FreezeGC) -> FreezeGC:
    """Add Triton retention without changing SGLang's GC policy."""

    @contextmanager
    def freeze_gc(enable_cudagraph_gc: bool) -> Iterator[None]:
        with original_freeze_gc(enable_cudagraph_gc), retain_triton_kernels():
            yield

    return freeze_gc


def apply_triton_kernel_retention_patch() -> None:
    """Install the capture guard without modifying the SGLang source tree.

    Decode and prefill runners import ``freeze_gc`` by value. Patch their local
    aliases when already loaded; runners imported later receive the patched
    symbol from ``base_cuda_graph_runner``.
    """

    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from sglang.srt.model_executor.runner import base_cuda_graph_runner

    freeze_gc = _wrap_freeze_gc(base_cuda_graph_runner.freeze_gc)
    base_cuda_graph_runner.freeze_gc = freeze_gc
    for module_name in (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            module.freeze_gc = freeze_gc

    _PATCH_APPLIED = True
