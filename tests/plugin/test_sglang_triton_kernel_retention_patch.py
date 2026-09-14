import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from atom.plugin.sglang.patches import triton_kernel_retention_patch as retention


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def test_apply_patch_updates_base_and_loaded_runner_aliases():
    @contextmanager
    def original_freeze_gc(enable_cudagraph_gc):
        yield

    base_module = SimpleNamespace(freeze_gc=original_freeze_gc)
    decode_module = SimpleNamespace(freeze_gc=original_freeze_gc)
    prefill_module = SimpleNamespace(freeze_gc=original_freeze_gc)
    runner_package = _package("sglang.srt.model_executor.runner")
    runner_package.base_cuda_graph_runner = base_module

    fake_modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.model_executor": _package("sglang.srt.model_executor"),
        "sglang.srt.model_executor.runner": runner_package,
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner": decode_module,
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner": prefill_module,
    }

    retention._PATCH_APPLIED = False
    with patch.dict(sys.modules, fake_modules):
        retention.apply_triton_kernel_retention_patch()

    assert base_module.freeze_gc is decode_module.freeze_gc
    assert base_module.freeze_gc is prefill_module.freeze_gc
    assert base_module.freeze_gc is not original_freeze_gc


def test_capture_guard_retains_kernel_and_restores_destructor():
    destructor_calls = []

    class CompiledKernel:
        def __del__(self):
            destructor_calls.append(self)

    compiler_module = ModuleType("triton.compiler.compiler")
    compiler_module.CompiledKernel = CompiledKernel
    fake_modules = {
        "triton": _package("triton"),
        "triton.compiler": _package("triton.compiler"),
        "triton.compiler.compiler": compiler_module,
    }
    original_destructor = CompiledKernel.__del__
    kernel = CompiledKernel()
    retention._CAPTURED_GRAPH_TRITON_KERNELS.clear()

    with patch.dict(sys.modules, fake_modules), retention.retain_triton_kernels():
        CompiledKernel.__del__(kernel)

    assert retention._CAPTURED_GRAPH_TRITON_KERNELS == [kernel]
    assert destructor_calls == []
    assert CompiledKernel.__del__ is original_destructor


def test_freeze_gc_wrapper_preserves_original_policy():
    events = []

    @contextmanager
    def original_freeze_gc(enable_cudagraph_gc):
        events.append(("original_enter", enable_cudagraph_gc))
        yield
        events.append(("original_exit", enable_cudagraph_gc))

    @contextmanager
    def retain_triton_kernels():
        events.append(("retention_enter", None))
        yield
        events.append(("retention_exit", None))

    wrapped_freeze_gc = retention._wrap_freeze_gc(original_freeze_gc)
    with (
        patch.object(retention, "retain_triton_kernels", retain_triton_kernels),
        wrapped_freeze_gc(enable_cudagraph_gc=False),
    ):
        events.append(("body", None))

    assert events == [
        ("original_enter", False),
        ("retention_enter", None),
        ("body", None),
        ("retention_exit", None),
        ("original_exit", False),
    ]
