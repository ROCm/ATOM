import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

from atom.plugin.sglang.patches import prefill_compile_only_patch as compile_only_patch
from atom.plugin.sglang.patches.prefill_compile_only_patch import (
    apply_prefill_compile_only_patch,
)


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_compile_only_patch_uses_live_batch_without_cudagraph(monkeypatch):
    monkeypatch.setenv("ATOM_SGLANG_PREFILL_COMPILE_ONLY", "1")
    retention_calls = []
    init_native_flags = []
    functionalization_flags = []

    @contextmanager
    def retain_triton_kernels():
        retention_calls.append("enter")
        yield
        retention_calls.append("exit")

    monkeypatch.setattr(
        compile_only_patch,
        "retain_triton_kernels",
        retain_triton_kernels,
    )

    class Backend:
        TC_PIECEWISE = "tc_piecewise"

    class ShapeKey:
        def __init__(self, size):
            self.size = size

    class CompileConfig:
        def __init__(self):
            self.capture_sizes = [8192, 16384]
            self.compiler = "inductor"

    class SGLangBackend:
        def __init__(self):
            self.inductor_config = {"enable_auto_functionalized_v2": False}

    class TcPiecewiseCudaGraphBackend:
        @staticmethod
        def build_compilation_config(server_args):
            return CompileConfig()

    class PrefillCudaGraphRunner:
        prefill_backend_name = Backend.TC_PIECEWISE
        max_num_tokens = 16384

        def __init__(self, model_runner):
            self.prefill_backend_name = (
                model_runner.server_args.cuda_graph_config.prefill.backend
            )
            self.model_runner = model_runner
            init_native_flags.append(
                compile_only_patch.is_compile_only_prefill_active()
            )
            backend = SGLangBackend()
            functionalization_flags.append(
                backend.inductor_config["enable_auto_functionalized_v2"]
            )

        def can_replay_locally(self, **kwargs):
            return kwargs.get("input_embeds") is None

        def capture(self):
            raise AssertionError("original graph capture must not run")

        def execute(self, forward_batch, **kwargs):
            raise AssertionError("original graph execute must not run")

    modules = {
        "sglang.srt.compilation.backend": _module(
            "sglang.srt.compilation.backend", SGLangBackend=SGLangBackend
        ),
        "sglang.srt.model_executor.cuda_graph_config": _module(
            "sglang.srt.model_executor.cuda_graph_config", Backend=Backend
        ),
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner": _module(
            "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
            PrefillCudaGraphRunner=PrefillCudaGraphRunner,
        ),
        "sglang.srt.model_executor.runner.shape_key": _module(
            "sglang.srt.model_executor.runner.shape_key", ShapeKey=ShapeKey
        ),
        "sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend": _module(
            "sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend",
            TcPiecewiseCudaGraphBackend=TcPiecewiseCudaGraphBackend,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert apply_prefill_compile_only_patch()
    backend = SGLangBackend()
    assert backend.inductor_config["enable_auto_functionalized_v2"] is False

    prefill_config = SimpleNamespace(backend="tc_piecewise", tc_compiler="eager")
    server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=prefill_config)
    )
    compile_config = TcPiecewiseCudaGraphBackend.build_compilation_config(server_args)
    assert prefill_config.tc_compiler == "eager"
    assert compile_config.compiler == "inductor"
    assert compile_config.capture_sizes == []

    calls = []

    class AttentionBackend:
        def init_forward_metadata(self, forward_batch):
            calls.append(("metadata", forward_batch))

    class CompiledBackend:
        @contextmanager
        def replay_session(self):
            calls.append(("session", "enter"))
            yield
            calls.append(("session", "exit"))

        def replay(self, shape_key, forward_batch, **kwargs):
            calls.append(
                (
                    "replay",
                    shape_key.size,
                    forward_batch,
                    kwargs,
                    compile_only_patch.is_compile_only_prefill_active(),
                )
            )
            return "compiled-output"

    runner = PrefillCudaGraphRunner(SimpleNamespace(server_args=server_args))
    assert init_native_flags == [True]
    assert functionalization_flags == [True]
    runner.backend = CompiledBackend()
    runner.model_runner = SimpleNamespace(attn_backend=AttentionBackend())
    runner.warmup = lambda: calls.append(
        ("warmup", compile_only_patch.is_compile_only_prefill_active())
    )
    runner._validate_capture_hidden_mode = lambda forward_batch: calls.append(
        ("validate", forward_batch)
    )

    @contextmanager
    def prefill_context(forward_batch, **kwargs):
        calls.append(("context", forward_batch, kwargs))
        yield

    runner._prefill_forward_context = prefill_context
    runner._finalize_execute_output = lambda output: ("final", output)

    live_batch = SimpleNamespace(input_ids=range(123), batch_size=2)
    runner.capture()
    result = runner.execute(live_batch, marker="live")

    assert result == ("final", "compiled-output")
    assert retention_calls == ["enter", "exit", "enter", "exit"]
    assert ("warmup", True) in calls
    assert runner.raw_num_tokens == 123
    assert runner.raw_bs == 2
    assert ("metadata", live_batch) in calls
    assert ("replay", 123, live_batch, {"marker": "live"}, True) in calls
    assert compile_only_patch.is_compile_only_prefill_active() is False

    base_kwargs = {
        "batch_size": 2,
        "num_tokens": 1024,
        "input_embeds": None,
        "replace_embeds": None,
        "prefix_lens": [0, 0],
        "is_target_verify": False,
        "capture_hidden_mode": None,
        "return_logprob": False,
    }
    assert runner.can_replay_locally(**base_kwargs)
    assert runner.can_replay_locally(**{**base_kwargs, "num_tokens": 16385}) is False
    assert (
        runner.can_replay_locally(**{**base_kwargs, "input_embeds": object()}) is False
    )
