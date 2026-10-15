# SPDX-License-Identifier: MIT
"""Runner retirement precedes V4.1 builder cleanup; failures retain its owner."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from atom.plugin.vllm import v41_shutdown as patch


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    guard = Mock(side_effect=AssertionError("CPU shutdown test touched a GPU"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", guard)
    monkeypatch.setattr(torch.cuda, "synchronize", guard)
    monkeypatch.setattr(torch.accelerator, "synchronize", guard)
    monkeypatch.setattr(torch.accelerator, "empty_cache", guard)
    yield
    guard.assert_not_called()


@pytest.fixture
def lifecycle(monkeypatch):
    events = []
    options = SimpleNamespace(original_error=None, fence_error=None, close_error=None)
    active_device = []
    device = torch.device("cuda:2")

    class DeviceContext:
        def __init__(self, selected):
            assert selected == device

        def __enter__(self):
            active_device.append(device)

        def __exit__(self, *exc):
            active_device.pop()

    def fence(selected):
        assert selected == device
        assert active_device == [device]
        events.append("fence")
        if options.fence_error is not None:
            raise options.fence_error

    monkeypatch.setattr(torch.cuda, "device", DeviceContext)
    monkeypatch.setattr(torch.cuda, "synchronize", fence)

    class Builder:
        def close(self):
            assert active_device == [device]
            events.append("builder.close")
            if options.close_error is not None:
                raise options.close_error

    builder = Builder()
    context = {"proxy": SimpleNamespace(_atom_v41_builder=builder)}

    class Runner:
        def __init__(self):
            self.device = device
            self.model = SimpleNamespace(_deepseek_v41_builder=builder)
            self.vllm_config = SimpleNamespace(
                compilation_config=SimpleNamespace(static_forward_context=context)
            )

        def shutdown(self, *args, **kwargs):
            events.append(("runner.shutdown", args, kwargs))
            # Match v2's destructive retirement: neither source survives.
            if hasattr(self, "model"):
                del self.model
            context.clear()
            if options.original_error is not None:
                raise options.original_error
            return "runner-result"

    monkeypatch.setattr(patch, "gpu_model_runner_classes", lambda: [Runner])
    patch.apply_vllm_v41_shutdown_patch()
    yield SimpleNamespace(
        runner=Runner(),
        runner_cls=Runner,
        builder=builder,
        events=events,
        options=options,
        context=context,
    )
    assert not active_device


def test_builder_survives_model_retirement_and_closes_once(lifecycle):
    e = lifecycle
    assert e.runner.shutdown(7, reason="stop") == "runner-result"
    assert e.events == [
        ("runner.shutdown", (7,), {"reason": "stop"}),
        "fence",
        "builder.close",
    ]
    assert not hasattr(e.runner, "model")
    assert not e.context
    assert e.runner.shutdown() == "runner-result"
    assert len(e.events) == 3


def test_proxy_builder_is_kept_when_model_is_not_available(lifecycle):
    e = lifecycle
    del e.runner.model
    e.runner.shutdown()
    assert e.events[-2:] == ["fence", "builder.close"]


def test_original_error_does_not_close_builder(lifecycle):
    e = lifecycle
    e.options.original_error = RuntimeError("runner did not stop")
    with pytest.raises(RuntimeError, match="runner did not stop"):
        e.runner.shutdown()
    assert e.events == [("runner.shutdown", (), {})]
    assert e.runner._atom_v41_shutdown.builders == [e.builder]
    assert not e.runner._atom_v41_shutdown.runner_done


@pytest.mark.parametrize("failed_phase", ["fence", "close"])
def test_cleanup_retry_does_not_repeat_original_shutdown(lifecycle, failed_phase):
    e = lifecycle
    setattr(e.options, failed_phase + "_error", RuntimeError("cleanup incomplete"))
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        e.runner.shutdown()
    assert e.runner._atom_v41_shutdown.builders == [e.builder]
    assert e.runner._atom_v41_shutdown.runner_done
    setattr(e.options, failed_phase + "_error", None)
    assert e.runner.shutdown() == "runner-result"
    assert e.events.count(("runner.shutdown", (), {})) == 1
    assert e.events.count("fence") == 2
    assert e.events.count("builder.close") == (2 if failed_phase == "close" else 1)
    assert e.runner._atom_v41_shutdown.builders == []


def test_successful_builder_is_not_reclosed_when_another_owner_fails(lifecycle):
    e = lifecycle
    other = SimpleNamespace(close=Mock(side_effect=RuntimeError("second builder")))
    e.context["other-proxy"] = SimpleNamespace(_atom_v41_builder=other)
    with pytest.raises(RuntimeError, match="second builder"):
        e.runner.shutdown()
    assert e.runner._atom_v41_shutdown.builders == [other]
    other.close.side_effect = None
    e.runner.shutdown()
    assert e.events.count("builder.close") == 1
    assert other.close.call_count == 2


def test_unrelated_runner_is_an_exact_passthrough(lifecycle):
    e = lifecycle
    e.runner.model = SimpleNamespace()
    e.context.clear()
    assert e.runner.shutdown(5, reason="plain") == "runner-result"
    assert e.runner.shutdown() == "runner-result"
    assert e.events == [
        ("runner.shutdown", (5,), {"reason": "plain"}),
        ("runner.shutdown", (), {}),
    ]
    assert not hasattr(e.runner, "_atom_v41_shutdown")


def test_installation_is_idempotent(lifecycle):
    method = lifecycle.runner_cls.shutdown
    assert getattr(method, "_atom_v41_shutdown_patched", False)
    assert patch.apply_vllm_v41_shutdown_patch() == 0
    assert lifecycle.runner_cls.shutdown is method


def test_worker_proc_keeps_tp_owner_when_builder_close_fails(lifecycle, monkeypatch):
    from vllm.v1.executor import multiproc_executor

    e = lifecycle
    tp_destroy = Mock(side_effect=lambda: e.events.append("tp.destroy"))
    env_destroy = Mock(side_effect=lambda: e.events.append("environment.destroy"))
    monkeypatch.setattr(multiproc_executor, "destroy_model_parallel", tp_destroy)
    monkeypatch.setattr(
        multiproc_executor, "destroy_distributed_environment", env_destroy
    )
    proc = multiproc_executor.WorkerProc.__new__(multiproc_executor.WorkerProc)
    proc.rpc_broadcast_mq = proc.worker_response_mq = None
    proc.worker = e.runner
    e.options.close_error = RuntimeError("private owner pending")
    with pytest.raises(RuntimeError, match="private owner pending"):
        proc.shutdown()
    tp_destroy.assert_not_called()
    env_destroy.assert_not_called()
    e.options.close_error = None
    proc.shutdown()
    assert e.events[-3:] == ["builder.close", "tp.destroy", "environment.destroy"]


def test_installs_on_actual_v1_and_v2_classes(monkeypatch):
    from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

    classes = gpu_model_runner_classes()
    assert {cls.__module__ for cls in classes} >= {
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    }
    for cls in classes:
        monkeypatch.setattr(cls, "shutdown", cls.shutdown)
    patch.apply_vllm_v41_shutdown_patch()
    for cls in classes:
        assert getattr(cls.shutdown, "_atom_v41_shutdown_patched", False)


def test_general_plugin_registration_installs_shutdown_hook():
    import inspect

    register = importlib.import_module("atom.plugin.vllm.register")
    assert "apply_vllm_v41_shutdown_patch()" in inspect.getsource(
        register.register_model
    )


@pytest.mark.parametrize("runner_error", [False, True])
def test_actual_v2_shutdown_retires_graph_owners_before_builder(
    lifecycle, monkeypatch, runner_error
):
    import inspect

    from vllm.v1.worker.gpu import model_runner

    e = lifecycle
    cls = model_runner.GPUModelRunner
    original = inspect.unwrap(cls.shutdown)
    monkeypatch.setattr(cls, "shutdown", original)
    monkeypatch.setattr(patch, "gpu_model_runner_classes", lambda: [cls])
    patch.apply_vllm_v41_shutdown_patch()
    runner = cls.__new__(cls)
    runner.device = e.runner.device
    runner.vllm_config = e.runner.vllm_config
    runner.aux_output_connector = None
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.kv_caches = [object()]
    runner.attn_groups = [object()]
    runner.kv_cache_config = object()
    runner.cudagraph_manager = object()
    runner.fast_prefill = runner.pooling_runner = object()

    class Model:
        def __init__(self, name):
            self.name = name
            self._deepseek_v41_builder = e.builder

        def modules(self):
            return iter([self])

    runner.model = Model("target")
    runner.speculator = SimpleNamespace(
        model=Model("draft"), cudagraph_manager=object()
    )
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: e.events.append("runner.fence")
    )
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: e.events.append("empty_cache")
    )
    monkeypatch.setattr(
        model_runner, "set_offloader", lambda value: e.events.append("offloader.clear")
    )
    monkeypatch.setattr(model_runner.gc, "collect", lambda: e.events.append("gc"))

    def clear_layer_caches(modules):
        e.events.append(next(modules).name + ".clear")

    monkeypatch.setattr(model_runner, "clear_layer_kv_caches", clear_layer_caches)

    def free_before_shutdown(config):
        assert runner.cudagraph_manager is None
        assert not runner.kv_caches and not runner.attn_groups
        config.compilation_config.static_forward_context.clear()
        e.events.append("static_context.clear")
        if runner_error:
            raise RuntimeError("v2 retirement failed")

    monkeypatch.setattr(model_runner, "free_before_shutdown", free_before_shutdown)
    if runner_error:
        with pytest.raises(RuntimeError, match="v2 retirement failed"):
            runner.shutdown()
        assert "builder.close" not in e.events
        assert runner._atom_v41_shutdown.builders == [e.builder]
        assert not runner._atom_v41_shutdown.runner_done
    else:
        assert runner.shutdown() is None
        assert not hasattr(runner, "model")
        assert runner.speculator is None
        assert not hasattr(runner, "model_state")
        assert e.events == [
            "runner.fence",
            "offloader.clear",
            "static_context.clear",
            "draft.clear",
            "target.clear",
            "gc",
            "empty_cache",
            "fence",
            "builder.close",
        ]
        assert runner.shutdown() is None
        assert e.events.count("builder.close") == 1
