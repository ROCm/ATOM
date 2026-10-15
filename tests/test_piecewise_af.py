# SPDX-License-Identifier: MIT
"""AF regions over backend-owned buffers, without Python state replay."""

from types import SimpleNamespace

import pytest
import torch

from atom.config import CUDAGraphMode
from atom.utils.attn_ffn_piecewise import piecewise_core
from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")


def _context(key=(), mode=CUDAGraphMode.PIECEWISE):
    return SimpleNamespace(
        in_hipgraph=False,
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(dummy=False, cudagraph_key=key),
        cudagraph_runtime_mode=mode,
    )


def test_backend_result_slots_are_explicit_and_isolated_by_view():
    storage = StableOutputs()
    main = storage.empty(("main", "selected"), (4, 8), dtype=torch.int32, device="cpu")
    child = storage.empty(("ub0", "selected"), (4, 8), dtype=torch.int32, device="cpu")
    main.fill_(7)
    child.fill_(9)
    assert (
        storage.empty(("main", "selected"), (4, 8), dtype=torch.int32, device="cpu")
        is main
    )
    assert main.data_ptr() != child.data_ptr()
    assert torch.all(main == 7)


@cuda
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("void", [False, True])
def test_replay_refreshes_buffers_and_applies_gpu_effect_exactly_once(direct, void):
    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor, md: object):
        layer.calls += 1
        layer.kv.add_(1)
        md.output.copy_(x + md.positions + 2)
        return None if void else md.output

    layer = SimpleNamespace(calls=0, kv=torch.zeros(1, device="cuda"))
    positions = torch.zeros(4, device="cuda")
    hidden = torch.zeros_like(positions)
    output = torch.empty_like(hidden)
    runner = CudagraphCaptureRunner()
    for i in range(1, 5):
        positions.fill_(i)
        hidden.fill_(i * 2)
        # Python wrappers may change; captured storage is owned by the backend.
        md = SimpleNamespace(positions=positions, output=output)
        result = core(
            layer,
            hidden,
            md,
            piecewise=True,
            capture=True,
            runner=runner,
            forward_context=_context(),
            zero_copy_inputs=("x",) if direct else (),
        )
        torch.testing.assert_close(output, hidden + positions + 2)
        if void:
            assert result is None
        else:
            torch.testing.assert_close(result, output)
        assert layer.kv.item() == i
    assert layer.calls == 2
    assert runner.release_graphs() == 1


@cuda
def test_layout_generation_and_budget_fallback():
    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor, source: object):
        layer.calls += 1
        return x + source

    layer = SimpleNamespace(calls=0)
    runner = CudagraphCaptureRunner(max_entries=2)
    x = torch.ones(4, device="cuda")
    sources = [torch.full_like(x, i) for i in range(3)]
    for i in (0, 1, 0, 2, 1):
        result = core(
            layer,
            x,
            sources[i],
            piecewise=True,
            capture=True,
            runner=runner,
            forward_context=_context(key=(i,)),
        )
        torch.testing.assert_close(result, x + sources[i])
    assert layer.calls == 5
    assert runner.release_graphs() == 2


@cuda
def test_eager_producer_then_graph_consumer_reads_current_result():
    @piecewise_core(capture_prefill=True)
    def produce(layer, x: torch.Tensor, results: object):
        results[0].copy_(x + 3)
        return results[0]

    @piecewise_core(capture_prefill=True)
    def consume(layer, x: torch.Tensor, results: object):
        layer.calls += 1
        return x + results[0]

    producer, consumer = CudagraphCaptureRunner(max_entries=1), CudagraphCaptureRunner()
    slot = torch.empty(4, device="cuda")
    layer = SimpleNamespace(calls=0)
    for i in range(5):
        x = torch.full((4,), float(i), device="cuda")
        results = {0: slot}
        produce(
            layer,
            x,
            results,
            piecewise=True,
            capture=True,
            runner=producer,
            forward_context=_context(key=(i,)),
            zero_copy_inputs=(),
        )
        actual = consume(
            layer,
            x,
            results,
            piecewise=True,
            capture=True,
            runner=consumer,
            forward_context=_context(),
            zero_copy_inputs=(),
        )
        torch.testing.assert_close(actual, x * 2 + 3)
    assert layer.calls == 2
    assert producer.release_graphs() == consumer.release_graphs() == 1


@cuda
def test_context_only_model_interface(monkeypatch):
    from atom.utils import forward_context

    @piecewise_core(capture_prefill=True, copy_per_step=("x",))
    def core(layer, x: torch.Tensor):
        return x + 1

    fc = _context()
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: fc)
    layer = SimpleNamespace()
    for i in range(3):
        x = torch.full((4,), float(i), device="cuda")
        torch.testing.assert_close(core(layer, x), x + 1)
    runner = next(iter(layer._piecewise_core_runners.values()))
    assert runner.release_graphs() == 1


@cuda
@pytest.mark.parametrize(
    "gate", ["dummy", "startup", "full", "disabled", "cap", "unsupported"]
)
def test_live_capture_guards(gate):
    @piecewise_core(capture_prefill=True, max_tokens=3 if gate == "cap" else None)
    def core(layer, x: torch.Tensor):
        return x + 1

    context = _context(
        key=None if gate == "unsupported" else (),
        mode=CUDAGraphMode.FULL if gate == "full" else CUDAGraphMode.PIECEWISE,
    )
    context.in_hipgraph = gate == "startup"
    context.attn_metadata.dummy = gate == "dummy"
    runner = CudagraphCaptureRunner()
    x = torch.ones(4, device="cuda")
    out = core(
        SimpleNamespace(),
        x,
        piecewise=True,
        capture=gate != "disabled",
        runner=runner,
        forward_context=context,
    )
    torch.testing.assert_close(out, x + 1)
    assert not runner._graphs


def test_context_interface_preserves_configured_output_capacity(monkeypatch):
    from atom.utils import forward_context
    from atom.utils.attn_ffn_piecewise import configure_piecewise_core

    @piecewise_core(capture_prefill=True, token_dim=1)
    def core(layer, x: torch.Tensor):
        return x * 2

    fc = _context(key=None)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: fc)
    layer = SimpleNamespace()
    outputs = StableOutputs(capacity=8, token_dim=1)
    configure_piecewise_core(layer, outputs=outputs, copy_per_step=("x",))
    addresses = []
    for tokens in (3, 8, 3):
        x = torch.arange(tokens * 2).reshape(1, tokens, 2)
        result = core(layer, x)
        torch.testing.assert_close(result, x * 2)
        addresses.append(result.data_ptr())
    assert len(set(addresses)) == 1
    assert not next(iter(layer._piecewise_core_runners.values()))._graphs
