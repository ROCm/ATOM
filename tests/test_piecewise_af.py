# SPDX-License-Identifier: MIT
"""AF regions over backend-owned buffers, without Python state replay."""

import gc
import weakref
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
        # Copied inputs may move every step; moving a direct input must create
        # a new graph, then reuse it while that producer address stays stable.
        if not direct or i == 3:
            hidden = hidden.clone()
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
    assert layer.calls == (4 if direct else 2)
    assert runner.release_graphs() == (2 if direct else 1)


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


@pytest.mark.parametrize("capacity", [0, 8])
def test_context_interface_preserves_configured_output_capacity(monkeypatch, capacity):
    from atom.utils import forward_context
    from atom.utils.attn_ffn_piecewise import configure_piecewise_core

    @piecewise_core(capture_prefill=True, token_dim=1)
    def core(layer, x: torch.Tensor):
        return x * 2

    fc = _context(key=None)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: fc)
    layer = SimpleNamespace()
    outputs = StableOutputs(capacity=capacity, token_dim=1)
    configure_piecewise_core(layer, outputs=outputs, copy_per_step=("x",))
    addresses = []
    for tokens in (3, 8, 3):
        x = torch.arange(tokens * 2).reshape(1, tokens, 2)
        result = core(layer, x)
        torch.testing.assert_close(result, x * 2)
        addresses.append(result.data_ptr())
    assert addresses[0] == addresses[2]
    assert (addresses[0] == addresses[1]) == bool(capacity)
    assert len(outputs._slots) == (1 if capacity else 2)
    assert not next(iter(layer._piecewise_core_runners.values()))._graphs


@cuda
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("limit", [0, 1, 2])
def test_live_keys_obey_budget_and_replay_existing_graphs(explicit, limit):
    @piecewise_core(capture_prefill=True, copy_per_step=("x",))
    def core(layer, x: torch.Tensor, source: object):
        layer.calls += 1
        return x + source

    runner, layer = CudagraphCaptureRunner(max_entries=limit), SimpleNamespace(calls=0)
    x = torch.ones(4, device="cuda")
    sources = [torch.full_like(x, key * 10) for key in range(3)]
    try:
        for i, key in enumerate((0, 1, 0, 2, 1, 0)):
            x.fill_(i)
            result = core(
                layer,
                x,
                sources[key],
                piecewise=True,
                capture=True,
                runner=runner,
                forward_context=_context(key=(key,)),
                capture_key=("live", key) if explicit else None,
            )
            torch.testing.assert_close(result, x + sources[key])
            assert len(runner._graphs) == min(limit, (1, 2, 2, 3, 3, 3)[i])
        # A new graph warms up and records; misses execute once and replays
        # never re-enter Python, even after the recording budget is exhausted.
        assert layer.calls == (6, 5, 5)[limit]
    finally:
        assert runner.release_graphs() == limit


@cuda
def test_legacy_startup_capture_is_not_limited_by_live_budget():
    @piecewise_core(copy_per_step=("x",))
    def core(layer, x: torch.Tensor):
        return x + 1

    runner, layer = CudagraphCaptureRunner(max_entries=0), SimpleNamespace()
    context = _context()
    context.in_hipgraph = True
    context.attn_metadata.state = SimpleNamespace(name="DECODE")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for n in (2, 4, 8):
                core(
                    layer,
                    x=torch.ones(n, device="cuda"),
                    piecewise=True,
                    capture=True,
                    runner=runner,
                    outputs=runner.outputs,
                    forward_context=context,
                )
        torch.cuda.current_stream().wait_stream(stream)
        context.in_hipgraph = False
        for n in (4, 2, 8):
            result = core(
                layer,
                x=torch.full((n,), 7.0, device="cuda"),
                piecewise=True,
                capture=True,
                runner=runner,
                outputs=runner.outputs,
                forward_context=context,
            )
            torch.testing.assert_close(result, torch.full_like(result, 8.0))
        assert len(runner._graphs) == 3
    finally:
        runner.release_graphs()


@cuda
@pytest.mark.parametrize("policy", ["call", "configured"])
def test_explicit_live_key_refreshes_fresh_inputs_with_effective_policy(policy):
    from atom.utils.attn_ffn_piecewise import configure_piecewise_core

    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor):
        layer.calls += 1
        return x * 2 + 1

    runner, layer = CudagraphCaptureRunner(), SimpleNamespace(calls=0)
    options = {"zero_copy_inputs": ()} if policy == "call" else {}
    if policy == "configured":
        configure_piecewise_core(layer, copy_per_step=("x",))
    inputs = [torch.full((4,), float(i), device="cuda") for i in range(3)]
    try:
        for x in inputs:
            actual = core(
                layer,
                x,
                piecewise=True,
                capture=True,
                runner=runner,
                forward_context=_context(),
                capture_key="same-layout",
                **options,
            )
            torch.testing.assert_close(actual, x * 2 + 1)
        assert layer.calls == 2
        assert len(runner._graphs) == 1
    finally:
        runner.release_graphs()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("invalid", ["x", ("unknown",)])
def test_live_capture_validates_effective_input_policy(explicit, invalid):
    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor):
        return x + 1

    with pytest.raises(ValueError, match="zero_copy_inputs"):
        core(
            SimpleNamespace(),
            torch.ones(4),
            piecewise=True,
            capture=True,
            runner=CudagraphCaptureRunner(),
            forward_context=_context(),
            capture_key="explicit" if explicit else None,
            zero_copy_inputs=invalid,
        )


@cuda
@pytest.mark.parametrize("explicit", [False, True])
def test_live_graph_retains_external_metadata_until_release(explicit):
    class Metadata:
        def __init__(self):
            self.values = torch.full((4,), 3.0, device="cuda")

    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor, metadata: object):
        return x + metadata.values

    runner, layer = CudagraphCaptureRunner(), SimpleNamespace()
    metadata = Metadata()
    reference = weakref.ref(metadata)
    x = torch.ones(4, device="cuda")
    options = {
        "piecewise": True,
        "capture": True,
        "runner": runner,
        "forward_context": _context(),
        "capture_key": "external" if explicit else None,
    }
    try:
        torch.testing.assert_close(core(layer, x, metadata, **options), x + 3)
        del metadata
        gc.collect()
        assert reference() is not None
        reference().values.fill_(7)
        torch.testing.assert_close(core(layer, x, reference(), **options), x + 7)
        torch.cuda.synchronize()
    finally:
        runner.release_graphs()
    gc.collect()
    assert reference() is None
