# SPDX-License-Identifier: MIT
"""Builder-owned buffers and Python result lifetimes, independent of V4.1."""

from types import SimpleNamespace

import pytest
import torch

from atom.config import CUDAGraphMode
from atom.utils.attn_ffn_piecewise import piecewise_core
from atom.utils.piecewise_af import (
    PiecewiseAFRunner,
    PiecewiseBuffers,
    PiecewiseState,
    ReplayCache,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")


def _context(**kwargs):
    return SimpleNamespace(
        in_hipgraph=False,
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(dummy=False),
        cudagraph_runtime_mode=kwargs.get("mode", CUDAGraphMode.PIECEWISE),
    )


def test_cache_transaction_rolls_back_and_publishes_only_its_writes():
    upstream = torch.ones(3)
    cache = ReplayCache({0: upstream, "remove": 4})
    state = PiecewiseState((), (), {"results": cache})
    with state.record() as tx:
        assert cache.get(0) is upstream
        assert "new" not in cache
        cache["new"] = upstream + 1
        del cache["remove"]
        assert "new" in cache
    assert cache[0] is upstream and "new" not in cache and cache["remove"] == 4
    assert state.matches(tx.reads)
    state.publish(tx.writes)
    assert cache[0] is upstream and "remove" not in cache
    torch.testing.assert_close(cache["new"], upstream + 1)
    assert not state.matches(tx.reads)


def test_exception_restores_caches_and_recording_context():
    cache = ReplayCache()
    state = PiecewiseState((), (), {"cache": cache})
    with pytest.raises(RuntimeError, match="failed"), state.record():
        cache[0] = 1
        raise RuntimeError("failed")
    assert not cache
    with state.record():
        cache[1] = 2
    assert not cache
    with pytest.raises(ValueError, match="outside"), state.record():
        ReplayCache()[0] = 1


@cuda
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("void", [False, True])
def test_replay_uses_published_buffers_and_applies_effect_once(direct, void):
    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor, md: object):
        layer.calls += 1
        layer.kv.add_(1)
        if "derived" not in md.cache:
            md.cache["derived"] = x + md.positions
        md.cache["output"] = md.cache["derived"] + 2
        return None if void else md.cache["output"]

    layer = SimpleNamespace(calls=0, kv=torch.zeros(1, device="cuda"))
    positions = torch.zeros(4, device="cuda")
    hidden = torch.zeros_like(positions)
    runner = PiecewiseAFRunner()
    outputs = PiecewiseBuffers()
    for i in range(1, 5):
        positions.fill_(i)
        hidden.fill_(i * 2)
        cache = ReplayCache(stable=True)
        state = PiecewiseState(
            (), (positions, layer.kv), {"cache": cache}, storage=outputs
        )
        md = SimpleNamespace(positions=positions, cache=cache)
        result = core(
            layer,
            hidden,
            md,
            piecewise=True,
            capture=True,
            runner=runner,
            capture_state=state,
            forward_context=_context(),
            zero_copy_inputs=("x",) if direct else (),
        )
        torch.testing.assert_close(cache["output"], hidden + positions + 2)
        if void:
            assert result is None
        else:
            torch.testing.assert_close(result, hidden + positions + 2)
        assert layer.kv.item() == i
    assert layer.calls == 2  # Warmup + record; replay executes no Python core.
    assert runner.release_graphs() == 1
    assert not runner._captures


@cuda
def test_cache_dependencies_changed_addresses_and_full_cache_fallback():
    @piecewise_core(capture_prefill=True)
    def core(layer, x: torch.Tensor, cache: object):
        layer.calls += 1
        return x + cache["source"]

    layer = SimpleNamespace(calls=0)
    runner = PiecewiseAFRunner()
    runner.max_entries = 2
    x = torch.ones(4, device="cuda")
    sources = [torch.full_like(x, i) for i in range(3)]
    for i in (0, 1, 0, 2, 1):
        cache = ReplayCache(
            {"source": sources[i], "irrelevant": torch.empty(i + 1, device="cuda")}
        )
        state = PiecewiseState((), (), {"cache": cache})
        result = core(
            layer,
            x,
            cache,
            piecewise=True,
            capture=True,
            runner=runner,
            capture_state=state,
            forward_context=_context(),
        )
        torch.testing.assert_close(result, x + sources[i])
    assert (
        layer.calls == 5
    )  # Two recordings + one fallback, unrelated writes don't key.
    assert runner.release_graphs() == 2


@cuda
def test_cross_layer_stable_results_survive_producer_capture_and_fallback():
    @piecewise_core(capture_prefill=True)
    def produce(layer, x: torch.Tensor, cache: object):
        cache[0] = x + 3
        return cache[0]

    @piecewise_core(capture_prefill=True)
    def consume(layer, x: torch.Tensor, cache: object):
        layer.calls += 1
        return x + cache[0]

    storage = PiecewiseBuffers()
    producer, consumer = PiecewiseAFRunner(), PiecewiseAFRunner()
    producer.max_entries = 1
    layer = SimpleNamespace(calls=0)
    for i in range(5):
        x = torch.full((4,), float(i), device="cuda")
        cache = ReplayCache(stable=True)
        state = PiecewiseState((), (), {"cache": cache}, storage=storage)
        opts = {
            "piecewise": True,
            "capture": True,
            "capture_state": state,
            "forward_context": _context(),
        }
        produce(layer, x, cache, runner=producer, **opts)
        actual = consume(layer, x, cache, runner=consumer, zero_copy_inputs=(), **opts)
        torch.testing.assert_close(actual, x * 2 + 3)
    assert layer.calls == 2
    assert producer.release_graphs() == consumer.release_graphs() == 1


@cuda
@pytest.mark.parametrize("gate", ["dummy", "startup", "full", "disabled", "cap"])
def test_live_capture_guards(gate):
    @piecewise_core(capture_prefill=True, max_tokens=3 if gate == "cap" else None)
    def core(layer, x: torch.Tensor):
        return x + 1

    context = _context(
        mode=CUDAGraphMode.FULL if gate == "full" else CUDAGraphMode.PIECEWISE
    )
    context.in_hipgraph = gate == "startup"
    context.attn_metadata.dummy = gate == "dummy"
    runner = PiecewiseAFRunner()
    x = torch.ones(4, device="cuda")
    out = core(
        SimpleNamespace(),
        x,
        piecewise=True,
        capture=gate != "disabled",
        runner=runner,
        forward_context=context,
        capture_state=PiecewiseState((), ()),
    )
    torch.testing.assert_close(out, x + 1)
    assert not runner._graphs
