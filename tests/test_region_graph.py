# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Region graphs: a recorded forward replays to what the same forward
computes eagerly, with regions that read live host state and run on side
streams."""

import gc
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context, region_graph
from atom.utils.graph_holders import release_registered_graphs
from atom.utils.region_graph import (
    TokenGraph,
    _record_segments,
    _weak,
    eager_region,
    join_stream,
    token_graph,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="region graphs record CUDA graphs"
)

# The live host state a region reads at call time, as V4.1's read the step.
LIVE = {"scale": 1.0, "rows": 0}


class Toy(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.a = torch.nn.Linear(dim, dim, device="cuda")
        self.b = torch.nn.Linear(dim, dim, device="cuda")
        self.side = torch.cuda.Stream()
        # One per bucket: a segment reads the buffer of its own row count.
        self.side_out = {}

    @eager_region(stream=lambda self: self.side)
    def beside(self, x: torch.Tensor):
        # Writes state only, like a compressor: nothing returned.
        self.side_out[x.shape[0]].copy_(x * LIVE["scale"])

    @eager_region()
    def in_place(self, x: torch.Tensor):
        # Host-sized work: only the first LIVE["rows"] rows change.
        x[: LIVE["rows"]] *= 2.0
        return x

    @eager_region()
    def allocating(self, x: torch.Tensor):
        return x + LIVE["scale"]

    def forward(self, x):
        h = torch.relu(self.a(x))
        self.beside(h)
        h = self.b(h)
        h = self.in_place(h)
        join_stream(self.side)
        h = h + self.side_out[h.shape[0]]
        return self.allocating(h) * 3.0


@pytest.fixture(scope="module", autouse=True)
def blas_ready():
    """hipBLASLt initialises on its first GEMM, which a recording may not do:
    a test run on its own would otherwise capture that first one."""
    torch.nn.functional.linear(
        torch.randn(8, 64, device="cuda"), torch.randn(64, 64, device="cuda")
    )
    torch.cuda.synchronize()


@pytest.fixture
def context(monkeypatch):
    ctx = SimpleNamespace(in_hipgraph=False, main_stream=torch.cuda.current_stream())
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    return ctx


def _eager(model, x):
    with torch.inference_mode():
        return model(x).clone()


def _set(scale, rows):
    LIVE["scale"], LIVE["rows"] = scale, rows


@torch.inference_mode()
def test_replay_matches_eager_under_changing_inputs_and_host_state(context):
    torch.manual_seed(0)
    model = Toy()
    n = 32
    model.side_out[n] = torch.zeros(n, 64, device="cuda")
    x = torch.randn(n, 64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.main_stream = stream
        _set(1.0, 4)
        model(x)  # warmup
        context.in_hipgraph = True
        segments, _ = _record_segments(model, x, pool=torch.cuda.graph_pool_handle())
        context.in_hipgraph = False
    torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()
    # four breaks (beside, in_place, the join, allocating): five segments
    assert segments.num_segments == 5

    for scale, rows in ((0.5, 0), (2.0, 7), (-1.0, 32)):
        _set(scale, rows)
        fresh = torch.randn(n, 64, device="cuda")
        x.copy_(fresh)
        got = segments.replay().clone()
        want = _eager(model, fresh)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@torch.inference_mode()
def test_unjoined_side_region_is_refused(context):
    model = Toy()
    model.side_out[8] = torch.zeros(8, 64, device="cuda")

    def forward(x):
        model.beside(x)
        return x * 2

    x = torch.randn(8, 64, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        context.main_stream = stream
        with pytest.raises(RuntimeError, match="unjoined"):
            _record_segments(forward, x)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_token_graph_replays_each_bucket_from_any_input_tensor(context):
    torch.manual_seed(1)
    model = Toy()
    graph = TokenGraph()
    stream = torch.cuda.Stream()
    _set(1.0, 3)
    for n in (16, 48):
        model.side_out[n] = torch.zeros(n, 64, device="cuda")
        x = torch.randn(n, 64, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            context.main_stream = stream
            model(x)
            context.in_hipgraph = True
            graph.record(model, (x,), {})
            context.in_hipgraph = False
        torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()

    _set(0.25, 5)
    for n in (48, 16, 48):
        x = torch.randn(n, 64, device="cuda")  # a new address every call
        got = graph.replay((x,), {}).clone()
        torch.testing.assert_close(got, _eager(model, x), rtol=0, atol=0)
    assert not graph.has(TokenGraph.key((torch.empty(32, 64),), {}))
    assert {16, 48} <= set(TokenGraph.captured_sizes())
    # Reached only because the graph registered itself: sleep finds a module's
    # token graphs nowhere else, and no step may pad to a bucket it dropped.
    release_registered_graphs()
    assert not graph.has(TokenGraph.key((x,), {}))
    assert not {16, 48} & set(TokenGraph.captured_sizes())


def test_a_bucket_replays_from_its_inputs_written_in_place(context):
    """A caller that writes the recording's own inputs (`TokenGraph.inputs`)
    replays them as it would a fresh tensor, with no copy in."""
    torch.manual_seed(2)
    model = Toy()
    graph = TokenGraph()
    stream = torch.cuda.Stream()
    _set(1.0, 3)
    model.side_out[32] = torch.zeros(32, 64, device="cuda")
    x = torch.randn(32, 64, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.main_stream = stream
        model(x)
        context.in_hipgraph = True
        graph.record(model, (x,), {})
        context.in_hipgraph = False
    torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()

    key = TokenGraph.key((x,), {})
    (recorded,) = graph.inputs(key)
    fresh = torch.randn(32, 64, device="cuda")
    recorded.copy_(fresh)
    got = graph.replay((recorded,), {}).clone()
    torch.testing.assert_close(got, _eager(model, fresh), rtol=0, atol=0)
    graph.release_graphs()


@token_graph
class Unit(torch.nn.Module):
    """A recording unit with no compile decorator: its Python counts runs."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 64, device="cuda")
        self.runs = 0

    def forward(self, x):
        self.runs += 1
        return torch.relu(self.a(x)) * 3.0


@torch.inference_mode()
def test_a_token_graph_unit_records_and_replays_without_torch_compile(context):
    torch.manual_seed(2)
    unit = Unit()
    context.tr_graph = None
    unit(torch.randn(24, 64, device="cuda"))  # warmup, run directly
    assert unit.runs == 1

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.main_stream = stream
        context.in_hipgraph, context.tr_graph = True, "record"
        unit(torch.randn(24, 64, device="cuda"))
        context.in_hipgraph, context.tr_graph = False, None
    torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()
    assert unit.runs == 2

    context.tr_graph = "replay"
    x = torch.randn(24, 64, device="cuda")
    got = unit(x).clone()
    assert unit.runs == 2  # replayed: no model Python ran
    context.tr_graph = None
    torch.testing.assert_close(got, unit(x), rtol=0, atol=0)
    assert unit.runs == 3
    unit.__dict__["_token_graph"].release_graphs()


@token_graph(per_phase=True)
class PhasedUnit(Unit):
    """A unit whose forward records different work in each phase."""

    def forward(self, x):
        self.runs += 1
        scale = 2.0 if forward_context.get_forward_context().context.is_prefill else 3.0
        return torch.relu(self.a(x)) * scale


def _record_phase(context, unit, rows, prefill):
    """One "record" pass of ``unit`` at ``rows`` in the given phase."""
    context.context = SimpleNamespace(is_prefill=prefill)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.main_stream = stream
        context.in_hipgraph, context.tr_graph = True, "record"
        unit(torch.randn(rows, 64, device="cuda"))
        context.in_hipgraph, context.tr_graph = False, None
    torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()


def _replay_phase(context, unit, x, prefill):
    context.context = SimpleNamespace(is_prefill=prefill)
    context.tr_graph = "replay"
    out = unit(x).clone()
    context.tr_graph = None
    return out


@torch.inference_mode()
def test_a_shared_unit_serves_prefill_with_its_decode_recording(context):
    torch.manual_seed(3)
    unit = Unit()
    _record_phase(context, unit, 32, prefill=False)
    _record_phase(context, unit, 32, prefill=True)  # already held: not recorded
    graph = unit.__dict__["_token_graph"]
    assert len(graph.keys()) == 1 and unit.runs == 2
    assert 32 in TokenGraph.captured_sizes("prefill")

    x = torch.randn(32, 64, device="cuda")
    runs = unit.runs
    got = _replay_phase(context, unit, x, prefill=True)
    assert unit.runs == runs  # replayed
    context.context = SimpleNamespace(is_prefill=True)
    torch.testing.assert_close(got, unit(x), rtol=0, atol=0)
    graph.release_graphs()


@torch.inference_mode()
def test_a_decode_only_recording_is_no_prefill_bucket(context):
    """A shared unit recorded only by a decode pass (V4's TR_HYBRID decode)
    is not a bucket a prefill was captured at: a scheduler that read it as one
    sent idle DP ranks a dummy prefill their builder cannot run."""
    torch.manual_seed(5)
    unit = Unit()
    _record_phase(context, unit, 24, prefill=False)
    assert 24 in TokenGraph.captured_sizes("decode")
    assert 24 in TokenGraph.captured_sizes()
    assert 24 not in TokenGraph.captured_sizes("prefill")
    unit.__dict__["_token_graph"].release_graphs()
    assert 24 not in TokenGraph.captured_sizes()


@torch.inference_mode()
def test_a_per_phase_unit_records_and_replays_each_phase_apart(context):
    torch.manual_seed(4)
    unit = PhasedUnit()
    _record_phase(context, unit, 40, prefill=False)
    assert 40 in TokenGraph.captured_sizes("decode")
    assert 40 not in TokenGraph.captured_sizes("prefill")  # no prefill recording yet

    x = torch.randn(40, 64, device="cuda")
    runs = unit.runs
    _replay_phase(context, unit, x, prefill=True)
    assert unit.runs == runs + 1  # ran directly, not the decode recording

    _record_phase(context, unit, 40, prefill=True)
    assert 40 in TokenGraph.captured_sizes("prefill")
    runs = unit.runs
    decode = _replay_phase(context, unit, x, prefill=False)
    prefill = _replay_phase(context, unit, x, prefill=True)
    assert unit.runs == runs
    torch.testing.assert_close(prefill * 1.5, decode, rtol=0, atol=0)
    unit.__dict__["_token_graph"].release_graphs()


@torch.inference_mode()
def test_a_recording_drains_the_nccl_watchdog_before_it_captures(context, monkeypatch):
    drained = []
    monkeypatch.setattr(
        region_graph,
        "drain_nccl_works",
        lambda: drained.append(torch.cuda.is_current_stream_capturing()),
    )
    graph = TokenGraph()
    x = torch.randn(8, 64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.main_stream = stream
        graph.record(lambda t: t * 2.0, (x,), {})
    torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()
    assert drained == [False]
    graph.release_graphs()


@torch.inference_mode()
def test_live_arguments_are_read_at_replay_not_recorded(context):
    step = {"value": SimpleNamespace(scale=1.0)}

    class Owner:
        @eager_region(live=lambda: {"state": step["value"]})
        def scaled(self, x: torch.Tensor, state):
            return x * state.scale

    owner = Owner()

    def forward(x):
        return owner.scaled(x + 1.0, step["value"]) - 1.0

    x = torch.randn(8, 16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.main_stream = stream
        forward(x)
        segments, _ = _record_segments(forward, x)
    torch.cuda.current_stream().wait_stream(stream)
    context.main_stream = torch.cuda.current_stream()

    step["value"] = SimpleNamespace(scale=3.0)  # a new object, as a new step is
    torch.testing.assert_close(segments.replay().clone(), forward(x), rtol=0, atol=0)


@torch.inference_mode()
def test_recording_goes_on_after_a_holder_goes_without_release(context):
    """One holder records, its output is still held, and the holder goes away
    without `release_graphs` -- its recordings with it. A pool left with no
    recording and live memory refuses another, so the next holder has to
    record into a fresh one."""
    model = Toy()
    stream = torch.cuda.Stream()

    def record(graph, n):
        model.side_out[n] = torch.zeros(n, 64, device="cuda")
        x = torch.randn(n, 64, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            context.main_stream = stream
            model(x)
            context.in_hipgraph = True
            output = graph.record(model, (x,), {})
            context.in_hipgraph = False
        torch.cuda.current_stream().wait_stream(stream)
        context.main_stream = torch.cuda.current_stream()
        return output

    first = TokenGraph()
    held = record(first, 16)
    del first
    gc.collect()
    second = TokenGraph()
    record(second, 16)
    assert second.has(TokenGraph.key((held,), {}))


def test_a_region_argument_keeps_its_named_tuple_type():
    """A replay hands a region its arguments without their storage; a named
    tuple stays one, so a region reading its fields by name (the indexer's
    `IndexCache`) still can."""
    Pair = namedtuple("Pair", "values width")
    value = Pair(torch.zeros(4, device="cuda"), 64)
    weak = _weak(value)
    assert type(weak) is Pair and weak.width == 64
    assert weak.values.data_ptr() == value.values.data_ptr()
    assert type(_weak((value.values, 1))) is tuple
