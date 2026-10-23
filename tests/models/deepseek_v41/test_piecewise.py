# SPDX-License-Identifier: MIT
"""Basic V4.1 piecewise: live metadata and graph boundary address contracts."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.config import CUDAGraphMode
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41 import runtime
from atom.utils import cuda_graph
from atom.utils.backends import split_graph
from atom.utils.cuda_graph import BatchDescriptor, CUDAGraphOptions, CUDAGraphWrapper


def _context(monkeypatch, tokens=8):
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(tokens),
        attn_metadata=SimpleNamespace(image_mask=None),
    )
    from atom.utils import forward_context

    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
    monkeypatch.setattr(cuda_graph, "get_forward_context", lambda: context)
    return context


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("empty_startup", [False, True])
def test_split_graph_replays_with_live_batch_state_and_engram(
    monkeypatch, empty_startup
):
    context = _context(monkeypatch)
    calls = []

    class Rows(runtime.EngramStagedRows):
        __init__ = dict.__init__
        get = dict.get

        def stage(self):
            calls.append("stage")

        def join(self):
            calls.append("join")

    class Engram:
        layer_id = 0

        def __call__(self, residual, embeddings, mask):
            return residual + context.attn_metadata.step.offset

    def begin():
        calls.append("begin")
        context.attn_metadata.step.selected.clear()

    def attention(hidden, *args, queries=None):
        step = context.attn_metadata.step
        assert not step.selected
        step.selected[0] = step.offset
        calls.append(("attention", len(step.requests)))
        return hidden + step.offset

    outputs = cuda_graph.StableOutputs(token_dim=1)
    layer = SimpleNamespace(
        layer_name="test.layer",
        attn=attention,
        engram=Engram(),
        piecewise_outputs=outputs,
        af_piecewise=False,
        af_zero_copy_hidden=True,
    )
    step = SimpleNamespace(
        requests=[1], width=8, offset=1.0, begin_forward=begin, selected={}
    )
    context.attn_metadata = SimpleNamespace(
        step=step,
        cache=None,
        queries=None,
        image_mask=None,
        engram_embeddings=Rows({0: torch.zeros(1, 8, 4, device="cuda")}),
    )
    context.no_compile_layers = {"test.layer": (layer, None)}
    config = SimpleNamespace(compilation_config=SimpleNamespace())
    pieces = []

    def backend(gm, inputs):
        split, items = split_graph(gm, [])
        boundaries = [item for item in items if item.is_splitting_graph]
        assert len(boundaries) == 4
        for item in items:
            if not item.is_splitting_graph:
                wrapper = CUDAGraphWrapper(
                    item.graph,
                    config,
                    CUDAGraphMode.PIECEWISE,
                    CUDAGraphOptions(weak_ref_output=False),
                )
                split.__dict__[item.submod_name] = wrapper
                pieces.append(wrapper)
        return split

    def forward(hidden):
        torch.ops.aiter.v41_piecewise_begin(hidden)
        x = torch.ops.aiter.v41_piecewise_engram(hidden, "test.layer")
        x = x * 2
        x = torch.ops.aiter.v41_piecewise_attention(x, "test.layer")
        x = x + 3
        torch.ops.aiter.v41_piecewise_end(x)
        return x

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
    buffers = {n: torch.zeros(1, n, 4, device="cuda") for n in (8, 16)}
    captured = set()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for n, bs, value in [(8, 1, 1.0), (8, 4, 2.0), (16, 2, 3.0), (8, 2, 4.0)]:
            step.width, step.requests, step.offset = n, list(range(bs)), value
            context.batch_descriptor = BatchDescriptor(n)
            hidden = buffers[n]
            hidden.fill_(value)
            # Warm up new kernels without capturing before each bucket's capture.
            if n not in captured:
                if empty_startup:
                    step.requests = []
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                compiled(hidden)
                context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                compiled(hidden)  # capture records work; it does not execute it
                captured.add(n)
                step.requests = list(range(bs))
                hidden.fill_(value)
            actual = compiled(hidden)
            torch.testing.assert_close(actual, torch.full_like(actual, 5 * value + 3))
    torch.cuda.current_stream().wait_stream(stream)
    assert calls.count("begin") == 8
    assert calls.count("stage") == (4 if empty_startup else 8)
    assert calls.count("join") == (8 if empty_startup else 16)
    assert ("attention", 4) in calls
    assert any(len(piece.concrete_cudagraph_entries) == 2 for piece in pieces)
    for piece in pieces:
        piece.release_graphs()


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.AF_PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_runtime_boundary_storage_stays_bounded_after_graph_budget(monkeypatch, mode):
    def initialize(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        layer = torch.nn.Module()
        layer.layer_name = "test.layer"
        layer.attn = lambda hidden, *args, **kwargs: hidden + 1
        self.layers = torch.nn.ModuleList([layer])

    monkeypatch.setattr(runtime.DeepseekV41MultimodalModel, "__init__", initialize)
    monkeypatch.setattr(runtime.DeepseekV41RuntimeModel, "_rope", lambda *args: None)
    monkeypatch.setattr(runtime, "_Backbone", lambda **kwargs: torch.nn.Identity())
    config = SimpleNamespace(
        hf_config=None,
        max_model_len=64,
        max_num_batched_tokens=64,
        prefill_capture_sizes=[8, 64],
        online_quant_config=None,
        enforce_eager=False,
        compilation_config=SimpleNamespace(
            level=3, cudagraph_mode=mode, static_forward_context={}
        ),
    )
    model = runtime.DeepseekV41RuntimeModel(config)
    layer = model.layers[0]
    context = _context(monkeypatch, tokens=64)
    context.no_compile_layers = config.compilation_config.static_forward_context
    context.attn_metadata = SimpleNamespace(
        step=SimpleNamespace(requests=[1]), cache=None, queries=None
    )
    sample = torch.ones(1, 64, 4)
    first = runtime.v41_attention(sample[:, :33], layer.layer_name)
    runner = next(iter(layer._piecewise_core_runners.values()))
    runner.max_entries = 0
    if mode != CUDAGraphMode.PIECEWISE:
        context.attn_metadata.cudagraph_key = ("live",)
    footprint = sum(
        t.untyped_storage().nbytes() for t in layer.piecewise_outputs._slots.values()
    )
    for n in (*range(34, 65), 33):
        sample.fill_(n)
        actual = runtime.v41_attention(sample[:, :n], layer.layer_name)
        torch.testing.assert_close(actual, torch.full_like(actual, n + 1))
        assert actual.data_ptr() == first.data_ptr()
        assert (
            sum(
                t.untyped_storage().nbytes()
                for t in layer.piecewise_outputs._slots.values()
            )
            == footprint
        )
    assert not runner._graphs


def test_image_stage_bypasses_text_graph(monkeypatch):
    context = _context(monkeypatch)
    context.attn_metadata.image_mask = torch.tensor([[False, True]])
    owner = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(owner)
    owner.piecewise = True

    class Stage:
        def __call__(self, x):
            raise AssertionError("must not replay the text graph")

        def forward(self, x):
            return x + context.attn_metadata.image_mask

    torch.testing.assert_close(
        owner._run_stage(Stage(), torch.zeros(1, 2)), torch.tensor([[0.0, 1.0]])
    )


def test_empty_piecewise_step_does_not_require_engram_rows(monkeypatch):
    context = _context(monkeypatch)
    calls = []
    context.attn_metadata = SimpleNamespace(
        step=SimpleNamespace(requests=[], begin_forward=lambda: calls.append("reset"))
    )
    hidden = torch.ones(1, 8, 4)
    runtime.v41_piecewise_begin(hidden)
    runtime.v41_piecewise_end(hidden)
    assert calls == ["reset"]
    assert torch.count_nonzero(hidden) == 0


def test_prefill_runner_pads_persistent_inputs_and_returns_only_live_rows(monkeypatch):
    from atom.model_engine import model_runner

    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
        attn_metadata=SimpleNamespace(step=SimpleNamespace(width=8, scheduled=5)),
    )
    monkeypatch.setattr(model_runner, "get_forward_context", lambda: context)
    ids, positions = torch.arange(16), torch.arange(16)
    pointers = []

    def model(x, pos):
        assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        assert context.batch_descriptor == BatchDescriptor(8)
        assert x.shape == pos.shape == (8,)
        assert not x[5:].any() and not pos[5:].any()
        pointers.append((x.data_ptr(), pos.data_ptr()))
        return x[:, None] * 3

    runner = SimpleNamespace(
        forward_vars={
            "input_ids": SimpleNamespace(gpu=ids),
            "positions": SimpleNamespace(gpu=positions),
        },
        model=model,
    )
    mode = SimpleNamespace(scheduled_tokens=5, running_tokens=8)
    run = model_runner.ModelRunner._run_piecewise_prefill
    for value in (2, 7):
        ids[:5].fill_(value)
        result = run(runner, mode)
        torch.testing.assert_close(result, torch.full((5, 1), value * 3))
        assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        assert context.batch_descriptor is None
    assert pointers[0] == pointers[1]

    def fail(*args):
        raise RuntimeError("prefill failed")

    runner.model = fail
    with pytest.raises(RuntimeError, match="prefill failed"):
        run(runner, mode)
    assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
    assert context.batch_descriptor is None


def test_backend_build_keeps_state_maintenance_before_padded_prefill(monkeypatch):
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    events = []
    monkeypatch.setattr(
        builder, "relocate_state_slots", lambda x: events.append("relocate")
    )
    monkeypatch.setattr(
        builder, "execute_paged_state_copies", lambda *x: events.append("state")
    )

    def prepare(batch, bs, tokens, **kwargs):
        events.append(("prepare", bs, tokens, kwargs["query_prefix_ready"]))
        return "metadata", "positions"

    monkeypatch.setattr(builder, "_prepare", prepare)
    batch = SimpleNamespace(
        total_tokens_num_prefill=600,
        total_tokens_num=600,
        state_maintenance_ops=SimpleNamespace(
            relocations=[1], checkpoint_stores=[1], checkpoint_restores=[]
        ),
    )
    assert builder.build(batch, 4, 768, 6) == ("metadata", "positions")
    assert events == ["relocate", "state", ("prepare", 4, 768, True)]


def test_padded_prefill_metadata_reuses_buffers_across_request_layouts():
    from atom.model_ops.attentions.deepseek_v41.metadata import (
        RequestSpan,
        prepare_batch_step,
    )
    from tests.attentions.deepseek_v41.helpers import metadata_buffers

    buffers = metadata_buffers(4, 768, 64, device="cpu")
    addresses = []
    for lengths in ((600,), (150, 450), (200, 200, 200)):
        offset = 0
        requests = []
        for i, length in enumerate(lengths):
            requests.append(RequestSpan(i, 1000 + i * 100, offset, length, i))
            offset += length
        step = prepare_batch_step(
            requests,
            "cpu",
            block_tables=[tuple(range(64)) for _ in requests],
            buffers=buffers,
            running_bs=4,
            running_tokens=768,
        )
        assert step.width == 768 and step.scheduled == 600
        assert torch.all(step.batch_ids[600:] == -1)
        assert torch.all(step.positions[600:] == 0)
        assert step.cu_seqlens_q[-1] == 600
        assert len(step.requests) == len(lengths)
        addresses.append((step.batch_ids.data_ptr(), step.positions.data_ptr()))
    assert addresses[0] == addresses[1] == addresses[2]


def _hc_state(tokens, value, device="cpu"):
    return SinglePassHCState(
        *(
            torch.full((1, tokens, width), value, device=device)
            for width in (2, 3, 4, 5, 6)
        )
    )


@pytest.fixture
def late_runtime(monkeypatch):
    context = _context(monkeypatch, tokens=32)
    context.in_hipgraph, context.dp_stages = True, None
    owner = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(owner)
    owner.piecewise = True
    owner._late_inputs = {}
    owner._late_input_storage = cuda_graph.StableOutputs(capacity=8)
    owner.late = torch.nn.Identity()
    owner.late.forward = lambda *inputs: tuple(t * 3 + 1 for t in inputs)
    return owner, context


@pytest.fixture
def captured_late(late_runtime, monkeypatch):
    owner, context = late_runtime
    monkeypatch.setattr(cuda_graph, "_graph_pools", {})
    monkeypatch.setenv("ATOM_PER_BUCKET_POOL", "1")
    compiled = CUDAGraphWrapper(
        owner.late.forward,
        SimpleNamespace(compilation_config=SimpleNamespace()),
        CUDAGraphMode.PIECEWISE,
        CUDAGraphOptions(weak_ref_output=False),
    )
    owner.late.forward = compiled
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for n in (8, 4):
            owner._late_layers(_hc_state(n, 1.0, "cuda"))
    torch.cuda.current_stream().wait_stream(stream)
    try:
        yield owner, context, compiled, stream
    finally:
        compiled.release_graphs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_late_replay_refreshes_all_inputs_and_keeps_unseen_shapes_uncaptured(
    captured_late,
):
    owner, context, compiled, _ = captured_late
    context.in_hipgraph = False
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    for n, value in ((4, 2.0), (8, 5.0), (3, 9.0), (4, 7.0)):
        result = owner._late_layers(_hc_state(n, value, "cuda"))
        for tensor in result.fields():
            torch.testing.assert_close(tensor, torch.full_like(tensor, value * 3 + 1))
    assert set(compiled.concrete_cudagraph_entries) == {
        BatchDescriptor(4),
        BatchDescriptor(8),
    }
    assert set(owner._late_inputs) == {4, 8}
    for small, large in zip(owner._late_inputs[4], owner._late_inputs[8]):
        assert small.data_ptr() == large.data_ptr()
    assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
    assert context.batch_descriptor == BatchDescriptor(32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@pytest.mark.parametrize("mode", [None, CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_late_full_capture_keeps_its_own_inputs(captured_late, mode):
    owner, context, compiled, stream = captured_late
    context.cudagraph_runtime_mode = mode
    state = _hc_state(4, 2.0, "cuda")
    stream.wait_stream(torch.cuda.current_stream())
    outer = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            owner._late_layers(state)
            with torch.cuda.graph(outer, stream=stream):
                result = owner._late_layers(state)
        torch.cuda.current_stream().wait_stream(stream)
        for value in (5.0, 7.0):
            for tensor in state.fields():
                tensor.fill_(value)
            outer.replay()
            for tensor in result.fields():
                torch.testing.assert_close(
                    tensor, torch.full_like(tensor, value * 3 + 1)
                )
        for saved in owner._late_inputs[4]:
            torch.testing.assert_close(saved, torch.ones_like(saved))
        assert context.cudagraph_runtime_mode == mode
        assert context.batch_descriptor == BatchDescriptor(32)
        assert set(compiled.concrete_cudagraph_entries) == {
            BatchDescriptor(4),
            BatchDescriptor(8),
        }
    finally:
        outer.reset()


@pytest.mark.parametrize("guard", ["tbo", "dp"])
@pytest.mark.parametrize("startup", [False, True])
def test_late_parallel_calls_do_not_use_serial_buffers(
    late_runtime, monkeypatch, guard, startup
):
    import threading

    from atom.utils.tbo import ubatching

    owner, context = late_runtime
    owner._late_layers(_hc_state(4, 1.0))
    context.in_hipgraph = startup
    if guard == "tbo":
        monkeypatch.setitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), 0)
    else:
        context.dp_stages = object()

    def late(*inputs):
        assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        assert context.batch_descriptor is None
        return inputs

    owner.late.forward = late
    for n in (4, 3):
        state = _hc_state(n, 7.0)
        result = owner._late_layers(state)
        assert result.residual.data_ptr() == state.residual.data_ptr()
    assert set(owner._late_inputs) == {4}
    for saved in owner._late_inputs[4]:
        torch.testing.assert_close(saved, torch.ones_like(saved))


def test_late_failure_restores_context_without_registering_inputs(late_runtime):
    owner, context = late_runtime

    def fail(*inputs):
        raise RuntimeError("late failed")

    owner.late.forward = fail
    with pytest.raises(RuntimeError, match="late failed"):
        owner._late_layers(_hc_state(4, 1.0))
    assert not owner._late_inputs
    assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    assert context.batch_descriptor == BatchDescriptor(32)
