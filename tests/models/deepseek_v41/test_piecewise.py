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
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
    monkeypatch.setattr(cuda_graph, "get_forward_context", lambda: context)
    return context


def test_late_stage_uses_its_own_rows_and_refreshes_input_addresses(monkeypatch):
    context = _context(monkeypatch, 32)
    owner = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(owner)
    owner.piecewise = True
    owner._late_piecewise_inputs = {}
    seen = []

    def late(*inputs):
        seen.append(
            (
                context.cudagraph_runtime_mode,
                context.batch_descriptor,
                inputs[0].data_ptr(),
                inputs[0].clone(),
            )
        )
        return inputs

    owner.late = late

    def state(n, value):
        return SinglePassHCState(torch.full((1, n, 2), value), None, None, None, None)

    owner._late_layers(state(8, 1.0))  # startup capture
    owner._late_layers(state(8, 7.0), allow_capture=False)
    owner._late_layers(state(5, 9.0), allow_capture=False)
    assert seen[0][1] == seen[1][1] == BatchDescriptor(8)
    assert seen[0][2] == seen[1][2]
    assert torch.all(seen[1][3] == 7)
    assert seen[2][0] == CUDAGraphMode.NONE
    assert 5 not in owner._late_piecewise_inputs
    assert context.batch_descriptor == BatchDescriptor(32)
    assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE

    def fail(*inputs):
        raise RuntimeError("stage failed")

    owner.late = fail
    with pytest.raises(RuntimeError, match="stage failed"):
        owner._late_layers(state(5, 0.0), allow_capture=False)
    assert context.batch_descriptor == BatchDescriptor(32)
    assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_split_graph_replays_with_live_batch_state_and_engram(monkeypatch):
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

    def attention(hidden, *args):
        step = context.attn_metadata.step
        assert not step.selected
        step.selected[0] = step.offset
        calls.append(("attention", len(step.requests)))
        return hidden + step.offset

    layer = SimpleNamespace(attn=attention, engram=Engram())
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
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                compiled(hidden)
                context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                compiled(hidden)  # capture records work; it does not execute it
                captured.add(n)
            actual = compiled(hidden)
            torch.testing.assert_close(actual, torch.full_like(actual, 5 * value + 3))
    torch.cuda.current_stream().wait_stream(stream)
    assert calls.count("begin") == 8
    assert calls.count("stage") == 8
    assert calls.count("join") == 16
    assert ("attention", 4) in calls
    assert any(len(piece.concrete_cudagraph_entries) == 2 for piece in pieces)
    for piece in pieces:
        piece.release_graphs()


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


def test_window_tail_capture_shapes_fill_only_missing_token_buckets():
    owner = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(owner)
    owner.piecewise, owner.replay = True, True
    owner.config = SimpleNamespace(sliding_window=128)
    extra = owner.extra_piecewise_capture_shapes([1, 2, 8, 32, 64, 128], [6])
    assert extra == [(128, 1), (128, 2), (128, 4), (128, 5)]
    assert all(bs <= 128 and q <= 6 for bs, q in extra)
    # Existing captures already contain 384 and 768; do not duplicate them.
    assert {bs * q for bs, q in extra} == {128, 256, 512, 640}
    owner.replay = False
    assert owner.extra_piecewise_capture_shapes([128], [6]) == []


def test_tail_replays_when_early_stage_missed_its_bucket(monkeypatch):
    context = _context(monkeypatch, 768)
    owner = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(owner)
    owner.piecewise = True
    owner._late_piecewise_inputs = {}
    seen = []

    def late(*inputs):
        seen.append((context.cudagraph_runtime_mode, context.batch_descriptor))
        return inputs

    owner.late = late
    state = SinglePassHCState(torch.ones(1, 128, 2), None, None, None, None)
    owner._late_layers(state)  # startup capture of T=128
    context.cudagraph_runtime_mode, context.batch_descriptor = CUDAGraphMode.NONE, None
    owner._late_layers(state, allow_capture=False)
    assert seen[-1] == (CUDAGraphMode.PIECEWISE, BatchDescriptor(128))
    assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
    assert context.batch_descriptor is None
