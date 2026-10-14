# SPDX-License-Identifier: MIT
"""Hybrid capture must preserve FULL decode and independent prefill buckets."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.model_engine import model_runner
from atom.utils.forward_context import ForwardMode


def _runner(mode):
    runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
    runner.enforce_eager = False
    runner.config = SimpleNamespace(
        prefill_context_parallel_size=1,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        max_num_seqs=256,
        max_num_batched_tokens=16384,
    )
    runner._extra_piecewise_capture_shapes = lambda sizes, qs: []
    return runner


@pytest.mark.parametrize("token_dim,shape", [(0, (3, 4)), (1, (1, 3, 4))])
def test_token_buffers_share_capacity_without_growing_recorded_storage(
    token_dim, shape
):
    from atom.models.deepseek_v41.piecewise_buffers import TokenBuffers

    buffers = TokenBuffers(8, token_dim)
    small = torch.zeros(shape, dtype=torch.bfloat16)
    large_shape = list(shape)
    large_shape[token_dim] = 8
    large = torch.zeros(large_shape, dtype=small.dtype)
    view = buffers.slot("attention", small)
    assert buffers.slot("attention", large).data_ptr() == view.data_ptr()
    assert buffers.nbytes() == large.numel() * large.element_size()
    assert buffers.slot("engram", small).data_ptr() != view.data_ptr()
    large_shape[token_dim] = 9
    with pytest.raises(ValueError, match="capacity"):
        buffers.slot("attention", torch.empty(large_shape, dtype=small.dtype))
    assert buffers.slot("attention", small).data_ptr() == view.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_token_views_replay_with_separate_bucket_graph_pools():
    from atom.models.deepseek_v41.piecewise_buffers import TokenBuffers

    buffers = TokenBuffers(8, token_dim=0)
    graphs, outputs = {}, {}
    for tokens in (8, 3):
        view = buffers.slot("attention", torch.ones(tokens, 4, device="cuda"))
        view.fill_(1)
        outputs[tokens] = torch.empty_like(view)
        outputs[tokens].copy_(view * 3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            outputs[tokens].copy_(view * 3)
        graphs[tokens] = graph
    assert graphs[8].pool() != graphs[3].pool()
    for tokens, value in [(3, 2), (8, 4), (3, 7), (8, 3)]:
        buffers.slot("attention", outputs[tokens]).fill_(value)
        graphs[tokens].replay()
        torch.testing.assert_close(
            outputs[tokens], torch.full_like(outputs[tokens], 3 * value)
        )


@pytest.mark.parametrize(
    "mode,expected",
    [
        (CUDAGraphMode.FULL, {CUDAGraphMode.FULL}),
        (CUDAGraphMode.PIECEWISE, {CUDAGraphMode.PIECEWISE}),
        (CUDAGraphMode.AF_PIECEWISE, {CUDAGraphMode.PIECEWISE}),
    ],
)
def test_capture_plan_keeps_tail_buckets_out_of_full_decode(mode, expected):
    runner = _runner(mode)
    runner.capture_sizes = [8, 2, 1]
    runner._extra_piecewise_capture_shapes = lambda sizes, qs: [(8, 1), (8, 2)]
    plan = runner._cudagraph_capture_plan([6])
    assert {kind for kind, _, _ in plan} == expected
    full = [(q, sizes) for kind, q, sizes in plan if kind == CUDAGraphMode.FULL]
    if CUDAGraphMode.FULL in expected:
        assert full == [(6, [8, 2, 1])]
    if CUDAGraphMode.PIECEWISE in expected:
        assert {
            (q, tuple(sizes))
            for kind, q, sizes in plan
            if kind == CUDAGraphMode.PIECEWISE
        } == {(6, (8, 2, 1)), (1, (8,)), (2, (8,))}


def test_hybrid_uses_one_token_ladder_with_query_aligned_full_decode():
    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 1025, 16))
    )
    plan = runner._cudagraph_capture_plan([6])
    pw, full = plan
    assert pw == (CUDAGraphMode.PIECEWISE, 1, runner.config.capture_sizes[::-1])
    assert full[0:2] == (CUDAGraphMode.FULL, 6)
    assert len(pw[2]) == 83 and len(full[2]) == 80
    assert full[2][0] == 168 and full[2][-1] == 1
    assert 102 in full[2]  # 600 real tokens dispatch to 612, BS=102.
    runner.config.max_num_seqs = 128
    pw, full = runner._cudagraph_capture_plan([6])
    assert pw[2][0] == 1024  # Prefill token capacity is independent of max BS.
    assert full[2][0] == 128


def test_hybrid_default_ladder_and_internal_tail_shapes():
    config = SimpleNamespace(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        )
    )
    assert len(Config.capture_sizes.fget(config)) == 83
    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = [6, 192, 768]
    runner._extra_piecewise_capture_shapes = lambda sizes, qs: [(128, 1), (256, 1)]
    pw, full = runner._cudagraph_capture_plan([6])
    assert pw[2] == [768, 256, 192, 128, 6]
    assert full[2] == [128, 32, 1]


@pytest.mark.parametrize("from_cli", [False, True])
@pytest.mark.parametrize(
    "mode", ["FULL", "PIECEWISE", "AF_PIECEWISE", "FULL_AND_PIECEWISE"]
)
def test_capture_defaults_reach_engine_config_without_explicit_sizes(mode, from_cli):
    import argparse

    from atom.model_engine.arg_utils import EngineArgs

    if from_cli:
        parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
        args = EngineArgs.from_cli_args(parser.parse_args(["--cudagraph-mode", mode]))
    else:
        args = EngineArgs(cudagraph_mode=mode)
    kwargs = args._get_engine_kwargs()
    assert "cudagraph_capture_sizes" not in kwargs
    config = SimpleNamespace(compilation_config=kwargs["compilation_config"])
    expected = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 1025, 16))
        if mode == "FULL_AND_PIECEWISE"
        else [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64, 128, 256, 512]
    )
    assert Config.capture_sizes.fget(config) == expected


@pytest.mark.parametrize(
    "sizes",
    [
        [1024],
        [8, 128, 704],
        [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64, 128, 256, 512],
    ],
)
def test_hybrid_cli_explicit_capture_sizes_override_default(sizes):
    import argparse

    from atom.model_engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--cudagraph-mode",
                "FULL_AND_PIECEWISE",
                "--cudagraph-capture-sizes",
                str(sizes),
            ]
        )
    )
    config = SimpleNamespace(
        compilation_config=args._get_engine_kwargs()["compilation_config"]
    )
    assert Config.capture_sizes.fget(config) == sizes


def test_python_hybrid_capture_ceiling_override_is_preserved():
    config = SimpleNamespace(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE, cuda_graph_sizes=[256]
        )
    )
    assert Config.capture_sizes.fget(config) == [1, 2, 4] + list(range(8, 257, 8))


@pytest.mark.parametrize("query,expected_bs", [(2, 100), (6, 102), (4, 100)])
def test_decode_padding_selects_ladder_for_query(query, expected_bs):
    import numpy as np

    batch = SimpleNamespace(
        total_tokens_num_prefill=0,
        total_tokens_num=100 * query,
        total_seqs_num=100,
        is_dummy_run=False,
    )
    mode = ForwardMode.decide(
        batch=batch,
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        step_needs_eager=False,
        capture_sizes=np.asarray([1, 100, 102, 128]),
        capture_sizes_by_q={2: np.asarray([1, 100, 128]), 6: np.asarray([1, 102, 128])},
        captured_tokens=None,
        is_block_drafter=True,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=query,
    )
    assert mode.running_bs == expected_bs
    assert mode.running_tokens == expected_bs * query
    assert mode.use_cudagraph == (query in (2, 6))
    assert not mode.piecewise_captured


def test_hybrid_decode_replays_full_graph_even_when_token_bucket_exists(monkeypatch):
    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner._detailed_label_suffix = lambda batch: ""
    runner._piecewise_sorted_tokens = [6, 12, 48]
    runner.logits_in_graph = False
    output = torch.zeros(48, 2)
    runner.forward_vars = {"outputs": output}
    calls = []

    def replay():
        calls.append("full")
        output.fill_(7)

    runner.graphs = {(8, 6): SimpleNamespace(replay=replay)}

    class Model:
        def __call__(self, *args):
            raise AssertionError("hybrid decode must not run piecewise Python")

        def compute_logits(self, hidden):
            return hidden + 1

    runner.model = Model()
    mode = ForwardMode(
        use_cudagraph=True,
        is_prefill=False,
        scheduled_bs=3,
        scheduled_tokens=18,
        running_bs=8,
        running_tokens=48,
        running_tokens_are_unified=True,
        max_seqlen_q=6,
        piecewise_captured=False,
        tbo_collective_active=False,
    )
    context = SimpleNamespace(
        context=SimpleNamespace(
            scheduled_bs=3,
            scheduled_tokens=18,
            running_bs=8,
            is_prefill=False,
            is_dummy_run=False,
            forward_mode=mode,
            positions=torch.zeros(48, dtype=torch.int64),
        ),
        attn_metadata=SimpleNamespace(max_seqlen_q=6),
        ubatch_slices=None,
    )
    monkeypatch.setattr(model_runner, "get_forward_context", lambda: context)
    logits, hidden = runner.run_model(torch.zeros(18, dtype=torch.int32))
    assert calls == ["full"]
    torch.testing.assert_close(hidden, torch.full((18, 2), 7.0))
    torch.testing.assert_close(logits, torch.full((18, 2), 8.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_full_decode_and_af_prefill_share_model_with_live_metadata(monkeypatch):
    import numpy as np

    from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
    from atom.models.deepseek_v41 import runtime
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from atom.utils import cuda_graph
    from atom.utils.backends import split_graph
    from atom.utils.cuda_graph import (
        BatchDescriptor,
        CUDAGraphOptions,
        CUDAGraphWrapper,
    )

    step = BatchStep(
        (RequestSpan(0, 0, 0, 4, 0),),
        torch.zeros(4, device="cuda", dtype=torch.int64),
        torch.tensor([0, 4], device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.zeros(4, device="cuda", dtype=torch.int32),
        torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        scheduled=4,
        max_q_len=4,
        request_cached_until=np.zeros(1, dtype=np.int32),
    )
    step.indptrs = {0: (step.cu_seqlens_q, None, 0)}
    cache = SimpleNamespace(pool=torch.zeros(1, device="cuda"))

    class Core:
        spec = LayerAttentionSpec(0, 0, AttentionMode.WINDOW)

        def __call__(self, hidden, scale, cache, step, rope, queries=None):
            cache.pool.add_(1)
            return hidden + step.positions[None, :, None]

    layer = SimpleNamespace(attn=Core(), af_piecewise=True, af_zero_copy_hidden=True)
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
        in_hipgraph=True,
        context=SimpleNamespace(is_dummy_run=False),
        no_compile_layers={"hybrid.layer": (layer, None)},
        attn_metadata=SimpleNamespace(
            step=step,
            cache=cache,
            queries=None,
            dummy=False,
            engram_embeddings={},
            image_mask=None,
        ),
    )
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
    monkeypatch.setattr(cuda_graph, "get_forward_context", lambda: context)
    config = SimpleNamespace(compilation_config=SimpleNamespace())
    pieces = []

    def backend(gm, inputs):
        split, items = split_graph(gm, [])
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
        output = torch.ops.aiter.v41_piecewise_attention(hidden * 2, "hybrid.layer") + 3
        torch.ops.aiter.v41_piecewise_end(output)
        return output

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
    hidden = torch.ones(1, 4, 2, device="cuda")
    destination = torch.empty_like(hidden)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(hidden)
        context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        context.batch_descriptor = BatchDescriptor(4)
        compiled(hidden)
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        context.batch_descriptor = None
        compiled(hidden)
        full = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full):
            destination.copy_(compiled(hidden))
    torch.cuda.current_stream().wait_stream(stream)
    assert not hasattr(layer, "_af_attention")
    context.in_hipgraph = False
    cache.pool.zero_()
    for i, use_full in enumerate([True, False, True, False], 1):
        hidden.fill_(i)
        step.positions.fill_(10 * i)
        if use_full:
            full.replay()
            actual = destination
        else:
            context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            context.batch_descriptor = BatchDescriptor(4)
            actual = compiled(hidden)
        torch.testing.assert_close(actual, torch.full_like(hidden, 12 * i + 3))
        assert cache.pool.item() == i
    assert len(layer._af_attention._graphs) == 1
    layer._af_attention.release_graphs()
    for piece in pieces:
        piece.release_graphs()
