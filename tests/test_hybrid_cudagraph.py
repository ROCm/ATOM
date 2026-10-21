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
    class RunnerConfig(SimpleNamespace):
        @property
        def prefill_capture_sizes(self):
            return getattr(self, "prefill_sizes", self.capture_sizes)

    runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
    runner.enforce_eager = False
    runner.attn_metadata_builder = SimpleNamespace(supports_piecewise_prefill=True)
    runner.config = RunnerConfig(
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
    from atom.utils.cuda_graph import StableOutputs

    buffers = StableOutputs(capacity=8, token_dim=token_dim)
    small = torch.zeros(shape, dtype=torch.bfloat16)
    large_shape = list(shape)
    large_shape[token_dim] = 8
    large = torch.zeros(large_shape, dtype=small.dtype)
    view = buffers.slot("attention", small)
    assert buffers.slot("attention", large).data_ptr() == view.data_ptr()
    assert view.untyped_storage().nbytes() == large.numel() * large.element_size()
    assert buffers.slot("engram", small).data_ptr() != view.data_ptr()
    large_shape[token_dim] = 9
    with pytest.raises(ValueError, match="capacity"):
        buffers.slot("attention", torch.empty(large_shape, dtype=small.dtype))
    assert buffers.slot("attention", small).data_ptr() == view.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_token_views_replay_with_separate_bucket_graph_pools():
    from atom.utils.cuda_graph import StableOutputs

    buffers = StableOutputs(capacity=8)
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
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.AF_PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_stage_declarations_extend_only_piecewise_capture(mode):
    runner = _runner(mode)
    del runner._extra_piecewise_capture_shapes
    runner.model = torch.nn.Sequential(torch.nn.Identity())
    runner.capture_sizes = runner.config.capture_sizes = [8, 2, 1]
    before = runner._cudagraph_capture_plan([3])
    runner.model[0].piecewise_capture_multiple = 4
    after = runner._cudagraph_capture_plan([3])
    assert [p for p in before if p[0] == CUDAGraphMode.FULL] == [
        p for p in after if p[0] == CUDAGraphMode.FULL
    ]
    if mode != CUDAGraphMode.FULL:
        tokens = {
            bs * q
            for kind, q, sizes in after
            if kind == CUDAGraphMode.PIECEWISE
            for bs in sizes
        }
        assert 4 in tokens
        if mode == CUDAGraphMode.FULL_AND_PIECEWISE:
            assert tokens == {1, 2, 4, 8}
        else:
            assert tokens == {3, 4, 6, 8, 12, 16, 24}


def test_hybrid_preserves_dense_small_decode_batches_with_token_prefill():
    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 1025, 16))
    )
    plan = runner._cudagraph_capture_plan([6])
    pw, full = plan
    assert pw == (CUDAGraphMode.PIECEWISE, 1, runner.config.capture_sizes[::-1])
    assert full[0:2] == (CUDAGraphMode.FULL, 6)
    assert len(pw[2]) == 83 and len(full[2]) == 87
    assert set(range(1, 33)) <= set(full[2])
    assert full[2][0] == 168 and full[2][-1] == 1
    assert 102 in full[2]  # 600 real tokens dispatch to 612, BS=102.
    runner.config.max_num_seqs = 128
    pw, full = runner._cudagraph_capture_plan([6])
    assert pw[2][0] == 1024  # Prefill token capacity is independent of max BS.
    assert full[2][0] == 128


@pytest.mark.parametrize("q_buckets", [[6], [2, 6]])
def test_hybrid_small_token_ceiling_keeps_prefill_with_eager_decode(q_buckets):
    import numpy as np

    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = [4]
    runner.config.max_num_seqs = 512
    runner.config.max_num_batched_tokens = 2048
    plan = runner._cudagraph_capture_plan(q_buckets)
    assert plan[0] == (CUDAGraphMode.PIECEWISE, 1, [4])
    by_q = {q: np.asarray(sizes[::-1]) for _, q, sizes in plan[1:]}
    union = np.asarray(sorted({int(bs) for sizes in by_q.values() for bs in sizes}))
    assert by_q[6].size == 0
    for prefill, q in [(True, 4), *[(False, q) for q in q_buckets]]:
        mode = ForwardMode.decide(
            batch=SimpleNamespace(
                total_tokens_num_prefill=4 if prefill else 0,
                total_tokens_num=q,
                total_seqs_num=1,
                is_dummy_run=False,
            ),
            dp_size=1,
            dp_group=None,
            enforce_eager=False,
            step_needs_eager=False,
            capture_sizes=union,
            capture_sizes_by_q=by_q,
            captured_tokens=[4] if prefill else None,
            supports_piecewise_prefill=True,
            is_block_drafter=True,
            tbo_on=False,
            local_tbo=(False, False, 0, 0),
            max_seqlen_q=q,
        )
        assert mode.piecewise_captured == prefill
        assert mode.use_cudagraph == (not prefill and q == 2)
        assert mode.running_bs == 1 and mode.running_tokens == q


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@pytest.mark.parametrize("supported", [False, True])
def test_prefill_only_plan_completes_runner_startup_and_replays(monkeypatch, supported):
    from contextlib import contextmanager

    from atom.distributed import indexer_cp
    from atom.spec_decode.drafter import Drafter
    from atom.utils import cuda_graph

    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = [4]
    runner.config.max_num_seqs = 512
    runner.config.max_num_batched_tokens = 2048
    runner.config.enable_tbo = False
    runner.config.gpu_memory_utilization = 0.9
    runner.device = torch.device("cuda")
    runner.rank, runner.world_size = 0, 1
    runner.use_mrope = runner.mark_trace = False
    runner.profiler_dir = None
    runner._force_aiter_unreg_capture_for_piecewise = lambda: None
    runner._piecewise_skip_capture = lambda tokens: False
    runner.get_dp_padding = lambda tokens: (0, None)
    runner.h2d_owner = SimpleNamespace(begin=lambda: None, finish=lambda: None)
    ids = torch.zeros(2048, device="cuda", dtype=torch.int32)
    positions = torch.arange(2048, device="cuda", dtype=torch.int64)
    runner.forward_vars = {
        "input_ids": SimpleNamespace(gpu=ids),
        "positions": SimpleNamespace(gpu=positions),
        "outputs": torch.empty(2048, 2, device="cuda"),
    }

    def no_decode(*args, **kwargs):
        pytest.fail("prefill-only startup must not build decode or draft metadata")

    runner.attn_metadata_builder = SimpleNamespace(
        supports_piecewise_prefill=supported,
        capture_owns_cu_seqlens_q=True,
        blank_cache_write_targets=lambda: None,
        build_for_cudagraph_capture=no_decode,
        build_for_piecewise_capture=lambda n: (
            SimpleNamespace(),
            SimpleNamespace(positions=positions[:n]),
        ),
    )
    runner.drafter = SimpleNamespace(
        runner=runner,
        mtp_k=5,
        uses_confidence_schedule=False,
        is_block_drafter=True,
        draft_graphs=(SimpleNamespace(warmup=no_decode),),
    )
    runner.drafter.warmup_draft_graphs = lambda builder, stream: (
        Drafter.warmup_draft_graphs(runner.drafter, builder, stream)
    )
    context = SimpleNamespace(cudagraph_runtime_mode=None, batch_descriptor=None)
    monkeypatch.setattr(model_runner, "set_forward_context", lambda **kw: None)
    monkeypatch.setattr(model_runner, "get_forward_context", lambda: context)
    monkeypatch.setattr(cuda_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(cuda_graph, "_graph_pools", {})
    monkeypatch.setenv("ATOM_PER_BUCKET_POOL", "1")
    monkeypatch.setattr(indexer_cp, "warmup_exchange", lambda device: None)

    @contextmanager
    def capture_stream():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)
        torch.cuda.current_stream().wait_stream(stream)

    monkeypatch.setattr(model_runner, "graph_capture", capture_stream)
    compiled = cuda_graph.CUDAGraphWrapper(
        lambda ids, pos: (ids.float() + pos.float())[:, None].repeat(1, 2),
        runner.config,
        CUDAGraphMode.PIECEWISE,
        cuda_graph.CUDAGraphOptions(weak_ref_output=False),
    )
    compiled.compute_logits = lambda hidden: hidden + 1
    runner.model = compiled
    try:
        if not supported:
            with pytest.raises(AssertionError, match="no cudagraph capture sizes"):
                runner.capture_cudagraph()
            return
        _, sizes, _ = runner.capture_cudagraph()
        assert sizes == [] and runner.capture_sizes_np.size == 0
        assert runner.capture_sizes_by_q[6].size == 0 and not runner.graphs
        assert runner._piecewise_sorted_tokens == [4]
        assert len(compiled.concrete_cudagraph_entries) == 1
        context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        context.batch_descriptor = cuda_graph.BatchDescriptor(4)
        ids[:4].fill_(7)
        torch.testing.assert_close(
            compiled(ids[:4], positions[:4])[:, 0], positions[:4].float() + 7
        )
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        ids[:6].fill_(9)
        torch.testing.assert_close(
            compiled(ids[:6], positions[:6])[:, 0], positions[:6].float() + 9
        )
        assert len(compiled.concrete_cudagraph_entries) == 1
    finally:
        compiled.release_graphs()


@pytest.mark.parametrize("max_tokens", [1536, 16384])
def test_default_2k_prefill_preserves_full_and_draft_decode_ladder(max_tokens):
    config = SimpleNamespace(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
        )
    )
    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.max_num_batched_tokens = max_tokens
    runner.config.capture_sizes = Config.capture_sizes.fget(config)
    assert len(runner.config.capture_sizes) == 83
    baseline = runner._cudagraph_capture_plan([1, 2, 4, 6])
    runner.config.prefill_sizes = Config.prefill_capture_sizes.fget(config)
    assert len(runner.config.prefill_sizes) == 91
    extended = runner._cudagraph_capture_plan([1, 2, 4, 6])
    assert extended[1:] == baseline[1:]  # Draft uses the union of these BS lists.
    expected = list(range(1152, min(max_tokens, 2048) + 1, 128))
    assert extended[0][2] == expected[::-1] + baseline[0][2]


@pytest.mark.parametrize(
    "max_seqs,max_tokens,ceiling",
    [(256, 16384, 1024), (7, 16384, 1024), (256, 48, 1024), (256, 16384, 16)],
)
def test_hybrid_small_decode_batches_respect_capacity_and_dispatch_exactly(
    max_seqs, max_tokens, ceiling
):
    import numpy as np

    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = [1, 2, 4, 8, ceiling]
    runner.config.max_num_seqs = max_seqs
    runner.config.max_num_batched_tokens = max_tokens
    plan = runner._cudagraph_capture_plan([2, 4, 6])
    by_q = {
        q: np.asarray(sizes[::-1])
        for mode, q, sizes in plan
        if mode == CUDAGraphMode.FULL
    }
    union = np.asarray(sorted({int(bs) for sizes in by_q.values() for bs in sizes}))
    for q, sizes in by_q.items():
        bound = min(max_seqs, max_tokens // 6, min(ceiling, max_tokens) // q)
        assert all(0 < bs <= bound for bs in sizes)
        for bs in range(1, min(32, bound) + 1):
            batch = SimpleNamespace(
                total_tokens_num_prefill=0,
                total_tokens_num=bs * q,
                total_seqs_num=bs,
                is_dummy_run=False,
            )
            mode = ForwardMode.decide(
                batch=batch,
                dp_size=1,
                dp_group=None,
                enforce_eager=False,
                step_needs_eager=False,
                capture_sizes=union,
                capture_sizes_by_q=by_q,
                captured_tokens=None,
                is_block_drafter=True,
                tbo_on=False,
                local_tbo=(False, False, 0, 0),
                max_seqlen_q=q,
            )
            assert mode.use_cudagraph and not mode.piecewise_captured
            assert mode.running_bs == bs and mode.running_tokens == bs * q


@pytest.mark.parametrize(
    "mode,from_cli",
    [
        ("FULL", True),
        ("PIECEWISE", False),
        ("AF_PIECEWISE", True),
        ("FULL_AND_PIECEWISE", False),
        ("FULL_AND_PIECEWISE", True),
    ],
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


@pytest.mark.parametrize("sizes", [[1024], [8, 128, 704]])
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
    config.capture_sizes = sizes
    assert Config.prefill_capture_sizes.fget(config) == sizes


def test_python_hybrid_capture_ceiling_override_is_preserved():
    config = SimpleNamespace(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE, cuda_graph_sizes=[256]
        )
    )
    assert Config.capture_sizes.fget(config) == [1, 2, 4] + list(range(8, 257, 8))
    config.capture_sizes = Config.capture_sizes.fget(config)
    assert Config.prefill_capture_sizes.fget(config) == config.capture_sizes


def test_independent_prefill_override_does_not_change_decode():
    config = SimpleNamespace(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[8, 128, 512],
            cudagraph_prefill_capture_sizes=[128, 512, 1024, 2048],
        )
    )
    assert Config.capture_sizes.fget(config) == [8, 128, 512]
    assert Config.prefill_capture_sizes.fget(config) == [128, 512, 1024, 2048]


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

    outputs = cuda_graph.StableOutputs(capacity=8, token_dim=1)
    layer = SimpleNamespace(
        layer_name="hybrid.layer",
        attn=Core(),
        af_piecewise=True,
        af_zero_copy_hidden=True,
        piecewise_outputs=outputs,
    )
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
            cudagraph_key=(4,),
            engram_embeddings={},
            image_mask=None,
        ),
    )
    from atom.utils import forward_context

    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
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
    runner = next(iter(layer._piecewise_core_runners.values()))
    assert not runner._graphs
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
    assert len(runner._graphs) == 1
    runner.release_graphs()
    for piece in pieces:
        piece.release_graphs()


@pytest.mark.parametrize("backend", ["base", "v4", "mha", "mla"])
def test_unsupported_token_prefill_keeps_hybrid_full_decode_ladder(backend):
    from atom.model_ops.attentions.aiter_attention import AiterAttentionMetadataBuilder
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.attentions.backends import AttentionMetadataBuilder
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    builders = {
        "base": AttentionMetadataBuilder,
        "v4": DeepseekV4AttentionMetadataBuilder,
        "mha": AiterAttentionMetadataBuilder,
        "mla": AiterMLAMetadataBuilder,
    }
    runner = _runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.config.capture_sizes = [6, 192, 608, 768]
    supported = runner._cudagraph_capture_plan([2, 6])
    runner.attn_metadata_builder = builders[backend]
    fallback = runner._cudagraph_capture_plan([2, 6])
    assert fallback == [entry for entry in supported if entry[0] == CUDAGraphMode.FULL]
    assert 102 in fallback[1][2]
    runner.capture_sizes = [8, 2, 1]
    runner.config.compilation_config.cudagraph_mode = CUDAGraphMode.AF_PIECEWISE
    assert runner._cudagraph_capture_plan([6]) == [
        (CUDAGraphMode.PIECEWISE, 6, [8, 2, 1])
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_capacity_results_keep_contiguous_planes_across_graph_shapes():
    from atom.utils.cuda_graph import StableOutputs

    outputs = StableOutputs()
    recorded = {}
    source = torch.zeros(48, device="cuda")
    for width in (8, 3, 5):
        shape = (2, width * 3)
        view = outputs.empty(
            "indices", shape, capacity=48, dtype=torch.float32, device="cuda"
        )
        assert view.is_contiguous()
        result = torch.empty_like(view)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            view.copy_(source[: view.numel()].view(shape))
            result.copy_(view + 1)
        recorded[width] = graph, view, result
    assert len({view.data_ptr() for _, view, _ in recorded.values()}) == 1
    for i, width in enumerate((3, 8, 5, 3, 8), 1):
        source.copy_(torch.arange(48, device="cuda") + i * 100)
        graph, view, result = recorded[width]
        graph.replay()
        torch.testing.assert_close(result, source[: view.numel()].view_as(view) + 1)
    assert len(outputs._slots) == 1
    with pytest.raises(ValueError, match="capacity"):
        outputs.empty("indices", (49,), capacity=48, dtype=torch.float32, device="cuda")


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.AF_PIECEWISE,
    ],
)
def test_memory_estimate_only_reserves_supported_hybrid_prefill(
    monkeypatch, supported, mode
):
    runner = _runner(mode)
    runner.config.enforce_eager = False
    runner.config.capture_sizes = [4, 8]
    runner.config.compilation_config.cudagraph_capture_sizes = [4, 8]
    runner.config.parallel_config = SimpleNamespace(data_parallel_size=1)
    runner.config.gpu_memory_utilization = 0.9
    runner.attn_metadata_builder.supports_piecewise_prefill = supported
    runner._piecewise_per_token_bytes = lambda: 10
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 1000,
            "allocated_bytes.all.current": 500,
        },
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_000_000, 1_000_000))
    expected = {
        CUDAGraphMode.FULL: 100,
        CUDAGraphMode.FULL_AND_PIECEWISE: 220 if supported else 100,
        CUDAGraphMode.PIECEWISE: 120,
        CUDAGraphMode.AF_PIECEWISE: 120,
    }[mode]
    assert runner._estimate_cudagraph_overhead() == expected
