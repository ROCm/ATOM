# SPDX-License-Identifier: MIT
"""V2 preparation with real CPU metadata, cache state and publication owners."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.plugin.vllm import deepseek_v41_bridge as bridge
from atom.plugin.vllm.v41_core_overrides import VllmDeepseekV41MetadataBuilder
from atom.utils.h2d import PublicationError
from tests.attentions.deepseek_v41.helpers import metadata_buffers


def _builder(*, speculative_tokens=2, max_tokens=12):
    builder = VllmDeepseekV41MetadataBuilder.__new__(VllmDeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(
        2, ((0, 1), (1, 2)), 32, 4, 512, 32, speculative_tokens=speculative_tokens
    )
    builder.cache = PagedAttentionCache(
        builder.geometry, 16, 4, "cpu", max_tokens=max_tokens
    )
    builder.block_size, builder.device = 32, "cpu"
    builder.max_bs, builder.max_num_batched_tokens = 4, max_tokens
    buffers = metadata_buffers(4, max_tokens, 4, "cpu", builder.geometry)
    for name in ("positions", "batch_id_per_q_token"):
        buffers[name].publication_group = "v41_step"
    buffers["cu_seqlens_q"].publication_group = "early"
    buffers["block_tables"].publication_group = "prefill"
    builder.model_runner = SimpleNamespace(
        forward_vars=buffers, v41_graphs_enabled=False
    )
    builder.config = SimpleNamespace(
        engram_layer_ids=(0,),
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    builder.engram = None
    return builder


@pytest.fixture
def builder():
    return _builder()


@pytest.mark.parametrize("speculative_tokens,width", [(0, 1), (5, 6)])
def test_profile_full_builder_prepares_scratch_before_forward(
    monkeypatch, speculative_tokens, width
):
    """V2 calls the factory twice in mode NONE before ordinary FULL capture."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import override_forward_context

    from atom.model_ops.engram.device.staging import EngramStagedRows
    from atom.plugin.vllm import req_id_passthrough_patch as passthrough
    from atom.plugin.vllm.deepseek_v4_prefix_patch import _mark_v4_proxy_cache_mode
    from atom.utils.forward_context import get_forward_context

    builder = _builder(speculative_tokens=speculative_tokens, max_tokens=24)
    builder.cache = None
    builder.score_workspace = None
    builder.model_runner.v41_graphs_enabled = True
    proxy = bridge.AtomDeepseekV41ProxyAttention()
    model = SimpleNamespace()
    proxy._atom_v41_model, proxy._atom_v41_builder = model, builder
    # Nonempty but deliberately too small to be a serving pool: the marker,
    # not an empty tensor, must keep bind from carving it.
    proxy.kv_cache = torch.full((1,), 73, dtype=torch.uint8)
    layer = bridge.ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME
    sfc = {layer: proxy}
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(static_forward_context=sfc),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=speculative_tokens)
            if speculative_tokens
            else None
        ),
    )
    factory = bridge.AtomDeepseekV41ProxyMetadataBuilder.__new__(
        bridge.AtomDeepseekV41ProxyMetadataBuilder
    )
    factory.vllm_config, factory.device = config, "cpu"
    _mark_v4_proxy_cache_mode(sfc, True)
    assert not bridge.bind_deepseek_v41_proxy_cache(model, builder, config)
    batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens_after_padding=4 * width,
        input_ids=torch.zeros(4 * width, dtype=torch.int32),
        req_ids=["profile-0", "profile-1"],
        idx_mapping_np=np.array([2, 3], dtype=np.int32),
    )
    monkeypatch.setattr(passthrough, "get_current_input_batch", lambda: batch)
    monkeypatch.setattr(
        passthrough, "get_current_cudagraph_mode", lambda: CUDAGraphMode.NONE
    )
    monkeypatch.setattr(bridge, "_V41_PENDING_SPECULATIVE_STEP", None)
    monkeypatch.setattr(bridge, "_V41_OPEN_ENGRAM_FORK", None)
    fallback = Mock(side_effect=AssertionError("FULL called host staging"))
    monkeypatch.setattr(bridge, "v41_stage_step", fallback)
    staging = SimpleNamespace(
        host=SimpleNamespace(buffers={}), start=Mock(), join=Mock()
    )
    original_inputs = builder.prepare_model_inputs

    def prepare_inputs(input_ids, metadata):
        original_inputs(input_ids, metadata)
        metadata.engram_embeddings = EngramStagedRows(staging, input_ids.numel())

    inputs_spy = Mock(side_effect=prepare_inputs)
    prepare_spy = Mock(wraps=builder._prepare)
    monkeypatch.setattr(builder, "prepare_model_inputs", inputs_spy)
    monkeypatch.setattr(builder, "_prepare", prepare_spy)
    scratch = None
    for count in (2, 2, 1):
        batch.num_reqs = count
        common = SimpleNamespace(
            num_reqs=count,
            max_query_len=width,
            query_start_loc_cpu=torch.arange(count + 1, dtype=torch.int32) * width,
            seq_lens=torch.full((count,), width, dtype=torch.int32),
            block_table_tensor=torch.full((count, 1), 999, dtype=torch.int32),
        )
        returned = factory.build_for_cudagraph_capture(common)
        assert returned is common and not common.atom_v41_snapshot.full_graph
        prepared = getattr(returned, "atom_v41_prepared", None)
        assert prepared is not None, "profiling FULL skipped graph-external staging"
        metadata, step = prepared.metadata, prepared.metadata.step
        if scratch is None:
            scratch = metadata.cache
        assert metadata.cache is scratch is builder._capture_dummy
        assert builder.cache is None and not hasattr(model, "_atom_v41_proxy_cache_ptr")
        assert proxy.kv_cache.tolist() == [73]
        assert prepared.running_bs == 4 and prepared.running_tokens == 4 * width
        assert step.scheduled_bs == count and step.max_q_len == width
        assert [span.slot for span in step.requests] == list(range(count))
        assert step.tentative == (width > 1) and step.decode
        if width > 1:
            assert all(
                span.position == builder.geometry.window_size for span in step.requests
            )
        assert step.block_tables.eq(0).all()
        assert metadata.dummy and metadata.staged_outside_forward
        assert scratch.pending is None and not bridge.v41_speculative_step_is_pending()
        for ratio, plan in step.plans.items():
            assert plan.compress_plan_gpu.shape[0] == 4 * -(-width // ratio)
            assert (plan.compress_plan_gpu[plan.num_compress :, 2] == -1).all()
        before = (
            prepare_spy.call_count,
            inputs_spy.call_count,
            staging.start.call_count,
        )
        context = SimpleNamespace(attn_metadata={layer: returned})
        with (
            override_forward_context(context),
            bridge.atom_deepseek_v41_forward_context(
                atom_config=builder.config,
                builder=builder,
                input_ids=batch.input_ids,
                positions=None,
                vllm_config=config,
                force_dummy=True,
            ) as positions,
        ):
            assert get_forward_context().attn_metadata is metadata
            assert positions.data_ptr() == prepared.positions.data_ptr()
        assert before == (
            prepare_spy.call_count,
            inputs_spy.call_count,
            staging.start.call_count,
        )
        assert staging.join.call_count == staging.start.call_count
    fallback.assert_not_called()
    assert prepare_spy.call_count == inputs_spy.call_count == 3

    # A subsequent serving allocation must replace the cache used by newly
    # captured graphs, never promote the profiling scratch to serving storage.
    config.scheduler_config.max_num_batched_tokens = 24
    config.cache_config = SimpleNamespace(num_gpu_blocks=8, cache_dtype="auto")
    monkeypatch.setattr(bridge, "v41_proxy_geometry", lambda _: builder.geometry)
    size = builder.geometry.paged_extents(8)[1] + 8 * builder.geometry.state_bytes
    proxy.kv_cache = torch.zeros(size, dtype=torch.uint8).reshape(1, 1, 1, -1)
    _mark_v4_proxy_cache_mode(sfc, False)
    common.block_table_tensor.zero_()
    serving = factory.build_for_cudagraph_capture(common).atom_v41_prepared
    assert serving.metadata.cache is builder.cache
    assert builder.cache is not scratch and builder._capture_dummy is scratch
    assert model._atom_v41_proxy_cache_ptr == proxy.kv_cache.data_ptr()
    assert [span.slot for span in serving.metadata.step.requests] == [4]
    assert serving.metadata.step.tentative == (width > 1)
    assert (
        builder.cache.pending is None and not bridge.v41_speculative_step_is_pending()
    )


def snapshot(ids=("A", "B"), slots=(2, 0), start=0, width=1):
    count = len(ids)
    return SimpleNamespace(
        num_reqs=count,
        num_reqs_padded=4,
        max_query_len=width,
        full_graph=True,
        req_ids=list(ids),
        state_slots=np.asarray(slots, dtype=np.int32),
        query_lens=np.full(count, width, dtype=np.int32),
        num_computed=np.full(count, start, dtype=np.int64),
        ends=np.full(count, start + width, dtype=np.int64),
        block_rows=[(int(slot) + 1,) for slot in slots],
        total_tokens=count * width,
    )


def test_real_preparation_reuses_metadata_and_publishes_cursor_once(builder):
    pointers = {
        name: buf.gpu.data_ptr()
        for name, buf in builder.model_runner.forward_vars.items()
    }
    for step_snapshot in (
        snapshot(),
        snapshot(start=1),
        snapshot(("A",), (2,), start=2),
    ):
        batch = bridge._v41_scheduled_batch(
            step_snapshot, slots=step_snapshot.state_slots
        )
        metadata, positions = builder._prepare(batch, 4, 4, max_q_len=1)
        builder.prepare_model_inputs(torch.zeros(4, dtype=torch.int32), metadata)
        count = step_snapshot.num_reqs
        assert metadata.step.slots.tolist() == list(step_snapshot.state_slots) + [0] * (
            4 - count
        )
        assert positions[:count].tolist() == step_snapshot.num_computed.tolist()
        assert (
            builder.cache.cursor[step_snapshot.state_slots, 0].tolist()
            == step_snapshot.ends.tolist()
        )
        for plan in metadata.step.plans.values():
            assert plan.compress_plan_gpu.shape[0] == 4
            assert (plan.compress_plan_gpu[plan.num_compress :, 2] == -1).all()
        assert builder.model_runner.h2d_owner._state == "sealed"
        assert builder.cache._vllm_h2d_owner._state == "sealed"
    assert pointers == {
        name: buf.gpu.data_ptr()
        for name, buf in builder.model_runner.forward_vars.items()
    }


def test_tentative_host_prefixes_share_epoch_until_sampler_commit(builder):
    initial = snapshot()
    batch = bridge._v41_scheduled_batch(initial, slots=initial.state_slots)
    metadata, _ = builder._prepare(batch, 4, 4, max_q_len=1)
    builder.prepare_model_inputs(torch.zeros(4, dtype=torch.int32), metadata)
    verify = snapshot(start=1, width=3)
    metadata, _ = builder._prepare(
        bridge._v41_scheduled_batch(verify, slots=verify.state_slots),
        4,
        12,
        max_q_len=3,
        tentative=True,
    )
    builder.prepare_model_inputs(torch.zeros(12, dtype=torch.int32), metadata)
    owner = builder.cache._vllm_h2d_owner
    epoch = owner.epoch
    builder.commit_speculative_state(metadata, torch.tensor([0, 4]))
    assert owner.epoch == epoch and owner._state == "sealed"
    assert builder.cache.pending is None
    assert builder.cache.cursor[[2, 0], 0].tolist() == [2, 3]


def test_partial_preparation_failure_poison_prevents_reuse(builder):
    bad = snapshot()
    bad.block_rows = [(), ()]
    batch = bridge._v41_scheduled_batch(bad, slots=bad.state_slots)
    with pytest.raises(ValueError, match="PAGE table"):
        builder._prepare(batch, 4, 4, max_q_len=1)
    assert builder.model_runner.h2d_owner._state == "failed"
    with pytest.raises(PublicationError, match="failed"):
        builder._prepare(batch, 4, 4, max_q_len=1)


def test_outside_forward_and_fallback_both_publish_full_capacity(builder, monkeypatch):
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    inputs = torch.zeros(4, dtype=torch.int32)
    monkeypatch.setattr(bridge, "_v41_step_input_ids", lambda _width: inputs)
    monkeypatch.setattr(bridge, "_v41_staged_running_tokens", lambda: 4)
    prepared = bridge._v41_stage_outside_forward(
        builder, None, config, snapshot(), capturing=False
    )
    assert prepared.running_bs == 4
    assert prepared.metadata.step.scheduled_bs == 2
    assert prepared.metadata.staged_outside_forward
    common = SimpleNamespace(atom_v41_snapshot=snapshot(start=1))
    monkeypatch.setattr(
        bridge,
        "get_deepseek_v41_proxy_metadata_from_vllm_context",
        lambda _name: common,
    )
    monkeypatch.setattr(bridge, "_v41_publish_metadata", lambda _metadata: None)
    metadata, _, running_bs, _, _ = bridge.v41_stage_step(
        builder, inputs, "proxy", False, config
    )
    assert running_bs == 4 and metadata.step.scheduled_bs == 2
    assert metadata.step.plans[2].compress_plan_gpu.shape[0] == 4
    assert builder.cache.cursor[[2, 0], 0].tolist() == [2, 2]


def _scratch_storage(cache):
    tensors = [cache.backing, cache.cursor, cache.tentative_staging.gpu]
    tensors.extend(
        tensor for planes in cache.index_planes.values() for tensor in planes
    )
    tensors.extend(tensor for pair in cache.indptr_buffers.values() for tensor in pair)
    return tuple(
        (tensor.data_ptr(), tensor.shape, tensor.stride()) for tensor in tensors
    )


def test_capture_scratch_covers_split_dummy_pages_and_keeps_addresses():
    """Artificial helper ordering, not V2's descending capture sequence."""
    builder = _builder(max_tokens=96)
    builder.cache = None
    builder.score_workspace = None
    builder.model_runner.v41_graphs_enabled = True
    scratch = builder._capture_dummy_cache(1, 1, 1)
    pointers = _scratch_storage(scratch)
    batch = bridge._v41_dummy_batch(96, max_req_tokens=64, max_reqs=4)
    assert batch.num_scheduled_tokens == (48, 48)
    for lengths, expected_rows in [
        ((48, 48), [[0, 1], [2, 3]]),
        ((32, 64), [[0], [1, 2]]),
        ((1, 1, 1, 1), [[0], [1], [2], [3]]),
        ((1,), [[0]]),
    ]:
        batch.num_scheduled_tokens = batch.context_lens = lengths
        batch.req_ids = tuple(range(len(lengths)))
        batch.total_seqs_num, batch.total_tokens_num = len(lengths), sum(lengths)
        metadata, _ = builder._prepare(batch, 4, sum(lengths), engram_live=False)
        assert metadata.cache is scratch and builder.cache is None
        assert [row.tolist() for row in metadata.block_table_rows] == expected_rows
        assert scratch.num_pages > max(page for row in expected_rows for page in row)
        assert _scratch_storage(scratch) == pointers
    assert builder._capture_dummy_cache(1, 1, 1) is scratch


@pytest.mark.parametrize("dimension", ["num_pages", "num_slots", "max_tokens"])
def test_capture_scratch_rejects_oversize_without_replacing_storage(builder, dimension):
    builder.cache = None
    builder.score_workspace = None
    builder.model_runner.v41_graphs_enabled = True
    scratch = builder._capture_dummy_cache(1, 1, 1)
    pointers = _scratch_storage(scratch)
    current_step = scratch.current_step = object()
    request = {
        "num_pages": scratch.num_pages,
        "num_slots": scratch.num_slots,
        "max_tokens": scratch.max_tokens,
    }
    request[dimension] += 1
    with pytest.raises(RuntimeError, match="capture scratch capacity"):
        builder._capture_dummy_cache(*request.values())
    assert builder._capture_dummy is scratch
    assert scratch.current_step is current_step
    assert _scratch_storage(scratch) == pointers
    assert builder._capture_dummy_cache(1, 1, 1) is scratch


def test_eager_scratch_has_exact_independent_storage(builder):
    builder.score_workspace = None
    first = builder._capture_dummy_cache(1, 1, 1)
    second = builder._capture_dummy_cache(3, 2, 6)
    assert first is not second and first.backing.data_ptr() != second.backing.data_ptr()
    assert (first.num_pages, first.num_slots, first.max_tokens) == (1, 1, 1)
    assert (second.num_pages, second.num_slots, second.max_tokens) == (3, 2, 6)
    assert not hasattr(builder, "_capture_dummy")
    builder.model_runner.v41_graphs_enabled = True
    captured = builder._capture_dummy_cache(1, 1, 1)
    pointers = _scratch_storage(captured)
    builder.model_runner.v41_graphs_enabled = False
    eager = builder._capture_dummy_cache(1, 1, 1)
    assert eager.backing.data_ptr() != captured.backing.data_ptr()
    assert builder._capture_dummy is captured
    assert _scratch_storage(captured) == pointers


@pytest.mark.parametrize("graphs_enabled", [False, True])
def test_capture_scratch_lifetime_follows_effective_mode(
    builder, monkeypatch, graphs_enabled
):
    from atom.plugin.vllm import v41_core_overrides as overrides

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    builder.model_runner.v41_graphs_enabled = graphs_enabled
    builder.score_workspace = object()
    allocated = []

    def allocate(geometry, pages, slots, device, *, max_tokens, workspace):
        cache = SimpleNamespace(num_pages=pages, num_slots=slots, max_tokens=max_tokens)
        allocated.append(cache)
        return cache

    monkeypatch.setattr(overrides, "PagedAttentionCache", allocate)
    first = builder._capture_dummy_cache(1, 1, 1)
    second = builder._capture_dummy_cache(1, 2, 6)
    if graphs_enabled:
        assert first is second and len(allocated) == 1
        assert (first.num_pages, first.num_slots, first.max_tokens) == (4, 4, 12)
        # Changing mode must not drop storage that an existing graph references.
        builder.model_runner.v41_graphs_enabled = False
        assert builder._capture_dummy_cache(1, 1, 1) is not first
        assert builder._capture_dummy is first
    else:
        assert first is not second and len(allocated) == 2
        assert not hasattr(builder, "_capture_dummy")
        assert (first.num_slots, first.max_tokens) == (1, 1)
        assert (second.num_slots, second.max_tokens) == (2, 6)


def test_outside_staging_is_not_repeated_by_forward_boundaries(builder, monkeypatch):
    from atom.model_ops.engram.device.staging import EngramStagedRows

    inputs = torch.zeros(4, dtype=torch.int32)
    staging = SimpleNamespace(
        host=SimpleNamespace(buffers={}), start=Mock(), join=Mock()
    )
    rows = EngramStagedRows(staging, 4)
    original = builder.prepare_model_inputs

    def prepare(input_ids, metadata):
        original(input_ids, metadata)
        metadata.engram_embeddings = rows

    monkeypatch.setattr(builder, "prepare_model_inputs", prepare)
    monkeypatch.setattr(bridge, "_v41_step_input_ids", lambda _width: inputs)
    monkeypatch.setattr(bridge, "_v41_staged_running_tokens", lambda: 4)
    monkeypatch.setattr(bridge, "_v41_join_covers_the_layers", lambda **_: False)
    prepared = bridge._v41_stage_outside_forward(
        builder,
        None,
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4)),
        snapshot(),
        capturing=False,
    )
    staging.start.assert_called_once_with(4)
    staging.join.assert_called_once_with()

    # Execute the actual guarded function bodies without invoking custom-op
    # registration, compilation, or the heavyweight model import on this CPU test.
    source = Path(__file__).resolve().parents[2] / "atom/models/deepseek_v41/runtime.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("v41_begin_forward", "v41_end_forward")
    ]
    assert len(functions) == 2
    for node in functions:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "EngramStagedRows": EngramStagedRows,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata=prepared.metadata),
    }
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    hidden = torch.zeros(1, 4, 2)
    namespace["v41_begin_forward"](hidden)
    namespace["v41_end_forward"](hidden)
    staging.start.assert_called_once_with(4)
    staging.join.assert_called_once_with()
