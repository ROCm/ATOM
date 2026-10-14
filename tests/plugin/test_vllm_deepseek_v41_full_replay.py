# SPDX-License-Identifier: MIT
"""GPU-only V4.1 metadata publication -> real compression graph regressions.

Run this file explicitly in a ROCm environment, with GPU visibility/resource
ownership established by the caller. It uses only the current device, never
selects a physical GPU, and needs neither model weights nor a vLLM engine.

The capture contains PagedAttentionCache.compress: the real fused pooling,
RMSNorm/RoPE/PAGE scatter and compressor-state writer, plus FP4 packing/scatter
for a packed pool. Inputs start at the projected values/scores boundary; this
is not coverage of projection GEMMs, index attention, or a whole-model FULL
vLLM capture. All host metadata publication happens OUTSIDE torch.cuda.graph.
The profile-to-serving cases also run real Engram hash/cursor/UVA staging through
proxy/factory/forward-context preparation. Only V2 orchestration is lightweight;
this does not invoke the upstream runner's full capture manager.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip(
        "Actual ROCm graph capture/replay requires a GPU", allow_module_level=True
    )

pytest.importorskip("aiter", reason="V4.1 compression imports AITER-backed operators")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.plugin.vllm.v41_core_overrides import (
    VllmDeepseekV41MetadataBuilder,
    VllmEngramStaging,
    _v41_publication_owner,
)
from atom.utils import CpuGpuBuffer
from tests.attentions.deepseek_v41.helpers import metadata_buffers

_CAPACITY = 4
_HEAD_DIM = 512
# Neither batch indices nor STATE slots equal PAGE ids. PAGE 0 and PAGE 5
# remain guards, and shrinking to the last two requests leaves slot 0 idle.
_SLOTS = (2, 0, 3, 1)
_PAGES = (3, 1, 4, 2)


def _builder(
    ratio,
    packed,
    *,
    max_tokens=_CAPACITY,
    num_slots=_CAPACITY,
    history_size=3,
    speculative_tokens=0,
    allocate_cache=True,
):
    """Same hand-assembled builder as the CPU publication tests, on real HBM."""
    geometry = V41PoolGeometry(
        1,
        ((0, ratio),),
        32,
        4,
        _HEAD_DIM,
        32,
        history_size=history_size,
        packed=packed,
        speculative_tokens=speculative_tokens,
        layer_ratios=(ratio,),
        index_topk=4,
    )
    builder = VllmDeepseekV41MetadataBuilder.__new__(VllmDeepseekV41MetadataBuilder)
    builder.geometry = geometry
    builder.cache = (
        PagedAttentionCache(
            geometry, _CAPACITY + 2, num_slots, "cuda", max_tokens=max_tokens
        )
        if allocate_cache
        else None
    )
    builder.score_workspace = None
    builder.block_size, builder.device = 32, "cuda"
    builder.max_bs, builder.max_num_batched_tokens = _CAPACITY, max_tokens
    buffers = metadata_buffers(_CAPACITY, max_tokens, 1, "cuda", geometry)
    for name in ("positions", "batch_id_per_q_token"):
        buffers[name].publication_group = "v41_step"
    buffers["cu_seqlens_q"].publication_group = "early"
    buffers["block_tables"].publication_group = "prefill"
    assert all(buffer.cpu.is_pinned() for buffer in buffers.values())
    builder.model_runner = SimpleNamespace(
        forward_vars=buffers, v41_graphs_enabled=True
    )
    builder.engram = None
    return builder


def _prepare(builder, position, slots=_SLOTS, pages=_PAGES, *, width=1):
    """Publish through the real plugin override, never patch the plan tensors."""
    assert len(slots) == len(pages)
    count = len(slots)
    batch = SimpleNamespace(
        is_dummy_run=False,
        is_final_chunk=None,
        replaying={},
        req_ids=tuple(f"request-{slot}" for slot in slots),
        state_slots_committed=list(slots),
        num_scheduled_tokens=np.full(count, width, dtype=np.int32),
        context_lens=np.broadcast_to(np.asarray(position) + width, (count,)).copy(),
        block_tables=[(page,) for page in pages],
        total_seqs_num=count,
        total_tokens_num=count * width,
    )
    metadata, _ = builder._prepare(
        batch,
        _CAPACITY,
        builder.max_num_batched_tokens,
        max_q_len=width,
        tentative=width > 1,
        engram_live=False,
    )
    assert builder.model_runner.h2d_owner._state == "sealed"
    return metadata.step


def _parameters(ratio):
    # These are the exact fields consumed by compress_batch, not a substitute
    # compressor implementation. Leave projection GEMMs / distributed setup out
    # of this kernel-boundary fixture. D=512 matches the existing V4.1 tests.
    compressor = SimpleNamespace(
        ratio=ratio,
        ape=torch.zeros(ratio, _HEAD_DIM, dtype=torch.float32, device="cuda"),
        norm=SimpleNamespace(
            weight=torch.linspace(0.7, 1.3, _HEAD_DIM, device="cuda"), eps=1e-6
        ),
    )
    with torch.device("cuda"):
        rope = RotaryEmbedding(64, 32, base=10000)
    return compressor, rope


def _projections(ratio, phase, tokens=_CAPACITY):
    # The two steps are not scalar multiples: RMSNorm must not erase the
    # difference between a proper ratio-2 pool and a missing historical tail.
    columns = torch.arange(tokens * _HEAD_DIM, dtype=torch.float32, device="cuda").view(
        1, tokens, _HEAD_DIM
    )
    values = torch.sin(columns * 0.013 + phase * 0.7)
    values += 0.5 * torch.cos(columns * 0.031 - phase * 0.4)
    if ratio == 1:
        return values.to(torch.bfloat16), None
    scores = 0.35 * torch.cos(columns * 0.021 + phase)
    return values, scores


def _reset_cache(cache, *, page_byte=0x5A, state_value=0.0):
    cache.backing.zero_()
    cache.page_bytes.fill_(page_byte)
    for state in cache.compress_state(0):
        state.fill_(state_value)


def _compress(builder, compressor, values, scores, step, rope):
    return builder.cache.compress(0, compressor, values, scores, step, rope)


def _capture_at_position_zero(builder, compressor, values, scores, rope, stream):
    # Warm the ACTIVE specialization too. Otherwise a zero-boundary capture
    # could hide first-use JIT work until after capture. Warmup is not capture.
    warm = _prepare(builder, 1)
    _compress(builder, compressor, values, scores, warm, rope)
    stream.synchronize()
    _reset_cache(builder.cache)
    captured_step = _prepare(builder, 0)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        latent = _compress(builder, compressor, values, scores, captured_step, rope)
    return graph, latent, captured_step


def _storage(step, ratio):
    plan = step.plans[ratio]
    tensors = (
        plan.compress_plan_gpu,
        plan.write_plan_gpu,
        plan.key_rope_positions_gpu,
        step.slots,
        step.block_tables,
        step.positions,
        step.cu_seqlens_q,
        step.batch_ids,
    )
    return tuple((t.data_ptr(), tuple(t.shape), t.stride()) for t in tensors)


def _assert_plan(step, ratio, boundaries, writes):
    plan = step.plans[ratio]
    assert plan.num_compress == boundaries and plan.num_write == writes
    assert plan.compress_plan_gpu.shape == (_CAPACITY, 4)
    assert plan.write_plan_gpu.shape == (_CAPACITY, 4)
    assert torch.all(plan.compress_plan_gpu[boundaries:] == -1)
    assert torch.all(plan.write_plan_gpu[writes:] == -1)


def _main_rows(cache):
    return cache.pages.view("main_0")[0]


def _assert_only_rows_changed(cache, before, pages, row):
    """Require real PAGE writes and check EVERY other main row, including guards."""
    actual = _main_rows(cache)
    changed = torch.zeros(actual.shape[:2], dtype=torch.bool, device=actual.device)
    for page in pages:
        changed[page, row] = True
        assert not torch.equal(
            actual[page, row], before[page, row]
        ), f"compressed PAGE {page}, row {row} was not written on replay"
    torch.testing.assert_close(actual[~changed], before[~changed], rtol=0, atol=0)


def _assert_matches_eager(actual, expected):
    torch.testing.assert_close(actual.page_bytes, expected.page_bytes, rtol=0, atol=0)
    for observed, reference in zip(
        actual.compress_state(0), expected.compress_state(0)
    ):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True], ids=["bf16-page", "fp4-page"])
def test_ratio2_zero_boundary_capture_replays_a_real_compressed_page(packed):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        builder, eager = _builder(2, packed), _builder(2, packed)
        compressor, rope = _parameters(2)
        values, scores = _projections(2, 0)
        graph, latent, captured_step = _capture_at_position_zero(
            builder, compressor, values, scores, rope, stream
        )
        _assert_plan(captured_step, 2, boundaries=0, writes=_CAPACITY)
        captured_storage = _storage(captured_step, 2)
        before = _main_rows(builder.cache).clone()
        before_pages = builder.cache.page_bytes.clone()
        # Capture itself is not execution. This first replay really writes the
        # position-0 ring, while the compression grid has only sentinels.
        graph.replay()
        stream.synchronize()
        torch.testing.assert_close(
            builder.cache.page_bytes, before_pages, rtol=0, atol=0
        )
        for state, projected in zip(builder.cache.compress_state(0), (values, scores)):
            torch.testing.assert_close(
                state[list(_SLOTS), 0], projected[0], rtol=0, atol=0
            )

        # Independent eager history: do not seed it from the replay's state.
        _reset_cache(eager.cache)
        _compress(eager, compressor, values, scores, _prepare(eager, 0), rope)
        next_values, next_scores = _projections(2, 1)
        values.copy_(next_values)
        scores.copy_(next_scores)
        replay_step = _prepare(builder, 1)
        _assert_plan(replay_step, 2, boundaries=_CAPACITY, writes=_CAPACITY)
        assert _storage(replay_step, 2) == captured_storage
        assert captured_step.plans[2].num_compress == 0  # host object stays frozen
        expected_latent = _compress(
            eager, compressor, values, scores, _prepare(eager, 1), rope
        )
        graph.replay()
        stream.synchronize()
        _assert_only_rows_changed(builder.cache, before, _PAGES, row=0)
        _assert_matches_eager(builder.cache, eager.cache)
        assert latent is not None and expected_latent is not None
        assert torch.isfinite(latent).all()
        torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True], ids=["bf16-page", "fp4-page"])
def test_ratio1_shrunk_replay_sentinels_stop_page_and_state_tail_writes(packed):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        builder, eager = _builder(1, packed), _builder(1, packed)
        compressor, rope = _parameters(1)
        values, scores = _projections(1, 0)
        graph, latent, captured_step = _capture_at_position_zero(
            builder, compressor, values, scores, rope, stream
        )
        _assert_plan(captured_step, 1, boundaries=_CAPACITY, writes=_CAPACITY)
        captured_storage = _storage(captured_step, 1)
        before_full = _main_rows(builder.cache).clone()
        graph.replay()
        stream.synchronize()
        # Positive control: every B row in this graph really was a writer.
        _assert_only_rows_changed(builder.cache, before_full, _PAGES, row=0)

        # B=4 -> b=2, retaining/reordering the latter requests. Nonzero padded
        # inputs and a fresh poison value make a stray tail write observable;
        # replaying the old values onto their old rows would be a vacuous test.
        for case in (builder, eager):
            _reset_cache(case.cache, page_byte=0x6B, state_value=73.0)
        before = _main_rows(builder.cache).clone()
        state_before = [state.clone() for state in builder.cache.compress_state(0)]
        next_values, _ = _projections(1, 1)
        values.copy_(next_values)
        slots, pages = _SLOTS[2:], _PAGES[2:]
        replay_step = _prepare(builder, 1, slots, pages)
        _assert_plan(replay_step, 1, boundaries=2, writes=2)
        assert _storage(replay_step, 1) == captured_storage
        assert captured_step.plans[1].num_compress == _CAPACITY
        expected_latent = _compress(
            eager, compressor, values, scores, _prepare(eager, 1, slots, pages), rope
        )
        graph.replay()
        stream.synchronize()
        _assert_only_rows_changed(builder.cache, before, pages, row=1)
        _assert_matches_eager(builder.cache, eager.cache)
        idle_slots = sorted(set(_SLOTS) - set(slots))
        for actual, old in zip(builder.cache.compress_state(0), state_before):
            torch.testing.assert_close(
                actual[idle_slots], old[idle_slots], rtol=0, atol=0
            )
        assert latent is not None and expected_latent is not None
        # Inactive latent rows are intentionally undefined. Their PAGE/STATE
        # stores, rather than their temporary values, are what must be absent.
        torch.testing.assert_close(
            latent[:, :2], expected_latent[:, :2], rtol=0, atol=0
        )


def _cache_storage(cache):
    tensors = [cache.backing, cache.cursor, cache.tentative_staging.gpu]
    tensors.extend(tensor for pair in cache.indptr_buffers.values() for tensor in pair)
    return tuple((t.data_ptr(), tuple(t.shape), t.stride()) for t in tensors)


def _assert_decode_indptrs(observed, geometry, starts, width, padded_tokens, ratio):
    # Independent counting oracle from test_cache's shared-indptr regression.
    positions = [start + j for start in starts for j in range(width)]
    windows = [max(0, p + 1 - max(p - geometry.window_size + 1, 0)) for p in positions]
    split = not geometry.packed
    counts = [
        (0 if split else window) + min((p + 1) // ratio, geometry.index_topk)
        for window, p in zip(windows, positions)
    ]
    pad = [0] * (padded_tokens - len(positions))
    for actual, rows in zip(observed, (counts, windows if split else counts)):
        expected = torch.tensor(
            [0] + list(np.cumsum(rows + pad)), dtype=torch.int32, device="cuda"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual[len(positions)].item() > 0
        assert torch.all(actual[len(positions) :] == actual[len(positions)])


@pytest.mark.parametrize(
    "speculative_tokens,width,packed,ratio",
    [(0, 1, False, 1), (5, 6, True, 2)],
    ids=["non-spec-width1-bf16", "dspark5-width6-fp4"],
)
def test_profile_to_serving_full_graph_rebinds_real_engram_and_cache(
    monkeypatch, speculative_tokens, width, packed, ratio
):
    """True plugin/kernel/graph integration; only V2 orchestration is synthetic."""
    pytest.importorskip("vllm", reason="requires vLLM plugin runtime")
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import override_forward_context

    from atom.model_ops.engram.device.hashing import (
        engram_cursor_rows_reference,
        engram_row_indices_reference,
    )
    from atom.model_ops.engram.device.runtime import EngramInputPreparer
    from atom.plugin.vllm import deepseek_v41_bridge as bridge
    from atom.plugin.vllm import req_id_passthrough_patch as passthrough
    from atom.plugin.vllm.deepseek_v4_prefix_patch import _mark_v4_proxy_cache_mode
    from atom.utils.forward_context import get_forward_context
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config
    from tests.model_ops.engram.test_overlap import make_staging

    mapping = build(tiny_config())
    total = _CAPACITY * width
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        staging, mmap_backing = make_staging(mapping, None, VllmEngramStaging)
        graph = None
        try:
            staging.host.overlap = staging
            builder = _builder(
                ratio,
                packed,
                max_tokens=total,
                history_size=2,
                speculative_tokens=speculative_tokens,
                allocate_cache=False,
            )
            builder.engram = EngramInputPreparer(mapping, staging.host)
            builder.config = SimpleNamespace(
                engram_layer_ids=mapping.config.layer_ids,
                engram_max_ngram_size=mapping.config.max_ngram_size,
                engram_n_heads=mapping.config.n_heads,
                engram_head_dim=mapping.config.head_dim,
                parallel_config=SimpleNamespace(data_parallel_size=1),
                compilation_config=SimpleNamespace(static_forward_context={}),
            )
            proxy = bridge.AtomDeepseekV41ProxyAttention()
            model = SimpleNamespace()
            proxy._atom_v41_model, proxy._atom_v41_builder = model, builder
            placeholder = torch.full((1,), 73, dtype=torch.uint8, device="cuda")
            proxy.kv_cache = placeholder
            layer = bridge.ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME
            sfc = {layer: proxy}
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=_CAPACITY,
                    max_num_batched_tokens=total,
                ),
                compilation_config=SimpleNamespace(static_forward_context=sfc),
                speculative_config=(
                    SimpleNamespace(num_speculative_tokens=speculative_tokens)
                    if speculative_tokens
                    else None
                ),
                cache_config=SimpleNamespace(
                    num_gpu_blocks=_CAPACITY + 2,
                    cache_dtype="nvfp4" if packed else "auto",
                ),
                # The real geometry factory's documented cached value, not a stub.
                _atom_v41_geometry=builder.geometry,
            )
            factory = bridge.AtomDeepseekV41ProxyMetadataBuilder.__new__(
                bridge.AtomDeepseekV41ProxyMetadataBuilder
            )
            factory.vllm_config, factory.device = config, "cuda"
            batch = SimpleNamespace(
                num_reqs_after_padding=_CAPACITY,
                num_tokens_after_padding=total,
                input_ids=torch.full((total,), 31, dtype=torch.int32, device="cuda"),
            )
            mode = CUDAGraphMode.NONE
            monkeypatch.setattr(passthrough, "get_current_input_batch", lambda: batch)
            monkeypatch.setattr(passthrough, "get_current_cudagraph_mode", lambda: mode)
            monkeypatch.setattr(bridge, "_V41_PENDING_SPECULATIVE_STEP", None)
            monkeypatch.setattr(bridge, "_V41_OPEN_ENGRAM_FORK", None)

            def forbidden_fallback(*args, **kwargs):
                raise AssertionError(
                    "FULL forward must consume factory-prepared metadata"
                )

            monkeypatch.setattr(bridge, "v41_stage_step", forbidden_fallback)
            calls = {}

            def check_outside(obj, name):
                original = getattr(obj, name)
                calls[name] = 0

                def checked(*args, **kwargs):
                    assert not torch.cuda.is_current_stream_capturing(), name
                    calls[name] += 1
                    return original(*args, **kwargs)

                monkeypatch.setattr(obj, name, checked)

            for obj, names in (
                (builder, ("_prepare", "prepare_model_inputs")),
                (bridge, ("_v41_stage_outside_forward",)),
                (
                    staging,
                    (
                        "prepare",
                        "start",
                        "_snapshot",
                        "_advance_cursor",
                        "_prefetch",
                        "join",
                        "_await_layer",
                        "_await_fork",
                    ),
                ),
            ):
                for name in names:
                    check_outside(obj, name)

            def common_inputs(starts, slots, pages):
                count = len(slots)
                batch.num_reqs = count
                batch.req_ids = [f"request-{slot}" for slot in slots]
                batch.idx_mapping_np = np.asarray(slots, dtype=np.int32)
                batch.num_draft_tokens_per_req = [speculative_tokens] * count
                return SimpleNamespace(
                    num_reqs=count,
                    max_query_len=width,
                    query_start_loc_cpu=torch.arange(count + 1, dtype=torch.int32)
                    * width,
                    seq_lens=torch.tensor(
                        np.asarray(starts) + width, dtype=torch.int32, device="cuda"
                    ),
                    block_table_tensor=torch.tensor(
                        pages, dtype=torch.int32, device="cuda"
                    ).view(count, 1),
                )

            compressor, rope = _parameters(ratio)
            values, scores = _projections(ratio, 0, total)
            output = {
                lid: torch.empty_like(buffer.gpu[:total])
                for lid, buffer in staging.host.buffers.items()
            }
            observed_indptrs = tuple(
                torch.empty(total + 1, dtype=torch.int32, device="cuda")
                for _ in range(2)
            )

            def staging_pointers():
                return tuple(
                    t.data_ptr()
                    for t in (
                        staging.snapshot,
                        staging.compressed,
                        staging.uva.row_ids,
                        *staging.flat.values(),
                        *(buffer.gpu for buffer in staging.host.buffers.values()),
                    )
                )

            staging_storage = staging_pointers()

            def forward(common):
                with (
                    override_forward_context(
                        SimpleNamespace(attn_metadata={layer: common})
                    ),
                    bridge.atom_deepseek_v41_forward_context(
                        atom_config=builder.config,
                        builder=builder,
                        input_ids=batch.input_ids,
                        positions=None,
                        vllm_config=config,
                    ),
                ):
                    metadata = get_forward_context().attn_metadata
                    assert metadata is common.atom_v41_prepared.metadata
                    assert metadata.staged_outside_forward
                    for lid, buffer in output.items():
                        buffer.copy_(metadata.engram_embeddings.get(lid)[0])
                    for observed, pointer in zip(
                        observed_indptrs, metadata.step.indptrs[ratio][:2]
                    ):
                        observed.copy_(pointer)
                    # Profiling deliberately has builder.cache=None. The prepared
                    # cache is the production consumer's authority in both phases.
                    return metadata.cache.compress(
                        0, compressor, values, scores, metadata.step, rope
                    )

            _mark_v4_proxy_cache_mode(sfc, True)
            scratch = builder._capture_dummy_cache(1, _CAPACITY, total)
            _reset_cache(scratch)
            # Poison AFTER the legitimate reset; live=0 must not touch either.
            scratch.cursor.fill_(-37)
            scratch.tentative_staging.gpu.fill_(-99)
            profile_cursor = scratch.cursor.clone()
            profile_candidates = scratch.tentative_staging.gpu.clone()
            common = factory.build_for_cudagraph_capture(
                common_inputs([0] * _CAPACITY, _SLOTS, [999] * _CAPACITY)
            )
            prepared = common.atom_v41_prepared
            assert not common.atom_v41_snapshot.full_graph
            assert builder.cache is None and prepared.metadata.cache is scratch
            assert not hasattr(model, "_atom_v41_proxy_cache_ptr")
            assert proxy.kv_cache is placeholder and placeholder.item() == 73
            assert prepared.metadata.dummy and staging.step.live.item() == 0
            profile_engram_step = staging.step
            assert profile_engram_step.cursor.data_ptr() == scratch.cursor.data_ptr()
            assert (staging.step.candidates is not None) == bool(speculative_tokens)
            if speculative_tokens:
                assert (
                    staging.step.candidates.data_ptr()
                    == scratch.tentative_staging.gpu.data_ptr()
                )
            assert (
                scratch.pending is None and not bridge.v41_speculative_step_is_pending()
            )
            before_calls = calls.copy()
            forward(common)  # JIT warmup, not a graph and not a second preparation.
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                latent = forward(common)
            graph.replay()
            stream.synchronize()
            assert calls == before_calls
            assert torch.equal(scratch.cursor, profile_cursor)
            assert torch.equal(scratch.tentative_staging.gpu, profile_candidates)
            assert all(torch.count_nonzero(rows) == 0 for rows in output.values())
            scratch_storage = _cache_storage(scratch)
            scratch_before = scratch.backing.clone()
            graph.reset()
            del graph, latent, common, prepared
            graph = None  # No profiling graph survives the serving allocation.

            slots_count = bridge.v41_num_state_slots(config)
            size = builder.geometry.paged_extents(_CAPACITY + 2)[1]
            size += slots_count * builder.geometry.state_bytes
            proxy.kv_cache = torch.zeros(size, dtype=torch.uint8, device="cuda").view(
                1, 1, 1, -1
            )
            _mark_v4_proxy_cache_mode(sfc, False)
            common = factory.build_for_cudagraph_capture(
                common_inputs([0] * _CAPACITY, _SLOTS, [0] * _CAPACITY)
            )
            cache = builder.cache
            captured = common.atom_v41_prepared.metadata.step
            assert common.atom_v41_prepared.metadata.cache is cache is not scratch
            assert model._atom_v41_proxy_cache_ptr == proxy.kv_cache.data_ptr()
            assert staging.step is not profile_engram_step
            assert staging.step.cursor.data_ptr() == cache.cursor.data_ptr()
            del profile_engram_step
            if speculative_tokens:
                assert (
                    staging.step.candidates.data_ptr()
                    == cache.tentative_staging.gpu.data_ptr()
                )
            serving_storage = _cache_storage(cache)
            assert all(a[0] != b[0] for a, b in zip(serving_storage, scratch_storage))
            captured_storage = _storage(captured, ratio)
            before_calls = calls.copy()
            forward(common)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                latent = forward(common)
            graph.replay()
            stream.synchronize()
            capture_latent = latent.clone()
            capture_output = {lid: rows.clone() for lid, rows in output.items()}
            assert calls == before_calls

            eager = _builder(
                ratio,
                packed,
                max_tokens=total,
                num_slots=slots_count,
                history_size=2,
                speculative_tokens=speculative_tokens,
            )
            for target in (cache, eager.cache):
                _reset_cache(target, page_byte=0x6B, state_value=0.25)
            cache.cursor.fill_(-17)
            cache.tentative_staging.gpu.fill_(-99)
            slots, pages = (3, 1), (2, 4)
            starts = [1, 3] if not speculative_tokens else [3, 7]
            histories = [np.array([41, 51]), np.array([61, 71])]
            for slot, start, history in zip(slots, starts, histories):
                cache.cursor[slot] = torch.tensor([start, *history], device="cuda")
            mode = CUDAGraphMode.FULL
            for iteration in (1, 2):
                tokens = [
                    np.arange(width, dtype=np.int64) + 10 * iteration + 20 * i + 1
                    for i in range(len(slots))
                ]
                count = len(slots) * width
                batch.input_ids.fill_(509)  # Valid, nonzero padding, never looked up.
                batch.input_ids[:count].copy_(
                    torch.tensor(np.concatenate(tokens), device="cuda")
                )
                projected, gates = _projections(ratio, iteration, total)
                values.copy_(projected)
                if scores is not None:
                    scores.copy_(gates)
                page_before = _main_rows(cache).clone()
                state_before = cache.state_bytes.clone()
                cursor_before = cache.cursor.clone()
                candidates_before = cache.tentative_staging.gpu.clone()
                common = common_inputs(starts, slots, pages)
                returned = factory.build(0, common)
                # Observations only: no extra preparation or metadata patching
                # between the real live factory and the first serving replay.
                staged_cursor = cache.cursor.clone()
                staged_candidates = cache.tentative_staging.gpu.clone()
                staged_scratch = scratch.backing.clone()
                graph.replay()
                stream.synchronize()
                prepared = returned.atom_v41_prepared
                step = prepared.metadata.step
                assert returned.atom_v41_snapshot.full_graph and step.decode
                assert step.scheduled == count < step.width == total
                assert [span.slot for span in step.requests] == list(slots)
                assert step.tentative == bool(speculative_tokens)
                assert staging.step.cursor.data_ptr() == cache.cursor.data_ptr()
                if speculative_tokens:
                    assert (
                        staging.step.candidates.data_ptr()
                        == cache.tentative_staging.gpu.data_ptr()
                    )
                assert builder.model_runner.h2d_owner._state == "sealed"
                assert _storage(step, ratio) == captured_storage
                assert _cache_storage(cache) == serving_storage
                assert _cache_storage(scratch) == scratch_storage
                assert torch.equal(staged_scratch, scratch_before)
                assert torch.equal(scratch.backing, scratch_before)
                assert torch.equal(scratch.tentative_staging.gpu, profile_candidates)
                assert torch.equal(
                    cache.cursor, staged_cursor
                ), "replay advanced Engram twice"
                assert torch.equal(cache.tentative_staging.gpu, staged_candidates)
                assert staging_pointers() == staging_storage
                _assert_decode_indptrs(
                    observed_indptrs, cache.geometry, starts, width, total, ratio
                )
                cursor_rows = engram_cursor_rows_reference(
                    mapping, tokens, histories, starts
                )
                for lid, table in staging.host.prefetcher._tables.items():
                    indices = engram_row_indices_reference(
                        mapping, lid, tokens, histories
                    )
                    expected = table._tensor[torch.as_tensor(indices)].reshape(
                        count, -1
                    )
                    torch.testing.assert_close(
                        output[lid][:count].cpu(), expected, rtol=0, atol=0
                    )
                    assert torch.all(output[lid][count:] == 0)
                assert any(
                    not torch.equal(output[lid][:count], capture_output[lid][:count])
                    for lid in output
                )
                if speculative_tokens:
                    assert torch.equal(cache.cursor, cursor_before)
                    for i, expected in enumerate(cursor_rows):
                        np.testing.assert_array_equal(
                            staged_candidates[i, :width].cpu().numpy(), expected
                        )
                    assert torch.equal(
                        staged_candidates[len(slots) :], candidates_before[len(slots) :]
                    )
                else:
                    expected_cursor = cursor_before.clone()
                    for slot, expected in zip(slots, cursor_rows):
                        expected_cursor[slot] = torch.tensor(
                            expected[-1], device="cuda"
                        )
                    assert torch.equal(cache.cursor, expected_cursor)
                    assert torch.equal(staged_candidates, candidates_before)
                idle = sorted(set(range(slots_count)) - set(slots))
                assert torch.equal(cache.state_bytes[idle], state_before[idle])
                changed = torch.zeros(
                    page_before.shape[:2], dtype=torch.bool, device="cuda"
                )
                for page, start in zip(pages, starts):
                    changed[page, start // ratio : (start + width) // ratio] = True
                assert torch.all(
                    torch.any(
                        _main_rows(cache)[changed] != page_before[changed], dim=-1
                    )
                )
                assert torch.equal(_main_rows(cache)[~changed], page_before[~changed])
                eager_step = _prepare(eager, starts, slots, pages, width=width)
                expected_latent = _compress(
                    eager, compressor, values, scores, eager_step, rope
                )
                _assert_matches_eager(cache, eager.cache)
                boundaries = step.plans[ratio].num_compress
                assert boundaries > 0 and torch.isfinite(latent[:, :boundaries]).all()
                assert torch.all(step.plans[ratio].compress_plan_gpu[boundaries:] == -1)
                writes = step.plans[ratio].num_write
                assert torch.all(step.plans[ratio].write_plan_gpu[writes:] == -1)
                torch.testing.assert_close(
                    latent[:, :boundaries],
                    expected_latent[:, :boundaries],
                    rtol=0,
                    atol=0,
                )
                assert not torch.equal(
                    latent[:, :boundaries], capture_latent[:, :boundaries]
                )
                accepted = [min(width, 2 * (i + 1)) for i in range(len(slots))]
                if speculative_tokens:
                    anchors = torch.tensor(
                        [i * width + n - 1 for i, n in enumerate(accepted)],
                        dtype=torch.int64,
                        device="cuda",
                    )
                    assert bridge.v41_commit_speculative_state(anchors)
                    assert (
                        cache.pending is None
                        and not bridge.v41_speculative_step_is_pending()
                    )
                # Keep the last request, so the second replay also tests shrinking
                # and a real committed cursor, rather than manually reseeding it.
                histories = [cursor_rows[-1][accepted[-1] - 1, 1:].copy()]
                starts = [starts[-1] + accepted[-1]]
                slots, pages = slots[-1:], pages[-1:]
            assert (
                calls["_prepare"]
                == calls["prepare_model_inputs"]
                == calls["start"]
                == 4
            )
            assert (
                calls["_snapshot"]
                == calls["_advance_cursor"]
                == calls["_prefetch"]
                == 4
            )
            assert calls["_await_layer"] == 0 and calls["_await_fork"] == 4
            assert placeholder.item() == 73
        finally:
            stream.synchronize()
            staging.stream.synchronize()
            if graph is not None:
                graph.reset()
            for table in staging.host.prefetcher._tables.values():
                table.disable_uva()
            # Keep mmap storage alive through synchronization and unregister.
            assert all(not memory.closed for memory in mmap_backing)


def test_plugin_publication_owner_waits_before_pinned_source_overwrite():
    if not callable(getattr(torch.cuda, "_sleep", None)):
        pytest.skip("torch.cuda._sleep is needed to delay the actual H2D reader")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buffer = CpuGpuBuffer(
            8,
            4,
            dtype=torch.int32,
            device="cuda",
            pin_memory=True,
            publication_group="v4_plans",
        )
        assert buffer.cpu.is_pinned() and buffer.gpu.is_cuda
        pointers = (buffer.cpu.data_ptr(), buffer.gpu.data_ptr())
        observations = [torch.empty_like(buffer.gpu) for _ in range(3)]
        read_done = [torch.cuda.Event() for _ in range(2)]
        # Warm dispatch and allocate everything before creating a pending read.
        torch.cuda._sleep(1)
        for observation in observations:
            observation.copy_(buffer.gpu)
        for event in read_done:
            event.record()
        stream.synchronize()

        # Constructor-time publication precedes binding. The plugin factory's
        # initial event must cover this read, not just later owner.finish calls.
        buffer.cpu.fill_(11)
        torch.cuda._sleep(100_000_000)
        buffer.copy_to_gpu()
        observations[0].copy_(buffer.gpu)
        read_done[0].record()
        owner, groups = _v41_publication_owner({"plan": buffer}, "cuda")
        assert not read_done[0].query(), "delay expired before the first reuse gate"
        owner.begin()
        assert read_done[0].query(), "begin returned before the constructor H2D read"
        buffer.cpu.fill_(22)  # only AFTER begin waited for the old source reader

        torch.cuda._sleep(100_000_000)
        groups["v4_plans"].publish((8,))
        observations[1].copy_(buffer.gpu)
        read_done[1].record()
        owner.finish()
        assert not read_done[1].query(), "delay expired before the epoch reuse gate"
        owner.begin()
        assert read_done[1].query(), "begin returned before the previous epoch H2D read"
        buffer.cpu.fill_(33)
        groups["v4_plans"].publish((8,))
        observations[2].copy_(buffer.gpu)
        owner.finish()
        owner.completion.synchronize()

        assert (buffer.cpu.data_ptr(), buffer.gpu.data_ptr()) == pointers
        assert owner.epoch == 2 and owner._state == "sealed"
        for observation, expected in zip(observations, (11, 22, 33)):
            assert torch.all(observation.cpu() == expected)


def test_engram_join_outside_full_capture_keeps_replay_prefetch_live(monkeypatch):
    """Actual stream dependency only; the prefetch payload is not Engram hash/TP."""
    if not callable(getattr(torch.cuda, "_sleep", None)):
        pytest.skip("torch.cuda._sleep is needed to keep the side-stream read pending")
    compute, prefetch = torch.cuda.Stream(), torch.cuda.Stream()
    compute.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(compute):
        width, layer = 4, 0
        source = torch.zeros(width, 8, device="cuda")
        staged = torch.empty_like(source)
        observed = torch.empty_like(source)
        staging = VllmEngramStaging.__new__(VllmEngramStaging)
        staging.host = SimpleNamespace(
            device=source.device,
            _tp_group=None,
            buffers={layer: SimpleNamespace(gpu=staged)},
        )
        staging.stream = prefetch
        staging.done = {layer: torch.cuda.Event()}
        staging.collective = None
        waits = {"layer": 0, "fork": 0}
        await_layer, await_fork = staging._await_layer, staging._await_fork

        def checked_layer_wait(event):
            # Guard before issuing an illegal external-event wait: report the
            # regression without poisoning the HIP context on that error path.
            assert (
                not torch.cuda.is_current_stream_capturing()
            ), "consume repeated an external-event wait inside FULL capture"
            waits["layer"] += 1
            await_layer(event)  # production wait, on a real event

        def checked_fork_wait():
            assert (
                not torch.cuda.is_current_stream_capturing()
            ), "join repeated the external side-stream wait inside FULL capture"
            waits["fork"] += 1
            await_fork()  # production wait, on the actual prefetch stream

        def prefetch_payload(count):
            # Only the hash/UVA/TP DATA producer is replaced. Keep an actual
            # fork, delayed GPU write and recorded event, not mocked completion.
            prefetch.wait_stream(torch.cuda.current_stream(source.device))
            with torch.cuda.stream(prefetch):
                torch.cuda._sleep(100_000_000)
                staged[:count].copy_(source[:count])
                staging.done[layer].record(prefetch)

        monkeypatch.setattr(staging, "_snapshot", lambda _width: None)
        monkeypatch.setattr(staging, "_advance_cursor", lambda _width: None)
        monkeypatch.setattr(staging, "_prefetch", prefetch_payload)
        monkeypatch.setattr(staging, "_await_layer", checked_layer_wait)
        monkeypatch.setattr(staging, "_await_fork", checked_fork_wait)

        # Positive control: without a prior join, production consume DOES wait
        # on the real event. This also warms both stream waits before capture.
        source.fill_(7)
        staging.start(width)
        staging.consume(layer, width)
        observed.copy_(staged)
        staging.join()
        compute.synchronize()
        assert torch.all(observed == 7)
        assert waits == {"layer": 1, "fork": 1}

        source.fill_(11)
        staging.start(width)
        assert not staging._vllm_prefetch_joined
        staging.join()  # the FULL preparation boundary, outside the graph
        assert staging._vllm_prefetch_joined
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=compute):
            staging.consume(layer, width)
            observed.copy_(staged)
            staging.join()
        assert waits == {"layer": 1, "fork": 2}
        graph.replay()
        compute.synchronize()
        assert torch.all(observed == 11)

        pointer = staged.data_ptr()
        for epoch, value in enumerate((22, 33), start=3):
            source.fill_(value)
            staging.start(width)
            assert not staging._vllm_prefetch_joined
            assert not staging.done[layer].query(), "prefetch delay expired before join"
            staging.join()
            graph.replay()
            # No host/device synchronization between start -> join -> replay.
            # A missing compute<-prefetch edge leaves the old payload visible.
            compute.synchronize()
            assert staged.data_ptr() == pointer
            assert torch.all(observed == value)
            assert waits == {"layer": 1, "fork": epoch}
