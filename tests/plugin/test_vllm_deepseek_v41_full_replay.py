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


def _builder(ratio, packed):
    """Same hand-assembled builder as the CPU publication tests, on real HBM."""
    geometry = V41PoolGeometry(
        1,
        ((0, ratio),),
        32,
        4,
        _HEAD_DIM,
        32,
        packed=packed,
        layer_ratios=(ratio,),
        index_topk=4,
    )
    builder = VllmDeepseekV41MetadataBuilder.__new__(VllmDeepseekV41MetadataBuilder)
    builder.geometry = geometry
    builder.cache = PagedAttentionCache(
        geometry, _CAPACITY + 2, _CAPACITY, "cuda", max_tokens=_CAPACITY
    )
    builder.block_size, builder.device = 32, "cuda"
    builder.max_bs = builder.max_num_batched_tokens = _CAPACITY
    buffers = metadata_buffers(_CAPACITY, _CAPACITY, 1, "cuda", geometry)
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


def _prepare(builder, position, slots=_SLOTS, pages=_PAGES):
    """Publish through the real plugin override, never patch the plan tensors."""
    assert len(slots) == len(pages)
    count = len(slots)
    batch = SimpleNamespace(
        is_dummy_run=False,
        is_final_chunk=None,
        replaying={},
        req_ids=tuple(f"request-{slot}" for slot in slots),
        state_slots_committed=list(slots),
        num_scheduled_tokens=np.ones(count, dtype=np.int32),
        context_lens=np.full(count, position + 1, dtype=np.int32),
        block_tables=[(page,) for page in pages],
        total_seqs_num=count,
        total_tokens_num=count,
    )
    metadata, _ = builder._prepare(
        batch, _CAPACITY, _CAPACITY, max_q_len=1, engram_live=False
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


def _projections(ratio, phase):
    # The two steps are not scalar multiples: RMSNorm must not erase the
    # difference between a proper ratio-2 pool and a missing historical tail.
    columns = torch.arange(
        _CAPACITY * _HEAD_DIM, dtype=torch.float32, device="cuda"
    ).view(1, _CAPACITY, _HEAD_DIM)
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
