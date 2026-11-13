# SPDX-License-Identifier: MIT
"""Quantized KV planes: ownership, source construction, and exact state movement."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the V4.1 cache and CSR kernels need Triton")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.indices import (
    build_indices,
    build_window_rows,
    fill_step_indptrs,
)
from atom.model_ops.attentions.deepseek_v41.planes import (
    prefill_window,
    reset_windows,
    split_source,
)
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry


def geometry(dtype, slack=0):
    return V41PoolGeometry(
        3,
        ((0, 2), (1, 1)),
        32,
        8,
        512,
        128,
        kv_dtype=dtype,
        speculative_tokens=slack,
        layer_ratios=(0, 1, 2),
        index_topk=4,
    )


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_staging_reuses_storage_with_live_plane_offsets(
    dtype, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    cache = PagedAttentionCache(geometry(dtype), 7, 4, device, max_tokens=16)
    pointer = None
    for requests in (2, 2, 1, 2):
        length = 16 // requests
        step = cache.begin_step(
            tuple(RequestSpan(b, 8, b * length, length, b) for b in range(requests)),
            block_tables=[[b] for b in range(requests)],
            is_prefill=True,
        )
        reserved = step.prefill_window_staging
        if pointer is None:
            pointer = reserved.data_ptr()
        assert reserved.data_ptr() == pointer
        step.begin_forward()
        assert step.prefill_window_staging is reserved
        for layer in range(3):
            history = cache.ring_view(layer)
            for i, plane in enumerate(history):
                plane.fill_(layer * 10 + i + 1)
            with monkeypatch.context() as patch:

                def no_allocation(*args, **kwargs):
                    pytest.fail("prefill_window allocated instead of reusing staging")

                patch.setattr(torch, "empty", no_allocation)
                source, tail = prefill_window(history, step, 8)
            rows = requests * 8
            assert source[0].data_ptr() == pointer
            assert source[1].data_ptr() == pointer + (rows + 16) * 512
            for i, (plane, current) in enumerate(zip(source, tail)):
                assert plane.is_contiguous() and current.is_contiguous()
                assert current.shape[0] == 16
                assert current.data_ptr() == plane.data_ptr() + rows * plane.shape[1]
                assert (plane[:rows] == layer * 10 + i + 1).all()
                current.fill_(99)


@pytest.mark.parametrize("dtype", ["bf16", "fp4", "fp8"])
def test_prefill_staging_grows_at_step_boundary_and_skips_decode(dtype):
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    cache = PagedAttentionCache(geometry(dtype), 7, 4, "cpu")
    cache.begin_step((RequestSpan(0, 8, 0, 1, 0),), block_tables=[[0]])
    assert cache.prefill_window_staging is None
    small = cache.begin_step(
        (RequestSpan(0, 8, 0, 2, 0),), block_tables=[[0]], is_prefill=True
    )
    large = cache.begin_step(
        (RequestSpan(0, 8, 0, 16, 0),), block_tables=[[0]], is_prefill=True
    )
    if dtype == "bf16":
        assert small.prefill_window_staging is large.prefill_window_staging is None
    else:
        assert (
            large.prefill_window_staging.numel() > small.prefill_window_staging.numel()
        )
        assert (
            large.prefill_window_staging.data_ptr()
            != small.prefill_window_staging.data_ptr()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("slack", [0, 3])
def test_prefill_staging_permuted_slots_and_graph_replay(dtype, slack):
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    cache = PagedAttentionCache(geometry(dtype, slack), 7, 4, "cuda", max_tokens=16)
    ring = cache.geometry.ring_slots
    step = cache.begin_step(
        (RequestSpan(0, 8, 0, 8, 3), RequestSpan(1, 8, 8, 8, 1)),
        block_tables=[[0], [1]],
        is_prefill=True,
    )
    history = cache.ring_view(1)
    for i, plane in enumerate(history):
        values = torch.arange(plane.numel(), device="cuda").view(plane.shape)
        plane.copy_((values + 31 * i) % 251)
    saved = [plane.clone() for plane in history]
    source, tail = prefill_window(history, step, ring)

    def check(ids):
        for output, old, current in zip(source, saved, tail):
            expected = torch.cat([old[slot * ring : (slot + 1) * ring] for slot in ids])
            torch.testing.assert_close(output[: 2 * ring], expected, rtol=0, atol=0)
            assert (current == 219).all()
        for actual, old in zip(history, saved):
            torch.testing.assert_close(actual, old, rtol=0, atol=0)
        assert source[0].data_ptr() == step.prefill_window_staging.data_ptr()

    for current in tail:
        current.fill_(219)
    check([3, 1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prefill_window(history, step, ring)
    torch.cuda.current_stream().wait_stream(stream)
    for ids in ([1, 3], [2, 0]):
        step.slots.copy_(torch.tensor(ids, dtype=step.slots.dtype, device="cuda"))
        for output in source:
            output[: 2 * ring].fill_(255)
        graph.replay()
        check(ids)


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("slack", [0, 3])
def test_published_decoder_tail_reuses_reserved_staging(
    dtype, device, slack, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
    from tests.attentions.deepseek_v41.helpers import decoder_publisher

    geo = geometry(dtype, slack)
    reserved = torch.empty(
        (2 * geo.ring_slots + 16) * geo.window_row_bytes,
        dtype=torch.uint8,
        device=device,
    )
    cache = PagedAttentionCache(
        geo, 7, 4, device, max_tokens=16, prefill_window_staging=reserved
    )
    publish = decoder_publisher(cache, 2, 16, 1, device)
    for requests in (2, 1, 0, 2):
        rows = [[b] for b in range(requests)]
        step = cache.begin_step(
            tuple(RequestSpan(b, 8, b * 8, 8, b) for b in range(requests)),
            block_tables=rows,
            is_prefill=True,
        )
        step.request_finals[:] = True
        tail = publish(step, rows).decoder.tail
        assert not tail.decode
        assert tail.prefill_window_staging is reserved
        tail.begin_forward()
        assert tail.prefill_window_staging is reserved
        for layer in range(3):
            history = cache.ring_view(layer)
            for i, plane in enumerate(history):
                plane.fill_(layer * 10 + i + 1)
            with monkeypatch.context() as patch:
                patch.setattr(
                    torch,
                    "empty",
                    lambda *a, **k: pytest.fail("decoder tail allocated staging"),
                )
                source, current = prefill_window(history, tail, geo.ring_slots)
            assert all(
                p.untyped_storage().data_ptr() == reserved.data_ptr()
                for p in (*source, *current)
            )
            for i, plane in enumerate(source):
                assert (plane[: requests * geo.ring_slots] == layer * 10 + i + 1).all()

    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), monkeypatch.context() as patch:
            patch.setattr(
                torch,
                "empty",
                lambda *a, **k: pytest.fail("capture allocated staging"),
            )
            for layer in range(3):
                source, _ = prefill_window(cache.ring_view(layer), tail, geo.ring_slots)
        for value in (41, 73):
            for plane in cache.ring_view(2):
                plane.fill_(value)
            graph.replay()
            assert all(
                p.untyped_storage().data_ptr() == reserved.data_ptr() for p in source
            )
            assert all((p[: 2 * geo.ring_slots] == value).all() for p in source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefill_staging_cannot_grow_under_capture(monkeypatch):
    cache = PagedAttentionCache(geometry("fp8"), 7, 4, "cuda")
    small = SimpleNamespace(
        decode=False, running_tokens=16, cu_seqlens_q=torch.tensor([0, 16])
    )
    large = SimpleNamespace(
        decode=False, running_tokens=32, cu_seqlens_q=torch.tensor([0, 32])
    )
    cache._reserve_prefill_window(small)
    pointer = small.prefill_window_staging.data_ptr()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    cache._reserve_prefill_window(small)
    assert small.prefill_window_staging.data_ptr() == pointer
    with pytest.raises(RuntimeError, match="reserved before capture"):
        cache._reserve_prefill_window(large)
    assert cache.prefill_window_staging.data_ptr() == pointer


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("slack", [0, 3])
def test_planes_are_dense_disjoint_and_cover_exact_budget(dtype, slack):
    cache = PagedAttentionCache(geometry(dtype, slack), 7, 4, "cpu")
    buffers = (*cache.main_sources.values(), *cache.window_sources)
    assert len({buf.untyped_storage().data_ptr() for buf in cache.allocations}) == len(
        cache.allocations
    )
    for owner, ratio in cache.geometry.owners:
        data, scale = cache.main_view(owner)
        assert data.shape == (7 * (32 // ratio), 256 if dtype == "fp4" else 512)
        assert scale.shape == (data.shape[0], 32 if dtype == "fp4" else 16)
        for plane in (data, scale):
            assert plane.is_contiguous()
            assert (
                plane.untyped_storage().data_ptr()
                == cache.main_sources[owner].data_ptr()
            )
    for layer in range(3):
        data, scale = cache.ring_view(layer)
        assert data.shape == (4 * (8 + slack), 512)
        assert scale.shape == (data.shape[0], 16)
        assert data.is_contiguous() and scale.is_contiguous()
    assert sum(n for _, n in cache.unit_regions()) == cache.geometry.paged_bytes
    assert sum(v.shape[1] for _, v in cache.state_regions) == cache.geometry.state_bytes
    regions = [v for _, v in cache.page_regions + cache.state_regions]
    regions += [v for planes in cache.index_planes.values() for v in planes]
    spans = sorted((v.data_ptr(), v.data_ptr() + v.numel()) for v in regions)
    assert all(a[1] <= b[0] for a, b in pairwise(spans))
    once_aligned = cache.geometry.paged_extents(7)[1] - 7 * cache.geometry.paged_bytes
    assert sum(end - start for start, end in spans) + once_aligned == sum(
        buf.numel() for buf in cache.allocations
    )
    assert cache.page_bytes is None
    for source in buffers:
        assert source.ndim == 1 and source.is_contiguous()
    assert cache.geometry.layout_id.startswith("source-buffer-v1:")


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
def test_prefill_source_retains_history_until_ring_commit(dtype):
    cache = PagedAttentionCache(geometry(dtype), 7, 4, "cpu")
    old = cache.ring_view(1)
    old[0].fill_(41)
    old[0][:8].fill_(7)
    old[1].fill_(127)
    step = SimpleNamespace(
        running_tokens=12,
        batch_ids=torch.tensor([0] * 11 + [-1]),
        positions=torch.arange(100, 112),
        cu_seqlens_q=torch.tensor([0, 11, 11]),
        slots=torch.tensor([2, -1]),
        max_q_len=11,
        request_cached_until=torch.tensor([104, 0]),
        swa_cached_until=torch.tensor([104, 0]),
    )
    source, current = prefill_window(old, step, 8)
    assert source[0].shape == (2 * 8 + 12, 512)
    assert (
        source[0].untyped_storage().data_ptr() == source[1].untyped_storage().data_ptr()
    )
    for prepared, history, new in zip(source, old, current):
        assert prepared.is_contiguous() and new.is_contiguous()
        torch.testing.assert_close(prepared[:8], history[16:24])
        assert new.data_ptr() == prepared[16:].data_ptr()
    # Simulate the producer writing directly into the source's current tail.
    current[0].copy_(torch.arange(12, dtype=torch.uint8)[:, None].expand(12, 512))
    current[1].fill_(128)
    before = old[0].clone()
    cache.write_window(1, current, step)
    expected = before.clone()
    for t in range(4, 11):
        expected[2 * 8 + (100 + t) % 8] = current[0][t]
    torch.testing.assert_close(old[0], expected)
    # History is a snapshot; the live ring is committed only after attention.
    assert (source[0][:8] == 41).all()
    assert cache.ring_view(0)[0].count_nonzero() == 0


@pytest.mark.parametrize("dtype", ["fp4", "fp8", "bf16"])
def test_checkpoint_and_overlapping_relocation_preserve_segments(dtype):
    cache = PagedAttentionCache(geometry(dtype, 3), 32, 4, "cpu")
    geo = cache.geometry
    spec = SimpleNamespace(
        page_unit_bytes=geo.paged_bytes,
        slot_bytes=geo.state_bytes,
        image_bytes=geo.checkpoint_image_bytes,
        layout_id=geo.layout_id,
        units_per_checkpoint=-(-geo.checkpoint_image_bytes // geo.paged_bytes),
    )
    copies = StateCopies(cache, spec, 4)
    for i in range(4):
        for view in cache.state_views(i):
            view.copy_(torch.randint(0, 256, view.shape, dtype=torch.uint8))
    saved = [v.clone() for v in cache.state_views(2, checkpoint=True)]
    omitted = [
        cache.state.view(name)[:, 1].clone()
        for name in ("compress_kv", "compress_score")
    ]
    units = tuple(range(1, 2 * spec.units_per_checkpoint, 2))
    store = SimpleNamespace(
        src_slot=2,
        unit_ids=units,
        total_bytes=spec.image_bytes,
        layout_id=spec.layout_id,
    )
    restore = SimpleNamespace(
        dst_slot=1,
        unit_ids=units,
        total_bytes=spec.image_bytes,
        layout_id=spec.layout_id,
    )
    copies.execute([store], [restore])
    for actual, expected in zip(cache.state_views(1, checkpoint=True), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, expected in zip(("compress_kv", "compress_score"), omitted):
        assert torch.equal(
            cache.state.view(name)[:, 1].contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    before = [[v.clone() for v in cache.state_views(i)] for i in range(4)]
    pairs = [(1, 0), (0, 1), (2, 3)]
    copies.relocate(pairs)
    for src, dst in pairs:
        for actual, expected in zip(cache.state_views(dst), before[src]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    unchanged = [v.clone() for v in cache.allocations]
    with pytest.raises(IndexError):
        copies.relocate([(0, 1), (2, 4)])
    for actual, saved in zip(cache.allocations, unchanged):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
def test_read_windows_uses_layer_and_slot_planes(dtype):
    cache = PagedAttentionCache(geometry(dtype, 3), 7, 4, "cpu")
    for layer in range(3):
        data, scale = cache.ring_view(layer)
        data.copy_(
            torch.full(data.shape, layer + 1.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        scale.fill_(127)
    actual = cache.read_windows(torch.tensor([2, 0]), torch.tensor([3, 1]))
    assert actual.shape == (2, 2, 11, 512)
    assert (actual[0] == 3).all() and (actual[1] == 1).all()


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
def test_external_slab_uses_same_source_layout_without_extra_allocations(dtype):
    from atom.plugin.vllm.v41_core_overrides import VllmBackedPagedAttentionCache

    geo = geometry(dtype, 3)
    size = geo.paged_extents(7)[1] + 4 * geo.state_bytes
    slab = torch.full((size + 256,), 37, dtype=torch.uint8)
    cache = VllmBackedPagedAttentionCache(geo, 7, 4, "cpu", backing=slab[:size])
    assert sum(buf.numel() for buf in cache.allocations) == size
    assert len({buf.untyped_storage().data_ptr() for buf in cache.allocations}) == 1
    assert (slab[size:] == 37).all()
    for owner, _ in geo.owners:
        data, scale = cache.main_view(owner)
        assert data.is_contiguous() and scale.is_contiguous()
        assert scale.data_ptr() == data.data_ptr() + data.numel()
    for layer in range(geo.layers):
        data, scale = cache.ring_view(layer)
        assert data.is_contiguous() and scale.is_contiguous()
        assert scale.data_ptr() == data.data_ptr() + data.numel()


@pytest.mark.parametrize("slots", [4, 97])
def test_prefill_scratch_depends_on_batch_capacity_and_keeps_slot_mapping(slots):
    cache = PagedAttentionCache(geometry("fp8", 3), 7, slots, "cpu")
    data, scale = cache.ring_view(0)
    for slot in range(slots):
        data[slot * 11 : (slot + 1) * 11].fill_(slot)
        scale[slot * 11 : (slot + 1) * 11].fill_(127 + slot % 2)
    step = SimpleNamespace(
        running_tokens=8,
        slots=torch.tensor([slots - 1, -1, 1]),
        cu_seqlens_q=torch.tensor([0, 3, 4, 7]),
    )
    source, current = prefill_window((data, scale), step, 11)
    assert source[0].shape == (3 * 11 + 8, 512)
    assert source[1].shape == (3 * 11 + 8, 16)
    assert (source[0][:11] == slots - 1).all()
    assert (source[0][22:33] == 1).all()
    assert (source[1][:11] == 127 + (slots - 1) % 2).all()
    assert current[0].data_ptr() == source[0][33:].data_ptr()
    assert current[1].data_ptr() == source[1][33:].data_ptr()


def test_a8w4_prefill_reuses_mxfp8_window_and_keeps_history(monkeypatch):
    import atom.model_ops.attentions.deepseek_v41.cache as module

    emitted = []

    def produce(*args, **kwargs):
        emitted.append(kwargs)
        kwargs["values"].view(torch.uint8).fill_(42)
        kwargs["scales"].view(torch.uint8).fill_(127)

    monkeypatch.setattr(module, "rope_quant_window", produce)
    cache = PagedAttentionCache(geometry("fp4"), 7, 4, "cpu")
    old = cache.ring_view(1)
    old[0].copy_(torch.arange(32, dtype=torch.uint8)[:, None].expand(32, 512))
    old[1].fill_(128)
    snapshot = tuple(plane.clone() for plane in old)
    step = SimpleNamespace(
        running_tokens=3,
        decode=False,
        positions=torch.tensor([8, 9, 13]),
        slots=torch.tensor([2, 0]),
        cu_seqlens_q=torch.tensor([0, 2, 3]),
    )
    rope = SimpleNamespace(
        rope_dim=64, cos_cache=torch.empty(16, 32), sin_cache=torch.empty(16, 32)
    )
    query = torch.empty(1, 3, 64, 512, dtype=torch.bfloat16)
    kv = torch.empty(1, 3, 512, dtype=torch.bfloat16)
    assert cache.query_fp8(query) is None
    q8 = cache.query_fp8(query, a8w4=True)
    assert q8[0].shape == query.shape and q8[1].shape == (1, 3, 64, 16)
    source, pending = cache.rope_quant_window(1, query, kv, rope, step, q8, a8w4=True)
    assert source[0].shape == (19, 512) and source[1].shape == (19, 16)
    for plane, before, current, width in zip(source, snapshot, pending, (512, 16)):
        torch.testing.assert_close(plane[:8], before[16:24])
        torch.testing.assert_close(plane[8:16], before[:8])
        assert plane[16:].data_ptr() == current.data_ptr()
        assert current.shape == (1, 3, width)
    assert (source[0][16:] == 42).all() and (source[1][16:] == 127).all()
    for plane, before in zip(old, snapshot):
        torch.testing.assert_close(plane, before)
    assert emitted[0]["qat"] is None and emitted[0]["query_fp8"] is q8
    assert cache.packed and not cache.kv_fp8


def test_a8w4_indptrs_are_owned_by_each_step_and_separate_from_fallback(monkeypatch):
    import atom.model_ops.attentions.deepseek_v41.cache as module

    built = []

    def fill(step, geo, buffers, *, split):
        assert split
        built.append(buffers)
        return {r: (*pair, geo.batch_topk(r)) for r, pair in buffers.items()}

    monkeypatch.setattr(module, "fill_step_indptrs", fill)
    cache = PagedAttentionCache(geometry("fp4"), 7, 4, "cpu")
    steps = [
        SimpleNamespace(
            running_tokens=3,
            positions=torch.arange(3),
            a8w4_indptrs={},
            indptrs={2: (torch.full((4,), 9), torch.full((4,), 7), 4)},
        )
        for _ in range(2)
    ]
    for step in steps:
        first = cache.indptrs(step, a8w4=True)
        assert cache.indptrs(step, a8w4=True) is first
        assert (step.indptrs[2][0] == 9).all() and (step.indptrs[2][1] == 7).all()
    assert len(built) == 2
    assert built[0][2][0].data_ptr() != built[1][2][0].data_ptr()
    assert built[0][2][0].data_ptr() != steps[0].indptrs[2][0].data_ptr()


@pytest.mark.parametrize("dtype", ["bf16", "fp4", "fp8"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("slack", [0, 3])
def test_reset_state_slots_preserves_unselected_slots_and_pages(dtype, device, slack):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    cache = PagedAttentionCache(geometry(dtype, slack), 7, 4, device)
    regions = [region for _, region in cache.state_regions]
    for region in regions:
        region.fill_(37)
    before = [region.clone() for region in regions]
    pages = [region.clone() for _, region in cache.page_regions]
    slots = torch.tensor([3, 1], dtype=torch.int64, device=device)
    cache._reset_state_slots(slots[:0])
    for actual, old in zip(regions, before):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
    cache._reset_state_slots(slots)
    for actual, old in zip(regions, before):
        old[slots] = 0
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
    for (_, actual), old in zip(cache.page_regions, pages):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
def test_reset_state_graph_uses_live_slot_ids(dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cache = PagedAttentionCache(geometry(dtype, 3), 7, 4, "cuda")
    slots = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    cache._reset_state_slots(slots)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cache._reset_state_slots(slots)
    torch.cuda.current_stream().wait_stream(stream)
    for ids in ([0, 2], [1, 3]):
        for _, region in cache.state_regions:
            region.fill_(91)
        slots.copy_(torch.tensor(ids, device="cuda"))
        graph.replay()
        for _, region in cache.state_regions:
            expected = torch.full_like(region, 91)
            expected[ids] = 0
            torch.testing.assert_close(region, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("count", [1500, 1600])
def test_reset_windows_above_grid_y_limit(count):
    # 43 * 1600 exceeds 65535. Two tiles also exercise partial data/scale masks.
    layers, ring, dim = 43, 3, 512
    capacity = count + 2
    sources = [
        torch.full(
            (capacity * ring * (dim + dim // 32),),
            layer + 1,
            dtype=torch.uint8,
            device="cuda",
        )
        for layer in range(layers)
    ]
    ptrs = torch.tensor(
        [p.data_ptr() for source in sources for p in split_source(source, dim)],
        dtype=torch.int64,
        device="cuda",
    )
    slots = torch.arange(count, 0, -1, device="cuda", dtype=torch.int64)
    reset_windows(sources, ptrs, slots, ring, dim)
    for layer, source in enumerate(sources):
        for plane in split_source(source, dim):
            expected = torch.full_like(plane.view(capacity, -1), layer + 1)
            expected[1:-1] = 0
            torch.testing.assert_close(
                plane.view_as(expected), expected, rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", ["fp4", "fp8", "fp4_a8w4"])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cached,replay", [(0, 0), (12, 0), (0, 10)])
def test_csr_resolves_to_causal_main_and_window_rows(
    dtype, decode, cached, replay, position_dtype
):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    a8w4 = dtype == "fp4_a8w4"
    geo = V41PoolGeometry(
        2,
        ((0, 2),),
        32,
        8,
        512,
        128,
        kv_dtype="fp4" if a8w4 else dtype,
        speculative_tokens=3,
        layer_ratios=(2,),
        index_topk=4,
    )
    device = "cuda"
    tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    positions = [10, 11, 12, 13, 31, 32, 0]
    batches = [0, 0, 0, 0, 1, 1, -1]
    tables = [[2, 0, 1], [3, 4, 5]]
    slots = [2, 0]
    step = SimpleNamespace(
        running_tokens=7,
        max_q_len=4,
        decode=decode,
        positions=tensor(positions).to(position_dtype),
        batch_ids=tensor(batches),
        cu_seqlens_q=tensor([0, 4, 6]),
        slots=tensor(slots),
        block_tables=tensor(tables),
        swa_cached_until=tensor([cached, 0]),
        swa_replay_start=tensor([replay, 0]),
    )
    indptrs = fill_step_indptrs(
        step, geo, {2: (tensor([0] * 8), tensor([0] * 8))}, split=True if a8w4 else None
    )[2]
    selected = tensor([[0, 1, 2, 3]] * 7)
    window = geo.window(0, 6)
    ring_rows = len(slots) * geo.ring_slots
    pi, pp, wi, wp = build_indices(
        selected, indptrs, step, geo, window, 0, 2, split=True if a8w4 else None
    )
    assert pi.dtype == (torch.int64 if dtype == "fp4" else torch.int32)
    if decode and (dtype == "fp8" or a8w4):
        wi = build_window_rows(step, geo, window, wp)
    pi, pp, wi, wp = pi[0].tolist(), pp.tolist(), wi.tolist(), wp.tolist()
    for t, (pos, batch) in enumerate(zip(positions, batches)):
        if batch < 0:
            assert pp[t + 1] == pp[t] and wp[t + 1] == wp[t]
            continue
        main = [tables[batch][0] * 16 + j for j in range(4)]
        if dtype == "fp8" or a8w4:
            assert pi[pp[t] : pp[t + 1]] == main
        start = 10 if batch == 0 else 31
        floor = max(0, pos - 7, (replay if batch == 0 else 0) if not decode else 0)
        cached_end = max(start, cached if batch == 0 else 0)
        history_end = (
            pos + 1 if decode else (cached_end if pos >= cached_end else start)
        )
        history, current = [], []
        for p in range(floor, pos + 1):
            if p < history_end:
                owner = slots[batch] if decode or dtype == "fp4" else batch
                history.append(owner * geo.ring_slots + p % geo.ring_slots)
            else:
                current.append((0 if batch == 0 else 4) + p - start)
        if dtype == "fp4":
            expected = [row * 2 for row in main] + [row * 2 + 1 for row in history]
            assert pi[pp[t] : pp[t + 1]] == expected
            if not decode:
                assert wi[wp[t] : wp[t + 1]] == current
        else:
            assert wi[wp[t] : wp[t + 1]] == history + [
                ring_rows + row for row in current
            ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_prefill_request_counts_reuse_indices_kernel(dtype, monkeypatch):
    import triton

    geo = V41PoolGeometry(
        2,
        ((0, 2),),
        32,
        8,
        512,
        128,
        kv_dtype=dtype,
        speculative_tokens=3,
        layer_ratios=(2,),
        index_topk=4,
    )
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    tokens = 16
    selected = tensor([[0, 1, 2, 3]] * tokens)
    positions = torch.empty(tokens, dtype=torch.int32, device="cuda")
    batches = torch.empty_like(positions)
    cu = torch.empty(5, dtype=torch.int32, device="cuda")
    pptr, eptr = tensor([0] * (tokens + 1)), tensor([0] * (tokens + 1))
    step = SimpleNamespace(
        running_tokens=tokens,
        max_q_len=tokens,
        decode=False,
        positions=positions,
        batch_ids=batches,
        slots=tensor([3, 1, 2, 0]),
        block_tables=tensor([[1, 0], [2, 0], [3, 0], [4, 0]]),
        swa_cached_until=tensor([0] * 4),
        swa_replay_start=tensor([0] * 4),
    )
    cache_misses = []

    def record_miss(**event):
        if event["fn"].name == "_indices":
            cache_misses.append(event["key"])

    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", record_miss)
    # Fixed token bucket and tensor alignment; only the live request count changes.
    for iteration, requests in enumerate((1, 2, 3, 4, 1)):
        lengths = [
            tokens // requests + (b < tokens % requests) for b in range(requests)
        ]
        offsets = [0]
        pos, batch_ids = [], []
        for batch, length in enumerate(lengths):
            offsets.append(offsets[-1] + length)
            pos.extend(range(10 + 32 * batch, 10 + 32 * batch + length))
            batch_ids.extend([batch] * length)
        positions.copy_(tensor(pos))
        batches.copy_(tensor(batch_ids))
        cu[: requests + 1].copy_(tensor(offsets))
        step.cu_seqlens_q = cu[: requests + 1]
        indptrs = fill_step_indptrs(step, geo, {2: (pptr, eptr)}, split=True)[2]
        pi, pp, wi, wp = build_indices(
            selected, indptrs, step, geo, geo.window(0, 6), 0, 2, split=True
        )
        pi, pp, wi, wp = pi[0].tolist(), pp.tolist(), wi.tolist(), wp.tolist()
        for t, (position, batch) in enumerate(zip(pos, batch_ids)):
            assert pi[pp[t] : pp[t + 1]] == [(batch + 1) * 16 + j for j in range(4)]
            start = 10 + 32 * batch
            expected = [
                (
                    batch * geo.ring_slots + p % geo.ring_slots
                    if p < start
                    else requests * geo.ring_slots + offsets[batch] + p - start
                )
                for p in range(position - geo.window_size + 1, position + 1)
            ]
            assert wi[wp[t] : wp[t + 1]] == expected
        if iteration == 0:
            cache_misses.clear()  # The first request count may compile once.
        else:
            assert not cache_misses, "request count introduced a new specialization"
