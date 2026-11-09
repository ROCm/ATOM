# SPDX-License-Identifier: MIT
"""Quantized KV planes: ownership, source construction, and exact state movement."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the V4.1 cache and CSR kernels need Triton")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.planes import prefill_window
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


@pytest.mark.parametrize("dtype", ["bf16", "fp4", "fp8"])
@pytest.mark.parametrize("index_fp4", [False, True])
def test_capture_validator_checks_real_index_allocations(dtype, index_fp4):
    from dataclasses import replace

    from tests.test_deepseek_v41_runtime_validation import capture_checks, make_runner

    geo = replace(
        geometry(dtype), index_fp4=index_fp4, index_layout=None, index_block_rows=8
    )
    cache = PagedAttentionCache(geo, 7, 4, "cpu")
    runner, _ = make_runner(dtype)
    runner.attn_metadata_builder.cache = cache
    runner.attn_metadata_builder.block_size = 32
    for owner, planes in cache.index_planes.items():
        for i, plane in enumerate(planes):

            def corrupt(plane=plane):
                plane[1].view(torch.uint8).fill_(99)

            runner.capture_cudagraph = corrupt
            with pytest.raises(
                AssertionError, match=f"non-scratch pages in index_{owner}.{i}"
            ):
                capture_checks(runner, [1, 2, 4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefill_staging_cannot_grow_under_capture(monkeypatch):
    cache = PagedAttentionCache(geometry("fp8"), 7, 4, "cuda")
    small = SimpleNamespace(decode=False, width=16, cu_seqlens_q=torch.tensor([0, 16]))
    large = SimpleNamespace(decode=False, width=32, cu_seqlens_q=torch.tensor([0, 32]))
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
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("plan_dtype", [torch.int32, torch.int64])
def test_main_scatter_preserves_other_owner_and_all_window_planes(
    dtype, ratio, device, plan_dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    cache = PagedAttentionCache(geometry(dtype), 7, 4, device)
    for buffer in cache.allocations:
        buffer.fill_(29)
    owner = 1 if ratio == 1 else 0
    pages = tuple(plane.view(7, 32 // ratio, -1) for plane in cache.main_view(owner))
    rpp = 32 // ratio
    plan = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, rpp * ratio, 0],
            [-1, -1, -1, -1],
            [0, 0, (rpp + 1) * ratio, 0],
        ],
        dtype=plan_dtype,
    )
    step = SimpleNamespace(
        plans={ratio: SimpleNamespace(compress_plan_gpu=plan.to(device))},
        block_tables=torch.tensor([[3, 1], [6, 2]], device=device),
    )
    values = tuple(
        torch.randint(0, 256, (1, 4, p.shape[-1]), dtype=torch.uint8, device=device)
        for p in pages
    )
    source = cache.main_sources[owner]
    expected = source.clone()
    untouched = [buf.clone() for buf in cache.allocations if buf is not source]
    for p, val in zip(pages, values):
        for t, page, row in ((0, 3, 0), (1, 2, 0), (3, 1, 1)):
            at = p.storage_offset() + page * p.stride(0) + row * p.stride(1)
            expected[at : at + p.shape[-1]] = val[0, t]
    cache._scatter_rows(pages, step, values, ratio)
    torch.testing.assert_close(source, expected, rtol=0, atol=0)
    for actual, saved in zip(
        (buf for buf in cache.allocations if buf is not source), untouched
    ):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
def test_prefill_source_retains_history_until_ring_commit(dtype):
    cache = PagedAttentionCache(geometry(dtype), 7, 4, "cpu")
    old = cache.ring_view(1)
    old[0].fill_(41)
    old[0][:8].fill_(7)
    old[1].fill_(127)
    step = SimpleNamespace(
        width=12,
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


@pytest.mark.parametrize("planar", [False, True])
def test_existing_fp4_gather_receives_source_planes_without_repacking(
    planar, monkeypatch
):
    from atom.model_ops.attentions.deepseek_v41 import packed_rows

    calls = []

    class Launch:
        def __getitem__(self, grid):
            return lambda *args: calls.append(args)

    monkeypatch.setattr(packed_rows, "_gather_prefix", Launch())
    cache = PagedAttentionCache(geometry("fp4"), 7, 4, "cpu")
    main, window = cache.main_view(0), cache.ring_view(1)
    pool = (main, window) if planar else torch.empty(1024, dtype=torch.uint8)
    indices = torch.tensor([8, 5, 2**45], dtype=torch.int64)
    ptr = torch.tensor([0, 2], dtype=torch.int32)
    output = torch.empty(3, 512, dtype=torch.bfloat16)
    packed_rows.gather_prefix_rows(pool, indices, ptr, output, 0, 1)
    (args,) = calls
    assert args[1] is indices and args[2] is ptr and args[3] is output
    assert args[-1] is planar
    if planar:
        assert args[0] is main[0] and args[7] is main[1]
        assert args[8] is window[0] and args[9] is window[1]
    else:
        assert args[0] is pool and args[7:10] == (None, None, None)


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
@pytest.mark.parametrize("decode", [False, True])
def test_quantized_csr_launch_contract(dtype, decode, monkeypatch):
    import atom.model_ops.attentions.deepseek_v41.indices as idx

    calls = []

    class Launch:
        def __getitem__(self, grid):
            def capture(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return capture

    monkeypatch.setattr(idx, "_indices", Launch())
    geo = geometry(dtype)
    step = SimpleNamespace(
        width=5,
        max_q_len=2,
        decode=decode,
        positions=torch.arange(5),
        batch_ids=torch.zeros(5, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 5]),
        slots=torch.tensor([3]),
        block_tables=torch.tensor([[2, 0, 1]]),
        swa_replay_start=torch.tensor([0]),
        swa_cached_until=torch.tensor([0]),
    )
    pp = torch.zeros(6, dtype=torch.int32)
    ep = torch.zeros_like(pp)
    prefix, _, extend, _ = idx.build_indices(
        torch.zeros((5, 4), dtype=torch.int32),
        (pp, ep, 4),
        step,
        geo,
        geo.window(0, 7),
        0,
        2,
    )
    assert prefix.shape == (1, 5 * (4 + (geo.window_size if dtype == "fp4" else 0)))
    assert prefix.dtype == (torch.int64 if dtype == "fp4" else torch.int32)
    assert extend.numel() == (
        0 if decode else 5 * (2 if dtype == "fp4" else geo.window_size)
    )
    assert calls[0][2]["PAGE_ROWS"] == 16
    assert calls[0][2]["WINDOW_ROWS"] == 8
    assert calls[0][2]["SPLIT_PREFILL"] == (dtype == "fp8" and not decode)
    assert calls[0][2]["HISTORY"] == (dtype == "fp4")
    assert calls[0][2]["PACKED"] == (dtype == "fp4")


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("decode", [False, True])
def test_rope_producer_writes_the_native_plane_destinations(dtype, decode, monkeypatch):
    import atom.model_ops.attentions.deepseek_v41.cache as module

    calls = []
    monkeypatch.setattr(
        module, "rope_quant_window", lambda *args, **kw: calls.append(kw)
    )
    cache = PagedAttentionCache(geometry(dtype), 7, 4, "cpu")
    query = torch.empty(1, 3, 4, 512, dtype=torch.bfloat16)
    kv = torch.empty(1, 3, 512, dtype=torch.bfloat16)
    step = SimpleNamespace(
        width=3,
        decode=decode,
        positions=torch.arange(3),
        slots=torch.tensor([2]),
        cu_seqlens_q=torch.tensor([0, 3]),
    )
    rope = SimpleNamespace(
        rope_dim=64, cos_cache=torch.empty(3, 32), sin_cache=torch.empty(3, 32)
    )
    current, owed = cache.rope_quant_window(1, query, kv, rope, step)
    if decode:
        assert current is None and owed is None
        (data, scale), called_step, window = calls[0]["ring"]
        assert called_step is step and window.ring_start == 0
        assert data.data_ptr() == cache.ring_view(1)[0].data_ptr()
        assert scale.data_ptr() == cache.ring_view(1)[1].data_ptr()
    else:
        if dtype == "fp8":
            assert current[0].shape == (8 + 3, 512)
            assert current[0][8:].data_ptr() == owed[0].data_ptr()
            assert current[1][8:].data_ptr() == owed[1].data_ptr()
            assert calls[0]["qat"] is None
        else:
            assert current.dtype == torch.bfloat16 and current.shape == kv.shape
            assert calls[0]["qat"] is current
        data, scale = owed
        assert data.is_contiguous() and scale.is_contiguous()
        assert data.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
        assert data.shape == (1, 3, 512) and scale.shape == (1, 3, 16)
        assert calls[0]["values"] is data and calls[0]["scales"] is scale
        assert "ring" not in calls[0]


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
def test_window_only_cache_has_empty_csa_and_dense_window(dtype):
    from dataclasses import replace

    cache = PagedAttentionCache(replace(geometry(dtype), owners=()), 7, 4, "cpu")
    main = cache.main_view(None)
    assert main[0].shape[0] == main[1].shape[0] == 0
    assert cache.ring_view(2)[0].is_contiguous()


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
        width=8,
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
        width=3,
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
            width=3,
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


@pytest.mark.parametrize("decode", [False, True])
def test_a8w4_indices_request_untagged_source_rows(decode, monkeypatch):
    import atom.model_ops.attentions.deepseek_v41.indices as module

    calls = []

    class Launch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((args, kwargs))

    monkeypatch.setattr(module, "_indices", Launch())
    geo = geometry("fp4")
    step = SimpleNamespace(
        width=3,
        max_q_len=2,
        decode=decode,
        positions=torch.tensor([8, 9, 13]),
        batch_ids=torch.tensor([0, 0, 1]),
        cu_seqlens_q=torch.tensor([0, 2, 3]),
        slots=torch.tensor([2, 0]),
        block_tables=torch.tensor([[2, 1], [4, 0]]),
        swa_replay_start=torch.zeros(2, dtype=torch.int32),
        swa_cached_until=torch.zeros(2, dtype=torch.int32),
    )
    ptr = torch.zeros(4, dtype=torch.int32)
    selected = torch.zeros(3, 4, dtype=torch.int32)
    prefix, _, extend, _ = module.build_indices(
        selected, (ptr, ptr.clone(), 4), step, geo, geo.window(0, 7), 0, 2, split=True
    )
    assert prefix.shape == (1, 12) and prefix.dtype == torch.int32
    assert extend.numel() == (0 if decode else 3 * geo.window_size)
    assert not calls[0][1]["PACKED"] and not calls[0][1]["HISTORY"]
    assert calls[0][1]["SPLIT_PREFILL"] == (not decode)
    assert calls[0][1]["PAGE_ROWS"] == 16 and calls[0][1]["WINDOW_ROWS"] == 16


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
