# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 prefix images on the vLLM plugin: slots, sizing, round trip."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the V4 bridge reaches AITER dtypes")

from atom.plugin.vllm.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    _v4_kv_fp8,
    _v4_state_layout,
    slice_deepseek_v4_proxy_cache_views,
)
from atom.plugin.vllm.deepseek_v4_bridge import (
    _V4StateSlotAllocator as StateSlotAllocator,
)
from atom.plugin.vllm.deepseek_v4_image import (
    V4ImagePlacement,
    deepseek_v4_image_layer_names,
    make_v4_image_copier,
    v4_image_sizing,
)
from atom.plugin.vllm.paged_state_image import (
    execute_image_restores,
    execute_image_stores,
)

RATIOS = [0, 0, 4, 128, 4, 128, 4, 0]


def _vcfg(max_num_seqs=4):
    hf = NS(
        compress_ratios=RATIOS,
        num_hidden_layers=len(RATIOS) - 1,
        head_dim=512,
        index_head_dim=128,
        qk_rope_head_dim=64,
        sliding_window=128,
        index_topk=512,
    )
    return NS(
        model_config=NS(
            hf_config=hf, max_model_len=4096, architectures=["DeepseekV4ForCausalLM"]
        ),
        scheduler_config=NS(max_num_seqs=max_num_seqs),
        cache_config=NS(cache_dtype="fp8", enable_prefix_caching=True),
        speculative_config=None,
    )


def test_event_driven_slots_never_take_a_live_requests_slot():
    alloc = StateSlotAllocator(2, evict_lru=False)
    slots, reset = alloc.assign(["a", "b"], np.array([0, 256]))
    assert alloc.last_fresh_rows == [0, 1] and reset == set(slots.tolist())
    # Request "a" was not scheduled this step; it is still alive.
    with pytest.raises(RuntimeError, match="finished or preempted"):
        alloc.assign(["b", "c"], np.array([300, 0]))
    assert alloc.release(["a"]) == [int(slots[0])]
    again, _ = alloc.assign(["b", "c"], np.array([301, 0]))
    assert again[0] == slots[1] and again[1] == slots[0]
    assert alloc.last_fresh_rows == [1]
    # A preempted request comes back fresh, wherever it resumes.
    alloc.release(["b"])
    alloc.assign(["b"], np.array([128]))
    assert alloc.last_fresh_rows == [0]


def test_lru_default_is_unchanged_for_other_bridges():
    alloc = StateSlotAllocator(1)
    first, _ = alloc.assign(["a"], np.array([0]))
    second, reset = alloc.assign(["b"], np.array([0]))
    assert first[0] == second[0] and reset == {int(second[0])}


def test_sizing_prices_real_blocks_and_native_checkpoint_ranges():
    vc = _vcfg()
    s = v4_image_sizing(vc)
    assert s.page_bytes % 256 == 0 and s.page_bytes - s.page_unit_bytes < 256
    assert s.k == -(-s.image_bytes // s.page_unit_bytes)
    # HCA compressor state is not in a 128-aligned image; the window rows are.
    assert s.image_bytes < s.slot_bytes
    assert "nocopy=hca_main_kv,hca_main_score" in s.layout_id
    assert len(deepseek_v4_image_layer_names(s.k)) == s.k
    # The slot tail comes out of the tensor, not out of every page.
    tail = -(-s.num_slots * s.slot_bytes // s.page_bytes)
    for blocks in (tail + 64, tail + 1000):
        usable = s.usable_blocks(blocks)
        assert 0 < usable <= blocks - tail
        assert s.carve_bytes(usable) <= blocks * s.page_bytes
        assert s.carve_bytes(usable + 1) > blocks * s.page_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_image_round_trip_through_the_plugin_carve():
    vc = _vcfg()
    s = v4_image_sizing(vc)
    tensor_blocks = -(-s.num_slots * s.slot_bytes // s.page_bytes) + 6 * s.k + 8
    usable = s.usable_blocks(tensor_blocks)
    head = s.page_bytes // (2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    # vLLM's layout: logical (2, blocks, 128, 1, head), physically block-major.
    raw = torch.zeros(tensor_blocks * s.page_bytes, dtype=torch.uint8, device="cuda")
    proxy = raw.view(tensor_blocks, 2, ATOM_DEEPSEEK_V4_BLOCK_SIZE, 1, head)
    proxy = proxy.permute(1, 0, 2, 3, 4)
    planes, arena_rows, widths = _v4_state_layout(vc, _v4_kv_fp8(vc))
    views = slice_deepseek_v4_proxy_cache_views(
        proxy,
        compress_ratios=RATIOS,
        num_slots=s.num_slots,
        window_size=s.win_with_spec,
        kv_fp8=True,
        arena_planes=planes,
        arena_rows=arena_rows,
        row_widths=widths,
        num_blocks=usable,
    )
    copier = make_v4_image_copier(s, views, num_blocks=usable, device=raw.device)
    geo = views["geometry"]

    def slot_bytes(group):
        start, stop = geo.slot_span(geo.physical_slot(group))
        return [
            p.view(torch.uint8).reshape(p.shape[0], -1)[start:stop].clone()
            for p in copier._kv_planes()
        ]

    def fill_slot(group, value=None):
        start, stop = geo.slot_span(geo.physical_slot(group))
        for p in copier._kv_planes():
            rows = p.view(torch.uint8).reshape(p.shape[0], -1)[start:stop]
            if value is None:
                noise = torch.randint(0, 256, rows.shape, dtype=torch.uint8)
                rows.copy_(noise.to(rows.device))
            else:
                rows.fill_(value)

    fill_slot(2)
    source = slot_bytes(2)
    # Non-contiguous, out-of-order ids, as a block pool hands them out.
    units = tuple(range(usable - 1, usable - 1 - 2 * s.k, -2))
    untouched = raw.clone()
    execute_image_stores(copier, [2], [units])
    fill_slot(1, 0)
    execute_image_restores(copier, [1], [units])
    torch.cuda.synchronize()
    restored = slot_bytes(1)
    ranges = copier._checkpoint_slot_ranges()
    carried = 0
    for src, dst, plane_ranges in zip(source, restored, ranges):
        flat_src, flat_dst = src.reshape(-1), dst.reshape(-1)
        mask = torch.zeros_like(flat_dst, dtype=torch.bool)
        for off, n in plane_ranges:
            assert torch.equal(flat_dst[off : off + n], flat_src[off : off + n])
            mask[off : off + n] = True
            carried += n
        # Bytes the image does not carry stay as the reset left them.
        assert int(flat_dst[~mask].count_nonzero()) == 0
    assert carried == s.image_bytes
    # Only the slots and the image's own units changed in the pool.
    changed = (raw != untouched).nonzero().flatten()
    assert changed.numel() > 0
    region_bases, region_sizes = copier._page_unit_regions()
    base_ptr = raw.data_ptr()
    allowed = []
    for u in units:
        for b, n in zip(region_bases, region_sizes):
            lo = int(b) + u * int(n) - base_ptr
            allowed.append((lo, lo + int(n)))
    for g in (1,):
        start, stop = geo.slot_span(geo.physical_slot(g))
        for p, w in zip(copier._kv_planes(), widths):
            lo = p.data_ptr() - base_ptr + start * w
            allowed.append((lo, lo + (stop - start) * w))
    ok = torch.zeros(raw.numel(), dtype=torch.bool, device=raw.device)
    for lo, hi in allowed:
        ok[lo:hi] = True
    assert bool(ok[changed].all())


def test_placement_follows_native_ladder():
    p = V4ImagePlacement(interval=8192, demand=True)
    # 12,288-token prompt: rung 8192, anchor one block short of the end.
    assert p.limit(12288) == 8192
    anchor = p.anchor(12288, 12288)
    assert anchor == 12160
    assert p.cut(0, 8192, 12288, 0, anchor) == 8192
    assert p.cut(8192, 12288, 12288, 0, anchor) == 12160
    assert p.cut(12160, 12288, 12288, 0, anchor) == 0
    # Earliest candidate wins: a demand below the rung is cut first.
    assert p.cut(0, 8192, 12288, 2944, anchor) == 2944
    # Keeps: rung, anchor, demand; never past the prompt (generation keeps none).
    assert p.keeps(8192, 12288, 0, anchor) and p.keeps(anchor, 12288, 0, anchor)
    assert p.keeps(2944, 12288, 2944, anchor)
    assert not p.keeps(4096, 12288, 0, anchor)
    assert not p.keeps(16384, 12288, 0, anchor)
    # Demand only when the proxy hit is longer than the image hit, and room.
    assert p.demand_pos(0, 2944, True) == 2944
    assert p.demand_pos(2944, 2944, True) == 0
    assert p.demand_pos(0, 2944, False) == 0
    assert V4ImagePlacement(interval=8192, demand=False).demand_pos(0, 2944, True) == 0
    assert p.images_per_request() == 2
    # Generation: native's spacing rule, one interval past the last checkpoint.
    assert p.decode_keeps(9344, 0) and not p.decode_keeps(9344, 2944)
    assert not p.decode_keeps(9345, 0)
    assert not V4ImagePlacement(interval=-1, demand=True).decode_keeps(9344, 0)


def test_placement_interval_off_and_ladder_off():
    off = V4ImagePlacement(interval=0, demand=True)
    assert off.anchor(3000, 3000) == 0
    assert off.demand_pos(0, 2944, True) == 0
    assert not off.keeps(2944, 3000, 2944, 2944)
    assert off.images_per_request() == 0
    ladder_off = V4ImagePlacement(interval=-1, demand=True)
    assert ladder_off.limit(20000) == 0
    assert ladder_off.anchor(3000, 3000) == 2944
    assert not ladder_off.keeps(8192, 20000, 0, 19968)
    assert ladder_off.keeps(19968, 20000, 0, 19968)


def test_partial_eviction_restores_by_group():
    """vLLM evicts block by block (hybrid Mamba semantics): losing one group's
    block leaves the other groups hashed, the boundary is stored again, and a
    hit takes each group's first cached block -- the new store for the evicted
    group, the old one for the rest."""
    from vllm.v1.core.block_pool import BlockPool

    from atom.plugin.vllm.deepseek_v4_image import V4SchedulerImages

    gids = [1, 2, 3]
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=128)
    st = object.__new__(V4SchedulerImages)
    st.block_pool = pool
    st.image_gids = gids
    st.k = len(gids)
    req = NS(block_hashes=[b"h0", b"h1"], request_id="r")

    def store():
        image = []
        for gid in gids:
            (blk,) = pool.get_new_blocks(1)
            pool.cache_full_blocks(
                request=req,
                blocks=[blk],
                num_cached_blocks=0,
                num_full_blocks=1,
                block_size=128,
                kv_cache_group_id=gid,
            )
            image.append(blk)
        pool.free_blocks(image)
        return image

    a = store()
    assert st._image_cached(req, 0)
    assert pool._maybe_evict_cached_block(a[1])
    assert a[0].block_hash is not None and a[2].block_hash is not None
    assert a[1].block_hash is None
    assert not st._image_cached(req, 0)
    b = store()
    assert st._image_cached(req, 0)
    hit = pool.get_cached_block(b"h0", gids)
    assert [blk.block_id for blk in hit] == [
        a[0].block_id,
        b[1].block_id,
        a[2].block_id,
    ]
