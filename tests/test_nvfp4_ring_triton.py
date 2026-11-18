# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton decode tail append matches the torch ops it replaces, bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DIM, REQS, LAYERS, PAGE = 576, 40, 3, 64


def _case(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing

    ring = Nvfp4HpRing(REQS + 1, LAYERS, DIM, device="cuda")
    ring.buf.copy_(torch.randn(ring.buf.shape, generator=gen, device="cuda"))
    live = 32
    req_ids = torch.randperm(REQS, generator=gen, device="cuda")[:live]
    # Tile starts, ends, and everything between, at assorted pages.
    positions = torch.randint(0, 4096, (live,), generator=gen, device="cuda")
    positions[:4] = torch.tensor([0, 15, 16, 63], device="cuda")
    slots = (
        torch.randint(0, 500, (live,), generator=gen, device="cuda") * PAGE
        + positions % PAGE
    )
    # Two padding rows on the scratch row, slot -1.
    req_ids = torch.cat((req_ids, torch.full((2,), REQS, device="cuda")))
    positions = torch.cat((positions, torch.zeros(2, dtype=torch.long, device="cuda")))
    slots = torch.cat((slots, torch.full((2,), -1, dtype=torch.long, device="cuda")))
    values = torch.randn(live + 2, DIM, generator=gen, device="cuda").to(torch.bfloat16)
    return ring, req_ids.int(), positions, values, slots, live


def _torch_path(ring, layer, req_ids, positions, values, slots):
    tiles, valid, _, tile_ids = ring.append_tokens(
        layer, req_ids, positions, values, PAGE, check=False
    )
    pages = torch.div(slots, PAGE, rounding_mode="floor").to(torch.int32)
    return tiles, valid, tile_ids, pages


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_triton_append_matches_torch(layer):
    from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_gather

    ring, req_ids, positions, values, slots, live = _case(layer)
    ref_ring = ring.buf.clone()
    got = ring_append_gather(
        ring.buf[:, layer], req_ids, positions, values, slots, PAGE
    )

    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing

    ref = Nvfp4HpRing(REQS + 1, LAYERS, DIM, device="cuda")
    ref.buf.copy_(ref_ring)
    want = _torch_path(ref, layer, req_ids, positions, values, slots)
    # Padding rows share one scratch row; only the live rows are defined.
    for g, w in zip(got, want):
        assert torch.equal(g[:live], w[:live])
    assert torch.equal(got[3][live:], want[3][live:])  # pages of padding: -1
    assert torch.equal(ring.buf[:REQS], ref.buf[:REQS])


def _fused_vs_producer(seed, pages=600):
    from atom.model_ops.attentions.nvfp4_ring_triton import (
        ring_append_gather,
        ring_append_quantize,
    )

    pytest.importorskip("aiter.ops.mla_nvfp4")
    from aiter.ops.mla_nvfp4 import mla_nvfp4_update_cache

    ring, req_ids, positions, values, slots, _ = _case(seed)
    # Heavy-tailed values, exact zeros and negative zeros, like a real latent.
    gen = torch.Generator(device="cuda").manual_seed(seed + 100)
    scale = torch.exp(torch.randn(DIM, generator=gen, device="cuda") * 2)
    ring.buf.mul_(scale.to(torch.bfloat16))
    values = (values.float() * scale).to(torch.bfloat16)
    values[:, ::41] = 0
    values[:, 7::53] = -0.0
    slots = torch.where(slots >= 0, slots % (pages * PAGE), slots)
    nope = torch.randint(
        0, 256, (pages, 16384), dtype=torch.uint8, device="cuda", generator=gen
    )
    aux = torch.randint(
        0, 256, (pages, 8192), dtype=torch.uint8, device="cuda", generator=gen
    )
    # One update per (page, tile): drop rows that would share one.
    key = (slots // 16).tolist()
    keep = [i for i, k in enumerate(key) if k < 0 or key.index(k) == i]
    pick = torch.tensor(keep, device="cuda")
    req_ids, positions, values, slots = (
        req_ids[pick],
        positions[pick],
        values[pick],
        slots[pick],
    )

    ref_ring, ref_nope, ref_aux = ring.buf.clone(), nope.clone(), aux.clone()
    ring_append_quantize(ring.buf[:, 1], req_ids, positions, values, slots, nope, aux)
    tiles, valid, tile_ids, page_ids = ring_append_gather(
        ref_ring[:, 1], req_ids, positions, values, slots, PAGE
    )
    mla_nvfp4_update_cache(tiles, page_ids, tile_ids, valid, ref_nope, ref_aux)
    torch.cuda.synchronize()
    return (ring.buf, nope, aux), (ref_ring, ref_nope, ref_aux)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_write_matches_the_aiter_producer_byte_for_byte(seed):
    (r, n, a), (rr, rn, ra) = _fused_vs_producer(seed)
    assert torch.equal(r[:REQS], rr[:REQS])
    bad = (n != rn).nonzero()
    assert bad.numel() == 0, f"kv_nope differs at {bad[:8].tolist()}"
    bad = (a != ra).nonzero()
    assert bad.numel() == 0, f"kv_aux differs at {bad[:8].tolist()}"


def test_fused_write_replays_in_a_graph():
    from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_quantize

    ring, req_ids, positions, values, slots, _ = _case(9)
    slots = torch.where(slots >= 0, slots % (600 * PAGE), slots)
    nope = torch.zeros((600, 16384), dtype=torch.uint8, device="cuda")
    aux = torch.zeros((600, 8192), dtype=torch.uint8, device="cuda")
    start = ring.buf.clone()
    ring_append_quantize(ring.buf[:, 0], req_ids, positions, values, slots, nope, aux)
    eager = (ring.buf.clone(), nope.clone(), aux.clone())
    ring.buf.copy_(start)
    nope.zero_()
    aux.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ring_append_quantize(
            ring.buf[:, 0], req_ids, positions, values, slots, nope, aux
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(ring.buf[:REQS], eager[0][:REQS])
    assert torch.equal(nope, eager[1]) and torch.equal(aux, eager[2])


def test_triton_append_replays_in_a_graph():
    from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_gather

    ring, req_ids, positions, values, slots, live = _case(7)
    start = ring.buf.clone()
    eager = [
        t.clone()
        for t in ring_append_gather(
            ring.buf[:, 1], req_ids, positions, values, slots, PAGE
        )
    ]
    eager_ring = ring.buf.clone()
    ring.buf.copy_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ring_append_gather(
            ring.buf[:, 1], req_ids, positions, values, slots, PAGE
        )
    ring.buf.copy_(start)
    graph.replay()
    torch.cuda.synchronize()
    for o, e in zip(out, eager):
        assert torch.equal(o[:live], e[:live])
    assert torch.equal(ring.buf[:REQS], eager_ring[:REQS])
