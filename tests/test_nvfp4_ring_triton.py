# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The Triton decode tail append and FP4 write match the torch tail append
followed by the AITER producer, bit for bit, and the prefill tile write matches
the producer on the gathered tiles."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
requires_gfx1250 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="AITER's NVFP4 MLA producers are gfx1250-only",
)

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


def _torch_append(buf, req_ids, positions, values, slots, layer=1):
    """The torch tail append on a copy of ``buf``: that copy, then the tiles."""
    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing

    ring = Nvfp4HpRing(REQS + 1, LAYERS, DIM, device="cuda")
    ring.buf.copy_(buf)
    return ring.buf, _torch_path(ring, layer, req_ids, positions, values, slots)


def _aiter_kwargs():
    """The N16 writer of the mounted AITER, or a skip."""
    import inspect

    pytest.importorskip("aiter.ops.mla_nvfp4")
    from aiter.ops.mla_nvfp4 import mla_nvfp4_update_cache

    from atom.model_ops.attentions.nvfp4_mla import aiter_aux_n16

    if "aux_layout" in inspect.signature(mla_nvfp4_update_cache).parameters:
        return {"aux_layout": "n16"}
    if not aiter_aux_n16():
        pytest.skip("this AITER has no N16 AUX writer")
    return {}


def _fused_vs_producer(seed, pages=600):
    from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_quantize

    kw = _aiter_kwargs()
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

    # The tile as earlier steps left it: the producer's write of the rows
    # before the new one (a tile the new token starts keeps random bytes).
    _, (tiles, valid, tile_ids, page_ids) = _torch_append(
        ring.buf, req_ids, positions, values, slots
    )
    prior = (valid > 1) & (page_ids >= 0)
    if prior.any():
        rows = torch.arange(16, device="cuda")[None, :, None]
        old = tiles.masked_fill(rows >= (valid - 1)[:, None, None], 0)[prior]
        mla_nvfp4_update_cache(
            old, page_ids[prior], tile_ids[prior], valid[prior] - 1, nope, aux, **kw
        )

    ref_nope, ref_aux = nope.clone(), aux.clone()
    ref_ring, (tiles, valid, tile_ids, page_ids) = _torch_append(
        ring.buf, req_ids, positions, values, slots
    )
    ring_append_quantize(ring.buf[:, 1], req_ids, positions, values, slots, nope, aux)
    mla_nvfp4_update_cache(tiles, page_ids, tile_ids, valid, ref_nope, ref_aux, **kw)
    torch.cuda.synchronize()
    return (ring.buf, nope, aux), (ref_ring, ref_nope, ref_aux)


@requires_gfx1250
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


@requires_gfx1250
def test_prefill_quantize_matches_the_producer_on_gathered_tiles():
    from atom.model_ops.attentions.nvfp4_ring_triton import prefill_quantize

    kw = _aiter_kwargs()
    from aiter.ops.mla_nvfp4 import mla_nvfp4_update_cache

    gen = torch.Generator(device="cuda").manual_seed(17)
    lengths = [16, 1, 7, 15, 16, 3, 11]
    tiles, total = len(lengths), sum(lengths)
    # Heavy-tailed values, exact zeros and negative zeros, like a real latent.
    scale = torch.exp(torch.randn(DIM, generator=gen, device="cuda") * 2)
    kv = (torch.randn(total, DIM, generator=gen, device="cuda") * scale).to(
        torch.bfloat16
    )
    kv[:, :512:37] = 0
    kv[:, 517::17] = -0.0
    # Rows past a tile's valid repeat its last token, as `plan_prefill` does.
    rows = torch.arange(16, device="cuda")
    starts = torch.tensor([0] + lengths[:-1], device="cuda").cumsum(0)
    last = torch.tensor(lengths, device="cuda") - 1
    gather = (starts[:, None] + torch.minimum(rows[None, :], last[:, None])).view(-1)
    pages = torch.arange(tiles, device="cuda", dtype=torch.int32) * 3
    tile_ids = torch.arange(tiles, device="cuda", dtype=torch.int32) % 4
    valid = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    page_count = int(pages.max()) + 1
    nope = torch.randint(
        0, 256, (page_count, 16384), dtype=torch.uint8, device="cuda", generator=gen
    )
    aux = torch.randint(
        0, 256, (page_count, 8192), dtype=torch.uint8, device="cuda", generator=gen
    )
    ref_nope, ref_aux = nope.clone(), aux.clone()

    ref_tiles = kv[gather].view(tiles, 16, DIM)
    mla_nvfp4_update_cache(ref_tiles, pages, tile_ids, valid, ref_nope, ref_aux, **kw)
    # NoPE as a strided view of the latent, RoPE as its own tensor.
    prefill_quantize(
        kv[:, :512], kv[:, 512:].clone(), gather, pages, tile_ids, valid, nope, aux
    )
    torch.cuda.synchronize()

    bad = (nope != ref_nope).nonzero()
    assert bad.numel() == 0, f"kv_nope differs at {bad[:8].tolist()}"
    bad = (aux != ref_aux).nonzero()
    assert bad.numel() == 0, f"kv_aux differs at {bad[:8].tolist()}"


@requires_gfx1250
def test_split_prefill_publish_matches_the_concat_publish():
    """Pages and the BF16 tail of each request's short last tile, which the
    next decode token requantizes."""
    pytest.importorskip("aiter.ops.mla_nvfp4")
    from atom.model_ops.attentions.nvfp4_mla import Nvfp4MlaCache, aiter_aux_n16

    if not aiter_aux_n16():
        pytest.skip("the fused cache writes the N16 AUX layout")

    gen = torch.Generator(device="cuda").manual_seed(23)
    lengths = [21, 16, 5, 1, 70]
    tables = [[3], [1], [6], [2], [7, 4]]
    kv = torch.randn(sum(lengths), DIM, generator=gen, device="cuda").to(torch.bfloat16)

    def planned_cache():
        cache = Nvfp4MlaCache(2, 8, 8, DIM, device="cuda")
        fill = torch.Generator(device="cuda").manual_seed(5)
        for plane in (cache.kv_nope, cache.kv_aux):
            plane.copy_(
                torch.randint(0, 256, plane.shape, generator=fill, device="cuda")
            )
        cache.ring.buf.copy_(
            torch.randn(cache.ring.buf.shape, generator=fill, device="cuda")
        )
        rows = cache.bind_rows([11, 12, 13, 14, 15])
        plan = cache.plan_prefill(rows, [0] * len(lengths), lengths, tables)
        return cache, cache.upload_plan(plan)

    concat, concat_plan = planned_cache()
    split, split_plan = planned_cache()
    concat.publish_prefill(1, kv, concat_plan)
    split.publish_prefill_split(1, kv[:, :512], kv[:, 512:].clone(), split_plan)
    torch.cuda.synchronize()

    assert torch.equal(split.kv_nope, concat.kv_nope), "kv_nope differs"
    assert torch.equal(split.kv_aux, concat.kv_aux), "kv_aux differs"
    assert torch.equal(split.ring.buf, concat.ring.buf), "BF16 tail differs"
