# SPDX-License-Identifier: MIT
"""The UVA gather through the device row cache returns exactly what the plain
gather returns: on a page-locked fp8 table with e8m0 scales, across hits,
misses, same-set collisions, repeated rows, dead ids, a head shard, and graph
replays that change the ids; and every filled way holds its row's bytes."""

import pytest
import torch

if not torch.cuda.is_available():
    # Before the import, which reaches Triton through the gather kernels.
    pytest.skip("needs a GPU for the UVA gather", allow_module_level=True)

from atom.model_ops.engram.device.uva import (
    UvaMissStaging,
    UvaRowCache,
    uva_gather_into,
)

ROWS, DIM, BLOCK, HEADS = 1 << 15, 256, 32, 24


class _Table:
    """The surface `uva_gather_into` reads off a page-locked HostEmbeddingTable."""

    head_dim, block_size, quantized = DIM, BLOCK, True

    def __init__(self, row_start=0, row_end=ROWS):
        g = torch.Generator().manual_seed(0)
        raw = torch.randint(0, 255, (ROWS, DIM), dtype=torch.uint8, generator=g)
        raw[(raw & 0x7F) == 0x7F] = 0  # no NaN encodings
        self.weight = raw[row_start:row_end].contiguous().pin_memory()
        self.scale = (
            torch.randint(
                110, 140, (ROWS, DIM // BLOCK), dtype=torch.uint8, generator=g
            )[row_start:row_end]
            .contiguous()
            .pin_memory()
        )
        self.range = (row_start, row_end)

    def registered_shard(self):
        return (
            self.weight.view(torch.float8_e4m3fn),
            self.scale,
            *self.range,
        )


def _ids(gen, tokens, hot=None):
    ids = torch.randint(0, ROWS, (tokens, HEADS), generator=gen)
    if hot is not None:  # many rows into few sets, and one row many times
        pick = hot[torch.randint(0, len(hot), (tokens, HEADS), generator=gen)]
        ids = torch.where(torch.rand(tokens, HEADS, generator=gen) < 0.6, pick, ids)
        ids[torch.rand(tokens, HEADS, generator=gen) < 0.1] = int(hot[0])
    ids[torch.rand(tokens, HEADS, generator=gen) < 0.02] = -1
    return ids.cuda()


def _same_set_rows(sets_log2, count=6):
    rows = torch.arange(ROWS, dtype=torch.int64)
    sets = ((rows * 2654435761) & 0xFFFFFFFF) >> (32 - sets_log2)
    return torch.cat([rows[sets == s][:8] for s in sets.unique()[:count]])


def _check_filled(cache, table):
    tag = cache.tag.cpu()
    filled = (tag >= 0).nonzero().squeeze(1)
    rows = tag[filled].long()
    assert torch.equal(cache.val.cpu()[filled], table.weight[rows])
    assert torch.equal(cache.scale.cpu()[filled], table.scale[rows])
    assert not bool((cache.claim >= 0).any())


@pytest.mark.parametrize("tokens", [1, 7, 96, 672])
def test_cached_gather_matches_plain(tokens):
    table = _Table()
    staging = UvaMissStaging(tokens * HEADS, DIM, BLOCK, "cuda")
    cache = UvaRowCache(DIM, BLOCK, staging, "cuda", slots_log2=10)
    hot = _same_set_rows(cache.sets_log2)
    gen = torch.Generator().manual_seed(tokens)
    plain = torch.empty(tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    cached = torch.empty_like(plain)
    for step in range(12):
        ids = _ids(gen, tokens, hot if step % 2 else None)
        uva_gather_into(
            table, ids, plain, head_start=0, local_heads=HEADS, total_heads=HEADS
        )
        uva_gather_into(
            table,
            ids,
            cached,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
            cache=cache,
        )
        assert torch.equal(plain.view(torch.int16), cached.view(torch.int16)), step
    _check_filled(cache, table)


def test_cached_gather_head_shard():
    start_head, local_heads, per_head = 6, 6, ROWS // HEADS
    lo, hi = start_head * per_head, (start_head + local_heads) * per_head
    table = _Table(lo, hi)
    staging = UvaMissStaging(96 * local_heads, DIM, BLOCK, "cuda")
    cache = UvaRowCache(DIM, BLOCK, staging, "cuda", slots_log2=8)
    gen = torch.Generator().manual_seed(3)
    plain = torch.empty(96, local_heads, DIM, dtype=torch.bfloat16, device="cuda")
    cached = torch.empty_like(plain)
    for _ in range(10):
        ids = _ids(gen, 96)  # rows outside [lo, hi) are not this shard's: zeros
        uva_gather_into(
            table,
            ids,
            plain,
            head_start=start_head,
            local_heads=local_heads,
            total_heads=HEADS,
        )
        uva_gather_into(
            table,
            ids,
            cached,
            head_start=start_head,
            local_heads=local_heads,
            total_heads=HEADS,
            cache=cache,
        )
        assert torch.equal(plain.view(torch.int16), cached.view(torch.int16))
    tag = cache.tag.cpu()
    filled = (tag >= 0).nonzero().squeeze(1)
    assert torch.equal(cache.val.cpu()[filled], table.weight[tag[filled].long()])


def test_cached_gather_graph_replay():
    tokens = 64
    table = _Table()
    staging = UvaMissStaging(tokens * HEADS, DIM, BLOCK, "cuda")
    cache = UvaRowCache(DIM, BLOCK, staging, "cuda", slots_log2=9)
    ids = torch.zeros(tokens, HEADS, dtype=torch.int64, device="cuda")
    out = torch.empty(tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    ref = torch.empty_like(out)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        uva_gather_into(
            table,
            ids,
            out,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
            cache=cache,
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        uva_gather_into(
            table,
            ids,
            out,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
            cache=cache,
        )
    gen = torch.Generator().manual_seed(5)
    hot = _same_set_rows(cache.sets_log2)
    clock = int(cache.clock)
    for step in range(40):
        ids.copy_(_ids(gen, tokens, hot if step % 3 == 0 else None))
        graph.replay()
        uva_gather_into(
            table, ids, ref, head_start=0, local_heads=HEADS, total_heads=HEADS
        )
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), step
    assert int(cache.clock) == clock + 40
    _check_filled(cache, table)
