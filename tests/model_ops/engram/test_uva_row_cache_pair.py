# SPDX-License-Identifier: MIT
"""Two tables' cached UVA gathers in one go (`uva_gather_pair_into`) return
exactly what two plain gathers return, across hits, misses, same-set
collisions, repeated rows, dead ids, a head shard and graph replays; each
table's cache holds its own rows; and a pair that does not fit the miss
scratch is left to the caller."""

import pytest
import torch

if not torch.cuda.is_available():
    # Before the imports, which reach Triton through the gather kernels.
    pytest.skip("needs a GPU for the UVA gather", allow_module_level=True)

from atom.model_ops.engram.device.uva import (
    UvaMissStaging,
    UvaRowCache,
    uva_gather_into,
    uva_gather_pair_into,
    uva_pair_fits,
)
from tests.model_ops.engram.test_uva_row_cache import (
    BLOCK,
    DIM,
    HEADS,
    ROWS,
    _check_filled,
    _ids,
    _same_set_rows,
    _Table,
)


class _OtherTable(_Table):
    """A second table: other bytes, other scales."""

    def __init__(self, row_start=0, row_end=ROWS):
        super().__init__(row_start, row_end)
        self.weight = self.weight.flip(0).contiguous().pin_memory()
        self.scale = (self.scale.flip(0) + 1).contiguous().pin_memory()


def _pair(tokens, local_heads=HEADS, slots_log2=10, row_range=(0, ROWS)):
    tables = (_Table(*row_range), _OtherTable(*row_range))
    pad = -(-tokens * local_heads // 64) * 64
    staging = UvaMissStaging(pad + tokens * local_heads, DIM, BLOCK, "cuda")
    caches = tuple(
        UvaRowCache(DIM, BLOCK, staging, "cuda", slots_log2=slots_log2) for _ in tables
    )
    return tables, staging, caches


def _plain(tables, ids, outs, head_start=0, local_heads=HEADS):
    for table, i, o in zip(tables, ids, outs):
        uva_gather_into(
            table,
            i,
            o,
            head_start=head_start,
            local_heads=local_heads,
            total_heads=HEADS,
        )


@pytest.mark.parametrize("tokens", [1, 7, 96, 672])
def test_pair_matches_plain(tokens):
    tables, staging, caches = _pair(tokens)
    hot = _same_set_rows(caches[0].sets_log2)
    gen = torch.Generator().manual_seed(tokens)
    plain = [
        torch.empty(tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        for _ in tables
    ]
    paired = [torch.empty_like(p) for p in plain]
    for step in range(12):
        ids = [_ids(gen, tokens, hot if step % 2 else None) for _ in tables]
        _plain(tables, ids, plain)
        assert uva_gather_pair_into(
            tables,
            ids,
            paired,
            caches,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
        )
        for p, q in zip(plain, paired):
            assert torch.equal(p.view(torch.int16), q.view(torch.int16)), step
        assert not bool(staging.count.any())
    for cache, table in zip(caches, tables):
        _check_filled(cache, table)
        assert int(cache.clock) == 1 + 12


def test_pair_same_rows_both_tables():
    """Both tables asked for the same row ids: each cache fills its own bytes."""
    tokens = 96
    tables, _, caches = _pair(tokens, slots_log2=8)
    gen = torch.Generator().manual_seed(11)
    plain = [
        torch.empty(tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        for _ in tables
    ]
    paired = [torch.empty_like(p) for p in plain]
    for _ in range(8):
        ids = _ids(gen, tokens, _same_set_rows(caches[0].sets_log2))
        _plain(tables, [ids, ids], plain)
        uva_gather_pair_into(
            tables,
            [ids, ids],
            paired,
            caches,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
        )
        for p, q in zip(plain, paired):
            assert torch.equal(p.view(torch.int16), q.view(torch.int16))
    for cache, table in zip(caches, tables):
        _check_filled(cache, table)


def test_pair_head_shard():
    start_head, local_heads, per_head = 6, 6, ROWS // HEADS
    lo, hi = start_head * per_head, (start_head + local_heads) * per_head
    tables, _, caches = _pair(96, local_heads, slots_log2=8, row_range=(lo, hi))
    gen = torch.Generator().manual_seed(3)
    plain = [
        torch.empty(96, local_heads, DIM, dtype=torch.bfloat16, device="cuda")
        for _ in tables
    ]
    paired = [torch.empty_like(p) for p in plain]
    for _ in range(10):
        ids = [_ids(gen, 96) for _ in tables]
        _plain(tables, ids, plain, start_head, local_heads)
        assert uva_gather_pair_into(
            tables,
            ids,
            paired,
            caches,
            head_start=start_head,
            local_heads=local_heads,
            total_heads=HEADS,
        )
        for p, q in zip(plain, paired):
            assert torch.equal(p.view(torch.int16), q.view(torch.int16))


def test_pair_graph_replay():
    tokens = 64
    tables, staging, caches = _pair(tokens, slots_log2=9)
    ids = [torch.zeros(tokens, HEADS, dtype=torch.int64, device="cuda") for _ in tables]
    outs = [
        torch.empty(tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        for _ in tables
    ]
    refs = [torch.empty_like(o) for o in outs]

    def run():
        return uva_gather_pair_into(
            tables,
            ids,
            outs,
            caches,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    gen = torch.Generator().manual_seed(5)
    hot = _same_set_rows(caches[0].sets_log2)
    clocks = [int(c.clock) for c in caches]
    for step in range(40):
        for i in ids:
            i.copy_(_ids(gen, tokens, hot if step % 3 == 0 else None))
        graph.replay()
        _plain(tables, ids, refs)
        for o, r in zip(outs, refs):
            assert torch.equal(o.view(torch.int16), r.view(torch.int16)), step
    assert [int(c.clock) for c in caches] == [c + 40 for c in clocks]
    assert not bool(staging.count.any())
    for cache, table in zip(caches, tables):
        _check_filled(cache, table)


def test_pair_that_does_not_fit_is_declined():
    tokens = 96
    tables = (_Table(), _OtherTable())
    staging = UvaMissStaging(tokens * HEADS, DIM, BLOCK, "cuda")  # one table's rows
    caches = tuple(
        UvaRowCache(DIM, BLOCK, staging, "cuda", slots_log2=8) for _ in tables
    )
    assert not uva_pair_fits(tables, caches, tokens, HEADS)
    assert uva_pair_fits(tables, caches, tokens // 3, HEADS)
    assert not uva_pair_fits(tables, (caches[0], None), 1, HEADS)
    assert not uva_pair_fits(tables, (caches[0], caches[0]), 1, HEADS)
    ids = [torch.zeros(tokens, HEADS, dtype=torch.int64, device="cuda")] * 2
    outs = [torch.full((tokens, HEADS, DIM), 7.0, dtype=torch.bfloat16, device="cuda")]
    outs.append(outs[0].clone())
    assert not uva_gather_pair_into(
        tables, ids, outs, caches, head_start=0, local_heads=HEADS, total_heads=HEADS
    )
    torch.cuda.synchronize()
    assert bool((outs[0] == 7.0).all()) and int(caches[0].clock) == 1
