"""The FlyDSL CP FP4 scorer (ATOM_M3_CP_SCORE_FLYDSL=1) against the Triton
`_context_score`: bit-identical scores on every live slot, dead slots untouched,
for every rank of a simulated group (gfx1250 only)."""

import pytest
import torch

from atom.utils import envs

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx_runtime  # noqa: E402

if get_gfx_runtime() != "gfx1250":
    pytest.skip("the FlyDSL CP scorer is gfx1250-only", allow_module_level=True)

from atom.model_ops.minimax_m3 import indexer_cp_score_flydsl as fly  # noqa: E402
from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store  # noqa: E402
from atom.model_ops.minimax_m3.indexer_context_parallel import (  # noqa: E402
    indexer_context_scores,
)

DEV, D, PAGE = "cuda", 128, 128


def _cache(bs, lens, bound_blocks, seed, mode):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = [-(-n // PAGE) for n in lens]
    pages = sum(nb) + 1
    payload = torch.empty(pages, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.empty(pages, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    if mode == "ties":
        base = (torch.randn(4 * PAGE, D, device=DEV, generator=g) * 3).to(torch.bfloat16)
        keys = base.repeat(-(-pages // 4), 1)[: pages * PAGE]
    elif mode == "zero":
        keys = torch.zeros(pages * PAGE, D, device=DEV, dtype=torch.bfloat16)
    else:
        keys = (torch.randn(pages * PAGE, D, device=DEV, generator=g) * 3).to(torch.bfloat16)
    index_fp4_store(keys, torch.arange(pages * PAGE, device=DEV), payload, scale)
    if mode == "nan":  # NaN E4M3 scales: single rows and whole pages
        scale.view(-1)[torch.randint(0, scale.numel(), (64,), device=DEV, generator=g)] = 0x7F
        scale[torch.randint(0, pages, (max(1, pages // 8),), device=DEV, generator=g)] = 0xFF
    if mode == "raw":  # every byte pattern, NaN scales included
        payload.random_(0, 256, generator=g)
        scale.random_(0, 256, generator=g)
    perm = torch.randperm(pages, device=DEV, generator=g).to(torch.int32)
    bt = torch.zeros(bs, bound_blocks, dtype=torch.int32, device=DEV)
    off = 0
    for b, n in enumerate(nb):
        bt[b, :n] = perm[off : off + n]
        off += n
    return payload, scale, bt, torch.tensor(lens, dtype=torch.int32, device=DEV)


CASES = [
    # bs, lengths, bound, query rows a request, heads, world, mode
    (16, [65536] * 16, 1 << 20, 8, 4, 4, "rand"),
    (4, [300001, 1, 5, 129], 400000, 8, 4, 4, "rand"),
    (8, [1, 2, 7, 8, 9, 127, 128, 129], 4096, 8, 4, 4, "rand"),
    (8, [1, 2, 7, 8, 9, 127, 128, 129], 4096, 1, 4, 4, "rand"),
    (5, [1000, 999, 4097, 70000, 3], 131072, 4, 4, 4, "rand"),
    (24, [int(x) for x in torch.randint(1, 300000, (24,), generator=torch.Generator().manual_seed(1))],
     307200, 8, 4, 4, "rand"),
    (12, [17000 + 997 * i for i in range(12)], 65536, 8, 4, 4, "ties"),
    (6, [17000, 3000, 129, 40000, 640, 100000], 131072, 8, 4, 4, "zero"),
    (6, [17000, 3000, 1290, 40000, 6400, 100000], 131072, 8, 4, 4, "nan"),
    (6, [17000, 3000, 1290, 40000, 6400, 100000], 131072, 8, 4, 4, "raw"),
    (8, [20000, 140000, 5, 130, 7000, 99999, 128, 256], 262144, 8, 2, 2, "rand"),
    (3, [70000, 5000, 129], 131072, 8, 4, 1, "rand"),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_flydsl_scores_match_triton(case, monkeypatch):
    bs, lens, bound, mq, heads, world, mode = CASES[case]
    bound_blocks = -(-bound // PAGE)
    payload, scale, bt, seq = _cache(bs, lens, bound_blocks, case, mode)
    g = torch.Generator(device=DEV).manual_seed(case + 1)
    q = torch.randn(bs * mq, heads, D, device=DEV, dtype=torch.bfloat16, generator=g)
    if mode == "zero":
        q = torch.ones_like(q)
    assert fly.supported(q, payload, scale, mq)
    for rank in range(world):
        args = (q, payload, bt, seq, bound, rank, world, mq, D**-0.5)
        monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "0")
        assert not envs.ATOM_M3_CP_SCORE_FLYDSL
        ref = indexer_context_scores(*args, index_scale=scale, fill_dead_slots=False)
        monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "1")
        got = torch.full_like(ref, float("nan"))
        fly.cp_scores_fp4(q, payload, scale, bt, seq, bound, rank, world, mq, D**-0.5, got)
        via = indexer_context_scores(*args, index_scale=scale, fill_dead_slots=False)
        torch.cuda.synchronize()
        rb = torch.clamp((seq.long() + PAGE - 1) // PAGE, max=bound_blocks)
        live = torch.clamp((rb - rank + world - 1) // world, min=0)
        m = torch.arange(ref.shape[2], device=DEV)[None, :] < live[:, None]
        m = m.repeat_interleave(mq, 0)[None].expand(heads, -1, -1)
        assert torch.equal(ref.view(torch.int32)[m], got.view(torch.int32)[m]), f"rank {rank}"
        assert torch.equal(ref.view(torch.int32)[m], via.view(torch.int32)[m]), f"rank {rank} via flag"
        assert torch.isnan(got[~m]).all(), f"rank {rank}: a dead slot was written"


def test_flydsl_scores_graph_replay(monkeypatch):
    """Captured once, replayed with other lengths and contents: the grid
    depends only on the batch, so the replay must still match."""
    monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "1")
    bs, bound, mq, heads, world = 8, 1 << 20, 8, 4, 4
    bb = bound // PAGE
    payload, scale, bt, seq = _cache(bs, [200000] * bs, bb, 7, "rand")
    q = torch.randn(bs * mq, heads, D, device=DEV, dtype=torch.bfloat16)
    local = -(-bb // world)
    out = torch.empty((heads, bs * mq, local), dtype=torch.float32, device=DEV)
    fly.cp_scores_fp4(q, payload, scale, bt, seq, bound, 1, world, mq, D**-0.5, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fly.cp_scores_fp4(q, payload, scale, bt, seq, bound, 1, world, mq, D**-0.5, out)
    for lens in ([1, 129, 70000, 200000, 4096, 5, 130000, 999], [199999] * 8):
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        q.normal_()
        out.fill_(float("nan"))
        graph.replay()
        monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "0")
        ref = indexer_context_scores(q, payload, bt, seq, bound, 1, world, mq, D**-0.5,
                                     index_scale=scale, fill_dead_slots=False)
        monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "1")
        torch.cuda.synchronize()
        rb = (seq.long() + PAGE - 1) // PAGE
        live = torch.clamp((rb - 1 + world - 1) // world, min=0)
        m = torch.arange(local, device=DEV)[None, :] < live[:, None]
        m = m.repeat_interleave(mq, 0)[None].expand(heads, -1, -1)
        assert torch.equal(ref.view(torch.int32)[m], out.view(torch.int32)[m])


@pytest.mark.parametrize("case", [0, 2, 4, 5, 6, 8])
@pytest.mark.parametrize("init,local_keep", [(1, 1), (0, 2)])
def test_flydsl_cp_selection_matches_tp(case, init, local_keep, monkeypatch):
    """End to end: FlyDSL shard scores -> shard top-k -> merge of the four
    shards' keys equals each TP rank's own selection (ids, page table, bounds)."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
        merge_candidate_keys,
    )

    bs, lens, bound, mq, heads, world, mode = CASES[case]
    if heads != world:
        pytest.skip("one head a rank")
    bb = -(-bound // PAGE)
    payload, scale, bt, seq = _cache(bs, lens, bb, 100 + case, mode)
    q = torch.randn(bs * mq, heads, D, device=DEV, dtype=torch.bfloat16,
                    generator=torch.Generator(device=DEV).manual_seed(case))
    monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "1")
    keys = [
        local_candidate_keys(
            indexer_context_scores(q, payload, bt, seq, bound, r, world, mq, D**-0.5,
                                   index_scale=scale, fill_dead_slots=False),
            seq, 16, r, world, mq, bb, init, local_keep,
        )
        for r in range(world)
    ]
    for h in range(heads):
        got = merge_candidate_keys(torch.stack([k[h] for k in keys]), bt, seq, 16, init, local_keep, mq)
        ref = minimax_m3_index_topk_decode(
            q[:, h : h + 1].contiguous(), payload, bt, seq, bound, 16, init, local_keep, 1, D**-0.5,
            emit_sparse_block_table=True, max_query_len=mq, index_kv_scale=scale,
        )
        for name, a, b in zip(("indices", "sparse_bt", "sparse_ctx"), ref, got):
            assert torch.equal(a, b), f"head {h}: {name}"
