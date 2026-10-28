# SPDX-License-Identifier: MIT
"""GPU DCP math against dense PyTorch over independently decoded native KV."""

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm/CUDA required"
)
from atom.model_ops.minimax_m3.dcp_attention import local_attention, merge_partials


def cache_fixture(fmt, page, world=4, scaled=False):
    torch.manual_seed(314)
    total, heads, dim = 768, 4, 128
    if fmt == "nvfp4":
        kb = torch.randint(0, 256, (total, heads, 64), device="cuda", dtype=torch.uint8)
        vb = torch.randint(0, 256, kb.shape, device="cuda", dtype=torch.uint8)
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device="cuda",
        )

        def decode(x):
            return lut[torch.stack((x & 15, x >> 4), -1).long()].flatten(-2).bfloat16()

        k, v = decode(kb), decode(vb)
    else:
        dtype = torch.bfloat16 if fmt == "bf16" else torch.float8_e4m3fn
        kb = (torch.randn(total, heads, dim, device="cuda") * 0.3).to(dtype)
        vb = torch.randn(total, heads, dim, device="cuda").to(dtype)
        k, v = kb.bfloat16(), vb.bfloat16()
    scale_k = scale_v = None
    if scaled and fmt != "bf16":
        width = 8 if fmt == "nvfp4" else 1
        scale_k = torch.rand(total, heads, width, device="cuda") * 0.4 + 0.1
        scale_v = torch.rand_like(scale_k) * 0.7 + 0.2
        if fmt == "nvfp4":
            scale_k = scale_k.to(torch.float8_e4m3fn).float()
            scale_v = scale_v.to(torch.float8_e4m3fn).float()
            k = (k.float() * scale_k.repeat_interleave(16, -1)).bfloat16()
            v = (v.float() * scale_v.repeat_interleave(16, -1)).bfloat16()
        else:
            k = (k.float() * scale_k).bfloat16()
            v = (v.float() * scale_v).bfloat16()
    block_ids = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    caches = []
    for rank in range(world):
        positions = [
            b * 128 + t for b in range(6) if b % world == rank for t in range(128)
        ]
        ids = torch.tensor(positions, device="cuda", dtype=torch.long)

        def pack(x, ids=ids, positions=positions):
            # Pad unused blocks, then permute the physical block IDs.
            local = torch.zeros((256, *x.shape[1:]), dtype=x.dtype, device="cuda")
            local[: len(positions)] = x.view(torch.uint8)[ids].view(x.dtype)
            local = local.reshape(2, 128, heads, -1).flip(0)
            if fmt == "nvfp4":
                return local.permute(0, 2, 1, 3).contiguous()
            xpack = 16 // x.element_size()
            physical = local.reshape(-1, page, heads, dim)
            if x is kb:
                return (
                    physical.reshape(-1, page, heads, dim // xpack, xpack)
                    .permute(0, 2, 3, 1, 4)
                    .contiguous()
                )
            return (
                physical.reshape(-1, page // xpack, xpack, heads, dim)
                .permute(0, 3, 1, 4, 2)
                .contiguous()
            )

        kc, vc = pack(kb), pack(vb)
        if fmt == "nvfp4":
            ks = torch.full((2, heads, 128, 8), 56, dtype=torch.uint8, device="cuda")
        elif fmt == "fp8":
            ks = torch.ones(
                (256 // page, heads, page), dtype=torch.float32, device="cuda"
            )
        else:
            ks = None
        vs = ks
        if scale_k is not None:

            def pack_scale(x, ids=ids, positions=positions):
                values = torch.zeros(256, heads, x.shape[-1], device="cuda")
                values[: len(positions)] = x[ids]
                values = values.reshape(2, 128, heads, -1).flip(0)
                if fmt == "nvfp4":
                    return (
                        values.permute(0, 2, 1, 3)
                        .contiguous()
                        .to(torch.float8_e4m3fn)
                        .view(torch.uint8)
                    )
                return values.reshape(-1, page, heads).permute(0, 2, 1).contiguous()

            ks, vs = pack_scale(scale_k), pack_scale(scale_v)
        caches.append((kc, vc, ks, vs))
    return k, v, block_ids, caches


@pytest.mark.parametrize(
    "fmt,page", [("bf16", 16), ("bf16", 128), ("fp8", 16), ("fp8", 128), ("nvfp4", 128)]
)
@pytest.mark.parametrize("sparse", [False, True])
def test_dcp_partials_match_unsharded_attention(fmt, page, sparse):
    k, v, table, caches = cache_fixture(fmt, page, scaled=True)
    rows, hq = 8, 64
    q = (torch.randn(rows, hq, 128, device="cuda") * 0.3).bfloat16()
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [1, 127, 128, 129, 511, 513, 767, 768], device="cuda", dtype=torch.int32
    )
    starts = torch.tensor(
        [0, 0, 16, 128, 256, 512, 528, 768], device="cuda", dtype=torch.int32
    )
    selected = (
        torch.tensor([0, 1, 4, 5, -1], device="cuda", dtype=torch.int32)
        .expand(4, rows, 5)
        .contiguous()
        if sparse
        else None
    )
    outputs, lses = [], []
    for rank, cache in enumerate(caches):
        o, lse = local_attention(
            q,
            cache,
            table,
            req,
            lengths,
            starts,
            world=4,
            rank=rank,
            scale=128**-0.5,
            topk=selected,
            nvfp4=fmt == "nvfp4",
        )
        outputs.append(o)
        lses.append(lse)
    actual, actual_lse = merge_partials(torch.stack(outputs), torch.stack(lses))
    expected = torch.zeros_like(actual)
    expected_lse = torch.full_like(actual_lse, -torch.inf)
    for row in range(rows):
        for head in range(hq):
            pos = torch.arange(768, device="cuda")
            mask = (pos >= starts[row]) & (pos < lengths[row])
            if sparse:
                mask &= torch.isin(pos // 128, selected[head // (hq // 4), row])
            kk, vv = (
                k[mask, head // (hq // 4)].float(),
                v[mask, head // (hq // 4)].float(),
            )
            if kk.shape[0]:
                score = kk @ q[row, head].float() * 128**-0.5
                expected[row, head] = score.softmax(0) @ vv
                expected_lse[row, head] = score.logsumexp(0)
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.002, rtol=0.002)
    assert torch.isfinite(actual).all()
    assert (actual[-1] == 0).all()


def test_graph_replay_replaces_empty_shard_with_live_tokens():
    _, _, table, caches = cache_fixture("fp8", 16)
    q = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    req = torch.zeros(1, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1], device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(lengths)

    def run():
        return local_attention(
            q, caches[3], table, req, lengths, starts, world=4, rank=3, scale=128**-0.5
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = run()
    graph.replay()
    assert not out.any()
    assert torch.isneginf(lse).all()
    lengths.fill_(513)
    graph.replay()
    expected, expected_lse = run()
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(lse, expected_lse)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("fp4", [False, True])
def test_owned_index_scores_and_global_topk(fp4):
    from atom.model_ops.minimax_m3.dcp_attention import _index_scores, index_topk

    k, _, table, caches = cache_fixture("nvfp4" if fp4 else "fp8", 128)
    rows, heads, blocks = 8, 4, 2
    q = (torch.randn(rows, heads, 128, device="cuda") * 0.1).bfloat16()
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [1, 127, 128, 129, 511, 513, 767, 768], device="cuda", dtype=torch.int32
    )
    scores = []
    indices = []
    for rank, cache in enumerate(caches):
        # Index K has one replicated vector per token, so use head 0 only.
        if fp4:
            key = cache[0][:, 0].contiguous()
            scale = cache[2][:, 0].contiguous()
        else:
            key = cache[0][:, 0].permute(0, 2, 1, 3).reshape(2, 128, 128).contiguous()
            scale = None
        score = torch.empty(heads, rows, blocks, device="cuda", dtype=torch.float32)
        _index_scores[(rows, blocks)](
            q,
            key,
            scale if fp4 else key,
            table,
            req,
            lengths,
            score,
            q.stride(0),
            q.stride(1),
            table.stride(0),
            heads,
            rows,
            blocks,
            4,
            rank,
            fp4,
            128**-0.5 * 1.4426950409,
            16,
        )
        scores.append(score)
        indices.append((key, scale))
    restored = torch.stack(scores, -1).reshape(heads, rows, 8)
    expected = torch.full_like(restored, -torch.inf)
    for block in range(6):
        for row in range(rows):
            end = min(int(lengths[row]), (block + 1) * 128)
            if end > block * 128:
                values = k[block * 128 : end, 0].float() @ q[row].float().T
                expected[:, row, block] = values.amax(0) * (128**-0.5 * 1.4426950409)
    torch.testing.assert_close(restored, expected, atol=0.002, rtol=0.002)

    class Gather:
        def all_gather(self, x, dim):
            return torch.cat(scores, dim=dim)

    result = index_topk(
        q,
        *indices[0],
        torch.nn.functional.pad(table, (0, 1022)),
        req,
        lengths,
        Gather(),
        0,
        4,
        3,
        1,
        1,
        128**-0.5,
        max_seq_len=768,
    )
    for row in range(rows):
        valid = (int(lengths[row]) + 127) // 128
        for head in range(heads):
            ranks = sorted(
                range(valid),
                key=lambda b: (
                    (
                        1e29
                        if b == valid - 1
                        else 1e30 if b == 0 else float(expected[head, row, b])
                    ),
                    b,
                ),
                reverse=True,
            )[:3]
            ranks += [-1] * (3 - len(ranks))
            assert result[head, row].tolist() == ranks


def test_dspark_dcp_allocation_cap_and_global_sliding_window():
    from types import SimpleNamespace

    import numpy as np

    from atom.models.minimax_m3.dspark import MiniMaxM3DSpark

    draft = object.__new__(MiniMaxM3DSpark)
    torch.nn.Module.__init__(draft)
    draft.max_num_seqs = 3
    draft.block_size = 128
    draft.dcp_world = 4
    draft.sliding_window = 4096
    draft._bt16 = None
    positions = torch.tensor(
        [[508 + t for t in range(8)], [5000 + t for t in range(8)], [0] * 8],
        device="cuda",
        dtype=torch.int64,
    )
    slots = torch.full_like(positions, 123)
    table = torch.arange(30, device="cuda", dtype=torch.int32).reshape(3, 10)
    lengths = torch.tensor([516, 5008, 0], device="cuda", dtype=torch.int32)
    md = SimpleNamespace()
    draft.prepare_paged_block(
        md, table, positions, slots, lengths, np.array([1, 10]), 2
    )
    expected_lens = torch.tensor(
        [512] * 8 + [5008] * 8 + [0] * 8, device="cuda", dtype=torch.int32
    )
    expected_start = torch.tensor(
        [max(0, int(p) - 4096 + 1) // 16 * 16 for p in positions.flatten().tolist()],
        device="cuda",
        dtype=torch.int64,
    )
    torch.testing.assert_close(md.dspark_dcp_lens, expected_lens)
    torch.testing.assert_close(md.dspark_dcp_start, expected_start)
    assert torch.all(slots[0, :4] == 123)
    assert torch.all(slots[0, 4:] == -1)
    # The next step allocates a second local page (512 additional global tokens).
    slots.fill_(123)
    draft.prepare_paged_block(
        md, table, positions, slots, lengths, np.array([2, 10]), 2
    )
    assert torch.all(md.dspark_dcp_lens[:8] == 516)
    assert torch.all(slots[0] == 123)
