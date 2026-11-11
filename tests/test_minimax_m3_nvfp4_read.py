# SPDX-License-Identifier: MIT
"""NVFP4 read path end to end, without the model: pages written by ``nvfp4_write``,
scored and selected by ``minimax_m3_index_topk{,_decode}(index_nvfp4=True)``, attended
by ``sparse_attn_nvfp4``. Oracles are plain torch on the *dequantized pages* (a
token-major reader written here from the page offsets of ``nvfp4_pages``, independent of
both the writer and the kernels):

* the selection: block score = max over the block's tokens of K . q (K = e2m1 * e4m3
  scale, exact; q as stored e4m3), tail block pinned, 15 more by score;
* the attention: softmax over the keys of the blocks the selector *picked*, fp64.

Needs gfx950 and aiter's FlyDSL (kernels are aiter's), like the MXFP4 tests.
"""

import pytest
import torch

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)

from atom.model_ops.minimax_m3.nvfp4_pages import (  # noqa: E402
    INDEX_PAGE_BYTES,
    head_regions,
    index_page_views,
    kv_page_bytes,
)
from atom.model_ops.minimax_m3.nvfp4_write import (  # noqa: E402
    nvfp4_insert_index,
    nvfp4_insert_kv,
)

TOPK, DIM = 16, 128
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _fp4_values(codes):
    mag = GRID.to(codes.device)[codes & 7]
    return torch.where(codes >= 8, -mag, mag)


def deq_index(pages):
    """fp32 [pages, 128, 128] of index pages."""
    data, scale = index_page_views(pages)
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    s = scale.view(torch.float8_e4m3fn).float()
    return (_fp4_values(codes).unflatten(-1, (8, 16)) * s[..., None]).flatten(-2)


def deq_kv(pages, heads, gs_k, gs_v):
    """(K, V) fp32 [pages, heads, 128 tokens, 128] of KV pages, read token by token
    from the offsets documented in nvfp4_pages."""
    dev = pages.device
    t = torch.arange(128, device=dev)[:, None]
    d = torch.arange(128, device=dev)[None, :]
    g = torch.arange(8, device=dev)[None, :]
    r = t % 32
    n = 4 * (r // 8) + r % 4
    out = []
    for side, gs in ((0, gs_k), (1, gs_v)):
        if side == 0:
            data_off = t * 64 + d // 2
            shift = (d % 2) * 4 + 0 * t
            scale_off = (
                (((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4
            ) * 2 + g % 2
        else:
            data_off = ((t // 8) * 128 + d) * 4 + (t % 8) // 2
            shift = (t % 2) * 4 + 0 * d
            scale_off = g * 128 + t
        per_head = []
        for h in range(heads):
            db, sb = head_regions(side, h, heads)
            codes = ((pages[:, db + data_off] >> shift.to(torch.uint8)) & 15).long()
            s = pages[:, sb + scale_off].view(torch.float8_e4m3fn).float()
            vals = _fp4_values(codes).unflatten(-1, (8, 16)) * s[..., None]
            per_head.append(vals.flatten(-2) * gs)
        out.append(torch.stack(per_head, 1))
    return out[0], out[1]


class Case:
    """Requests with ragged lengths, scattered physical pages, written through the
    writer; the oracles read them back."""

    def __init__(self, lens, heads=1, group=16, seed=0, gs=(1.0, 1.0)):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.heads, self.group = heads, group
        self.lens = lens
        self.gs = gs
        blocks = max(-(-n // 128) for n in lens)
        self.pages = len(lens) * blocks + 3
        perm = torch.randperm(self.pages, device="cuda", generator=g)
        self.table = torch.stack(
            [perm[i * blocks : (i + 1) * blocks] for i in range(len(lens))]
        ).to(torch.int32)
        self.kv = torch.zeros(
            (self.pages, kv_page_bytes(heads)), dtype=torch.uint8, device="cuda"
        )
        self.idx = torch.zeros(
            (self.pages, INDEX_PAGE_BYTES), dtype=torch.uint8, device="cuda"
        )
        ks = torch.tensor([gs[0]], device="cuda")
        vs = torch.tensor([gs[1]], device="cuda")
        slots, k, v, ik = [], [], [], []
        for b, n in enumerate(lens):
            pos = torch.arange(n, device="cuda")
            slots.append(self.table[b, pos // 128].long() * 128 + pos % 128)
            scale = torch.exp2(
                torch.randint(
                    -3, 3, (n, heads, 8, 1), device="cuda", generator=g
                ).float()
            )
            k.append(
                (
                    torch.randn(n, heads, 8, 16, device="cuda", generator=g) * scale
                ).flatten(-2)
            )
            v.append(torch.randn(n, heads, DIM, device="cuda", generator=g))
            ik.append(torch.randn(n, DIM, device="cuda", generator=g) * 0.8)
        slots = torch.cat(slots)
        k, v, ik = (torch.cat(x).to(torch.bfloat16) for x in (k, v, ik))
        nvfp4_insert_kv(k, v, self.kv, slots, ks, vs)
        nvfp4_insert_index(ik, self.idx, slots)
        self.ks, self.vs = ks, vs
        self.K, self.V = deq_kv(self.kv, heads, gs[0], gs[1])
        self.IK = deq_index(self.idx)
        self.gen = g

    def queries(self, rows):
        hq = self.heads * self.group
        q = (torch.randn(rows, hq, DIM, device="cuda", generator=self.gen) * 0.7).to(
            torch.bfloat16
        )
        iq = (
            torch.randn(rows, self.heads, DIM, device="cuda", generator=self.gen) * 0.7
        ).to(torch.float8_e4m3fn)
        return q, iq


def expected_blocks(case, iq, rows_req, rows_causal, sm_scale):
    """Per row: set of the 16 blocks the selection must take (tail + top 15)."""
    out = []
    for r, (b, causal) in enumerate(zip(rows_req, rows_causal)):
        last = (causal - 1) // 128
        per_head = []
        for h in range(case.heads):
            qr = iq[r, h].float()
            sc = []
            for p in range(last):
                kb = case.IK[case.table[b, p].long()]
                sc.append((kb @ qr).max() * sm_scale)
            take = min(TOPK - 1, last)
            chosen = set(torch.stack(sc).topk(take).indices.tolist()) if take else set()
            per_head.append(chosen | {last})
        out.append(per_head)
    return out


def attend_reference(case, q, topk_idx, rows_req, rows_causal, sm_scale):
    """fp64 attention of every row over the keys of its picked blocks."""
    rows, hq, _ = q.shape
    out = torch.zeros(rows, hq, DIM, dtype=torch.float64, device="cuda")
    for r, (b, causal) in enumerate(zip(rows_req, rows_causal)):
        for h in range(case.heads):
            ks, vs = [], []
            for blk in sorted(x for x in topk_idx[h, r].tolist() if x >= 0):
                page = case.table[b, blk].long()
                n = min(128, causal - blk * 128)
                if n <= 0:
                    continue
                ks.append(case.K[page, h, :n].double())
                vs.append(case.V[page, h, :n].double())
            kk, vv = torch.cat(ks), torch.cat(vs)
            qh = q[r, h * case.group : (h + 1) * case.group].double()
            p = torch.softmax(qh @ kk.T * sm_scale, -1)
            out[r, h * case.group : (h + 1) * case.group] = p @ vv
    return out


def rel_l2(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@pytest.mark.parametrize(
    "lens,S,heads,gs",
    [
        ([128 * 40 + 17, 128 * 9 + 128, 300, 128 * 33 - 5], 1, 1, (1.0, 1.0)),
        ([128 * 40 + 17, 128 * 9 + 5, 129], 4, 1, (1.0, 1.0)),
        ([128 * 25, 128 * 17 + 3, 700], 1, 2, (0.37, 1.9)),
    ],
)
def test_decode_select_and_attend(lens, S, heads, gs):
    from aiter.ops.flydsl import (
        minimax_m3_index_score_work_map,
        minimax_m3_index_score_work_map_capacity,
    )

    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import sparse_pa_nvfp4_decode

    case = Case(lens, heads=heads, seed=len(lens) * 7 + S, gs=gs)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    rows = len(lens) * S
    q, iq = case.queries(rows)
    sm = DIM**-0.5
    bound = case.table.shape[1]
    wm = torch.empty(
        (minimax_m3_index_score_work_map_capacity(len(lens), bound, S, heads), 2),
        dtype=torch.int32,
        device="cuda",
    )
    wm = minimax_m3_index_score_work_map(seq, bound, S, heads, out=wm)
    topk = minimax_m3_index_topk_decode(
        iq, case.idx, case.table, seq, bound * 128, TOPK, 0, 1, heads, sm,
        emit_sparse_block_table=False, max_query_len=S, index_nvfp4=True,
        index_work_map=wm, index_work_map_max_block=bound,
    )  # fmt: skip
    assert topk.shape == (heads, rows, TOPK) and topk.dtype == torch.int32
    rows_req = [r // S for r in range(rows)]
    rows_causal = [lens[r // S] - S + r % S + 1 for r in range(rows)]
    for r, want in enumerate(expected_blocks(case, iq, rows_req, rows_causal, sm)):
        for h in range(heads):
            got = {x for x in topk[h, r].tolist() if x >= 0}
            assert got == want[h], f"row {r} head {h}"
    out = torch.empty_like(q)
    sparse_pa_nvfp4_decode(
        q, case.kv, topk, case.table, seq, sm, case.ks, case.vs, S, out
    )
    ref = attend_reference(case, q, topk, rows_req, rows_causal, sm)
    assert rel_l2(out, ref) < 2.5e-3


@pytest.mark.parametrize(
    "qlens,prefix,heads",
    [([37, 200, 5], [0, 128 * 12, 128 * 30 + 3], 1), ([64, 9], [128 * 8, 0], 2)],
)
def test_prefill_select_and_attend(qlens, prefix, heads):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        prefill_row_positions,
        sparse_pa_nvfp4_prefill,
    )

    lens = [p + n for p, n in zip(prefix, qlens)]
    case = Case(lens, heads=heads, seed=11 + len(lens), gs=(1.0, 1.0))
    rows = sum(qlens)
    q, iq = case.queries(rows)
    sm = DIM**-0.5
    cu = torch.tensor([0] + qlens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    pre = torch.tensor(prefix, dtype=torch.int32, device="cuda")
    topk = minimax_m3_index_topk(
        iq, case.idx, case.table, cu, seq, pre, max(qlens), max(lens), TOPK, 0, 1,
        heads, sm, emit_sparse_block_table=False, index_nvfp4=True,
    )  # fmt: skip
    req, kvl = prefill_row_positions(cu, pre, rows)
    rows_req, rows_causal = req.tolist(), kvl.tolist()
    for r, want in enumerate(expected_blocks(case, iq, rows_req, rows_causal, sm)):
        for h in range(heads):
            got = {x for x in topk[h, r].tolist() if x >= 0}
            assert got == want[h], f"row {r} head {h}"
    out = sparse_pa_nvfp4_prefill(
        q, case.kv, topk, case.table, req, kvl, sm, case.ks, case.vs
    )
    ref = attend_reference(case, q, topk, rows_req, rows_causal, sm)
    assert rel_l2(out, ref) < 2.5e-3
