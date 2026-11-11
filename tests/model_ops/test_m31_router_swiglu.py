# SPDX-License-Identifier: MIT
"""The 1..128-row M3.1 router + shared SwiGLU (`m31_route_swiglu`) against
what it replaces.

- SwiGLU: the values and scales equal `swiglu_oai_split_mxfp8` byte for byte.
- Selection: per row, the top-4 experts of sigmoid(fp32 logits) + bias, with
  renormalized weights, as an fp32 torch reference computes them.
- Sort: both layouts equal aiter's own sort of the same routes bit for bit
  (`moe_sorting(..., output_aux=True)` and plain `moe_sorting`).
- Rows past `n_local` are padding: no routes, reverse_sorted -1, num_valid[1]
  the real count; the MoE output is zeroed, also on a CUDA graph replay.
"""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
    pytest.skip("the fused router is gfx950-only", allow_module_level=True)

pytest.importorskip("aiter")

from atom.model_ops.minimax_m3.router.router_swiglu import (
    m31_route_swiglu,
    router_swiglu_supported,
)
from atom.model_ops.swiglu_oai import swiglu_oai_split_mxfp8

E, K, TOPK, INTER = 128, 6144, 4, 768
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0


@pytest.fixture(scope="module")
def gate():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(E, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda", generator=g) * 0.05
    return w, bias


def _inputs(m, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, device="cuda", generator=g).to(torch.bfloat16)
    gu = (torch.randn(m, 2 * INTER, device="cuda", generator=g) * 3).to(torch.bfloat16)
    return x, gu


def _route(x, w, bias, gu, n_local=None):
    assert router_swiglu_supported(x, w, bias, gu)
    nl = (
        None
        if n_local is None
        else torch.tensor([n_local], dtype=torch.int32, device="cuda")
    )
    out = m31_route_swiglu(x, w, bias, gu, ALPHA, BETA, LIMIT, 1.0, nl)
    assert out is not None
    torch.cuda.synchronize()
    return out


def _reference(x, w, bias):
    act = torch.sigmoid(x.float() @ w.float().t())
    ids = torch.topk(act + bias, TOPK, dim=-1).indices
    sel = act.gather(1, ids)
    return ids, sel / sel.sum(-1, keepdim=True)


@pytest.mark.parametrize("m", [1, 2, 8, 16, 17, 24, 32, 40, 64, 96, 128])
def test_swiglu_matches_separate_kernel(gate, m):
    w, bias = gate
    x, gu = _inputs(m)
    *_, (q, s) = _route(x, w, bias, gu)
    rq, rs = swiglu_oai_split_mxfp8(gu, ALPHA, BETA, LIMIT)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8), rs.view(torch.uint8))


@pytest.mark.parametrize("m", [1, 3, 8, 16, 17, 24, 32, 48, 64, 100, 128])
def test_selection_matches_fp32_reference(gate, m):
    w, bias = gate
    x, gu = _inputs(m)
    topk_w, topk_ids, _, _ = _route(x, w, bias, gu)
    rids, rw = _reference(x, w, bias)
    assert torch.equal(
        torch.sort(topk_ids.long(), -1).values, torch.sort(rids, -1).values
    )
    pos = (rids[:, None, :] == topk_ids.long()[:, :, None]).float().argmax(-1)
    assert torch.allclose(topk_w, rw.gather(1, pos), atol=2e-6, rtol=0)


def _canonical(ids, weights, eids, nv):
    """Per expert, its rows sorted by id (pads, the largest ids, last).
    Past 16 rows aiter's own sort orders an expert's pairs differently from
    run to run, so only this form is comparable."""
    ids, weights, eids = ids[:nv].cpu(), weights[:nv].cpu(), eids[: nv // 16].cpu()
    rows = eids.repeat_interleave(16)
    order = torch.argsort(rows.long() << 32 | ids.long(), stable=True)
    return rows[order], ids[order], weights[order]


@pytest.mark.parametrize("m", [1, 2, 8, 16, 17, 24, 32, 64, 128])
def test_sort_equals_aiter_sort(gate, m):
    from aiter.fused_moe import moe_sorting

    w, bias = gate
    x, gu = _inputs(m)
    tw, ti, ps, _ = _route(x, w, bias, gu)
    aux = moe_sorting(ti, tw, E, K, torch.bfloat16, 16, output_aux=True)
    plain = moe_sorting(ti, tw, E, K, torch.bfloat16, 16, None, None)
    torch.cuda.synchronize()
    nv = int(aux[3][0])
    assert ps["num_valid_ids"].tolist() == aux[3].tolist() == plain[3].tolist()
    eids = ps["sorted_expert_ids"]
    assert torch.equal(eids[: nv // 16], aux[2][: nv // 16])
    mine = _canonical(ps["sorted_token_ids"], ps["sorted_weights"], eids, nv)
    theirs = _canonical(aux[0], aux[1], aux[2], nv)
    for a, b in zip(mine, theirs):
        assert torch.equal(a, b)
    mine_ms = _canonical(
        ps["sorted_token_ids_moe_sorting"], ps["sorted_weights"], eids, nv
    )
    theirs_ms = _canonical(plain[0], plain[1], plain[2], nv)
    for a, b in zip(mine_ms, theirs_ms):
        assert torch.equal(a, b)
    # the aux extras agree with this sort's own order
    ids = ps["sorted_token_ids"][:nv]
    live = ids != m
    assert torch.equal(ps["m_indices"][:nv], torch.where(live, ids & 0xFFFFFF, m))
    route = (ids & 0xFFFFFF) * TOPK + (ids >> 24)
    pos = torch.arange(nv, device="cuda", dtype=torch.int32)
    rev = torch.full((m * TOPK,), -1, dtype=torch.int32, device="cuda")
    rev[route[live].long()] = pos[live]
    assert torch.equal(ps["reverse_sorted"], rev)
    # pads share their rows with nothing; the live pairs keep (token, slot) order
    assert torch.equal(
        ids == m, ps["sorted_token_ids_moe_sorting"][:nv] == ((TOPK << 24) | m)
    )
    assert bool((ps["sorted_weights"][:nv][~live] == 0).all())
    assert bool((ps["moe_buf"] == 0).all())


def test_sort_is_deterministic(gate):
    w, bias = gate
    x, gu = _inputs(128)
    first = _route(x, w, bias, gu)[2]
    nv = int(first["num_valid_ids"][0])
    for _ in range(5):
        again = _route(x, w, bias, gu)[2]
        for key in ("sorted_token_ids", "sorted_weights", "m_indices"):
            assert torch.equal(first[key][:nv], again[key][:nv])
        assert torch.equal(first["reverse_sorted"], again["reverse_sorted"])


@pytest.mark.parametrize(
    "m,n_local", [(2, 1), (16, 13), (17, 9), (32, 25), (64, 33), (128, 120)]
)
def test_padding_rows_carry_no_routes(gate, m, n_local):
    w, bias = gate
    x, gu = _inputs(m)
    tw, _, ps, _ = _route(x, w, bias, gu, n_local)
    assert ps["num_valid_ids"][1].item() == n_local
    rev = ps["reverse_sorted"].view(m, TOPK)
    assert bool((rev[n_local:] == -1).all()) and bool((rev[:n_local] >= 0).all())
    assert bool((tw[n_local:] == 0).all())
    ids = ps["sorted_token_ids"][: int(ps["num_valid_ids"][0])]
    real = ids[ids != m] & 0xFFFFFF
    assert sorted(real.tolist()) == sorted(
        t for t in range(n_local) for _ in range(TOPK)
    )
    # every padded row is the pad id in both layouts
    ms = ps["sorted_token_ids_moe_sorting"][: int(ps["num_valid_ids"][0])]
    assert bool(((ids == m) == (ms == ((TOPK << 24) | m))).all())


def test_graph_replay(gate):
    w, bias = gate
    x, gu = _inputs(32)
    _route(x, w, bias, gu)  # workspace outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = m31_route_swiglu(x, w, bias, gu, ALPHA, BETA, LIMIT)
    out[2]["moe_buf"].fill_(1.0)
    x2, gu2 = _inputs(32, seed=7)
    x.copy_(x2)
    gu.copy_(gu2)
    graph.replay()
    torch.cuda.synchronize()
    rids, _ = _reference(x, w, bias)
    assert torch.equal(
        torch.sort(out[1].long(), -1).values, torch.sort(rids, -1).values
    )
    assert bool((out[2]["moe_buf"] == 0).all())
    rq, _ = swiglu_oai_split_mxfp8(gu, ALPHA, BETA, LIMIT)
    assert torch.equal(out[3][0].view(torch.uint8), rq.view(torch.uint8))
