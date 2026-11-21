# SPDX-License-Identifier: MIT
"""The one-launch M3.1 decode router (`m31_route`) against what it replaces.

- Selection: per row, the top-4 experts of sigmoid(fp32 logits) + bias, with
  renormalized weights, as an fp32 torch reference computes them.
- Sort: both layouts it hands `fused_moe` equal aiter's own sort of the same
  routes bit for bit -- the aux sort (`moe_sorting(..., output_aux=True)`:
  ids, weights, expert ids, num_valid, m_indices, reverse_sorted) and the
  `moe_sorting` layout (its padding ids carry the slot bits).
- Rows past `n_local` are padding: no routes, reverse_sorted -1, num_valid[1]
  the real count.
- The self-resetting ticket counter survives back-to-back launches and a CUDA
  graph replay; the MoE output it zeroes is zero.
"""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
    pytest.skip("the fused router is gfx950-only", allow_module_level=True)

pytest.importorskip("aiter")

from atom.model_ops.minimax_m3.router import m31_route

E, K, TOPK = 128, 6144, 4


@pytest.fixture(scope="module")
def gate():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(E, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda", generator=g) * 0.05
    return w, bias


def _route(x, w, bias, n_local=None):
    nl = (
        None
        if n_local is None
        else torch.tensor([n_local], dtype=torch.int32, device="cuda")
    )
    out = m31_route(x, w, bias, routed_scaling_factor=1.0, n_local=nl)
    assert out is not None, "fused router failed to build or load"
    torch.cuda.synchronize()
    return out


def _reference(x, w, bias):
    act = torch.sigmoid(x.float() @ w.float().t())
    ids = torch.topk(act + bias, TOPK, dim=-1).indices
    sel = act.gather(1, ids)
    return ids, sel / sel.sum(-1, keepdim=True)


@pytest.mark.parametrize("m", [1, 2, 3, 8, 9, 15, 16])
def test_selection_matches_fp32_reference(gate, m):
    w, bias = gate
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    topk_w, topk_ids, _ = _route(x, w, bias)
    rids, rw = _reference(x, w, bias)
    assert torch.equal(
        torch.sort(topk_ids.long(), -1).values, torch.sort(rids, -1).values
    )
    for r in range(m):
        for s in range(TOPK):
            j = (rids[r] == int(topk_ids[r, s])).nonzero().item()
            assert abs(float(topk_w[r, s]) - float(rw[r, j])) < 2e-6


@pytest.mark.parametrize("m", [1, 2, 8, 16])
def test_sort_equals_aiter_sort(gate, m):
    from aiter.fused_moe import moe_sorting

    w, bias = gate
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    tw, ti, ps = _route(x, w, bias)
    aux = moe_sorting(ti, tw, E, K, torch.bfloat16, 16, output_aux=True)
    plain = moe_sorting(ti, tw, E, K, torch.bfloat16, 16, None, None)
    torch.cuda.synchronize()
    nv = int(aux[3][0])
    assert ps["num_valid_ids"].tolist() == aux[3].tolist() == plain[3].tolist()
    assert torch.equal(ps["sorted_token_ids"][:nv], aux[0][:nv])
    assert torch.equal(ps["sorted_weights"][:nv], aux[1][:nv])
    assert torch.equal(ps["sorted_expert_ids"][: nv // 16], aux[2][: nv // 16])
    assert torch.equal(ps["m_indices"][:nv], aux[5][:nv])
    assert torch.equal(ps["reverse_sorted"], aux[6])
    assert torch.equal(ps["sorted_token_ids_moe_sorting"][:nv], plain[0][:nv])


@pytest.mark.parametrize("m,n_local", [(2, 1), (8, 5), (16, 13)])
def test_padding_rows_carry_no_routes(gate, m, n_local):
    w, bias = gate
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    tw, _, ps = _route(x, w, bias, n_local)
    assert ps["num_valid_ids"][1].item() == n_local
    rev = ps["reverse_sorted"].view(m, TOPK)
    assert bool((rev[n_local:] == -1).all()) and bool((rev[:n_local] >= 0).all())
    assert bool((tw[n_local:] == 0).all())
    ids = ps["sorted_token_ids"][: int(ps["num_valid_ids"][0])]
    real = ids[ids != m] & 0xFFFFFF
    assert sorted(real.tolist()) == sorted(
        t for t in range(n_local) for _ in range(TOPK)
    )


def test_repeat_and_graph_replay(gate):
    w, bias = gate
    x = torch.randn(16, K, device="cuda").to(torch.bfloat16)
    first = _route(x, w, bias)
    for _ in range(10):
        again = _route(x, w, bias)
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = m31_route(x, w, bias)
    out[2]["moe_buf"].fill_(1.0)
    x.copy_(torch.randn(16, K, device="cuda").to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    rids, _ = _reference(x, w, bias)
    assert torch.equal(
        torch.sort(out[1].long(), -1).values, torch.sort(rids, -1).values
    )
    assert bool((out[2]["moe_buf"] == 0).all())
