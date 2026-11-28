# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`FusedRoute` against the three launches it replaces -- aiter `topk_gating`,
`eplb._eplb_route_kernel`, `remap_topk_to_dispatch` -- bit for bit: ids,
weights and expert-load counts, with pad rows, ties (across and within a
wave lane), NaN and +Inf rows, EPLB map and record on and off, forced-remote experts
spread over their replicas, with this rank's shared column (LOCAL_REPLICA) and
without a shared expert (no dispatch remap)."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("requires a real ROCm GPU", allow_module_level=True)

aiter = pytest.importorskip("aiter")

from atom.model_ops import eplb
from atom.model_ops.fused_moe import fused_route
from atom.model_ops.fused_moe.shared_expert_dispatch import (
    remap_topk_to_dispatch,
)
from atom.utils import forward_context

EXPERTS, TOP_K, EP, SCALE = 384, 6, 8, 1.5
PER_RANK = EXPERTS // EP


def _layer(rank, bias, num_shared):
    layout = SimpleNamespace(
        num_routed=EXPERTS,
        num_routed_physical=EXPERTS,
        routed_physical_per_rank=PER_RANK,
    )
    return SimpleNamespace(
        expert_layout=layout,
        e_score_correction_bias=bias,
        top_k=TOP_K,
        num_fused_shared_experts=num_shared,
        routed_scaling_factor=SCALE,
        shared_dispatch_base=rank * (PER_RANK + 1) + PER_RANK,
        shared_expert_weight=1.0,
    )


def _tables(spread, record):
    """A random logical -> physical placement; with `spread` a quarter of the
    experts are forced remote (dispatch -1) over two replicas."""
    perm = torch.randperm(EXPERTS, device="cuda").to(torch.int32)
    dispatch = perm.clone()
    l2p = perm.view(EXPERTS, 1).clone()
    cnt = torch.ones(EXPERTS, dtype=torch.int32, device="cuda")
    if spread:
        remote = torch.randperm(EXPERTS, device="cuda")[: EXPERTS // 4]
        dispatch[remote] = -1
        l2p = torch.cat((l2p, perm.roll(1).view(EXPERTS, 1)), dim=1)
        cnt[remote] = 2
    return eplb.EplbRouteTables(
        dispatch=dispatch,
        l2p=l2p.contiguous(),
        cnt=cnt,
        load=torch.zeros(EXPERTS, dtype=torch.int32, device="cuda") if record else None,
        num_logical=EXPERTS,
        id_delta=0,
        num_physical=EXPERTS,
        num_replicas=int(l2p.shape[1]),
    )


def _reference(layer, logits, num_valid, tables, dp):
    ids = torch.empty(logits.shape[0], TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.empty(logits.shape[0], TOP_K, dtype=torch.float32, device="cuda")
    aiter.topk_gating(
        weights,
        ids,
        logits,
        layer.e_score_correction_bias,
        True,
        SCALE,
        score_func="sqrtsoftplus",
        num_valid_rows=num_valid,
    )
    load = None
    phys = ids
    if tables is not None:
        phys = torch.empty_like(ids)
        record = tables.load is not None
        load = torch.zeros_like(tables.load) if record else None
        numel = ids.numel()
        eplb._eplb_route_kernel[(triton_cdiv(numel, 256),)](
            ids,
            None,
            phys,
            None,
            tables.dispatch,
            tables.l2p,
            tables.cnt,
            load,
            tables.num_logical,
            tables.id_delta,
            tables.num_physical,
            tables.num_replicas,
            numel,
            0,
            0.0,
            *dp,
            BLOCK=256,
            TOP_K=TOP_K,
            NUM_SHARED=0,
            MAP=True,
            RECORD=record,
            NUM_BINS=1 << tables.num_physical.bit_length() if record else 1,
        )
    if not layer.num_fused_shared_experts:
        # No shared expert in the routed call: `to_dispatch_space` is the
        # identity.
        return weights, phys, load
    w, i = remap_topk_to_dispatch(
        weights,
        phys,
        EXPERTS,
        PER_RANK,
        layer.shared_dispatch_base,
        1,
        layer.shared_expert_weight,
    )
    return w, i, load


def triton_cdiv(a, b):
    return (a + b - 1) // b


CASES = [
    # rows, pad rows, ties, EPLB map, record, forced-remote spread
    (1, 0, None, False, False, False),
    (2, 1, None, True, True, False),
    (7, 0, "across", True, True, False),
    (16, 5, None, True, False, False),
    (24, 0, "lane", True, True, False),
    (40, 0, None, True, True, True),
    (64, 13, "across", True, True, True),
    (96, 0, "lane", True, True, True),
    (100, 0, "nan", True, True, False),
    (128, 0, None, False, False, False),
    (128, 37, "lane", True, True, True),
]


def _with_ties(logits, ties):
    if ties == "across":
        logits[:, ::7] = logits[:, :1]
    elif ties == "lane":
        # A few logit levels (with a lane-periodic bias, below): equal values
        # among a wave lane's six experts (e % 64 == lane) everywhere, which
        # the compare-swaps of aiter's sort_network_desc<6> reorder.
        logits = (torch.randint_like(logits, 0, 4) * 0.5).to(logits.dtype)
    elif ties == "nan":
        # Whole-row NaN logits (e.g. a warmup batch's), and whole-row +Inf.
        logits[::9] = float("nan")
        logits[4::9] = float("inf")
    return logits


@pytest.mark.parametrize("rows, pad, ties, mapped, record, spread", CASES)
@pytest.mark.parametrize("rank", [0, 5])
@pytest.mark.parametrize("num_shared", [1, 0])
def test_routes_match_the_separate_launches(
    monkeypatch, rows, pad, ties, mapped, record, spread, rank, num_shared
):
    torch.manual_seed(rows * 31 + rank)
    logits = (torch.randn(rows, EXPERTS, device="cuda") * (0.5 + rows % 3)).to(
        torch.bfloat16
    )
    logits = _with_ties(logits, ties)
    bias = (torch.randn(EXPERTS, device="cuda") * 0.05).float()
    if ties == "lane":
        bias = bias[:64].repeat(EXPERTS // 64)
    layer = _layer(rank, bias, num_shared)
    num_valid = (
        torch.tensor([rows - pad], dtype=torch.int32, device="cuda") if pad else None
    )
    tables = _tables(spread, record) if mapped else None
    dp = (EP, rank)
    monkeypatch.setattr(eplb, "eplb_route_tables", lambda *_: tables)
    monkeypatch.setattr(eplb, "_replica_spread", lambda gathered=False: dp)
    monkeypatch.setattr(
        forward_context, "pass_scheduled_tokens_gpu", lambda _rows: num_valid
    )

    w_ref, id_ref, load_ref = _reference(layer, logits, num_valid, tables, dp)
    w, ids = fused_route.FusedRoute(layer)(logits)
    torch.cuda.synchronize()

    assert ids.shape == (rows, TOP_K + num_shared)
    assert torch.equal(ids, id_ref.to(torch.int32))
    assert torch.equal(w.view(torch.int32), w_ref.float().view(torch.int32))
    if load_ref is not None:
        assert torch.equal(tables.load, load_ref)


def test_wider_steps_keep_the_separate_launches():
    layer = _layer(0, torch.zeros(EXPERTS, device="cuda"), 0)
    route = fused_route.FusedRoute(layer)
    assert route.runs(torch.zeros(fused_route.MAX_ROWS, EXPERTS, device="cuda"))
    assert route(torch.zeros(fused_route.MAX_ROWS + 1, EXPERTS, device="cuda")) is None
