# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`FusedRoute` against the three launches it replaces -- aiter `topk_gating`,
`eplb._eplb_route_kernel`, `remap_topk_to_dispatch` -- bit for bit: ids,
weights and expert-load counts, with pad rows, ties, EPLB map and record on
and off, and forced-remote experts spread over their replicas."""

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


def _layer(rank, bias):
    layout = SimpleNamespace(
        num_routed=EXPERTS,
        num_routed_physical=EXPERTS,
        routed_physical_per_rank=PER_RANK,
    )
    return SimpleNamespace(
        expert_layout=layout,
        e_score_correction_bias=bias,
        top_k=TOP_K,
        num_fused_shared_experts=1,
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
    (1, 0, False, False, False, False),
    (2, 1, False, True, True, False),
    (7, 0, True, True, True, False),
    (16, 5, False, True, False, False),
    (40, 0, False, True, True, True),
    (64, 13, True, True, True, True),
    (128, 0, False, False, False, False),
    (200, 40, False, True, True, True),
    (240, 0, True, True, True, False),
    (256, 85, False, True, True, True),
]


@pytest.mark.parametrize("rows, pad, ties, mapped, record, spread", CASES)
@pytest.mark.parametrize("rank", [0, 5])
def test_routes_match_the_three_launches(
    monkeypatch, rows, pad, ties, mapped, record, spread, rank
):
    torch.manual_seed(rows * 31 + rank)
    logits = (torch.randn(rows, EXPERTS, device="cuda") * (0.5 + rows % 3)).to(
        torch.bfloat16
    )
    if ties:
        logits[:, ::7] = logits[:, :1]
    bias = (torch.randn(EXPERTS, device="cuda") * 0.05).float()
    layer = _layer(rank, bias)
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

    assert torch.equal(ids, id_ref.to(torch.int32))
    assert torch.equal(w.view(torch.int32), w_ref.float().view(torch.int32))
    if load_ref is not None:
        assert torch.equal(tables.load, load_ref)
