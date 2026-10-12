# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which dispatch kernel MegaMoE runs on, and the VMM its compact plan needs.

mega_transport_config holds these rules without aiter, so they run on CPU CI.
"""

import pytest

from atom.model_ops.fused_moe.mega_transport_config import (
    compact_plan_vmm_bytes,
    mega_dispatch_kwargs,
)


@pytest.mark.parametrize("backend", ["mori", "flydsl"])
def test_bf16_wire_leaves_the_backend_to_aiter(backend):
    assert mega_dispatch_kwargs("bf16", backend) == {}


@pytest.mark.parametrize("wire", ["fp8", "fp4"])
def test_mori_quant_wire_runs_mori(wire):
    assert mega_dispatch_kwargs(wire, "mori") == {"dispatch_backend": "mori"}


@pytest.mark.parametrize("wire", ["fp8", "fp4"])
def test_flydsl_quant_wire_runs_the_compact_plan(wire):
    assert mega_dispatch_kwargs(wire, "flydsl") == {
        "dispatch_backend": "flydsl",
        "stage1_fused": True,
    }


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="MEGA_DISPATCH"):
        mega_dispatch_kwargs("fp4", "deepep")


def _compact_row_capacity(*, max_recv, topk, experts_per_rank, tile_m):
    # aiter's compact_plan.compact_row_capacity: the rows the arena is cut for.
    ub = max_recv * topk + experts_per_rank * tile_m - topk
    return max(tile_m, (ub + tile_m - 1) // tile_m * tile_m)


@pytest.mark.parametrize("tile_m", [16, 32, 64, 128, 256])
@pytest.mark.parametrize(
    ("wire", "wire_row"),
    # payload (fp4 packs two per byte) + hidden/32 e8m0 scales, 128-B aligned
    [("fp4", 1920), ("fp8", 3712)],
)
@pytest.mark.parametrize(
    ("ep_size", "max_tokens_per_rank", "topk", "num_experts"),
    [(8, 2048, 16, 896), (8, 128, 16, 896), (16, 1024, 8, 256), (72, 64, 16, 1152)],
)
def test_compact_vmm_covers_the_compact_rows(
    tile_m, wire, wire_row, ep_size, max_tokens_per_rank, topk, num_experts
):
    hidden_dim = 3584
    rows = _compact_row_capacity(
        max_recv=ep_size * max_tokens_per_rank,
        topk=topk,
        experts_per_rank=num_experts // ep_size,
        tile_m=tile_m,
    )
    disp_out_and_rowmap = rows * wire_row + (rows + 1) * 8
    assert disp_out_and_rowmap <= compact_plan_vmm_bytes(
        ep_size=ep_size,
        hidden_dim=hidden_dim,
        max_tokens_per_rank=max_tokens_per_rank,
        topk=topk,
        num_experts=num_experts,
        dispatch_wire=wire,
    )


def test_compact_vmm_at_k3_dp8ep8():
    # 8 * 2048 * 16 routes + 113 tiles of 256, each 1920 + 8 bytes: ~535 MiB on
    # top of the 736 MiB token-major budget, which the ~753 MiB arena outgrows.
    assert (
        compact_plan_vmm_bytes(
            ep_size=8,
            hidden_dim=3584,
            max_tokens_per_rank=2048,
            topk=16,
            num_experts=896,
            dispatch_wire="fp4",
        )
        == (8 * 2048 * 16 + 113 * 256) * 1928
    )
