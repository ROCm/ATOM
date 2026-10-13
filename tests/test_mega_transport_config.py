# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which dispatch kernel MegaMoE runs on, and the VMM its compact plan needs.

mega_transport_config holds these rules without aiter, so they run on CPU CI.
"""

import pytest

from atom.model_ops.fused_moe.mega_transport_config import (
    DISPATCH_KERNELS,
    compact_plan_vmm_bytes,
    dispatch_kernel_warning,
    mega_dispatch_kwargs,
    resolve_dispatch_kernel,
)

QUANT_WIRES = ["fp8", "fp4"]


@pytest.mark.parametrize("wire", QUANT_WIRES)
@pytest.mark.parametrize("kernel", sorted(DISPATCH_KERNELS))
def test_every_kernel_is_selectable_on_a_quant_wire(wire, kernel):
    assert resolve_dispatch_kernel(wire, kernel, None) == kernel


@pytest.mark.parametrize("kernel", ["mori", "flydsl_token_major"])
def test_token_major_kernels_are_selectable_on_bf16(kernel):
    assert resolve_dispatch_kernel("bf16", kernel, None) == kernel


def test_compact_on_bf16_is_rejected():
    with pytest.raises(ValueError, match="fp8 or fp4"):
        resolve_dispatch_kernel("bf16", "flydsl_compact", None)


@pytest.mark.parametrize(
    ("wire", "expected"),
    [("fp8", "mori"), ("fp4", "mori"), ("bf16", "flydsl_token_major")],
)
def test_defaults_are_unchanged(wire, expected):
    assert resolve_dispatch_kernel(wire, None, None) == expected


@pytest.mark.parametrize(
    ("wire", "legacy", "expected"),
    [
        ("fp4", "mori", "mori"),
        ("fp8", "mori", "mori"),
        ("bf16", "mori", "mori"),
        ("fp4", "flydsl", "flydsl_compact"),
        ("fp8", "flydsl", "flydsl_compact"),
        ("bf16", "flydsl", "flydsl_token_major"),
    ],
)
def test_legacy_mega_dispatch_keeps_its_meaning(wire, legacy, expected):
    assert resolve_dispatch_kernel(wire, None, legacy) == expected


@pytest.mark.parametrize(
    ("kernel", "legacy"),
    [("flydsl_token_major", "flydsl"), ("flydsl_compact", "flydsl"), ("mori", "mori")],
)
def test_kernel_with_a_matching_legacy_backend_is_accepted(kernel, legacy):
    assert resolve_dispatch_kernel("fp4", kernel, legacy) == kernel


@pytest.mark.parametrize(
    ("kernel", "legacy"),
    [("mori", "flydsl"), ("flydsl_compact", "mori"), ("flydsl_token_major", "mori")],
)
def test_kernel_and_legacy_backend_disagreeing_is_rejected(kernel, legacy):
    with pytest.raises(ValueError, match="disagrees"):
        resolve_dispatch_kernel("fp4", kernel, legacy)


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="ATOM_MEGA_DISPATCH_KERNEL"):
        resolve_dispatch_kernel("fp4", "deepep", None)


def test_unknown_legacy_backend_is_rejected():
    with pytest.raises(ValueError, match="MEGA_DISPATCH"):
        resolve_dispatch_kernel("fp4", None, "deepep")


@pytest.mark.parametrize(
    ("kernel", "kwargs"),
    [
        ("mori", {"dispatch_backend": "mori"}),
        ("flydsl_token_major", {"dispatch_backend": "flydsl"}),
        ("flydsl_compact", {"dispatch_backend": "flydsl", "stage1_fused": True}),
    ],
)
def test_kwargs_name_the_kernel(kernel, kwargs):
    assert mega_dispatch_kwargs(kernel) == kwargs


@pytest.mark.parametrize("wire", QUANT_WIRES)
def test_token_major_flydsl_on_a_quant_wire_warns(wire):
    assert "wrong" in dispatch_kernel_warning("flydsl_token_major", wire)


@pytest.mark.parametrize(
    ("kernel", "wire"),
    [
        ("mori", "fp4"),
        ("flydsl_compact", "fp4"),
        ("flydsl_token_major", "bf16"),
        ("mori", "bf16"),
    ],
)
def test_other_combinations_do_not_warn(kernel, wire):
    assert dispatch_kernel_warning(kernel, wire) is None


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), ("", None), ("mori", "mori")]
)
def test_kernel_env_var(monkeypatch, value, expected):
    from atom.utils import envs

    if value is None:
        monkeypatch.delenv("ATOM_MEGA_DISPATCH_KERNEL", raising=False)
    else:
        monkeypatch.setenv("ATOM_MEGA_DISPATCH_KERNEL", value)
    assert envs.ATOM_MEGA_DISPATCH_KERNEL == expected


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
    # 8 * 2048 * 16 routes + 113 tiles of 256, each 1920 + 8 bytes: ~535 MiB of
    # compact rows on top of the token-major budget.
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
