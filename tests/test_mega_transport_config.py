# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which dispatch kernel MegaMoE runs on, and the VMM its compact plan needs.

mega_transport_config holds these rules without aiter, so they run on CPU CI.
"""

import pytest

from atom.model_ops.fused_moe.mega_transport_config import (
    compact_plan_vmm_bytes,
    mega_dispatch_kwargs,
    resolve_dispatch_kernel,
)

QUANT_WIRES = ["fp8", "fp4"]

# Kimi-K3's routed MoE: 896 experts, top-16, hidden 3584.
K3 = {"num_experts": 896, "topk": 16, "hidden_dim": 3584}
# A MoE shape with no measurement (DeepSeek-R1's).
UNMEASURED = {"num_experts": 256, "topk": 8, "hidden_dim": 7168}


def _resolve(
    wire="fp4",
    mega_dispatch=None,
    needs_token_major=False,
    max_tokens_per_rank=16384,
    **shape,
):
    return resolve_dispatch_kernel(
        dispatch_wire=wire,
        mega_dispatch=mega_dispatch,
        needs_token_major=needs_token_major,
        max_tokens_per_rank=max_tokens_per_rank,
        **{**K3, **shape},
    )


@pytest.mark.parametrize(
    ("wire", "expected"),
    [("fp8", "mori"), ("fp4", "mori"), ("bf16", "flydsl_token_major")],
)
def test_unmeasured_shapes_keep_the_defaults(wire, expected):
    assert _resolve(wire, **UNMEASURED) == (expected, "default")


@pytest.mark.parametrize(
    ("wire", "mega_dispatch", "expected"),
    [
        ("fp4", "mori", "mori"),
        ("fp8", "mori", "mori"),
        ("bf16", "mori", "mori"),
        ("fp4", "flydsl", "flydsl_compact"),
        ("fp8", "flydsl", "flydsl_compact"),
        ("bf16", "flydsl", "flydsl_token_major"),
    ],
)
def test_mega_dispatch_overrides_the_choice(wire, mega_dispatch, expected):
    assert _resolve(wire, mega_dispatch) == (expected, "MEGA_DISPATCH")


def test_unknown_mega_dispatch_is_rejected():
    with pytest.raises(ValueError, match="MEGA_DISPATCH"):
        _resolve(mega_dispatch="deepep")


@pytest.mark.parametrize("wire", QUANT_WIRES)
def test_mega_dispatch_flydsl_is_rejected_under_triton_decode(wire):
    with pytest.raises(ValueError, match="ATOM_USE_TRITON_MOE_DECODE"):
        _resolve(wire, "flydsl", needs_token_major=True)


def test_mega_dispatch_mori_is_accepted_under_triton_decode():
    assert _resolve(mega_dispatch="mori", needs_token_major=True) == (
        "mori",
        "MEGA_DISPATCH",
    )


@pytest.mark.parametrize("max_tokens_per_rank", [1, 2048, 16384])
def test_k3_on_an_fp4_wire_runs_the_compact_plan(max_tokens_per_rank):
    assert _resolve(max_tokens_per_rank=max_tokens_per_rank) == (
        "flydsl_compact",
        "measured",
    )


def test_k3_past_the_measured_capacity_keeps_the_default():
    assert _resolve(max_tokens_per_rank=16385) == ("mori", "default")


@pytest.mark.parametrize(
    ("wire", "expected"), [("fp8", "mori"), ("bf16", "flydsl_token_major")]
)
def test_k3_on_an_unmeasured_wire_keeps_the_default(wire, expected):
    assert _resolve(wire) == (expected, "default")


@pytest.mark.parametrize(
    "shape", [{"num_experts": 384}, {"topk": 8}, {"hidden_dim": 7168}]
)
def test_a_shape_close_to_k3_keeps_the_default(shape):
    assert _resolve(**shape) == ("mori", "default")


def test_triton_decode_keeps_k3_off_the_compact_plan():
    assert _resolve(needs_token_major=True) == ("mori", "default")


def test_mega_dispatch_beats_the_measurement():
    assert _resolve(mega_dispatch="mori") == ("mori", "MEGA_DISPATCH")


def test_the_kernel_env_var_is_gone():
    from atom.utils import envs

    assert not hasattr(envs, "ATOM_MEGA_DISPATCH_KERNEL")


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
