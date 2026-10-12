# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""How ATOM configures aiter's MegaMoEGfx1250 transport.

Which dispatch kernel runs on a quantizing wire and how much cco VMM its arena
then needs. Kept free of aiter and mori imports so the rules are testable on
CPU-only CI.
"""

_QUANT_WIRES = ("fp8", "fp4")
_DISPATCH_BACKENDS = ("mori", "flydsl")

# Upper bound on the expert GEMM's tile_m, which pads each local expert's slice
# of the compact plan. aiter picks it from its tuned grouped-GEMM table, whose
# widest tile is 256.
_MAX_COMPACT_TILE_M = 256
# The compact plan's row map: one 8-byte entry per dispatch row.
_ROWMAP_ENTRY_BYTES = 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def mega_dispatch_kwargs(dispatch_wire: str, dispatch_backend: str) -> dict:
    """MegaMoEGfx1250 constructor kwargs that pin its dispatch kernel.

    A quantizing wire (fp8/fp4) runs on ``dispatch_backend`` ($MEGA_DISPATCH,
    mori unless set): aiter's flydsl TDM dispatch carries the per-token scale
    row as mori's does. It is named explicitly because aiter's own read of
    $MEGA_DISPATCH defaults to flydsl. A bf16 wire passes nothing, leaving that
    read to aiter as before.

    flydsl on a quantizing wire runs aiter's compact plan (``stage1_fused``):
    its token-major path gave wrong, nondeterministic output there (Kimi-K3
    dp8ep8, aiter e965ebfe), while the compact plan matched mori.
    """
    if dispatch_backend not in _DISPATCH_BACKENDS:
        raise ValueError(
            f"MEGA_DISPATCH must be one of {_DISPATCH_BACKENDS}, "
            f"got {dispatch_backend!r}"
        )
    if dispatch_wire not in _QUANT_WIRES:
        return {}
    kwargs = {"dispatch_backend": dispatch_backend}
    if dispatch_backend == "flydsl":
        kwargs["stage1_fused"] = True
    return kwargs


def compact_plan_vmm_bytes(
    *,
    ep_size: int,
    hidden_dim: int,
    max_tokens_per_rank: int,
    topk: int,
    num_experts: int,
    dispatch_wire: str,
) -> int:
    """cco VMM per rank for the compact plan's dispatch rows, on top of the
    token-major budget.

    A token-major arena has one dispatch row per received token. The compact
    plan has one per route, grouped per local expert and padded to the expert
    GEMM's tile: ``ep_size * max_tokens_per_rank * topk`` rows plus up to a tile
    per local expert, and one more for aligning the total. Each row is the wire
    payload plus its e8m0 scales, 128-byte aligned, plus a row-map entry. The
    plan's histogram and counters (tens of KiB) fit in the token-major rows it no
    longer allocates. At K3 dp8ep8 (M=2048, topk 16, 112 experts per rank,
    hidden 3584, fp4) the compact arena is ~753 MiB, past the 736 MiB budget.
    """
    payload = hidden_dim // 2 if dispatch_wire == "fp4" else hidden_dim
    wire_row = _align_up(payload + hidden_dim // 32, 128)
    experts_per_rank = num_experts // ep_size
    rows = (
        ep_size * max_tokens_per_rank * topk
        + (experts_per_rank + 1) * _MAX_COMPACT_TILE_M
    )
    return rows * (wire_row + _ROWMAP_ENTRY_BYTES)
