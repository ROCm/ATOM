# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""How ATOM configures aiter's MegaMoEGfx1250 transport.

Which dispatch kernel it runs and how much cco VMM its arena then needs. Kept free
of aiter and mori imports so the rules are testable on CPU-only CI.
"""

_QUANT_WIRES = ("fp8", "fp4")
_DISPATCH_BACKENDS = ("mori", "flydsl")

# Every dispatch kernel MegaMoEGfx1250 can run, by the name
# $ATOM_MEGA_DISPATCH_KERNEL takes: (aiter's dispatch_backend, stage1_fused).
#   mori                mori's HIP/JIT dispatch, token-major
#   flydsl_token_major  aiter's FlyDSL TDM dispatch, token-major
#   flydsl_compact      the same dispatch fed by aiter's compact plan: each route
#                       lands as its own row, grouped per local expert
DISPATCH_KERNELS = {
    "mori": ("mori", False),
    "flydsl_token_major": ("flydsl", False),
    "flydsl_compact": ("flydsl", True),
}

# Upper bound on the expert GEMM's tile_m, which pads each local expert's slice
# of the compact plan. aiter picks it from its tuned grouped-GEMM table, whose
# widest tile is 256.
_MAX_COMPACT_TILE_M = 256
# The compact plan's row map: one 8-byte entry per dispatch row.
_ROWMAP_ENTRY_BYTES = 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def resolve_dispatch_kernel(
    dispatch_wire: str, kernel: str | None, legacy_backend: str | None
) -> str:
    """The dispatch kernel a transport on ``dispatch_wire`` runs.

    ``kernel`` ($ATOM_MEGA_DISPATCH_KERNEL) names it outright, on any wire. Unset,
    aiter's older backend knob ``legacy_backend`` ($MEGA_DISPATCH) still selects
    what it did before: mori -> mori; flydsl -> the compact plan on an fp8/fp4
    wire, token-major on bf16. With neither set, an fp8/fp4 wire runs mori and a
    bf16 wire flydsl token-major, the previous defaults.

    Set together, the two must name the same backend: a launch script carrying an
    old MEGA_DISPATCH next to a new kernel name would otherwise run one of them
    without saying which.
    """
    quant = dispatch_wire in _QUANT_WIRES
    if legacy_backend is not None and legacy_backend not in _DISPATCH_BACKENDS:
        raise ValueError(
            f"MEGA_DISPATCH must be one of {_DISPATCH_BACKENDS}, "
            f"got {legacy_backend!r}"
        )
    if kernel is None:
        if legacy_backend is None:
            kernel = "mori" if quant else "flydsl_token_major"
        elif legacy_backend == "mori":
            kernel = "mori"
        else:
            kernel = "flydsl_compact" if quant else "flydsl_token_major"
    elif kernel not in DISPATCH_KERNELS:
        raise ValueError(
            f"ATOM_MEGA_DISPATCH_KERNEL must be one of {tuple(DISPATCH_KERNELS)}, "
            f"got {kernel!r}"
        )
    elif legacy_backend is not None and legacy_backend != DISPATCH_KERNELS[kernel][0]:
        raise ValueError(
            f"MEGA_DISPATCH={legacy_backend} disagrees with "
            f"ATOM_MEGA_DISPATCH_KERNEL={kernel}; unset MEGA_DISPATCH, which "
            "ATOM_MEGA_DISPATCH_KERNEL replaces"
        )
    if DISPATCH_KERNELS[kernel][1] and not quant:
        raise ValueError(
            f"ATOM_MEGA_DISPATCH_KERNEL={kernel} needs an fp8 or fp4 "
            f"MEGA_DISPATCH_WIRE, got {dispatch_wire!r}: the compact plan carries "
            "each row's e8m0 scales inside the row"
        )
    return kernel


def mega_dispatch_kwargs(kernel: str) -> dict:
    """MegaMoEGfx1250 constructor kwargs that run ``kernel``.

    The backend is passed on every wire, so aiter never falls back to its own
    read of $MEGA_DISPATCH and the transport runs what its key names.
    stage1_fused is passed only when on: an aiter that predates the compact plan
    has no such kwarg.
    """
    backend, compact = DISPATCH_KERNELS[kernel]
    kwargs = {"dispatch_backend": backend}
    if compact:
        kwargs["stage1_fused"] = True
    return kwargs


def dispatch_kernel_warning(kernel: str, dispatch_wire: str) -> str | None:
    """Why ``kernel`` on ``dispatch_wire`` is not to be trusted yet, else None.

    Selectable all the same: an A/B between the two FlyDSL layouts, and a fix to
    the token-major path, both need to reach it.
    """
    if kernel == "flydsl_token_major" and dispatch_wire in _QUANT_WIRES:
        return (
            "flydsl_token_major on an fp8/fp4 MEGA_DISPATCH_WIRE gave wrong, "
            "nondeterministic output on Kimi-K3 dp8ep8 (aiter e965ebfe), where "
            "flydsl_compact and mori matched; use it for A/B and debugging"
        )
    return None


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
    longer allocates.

    cco rounds the per-rank budget up to 4 GiB, so a small M already fits: at
    Kimi-K3 dp8ep8 (topk 16, 112 experts per rank, hidden 3584, fp4) and M=2048 the
    compact arena is ~753 MiB. At M=16384 it is ~5.8 GiB against a ~3.8 GiB
    token-major budget (4 GiB reserved), and only these rows make it fit.
    """
    payload = hidden_dim // 2 if dispatch_wire == "fp4" else hidden_dim
    wire_row = _align_up(payload + hidden_dim // 32, 128)
    experts_per_rank = num_experts // ep_size
    rows = (
        ep_size * max_tokens_per_rank * topk
        + (experts_per_rank + 1) * _MAX_COMPACT_TILE_M
    )
    return rows * (wire_row + _ROWMAP_ENTRY_BYTES)
