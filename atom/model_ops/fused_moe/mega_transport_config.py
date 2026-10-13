# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""How ATOM configures aiter's MegaMoEGfx1250 transport.

Which dispatch kernel it runs and how much cco VMM its arena then needs. Kept free
of aiter and mori imports so the rules are testable on CPU-only CI.
"""

_QUANT_WIRES = ("fp8", "fp4")
_DISPATCH_BACKENDS = ("mori", "flydsl")

# Every dispatch kernel MegaMoEGfx1250 can run: (aiter's dispatch_backend,
# stage1_fused).
#   mori                mori's HIP/JIT dispatch, token-major
#   flydsl_token_major  aiter's FlyDSL TDM dispatch, token-major
#   flydsl_compact      the same dispatch fed by aiter's compact plan: each route
#                       lands as its own row, grouped per local expert
DISPATCH_KERNELS = {
    "mori": ("mori", False),
    "flydsl_token_major": ("flydsl", False),
    "flydsl_compact": ("flydsl", True),
}

# The fastest dispatch kernel per measured gfx1250 MoE shape, keyed by
# (num_experts, topk, hidden_dim, dispatch_wire), at any EP size. Each value is
# (the largest max_tokens_per_rank measured, kernel): a transport built for more
# tokens per rank than that, or a shape or wire not listed here, keeps the
# defaults.
#
# Kimi-K3 (896 experts, top-16, hidden 3584, fp4 wire): measured 2026-10-12 on
# MI455 (gfx1250 B0), EP8 over two trays, aiter e965ebfe, mori 066fd10e, FlyDSL
# 0.3.4.1. aiter's bench_mega_moe drove an 8-layer MegaMoE chain in one CUDA
# graph, built for 16384 tokens per rank and called with 1 to 16384, uniform
# and skewed routing. Per MoE layer the compact plan beat mori by 40-60 us at 1
# to 512 tokens per rank and trailed it by at most 0.6% at 8192-16384; serving
# TPOT was 92.5 ms against mori's 95.5 ms. flydsl_token_major was fastest from
# 1024 tokens per rank up but gave wrong answers in ATOM serving, so it is not a
# candidate. mori was measured before aiter #6115, which speeds up its expert
# table at 896 experts.
_MEASURED_FASTEST: dict[tuple[int, int, int, str], tuple[int, str]] = {
    (896, 16, 3584, "fp4"): (16384, "flydsl_compact"),
}

# Upper bound on the expert GEMM's tile_m, which pads each local expert's slice
# of the compact plan. aiter picks it from its tuned grouped-GEMM table, whose
# widest tile is 256.
_MAX_COMPACT_TILE_M = 256
# The compact plan's row map: one 8-byte entry per dispatch row.
_ROWMAP_ENTRY_BYTES = 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _measured_fastest(
    *,
    num_experts: int,
    topk: int,
    hidden_dim: int,
    dispatch_wire: str,
    max_tokens_per_rank: int,
) -> str | None:
    measured = _MEASURED_FASTEST.get((num_experts, topk, hidden_dim, dispatch_wire))
    if measured is None or max_tokens_per_rank > measured[0]:
        return None
    return measured[1]


def resolve_dispatch_kernel(
    *,
    dispatch_wire: str,
    mega_dispatch: str | None,
    needs_token_major: bool,
    num_experts: int,
    topk: int,
    hidden_dim: int,
    max_tokens_per_rank: int,
) -> tuple[str, str]:
    """The dispatch kernel a transport runs, and what chose it.

    ``mega_dispatch`` ($MEGA_DISPATCH, aiter's backend knob) overrides the choice:
    mori -> mori; flydsl -> the compact plan on an fp8/fp4 wire, token-major on
    bf16. Unset, a shape in _MEASURED_FASTEST runs its measured fastest kernel,
    and anything else the defaults: mori on an fp8/fp4 wire, flydsl token-major on
    bf16.

    ``needs_token_major`` is set when Triton decode experts
    ($ATOM_USE_TRITON_MOE_DECODE) read the dispatch output: they read it
    token-major, so the compact plan is never picked for them and asking for it
    is an error.

    Returns (kernel, source), source being "MEGA_DISPATCH", "measured" or
    "default".
    """
    quant = dispatch_wire in _QUANT_WIRES
    if mega_dispatch is not None:
        if mega_dispatch not in _DISPATCH_BACKENDS:
            raise ValueError(
                f"MEGA_DISPATCH must be one of {_DISPATCH_BACKENDS}, "
                f"got {mega_dispatch!r}"
            )
        if mega_dispatch == "mori":
            kernel = "mori"
        else:
            kernel = "flydsl_compact" if quant else "flydsl_token_major"
        if needs_token_major and DISPATCH_KERNELS[kernel][1]:
            raise ValueError(
                f"MEGA_DISPATCH={mega_dispatch} runs {kernel} on a {dispatch_wire} "
                "wire, whose per-route rows the Triton decode experts "
                "(ATOM_USE_TRITON_MOE_DECODE=1) cannot read; set MEGA_DISPATCH=mori "
                "or unset it"
            )
        return kernel, "MEGA_DISPATCH"

    kernel = _measured_fastest(
        num_experts=num_experts,
        topk=topk,
        hidden_dim=hidden_dim,
        dispatch_wire=dispatch_wire,
        max_tokens_per_rank=max_tokens_per_rank,
    )
    if kernel is not None and not (needs_token_major and DISPATCH_KERNELS[kernel][1]):
        return kernel, "measured"
    return ("mori" if quant else "flydsl_token_major"), "default"


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
