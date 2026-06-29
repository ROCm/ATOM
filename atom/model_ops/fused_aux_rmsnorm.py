# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-group RMSNorm for EAGLE3 aux hidden-state fusion.

EAGLE3's ``combine_hidden_states`` normalizes ``num_aux`` aux chunks (each with
its own ``fc_norm`` weight) and concatenates them into the ``[N, num_aux*H]``
input of the ``fc`` projection.  The naive path launches one RMSNorm per chunk
plus a concat; this kernel does all chunks in a single launch, writing straight
into the contiguous ``fc`` input buffer.

Input layout: ``x`` is the concatenated aux ``[N, num_aux*H]`` (view as groups
of ``H`` along the last dim).  ``weight`` is the per-group RMSNorm weights
stacked to ``[num_aux, H]``.  Plain RMSNorm (``x * rstd * w``, fp32 reduction) —
matches ``atom.model_ops.layernorm.RMSNorm`` (NOT the Gemma ``1+w`` variant).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_group_rmsnorm_kernel(
    x_ptr,  # [N, G*H] contiguous
    w_ptr,  # [G, H] contiguous
    out_ptr,  # [N, G*H] contiguous
    n_rows,
    G: tl.constexpr,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1)
    col = tl.arange(0, BLOCK_H)
    mask = col < H

    row_base = row * (G * H) + g * H
    x = tl.load(x_ptr + row_base + col, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + g * H + col, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row_base + col, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_group_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    """Per-group RMSNorm over a concatenated ``[N, num_groups*H]`` tensor.

    Args:
        x: contiguous ``[N, num_groups*H]`` (groups of ``H`` along dim -1).
        weight: per-group weights stacked to ``[num_groups, H]`` (contiguous).
        eps: RMSNorm epsilon.
        num_groups: number of aux groups (``G``).

    Returns:
        ``[N, num_groups*H]`` with each group RMS-normalized by its own weight.
    """
    assert x.is_cuda, "fused_group_rmsnorm requires a CUDA tensor."
    assert x.dim() == 2 and x.is_contiguous()
    n_rows, total = x.shape
    assert total % num_groups == 0
    H = total // num_groups
    assert weight.shape == (
        num_groups,
        H,
    ), f"weight must be [{num_groups}, {H}], got {tuple(weight.shape)}"

    out = torch.empty_like(x)
    BLOCK_H = triton.next_power_of_2(H)
    num_warps = 8 if BLOCK_H >= 4096 else (4 if BLOCK_H >= 1024 else 2)
    grid = (n_rows, num_groups)
    _fused_group_rmsnorm_kernel[grid](
        x,
        weight.contiguous(),
        out,
        n_rows,
        num_groups,
        H,
        float(eps),
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Dual-input RMSNorm + concat (EAGLE3 draft decoder-layer attention input)
#
# The Eagle3 draft decoder layer normalizes two same-shaped ``[N, H]`` inputs
# (``embeds`` with ``input_layernorm``, ``hidden_states`` with ``hidden_norm``)
# and concatenates them into the ``[N, 2H]`` QKV input.  The naive path is two
# RMSNorm launches + a concat (3 launches; the concat re-reads + re-writes 2NH).
# This kernel does it in a single launch that writes each normalized half
# straight into the contiguous ``[N, 2H]`` output, cutting memory traffic from
# ~8NH (norm+norm+cat) to ~4NH.  Plain RMSNorm math (``x * rstd * w``, fp32
# reduction) — matches ``atom.model_ops.layernorm.RMSNorm`` and the sibling
# ``fused_group_rmsnorm`` above.
#
# Raw Triton (no custom-op wrapper): the EAGLE3 draft is built with
# ``CompilationLevel.NO_COMPILATION`` (eagle.py), so its forward always runs
# eager and never enters Dynamo — same as ``fused_group_rmsnorm`` above.
#
# grid = (n_rows, 2): program (row, 0) normalizes ``a`` -> out[:, :H], program
# (row, 1) normalizes ``b`` -> out[:, H:].  2*n_rows programs (vs n_rows) keeps
# occupancy up at small batch (EAGLE decode N == bs).
# ---------------------------------------------------------------------------


@triton.jit
def _fused_dual_rmsnorm_cat_kernel(
    a_ptr,  # [N, H] contiguous
    b_ptr,  # [N, H] contiguous
    wa_ptr,  # [H]
    wb_ptr,  # [H]
    out_ptr,  # [N, 2H] contiguous
    H,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1)  # 0 -> (a, wa) into out[:, :H]; 1 -> (b, wb) into out[:, H:]
    col = tl.arange(0, BLOCK_H)
    mask = col < H

    # g is uniform across the program (one (row, half) per program), so this is
    # uniform control flow — no divergence, and avoids selecting between two
    # base pointers (unsupported in Triton). Weights are reused across rows of
    # the same half, so keep them resident with evict_last.
    if g == 0:
        x = tl.load(a_ptr + row * H + col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(
            wa_ptr + col, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
    else:
        x = tl.load(b_ptr + row * H + col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(
            wb_ptr + col, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)

    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    y = x * rstd * w
    tl.store(
        out_ptr + row * (2 * H) + g * H + col,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_dual_rmsnorm_cat(
    a: torch.Tensor,
    b: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMS-norm two ``[N, H]`` inputs by their own weights into one ``[N, 2H]``.

    ``out[:, :H] = rmsnorm(a, w_a)``, ``out[:, H:] = rmsnorm(b, w_b)`` — the
    concatenated attention input for the Eagle3 draft decoder layer, produced
    in a single Triton launch (no separate per-input norm + concat).

    Args:
        a, b: contiguous ``[N, H]`` inputs (same shape).
        w_a, w_b: per-input RMSNorm weights ``[H]``.
        eps: RMSNorm epsilon (shared by both norms).

    Returns:
        contiguous ``[N, 2H]`` with the two normalized halves side by side.
    """
    n_rows, H = a.shape
    out = torch.empty((n_rows, 2 * H), dtype=a.dtype, device=a.device)
    BLOCK_H = triton.next_power_of_2(H)
    num_warps = 8 if BLOCK_H >= 4096 else (4 if BLOCK_H >= 1024 else 2)
    grid = (n_rows, 2)
    _fused_dual_rmsnorm_cat_kernel[grid](
        a,
        b,
        w_a,
        w_b,
        out,
        H,
        float(eps),
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return out
