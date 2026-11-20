# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This file contains code adapted from the flash-linear-attention project
# (fla/ops/attnres/fused.py). The original source code was licensed under the
# MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused attention-residual operations for Kimi-K3.

The algorithm is flash-linear-attention's ``fused_attnres``
(``fla/ops/attnres/fused.py``, MIT; read against fla 0.5.2), which is what the
reference KDA model calls -- see ``fla/models/kda/modeling_kda.py:135``.
Attention Residuals: https://arxiv.org/abs/2603.15031

Four deliberate divergences from that reference:

* ``residuals`` is a Sequence of separate ``[..., D]`` tensors there; here it is
  one ``[T, B, H]`` block_residual (read by stride) plus ``prefix_sum`` read as
  the final candidate. Not just cheaper -- their pointer-table gather cannot run
  on ROCm at all (details at the ``Adapted from FLA`` note in the kernel body).
* the caller's ``prefix_sum = prefix_sum + ...`` adds are folded into the last
  candidate's on-load (``DO_ADD``/``DO_ADD2``), which fla leaves to the caller.
  That fold is what lets the decoder layers defer their FFN and routed/shared
  expert adds across the layer boundary.
* ``score_weight`` arrives pre-multiplied: fla passes ``query`` and
  ``rms_weight`` separately, while ``AttnRes`` folds their product once at load
  time (see ``AttnRes.process_weights_after_loading``).
* forward only -- ATOM is inference-only, so there is no bwd and no
  ``checkpoint_level`` counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import mark_trace


def _column_tail_op(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_proj_rows: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    """Normalize the latent, then project and add one rank's BF16 columns."""
    from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_mul_add import (
        fused_gemm_a16w16_mul_add,
    )

    from atom.model_ops.layernorm import rmsnorm2d_fwd_

    if latent.ndim != 2 or up_proj_rows.ndim != 2:
        raise ValueError("column tail expects two-dimensional latent and weights")
    tokens, width = latent.shape
    columns, weight_width = up_proj_rows.shape
    if not 1 <= tokens <= 64 or width != 3584 or columns != 896:
        raise ValueError("column tail supports 1..64 rows and the TP8 Kimi geometry")
    if weight_width != width or shared.shape != (tokens, columns):
        raise ValueError("column tail weight and shared shapes must match the latent")
    if norm_weight.shape != (width,):
        raise ValueError("column tail norm weight must match the latent width")
    for tensor in (latent, norm_weight, up_proj_rows, shared):
        if tensor.dtype != torch.bfloat16 or tensor.device != latent.device:
            raise ValueError("column tail operands must be BF16 on one device")
        if tensor.stride(-1) != 1:
            raise ValueError("column tail operands require a contiguous feature axis")
    normalized = rmsnorm2d_fwd_(latent, norm_weight, eps, width)
    return fused_gemm_a16w16_mul_add(
        normalized, up_proj_rows, a=1.0, b=shared, dtype=torch.bfloat16
    )


def _column_tail_fake(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_proj_rows: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    return latent.new_empty((latent.shape[0], up_proj_rows.shape[0]))


direct_register_custom_op(
    op_name="kimi_k3_column_tail",
    op_func=_column_tail_op,
    mutates_args=[],
    fake_impl=_column_tail_fake,
)


@mark_trace
def column_tail(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_proj_rows: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    """Return the rank's [T, 896] columns, rounding projection plus add once.

    The BF16 latent is [T, 3584], with 1..64 rows. Weight and shared views
    can have padded row strides. Normalization keeps the model-sensitive
    RMSNorm setting; the existing AITER GEMM/add owns the FP32 epilogue.
    """
    return torch.ops.aiter.kimi_k3_column_tail(
        latent, norm_weight, eps, up_proj_rows, shared
    )


def _column_moe_op(hidden_states: torch.Tensor, layer_name: str) -> torch.Tensor:
    from atom.config import get_current_atom_config

    module = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return module._column_moe_forward(hidden_states)


def _column_moe_fake(hidden_states: torch.Tensor, layer_name: str) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.shape[0], 896))


direct_register_custom_op(
    op_name="kimi_k3_moe_columns",
    op_func=_column_moe_op,
    mutates_args=[],
    fake_impl=_column_moe_fake,
)


@dataclass(frozen=True)
class ColumnTail:
    """A rank's real columns; ordinary readers must gather them explicitly."""

    columns: torch.Tensor
    gather_name: str
    world: int


def _gather_columns_op(columns: torch.Tensor, world: int) -> torch.Tensor:
    from aiter.dist.communication_op import tensor_model_parallel_all_gather

    output = tensor_model_parallel_all_gather(columns, use_custom=True, dim=-1)
    if output.shape != (columns.shape[0], columns.shape[1] * world):
        raise RuntimeError("column gather returned the wrong TP geometry")
    return output


def _gather_columns_fake(columns: torch.Tensor, world: int) -> torch.Tensor:
    return columns.new_empty((columns.shape[0], columns.shape[1] * world))


direct_register_custom_op(
    op_name="kimi_k3_gather_columns",
    op_func=_gather_columns_op,
    mutates_args=[],
    fake_impl=_gather_columns_fake,
)


def materialize_column_output(value):
    """Return a complete tensor for an ordinary reader of a pending addend."""
    if isinstance(value, ColumnTail):
        return torch.ops.aiter.kimi_k3_gather_columns(value.columns, value.world)
    return value


class PeerColumnGather:
    """One model's two-parity exchange, consumed by its next residual mix.

    Each int64 word carries two BF16 columns and their row's epoch. A rank
    cannot reuse a parity until its peers have read the preceding parity.
    Every rank calls this exchange in the same order on its compute stream.
    """

    def __init__(self, hidden: int, rows: int, rank: int, world: int, group):
        from atom.mono.runtime.peer_memory import SymmetricPeerBuffer

        if not self.fits(hidden, world) or not 0 < rows <= 64:
            raise ValueError("column gather requires paired columns and 1..64 rows")
        self.rank, self.world, self.rows = rank, world, rows
        self.cols = hidden // world
        words = 2 * world * rows * (self.cols // 2)
        self.buffer = SymmetricPeerBuffer(8 * words, rank, world, group)
        self.scratch = self.buffer.storage.view(torch.int64)
        self.epochs = torch.zeros(rows, dtype=torch.int32, device=self.scratch.device)
        self.scratch.zero_()
        torch.cuda.synchronize(self.scratch.device)
        if world > 1:
            torch.distributed.barrier(group=group)
        self.closed = False

    @staticmethod
    def fits(hidden: int, world: int) -> bool:
        return (
            hidden > 0
            and world > 1
            and world & (world - 1) == 0
            and hidden % (2 * world) == 0
        )

    def close(self):
        """Close mappings after every rank's work has finished, collectively."""
        if not self.closed:
            self.buffer.close()
            self.closed = True

    def kernel_args(self, share: torch.Tensor) -> dict:
        if self.closed:
            raise RuntimeError("column gather is closed")
        if (
            share.ndim != 2
            or not 0 < share.shape[0] <= self.rows
            or share.shape[1] != self.cols
            or share.dtype != torch.bfloat16
            or share.device != self.scratch.device
            or share.stride(1) != 1
            or share.stride(0) % 2
        ):
            raise ValueError("column share does not match its exchange")
        return {
            "share_ptr": share,
            "scratch_ptr": self.scratch,
            "peers_ptr": self.buffer.addresses,
            "epochs_ptr": self.epochs,
            "rank": self.rank,
            "stride_share_t": share.stride(0),
            "PEERS": self.world,
            "COLS": self.cols,
            "ROWS": self.rows,
            "BW": triton.next_power_of_2(self.cols // 2),
            "GATHER": True,
        }


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _mix_tile(
        br_row, stride_br_b, i_l, o_d, ps, sw, b_m, b_acc, b_o, B, H, eps, BL
    ):
        # One tile of the online softmax over a row's candidates; candidate B is ps.
        o_l = i_l * BL + tl.arange(0, BL)
        m_d = o_d < H
        v = tl.load(
            br_row + o_l[:, None] * stride_br_b + o_d[None, :],
            mask=(o_l < B)[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.where((o_l == B)[:, None], ps[None, :], v)

        # score_weight = norm_weight * proj_weight, precomputed at load time
        rstd = tl.rsqrt(tl.sum(v * v, axis=1) / H + eps)
        s = tl.where(o_l <= B, tl.sum(v * sw[None, :], axis=1) * rstd, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(s, axis=0)), b_m
        r = tl.exp(b_mp - b_m)  # rescale for the new max
        p = tl.exp(s - b_m)
        b_acc = b_acc * r + tl.sum(p, axis=0)
        b_o = b_o * r + tl.sum(p[:, None] * v, axis=0)
        return b_m, b_acc, b_o

    @triton.jit
    def _finish_prefix(
        ps,
        t,
        o_d,
        m_d,
        flag,
        rank,
        scratch_ptr,
        share_ptr,
        epochs_ptr,
        hs2_ptr,
        pref_ptr,
        stride_share_t,
        stride_hs2_t,
        stride_pref_t,
        PEERS: tl.constexpr,
        COLS: tl.constexpr,
        ROWS: tl.constexpr,
        GATHER: tl.constexpr,
        DO_ADD2: tl.constexpr,
        WRITE_PREF: tl.constexpr,
    ):
        # The prefix's last adds and its store: the gathered columns, the
        # second addend, then the summed prefix back to pref_ptr.
        W: tl.constexpr = COLS // 2
        if GATHER:
            # This rank's columns from its share, the others' from the scratch
            # once their words carry the epoch. .cv reads past the caches,
            # where a stale line would hide the peer's store.
            src = o_d // COLS
            col = o_d % COLS
            words = scratch_ptr + (((flag & 1) * PEERS + src) * ROWS + t) * W + col // 2
            remote = m_d & (src != rank)
            ready = flag.to(tl.int64) << 32
            w = tl.load(
                words, mask=remote, other=ready, volatile=True, cache_modifier=".cv"
            )
            while tl.min(((w >> 32).to(tl.int32) == flag).to(tl.int32), axis=0) == 0:
                w = tl.load(
                    words, mask=remote, other=ready, volatile=True, cache_modifier=".cv"
                )
            bits = ((w >> ((col % 2) * 16).to(tl.int64)) & 0xFFFF).to(tl.int32)
            mine = tl.load(
                share_ptr + t * stride_share_t + o_d - rank * COLS,
                mask=m_d & (src == rank),
                other=0.0,
            ).to(tl.float32)
            ps += tl.where(src == rank, mine, (bits << 16).to(tl.float32, bitcast=True))
            # Every warp loaded the epoch before the poll's cross-warp min, so
            # no warp reads this store back as its own.
            tl.store(epochs_ptr + t, flag)
        if DO_ADD2:
            ps += tl.load(hs2_ptr + t * stride_hs2_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(
                pref_ptr + t * stride_pref_t + o_d,
                ps.to(pref_ptr.dtype.element_ty),
                mask=m_d,
            )
        return ps

    @triton.jit
    def _attn_res_fused_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        y_ptr,
        ys_ptr,
        hs_ptr,
        hs2_ptr,
        pref_ptr,
        ow_ptr,
        # Shapes are known at dispatch; specialize loop bounds and masks
        # without changing the fixed BF16 reduction layout below.
        B: tl.constexpr,
        Bp: tl.constexpr,
        H: tl.constexpr,
        eps,
        out_eps,
        fp8_max,
        inv_fp8_max,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_yt,
        stride_hs_t,
        stride_hs2_t,
        stride_pref_t,
        BL: tl.constexpr,  # candidates per tile
        BD: tl.constexpr,  # next_pow2(H) -- one tile spans all of H
        DO_ADD: tl.constexpr,  # fold prefix += add_hidden on-load
        DO_ADD2: tl.constexpr,  # fold a second addend (shared-expert output)
        WRITE_PREF: tl.constexpr,  # write the (summed) prefix back to pref_ptr
        OUT_NORM: tl.constexpr,  # fold the caller's output rmsnorm into the store
        QUANT: tl.constexpr,  # fold the consumer's per-token quant into the store
        share_ptr=None,
        scratch_ptr=None,
        peers_ptr=None,
        epochs_ptr=None,
        rank=0,
        stride_share_t=0,
        PEERS: tl.constexpr = 1,
        COLS: tl.constexpr = 2,
        ROWS: tl.constexpr = 1,
        BW: tl.constexpr = 1,
        GATHER: tl.constexpr = False,
    ):
        # One program per row t: rmsnorm each of the Bp = B+1 candidates, score =
        # <normed, score_weight>, softmax over Bp, then weighted sum -> y[t].
        # Candidates 0..B-1 are block_residual rows; candidate B is prefix_sum.
        # Read both source tensors directly (no torch.cat materialization).
        #
        # SINGLE PASS. The tile spans all of H and the running output stays in
        # registers, so the softmax runs online (flash-style): each new candidate
        # tile rescales the accumulator by exp(m_prev - m_new) instead of waiting
        # for a completed reduction over the candidate axis. Every candidate is
        # therefore read exactly ONCE.
        #
        # The tiling axis is what makes that work: tiling over CANDIDATES (not
        # over H, as an earlier version did) is what lets the whole output live in
        # registers. Tiling over H forces two passes -- probs aren't known until
        # the H-reduction completes, so the combine has to re-read everything --
        # and forces a third to fold OUT_NORM, since sum_h y_h^2 needs a formed y.
        # Here y is already formed in registers when the loop ends, so OUT_NORM is
        # free, and the token-count gate that the H-tiled version needed (the fold
        # stopped paying once a row's [Bp, H] reload spilled L2) is gone with it.
        #
        # Cost is registers: BD = next_pow2(H) floats of accumulator, 32 KB of VGPR
        # at H=7168, plus the [BL, BD] tile. BL is kept small for that reason.
        #
        # Adapted from FLA's fused_attnres (see the module docstring). Theirs
        # gathers from a tuple of separate residual tensors via a pointer table;
        # ours indexes one [T, B, H] block_residual by stride, which is both
        # cheaper and necessary here -- the pointer-table form miscompiles on
        # ROCm (TritonAMDGPUCanonicalizePointers rejects arith.select on
        # tensor<Nx!tt.ptr>), so their kernel cannot run on this backend at all.
        #
        # DO_ADD folds the caller's ``prefix_sum = prefix_sum + add_hidden``
        # elementwise add into the last-candidate on-load (saving a separate
        # kernel launch + HBM round-trip); WRITE_PREF then stores that summed
        # prefix so downstream layers reuse it. DO_ADD2 folds a SECOND addend the
        # same way, so an MoE layer can hand over its routed and shared expert
        # outputs unsummed and skip an entire [T, H] elementwise kernel.
        t = tl.program_id(0)
        o_d = tl.arange(0, BD)
        m_d = o_d < H
        flag = 0
        if GATHER:
            W: tl.constexpr = COLS // 2
            flag = tl.load(epochs_ptr + t) + 1
            o_w = tl.arange(0, BW)
            o_p = tl.arange(0, PEERS)
            data = tl.load(
                share_ptr.to(tl.pointer_type(tl.int32))
                + t * (stride_share_t // 2)
                + o_w,
                mask=o_w < W,
                other=0,
            )
            word = (flag.to(tl.int64) << 32) | (data.to(tl.int64) & 0xFFFFFFFF)
            slot = (((flag & 1) * PEERS + rank) * ROWS + t) * W + o_w
            peers = tl.load(peers_ptr + o_p).to(tl.pointer_type(tl.int64))
            tl.store(
                peers[:, None] + slot[None, :],
                word[None, :],
                mask=(o_p < PEERS)[:, None] & (o_w < W)[None, :],
            )
        sw = tl.load(sw_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)

        # prefix (the last candidate) is loaded once and reused across tiles;
        # re-reading it per tile would undo the single-pass property.
        ps = tl.load(ps_ptr + t * stride_ps_t + o_d, mask=m_d, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(hs_ptr + t * stride_hs_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        b_m = tl.full([], float("-inf"), dtype=tl.float32)  # running max
        b_acc = tl.zeros([], dtype=tl.float32)  # running softmax denominator
        b_o = tl.zeros([BD], dtype=tl.float32)  # running weighted sum

        # Keep matching loop structures for gathered and full inputs. A lone
        # bank-only tile stays with the prefix tile to preserve contraction.
        BANK_TILES: tl.constexpr = B // BL if B >= 2 * BL else 0
        FINISH_AFTER: tl.constexpr = 0 if GATHER and B // BL == 1 else -1
        br_row = br_ptr + t * stride_br_t
        for i_l in range(BANK_TILES):
            b_m, b_acc, b_o = _mix_tile(
                br_row, stride_br_b, i_l, o_d, ps, sw, b_m, b_acc, b_o, B, H, eps, BL
            )
        if FINISH_AFTER < 0:
            ps = _finish_prefix(
                ps,
                t,
                o_d,
                m_d,
                flag,
                rank,
                scratch_ptr,
                share_ptr,
                epochs_ptr,
                hs2_ptr,
                pref_ptr,
                stride_share_t,
                stride_hs2_t,
                stride_pref_t,
                PEERS,
                COLS,
                ROWS,
                GATHER,
                DO_ADD2,
                WRITE_PREF,
            )
        for i_l in range(BANK_TILES, tl.cdiv(Bp, BL)):
            b_m, b_acc, b_o = _mix_tile(
                br_row, stride_br_b, i_l, o_d, ps, sw, b_m, b_acc, b_o, B, H, eps, BL
            )
            if i_l == FINISH_AFTER:
                ps = _finish_prefix(
                    ps,
                    t,
                    o_d,
                    m_d,
                    flag,
                    rank,
                    scratch_ptr,
                    share_ptr,
                    epochs_ptr,
                    hs2_ptr,
                    pref_ptr,
                    stride_share_t,
                    stride_hs2_t,
                    stride_pref_t,
                    PEERS,
                    COLS,
                    ROWS,
                    GATHER,
                    DO_ADD2,
                    WRITE_PREF,
                )

        b_o = b_o / b_acc
        if OUT_NORM:
            # Free: b_o is already fully formed in registers.
            rs = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) / H + out_eps)
            b_o = b_o * rs * tl.load(ow_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
        if QUANT:
            # Also free, and for the same reason OUT_NORM is: the row is already
            # in registers, so the per-token amax is a register reduction and the
            # store just narrows. Folding it here is what lets the consuming GEMM
            # skip a standalone quant of this same [T, H] -- the fusion the
            # RMSNorm module does on the paths where it, rather than this kernel,
            # applies out_norm.
            amax = tl.max(tl.where(m_d, tl.abs(b_o), 0.0), axis=0)
            # amax * (1/fp8_max), not amax / fp8_max -- triton's ROCm fdiv is
            # 1 ulp off IEEE and would put this row's scale on a different code
            # than aiter's per-token quant, which the non-fused path uses for the
            # same activation. See the same note in kimi_k3/quant.py.
            scale = amax * inv_fp8_max
            inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
            b_o = tl.minimum(tl.maximum(b_o * inv, -fp8_max), fp8_max)
            tl.store(ys_ptr + t, scale)
        tl.store(y_ptr + t * stride_yt + o_d, b_o.to(y_ptr.dtype.element_ty), mask=m_d)


# (num_warps, num_stages, BL) by token count. One program per token, so at small T
# the grid alone cannot fill the GPU and wider warps are what recover occupancy;
# BL stays small throughout because the [BL, BD] tile competes with the [BD]
# accumulator for the register file.
_ATTN_RES_CONFIGS = (
    (8, 8, 2, 2),  # T <= 8
    (64, 8, 2, 2),
    (512, 8, 2, 2),
    (2048, 4, 2, 2),
)
_ATTN_RES_CATCHALL = (4, 2, 2)  # T > largest bucket
# Use one FP32 reduction layout for unquantized BF16 output. Changing the
# warp count at T=512 can move a value across a BF16 midpoint and alter MoE
# routing when the same prefix is evaluated with different batch peers.
_ATTN_RES_BF16_CONFIG = (8, 2, 2)


def _pick_attn_res_config(tokens: int):
    for max_tokens, nw, ns, bl in _ATTN_RES_CONFIGS:
        if tokens <= max_tokens:
            return nw, ns, bl
    return _ATTN_RES_CATCHALL


def _apply_attn_res_impl(
    prefix_sum: torch.Tensor,  # [T, H]
    block_residual: torch.Tensor,  # [T, B, H]; by stride, H unit-stride or copied
    score_weight: torch.Tensor,  # [H] (norm_weight * proj_weight, precomputed)
    eps: float,
    add_hidden: torch.Tensor | None = None,  # [T, H], folded: prefix += add_hidden
    out_norm_weight: torch.Tensor | None = None,  # [H], folded: y = rmsnorm(y)
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,  # [T, H], folded the same way
    quant_dtype: torch.dtype | None = None,  # folded: y = per-token quant(y)
    column_gather: str = "",
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Block-residual soft-attention mix: rmsnorm each of the B+1 candidates,
    score = <normed, score_weight>, softmax over B+1, weighted sum.

    Candidates are the B rows of ``block_residual`` plus ``prefix_sum``, so
    ``score_weight`` must already fold the rmsnorm gain into the scoring
    projection (see ``_attn_res_score_weight`` on the model side).

    Returns ``(mixed_output, prefix_out)``. When ``add_hidden`` (and optionally
    ``add_hidden2``) is given, the caller's ``prefix_sum = prefix_sum + ...``
    elementwise add is folded into the kernel on-load and ``prefix_out`` is that
    sum; otherwise ``prefix_out`` is ``prefix_sum`` unchanged. Two addends exist
    so an MoE layer can pass its routed and shared expert outputs separately and
    skip the [T, H] elementwise add that would otherwise combine them.

    When ``out_norm_weight`` is given, the caller's rmsnorm OF THE RESULT (every
    apply_attn_res call site in kimi_k3.py feeds one) is folded in too, so the
    returned ``y`` is already normed and scaled.

    ``quant_dtype`` folds the CONSUMING GEMM's per-token activation quant in as
    well, returning ``(y_quantized, y_scale, prefix_out)`` instead of
    ``(y, prefix_out)``. It requires ``out_norm_weight``: quantizing an unnormed
    mix would hand the consumer an activation on the wrong scale entirely.
    """
    T, B, H = block_residual.shape
    Bp = B + 1
    do_add = add_hidden is not None
    do_add2 = add_hidden2 is not None
    gather = None
    if column_gather:
        from atom.config import get_current_atom_config

        gather = get_current_atom_config().compilation_config.static_forward_context[
            column_gather
        ]
        if not isinstance(gather, PeerColumnGather) or H != gather.world * gather.cols:
            raise ValueError("residual mix does not match its column exchange")
        if not do_add:
            raise ValueError("column exchange requires a local column input")
    load_add = do_add and gather is None
    if do_add2 and not do_add:
        raise ValueError("add_hidden2 requires add_hidden")
    out_norm = out_norm_weight is not None
    quant = quant_dtype is not None
    if quant and not out_norm:
        raise ValueError("quant_dtype requires out_norm_weight")
    fp8_max = float(torch.finfo(quant_dtype).max) if quant else 1.0
    # the kernel reads T and B by stride (i32 offsets) and H unit-stride, so a
    # slice of a wider block bank needs no copy
    br = (
        block_residual
        if block_residual.stride(-1) == 1
        else block_residual.contiguous()
    )
    ps = prefix_sum.contiguous()
    sw = score_weight.contiguous()
    y = torch.empty(
        (T, H),
        device=block_residual.device,
        dtype=quant_dtype if quant else prefix_sum.dtype,
    )
    # Always allocated (triton needs a tensor); size 1 and never dereferenced
    # when QUANT is False.
    y_scale = torch.empty(
        (T, 1) if quant else (1,), device=block_residual.device, dtype=torch.float32
    )
    ow = out_norm_weight.contiguous() if out_norm else sw
    # hs/hs2/pref pointers are always passed (triton needs a tensor); when not
    # adding they alias ps and are never dereferenced (DO_ADD / DO_ADD2 /
    # WRITE_PREF are False).
    hs = add_hidden.contiguous() if load_add else ps
    hs2 = add_hidden2.contiguous() if do_add2 else ps
    pref = torch.empty_like(ps) if do_add else ps

    nw, ns, bl = (
        _ATTN_RES_BF16_CONFIG
        if ps.dtype == torch.bfloat16 and quant_dtype is None
        else _pick_attn_res_config(T)
    )
    _attn_res_fused_kernel[(T,)](
        br,
        ps,
        sw,
        y,
        y_scale,
        hs,
        hs2,
        pref,
        ow,
        B,
        Bp,
        H,
        float(eps),
        float(out_eps),
        fp8_max,
        1.0 / fp8_max,
        br.stride(0),
        br.stride(1),
        ps.stride(0),
        y.stride(0),
        hs.stride(0),
        hs2.stride(0),
        pref.stride(0),
        BL=bl,
        BD=triton.next_power_of_2(H),
        num_stages=ns,
        num_warps=nw,
        DO_ADD=load_add,
        DO_ADD2=do_add2,
        WRITE_PREF=do_add,
        OUT_NORM=out_norm,
        QUANT=quant,
        **(gather.kernel_args(add_hidden) if gather is not None else {}),
    )
    prefix_out = pref if do_add else prefix_sum
    if quant:
        return y, y_scale, prefix_out
    return y, prefix_out


def _apply_attn_res_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
) -> torch.Tensor:
    mixed_output, _ = _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
    )
    return mixed_output


def _apply_attn_res_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(prefix_sum)


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res",
    op_func=_apply_attn_res_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_op_fake,
)


def _apply_attn_res_add_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,
    column_gather: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
        add_hidden2=add_hidden2,
        column_gather=column_gather,
    )


def _apply_attn_res_add_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,
    column_gather: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(prefix_sum), torch.empty_like(prefix_sum)


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res_add",
    op_func=_apply_attn_res_add_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_add_op_fake,
)


# Quantizing variants. Separate ops rather than an optional `quant_dtype` on the
# two above because the return arity differs (the scale), and a schema's return
# type is fixed at registration.


def _apply_attn_res_quant_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, y_scale, _ = _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
        quant_dtype=quant_dtype,
    )
    return y, y_scale


def _apply_attn_res_quant_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(prefix_sum, dtype=quant_dtype),
        torch.empty(
            (prefix_sum.shape[0], 1), device=prefix_sum.device, dtype=torch.float32
        ),
    )


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res_quant",
    op_func=_apply_attn_res_quant_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_quant_op_fake,
)


def _apply_attn_res_add_quant_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
    add_hidden2: torch.Tensor | None = None,
    column_gather: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
        add_hidden2=add_hidden2,
        quant_dtype=quant_dtype,
        column_gather=column_gather,
    )


def _apply_attn_res_add_quant_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
    add_hidden2: torch.Tensor | None = None,
    column_gather: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(prefix_sum, dtype=quant_dtype),
        torch.empty(
            (prefix_sum.shape[0], 1), device=prefix_sum.device, dtype=torch.float32
        ),
        torch.empty_like(prefix_sum),
    )


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res_add_quant",
    op_func=_apply_attn_res_add_quant_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_add_quant_op_fake,
)


@mark_trace
def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor | ColumnTail | None = None,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Dispatch an opaque custom op whose CUDA implementation selects by concrete T.

    ``out_norm_weight`` folds the caller's rmsnorm of the result into the kernel;
    the returned mixed output is then already normed and scaled by it.
    ``add_hidden2`` folds a second addend into the prefix (see the impl).

    ``quant_dtype`` additionally folds the consuming GEMM's per-token quant in,
    making the first element of the returned pair a ``(quantized, scale)`` tuple
    rather than a tensor -- the same shape the fused-quant RMSNorm hands its
    consumers, so the call sites that already unpack one need no new branch."""
    column_gather = ""
    if isinstance(add_hidden, ColumnTail):
        column_gather, add_hidden = add_hidden.gather_name, add_hidden.columns
    if isinstance(add_hidden2, ColumnTail):
        raise TypeError("the column tail must be the first pending addend")
    if quant_dtype is not None:
        if add_hidden is None:
            if add_hidden2 is not None:
                raise ValueError("add_hidden2 requires add_hidden")
            y, y_scale = torch.ops.aiter.kimi_k3_apply_attn_res_quant(
                prefix_sum,
                block_residual,
                score_weight,
                eps,
                out_norm_weight,
                out_eps,
                quant_dtype,
            )
            return (y, y_scale), prefix_sum
        y, y_scale, prefix_out = torch.ops.aiter.kimi_k3_apply_attn_res_add_quant(
            prefix_sum,
            block_residual,
            score_weight,
            eps,
            add_hidden,
            out_norm_weight,
            out_eps,
            quant_dtype,
            add_hidden2,
            column_gather,
        )
        return (y, y_scale), prefix_out
    if add_hidden is None:
        if add_hidden2 is not None:
            raise ValueError("add_hidden2 requires add_hidden")
        return (
            torch.ops.aiter.kimi_k3_apply_attn_res(
                prefix_sum, block_residual, score_weight, eps, out_norm_weight, out_eps
            ),
            prefix_sum,
        )
    return torch.ops.aiter.kimi_k3_apply_attn_res_add(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        out_norm_weight,
        out_eps,
        add_hidden2,
        column_gather,
    )
