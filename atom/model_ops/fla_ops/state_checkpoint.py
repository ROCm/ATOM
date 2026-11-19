# SPDX-License-Identifier: MIT
"""Write interior SSM+conv state checkpoints in one launch.

A checkpoint at a *prompt-end* position is free: the chunk kernel already
leaves the final state in the slot. A checkpoint *inside* the step is not —
that state exists only as a slice of the chunk kernel's per-chunk
intermediates, ``h[:, j]``, and its conv half as a window of the conv input.
Both have to be copied out before the next step overwrites them.

The two halves share everything that varies per target — the row, the
destination slot, the token offset — and differ only in which buffer they read
and how they index it. So they are one kernel over a split grid rather than
two launches: ``program_id(1) < NBLK_SSM`` copies state, the rest copy conv
windows. At 48 GDN layers that halves the launch count on the hot path.

Doing this in Python instead cost a `copy_` pair per target per layer and —
worse — got varlen indexing wrong: ``h`` and the conv input are
batch-concatenated, so a target's chunk index is relative to
``chunk_offsets[row]`` and its token offset to ``cu_seqlens[row]``, not to the
start of the batch. Those bases are device tensors here and the arithmetic
happens per program, so there is no host-side loop and no base to forget.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.utils import envs


@triton.jit
def _copy_checkpoints_kernel(
    h,  # [1, NT_total, H, K, V] per-chunk states, batch-concatenated (a view of [.., V, K] when H_VK)
    ssm_dst,  # [num_slots, H, K, V] the state pool
    x,  # [total_tokens, D] the conv input, batch-concatenated
    conv_dst,  # [num_slots, D, >= STATE_LEN] the conv state pool
    rows,  # [T] batch row of each target
    slots,  # [T] absolute destination slot of each target
    offs,  # [T] token offset of the target within its slice of this step
    is_end,  # [T] 1 if the target sits AT the end of the row's tokens
    runtime_slots,  # [T] the row's runtime slot, the source when is_end
    chunk_offsets,  # [N + 1] first chunk index of each sequence in `h`
    cu_seqlens,  # [N + 1] first token of each sequence in `x`
    stride_x_t,  # row (token) stride of `x`; NOT D — see write_state_checkpoints
    stride_cd_s,  # conv_dst slot stride
    stride_cd_d,  # conv_dst channel stride
    stride_cd_l,  # conv_dst window-position stride
    HKV: tl.constexpr,  # H * K * V, one full recurrent state
    D: tl.constexpr,
    STATE_LEN: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLK_SSM: tl.constexpr,  # blocks covering HKV; the grid split point
    K: tl.constexpr,
    V: tl.constexpr,
    STATE_VK: tl.constexpr = False,
    SSM_FROM_H: tl.constexpr = True,
    H_VK: tl.constexpr = False,
):
    i_t = tl.program_id(0)
    i_blk = tl.program_id(1)

    row = tl.load(rows + i_t).to(tl.int64)
    slot = tl.load(slots + i_t).to(tl.int64)
    off = tl.load(offs + i_t).to(tl.int64)
    # A fixed-capacity target table (the token-graph path) pads with slot -1:
    # those programs copy nothing.
    valid = slot >= 0

    if i_blk < NBLK_SSM:
        # ── recurrent state -> slot ────────────────────────────────────────
        # Interior target: the state at `off` is a chunk boundary in `h`.
        # End target: it is NOT in `h` — `h` holds boundaries strictly before
        # the end, and `chunk_offsets[row] + T // CHUNK` is the NEXT
        # sequence's first chunk. The final state exists only in the row's
        # runtime slot, which the chunk kernel just wrote.
        # idx walks the pool's order; h_idx is the same element in h's order.
        idx = i_blk * BLOCK + tl.arange(0, BLOCK)
        mask = (idx < HKV) & valid
        h_idx = idx
        if STATE_VK != H_VK:
            INNER: tl.constexpr = K if STATE_VK else V
            h_idx = (
                idx // (K * V) * (K * V)
                + (idx % INNER) * (K * V // INNER)
                + (idx % (K * V)) // INNER
            )
        # Both branches cast to the pool's dtype: `h` is bf16 (the chunk
        # kernel allocates it with `k.new_empty`) while the pool is fp32, so
        # the two loads have different types and Triton requires them to
        # agree before the store.
        if tl.load(is_end + i_t) != 0:
            src_slot = tl.load(runtime_slots + i_t).to(tl.int64)
            sv = tl.load(ssm_dst + src_slot * HKV + idx, mask=mask, other=0.0).to(
                ssm_dst.dtype.element_ty
            )
            tl.store(ssm_dst + slot * HKV + idx, sv, mask=mask)
        elif SSM_FROM_H:
            src = tl.load(chunk_offsets + row).to(tl.int64) + off // CHUNK
            sv = tl.load(h + src * HKV + h_idx, mask=mask, other=0.0).to(
                ssm_dst.dtype.element_ty
            )
            tl.store(ssm_dst + slot * HKV + idx, sv, mask=mask)
        # Otherwise the chunk kernel already stored this interior state, in
        # fp32, straight into `slot` (`chunk_gated_delta_rule`'s
        # `checkpoint_slots`).
    else:
        # ── conv window: the STATE_LEN tokens ending at the target, ────────
        # oldest first into positions [0, STATE_LEN) of the pool's [D, W]
        # layout -- where the prefill conv kernel reads a resumed state from.
        end = tl.load(cu_seqlens + row).to(tl.int64) + off
        d = (i_blk - NBLK_SSM) * BLOCK + tl.arange(0, BLOCK)
        dmask = (d < D) & valid
        for j in tl.static_range(STATE_LEN):
            cv = tl.load(
                x + (end - STATE_LEN + j) * stride_x_t + d, mask=dmask, other=0.0
            )
            tl.store(
                conv_dst + slot * stride_cd_s + d * stride_cd_d + j * stride_cd_l,
                cv.to(conv_dst.dtype.element_ty),
                mask=dmask,
            )


@triton.jit
def _copy_checkpoints_loop_kernel(
    h,
    ssm_dst,
    x,
    conv_dst,
    rows,
    slots,
    offs,
    is_end,
    runtime_slots,
    chunk_offsets,
    cu_seqlens,
    n_targets,
    stride_x_t,
    stride_cd_s,
    stride_cd_d,
    stride_cd_l,
    HKV: tl.constexpr,
    D: tl.constexpr,
    STATE_LEN: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLK_SSM: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    TARGETS: tl.constexpr,
    STATE_VK: tl.constexpr = False,
    SSM_FROM_H: tl.constexpr = True,
    H_VK: tl.constexpr = False,
):
    """`_copy_checkpoints_kernel` with one program a block, looping over the targets.

    The whole target table is read in one vector load, so a step without targets
    (padding only) costs one load a program instead of one program a padded target.
    """
    i_blk = tl.program_id(0)
    t_idx = tl.arange(0, TARGETS)
    t_mask = t_idx < n_targets
    slot_v = tl.load(slots + t_idx, mask=t_mask, other=-1)
    row_v = tl.load(rows + t_idx, mask=t_mask, other=0)
    off_v = tl.load(offs + t_idx, mask=t_mask, other=0)
    end_v = tl.load(is_end + t_idx, mask=t_mask, other=0)
    run_v = tl.load(runtime_slots + t_idx, mask=t_mask, other=0)
    # targets past the last valid one are skipped without a single iteration
    n_live = tl.max(tl.where(slot_v >= 0, t_idx + 1, 0), 0)
    for t in range(n_live):
        sel = t_idx == t
        slot = tl.sum(tl.where(sel, slot_v, 0), 0).to(tl.int64)
        if slot >= 0:
            row = tl.sum(tl.where(sel, row_v, 0), 0).to(tl.int64)
            off = tl.sum(tl.where(sel, off_v, 0), 0).to(tl.int64)
            if i_blk < NBLK_SSM:
                idx = i_blk * BLOCK + tl.arange(0, BLOCK)
                mask = idx < HKV
                h_idx = idx
                if STATE_VK != H_VK:
                    INNER: tl.constexpr = K if STATE_VK else V
                    h_idx = (
                        idx // (K * V) * (K * V)
                        + (idx % INNER) * (K * V // INNER)
                        + (idx % (K * V)) // INNER
                    )
                if tl.sum(tl.where(sel, end_v, 0), 0) != 0:
                    src_slot = tl.sum(tl.where(sel, run_v, 0), 0).to(tl.int64)
                    sv = tl.load(ssm_dst + src_slot * HKV + idx, mask=mask, other=0.0)
                    tl.store(ssm_dst + slot * HKV + idx, sv, mask=mask)
                elif SSM_FROM_H:
                    src = tl.load(chunk_offsets + row).to(tl.int64) + off // CHUNK
                    hv = tl.load(h + src * HKV + h_idx, mask=mask, other=0.0)
                    tl.store(
                        ssm_dst + slot * HKV + idx,
                        hv.to(ssm_dst.dtype.element_ty),
                        mask=mask,
                    )
            else:
                end = tl.load(cu_seqlens + row).to(tl.int64) + off
                d = (i_blk - NBLK_SSM) * BLOCK + tl.arange(0, BLOCK)
                dmask = d < D
                for j in tl.static_range(STATE_LEN):
                    cv = tl.load(
                        x + (end - STATE_LEN + j) * stride_x_t + d,
                        mask=dmask,
                        other=0.0,
                    )
                    tl.store(
                        conv_dst
                        + slot * stride_cd_s
                        + d * stride_cd_d
                        + j * stride_cd_l,
                        cv.to(conv_dst.dtype.element_ty),
                        mask=dmask,
                    )


def write_state_checkpoints(
    h: torch.Tensor,
    ssm_state: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    offs: torch.Tensor,
    is_end: torch.Tensor,
    runtime_slots: torch.Tensor,
    chunk_offsets: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    conv_state_len: int,
    ssm_from_h: bool = True,
) -> None:
    """Write both halves of every checkpoint this step reached, in one launch.

    ``ssm_state`` is ``[slots, H, K, V]`` and ``conv_state``
    ``[slots, D, state_len]``; both are indexed by the same ``slots``, so a
    checkpoint's two halves always describe the same token position. Writing
    only one of them would pair a state at P with a window from elsewhere,
    which is silently wrong — hence the single entry point.

    ``is_end`` picks the SSM source per target: 0 reads the chunk kernel's
    intermediates (an interior position), 1 reads ``runtime_slots[i]`` (a
    position at the end of the row's tokens, which is only ever in the runtime
    slot). The conv half is the same window slice either way.

    ``conv_state_len`` is ``conv_kernel - 1`` and is passed, not read off
    ``conv_state``: the pool is ``conv_kernel - 1 + num_spec`` wide for the
    decode path, while ``causal_conv1d_fn`` resumes a prefill from positions
    ``[0, conv_kernel - 1)`` and nothing else. Filling the whole width put the
    newest token at the last position and left the prefill kernel reading
    tokens ``num_spec`` older than the checkpoint -- and read that many rows
    before the target, into the previous sequence's tokens.

    ``ssm_from_h=False``: the chunk kernel already stored every interior
    target's state, in fp32, into its slot (``chunk_gated_delta_rule``'s
    ``checkpoint_slots``), so only ``is_end`` targets copy an SSM state here
    and ``h`` is not read (any tensor will do). A target whose slot is negative
    is padding and copies nothing.

    Neither conv tensor may be assumed contiguous, so the conv half indexes by
    real strides. ``conv_state`` reaches us as a ``transpose(-1, -2)`` view of a
    ``[slots, state_len, D]`` allocation, and ``x`` is a column slice of the
    fused in-projection whose row stride is the *fused* width, not ``D``.
    Deriving either from the shape writes a correctly-shaped checkpoint out of
    the wrong bytes, which nothing downstream can detect.
    """
    n = rows.numel()
    if n == 0:
        return
    hkv = ssm_state.stride(0)
    d, state_len = conv_state.shape[1], conv_state_len
    assert 0 < state_len <= conv_state.shape[2], "conv window wider than the pool"
    assert x.stride(-1) == 1, "conv input must be contiguous along the feature dim"
    if envs.ATOM_GDN_CKPT_COPY_LOOP:
        # One program a block of the state / conv window, each looping over the targets.
        BLOCK = 2048
        nblk_ssm = triton.cdiv(hkv, BLOCK)
        _copy_checkpoints_loop_kernel[(nblk_ssm + triton.cdiv(d, BLOCK),)](
            h,
            ssm_state,
            x,
            conv_state,
            rows,
            slots,
            offs,
            is_end,
            runtime_slots,
            chunk_offsets,
            cu_seqlens,
            n,
            x.stride(0),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            HKV=hkv,
            D=d,
            STATE_LEN=state_len,
            CHUNK=chunk_size,
            BLOCK=BLOCK,
            NBLK_SSM=nblk_ssm,
            K=ssm_state.shape[-2],
            V=ssm_state.shape[-1],
            TARGETS=triton.next_power_of_2(n),
            STATE_VK=ssm_state.stride(-2) == 1,
            H_VK=h.stride(-2) == 1,
            SSM_FROM_H=ssm_from_h,
            num_warps=8,
        )
        return
    BLOCK = 1024
    nblk_ssm = triton.cdiv(hkv, BLOCK)
    grid = (n, nblk_ssm + triton.cdiv(d, BLOCK))
    _copy_checkpoints_kernel[grid](
        h,
        ssm_state,
        x,
        conv_state,
        rows,
        slots,
        offs,
        is_end,
        runtime_slots,
        chunk_offsets,
        cu_seqlens,
        x.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        HKV=hkv,
        D=d,
        STATE_LEN=state_len,
        CHUNK=chunk_size,
        BLOCK=BLOCK,
        NBLK_SSM=nblk_ssm,
        K=ssm_state.shape[-2],
        V=ssm_state.shape[-1],
        STATE_VK=ssm_state.stride(-2) == 1,
        H_VK=h.stride(-2) == 1,
        SSM_FROM_H=ssm_from_h,
    )
