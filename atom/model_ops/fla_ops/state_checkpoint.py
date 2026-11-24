# SPDX-License-Identifier: MIT
"""Write interior SSM+conv state checkpoints in one launch.

A target at the end of a row reads its settled runtime state. An interior
target reads ``h``: GDN supplies chunk snapshots, while Kimi KDA supplies the
final states of head re-runs. The conv half is a window of the step's input.
Both halves are copied into the reserved slot before the next step writes it.

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


@triton.jit
def _copy_checkpoints_kernel(
    h,  # Packed interior states; explicit h_rows or per-sequence chunk offsets.
    ssm_dst,  # [num_slots, H, K, V] the state pool
    x,  # [total_tokens, D] the conv input, batch-concatenated
    conv_dst,  # [num_slots, D, >= STATE_LEN] the conv state pool
    rows,  # [T] batch row of each target
    slots,  # [T] absolute destination slot of each target
    offs,  # [T] token offset of the target within its slice of this step
    is_end,  # [T] 1 if the target sits AT the end of the row's tokens
    runtime_slots,  # [T] the row's runtime slot, the source when is_end
    chunk_offsets,  # [N + 1] first chunk index of each sequence in `h`
    h_rows,  # [T] row of `h` holding each interior target, if HAS_H_ROWS
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
    HAS_H_ROWS: tl.constexpr = False,
):
    i_t = tl.program_id(0)
    i_blk = tl.program_id(1)

    row = tl.load(rows + i_t).to(tl.int64)
    slot = tl.load(slots + i_t).to(tl.int64)
    off = tl.load(offs + i_t).to(tl.int64)

    if i_blk < NBLK_SSM:
        # ── recurrent state -> slot ────────────────────────────────────────
        # Interior targets read their snapshot in `h`. End targets read the
        # settled runtime slot; chunk-offset arithmetic would point past the
        # row's interior snapshots.
        idx = i_blk * BLOCK + tl.arange(0, BLOCK)
        mask = idx < HKV
        dst_idx = idx
        if STATE_VK:
            dst_idx = idx // (K * V) * (K * V) + (idx % V) * K + (idx % (K * V)) // V
        # Snapshot and pool dtypes can differ. Both branches must agree on
        # the pool dtype before the store.
        if tl.load(is_end + i_t) != 0:
            src_slot = tl.load(runtime_slots + i_t).to(tl.int64)
            sv = tl.load(ssm_dst + src_slot * HKV + dst_idx, mask=mask, other=0.0).to(
                ssm_dst.dtype.element_ty
            )
        else:
            if HAS_H_ROWS:
                src = tl.load(h_rows + i_t).to(tl.int64)
            else:
                src = tl.load(chunk_offsets + row).to(tl.int64) + off // CHUNK
            sv = tl.load(h + src * HKV + idx, mask=mask, other=0.0).to(
                ssm_dst.dtype.element_ty
            )
        tl.store(ssm_dst + slot * HKV + dst_idx, sv, mask=mask)
    else:
        # ── conv window: the STATE_LEN tokens ending at the target, ────────
        # oldest first into positions [0, STATE_LEN) of the pool's [D, W]
        # layout -- where the prefill conv kernel reads a resumed state from.
        end = tl.load(cu_seqlens + row).to(tl.int64) + off
        d = (i_blk - NBLK_SSM) * BLOCK + tl.arange(0, BLOCK)
        dmask = d < D
        for j in tl.static_range(STATE_LEN):
            cv = tl.load(
                x + (end - STATE_LEN + j) * stride_x_t + d, mask=dmask, other=0.0
            )
            tl.store(
                conv_dst + slot * stride_cd_s + d * stride_cd_d + j * stride_cd_l,
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
    h_rows: torch.Tensor | None = None,
) -> None:
    """Write both halves of every checkpoint this step reached, in one launch.

    ``ssm_state`` is ``[slots, H, K, V]`` and ``conv_state``
    ``[slots, D, state_len]``; both are indexed by the same ``slots``, so a
    checkpoint's two halves always describe the same token position. Writing
    only one of them would pair a state at P with a window from elsewhere,
    which is silently wrong — hence the single entry point.

    ``is_end`` picks the SSM source per target: 0 reads an interior snapshot
    from ``h``, 1 reads ``runtime_slots[i]`` (a
    position at the end of the row's tokens, which is only ever in the runtime
    slot). The conv half is the same window slice either way.

    ``conv_state_len`` is ``conv_kernel - 1`` and is passed, not read off
    ``conv_state``: the pool is ``conv_kernel - 1 + num_spec`` wide for the
    decode path, while ``causal_conv1d_fn`` resumes a prefill from positions
    ``[0, conv_kernel - 1)`` and nothing else. Filling the whole width put the
    newest token at the last position and left the prefill kernel reading
    tokens ``num_spec`` older than the checkpoint -- and read that many rows
    before the target, into the previous sequence's tokens.

    Neither conv tensor may be assumed contiguous, so the conv half indexes by
    real strides. ``conv_state`` reaches us as a ``transpose(-1, -2)`` view of a
    ``[slots, state_len, D]`` allocation, and ``x`` is a column slice of the
    fused in-projection whose row stride is the *fused* width, not ``D``.
    Deriving either from the shape writes a correctly-shaped checkpoint out of
    the wrong bytes, which nothing downstream can detect.

    ``h_rows`` names, per target, the row of ``h`` an interior target reads.
    When snapshots are needed, ``interior_state_snapshots`` plans those rows
    and the chunk kernel fills the ``h_snap`` buffer allocated by
    ``GatedDeltaNet``; pass that buffer as ``h`` with the planned ``h_rows``.
    Kimi KDA instead supplies head re-runs' final states with explicit rows.
    Without ``h_rows`` the row is the chunk boundary,
    ``chunk_offsets[row] + off // chunk``.
    """
    n = rows.numel()
    if n == 0:
        return
    hkv = ssm_state.stride(0)
    d, state_len = conv_state.shape[1], conv_state_len
    assert 0 < state_len <= conv_state.shape[2], "conv window wider than the pool"
    assert x.stride(-1) == 1, "conv input must be contiguous along the feature dim"
    BLOCK = 1024
    nblk_ssm = (hkv + BLOCK - 1) // BLOCK
    grid = (n, nblk_ssm + (d + BLOCK - 1) // BLOCK)
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
        h_rows,
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
        HAS_H_ROWS=h_rows is not None,
    )
