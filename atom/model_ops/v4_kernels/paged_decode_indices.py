# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 paged-decode index scatter — two Triton kernels, one per half of the job.

`build_v4_paged_decode_indptr` sizes each token's slice and prefix-sums the
three ragged layouts; `write_v4_paged_decode_indices` then fills the SWA
window-prefix paged offsets into the three destination buffers
(`kv_indices_swa` / `kv_indices_csa` / `kv_indices_hca`). Call them in that
order — the writer reads the indptrs the builder wrote.

The window row is computed inline inside the kernel from `positions[t]` via
`pool_index.window_row` — no `[T, win]` window_topk staging buffer, no separate
CPU build + H2D copy. One row formula per compress class: the three output
buffers each serve one class, and the classes interleave their windows by
different strides.

Layout: ragged-packed. Each token's slice holds an SWA prefix of length
`n = min(positions[t]+1, win)` plus a per-buffer compress section; the
`swa_indptr` / `csa_indptr` / `hca_indptr` cumsums reflect this ragged
sizing. Within each token's slice the SWA prefix is written at the TAIL
(`[indptr[t+1] - n, indptr[t+1])`) and the compress section (CSA topk /
HCA committed) occupies the head.

Caller contract:
- Grid = T (one program per token).
- `batch_id_per_token[:T]` may carry `-1` sentinels in the CG-padded tail —
  kernel checks and bails (matches `_attach_v4_per_fwd_meta` convention).
- `swa_indptr` / `csa_indptr` / `hca_indptr` must reflect the ragged-packed
  sizing: per-token slot count = `min(positions[t]+1, win) + n_compress[t]`,
  where `n_compress[t]` is 0 for SWA, `min((pos+1)//4, index_topk)` for CSA and
  `(pos+1)//128` for HCA — per-token counts, see `v4_pool_geometry`. The
  builder above does that; a caller supplying its own indptrs owes the same.
- `swa_indices` / `csa_indices` / `hca_indices` capacity ≥ corresponding
  indptr[T]; this kernel writes the SWA-prefix segment at the slice tail
  `[indptr[t+1] - n, indptr[t+1])` per token, and — given
  `hca_block_tables` — the HCA compress section at the head, which then
  tile the slice exactly. CSA's head is filled per layer by
  `csa_translate_pack`; a caller that does not pass block tables fills
  HCA's itself.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.v4_kernels.pool_index import (
    compress_row,
    served_window_params,
    window_constexprs,
    window_row,
)
from atom.utils.decorators import mark_trace


def hca_compress_paged_offsets(
    entry_idx, bid_per_entry, block_tables_np, envelope_rows, hca_rows_per_block
):
    """HCA compress entry -> pool row (numpy, decode index build).

    The compressor packs ``hca_rows_per_block = block_size // hca_ratio`` rows
    per physical block, so entry ``e`` lives in physical block
    ``block_tables[bid, e // hca_rows_per_block]`` at row
    ``e % hca_rows_per_block`` — which is `pool_index.compress_row`, restated in
    numpy because this section is built on the host.

    ``entry_idx`` / ``bid_per_entry`` are int arrays of equal length; returns an
    int32 array of the same length. Shared by ``_attach_v4_paged_decode_meta`` and
    covered by ``tests/test_decode_indices_paged.py`` so the packing stays correct
    for V4 ``block_size=256`` (``hca_rows_per_block=2``).
    """
    blk = entry_idx // hca_rows_per_block
    row = entry_idx % hca_rows_per_block
    return (block_tables_np[bid_per_entry, blk] * envelope_rows + row).astype(np.int32)


@triton.jit
def _v4_paged_decode_indices_kernel(
    state_slot_per_seq_ptr,  # [bs] int32 — per-request SWA ring slot
    batch_id_per_token_ptr,  # [T+pad] int — sentinel -1 in pad tail
    positions_ptr,  # [T+pad] int — global token position
    swa_indptr_ptr,  # [T+1] int32 — ragged SWA-prefix cumsum
    csa_indptr_ptr,  # [T+1] int32 — ragged (SWA + CSA topk)
    hca_indptr_ptr,  # [T+1] int32 — ragged (SWA + HCA committed)
    swa_indices_ptr,  # [swa_total] int32, output
    csa_indices_ptr,  # [csa_total] int32, output (writes SWA-prefix segment only)
    hca_indices_ptr,  # [hca_total] int32, output (writes SWA-prefix segment only)
    dense_dest_ptr,  # [>=T] int32, output — where THIS token's own KV row goes
    csa_dest_ptr,
    hca_dest_ptr,
    dense_ring_start,  # per-class window bases; the only terms the boundary moves
    csa_ring_start,
    hca_ring_start,
    hca_block_tables_ptr,  # [bs, bt_stride_bs] int32 — only read if HCA_FROM_BT
    hca_bt_stride_bs,
    win: tl.constexpr,  # window_size — max SWA prefix slots
    BLOCK_N: tl.constexpr,  # next_pow2(win)
    HAS_CSA: tl.constexpr,  # caller has layers of this class to serve
    HAS_HCA: tl.constexpr,
    HAS_DENSE: tl.constexpr,
    DENSE_RING_SLOTS: tl.constexpr,
    DENSE_SLOT_ROWS: tl.constexpr,
    DENSE_RING_STRIDE: tl.constexpr,
    DENSE_RUN_ROWS: tl.constexpr,
    CSA_RING_SLOTS: tl.constexpr,
    CSA_SLOT_ROWS: tl.constexpr,
    CSA_RING_STRIDE: tl.constexpr,
    CSA_RUN_ROWS: tl.constexpr,
    HCA_RING_SLOTS: tl.constexpr,
    HCA_SLOT_ROWS: tl.constexpr,
    HCA_RING_STRIDE: tl.constexpr,
    HCA_RUN_ROWS: tl.constexpr,
    HCA_FROM_BT: tl.constexpr,  # also fill the HCA compress section
    HCA_ENVELOPE_ROWS: tl.constexpr,
    HCA_ROWS_PER_BLOCK: tl.constexpr,
):
    """One program per token. Writes `n = min(positions[t]+1, win)` pool rows
    to the SWA prefix segment, placed at the TAIL of each token's slice in the
    SWA/CSA/HCA index buffers (the compress section occupies the head).

    The three buffers no longer share one value. Each serves the layers of one
    compress class, and a class's layer stride is what its window rows are
    interleaved by, so the same `(slot, abs_pos)` names a different row in each.
    They agreed before only because the window was one flat region ahead of the
    blocks, which is exactly the arrangement that pinned the pool's split.
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched

    pos = tl.load(positions_ptr + t)
    # `n` = actual valid SWA prefix count. Cast to match `win` (compile-time
    # int) — pos is i32/i64 from positions buffer.
    n = tl.minimum(pos + 1, win)
    i = tl.arange(0, BLOCK_N)
    mask = i < n
    abs_pos = pos - n + 1 + i  # ∈ [0, pos] for valid i
    # Ring: `n <= win <= ring_slots`, and the newest position this request has
    # written is `pos` (plus its drafts), so every `abs_pos` here is inside the
    # ring's last lap.
    slot = tl.load(state_slot_per_seq_ptr + bid)

    if HAS_DENSE:
        # SWA prefix segment lives at the TAIL of each token's slice (compress
        # section fills the head). Write base = slice END (indptr[t+1]) - n.
        # For the SWA buffer (compress=0) end-n == indptr[t], same as a head
        # write.
        swa_end = tl.load(swa_indptr_ptr + t + 1)
        tl.store(
            swa_indices_ptr + swa_end - n + i,
            window_row(
                slot,
                abs_pos,
                dense_ring_start,
                DENSE_RING_SLOTS,
                DENSE_SLOT_ROWS,
                DENSE_RING_STRIDE,
                DENSE_RUN_ROWS,
            ),
            mask=mask,
        )
        # Where this token's own KV row lands. Same formula at `pos` — the last
        # position of its own window — handed to the fused SWA write so the two
        # cache-writing paths cannot drift, and so no kernel outside this file
        # has to know how a window is laid out.
        tl.store(
            dense_dest_ptr + t,
            window_row(
                slot,
                pos,
                dense_ring_start,
                DENSE_RING_SLOTS,
                DENSE_SLOT_ROWS,
                DENSE_RING_STRIDE,
                DENSE_RUN_ROWS,
            ),
        )
    if HAS_CSA:
        csa_end = tl.load(csa_indptr_ptr + t + 1)
        tl.store(
            csa_indices_ptr + csa_end - n + i,
            window_row(
                slot,
                abs_pos,
                csa_ring_start,
                CSA_RING_SLOTS,
                CSA_SLOT_ROWS,
                CSA_RING_STRIDE,
                CSA_RUN_ROWS,
            ),
            mask=mask,
        )
        tl.store(
            csa_dest_ptr + t,
            window_row(
                slot,
                pos,
                csa_ring_start,
                CSA_RING_SLOTS,
                CSA_SLOT_ROWS,
                CSA_RING_STRIDE,
                CSA_RUN_ROWS,
            ),
        )
    if HAS_HCA:
        hca_end = tl.load(hca_indptr_ptr + t + 1)
        tl.store(
            hca_indices_ptr + hca_end - n + i,
            window_row(
                slot,
                abs_pos,
                hca_ring_start,
                HCA_RING_SLOTS,
                HCA_SLOT_ROWS,
                HCA_RING_STRIDE,
                HCA_RUN_ROWS,
            ),
            mask=mask,
        )
        tl.store(
            hca_dest_ptr + t,
            window_row(
                slot,
                pos,
                hca_ring_start,
                HCA_RING_SLOTS,
                HCA_SLOT_ROWS,
                HCA_RING_STRIDE,
                HCA_RUN_ROWS,
            ),
        )
        if HCA_FROM_BT:
            # Compress section, at the slice HEAD. Its length is not passed in:
            # the slice was sized `n + (positions[t]+1)//HCA_RATIO`, so what the SWA
            # prefix leaves is exactly that count — the two tile the
            # slice, which is why nothing pre-fills it.
            hca_start = tl.load(hca_indptr_ptr + t)
            n_hca = hca_end - hca_start - n
            # Entry k is row k % HCA_ROWS_PER_BLOCK of physical block
            # k // HCA_ROWS_PER_BLOCK, matching the compressor's cache view.
            bt_row = bid * hca_bt_stride_bs
            for j in tl.range(0, n_hca, BLOCK_N):
                k = j + i
                k_mask = k < n_hca
                bt = tl.load(
                    hca_block_tables_ptr + bt_row + k // HCA_ROWS_PER_BLOCK,
                    mask=k_mask,
                    other=0,
                )
                tl.store(
                    hca_indices_ptr + hca_start + k,
                    compress_row(bt, k % HCA_ROWS_PER_BLOCK, HCA_ENVELOPE_ROWS),
                    mask=k_mask,
                )


@mark_trace
def write_v4_paged_decode_indices(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor | None,
    hca_indptr: torch.Tensor | None,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor | None,
    hca_indices: torch.Tensor | None,
    dest_rows: dict[int, torch.Tensor],
    T: int,
    win: int,
    geometry: UnifiedPoolGeometry,
    hca_block_tables: torch.Tensor | None = None,
    hca_rows_per_block: int = 0,
    prefix: str = "",
) -> None:
    """In-place fill SWA / CSA / HCA window-prefix offsets via a single
    Triton kernel. Replaces the prior `_build_window_topk_np` (CPU O(T·win))
    + `index_copy_` chain. All inputs are persistent forward_vars buffers —
    no allocator churn.

    SWA rows are ring-addressed via the request's state slot, one formula per
    compress class (`geometry.window_params`) — the classes interleave their
    windows differently, so the three buffers get three different values for the
    same token.

    Args (all GPU tensors except T/win/geometry):
      state_slot_per_seq:  [bs] int32 — per-request SWA ring slot.
      batch_id_per_token:  [>=T]  int   — token→seq map; -1 sentinel skipped.
      positions:           [>=T]  int   — global token position
                                   (forward_vars["positions"]); used to derive
                                   `n = min(pos+1, win)` per token + the paged
                                   offset for each window position.
      swa_indptr:          [>=T+1] int32 — ragged SWA-prefix cumsum, where
                                   `swa_indptr[t+1] - swa_indptr[t] =
                                    min(positions[t]+1, win)`.
      csa_indptr:          [>=T+1] int32 — ragged CSA buffer indptr (SWA
                                   prefix + CSA topk per token). `None` with
                                   `csa_indices` when the caller runs no layer
                                   of that class.
      hca_indptr:          [>=T+1] int32 — ragged HCA buffer indptr (SWA
                                   prefix + HCA committed per token). `None`
                                   likewise.
      swa_indices:         [>=swa_indptr[T]] int32 OUT — fully written by
                                   this kernel (no other source), unless no
                                   layer is dense, when it is left untouched:
                                   the dense class is the only reader and a
                                   geometry can turn out not to have one.
      csa_indices:         [>=csa_indptr[T]] int32 OUT — SWA prefix written
                                   here at the slice tail
                                   `[csa_indptr[t+1] - n, csa_indptr[t+1])`;
                                   CSA topk section (slice head) filled
                                   per-layer by `csa_translate_pack`. `None`
                                   skips the class — the three buffers hold
                                   DIFFERENT rows now, so a caller that only
                                   needs one can no longer alias them all onto
                                   it and let the writes agree.
      hca_indices:         [>=hca_indptr[T]] int32 OUT — same semantics, plus
                                   the compress section at the slice head when
                                   `hca_block_tables` is given. Opt-in because
                                   the bridges fill that section themselves,
                                   from a different row formula.
      hca_block_tables:    [bs, cols] int32 — the source for that fill. Must
                                   be numbered like `batch_id_per_token`: in a
                                   TBO ubatch, the ubatch-sliced buffer, not
                                   the global one.
      hca_rows_per_block:  int — `block_size // HCA_RATIO`, the rows the
                                   compressor packs per physical block.
      dest_rows:           {ratio: [>=T] int32 OUT} — the row token `t`'s own
                                   KV goes to in a layer of that class. Needed
                                   for every class that is both served by the
                                   geometry and enabled here; the rest are
                                   handed some other class's buffer and never
                                   written. The fused SWA write reads it
                                   instead of deriving
                                   the row itself, which is what keeps the pool
                                   layout out of the fused kernels.
                                   **Defined only where `batch_id_per_token[t]
                                   >= 0`**, and only for `t < T`: these are
                                   persistent buffers, so everywhere else holds
                                   an earlier forward's rows. Every consumer
                                   gates on the same batch id, which is what
                                   makes both stale ranges unreachable — a
                                   partial sentinel fill here would cover the
                                   first and not the second, and would read as
                                   a guarantee the buffer does not give.
      T:                   int — number of real tokens (grid size).
      win:                 int — SWA window size (typically 128 for V4-Pro).
      geometry:            the pool's `UnifiedPoolGeometry`; supplies one
                                 `WindowParams` per compress class.
    """
    if T == 0:
        return
    assert state_slot_per_seq.dim() == 1
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert swa_indptr.dim() == 1 and swa_indptr.shape[0] >= T + 1
    assert swa_indices.dim() == 1
    has_csa = csa_indices is not None
    has_hca = hca_indices is not None
    assert has_csa == (csa_indptr is not None)
    assert has_hca == (hca_indptr is not None)
    hca_from_bt = hca_block_tables is not None
    if hca_from_bt:
        assert has_hca, "hca_block_tables given without an HCA buffer to fill"
        assert hca_rows_per_block > 0
        assert hca_block_tables.dim() == 2

    # A class that is off — not in the geometry, or one the caller did not ask
    # for — still needs a pointer and a value for every `constexpr`, because
    # Triton takes them by position. It borrows another class's; its `HAS_`
    # flag is what keeps them away from a store.
    #
    # The classes that are ON index `served` directly, on purpose. A caller
    # asking for rows of a class no layer belongs to then gets a KeyError here
    # instead of a plausible row written from borrowed parameters — and unlike
    # an `assert`, that holds under `python -O`.
    served = served_window_params(geometry)
    if not served or not dest_rows:
        raise ValueError(
            "a V4 index build needs at least one compress class and one "
            f"destination buffer; got {sorted(served)} and {sorted(dest_rows)}"
        )
    has_dense = DENSE_RATIO in served
    borrowed = next(iter(served.values()))
    borrowed_dest = next(iter(dest_rows.values()))
    dense = served[DENSE_RATIO] if has_dense else borrowed
    csa = served[CSA_RATIO] if has_csa else borrowed
    hca = served[HCA_RATIO] if has_hca else borrowed
    BLOCK_N = triton.next_power_of_2(win)
    _v4_paged_decode_indices_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        positions,
        swa_indptr,
        # Triton wants a pointer even for a class the caller switched off, and
        # the SWA buffer is the one that is always there.
        csa_indptr if has_csa else swa_indptr,
        hca_indptr if has_hca else swa_indptr,
        swa_indices,
        csa_indices if has_csa else swa_indices,
        hca_indices if has_hca else swa_indices,
        dest_rows[DENSE_RATIO] if has_dense else borrowed_dest,
        dest_rows[CSA_RATIO] if has_csa else borrowed_dest,
        dest_rows[HCA_RATIO] if has_hca else borrowed_dest,
        dense.ring_start,
        csa.ring_start,
        hca.ring_start,
        hca_block_tables if hca_from_bt else swa_indices,
        hca_block_tables.stride(0) if hca_from_bt else 0,
        win=win,
        BLOCK_N=BLOCK_N,
        HAS_CSA=has_csa,
        HAS_HCA=has_hca,
        HAS_DENSE=has_dense,
        **window_constexprs(dense, "DENSE_"),
        **window_constexprs(csa, "CSA_"),
        **window_constexprs(hca, "HCA_"),
        HCA_FROM_BT=hca_from_bt,
        HCA_ENVELOPE_ROWS=geometry.envelope_rows,
        HCA_ROWS_PER_BLOCK=hca_rows_per_block,
    )


def write_v4_paged_decode_indices_reference(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor | None,
    hca_indptr: torch.Tensor | None,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor | None,
    hca_indices: torch.Tensor | None,
    dest_rows: dict[int, torch.Tensor],
    T: int,
    win: int,
    geometry: UnifiedPoolGeometry,
) -> None:
    """Pure-PyTorch reference equivalent of `write_v4_paged_decode_indices`.
    For unit tests and bisect verification. Mirrors the kernel: per-token
    ragged-packed write, ring-addressed via the state slot, with the row taken
    from the geometry rather than restated here.
    """
    if T == 0:
        return
    params = served_window_params(geometry)
    outputs = {
        DENSE_RATIO: (swa_indices, swa_indptr),
        CSA_RATIO: (csa_indices, csa_indptr),
        HCA_RATIO: (hca_indices, hca_indptr),
    }
    bid = batch_id_per_token[:T].long()
    pos_t = positions[:T].long()
    valid = bid >= 0
    # n = min(pos+1, win) per token; clamp invalid rows to 0 to skip writes.
    n_per_tok = torch.minimum(pos_t + 1, torch.full_like(pos_t, win))
    n_per_tok = torch.where(valid, n_per_tok, torch.zeros_like(n_per_tok))
    for t in range(T):
        n = int(n_per_tok[t].item())
        if n == 0:
            continue
        p = int(pos_t[t].item())
        b = int(bid[t].item())
        abs_pos = range(p - n + 1, p + 1)
        slot = int(state_slot_per_seq[b])
        # SWA prefix segment at the slice TAIL (compress section fills the head).
        for ratio, (buf, indptr) in outputs.items():
            # A class the caller switched off, or one the geometry does not
            # have at all — the kernel skips both, on `HAS_*` respectively
            # `HAS_DENSE`.
            if buf is None or ratio not in params:
                continue
            end = int(indptr[t + 1].item())
            rows = [params[ratio].index(slot, int(q)) for q in abs_pos]
            buf[end - n : end] = torch.tensor(rows, dtype=buf.dtype, device=buf.device)
            dest_rows[ratio][t] = params[ratio].index(slot, p)


# ---------------------------------------------------------------------------
# The three indptr cumsums the writer above consumes, plus the CSA per-token
# visibility the indexer consumes. One launch, no host arithmetic and no H2D:
# every input already lives on the device for other reasons.
# ---------------------------------------------------------------------------


@triton.jit
def _v4_decode_indptr_kernel(
    batch_id_per_token_ptr,  # [T_pad] int — -1 sentinel in the CG-padded tail
    positions_ptr,  # [T_pad] int — global token position (int64 in production)
    ragged_lens_ptr,  # [bs] int32 — DSpark per-seq query count (RECT only)
    cu_q_ptr,  # [bs] int32 — per-seq first token index (RECT only)
    swa_indptr_ptr,  # [T_pad+1] int32 OUT
    csa_indptr_ptr,  # [T_pad+1] int32 OUT
    hca_indptr_ptr,  # [T_pad+1] int32 OUT
    csa_n_committed_ptr,  # [T_pad] or [rect_bs*rect_full_q] int32 OUT
    t_pad,  # token count including the CG pad — RUNTIME, see below
    n_committed_rows,  # length of the visibility output, whichever layout
    rect_full_q,  # rectangle width; only read when IS_RECT
    WIN: tl.constexpr,
    CSA_R: tl.constexpr,
    HCA_R: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    IS_RECT: tl.constexpr,  # False = token-indexed output; True = DSpark rect
    BLOCK: tl.constexpr,
):
    """One program: a running offset per class over `t_pad` tokens, then the
    CSA visibility over `n_committed_rows` destination rows.

    A prefix sum has to see every earlier token, so this is one serial scan
    rather than a grid. Against a 1.7us launch floor every multi-launch
    alternative loses on the floor alone, and the whole path is launch-bound
    anyway (host 13-15us vs 3.5-5.7us on device), so the parallel rewrites that
    look tempting here buy stream time nobody is waiting on. Measurements in
    `/app/logs_claude/v4_indptr_scan_measure.md`.

    `t_pad` is a runtime argument on purpose: eager decode hands a fresh token
    count almost every step, and a `constexpr` would make each one its own JIT
    variant -- a build meant to save host work paying for it in recompiles.

    Both compress counts come from the token's own position; the rule and why
    the per-sequence one was dropped live next to `CSA_RATIO` in
    `v4_pool_geometry`. The draft mid-step never gets here -- `prepare_mtp_decode`
    rebuilds SWA alone.
    """
    tl.store(swa_indptr_ptr, 0)
    tl.store(csa_indptr_ptr, 0)
    tl.store(hca_indptr_ptr, 0)
    acc_swa = 0
    acc_csa = 0
    acc_hca = 0
    for base in tl.range(0, t_pad, BLOCK):
        idx = base + tl.arange(0, BLOCK)
        in_range = idx < t_pad
        bid = tl.load(batch_id_per_token_ptr + idx, mask=in_range, other=-1)
        # A padded slot contributes 0 to every class, which is what makes the
        # tail of each indptr flat — the `kv_len == 0` the readers bail on.
        live = in_range & (bid >= 0)
        pos = tl.load(positions_ptr + idx, mask=live, other=0).to(tl.int32)
        n = tl.minimum(pos + 1, WIN)
        n_csa = (pos + 1) // CSA_R
        n_hca = (pos + 1) // HCA_R
        # Visibility and output capacity are different quantities: the indexer
        # scans every causally visible row, the translated output reserves at
        # most `index_topk` of them.
        cnt_swa = tl.where(live, n, 0)
        cnt_csa = tl.where(live, n + tl.minimum(n_csa, INDEX_TOPK), 0)
        cnt_hca = tl.where(live, n + n_hca, 0)
        tl.store(
            swa_indptr_ptr + idx + 1,
            acc_swa + tl.cumsum(cnt_swa, axis=0),
            mask=in_range,
        )
        tl.store(
            csa_indptr_ptr + idx + 1,
            acc_csa + tl.cumsum(cnt_csa, axis=0),
            mask=in_range,
        )
        tl.store(
            hca_indptr_ptr + idx + 1,
            acc_hca + tl.cumsum(cnt_hca, axis=0),
            mask=in_range,
        )
        acc_swa += tl.sum(cnt_swa)
        acc_csa += tl.sum(cnt_csa)
        acc_hca += tl.sum(cnt_hca)

        if not IS_RECT:
            # row == token, so the lane that owns a slot here owns it in the
            # destination too, and the CG pad takes its 0 from the same store.
            tl.store(csa_n_committed_ptr + idx, tl.where(live, n_csa, 0), mask=in_range)

    if IS_RECT:
        # ----- CSA visibility, one pass over the DESTINATION -----
        # DSpark fp8 lays this out as a `full_q`-wide rectangle with each
        # sequence's tokens flushed right, so token -> row is a scatter and some
        # rows hold no token. Walking rows writes every one exactly once from
        # its owning lane; walking tokens would leave the rest to a clearing
        # pass another warp could still be in -- a hazard to fence rather than
        # one that cannot arise. They do get read: the indexer runs the captured
        # grid and takes this as each row's end.
        for row_base in tl.range(0, n_committed_rows, BLOCK):
            row = row_base + tl.arange(0, BLOCK)
            in_range = row < n_committed_rows
            seq = row // rect_full_q
            offset = row % rect_full_q
            # A row holds a token only past its band's leading slack; bands past
            # the batch read a zero `ragged_lens`, so the whole band is slack.
            lead = rect_full_q - tl.load(ragged_lens_ptr + seq, mask=in_range, other=0)
            token = tl.load(cu_q_ptr + seq, mask=in_range, other=0) + offset - lead
            mapped = in_range & (offset >= lead)
            # `mapped` already implies a token inside this sequence's span, so
            # this gather is redundant WHILE `cu_q` is the cumsum of
            # `ragged_lens`. Nothing here can check that, and the two now come
            # from different arrays built in different files, so it stays.
            bid = tl.load(batch_id_per_token_ptr + token, mask=mapped, other=-1)
            live = mapped & (bid >= 0)
            pos = tl.load(positions_ptr + token, mask=live, other=0).to(tl.int32)
            tl.store(
                csa_n_committed_ptr + row,
                tl.where(live, (pos + 1) // CSA_R, 0),
                mask=in_range,
            )


@mark_trace
def build_v4_paged_decode_indptr(
    *,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    csa_n_committed_per_token: torch.Tensor,
    T_pad: int,
    win: int,
    index_topk: int,
    rect_full_q: int = 0,
    ragged_lens: torch.Tensor | None = None,
    cu_q_per_seq: torch.Tensor | None = None,
) -> None:
    """Fill the three ragged indptr cumsums and the CSA per-token visibility.

    Every argument is a persistent forward_vars buffer already on the device,
    so this replaces the host's per-token arithmetic, three numpy cumsums and
    four H2D copies with one launch. Call it before
    `write_v4_paged_decode_indices`, which reads all three indptrs.

    A token's compress counts follow from its position alone -- see the kernel
    on why the sequence-level `ctx // ratio` is not a second bound.

    `csa_n_committed_per_token` is the visibility `(pos+1)//CSA_RATIO`, NOT
    capped by `index_topk`: that bounds what the CSA slice reserves, not what
    the indexer may look at.

    `rect_full_q > 0` switches the output to DSpark's right-aligned
    `[bs, full_q]` rectangle; `ragged_lens` / `cu_q_per_seq` are then required.

    The whole of `csa_n_committed_per_token` is this call's to fill, however it
    is laid out: pass the slice you want written and the rows no token maps to
    are set to 0. That filler is right only because the decode scorers pass
    `next_n=1`, making `row_len == rowEnds[r]`; the per-seq twin pads with
    `index_topk` for the opposite reason (`_attach_v4_per_fwd_meta` -- there,
    too SMALL hangs the radix loop).
    """
    # Every bound below fixes a device-side index, so violating one is an
    # out-of-bounds read rather than a wrong number -- `ValueError` for the same
    # reason `write_v4_paged_decode_indices` gives above: it holds under
    # `python -O`.
    for name, buf, want in (
        ("batch_id_per_token", batch_id_per_token, T_pad),
        ("positions", positions, T_pad),
        ("swa_indptr", swa_indptr, T_pad + 1),
        ("csa_indptr", csa_indptr, T_pad + 1),
        ("hca_indptr", hca_indptr, T_pad + 1),
    ):
        if buf.shape[0] < want:
            raise ValueError(
                f"{name} holds {buf.shape[0]} entries but this build indexes "
                f"{want} of them (T_pad={T_pad})"
            )
    n_rows = csa_n_committed_per_token.shape[0]
    if rect_full_q > 0:
        if ragged_lens is None or cu_q_per_seq is None:
            raise ValueError(
                "the DSpark rectangle needs the per-seq query counts and starts"
            )
        # The rows carry `seq = row // full_q` straight into both per-seq
        # tensors, so their length is what bounds that index. Nothing else does:
        # `rect_bs` is decided by the caller, and a short `ragged_lens` would be
        # an unmasked device read of whatever follows it.
        bands = -(-n_rows // rect_full_q)
        for name, buf in (("ragged_lens", ragged_lens), ("cu_q_per_seq", cu_q_per_seq)):
            if buf.shape[0] < bands:
                raise ValueError(
                    f"{name} has {buf.shape[0]} entries but the rectangle spans "
                    f"{bands} bands ({n_rows} rows of {rect_full_q})"
                )
    elif n_rows != T_pad:
        # Token-indexed: the destination IS the token axis, one row each. A
        # longer buffer would leave its tail reading tokens this forward never
        # declared -- live values from the last one, on a persistent buffer.
        raise ValueError(
            f"token-indexed output must be exactly T_pad={T_pad} rows, got "
            f"{n_rows}; slice it, or pass the rectangle's parameters"
        )
    _v4_decode_indptr_kernel[(1,)](
        batch_id_per_token,
        positions,
        # Triton takes a pointer per parameter whatever the constexpr says; the
        # rect-only ones borrow a buffer of the right dtype when switched off.
        ragged_lens if rect_full_q > 0 else batch_id_per_token,
        cu_q_per_seq if rect_full_q > 0 else batch_id_per_token,
        swa_indptr,
        csa_indptr,
        hca_indptr,
        csa_n_committed_per_token,
        T_pad,
        # The destination's own length, so the kernel fills exactly what it was
        # handed: `T_pad` rows token-indexed, `rect_bs * full_q` as a rectangle.
        csa_n_committed_per_token.shape[0],
        rect_full_q,
        WIN=win,
        CSA_R=CSA_RATIO,
        HCA_R=HCA_RATIO,
        INDEX_TOPK=index_topk,
        IS_RECT=rect_full_q > 0,
        BLOCK=1024,
    )


def build_v4_paged_decode_indptr_reference(
    *,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    csa_n_committed_per_token: torch.Tensor,
    T_pad: int,
    win: int,
    index_topk: int,
    rect_full_q: int = 0,
    ragged_lens: torch.Tensor | None = None,
    cu_q_per_seq: torch.Tensor | None = None,
) -> None:
    """Pure-PyTorch equivalent of `build_v4_paged_decode_indptr`, for the
    kernel-vs-reference tests. Same argument contract, including owning the
    whole of `csa_n_committed_per_token`.

    Stated as a scatter where the kernel walks the destination, so the parity
    test compares two derivations of the layout rather than one twice.
    """
    bid = batch_id_per_token[:T_pad].long()
    live = bid >= 0
    safe_bid = torch.where(live, bid, torch.zeros_like(bid))
    pos = positions[:T_pad].long()
    n = torch.minimum(pos + 1, torch.full_like(pos, win))
    n_csa = (pos + 1) // CSA_RATIO
    n_hca = (pos + 1) // HCA_RATIO
    zero = torch.zeros_like(n)
    counts = {
        "swa": torch.where(live, n, zero),
        "csa": torch.where(
            live, n + torch.minimum(n_csa, torch.full_like(n_csa, index_topk)), zero
        ),
        "hca": torch.where(live, n + n_hca, zero),
    }
    for key, out in (("swa", swa_indptr), ("csa", csa_indptr), ("hca", hca_indptr)):
        out[0] = 0
        out[1 : T_pad + 1] = torch.cumsum(counts[key], dim=0).to(out.dtype)
    if rect_full_q > 0:
        assert ragged_lens is not None and cu_q_per_seq is not None
        dst = (
            safe_bid * rect_full_q
            + (rect_full_q - ragged_lens[safe_bid].long())
            + (torch.arange(T_pad, device=bid.device) - cu_q_per_seq[safe_bid].long())
        )
    else:
        dst = torch.arange(T_pad, device=bid.device)
    csa_n_committed_per_token.zero_()
    csa_n_committed_per_token[dst[live]] = n_csa[live].to(
        csa_n_committed_per_token.dtype
    )
