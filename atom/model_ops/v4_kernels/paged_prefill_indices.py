# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 paged-prefill index scatter — single Triton kernel writes the four
per-fwd index buffers consumed by `sparse_attn_v4_paged_prefill`:

  - ``kv_indices_extend``       : per-fwd `kv` tensor row indices for the
                                  in-chunk SWA tail (one shared buffer).
  - ``kv_indices_prefix_swa``   : Dense path — SWA prior-chunk paged offsets
                                  into `unified_kv`.
  - ``kv_indices_prefix_csa``   : CSA path — SWA prefix segment written at the
                                  slice TAIL; the CSA topk HEAD section is filled
                                  per layer by ``csa_translate_pack`` (head-CSA /
                                  tail-SWA convention, matching decode, #1116).
  - ``kv_indices_prefix_hca``   : HCA path — SWA prefix segment + the HCA
                                  groups closed at or before the token's own
                                  position, both fully written.

Replaces the CPU numpy build in
``DeepseekV4AttentionMetadataBuilder._build_paged_prefill_meta`` (per-fwd
`_segment_indices` + cumsum + scatter chain + pinned H2D). The kernel runs
entirely on GPU and is invoked AFTER the caller has computed the four
indptrs via ``torch.cumsum`` (also on GPU).

Caller responsibilities (no copies done here):
  - The CSA slice is fully covered without any ``-1`` pre-fill: this kernel
    writes the SWA prefix at the slice TAIL (length ``prefix_swa_count``) and
    ``csa_translate_pack`` writes the CSA topk at the HEAD (length
    ``valid_k = slice_len - prefix_swa_count``) per layer — together they cover
    ``[indptr[t], indptr[t+1])`` with no gap. (HCA / Dense buffers are likewise
    fully written by this kernel.)
  - Compute and stage the four indptr buffers and the per-seq scalar inputs.

Per-token quantities (kernel-computed from inputs; mirror the formulas in
``_build_paged_prefill_meta``):
  token_pos_in_chunk[t] = positions[t] - chunk_start[bid]
  swa_low[t]            = max(positions[t] - win + 1, 0)
  extend_count[t]       = min(token_pos_in_chunk[t] + 1, win)
  prefix_swa_count[t]   = max(chunk_start[bid] - swa_low[t], 0)

Per-token pool row for SWA prefix entries (the same formula `swa_write` and
`_attach_v4_paged_decode_meta` use, from `pool_index.window_row`):
  row[t,k] = window.index(state_slot[bid], swa_low[t] + k)
one `window` per compress class, since the three output buffers each serve one
class and the classes interleave their windows by different layer strides.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
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


@triton.jit
def _v4_paged_prefill_indices_kernel(
    # Per-token inputs.
    positions_ptr,  # [T] int — global token position
    bid_per_token_ptr,  # [T] int — batch id per token (==`np.repeat(arange(bs), tnps)`)
    # Per-seq inputs (indexed by bid).
    chunk_start_per_seq_ptr,  # [bs] int — current chunk's absolute start position
    cu_seqlens_q_per_seq_ptr,  # [bs] int — per-seq prefix sum start in per-fwd kv tensor
    state_slot_per_seq_ptr,  # [bs] int — per-seq SWA ring slot
    block_tables_ptr,  # [bs, MAX_BLOCKS] int — compressed pool block ids (HCA)
    bt_stride_bs,  # row stride of block_tables
    # Indptrs (already cumsum'd by caller, all length [T+1]).
    extend_indptr_ptr,
    prefix_swa_indptr_ptr,
    prefix_csa_indptr_ptr,
    prefix_hca_indptr_ptr,
    # Output buffers.
    extend_indices_ptr,
    prefix_swa_indices_ptr,
    prefix_csa_indices_ptr,
    prefix_hca_indices_ptr,
    # Constants.
    win: tl.constexpr,
    dense_ring_start,  # per-class window bases; the only terms the boundary moves
    csa_ring_start,
    hca_ring_start,
    HCA_RATIO: tl.constexpr,  # HCA compress ratio (128) for per-token causal cap
    HCA_ROWS_PER_BLOCK: tl.constexpr,  # HCA rows per block (block_size // HCA_RATIO)
    ENVELOPE_ROWS: tl.constexpr,  # rows one block occupies across all layers
    BLOCK_N: tl.constexpr,  # next_pow2(win) — covers SWA prefix and extend segments
    HAS_DENSE: tl.constexpr,  # geometry has layers of this class to serve
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
):
    """One program per token. Writes four per-token segments:

    - extend         : ``[extend_indptr[t], extend_indptr[t]+extend_count[t])``
    - prefix SWA     : in swa / hca prefix buffers at the slice HEAD
                        ``[*_indptr[t], *_indptr[t]+prefix_swa_count[t])``; in the
                        csa prefix buffer at the slice TAIL
                        ``[csa_indptr[t+1]-prefix_swa_count[t], csa_indptr[t+1])``
    - HCA compress   : ``[prefix_hca_indptr[t]+prefix_swa_count[t], +n_hca[bid])``
                        in prefix_hca_indices

    Per-token bounded segments (extend, SWA prefix) fit in one ``BLOCK_N``
    vector. HCA compress can be up to ``max_model_len // 128`` per token
    (e.g. 8192 at V4-Pro 1M ctx) — looped in ``BLOCK_N`` chunks.
    """
    t = tl.program_id(0)

    bid = tl.load(bid_per_token_ptr + t)
    pos = tl.load(positions_ptr + t)
    chunk_start = tl.load(chunk_start_per_seq_ptr + bid)
    cu_q = tl.load(cu_seqlens_q_per_seq_ptr + bid)
    # Per-token causal HCA visibility (see `v4_pool_geometry`; matches the
    # reference `get_compress_topk_idxs` prefill mask). Here specifically, the
    # sequence's `ctx_end//128` would make a token's output depend on the
    # forward's total length -- chunked and single-shot would disagree.
    n_hca = (pos + 1) // HCA_RATIO

    # Per-token derived quantities (single-pass arithmetic).
    token_pos_in_chunk = pos - chunk_start
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(token_pos_in_chunk + 1, win)
    prefix_swa_count = tl.maximum(chunk_start - swa_low, 0)

    i = tl.arange(0, BLOCK_N)

    # ---- Extend kv_indices: rows in per-fwd kv tensor ----
    # row = cu_q + token_pos_in_chunk - extend_count + 1 + k, k in [0, extend_count)
    ext_base = tl.load(extend_indptr_ptr + t)
    ext_mask = i < extend_count
    ext_start_row = cu_q + token_pos_in_chunk - extend_count + 1
    tl.store(extend_indices_ptr + ext_base + i, ext_start_row + i, mask=ext_mask)

    # ---- SWA prefix rows: written to all three prefix buffers ----
    #   row = window.index(state_slot_per_seq[bid], gp) for that buffer's class,
    #   gp = swa_low + k, k in [0, prefix_swa_count)
    # `prefix_swa_count <= win - 1 < ring_slots` (it is `chunk_start - swa_low`
    # and `swa_low >= pos - win + 1 >= chunk_start - win + 1`), so every position
    # this reads is inside the ring's last lap. That bound is what lets a ring
    # serve chunked prefill at all — the in-chunk part comes from the extend
    # tensor, never from the pool.
    swa_base_hca = tl.load(prefix_hca_indptr_ptr + t)
    swa_mask = i < prefix_swa_count
    global_pos = swa_low + i
    swa_slot = tl.load(state_slot_per_seq_ptr + bid)
    if HAS_DENSE:
        swa_base_swa = tl.load(prefix_swa_indptr_ptr + t)
        tl.store(
            prefix_swa_indices_ptr + swa_base_swa + i,
            window_row(
                swa_slot,
                global_pos,
                dense_ring_start,
                DENSE_RING_SLOTS,
                DENSE_SLOT_ROWS,
                DENSE_RING_STRIDE,
                DENSE_RUN_ROWS,
            ),
            mask=swa_mask,
        )
    # CSA buffer: the SWA prefix goes at the slice TAIL. `csa_translate_pack`
    # writes the CSA topk section at the slice HEAD
    # `[indptr[t], indptr[t]+valid_k)` (valid_k = slice_len - prefix_swa_count),
    # so the SWA prefix must occupy `[indptr[t+1]-prefix_swa_count, indptr[t+1])`.
    # Writing it at the head (the pre-#1116 layout) collides with the CSA topk
    # head write and leaves the tail uninitialized — #1116 moved decode and
    # csa_translate_pack to this head-CSA / tail-SWA convention but missed this
    # prefill writer, corrupting chunked-prefill CSA slices (prefix_swa_count>0).
    csa_end = tl.load(prefix_csa_indptr_ptr + t + 1)
    csa_tail_base = csa_end - prefix_swa_count
    tl.store(
        prefix_csa_indices_ptr + csa_tail_base + i,
        window_row(
            swa_slot,
            global_pos,
            csa_ring_start,
            CSA_RING_SLOTS,
            CSA_SLOT_ROWS,
            CSA_RING_STRIDE,
            CSA_RUN_ROWS,
        ),
        mask=swa_mask,
    )
    tl.store(
        prefix_hca_indices_ptr + swa_base_hca + i,
        window_row(
            swa_slot,
            global_pos,
            hca_ring_start,
            HCA_RING_SLOTS,
            HCA_SLOT_ROWS,
            HCA_RING_STRIDE,
            HCA_RUN_ROWS,
        ),
        mask=swa_mask,
    )

    # ---- HCA compress section: HCA entry k -> paged offset for k in [0, n_hca) ----
    # Written at offset prefix_swa_count past the SWA prefix segment in HCA buffer.
    # Each physical block packs HCA_ROWS_PER_BLOCK rows (block_size // ratio),
    # matching the compressor's cache view [num_blocks, HCA_ROWS_PER_BLOCK,
    # head_dim]: entry k lives in physical block
    # block_tables[bid, k // HCA_ROWS_PER_BLOCK] at row k % HCA_ROWS_PER_BLOCK.
    hca_dst_base = swa_base_hca + prefix_swa_count
    # block_tables row stride is `bt_stride_bs` int32 elements (== max_num_blocks_per_seq).
    bt_row_base = bid * bt_stride_bs
    for j in tl.range(0, n_hca, BLOCK_N):
        k = j + i
        hca_mask = k < n_hca
        blk = k // HCA_ROWS_PER_BLOCK
        slot = k % HCA_ROWS_PER_BLOCK
        bt = tl.load(block_tables_ptr + bt_row_base + blk, mask=hca_mask, other=0)
        tl.store(
            prefix_hca_indices_ptr + hca_dst_base + k,
            compress_row(bt, slot, ENVELOPE_ROWS),
            mask=hca_mask,
        )


@mark_trace
def write_v4_paged_prefill_indices(
    *,
    positions: torch.Tensor,
    bid_per_token: torch.Tensor,
    chunk_start_per_seq: torch.Tensor,
    cu_seqlens_q_per_seq: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    extend_indptr: torch.Tensor,
    prefix_swa_indptr: torch.Tensor,
    prefix_csa_indptr: torch.Tensor,
    prefix_hca_indptr: torch.Tensor,
    extend_indices: torch.Tensor,
    prefix_swa_indices: torch.Tensor,
    prefix_csa_indices: torch.Tensor,
    prefix_hca_indices: torch.Tensor,
    T: int,
    win: int,
    geometry: UnifiedPoolGeometry,
    hca_ratio: int = 128,
    hca_rows_per_block: int = 1,
    prefix: str = "",
) -> None:
    """One-shot GPU build of the V4 paged-prefill index buffers.

    Replaces the CPU numpy build in
    ``DeepseekV4AttentionMetadataBuilder._build_paged_prefill_meta`` (the
    `_segment_indices` + scatter chain). All inputs/outputs are GPU tensors;
    no D2H, no allocator churn beyond the persistent buffers the caller owns.

    Caller is responsible for:
      1. Sizing ``prefix_csa_indices`` so each token's slice is
         ``prefix_swa_count[t] + csa_valid_k[t]`` long. No ``-1`` pre-fill is
         needed: this kernel writes the SWA prefix at the slice tail and
         ``csa_translate_pack`` writes the CSA topk at the head per layer,
         jointly covering the whole slice.
      2. Computing the four indptr cumsums (e.g. via ``torch.cumsum`` over
         the per-token count vectors).
      3. Computing ``bid_per_token`` (e.g.
         ``torch.repeat_interleave(arange(bs), token_num_per_seq)``).

    Per-seq inputs MUST be indexed by ``bid_per_token`` (the kernel reads
    ``chunk_start_per_seq[bid_per_token[t]]`` etc. inline — no per-token
    pre-gather needed by the caller).

    Args (all GPU tensors):
      positions:                 ``[T]``    int — global token positions.
      bid_per_token:             ``[T]``    int — batch id per token.
      chunk_start_per_seq:       ``[bs]``   int — per-seq chunk start.
      cu_seqlens_q_per_seq:      ``[bs]``   int — per-seq cu_seqlens_q[bid]
                                            (NOT the full ``[bs+1]`` cumsum
                                            — caller passes the leading
                                            ``bs`` entries).
      state_slot_per_seq:        ``[bs]``   int — per-seq SWA ring slot.
      block_tables:              ``[bs, mnbs]`` int — per-seq paged blocks.
      extend_indptr:             ``[T+1]``  int.
      prefix_swa_indptr:         ``[T+1]``  int.
      prefix_csa_indptr:         ``[T+1]``  int.
      prefix_hca_indptr:         ``[T+1]``  int.
      extend_indices:            ``[ext_total]`` int OUT — fully written.
      prefix_swa_indices:        ``[swa_total]`` int OUT — fully written,
                                  unless no layer is dense, when it is left
                                  untouched: the dense class is its only
                                  reader and a geometry can turn out not to
                                  have one. The caller allocates it with
                                  ``torch.empty`` and publishes it either way,
                                  so in that case it holds whatever was there.
      prefix_csa_indices:        ``[csa_total]`` int OUT — SWA prefix
                                  segment written at the slice TAIL; CSA topk
                                  HEAD section filled per layer by
                                  ``csa_translate_pack``.
      prefix_hca_indices:        ``[hca_total]`` int OUT — fully written.
      T:                         int — token count (grid size).
      win:                       int — SWA window size (per-token SWA cap).
      geometry:                  the pool's `UnifiedPoolGeometry`; supplies one
                                  `WindowParams` per compress class plus the
                                  envelope stride the compress section needs.
    """
    if T == 0:
        return
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert bid_per_token.dim() == 1 and bid_per_token.shape[0] >= T
    assert chunk_start_per_seq.dim() == 1
    assert cu_seqlens_q_per_seq.dim() == 1
    assert state_slot_per_seq.dim() == 1
    assert block_tables.dim() == 2
    for idp in (extend_indptr, prefix_swa_indptr, prefix_csa_indptr, prefix_hca_indptr):
        assert idp.dim() == 1 and idp.shape[0] >= T + 1
    for idx in (
        extend_indices,
        prefix_swa_indices,
        prefix_csa_indices,
        prefix_hca_indices,
    ):
        assert idx.dim() == 1

    # DENSE is the one class a V4 config can turn out not to have: a layer that
    # carries its window in a state field leaves the row space entirely, and on
    # a trunk that is all CSA and HCA the draft layer is the only ratio-0 one
    # there was. `prefix_swa_indices` then has no reader, so `HAS_DENSE` skips
    # it and the borrowed parameters below never reach a store. CSA and HCA
    # have no such exit today; the assert is there so the day one appears it
    # says so instead of raising a bare KeyError out of the geometry.
    served = served_window_params(geometry)
    assert CSA_RATIO in served and HCA_RATIO in served, (
        "V4 paged prefill writes the CSA and HCA prefix buffers unconditionally; "
        f"this pool serves only {sorted(served)}"
    )
    has_dense = DENSE_RATIO in served
    csa = served[CSA_RATIO]
    hca = served[HCA_RATIO]
    dense = served.get(DENSE_RATIO, csa)
    BLOCK_N = triton.next_power_of_2(win)
    _v4_paged_prefill_indices_kernel[(T,)](
        positions,
        bid_per_token,
        chunk_start_per_seq,
        cu_seqlens_q_per_seq,
        state_slot_per_seq,
        block_tables,
        block_tables.stride(0),
        extend_indptr,
        prefix_swa_indptr,
        prefix_csa_indptr,
        prefix_hca_indptr,
        extend_indices,
        prefix_swa_indices,
        prefix_csa_indices,
        prefix_hca_indices,
        win=win,
        dense_ring_start=dense.ring_start,
        csa_ring_start=csa.ring_start,
        hca_ring_start=hca.ring_start,
        HCA_RATIO=hca_ratio,
        HCA_ROWS_PER_BLOCK=hca_rows_per_block,
        ENVELOPE_ROWS=geometry.envelope_rows,
        BLOCK_N=BLOCK_N,
        HAS_DENSE=has_dense,
        **window_constexprs(dense, "DENSE_"),
        **window_constexprs(csa, "CSA_"),
        **window_constexprs(hca, "HCA_"),
    )


def write_v4_paged_prefill_indices_reference(
    *,
    positions: torch.Tensor,
    bid_per_token: torch.Tensor,
    chunk_start_per_seq: torch.Tensor,
    cu_seqlens_q_per_seq: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    extend_indptr: torch.Tensor,
    prefix_swa_indptr: torch.Tensor,
    prefix_csa_indptr: torch.Tensor,
    prefix_hca_indptr: torch.Tensor,
    extend_indices: torch.Tensor,
    prefix_swa_indices: torch.Tensor,
    prefix_csa_indices: torch.Tensor,
    prefix_hca_indices: torch.Tensor,
    T: int,
    win: int,
    geometry: UnifiedPoolGeometry,
    hca_ratio: int = 128,
    hca_rows_per_block: int = 1,
) -> None:
    """Pure-Python equivalent of ``write_v4_paged_prefill_indices``.
    Per-token Python loop — slow but readable; used for unit-test bit-exact
    verification against the Triton kernel and dump-bisect debugging.

    Same caller contract: the SWA prefix is written to the CSA slice TAIL and
    the CSA topk head is filled per layer by ``csa_translate_pack`` — together
    they cover the whole slice, so no ``-1`` pre-fill is needed.
    """
    if T == 0:
        return
    served = served_window_params(geometry)
    dense = served.get(DENSE_RATIO)
    csa = served[CSA_RATIO]
    hca = served[HCA_RATIO]
    bid_cpu = bid_per_token[:T].cpu().tolist()
    pos_cpu = positions[:T].cpu().tolist()
    cs_per_seq_cpu = chunk_start_per_seq.cpu().tolist()
    cu_q_cpu = cu_seqlens_q_per_seq.cpu().tolist()
    block_tables_cpu = block_tables.cpu()
    ext_indptr_cpu = extend_indptr.cpu().tolist()
    swa_indptr_cpu = prefix_swa_indptr.cpu().tolist()
    csa_indptr_cpu = prefix_csa_indptr.cpu().tolist()
    hca_indptr_cpu = prefix_hca_indptr.cpu().tolist()
    device = extend_indices.device

    for t in range(T):
        bid = bid_cpu[t]
        pos = pos_cpu[t]
        chunk_start = cs_per_seq_cpu[bid]
        cu_q = cu_q_cpu[bid]
        # Per-token causal HCA visibility (mirrors kernel + reference
        # get_compress_topk_idxs).
        n_hca = (pos + 1) // hca_ratio

        token_pos_in_chunk = pos - chunk_start
        swa_low = max(pos - win + 1, 0)
        extend_count = min(token_pos_in_chunk + 1, win)
        prefix_swa_count = max(chunk_start - swa_low, 0)

        # Extend
        ext_base = ext_indptr_cpu[t]
        ext_start_row = cu_q + token_pos_in_chunk - extend_count + 1
        ext_rows = torch.arange(
            ext_start_row,
            ext_start_row + extend_count,
            device=device,
            dtype=extend_indices.dtype,
        )
        extend_indices[ext_base : ext_base + extend_count] = ext_rows

        # SWA prefix (written to swa / csa / hca prefix buffers)
        sb_swa = swa_indptr_cpu[t]
        sb_hca = hca_indptr_cpu[t]
        if prefix_swa_count > 0:
            global_pos = range(swa_low, swa_low + prefix_swa_count)
            slot = int(state_slot_per_seq[bid])

            def rows(params, buf, positions=global_pos, s=slot):
                return torch.tensor(
                    [params.index(s, p) for p in positions],
                    dtype=buf.dtype,
                    device=device,
                )

            # No dense layer means no reader for this buffer — the kernel
            # leaves it alone too, on `HAS_DENSE`.
            if dense is not None:
                prefix_swa_indices[sb_swa : sb_swa + prefix_swa_count] = rows(
                    dense, prefix_swa_indices
                )
            # CSA: SWA prefix at the slice TAIL (head holds the CSA topk section
            # filled by csa_translate_pack). See the kernel comment above.
            csa_end = csa_indptr_cpu[t + 1]
            prefix_csa_indices[csa_end - prefix_swa_count : csa_end] = rows(
                csa, prefix_csa_indices
            )
            prefix_hca_indices[sb_hca : sb_hca + prefix_swa_count] = rows(
                hca, prefix_hca_indices
            )

        # HCA compress: entry k lives in physical block
        # block_tables[bid, k // hca_rows_per_block] at row
        # k % hca_rows_per_block.
        if n_hca > 0:
            ks = torch.arange(n_hca, device=device)
            blk = (ks // hca_rows_per_block).cpu()
            row = (ks % hca_rows_per_block).to(prefix_hca_indices.dtype)
            bt = block_tables_cpu[bid, blk].to(device).to(prefix_hca_indices.dtype)
            hca_dst = sb_hca + prefix_swa_count
            prefix_hca_indices[hca_dst : hca_dst + n_hca] = (
                bt * geometry.envelope_rows + row
            )
