# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from atom.utils import envs
from atom.utils.forward_context import AttentionMetaData

logger = logging.getLogger("atom")


def derive_prefill_lens_from_positions(
    positions,
    full_cu_seqlens_q,
    ub_slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (extend_lens, seq_lens) for a prefill ubatch.

    ``positions`` are the absolute positions for this ubatch's new tokens.
    ``seq_lens`` are the total visible sequence lengths for sparse metadata,
    preserving the absolute prefix length of a straddled request instead of
    treating the ubatch as a standalone prompt.
    """
    rs = ub_slice.request_slice
    ts = ub_slice.token_slice
    ub_num_reqs = rs.stop - rs.start
    positions_np = np.asarray(positions)
    full_cu = np.asarray(full_cu_seqlens_q)

    req_global_starts = full_cu[rs.start : rs.stop].astype(np.int64)
    req_global_ends = full_cu[rs.start + 1 : rs.stop + 1].astype(np.int64)
    clamped_starts = np.maximum(req_global_starts, ts.start)
    clamped_ends = np.minimum(req_global_ends, ts.stop)
    extend_lens = (clamped_ends - clamped_starts).astype(np.int32)

    ub_cu = np.zeros(ub_num_reqs + 1, dtype=np.int32)
    np.cumsum(extend_lens, dtype=np.int32, out=ub_cu[1:])
    start_positions = positions_np[ub_cu[:ub_num_reqs]].astype(np.int32)
    seq_lens = (start_positions + extend_lens).astype(np.int32)
    return extend_lens, seq_lens


@dataclass
class UBatchSlice:
    """Describes which portion of a batch belongs to a micro-batch."""

    request_slice: slice  # which requests belong to this micro-batch
    token_slice: slice  # which tokens belong to this micro-batch (for prefill)


def maybe_create_ubatch_slices(
    num_reqs: int,
    num_tokens: int,
    num_ubatches: int = 2,
    is_prefill: bool = False,
    num_scheduled_tokens: Optional[list[int]] = None,
    max_tokens_per_ubatch: Optional[int] = None,
) -> Optional[list[UBatchSlice]]:
    """Split a batch into N micro-batch slices.

    For decode: split by request count (uniform tokens per request).
    For prefill: token-balanced split so each ubatch has roughly equal
                 token count, respecting request boundaries.

    Returns None if the batch is too small to split or if the split
    would produce a ubatch exceeding max_tokens_per_ubatch.
    """
    if num_ubatches <= 1:
        return None

    # Token-midpoint prefill split can cut *inside* a request, so it works
    # even with a single request (bs=1) — only require that there are at
    # least `num_ubatches` tokens to go around. The request-boundary path
    # below still needs num_reqs >= num_ubatches.
    token_split = num_scheduled_tokens is not None and envs.ATOM_TBO_PREFILL_TOKEN_SPLIT
    if not token_split and num_reqs < num_ubatches:
        return None

    if num_scheduled_tokens is not None:
        # Prefill: token-balanced split
        if token_split:
            if num_tokens < num_ubatches:
                return None
            return _split_prefill_token_midpoint(
                num_reqs,
                num_scheduled_tokens,
                num_ubatches,
                max_tokens_per_ubatch,
            )
        return _split_prefill_balanced(
            num_reqs,
            num_scheduled_tokens,
            num_ubatches,
            max_tokens_per_ubatch,
        )

    # Decode: uniform split by request count
    slices = []
    reqs_per_ub = num_reqs // num_ubatches
    tokens_per_req = num_tokens // num_reqs if num_reqs > 0 else 1

    for i in range(num_ubatches):
        req_start = i * reqs_per_ub
        req_end = (i + 1) * reqs_per_ub if i < num_ubatches - 1 else num_reqs
        tok_start = req_start * tokens_per_req
        tok_end = req_end * tokens_per_req
        slices.append(
            UBatchSlice(
                request_slice=slice(req_start, req_end),
                token_slice=slice(tok_start, tok_end),
            )
        )
    return slices


def _split_prefill_balanced(
    num_reqs: int,
    num_scheduled_tokens: list[int],
    num_ubatches: int,
    max_tokens_per_ubatch: Optional[int],
) -> Optional[list[UBatchSlice]]:
    """Split prefill requests into ubatches balanced by token count.

    Finds the request boundary closest to total_tokens / num_ubatches,
    then validates that no ubatch exceeds max_tokens_per_ubatch.
    """
    total_tokens = sum(num_scheduled_tokens)
    target = total_tokens // num_ubatches

    # Find split point where cumulative tokens first reaches target
    cumsum = 0
    split_idx = 1  # at least 1 request in first ubatch
    for j in range(num_reqs):
        cumsum += num_scheduled_tokens[j]
        if cumsum >= target:
            # Pick the boundary (j or j+1) closer to the target
            prev = cumsum - num_scheduled_tokens[j]
            if j > 0 and (target - prev) < (cumsum - target):
                split_idx = j
            else:
                split_idx = j + 1
            break

    # Ensure at least 1 request per ubatch
    split_idx = max(1, min(split_idx, num_reqs - 1))

    tok_boundary = sum(num_scheduled_tokens[:split_idx])
    ub0_tokens = tok_boundary
    ub1_tokens = total_tokens - tok_boundary

    # Reject if either ubatch exceeds the AsyncLL buffer
    if max_tokens_per_ubatch is not None:
        if ub0_tokens > max_tokens_per_ubatch or ub1_tokens > max_tokens_per_ubatch:
            logger.info(
                f"[TBO] prefill split rejected: ubatch tokens "
                f"({ub0_tokens}, {ub1_tokens}) exceed buffer "
                f"{max_tokens_per_ubatch}"
            )
            return None

    return [
        UBatchSlice(
            request_slice=slice(0, split_idx),
            token_slice=slice(0, tok_boundary),
        ),
        UBatchSlice(
            request_slice=slice(split_idx, num_reqs),
            token_slice=slice(tok_boundary, total_tokens),
        ),
    ]


def _split_prefill_token_midpoint(
    num_reqs: int,
    num_scheduled_tokens: list[int],
    num_ubatches: int,
    max_tokens_per_ubatch: Optional[int],
) -> Optional[list[UBatchSlice]]:
    """split prefill at the exact token midpoint."""
    toks = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
    total_tokens = int(toks.sum())
    # Exclusive-prefix cumulative token offsets: cu[i] = first token of req i.
    cu = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(toks, out=cu[1:])

    # Token split points at exact fractions of the total.
    split_points = [(total_tokens * i) // num_ubatches for i in range(1, num_ubatches)]

    slices: list[UBatchSlice] = []
    tok_start = 0
    for tok_end in split_points + [total_tokens]:
        if tok_end <= tok_start:
            # Degenerate (e.g. total_tokens < num_ubatches) — bail out and
            # let the caller fall back to no-split.
            return None
        # Requests overlapping [tok_start, tok_end):
        #   first req = the one containing tok_start
        #   last req  = the one containing tok_end - 1
        req_start = int(np.searchsorted(cu, tok_start, side="right") - 1)
        req_stop = int(np.searchsorted(cu, tok_end - 1, side="right"))
        slices.append(
            UBatchSlice(
                request_slice=slice(req_start, req_stop),
                token_slice=slice(tok_start, tok_end),
            )
        )
        tok_start = tok_end

    if max_tokens_per_ubatch is not None:
        for s in slices:
            if (s.token_slice.stop - s.token_slice.start) > max_tokens_per_ubatch:
                logger.info(
                    "[TBO] token-midpoint split rejected: ubatch tokens "
                    "exceed buffer %s",
                    max_tokens_per_ubatch,
                )
                return None

    return slices


def split_attn_metadata(
    attn_metadata: AttentionMetaData,
    ub_slice: UBatchSlice,
    padded_bs: int,
) -> AttentionMetaData:
    """Split AttentionMetaData for a single micro-batch."""
    rs = ub_slice.request_slice
    ts = ub_slice.token_slice
    req_start = rs.start
    req_end = rs.stop
    ub_num_reqs = req_end - req_start

    ub_cu_seqlens_q = None
    if attn_metadata.cu_seqlens_q is not None:
        orig_cu = attn_metadata.cu_seqlens_q
        seg = orig_cu[req_start : req_end + 1]
        # clamp absolute token offsets to [ts.start, ts.stop], then re-base
        clamped = torch.clamp(seg, min=ts.start, max=ts.stop)
        ub_cu_seqlens_q = clamped - clamped[0]
        # Pad remaining entries up to padded_bs + 1
        if padded_bs > ub_num_reqs:
            last_val = ub_cu_seqlens_q[-1]
            pad_size = padded_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_cu_seqlens_q = torch.cat([ub_cu_seqlens_q, padding])

    # slot_mapping: slice by token
    ub_slot_mapping = None
    if attn_metadata.slot_mapping is not None:
        ub_slot_mapping = attn_metadata.slot_mapping[ts]
        # Pad with -1 for padded positions
        tok_count = ts.stop - ts.start
        # max_q_len = attn_metadata.max_seqlen_q
        # padded_tok_count = padded_bs * max_q_len
        padded_tok_count = padded_bs
        if padded_tok_count > tok_count:
            pad = torch.full(
                (padded_tok_count - tok_count,),
                -1,
                dtype=attn_metadata.slot_mapping.dtype,
                device=attn_metadata.slot_mapping.device,
            )
            ub_slot_mapping = torch.cat([ub_slot_mapping, pad])

    # context_lens: slice by request
    ub_context_lens = None
    if attn_metadata.context_lens is not None:
        ub_context_lens = attn_metadata.context_lens[rs]
        if padded_bs > ub_num_reqs:
            pad = torch.zeros(
                padded_bs - ub_num_reqs,
                dtype=attn_metadata.context_lens.dtype,
                device=attn_metadata.context_lens.device,
            )
            ub_context_lens = torch.cat([ub_context_lens, pad])

    # block_tables: slice by request
    ub_block_tables = None
    if attn_metadata.block_tables is not None:
        ub_block_tables = attn_metadata.block_tables[rs]
        if padded_bs > ub_num_reqs:
            pad = torch.zeros(
                padded_bs - ub_num_reqs,
                attn_metadata.block_tables.shape[1],
                dtype=attn_metadata.block_tables.dtype,
                device=attn_metadata.block_tables.device,
            )
            ub_block_tables = torch.cat([ub_block_tables, pad])

    # kv_indptr: re-base from 0
    ub_kv_indptr = None
    if attn_metadata.kv_indptr is not None:
        orig_indptr = attn_metadata.kv_indptr
        base = orig_indptr[req_start]
        ub_kv_indptr = orig_indptr[req_start : req_end + 1] - base
        if padded_bs > ub_num_reqs:
            last_val = ub_kv_indptr[-1]
            pad_size = padded_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_kv_indptr = torch.cat([ub_kv_indptr, padding])

    # kv_indices: shared (indexed via kv_indptr, no slicing needed)
    ub_kv_indices = attn_metadata.kv_indices

    # kv_last_page_lens: slice by request
    ub_kv_last_page_lens = None
    if attn_metadata.kv_last_page_lens is not None:
        ub_kv_last_page_lens = attn_metadata.kv_last_page_lens[rs]
        if padded_bs > ub_num_reqs:
            pad = torch.zeros(
                padded_bs - ub_num_reqs,
                dtype=attn_metadata.kv_last_page_lens.dtype,
                device=attn_metadata.kv_last_page_lens.device,
            )
            ub_kv_last_page_lens = torch.cat([ub_kv_last_page_lens, pad])

    # Prefix-cache prefill needs these fields to gather cached KV before
    # varlen attention. Dropping them makes cu_seqlens_k describe cached+new
    # tokens while K/V only contain new tokens, which can OOB in flash-attn.
    ub_num_cached_tokens = None
    if attn_metadata.num_cached_tokens is not None:
        ub_num_cached_tokens = attn_metadata.num_cached_tokens[rs]
        if padded_bs > ub_num_reqs:
            pad = torch.zeros(
                padded_bs - ub_num_reqs,
                dtype=attn_metadata.num_cached_tokens.dtype,
                device=attn_metadata.num_cached_tokens.device,
            )
            ub_num_cached_tokens = torch.cat([ub_num_cached_tokens, pad])

    ub_seq_starts = None
    if attn_metadata.seq_starts is not None:
        ub_seq_starts = attn_metadata.seq_starts[rs]
        if padded_bs > ub_num_reqs:
            pad = torch.zeros(
                padded_bs - ub_num_reqs,
                dtype=attn_metadata.seq_starts.dtype,
                device=attn_metadata.seq_starts.device,
            )
            ub_seq_starts = torch.cat([ub_seq_starts, pad])

    # sparse_kv_indptr: slice and re-base if present (per-request dimension).
    # NOTE: In MLA prefill sparse mode, sparse_kv_indptr is per-token — that
    # case is handled by the MLA builder's build_ubatch_prefill_metadata override.
    ub_sparse_kv_indptr = None
    if attn_metadata.sparse_kv_indptr is not None:
        orig = attn_metadata.sparse_kv_indptr
        base = orig[req_start]
        ub_sparse_kv_indptr = orig[req_start : req_end + 1] - base
        if padded_bs > ub_num_reqs:
            last_val = ub_sparse_kv_indptr[-1]
            pad_size = padded_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_sparse_kv_indptr = torch.cat([ub_sparse_kv_indptr, padding])

    # max_seqlen_k: recompute from the sliced context_lens
    ub_max_seqlen_k = attn_metadata.max_seqlen_k
    if ub_context_lens is not None and ub_num_reqs > 0:
        ub_max_seqlen_k = int(ub_context_lens[:ub_num_reqs].max().item())

    # cu_seqlens_k: re-base like cu_seqlens_q (needed for prefill attention)
    ub_cu_seqlens_k = None
    if attn_metadata.cu_seqlens_k is not None:
        orig_cu_k = attn_metadata.cu_seqlens_k
        seg_k = orig_cu_k[req_start : req_end + 1]
        if not getattr(attn_metadata, "has_cached", False):
            clamped_k = torch.clamp(seg_k, min=ts.start, max=ts.stop)
            ub_cu_seqlens_k = clamped_k - clamped_k[0]
        else:
            ub_cu_seqlens_k = seg_k - seg_k[0]
        if padded_bs > ub_num_reqs:
            last_val = ub_cu_seqlens_k[-1]
            pad_size = padded_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_cu_seqlens_k = torch.cat([ub_cu_seqlens_k, padding])

    # max_seqlen_q: recompute from cu_seqlens_q for this ubatch
    ub_max_seqlen_q = attn_metadata.max_seqlen_q
    if ub_cu_seqlens_q is not None and ub_num_reqs > 0:
        # Per-request q lengths are the diffs of consecutive cu_seqlens entries
        per_req_q = ub_cu_seqlens_q[1 : ub_num_reqs + 1] - ub_cu_seqlens_q[:ub_num_reqs]
        ub_max_seqlen_q = int(per_req_q.max().item())

    ub_total_kv = None
    if attn_metadata.has_cached:
        if ub_cu_seqlens_k is not None and ub_num_reqs > 0:
            ub_total_kv = int(ub_cu_seqlens_k[ub_num_reqs].item())
        elif ub_context_lens is not None and ub_num_reqs > 0:
            ub_total_kv = int(ub_context_lens[:ub_num_reqs].sum().item())
        else:
            ub_total_kv = 0

    # MLA work buffers are set to None here — they will be recomputed
    # by decode_update_mla_metadata_v1 in UBatchWrapper before each micro-batch run.
    # Backend-specific fields (e.g. MLA sparse prefill) are handled by
    # the builder's build_ubatch_prefill_metadata override.
    ub_attn = AttentionMetaData(
        cu_seqlens_q=ub_cu_seqlens_q,
        cu_seqlens_k=ub_cu_seqlens_k,
        max_seqlen_q=ub_max_seqlen_q,
        max_seqlen_k=ub_max_seqlen_k,
        min_seqlen_q=attn_metadata.min_seqlen_q,
        slot_mapping=ub_slot_mapping,
        context_lens=ub_context_lens,
        block_tables=ub_block_tables,
        dropout_p=attn_metadata.dropout_p,
        kv_indptr=ub_kv_indptr,
        kv_indices=ub_kv_indices,
        kv_last_page_lens=ub_kv_last_page_lens,
        sparse_kv_indptr=ub_sparse_kv_indptr,
        has_cached=attn_metadata.has_cached,
        total_kv=ub_total_kv,
        num_cached_tokens=ub_num_cached_tokens,
        seq_starts=ub_seq_starts,
        work_meta_data=None,
        work_indptr=None,
        work_info_set=None,
        reduce_indptr=None,
        reduce_final_map=None,
        reduce_partial_map=None,
    )
    # Carry over dtype_q if set
    if hasattr(attn_metadata, "dtype_q"):
        ub_attn.dtype_q = attn_metadata.dtype_q
    return ub_attn
