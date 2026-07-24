"""Prefill metadata — target prefill and draft_extend_for_prefill path."""

from __future__ import annotations

import numpy as np
import torch

from atom.plugin.sglang.glm52_mtp.common import (
    attention_page_size,
    build_token_table,
    counts_to_indptr,
    ensure_shared_sparse_buffer,
    flatten_kv_indices,
    get_extend_lens_cpu,
    get_index_topk,
    get_seq_lens_cpu,
    is_draft_extend_mode,
    local_num_attention_heads,
    make_mla_work_buffers,
    metadata_dtype,
    validate_page_size,
)


def is_draft_extend_prefill(forward_batch) -> bool:
    """Eagle ``forward_draft_extend`` after target prefill (EXTEND on draft pool only)."""
    if is_draft_extend_mode(forward_batch):
        return False
    if forward_batch.forward_mode.is_decode_or_idle():
        return False
    if getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)():
        return False
    mode = forward_batch.forward_mode
    if not bool(getattr(mode, "is_extend", lambda: False)()):
        return False
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is None:
        return False
    hidden = getattr(spec_info, "hidden_states", None)
    if not torch.is_tensor(hidden) or int(hidden.shape[0]) <= 0:
        return False
    bonus = getattr(spec_info, "bonus_tokens", None)
    if not torch.is_tensor(bonus):
        return False
    num_tokens_per_req = getattr(spec_info, "num_tokens_per_req", None)
    if num_tokens_per_req is not None and int(num_tokens_per_req) != 1:
        return False
    return True


def build_mtp_draft_extend_prefill_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    """Draft-pool prefill metadata for ``forward_draft_extend`` (native propose i0 semantics).

    Target prefill keeps ``build_prefill_metadata`` unchanged.
    """
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    bs = int(forward_batch.batch_size)
    seq_lens = get_seq_lens_cpu(forward_batch, bs)
    extend_lens = get_extend_lens_cpu(forward_batch, positions, bs)
    topk = get_index_topk(atom_config)
    page_size = validate_page_size(token_to_kv_pool, atom_config)

    cached_lens = np.maximum(seq_lens - extend_lens, 0).astype(np.int32)
    seq_lens = np.maximum(seq_lens, cached_lens + extend_lens).astype(np.int32)
    has_cached = bool(np.any(cached_lens > 0))

    q_np = np.zeros(bs + 1, dtype=np.int32)
    q_np[1:] = np.cumsum(extend_lens, dtype=np.int32)
    cu_q = torch.from_numpy(q_np).to(device=device)
    kv_indptr = counts_to_indptr(seq_lens, device)
    block_tables = build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=page_size,
    )
    token_table = build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=1,
    )
    kv_indices = flatten_kv_indices(token_table, seq_lens)
    state = AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE
    total_tokens = int(extend_lens.sum())
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr if has_cached else cu_q,
        max_seqlen_q=int(extend_lens.max(initial=1)),
        max_seqlen_k=int(seq_lens.max(initial=1)),
        slot_mapping=forward_batch.out_cache_loc[:total_tokens],
        context_lens=forward_batch.seq_lens[:bs].to(dtype=torch.int32),
        block_tables=block_tables,
        state=state,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        has_cached=has_cached,
        total_kv=int(seq_lens.sum()),
        num_cached_tokens=torch.from_numpy(cached_lens).to(device=device),
        seq_starts=torch.from_numpy(cached_lens).to(device=device),
    )
    dtype_q = metadata_dtype(atom_config)
    md.dtype_q = dtype_q

    if md.max_seqlen_k > topk:
        counts = extend_lens.astype(np.int32)
        local_offsets = np.concatenate(
            [np.arange(int(count), dtype=np.int32) for count in counts]
        )
        if has_cached:
            seq_starts = kv_indptr[:-1].detach().cpu().numpy().astype(np.int32)
            repeated_seq_starts = np.repeat(seq_starts, counts)
            repeated_cached_lens = np.repeat(cached_lens, counts)
            cu_ks = repeated_seq_starts
            cu_ke = repeated_seq_starts + repeated_cached_lens + local_offsets + 1
            sparse_counts = repeated_cached_lens + local_offsets + 1
        else:
            cu_ks = np.repeat(q_np[:bs], counts)
            cu_ke = np.arange(total_tokens, dtype=np.int32) + 1
            sparse_counts = local_offsets + 1

        sparse_cu = torch.arange(total_tokens + 1, dtype=torch.int32, device=device)
        sparse_kv_indptr = counts_to_indptr(
            np.minimum(sparse_counts, topk).astype(np.int32), device
        )
        sparse_last_page_lens = torch.ones(
            total_tokens, dtype=torch.int32, device=device
        )
        md.cu_seqlen_ks = torch.from_numpy(cu_ks.astype(np.int32)).to(device=device)
        md.cu_seqlen_ke = torch.from_numpy(cu_ke.astype(np.int32)).to(device=device)
        md.sparse_cu_seqlens_q = sparse_cu
        md.sparse_kv_indptr = sparse_kv_indptr
        md.sparse_kv_last_page_lens = sparse_last_page_lens
        md.token_to_seq_idxs = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            torch.from_numpy(counts.astype(np.int64)).to(device=device),
        )
        ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=total_tokens,
            topk=topk,
            device=device,
        )
        sparse_work = make_mla_work_buffers(
            cu_seqlens_q=sparse_cu,
            kv_indptr=sparse_kv_indptr,
            kv_last_page_lens=sparse_last_page_lens,
            num_heads=local_num_attention_heads(atom_config),
            dtype_q=dtype_q,
            dtype_kv=dtype_q,
            page_size=attention_page_size(token_to_kv_pool),
        )
        for key, value in sparse_work.items():
            setattr(md, f"sparse_prefill_{key}", value)
    else:
        ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=max(1, total_tokens),
            topk=topk,
            device=device,
        )

    return md


def build_prefill_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    bs = int(forward_batch.batch_size)
    seq_lens = get_seq_lens_cpu(forward_batch, bs)
    extend_lens = get_extend_lens_cpu(forward_batch, positions, bs)
    topk = get_index_topk(atom_config)
    page_size = validate_page_size(token_to_kv_pool, atom_config)

    q_np = np.zeros(bs + 1, dtype=np.int32)
    q_np[1:] = np.cumsum(extend_lens, dtype=np.int32)
    cu_q = torch.from_numpy(q_np).to(device=device)
    kv_indptr = counts_to_indptr(seq_lens, device)
    block_tables = build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=page_size,
    )
    token_table = build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=1,
    )
    kv_indices = flatten_kv_indices(token_table, seq_lens)
    has_cached = bool(np.any(seq_lens - extend_lens > 0))
    state = AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE
    total_tokens = int(extend_lens.sum())
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr if has_cached else cu_q,
        max_seqlen_q=int(extend_lens.max(initial=1)),
        max_seqlen_k=int(seq_lens.max(initial=1)),
        slot_mapping=forward_batch.out_cache_loc[:total_tokens],
        context_lens=forward_batch.seq_lens[:bs].to(dtype=torch.int32),
        block_tables=block_tables,
        state=state,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        has_cached=has_cached,
        total_kv=int(seq_lens.sum()),
        num_cached_tokens=torch.from_numpy(seq_lens - extend_lens).to(device=device),
        seq_starts=torch.from_numpy(seq_lens - extend_lens).to(device=device),
    )
    dtype_q = metadata_dtype(atom_config)
    md.dtype_q = dtype_q

    if md.max_seqlen_k > topk:
        counts = extend_lens.astype(np.int32)
        local_offsets = np.concatenate(
            [np.arange(int(count), dtype=np.int32) for count in counts]
        )
        if has_cached:
            seq_starts = kv_indptr[:-1].detach().cpu().numpy().astype(np.int32)
            cached_lens = seq_lens - counts
            repeated_seq_starts = np.repeat(seq_starts, counts)
            repeated_cached_lens = np.repeat(cached_lens, counts)
            cu_ks = repeated_seq_starts
            cu_ke = repeated_seq_starts + repeated_cached_lens + local_offsets + 1
            sparse_counts = repeated_cached_lens + local_offsets + 1
        else:
            cu_ks = np.repeat(q_np[:bs], counts)
            cu_ke = np.arange(total_tokens, dtype=np.int32) + 1
            sparse_counts = local_offsets + 1

        sparse_cu = torch.arange(total_tokens + 1, dtype=torch.int32, device=device)
        sparse_kv_indptr = counts_to_indptr(
            np.minimum(sparse_counts, topk).astype(np.int32), device
        )
        sparse_last_page_lens = torch.ones(
            total_tokens, dtype=torch.int32, device=device
        )
        md.cu_seqlen_ks = torch.from_numpy(cu_ks.astype(np.int32)).to(device=device)
        md.cu_seqlen_ke = torch.from_numpy(cu_ke.astype(np.int32)).to(device=device)
        md.sparse_cu_seqlens_q = sparse_cu
        md.sparse_kv_indptr = sparse_kv_indptr
        md.sparse_kv_last_page_lens = sparse_last_page_lens
        md.token_to_seq_idxs = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            torch.from_numpy(counts.astype(np.int64)).to(device=device),
        )
        ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=total_tokens,
            topk=topk,
            device=device,
        )
        sparse_work = make_mla_work_buffers(
            cu_seqlens_q=sparse_cu,
            kv_indptr=sparse_kv_indptr,
            kv_last_page_lens=sparse_last_page_lens,
            num_heads=local_num_attention_heads(atom_config),
            dtype_q=dtype_q,
            dtype_kv=dtype_q,
            page_size=attention_page_size(token_to_kv_pool),
        )
        for key, value in sparse_work.items():
            setattr(md, f"sparse_prefill_{key}", value)
    else:
        ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=max(1, total_tokens),
            topk=topk,
            device=device,
        )

    return md
