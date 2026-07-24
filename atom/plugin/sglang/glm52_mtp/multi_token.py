"""Multi-token MTP metadata shared by target_verify and draft_extend."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from atom.plugin.sglang.glm52_mtp.common import (
    attention_page_size,
    build_token_table,
    compute_mtp_sparse_per_token_kv_lens,
    counts_to_indptr,
    ensure_shared_sparse_buffer,
    flatten_kv_indices,
    get_index_topk,
    is_draft_extend_mode,
    local_num_attention_heads,
    make_mla_work_buffers,
    make_sparse_mtp_work_buffers,
    metadata_dtype,
    validate_page_size,
)


logger = logging.getLogger("atom")


def build_mtp_multi_token_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
    draft_token_num: int,
    resolve_lens_fn,
):
    """Build decode-style metadata for multi-token MTP phases (verify / draft_extend)."""
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    bs = int(forward_batch.batch_size)
    if draft_token_num <= 0:
        raise RuntimeError(
            "GLM-5.2 DSA multi-token decode metadata requires draft_token_num > 0"
        )

    max_seqlen_q = draft_token_num
    prefix_lens_np, context_lens_np = resolve_lens_fn(
        forward_batch, positions, bs, draft_token_num
    )
    extend_lens = np.full(bs, draft_token_num, dtype=np.int32)
    sum_scheduled_tokens = bs * max_seqlen_q

    q_np = np.zeros(bs + 1, dtype=np.int32)
    q_np[1:] = np.cumsum(extend_lens, dtype=np.int32)
    cu_q = torch.from_numpy(q_np).to(device=device)

    topk = get_index_topk(atom_config)
    page_size = validate_page_size(token_to_kv_pool, atom_config)
    block_tables = build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=context_lens_np,
        extend_lens=extend_lens,
        page_size=page_size,
    )
    token_table = build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=context_lens_np,
        extend_lens=extend_lens,
        page_size=1,
    )
    kv_indptr = counts_to_indptr(context_lens_np, device)
    kv_indices = flatten_kv_indices(token_table, context_lens_np)
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    slot_mapping = forward_batch.out_cache_loc[:sum_scheduled_tokens]
    context_lens = torch.from_numpy(context_lens_np).to(device=device, dtype=torch.int32)

    draft_extend = is_draft_extend_mode(forward_batch)
    per_token_kv_lens = compute_mtp_sparse_per_token_kv_lens(
        prefix_lens_np=prefix_lens_np,
        context_lens_np=context_lens_np,
        max_seqlen_q=max_seqlen_q,
        bs=bs,
        draft_extend=draft_extend,
    )
    sparse_per_token_lens = np.clip(per_token_kv_lens, 0, topk).astype(np.int32)
    sparse_kv_indptr = counts_to_indptr(sparse_per_token_lens, device)
    sparse_cu = torch.arange(sum_scheduled_tokens + 1, dtype=torch.int32, device=device)
    sparse_kv_last_page_lens = torch.ones(
        sum_scheduled_tokens, dtype=torch.int32, device=device
    )
    token_to_seq_idxs = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32, device=device),
        torch.from_numpy(extend_lens.astype(np.int64)).to(device=device),
    )

    ensure_shared_sparse_buffer(
        token_to_kv_pool,
        num_tokens=sum_scheduled_tokens,
        topk=topk,
        device=device,
    )
    dtype_q = metadata_dtype(atom_config)
    attn_page_size = attention_page_size(token_to_kv_pool)
    num_heads = local_num_attention_heads(atom_config)
    max_seqlen_k = int(
        max(
            int(context_lens_np.max(initial=1)),
            int(per_token_kv_lens.max(initial=1)),
        )
    )

    work = make_mla_work_buffers(
        cu_seqlens_q=cu_q,
        kv_indptr=kv_indptr,
        kv_last_page_lens=kv_last_page_lens,
        num_heads=num_heads,
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        page_size=attn_page_size,
    )
    sparse_mtp_work = make_sparse_mtp_work_buffers(
        sparse_cu_seqlens_q=sparse_cu,
        sparse_kv_indptr=sparse_kv_indptr,
        sparse_kv_last_page_lens=sparse_kv_last_page_lens,
        num_heads=num_heads,
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        page_size=attn_page_size,
    )

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        state=AttnState.DECODE,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sparse_kv_indptr=sparse_kv_indptr,
        sparse_cu_seqlens_q=sparse_cu,
        token_to_seq_idxs=token_to_seq_idxs,
        **work,
    )
    md.dtype_q = dtype_q
    md.sparse_kv_last_page_lens = sparse_kv_last_page_lens
    for key, value in sparse_mtp_work.items():
        setattr(md, key, value)

    if draft_extend and os.environ.get("ATOM_GLM52_MTP_DUMP", "0") in (
        "1",
        "true",
        "True",
    ):
        forward_batch._atom_glm52_draft_extend_dump_metadata = md

    if draft_extend and os.environ.get("ATOM_GLM52_MTP_DEBUG", "0") in (
        "1",
        "true",
        "True",
    ):
        req_pool_indices = forward_batch.req_pool_indices[:bs]
        raw = req_to_token_pool.req_to_token[req_pool_indices]
        row = 0
        prefix = int(prefix_lens_np[row])
        context = int(context_lens_np[row])
        token_suffix = token_table[row, prefix:context]
        raw_suffix = raw[row, prefix:context]
        slots = slot_mapping[:draft_token_num]
        sparse_lens = per_token_kv_lens[:draft_token_num]
        expected_sparse_lens = prefix + np.arange(
            1, draft_token_num + 1, dtype=np.int32
        )
        logger.info(
            "[GLM52_DRAFT_EXTEND_METADATA] req_idx=%s pool_ids=(kv=%s,req=%s) "
            "prefix=%s context=%s raw_suffix=%s table_suffix=%s slots=%s "
            "table_matches_raw=%s table_matches_slots=%s sparse_lens=%s "
            "sparse_matches_causal=%s",
            int(req_pool_indices[row].item()),
            id(token_to_kv_pool),
            id(req_to_token_pool),
            prefix,
            context,
            raw_suffix.detach().cpu().tolist(),
            token_suffix.detach().cpu().tolist(),
            slots.detach().cpu().tolist(),
            bool(torch.equal(token_suffix, raw_suffix)),
            bool(torch.equal(token_suffix, slots)),
            sparse_lens.tolist(),
            bool(np.array_equal(sparse_lens, expected_sparse_lens)),
        )
    return md
