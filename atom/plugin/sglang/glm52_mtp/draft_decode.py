"""Draft forward decode metadata — draft pool, 1 tok/query, multi-step sub_step."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger("atom.plugin.sglang.glm52_mtp.draft_decode")

from atom.plugin.sglang.glm52_mtp.common import (
    DRAFT_SUB_STEP_ATTR,
    attention_page_size,
    build_draft_decode_token_table,
    build_token_table,
    compute_mtp_sparse_per_token_kv_lens,
    counts_to_indptr,
    ensure_shared_sparse_buffer,
    flatten_kv_indices,
    get_index_topk,
    get_seq_lens_cpu,
    local_num_attention_heads,
    make_mla_work_buffers,
    make_sparse_mtp_work_buffers,
    metadata_dtype,
    resolve_draft_decode_context_lens,
    resolve_speculative_num_steps,
    validate_page_size,
)


def get_draft_decode_sub_step(forward_batch) -> int:
    return int(getattr(forward_batch, DRAFT_SUB_STEP_ATTR, 0) or 0)


def set_draft_decode_sub_step(forward_batch, sub_step: int) -> None:
    setattr(forward_batch, DRAFT_SUB_STEP_ATTR, int(sub_step))


def clear_draft_decode_sub_step(forward_batch) -> None:
    if hasattr(forward_batch, DRAFT_SUB_STEP_ATTR):
        delattr(forward_batch, DRAFT_SUB_STEP_ATTR)


def is_draft_decode_metadata(forward_batch) -> bool:
    return hasattr(forward_batch, DRAFT_SUB_STEP_ATTR)


def resolve_committed_lens_cpu(forward_batch, bs: int) -> np.ndarray:
    """Post-verify committed length (pre_verify + accept), from scheduler state."""
    spec_info = getattr(forward_batch, "spec_info", None)
    new_seq_lens = (
        getattr(spec_info, "new_seq_lens", None) if spec_info is not None else None
    )
    if torch.is_tensor(new_seq_lens) and int(new_seq_lens.numel()) >= bs:
        return new_seq_lens[:bs].detach().cpu().numpy().astype(np.int32)
    return get_seq_lens_cpu(forward_batch, bs)


def resolve_draft_decode_lens(
    forward_batch,
    positions: torch.Tensor | None,
    bs: int,
    sub_step: int,
) -> np.ndarray:
    """Effective dense/sparse KV lengths for EAGLE draft decode sub-steps."""
    return resolve_draft_decode_context_lens(forward_batch, positions, bs, sub_step)


def resolve_spec_decode_kv(
    forward_batch,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Reuse SGLang EAGLE spec_info KV routing when the draft backend pre-built it."""
    if is_draft_decode_metadata(forward_batch):
        return None
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is None:
        return None
    kv_indptr = getattr(spec_info, "kv_indptr", None)
    kv_indices = getattr(spec_info, "kv_indices", None)
    if not torch.is_tensor(kv_indptr) or not torch.is_tensor(kv_indices):
        return None
    bs = int(kv_indptr.shape[0]) - 1
    if bs <= 0:
        return None
    return kv_indptr[: bs + 1], kv_indices, bs


def _build_mtp_draft_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
    sub_step: int,
):
    """Eager draft_forward decode metadata (Implementation B rebuild)."""
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    # Authoritative batch width from SGLang's compacted ScheduleBatch. Do NOT
    # infer a defensive minimum: SGLang's filter_batch keeps seq_lens /
    # req_pool_indices / spec_info consistent with batch_size, so any mismatch
    # is a real lifecycle bug that must surface, not be masked.
    bs = int(forward_batch.batch_size)
    topk = get_index_topk(atom_config)
    page_size = validate_page_size(token_to_kv_pool, atom_config)
    attn_page_size = attention_page_size(token_to_kv_pool)
    num_heads = local_num_attention_heads(atom_config)

    context_lens_np = resolve_draft_decode_lens(forward_batch, positions, bs, sub_step)
    committed_np = get_seq_lens_cpu(forward_batch, bs)
    num_steps = resolve_speculative_num_steps(forward_batch, default=max(sub_step + 1, 1))

    token_table = build_draft_decode_token_table(
        forward_batch,
        req_to_token_pool,
        bs=bs,
        seq_lens_np=committed_np,
        sub_step=sub_step,
        num_steps=num_steps,
        topk=1,
        page_size=1,
    )
    block_tables = build_draft_decode_token_table(
        forward_batch,
        req_to_token_pool,
        bs=bs,
        seq_lens_np=committed_np,
        sub_step=sub_step,
        num_steps=num_steps,
        topk=1,
        page_size=page_size,
    )
    kv_indptr = counts_to_indptr(context_lens_np, device)
    kv_indices = flatten_kv_indices(token_table, context_lens_np)
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    context_lens = torch.from_numpy(context_lens_np).to(device=device, dtype=torch.int32)

    # For single-token decode, sparse KV count = context_lens (not context+1).
    # compute_mtp_sparse_per_token_kv_lens adds +1 for causal extend semantics,
    # but decode's indexer scores exactly context_lens entries, not context+1.
    # Using context+1 causes the kernel to read 1 stale entry from the indexer
    # buffer → garbage attention → accumulated prediction error.
    sparse_per_token_lens = np.clip(context_lens_np, 0, topk).astype(np.int32)
    sparse_kv_indptr = counts_to_indptr(sparse_per_token_lens, device)
    sparse_cu = torch.arange(bs + 1, dtype=torch.int32, device=device)
    sparse_kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    token_to_seq_idxs = torch.arange(bs, dtype=torch.int32, device=device)

    ensure_shared_sparse_buffer(
        token_to_kv_pool,
        num_tokens=bs,
        topk=topk,
        device=device,
    )

    cu_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
    dtype_q = metadata_dtype(atom_config)
    max_seqlen_k = int(
        max(
            int(context_lens_np.max(initial=1)),
            int(sparse_per_token_lens.max(initial=1)),
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

    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_cache_loc):
        slot_mapping = out_cache_loc[:bs].to(dtype=torch.int32)
    else:
        slot_mapping = torch.zeros(bs, dtype=torch.int32, device=device)

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr,
        max_seqlen_q=1,
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

    return md


def build_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    """Rebuild sparse draft-decode metadata from forward_batch."""
    from atom.utils.forward_context import AttentionMetaData, AttnState

    sub_step = get_draft_decode_sub_step(forward_batch)
    if is_draft_decode_metadata(forward_batch):
        return _build_mtp_draft_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
            sub_step=sub_step,
        )

    device = forward_batch.seq_lens.device
    bs = int(forward_batch.batch_size)
    topk = get_index_topk(atom_config)
    page_size = validate_page_size(token_to_kv_pool, atom_config)

    spec_kv = resolve_spec_decode_kv(forward_batch)
    if spec_kv is not None:
        kv_indptr, kv_indices, bs = spec_kv
        seq_lens = (
            (kv_indptr[1:] - kv_indptr[:-1]).detach().cpu().numpy().astype(np.int32)
        )
        block_tables = build_token_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=None,
            page_size=page_size,
        )
    else:
        seq_lens = get_seq_lens_cpu(forward_batch, bs)
        block_tables = build_token_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=None,
            page_size=page_size,
        )
        token_table = build_token_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=None,
            page_size=1,
        )
        kv_indptr = counts_to_indptr(seq_lens, device)
        kv_indices = flatten_kv_indices(token_table, seq_lens)

    cu_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    sparse_kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    sparse_counts = np.minimum(seq_lens, topk).astype(np.int32)
    sparse_kv_indptr = counts_to_indptr(sparse_counts, device)
    context_lens = torch.from_numpy(seq_lens).to(device=device, dtype=torch.int32)

    ensure_shared_sparse_buffer(
        token_to_kv_pool,
        num_tokens=bs,
        topk=topk,
        device=device,
    )
    dtype_q = metadata_dtype(atom_config)
    work = make_mla_work_buffers(
        cu_seqlens_q=cu_q,
        kv_indptr=sparse_kv_indptr,
        kv_last_page_lens=kv_last_page_lens,
        num_heads=local_num_attention_heads(atom_config),
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        page_size=attention_page_size(token_to_kv_pool),
    )

    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_cache_loc):
        slot_mapping = out_cache_loc[: int(out_cache_loc.numel())]
    else:
        slot_mapping = torch.zeros(bs, dtype=torch.int32, device=device)
    slot_mapping = slot_mapping.to(dtype=torch.int32)

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr,
        max_seqlen_q=1,
        max_seqlen_k=int(seq_lens.max(initial=1)),
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        state=AttnState.DECODE,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sparse_kv_indptr=sparse_kv_indptr,
        **work,
    )
    md.dtype_q = dtype_q
    md.sparse_kv_last_page_lens = sparse_kv_last_page_lens
    return md
