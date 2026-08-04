"""GLM-5.2 DSA and MTP integration for the SGLang plugin.

This module owns GLM-5.2 metadata construction, KV cache binding, and decode CUDA Graph support.
"""

from __future__ import annotations

# --- common.py ---
"""Shared GLM-5.2 DSA bridge helpers (all MTP phases)."""

import logging
import os

import numpy as np
import torch
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

from atom.plugin.sglang.runtime.attention_backend_resolver import (
    resolve_sglang_runtime,
)
from atom.plugin.sglang.runtime.model_arch import is_glm52_dsa_config

DECODE_GRAPH_BUFFERS_ATTR = "_atom_glm52_decode_graph_buffers"
EMPTY_VALUE_CACHE_ATTR = "_atom_glm52_empty_value_cache"
INDEXER_PAGE_SIZE_ATTR = "_atom_glm52_indexer_page_size"
ATTENTION_PAGE_SIZE_ATTR = "_atom_glm52_attention_page_size"
SHARED_SPARSE_INDICES_ATTR = "_atom_glm52_shared_sparse_kv_indices"
DRAFT_SUB_STEP_ATTR = "_atom_glm52_draft_decode_sub_step"
INDEXER_CONTEXT_LENS_ATTR = "_atom_glm52_indexer_context_lens"
GLM52_GRAPH_SEQ_LEN_CAPACITY = 10240


def is_glm52_dsa_arch(config) -> bool:
    return is_glm52_dsa_config(config)


def maybe_get_glm52_dsa_pools_from_sglang_backend(forward_batch=None):
    runtime = resolve_sglang_runtime(forward_batch)
    return runtime.token_to_kv_pool, runtime.req_to_token_pool


def get_seq_lens_cpu(forward_batch, bs: int) -> np.ndarray:
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu()
    if torch.is_tensor(seq_lens_cpu):
        seq_lens_cpu = seq_lens_cpu.detach().cpu().numpy()
    return np.asarray(seq_lens_cpu[:bs], dtype=np.int32)


def get_extend_prefix_lens_cpu(forward_batch, bs: int) -> np.ndarray | None:
    """Committed prefix length before the current draft_extend suffix."""
    prefix = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix is not None:
        if isinstance(prefix, list):
            return np.asarray(prefix[:bs], dtype=np.int32)
        if torch.is_tensor(prefix):
            return prefix[:bs].detach().cpu().numpy().astype(np.int32)
    prefix = getattr(forward_batch, "extend_prefix_lens", None)
    if prefix is None:
        return None
    if torch.is_tensor(prefix):
        return prefix[:bs].detach().cpu().numpy().astype(np.int32)
    return np.asarray(prefix[:bs], dtype=np.int32)


def resolve_draft_decode_context_lens(
    forward_batch,
    positions: torch.Tensor | None,
    bs: int,
    sub_step: int,
) -> np.ndarray:
    """Logical KV/context length for draft_forward sub-steps (SGLang layout).

    Uses committed ``seq_lens`` + ``sub_step + 1``, matching
    ``generate_draft_decode_kv_indices`` (history len + draft slots written).
    """
    # Authoritative batch width from SGLang's compacted ScheduleBatch. The caller
    # already passes forward_batch.batch_size; only clamp non-positive values.
    if bs <= 0:
        bs = int(forward_batch.batch_size)

    committed = get_seq_lens_cpu(forward_batch, bs)
    effective = (committed + int(sub_step) + 1).astype(np.int32)

    if torch.is_tensor(positions) and int(positions.numel()) > 0:
        pos_rows = positions.detach().cpu().numpy().astype(np.int32).reshape(-1)
        spec_info = getattr(forward_batch, "spec_info", None)
        topk = int(getattr(spec_info, "num_tokens_per_req", 0) or 0)
        if topk <= 0:
            topk = 1
        for row in range(int(committed.size)):
            idx = row * topk if pos_rows.size >= bs * topk else row
            if idx < pos_rows.size:
                effective[row] = max(int(effective[row]), int(pos_rows[idx]) + 1)
    return effective


def set_indexer_context_lens(forward_batch, context_lens_np: np.ndarray) -> None:
    """Indexer-only logical KV length for draft_forward sub-steps.

    Must not mutate ``forward_batch.seq_lens`` — that tensor is shared with the
    scheduler batch and would corrupt the subsequent target_verify pass.
    """
    bs = int(context_lens_np.size)
    if bs <= 0:
        return
    device = forward_batch.seq_lens.device
    lens = torch.from_numpy(context_lens_np.astype(np.int32)).to(device=device)
    setattr(forward_batch, INDEXER_CONTEXT_LENS_ATTR, lens)


def clear_indexer_context_lens(forward_batch) -> None:
    if hasattr(forward_batch, INDEXER_CONTEXT_LENS_ATTR):
        delattr(forward_batch, INDEXER_CONTEXT_LENS_ATTR)


def resolve_indexer_seq_lens(forward_batch, bs: int) -> torch.Tensor:
    """Return seq lens for sparse MLA indexer decode (may differ from batch)."""
    ctx = getattr(forward_batch, INDEXER_CONTEXT_LENS_ATTR, None)
    if torch.is_tensor(ctx) and int(ctx.numel()) >= bs:
        return ctx[:bs].to(dtype=torch.int32)
    return forward_batch.seq_lens[:bs].to(dtype=torch.int32)


def gather_draft_decode_token_row(
    req_to_token_row: torch.Tensor,
    seq_len: int,
    sub_step: int,
    *,
    topk_id: int = 0,
    num_steps: int = 1,
    page_size: int = 1,
) -> torch.Tensor:
    """Gather one draft-decode KV row using SGLang cache_locs semantics.

    Mirrors ``generate_draft_decode_kv_indices`` for page_size=1 / topk=1:
    history ``req_to_token[:seq_len]`` plus draft slots
    ``req_to_token[seq_len + topk_id * num_steps : seq_len + topk_id * num_steps + sub_step + 1]``.

    ``seq_len`` must be post-verify committed length (``batch.seq_lens``), not
    ``prefix + K``. Reject slots inside the draft_extend K-window are skipped
    because history ends at committed.
    """
    seq_len = int(seq_len)
    iters = int(sub_step) + 1
    if seq_len > int(req_to_token_row.numel()):
        raise RuntimeError(
            f"gather_draft_decode_token_row: seq_len={seq_len} exceeds row width "
            f"{int(req_to_token_row.numel())}"
        )
    if page_size == 1 or topk_id == 0:
        history = req_to_token_row[:seq_len]
        extend_start = seq_len + int(topk_id) * int(num_steps)
        extend_end = min(extend_start + iters, int(req_to_token_row.numel()))
        extend = req_to_token_row[extend_start:extend_end]
        if extend.numel() < iters:
            pad = torch.zeros(
                iters - int(extend.numel()),
                dtype=history.dtype,
                device=history.device,
            )
            extend = torch.cat([extend, pad])
        return torch.cat([history, extend.to(dtype=history.dtype)])

    prefix_len = seq_len
    last_page_len = prefix_len % page_size
    num_new_pages_per_topk = (last_page_len + num_steps + page_size - 1) // page_size
    prefix_base = prefix_len // page_size * page_size
    start = prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
    history = req_to_token_row[:seq_len]
    extend = req_to_token_row[start : start + iters]
    if extend.numel() < iters:
        pad = torch.zeros(
            iters - int(extend.numel()),
            dtype=history.dtype,
            device=history.device,
        )
        extend = torch.cat([extend, pad])
    return torch.cat([history, extend.to(dtype=history.dtype)])


def build_draft_decode_token_table(
    forward_batch,
    req_to_token_pool,
    *,
    bs: int,
    seq_lens_np: np.ndarray,
    sub_step: int,
    num_steps: int = 1,
    topk: int = 1,
    page_size: int = 1,
) -> torch.Tensor:
    """Build per-request token table for draft_forward sub-steps (SGLang layout)."""
    req_n = int(forward_batch.req_pool_indices.numel())
    bs = min(int(bs), req_n, int(seq_lens_np.size))
    if bs <= 0:
        raise RuntimeError("build_draft_decode_token_table: empty batch")

    device = forward_batch.req_pool_indices.device
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    raw = req_to_token_pool.req_to_token[req_pool_indices]
    context_lens_np = (seq_lens_np[:bs] + int(sub_step) + 1).astype(np.int32)
    max_len = int(context_lens_np.max(initial=1))
    out = torch.zeros(bs, max_len, dtype=torch.int32, device=device)

    for row in range(bs):
        row_tokens = gather_draft_decode_token_row(
            raw[row],
            int(seq_lens_np[row]),
            int(sub_step),
            topk_id=0,
            num_steps=int(num_steps),
            page_size=int(page_size),
        )
        length = min(int(row_tokens.numel()), max_len)
        out[row, :length] = row_tokens[:length].to(dtype=torch.int32)

    # NOTE: Do NOT overwrite out[row, ctx-1] with out_cache_loc[row]. The gather
    # above already reads the authoritative draft slot from req_to_token
    # (identical to native generate_draft_decode_kv_indices). out_cache_loc is
    # the KV *write* target and belongs only in slot_mapping; using it to build
    # KV *read* indices duplicates SGLang's slot bookkeeping and can diverge
    # under topk>1 / page_size>1 / post-filter reordering.

    if int(page_size) == 1:
        return out.contiguous()
    return (out[:, ::page_size] // page_size).to(dtype=torch.int32).contiguous()


def resolve_speculative_num_steps(forward_batch, default: int = 1) -> int:
    """Draft tree depth for ``generate_draft_decode_kv_indices`` extend offset."""
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is not None:
        for attr in ("speculative_num_steps", "_speculative_num_steps"):
            value = getattr(spec_info, attr, None)
            if value is not None:
                return max(1, int(value))
    cached = getattr(forward_batch, "_atom_glm52_speculative_num_steps", None)
    if cached is not None:
        return max(1, int(cached))
    return max(1, int(default))


def build_accept_compacted_token_table(
    forward_batch,
    req_to_token_pool,
    *,
    bs: int,
    context_lens_np: np.ndarray,
    prefix_lens_np: np.ndarray,
    accept_lens_np: np.ndarray,
    sub_step: int,
    page_size: int,
) -> torch.Tensor:
    """Compact draft-pool routing: skip reject tail [prefix+accept, prefix+K)."""
    req_n = int(forward_batch.req_pool_indices.numel())
    bs = min(
        int(bs),
        req_n,
        int(context_lens_np.size),
        int(prefix_lens_np.size),
        int(accept_lens_np.size),
    )
    if bs <= 0:
        raise RuntimeError("build_accept_compacted_token_table: empty batch")
    device = forward_batch.req_pool_indices.device
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    raw = req_to_token_pool.req_to_token[req_pool_indices].clone()
    max_len = int(context_lens_np.max(initial=1))
    out = torch.zeros(bs, max_len, dtype=torch.int32, device=device)
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)

    for row in range(bs):
        prefix = int(prefix_lens_np[row])
        accept = int(accept_lens_np[row])
        context = int(context_lens_np[row])
        parts: list[torch.Tensor] = []
        if prefix > 0:
            parts.append(raw[row, :prefix])
        if accept > 0:
            parts.append(raw[row, prefix : prefix + accept])
        if sub_step > 0:
            parts.append(raw[row, prefix + accept : prefix + accept + sub_step])

        compact = torch.cat(parts) if parts else raw[row, :0]
        past_len = max(context - 1, 0)
        if compact.numel() > past_len:
            compact = compact[:past_len]
        if compact.numel() < past_len:
            tail = raw[row, compact.numel() : past_len]
            if tail.numel() > 0:
                compact = torch.cat([compact, tail.to(dtype=compact.dtype)])

        if torch.is_tensor(out_cache_loc) and int(out_cache_loc.numel()) > row:
            slot = out_cache_loc[row].reshape(1).to(dtype=compact.dtype)
            compact = torch.cat([compact, slot])
        elif compact.numel() < context:
            pad = raw[row, compact.numel() : context]
            if pad.numel() > 0:
                compact = torch.cat([compact, pad.to(dtype=compact.dtype)])

        if compact.numel() > context:
            compact = compact[:context]
        out[row, : compact.numel()] = compact.to(dtype=torch.int32)

    if page_size == 1:
        return out.contiguous()
    return (out[:, ::page_size] // page_size).to(dtype=torch.int32).contiguous()


def get_extend_lens_cpu(forward_batch, positions: torch.Tensor, bs: int) -> np.ndarray:
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is not None:
        if torch.is_tensor(extend_lens):
            extend_lens = extend_lens.detach().cpu().numpy()
        return np.asarray(extend_lens[:bs], dtype=np.int32)

    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is None:
        tokens_per_req = max(1, int(positions.numel()) // max(1, bs))
    return np.full(bs, int(tokens_per_req), dtype=np.int32)


def build_token_table(
    forward_batch,
    req_to_token_pool,
    *,
    seq_lens: np.ndarray,
    extend_lens: np.ndarray | None,
    page_size: int,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    if extend_lens is not None and not forward_batch.forward_mode.is_decode_or_idle():
        prefix_lens = np.maximum(seq_lens - extend_lens, 0).astype(np.int32)
        table_lens = np.maximum(seq_lens, prefix_lens + extend_lens)
    else:
        prefix_lens = None
        table_lens = seq_lens
    max_seq_len = int(table_lens.max(initial=1))
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    token_table = req_to_token_pool.req_to_token[req_pool_indices, :max_seq_len].clone()

    # NOTE: Do NOT overwrite token_table[:, prefix:prefix+K] with out_cache_loc.
    # SGLang already writes the extend/verify draft slots into req_to_token
    # (assign_extend_cache_locs during verify, and the draft-extend KV fill),
    # so req_to_token is the authoritative KV-index source — exactly what native
    # create_flashinfer_kv_indices_triton reads. out_cache_loc is the KV *write*
    # target and belongs only in slot_mapping; using it to build read indices
    # duplicates SGLang slot bookkeeping and can diverge after filter_batch.

    if page_size == 1:
        return token_table.to(dtype=torch.int32).contiguous()
    return (token_table[:, ::page_size] // page_size).to(dtype=torch.int32).contiguous()


def flatten_kv_indices(token_table: torch.Tensor, lengths: np.ndarray) -> torch.Tensor:
    pieces = []
    for row, length in enumerate(lengths):
        if int(length) > 0:
            pieces.append(token_table[row, : int(length)])
    if not pieces:
        return torch.empty(0, dtype=torch.int32, device=token_table.device)
    return torch.cat(pieces).to(dtype=torch.int32).contiguous()


def counts_to_indptr(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    indptr = np.zeros(len(counts) + 1, dtype=np.int32)
    if len(counts):
        indptr[1:] = np.cumsum(counts, dtype=np.int32)
    return torch.from_numpy(indptr).to(device=device)


def get_index_topk(atom_config) -> int:
    topk = getattr(atom_config.hf_config, "index_topk", None)
    if topk is None:
        raise RuntimeError("GLM-5.2 DSA bridge requires hf_config.index_topk")
    return int(topk)


def local_num_attention_heads(atom_config) -> int:
    hf_config = atom_config.hf_config
    num_heads = int(hf_config.num_attention_heads)
    tp_size = int(getattr(atom_config, "tensor_parallel_size", 1))
    return max(1, num_heads // max(1, tp_size))


def metadata_dtype(atom_config):
    kv_dtype = getattr(atom_config, "kv_cache_dtype", "bf16")
    if str(kv_dtype).startswith("fp8"):
        return dtypes.fp8
    return getattr(dtypes, "d_dtypes", {}).get(kv_dtype, torch.bfloat16)


def is_draft_extend_mode(forward_batch) -> bool:
    return bool(
        getattr(forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False)(
            include_v2=True
        )
    )


def compute_mtp_sparse_per_token_kv_lens(
    *,
    prefix_lens_np: np.ndarray,
    context_lens_np: np.ndarray,
    max_seqlen_q: int,
    bs: int,
    draft_extend: bool,
) -> np.ndarray:
    """Per-query KV lengths for sparse MTP target_verify / draft_extend."""
    if draft_extend:
        return (
            np.repeat(prefix_lens_np, max_seqlen_q)
            + np.tile(np.arange(1, max_seqlen_q + 1, dtype=np.int32), bs)
        ).astype(np.int32)
    return (
        np.repeat(context_lens_np, max_seqlen_q)
        - max_seqlen_q
        + np.tile(np.arange(1, max_seqlen_q + 1, dtype=np.int32), bs)
    ).astype(np.int32)


def make_mla_work_buffers(
    *,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_heads: int,
    dtype_q,
    dtype_kv,
    page_size: int,
) -> dict[str, torch.Tensor]:
    num_seqs = max(1, int(cu_seqlens_q.numel()) - 1)
    max_q_len = 1
    if cu_seqlens_q.numel() > 1:
        q_counts = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_q_len = max(1, int(q_counts.max().item()))
    padded_heads = max(num_heads, 16)
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        num_seqs,
        max_q_len,
        padded_heads,
        dtype_q,
        dtype_kv,
        is_sparse=True,
        fast_mode=True,
    )
    device = cu_seqlens_q.device
    work = {
        "work_meta_data": torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        ),
        "work_indptr": torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        ),
        "work_info_set": torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        ),
        "reduce_indptr": torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        ),
        "reduce_final_map": torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        ),
        "reduce_partial_map": torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        ),
    }
    get_mla_metadata_v1(
        cu_seqlens_q,
        kv_indptr,
        kv_last_page_lens,
        padded_heads,
        1,
        True,
        work["work_meta_data"],
        work["work_info_set"],
        work["work_indptr"],
        work["reduce_indptr"],
        work["reduce_final_map"],
        work["reduce_partial_map"],
        page_size=page_size,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=True,
    )
    return work


def make_sparse_mtp_work_buffers(
    *,
    sparse_cu_seqlens_q: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    sparse_kv_last_page_lens: torch.Tensor,
    num_heads: int,
    dtype_q,
    dtype_kv,
    page_size: int,
) -> dict[str, torch.Tensor]:
    num_tokens = max(1, int(sparse_cu_seqlens_q.numel()) - 1)
    padded_heads = max(num_heads, 16)
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        num_tokens,
        1,
        padded_heads,
        dtype_q,
        dtype_kv,
        is_sparse=True,
        fast_mode=True,
    )
    device = sparse_cu_seqlens_q.device
    work = {
        "sparse_mtp_work_meta_data": torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        ),
        "sparse_mtp_work_indptr": torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        ),
        "sparse_mtp_work_info_set": torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        ),
        "sparse_mtp_reduce_indptr": torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        ),
        "sparse_mtp_reduce_final_map": torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        ),
        "sparse_mtp_reduce_partial_map": torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        ),
    }
    get_mla_metadata_v1(
        sparse_cu_seqlens_q,
        sparse_kv_indptr,
        sparse_kv_last_page_lens,
        padded_heads,
        1,
        True,
        work["sparse_mtp_work_meta_data"],
        work["sparse_mtp_work_info_set"],
        work["sparse_mtp_work_indptr"],
        work["sparse_mtp_reduce_indptr"],
        work["sparse_mtp_reduce_final_map"],
        work["sparse_mtp_reduce_partial_map"],
        page_size=page_size,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
    )
    return work


def ensure_shared_sparse_buffer(
    token_to_kv_pool,
    *,
    num_tokens: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    required = max(1, int(num_tokens) * int(topk))
    buffer = getattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, None)
    if (
        buffer is None
        or buffer.device != device
        or buffer.dtype != torch.int32
        or buffer.numel() < required
    ):
        buffer = torch.empty(required, dtype=torch.int32, device=device)
        setattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, buffer)
    return buffer[:required]


def validate_page_size(token_to_kv_pool, atom_config) -> int:
    page_size = int(getattr(token_to_kv_pool, "page_size", 1))
    from atom.utils import envs

    atom_config.kv_cache_block_size = page_size
    setattr(token_to_kv_pool, INDEXER_PAGE_SIZE_ATTR, page_size)
    setattr(token_to_kv_pool, ATTENTION_PAGE_SIZE_ATTR, int(envs.ATOM_MLA_PAGE_SIZE))
    return page_size


def attention_page_size(token_to_kv_pool) -> int:
    return int(getattr(token_to_kv_pool, ATTENTION_PAGE_SIZE_ATTR, 1))


# --- cache_bind.py ---
"""Bind SGLang KV pool views to ATOM GLM-5.2 sparse MLA modules."""


def bind_glm52_dsa_cache_views(model, token_to_kv_pool) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_key_buffer"):
        return False
    if not hasattr(token_to_kv_pool, "get_index_k_with_scale_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    shared_sparse = getattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, None)
    if shared_sparse is None:
        return False

    page_size = int(
        getattr(
            token_to_kv_pool,
            INDEXER_PAGE_SIZE_ATTR,
            getattr(token_to_kv_pool, "page_size", 1),
        )
    )
    empty_value_cache = getattr(token_to_kv_pool, EMPTY_VALUE_CACHE_ATTR, None)
    if empty_value_cache is None or empty_value_cache.device != shared_sparse.device:
        empty_value_cache = torch.empty(0, device=shared_sparse.device)
        setattr(token_to_kv_pool, EMPTY_VALUE_CACHE_ATTR, empty_value_cache)
    kv_cache_data = {}
    for module in model.modules():
        if not isinstance(module, DeepseekV2MLAAttention):
            continue

        layer_id = int(module.layer_num)
        mla_attn = module.mla_attn
        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=token_to_kv_pool.get_key_buffer(layer_id),
            v_cache=empty_value_cache,
            k_scale=getattr(mla_attn, "_k_scale", None),
            v_scale=getattr(mla_attn, "_k_scale", None),
        )

        indexer = getattr(module, "indexer", None)
        if indexer is not None:
            index_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
            index_entry_dim = int(indexer.head_dim) + 4
            indexer.k_cache.kv_cache[0] = index_cache.view(
                -1, page_size, index_entry_dim
            )
            indexer.sparse_kv_indices_buffer = shared_sparse

        if hasattr(mla_attn, "sparse_kv_indices_buffer"):
            mla_attn.sparse_kv_indices_buffer = shared_sparse

    if not kv_cache_data:
        return False

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return True


# --- multi_token.py ---
"""Multi-token MTP metadata shared by target_verify and draft_extend."""


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
    context_lens = torch.from_numpy(context_lens_np).to(
        device=device, dtype=torch.int32
    )

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

    return md


# --- target_verify.py ---
"""Target verify metadata — target pool, bs×K query rows."""


def resolve_target_verify_lens(
    forward_batch,
    positions: torch.Tensor,
    bs: int,
    draft_token_num: int,
    *,
    use_positions: bool = False,
):
    """Return committed prefix lengths and total KV lengths for target_verify."""
    position_rows = positions.detach().cpu().numpy().astype(np.int32)
    required = bs * draft_token_num
    if position_rows.size < required:
        raise RuntimeError(
            "GLM-5.2 DSA target_verify positions are shorter than "
            f"bs*draft_token_num: positions={position_rows.size}, "
            f"bs={bs}, draft_token_num={draft_token_num}"
        )
    prefix_lens = position_rows[:required:draft_token_num].astype(np.int32)
    if use_positions:
        context_lens = prefix_lens + draft_token_num
    else:
        position_context_lens = prefix_lens + draft_token_num
        context_lens = np.maximum(
            get_seq_lens_cpu(forward_batch, bs), position_context_lens
        )
        prefix_lens = context_lens - draft_token_num
    return prefix_lens, context_lens


def should_use_mtp_verify_prefill_path(
    forward_batch,
    positions: torch.Tensor,
    atom_config,
) -> bool:
    """Choose prefill vs decode metadata for eager target_verify."""
    del forward_batch, positions, atom_config
    override = os.environ.get("ATOM_GLM52_TV_VERIFY_PATH", "").lower()
    if override in ("prefill", "prefill_prefix"):
        return True
    if override in ("decode",):
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    return False


def build_mtp_verify_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
    use_positions: bool = False,
):
    """Build ATOM-native decode-style metadata for SGLang target_verify."""
    draft_token_num = int(
        getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0) or 0
    )
    if draft_token_num <= 0:
        raise RuntimeError("GLM-5.2 DSA target_verify requires draft_token_num")

    def resolve_lens(batch, position_rows, batch_size, num_draft_tokens):
        return resolve_target_verify_lens(
            batch,
            position_rows,
            batch_size,
            num_draft_tokens,
            use_positions=use_positions,
        )

    return build_mtp_multi_token_decode_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
        draft_token_num=draft_token_num,
        resolve_lens_fn=resolve_lens,
    )


# --- draft_decode.py ---
"""Draft forward decode metadata — draft pool, 1 tok/query, multi-step sub_step."""

logger = logging.getLogger("atom.plugin.sglang.glm52_dsa_bridge")


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
    num_steps = resolve_speculative_num_steps(
        forward_batch, default=max(sub_step + 1, 1)
    )

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
    context_lens = torch.from_numpy(context_lens_np).to(
        device=device, dtype=torch.int32
    )

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
        slot_mapping = out_cache_loc[:bs]
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


# --- draft_extend.py ---
"""Draft extend metadata — draft pool, DRAFT_EXTEND_V2 bs×K fill after verify."""


def draft_extend_k_only() -> bool:
    """Debug opt-in: attend only the current K-token extend chunk (legacy workaround)."""
    return os.environ.get("ATOM_GLM52_DRAFT_EXTEND_K_ONLY", "0") in (
        "1",
        "true",
        "True",
    )


def resolve_draft_extend_lens(
    forward_batch,
    positions: torch.Tensor,
    bs: int,
    draft_token_num: int,
):
    """Return prefix and total KV lengths for DRAFT_EXTEND_V2."""
    draft_token_num = int(draft_token_num)
    if draft_extend_k_only():
        prefix_lens = np.zeros(bs, dtype=np.int32)
        context_lens = np.full(bs, draft_token_num, dtype=np.int32)
        return prefix_lens, context_lens

    seq_lens = get_seq_lens_cpu(forward_batch, bs)
    position_rows = positions.detach().cpu().numpy().astype(np.int32)
    required = bs * draft_token_num
    if position_rows.size >= required:
        prefix_lens = position_rows[:required:draft_token_num].astype(np.int32)
    else:
        prefix_lens = get_extend_prefix_lens_cpu(forward_batch, bs)
        if prefix_lens is None:
            prefix_lens = np.maximum(seq_lens - draft_token_num, 0).astype(np.int32)
        else:
            prefix_lens = prefix_lens.astype(np.int32)

    context_lens = (prefix_lens + draft_token_num).astype(np.int32)
    context_lens = np.maximum(context_lens, seq_lens).astype(np.int32)
    return prefix_lens.astype(np.int32), context_lens.astype(np.int32)


def draft_extend_token_num(forward_batch, positions: torch.Tensor, bs: int) -> int:
    extend_lens = get_extend_lens_cpu(forward_batch, positions, bs)
    if extend_lens.size:
        return int(extend_lens.max(initial=1))
    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is not None:
        return int(tokens_per_req)
    if bs > 0 and int(positions.numel()) >= bs:
        return max(1, int(positions.numel()) // bs)
    return 1


def should_use_mtp_draft_extend_decode_path(forward_batch) -> bool:
    """Use decode-style draft_extend metadata (native propose i=0 semantics)."""
    override = os.environ.get("ATOM_GLM52_DRAFT_EXTEND_PATH", "").lower()
    if override in ("prefill", "prefill_prefix"):
        return False
    if override in ("decode",):
        return True
    if is_draft_extend_mode(forward_batch):
        return True
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return True
    return int(getattr(forward_batch, "_graph_cache_bs", 0) or 0) > 0


def build_mtp_draft_extend_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    """Build decode-style metadata for SGLang DRAFT_EXTEND_V2 (draft step i=0)."""
    bs = int(forward_batch.batch_size)
    draft_token_num = draft_extend_token_num(forward_batch, positions, bs)
    if draft_token_num <= 0:
        raise RuntimeError("GLM-5.2 DSA draft_extend requires draft_token_num")
    return build_mtp_multi_token_decode_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
        draft_token_num=draft_token_num,
        resolve_lens_fn=resolve_draft_extend_lens,
    )


# --- prefill.py ---
"""Prefill metadata — target prefill and draft_extend_for_prefill path."""


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
    return num_tokens_per_req is None or int(num_tokens_per_req) == 1


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


# --- decode_graph.py ---
"""Target decode CUDA-graph metadata buffers."""


class GLM52DecodeGraphBuffers:
    def __init__(
        self,
        *,
        max_bs: int,
        max_context_len: int,
        indexer_page_size: int,
        attention_page_size: int,
        index_topk: int,
        num_heads: int,
        dtype_q,
        dtype_kv,
        device: torch.device,
    ) -> None:
        self.max_bs = int(max_bs)
        self.max_context_len = int(max_context_len)
        self.indexer_page_size = int(indexer_page_size)
        self.attention_page_size = int(attention_page_size)
        self.index_topk = int(index_topk)
        self.device = device

        max_blocks = max(
            1,
            (self.max_context_len + self.indexer_page_size - 1)
            // self.indexer_page_size,
        )
        self.cu_q = torch.arange(self.max_bs + 1, dtype=torch.int32, device=device)
        self.kv_indptr = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=device)
        self.sparse_kv_indptr = torch.zeros(
            self.max_bs + 1, dtype=torch.int32, device=device
        )
        self.kv_indices = torch.empty(
            self.max_bs * self.max_context_len, dtype=torch.int32, device=device
        )
        self.kv_last_page_lens = torch.ones(
            self.max_bs, dtype=torch.int32, device=device
        )
        self.block_tables = torch.empty(
            self.max_bs, max_blocks, dtype=torch.int32, device=device
        )
        self.context_lens = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self.slot_mapping = torch.zeros(self.max_bs, dtype=torch.int64, device=device)
        self.shared_sparse = torch.empty(
            self.max_bs * self.index_topk, dtype=torch.int32, device=device
        )

        work = make_mla_work_buffers(
            cu_seqlens_q=self.cu_q,
            kv_indptr=self.sparse_kv_indptr,
            kv_last_page_lens=self.kv_last_page_lens,
            num_heads=num_heads,
            dtype_q=dtype_q,
            dtype_kv=dtype_kv,
            page_size=self.attention_page_size,
        )
        self.work_meta_data = work["work_meta_data"]
        self.work_indptr = work["work_indptr"]
        self.work_info_set = work["work_info_set"]
        self.reduce_indptr = work["reduce_indptr"]
        self.reduce_final_map = work["reduce_final_map"]
        self.reduce_partial_map = work["reduce_partial_map"]

    def stage_block_tables(self, req_to_token_pool, req_pool_indices, bs: int) -> None:
        req_to_token = req_to_token_pool.req_to_token
        live = req_to_token[
            req_pool_indices[:bs],
            : self.max_context_len : self.indexer_page_size,
        ]
        self.block_tables[:bs, : live.shape[1]].copy_(
            (live // self.indexer_page_size).to(torch.int32)
        )


def get_or_create_decode_graph_buffers(
    token_to_kv_pool,
    *,
    max_bs: int,
    max_context_len: int,
    indexer_page_size: int,
    attention_page_size_val: int,
    atom_config,
    device: torch.device,
) -> GLM52DecodeGraphBuffers:
    topk = get_index_topk(atom_config)
    dtype_q = metadata_dtype(atom_config)
    bufs = getattr(token_to_kv_pool, DECODE_GRAPH_BUFFERS_ATTR, None)
    if (
        bufs is None
        or bufs.max_bs < int(max_bs)
        or bufs.max_context_len < int(max_context_len)
        or bufs.indexer_page_size != int(indexer_page_size)
        or bufs.attention_page_size != int(attention_page_size_val)
        or bufs.index_topk != int(topk)
        or bufs.device != device
    ):
        bufs = GLM52DecodeGraphBuffers(
            max_bs=max_bs,
            max_context_len=max_context_len,
            indexer_page_size=indexer_page_size,
            attention_page_size=attention_page_size_val,
            index_topk=topk,
            num_heads=local_num_attention_heads(atom_config),
            dtype_q=dtype_q,
            dtype_kv=dtype_q,
            device=device,
        )
        setattr(token_to_kv_pool, DECODE_GRAPH_BUFFERS_ATTR, bufs)
        setattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, bufs.shared_sparse)
    return bufs


def build_atom_glm52_decode_graph_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
    max_bs: int | None = None,
    max_context_len: int | None = None,
):
    from atom.utils.forward_context import AttentionMetaData, AttnState

    del positions
    device = forward_batch.seq_lens.device
    bs = int(forward_batch.batch_size)
    seq_lens = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
    if max_context_len is None:
        req_to_token = req_to_token_pool.req_to_token
        max_context_len = int(req_to_token.shape[1])
    if max_bs is None:
        max_bs = max(bs, int(getattr(req_to_token_pool, "size", bs)))

    indexer_page_size = validate_page_size(token_to_kv_pool, atom_config)
    attn_page_size = attention_page_size(token_to_kv_pool)
    topk = get_index_topk(atom_config)
    dtype_q = metadata_dtype(atom_config)

    bufs = get_or_create_decode_graph_buffers(
        token_to_kv_pool,
        max_bs=max_bs,
        max_context_len=max_context_len,
        indexer_page_size=indexer_page_size,
        attention_page_size_val=attn_page_size,
        atom_config=atom_config,
        device=device,
    )

    bufs.kv_indptr.zero_()
    bufs.kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
    bufs.sparse_kv_indptr.zero_()
    bufs.sparse_kv_indptr[1 : bs + 1] = torch.cumsum(
        torch.clamp(seq_lens, max=topk), dim=0
    )
    bufs.context_lens[:bs].copy_(seq_lens)
    bufs.kv_last_page_lens[:bs].fill_(1)

    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_cache_loc):
        copy_n = min(bs, int(out_cache_loc.numel()))
        if copy_n:
            bufs.slot_mapping[:copy_n].copy_(out_cache_loc[:copy_n])
        if bs > copy_n:
            scratch_slot = max(0, int(getattr(token_to_kv_pool, "size", 1)) - 1)
            bufs.slot_mapping[copy_n:bs].fill_(scratch_slot)
    else:
        scratch_slot = max(0, int(getattr(token_to_kv_pool, "size", 1)) - 1)
        bufs.slot_mapping[:bs].fill_(scratch_slot)

    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token_pool.req_to_token,
        forward_batch.req_pool_indices[:bs],
        seq_lens,
        bufs.kv_indptr[: bs + 1],
        None,
        bufs.kv_indices,
        req_to_token_pool.req_to_token.stride(0),
    )
    bufs.stage_block_tables(req_to_token_pool, forward_batch.req_pool_indices, bs)

    get_mla_metadata_v1(
        bufs.cu_q[: bs + 1],
        bufs.sparse_kv_indptr[: bs + 1],
        bufs.kv_last_page_lens[:bs],
        max(local_num_attention_heads(atom_config), 16),
        1,
        True,
        bufs.work_meta_data,
        bufs.work_info_set,
        bufs.work_indptr,
        bufs.reduce_indptr,
        bufs.reduce_final_map,
        bufs.reduce_partial_map,
        page_size=attn_page_size,
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        kv_granularity=max(attn_page_size, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
    )

    setattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, bufs.shared_sparse)
    md = AttentionMetaData(
        cu_seqlens_q=bufs.cu_q[: bs + 1],
        cu_seqlens_k=bufs.kv_indptr[: bs + 1],
        max_seqlen_q=1,
        max_seqlen_k=int(seq_lens.max().item()) if bs else 1,
        slot_mapping=bufs.slot_mapping[:bs],
        context_lens=bufs.context_lens[:bs],
        block_tables=bufs.block_tables[:bs],
        state=AttnState.DECODE,
        kv_indptr=bufs.kv_indptr[: bs + 1],
        kv_indices=bufs.kv_indices,
        kv_last_page_lens=bufs.kv_last_page_lens[:bs],
        sparse_kv_indptr=bufs.sparse_kv_indptr[: bs + 1],
        work_meta_data=bufs.work_meta_data,
        work_indptr=bufs.work_indptr,
        work_info_set=bufs.work_info_set,
        reduce_indptr=bufs.reduce_indptr,
        reduce_final_map=bufs.reduce_final_map,
        reduce_partial_map=bufs.reduce_partial_map,
    )
    md.sparse_kv_last_page_lens = bufs.kv_last_page_lens[:bs]
    md.dtype_q = dtype_q
    return md


# --- dispatcher.py ---
"""Forward-mode router: target_verify / draft_extend / draft_decode / prefill."""


def build_atom_glm52_attention_metadata_from_sglang(
    forward_batch,
    positions,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    if getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)():
        if should_use_mtp_verify_prefill_path(forward_batch, positions, atom_config):
            raise RuntimeError(
                "GLM-5.2 DSA target_verify prefill metadata is not ported yet; "
                "use decode path (default) or unset ATOM_GLM52_TV_VERIFY_PATH"
            )
        return build_mtp_verify_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    if forward_batch.forward_mode.is_decode_or_idle():
        return build_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    if is_draft_extend_prefill(forward_batch):
        return build_mtp_draft_extend_prefill_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    if getattr(forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False)(
        include_v2=True
    ):
        if should_use_mtp_draft_extend_decode_path(forward_batch):
            return build_mtp_draft_extend_decode_metadata(
                forward_batch,
                positions,
                token_to_kv_pool=token_to_kv_pool,
                req_to_token_pool=req_to_token_pool,
                atom_config=atom_config,
            )
        return build_prefill_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    return build_prefill_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
    )
