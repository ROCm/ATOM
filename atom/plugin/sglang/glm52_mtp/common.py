"""Shared GLM-5.2 DSA bridge helpers (all MTP phases)."""

from __future__ import annotations

import numpy as np
import torch
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

from atom.plugin.sglang.runtime.model_arch import is_glm52_dsa_config

DECODE_GRAPH_BUFFERS_ATTR = "_atom_glm52_decode_graph_buffers"
EMPTY_VALUE_CACHE_ATTR = "_atom_glm52_empty_value_cache"
INDEXER_PAGE_SIZE_ATTR = "_atom_glm52_indexer_page_size"
ATTENTION_PAGE_SIZE_ATTR = "_atom_glm52_attention_page_size"
SHARED_SPARSE_INDICES_ATTR = "_atom_glm52_shared_sparse_kv_indices"
DRAFT_SUB_STEP_ATTR = "_atom_glm52_draft_decode_sub_step"
INDEXER_CONTEXT_LENS_ATTR = "_atom_glm52_indexer_context_lens"


def is_glm52_dsa_arch(config) -> bool:
    return is_glm52_dsa_config(config)


def maybe_get_glm52_dsa_pools_from_sglang_backend(forward_batch=None):
    if forward_batch is not None:
        token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        if token_to_kv_pool is not None and req_to_token_pool is not None:
            return token_to_kv_pool, req_to_token_pool

    # Fallback only for draft-side forwards that may omit pools on ForwardBatch.
    if forward_batch is not None:
        try:
            from atom.plugin.sglang.glm52_mtp.draft_decode import is_draft_decode_metadata

            draft_side = is_draft_decode_metadata(forward_batch) or is_draft_extend_mode(
                forward_batch
            )
        except Exception:
            draft_side = False
        if not draft_side:
            return None, None

    backend = None
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
    except Exception:
        backend = None

    if backend is not None:
        token_to_kv_pool = getattr(backend, "token_to_kv_pool", None)
        req_to_token_pool = getattr(backend, "req_to_token_pool", None)
        if token_to_kv_pool is not None and req_to_token_pool is not None:
            return token_to_kv_pool, req_to_token_pool

    return None, None


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
        extend_end = extend_start + iters
        if extend_end > int(req_to_token_row.numel()):
            extend_end = int(req_to_token_row.numel())
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
    num_heads = int(getattr(hf_config, "num_attention_heads"))
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
