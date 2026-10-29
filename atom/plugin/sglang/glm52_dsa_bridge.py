"""GLM-5.2 DSA and MTP integration for the SGLang plugin.

This module owns GLM-5.2 metadata construction, KV cache binding, and decode CUDA Graph support.
"""

from __future__ import annotations

# --- common.py ---
# Shared GLM-5.2 DSA bridge helpers (all MTP phases).
import logging

import numpy as np
import torch
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton

from atom.plugin.sglang.runtime.attention_backend_resolver import (
    resolve_sglang_runtime,
)
from atom.plugin.sglang.runtime.model_arch import is_glm52_dsa_config

logger = logging.getLogger("atom.plugin.sglang.glm52_dsa_bridge")

DECODE_GRAPH_BUFFERS_ATTR = "_atom_glm52_decode_graph_buffers"
EMPTY_VALUE_CACHE_ATTR = "_atom_glm52_empty_value_cache"
INDEXER_PAGE_SIZE_ATTR = "_atom_glm52_indexer_page_size"
ATTENTION_PAGE_SIZE_ATTR = "_atom_glm52_attention_page_size"
SHARED_SPARSE_INDICES_ATTR = "_atom_glm52_shared_sparse_kv_indices"
DRAFT_SUB_STEP_ATTR = "_atom_glm52_draft_decode_sub_step"
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


def _is_graph_fill(length: int) -> bool:
    """True when ``length`` is the CUDA-graph capacity, not a written context."""
    return int(length) >= int(GLM52_GRAPH_SEQ_LEN_CAPACITY)


def _prefer_real_length(primary: int, alternate: int) -> int:
    """Keep a real context when the other source is still the graph fill.

    Capture fills ``seq_lens`` (and sometimes the positions buffer) with
    ``GLM52_GRAPH_SEQ_LEN_CAPACITY``. ``max(fill, real)`` then walks
    ``req_to_token`` past the tokens prefill wrote, and the verify indexer
    memory-faults. A genuine long context, where both sources are past the
    fill, is left unchanged.
    """
    primary_fill = _is_graph_fill(primary)
    alternate_fill = _is_graph_fill(alternate)
    if primary_fill and not alternate_fill:
        return int(alternate)
    if alternate_fill and not primary_fill:
        return int(primary)
    return max(int(primary), int(alternate))


def _drop_unconfirmed_graph_fill(lengths: np.ndarray, forward_batch) -> np.ndarray:
    """Drop a CUDA-graph capacity sentinel the host sum does not confirm.

    Capture fills ``seq_lens`` with ``GLM52_GRAPH_SEQ_LEN_CAPACITY``. Replay
    padding keeps that sentinel, and a live row can keep it too while
    ``seq_lens_sum`` still counts the tokens prefill wrote. The verify indexer
    then loads ``req_to_token`` past the KV cache and the GPU faults. A real
    context at the capacity has a matching host sum and is preserved. No host
    sum means this is the capture allocation, which must stay wide.
    """

    out = np.asarray(lengths, dtype=np.int32).copy()
    if out.size == 0 or not np.any(out >= int(GLM52_GRAPH_SEQ_LEN_CAPACITY)):
        return out
    seq_lens_sum = getattr(forward_batch, "seq_lens_sum", None)
    if seq_lens_sum is None:
        return out

    fill = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    num_padding = int(getattr(forward_batch, "num_padding", 0) or 0)
    num_padding = max(0, min(num_padding, int(out.size)))
    raw_bs = int(out.size) - num_padding
    host_sum = int(seq_lens_sum)
    if num_padding and host_sum >= num_padding * fill:
        host_sum -= num_padding * fill
    if num_padding:
        out[raw_bs:] = 1
    if raw_bs <= 0 or not np.any(out[:raw_bs] >= fill):
        return out
    raw = out[:raw_bs]
    if int(raw.sum()) == host_sum or host_sum >= fill * raw_bs:
        return out
    if raw_bs == 1 and host_sum > 0:
        raw[0] = np.int32(host_sum)
    else:
        raw[raw >= fill] = 1
    logger.warning(
        "GLM graph-fill context %s does not match host sum %s; walking %s",
        np.asarray(lengths, dtype=np.int32).tolist(),
        int(seq_lens_sum),
        out.tolist(),
    )
    return out


def _cap_context_to_host_seq(
    context_lens: np.ndarray, host_lens: np.ndarray, query_len: int
) -> np.ndarray:
    """Do not let a graph-fill context outrun the scheduler's CPU seq len.

    ``host_lens`` is the committed prefix. ``query_len`` is the tokens this
    step adds (1 for decode, the draft width for verify). A CUDA-graph buffer
    can still say 10240 after replay, and a stale ``req_to_token`` row can
    look full, so the CPU length is the bound the indexer is allowed to walk.
    """

    context = np.asarray(context_lens, dtype=np.int32).copy()
    host = np.asarray(host_lens, dtype=np.int32).reshape(-1)
    n = min(int(context.size), int(host.size))
    if n == 0:
        return context
    fill = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    query_len = max(int(query_len), 1)
    real = (host[:n] > 0) & (host[:n] < fill)
    allow = host[:n] + np.int32(query_len)
    shrink = real & (context[:n] > allow)
    context[:n][shrink] = allow[shrink]
    return context


def real_prefill_seq_lens(
    seq_lens: np.ndarray,
    extend_lens: np.ndarray,
    forward_batch,
) -> np.ndarray:
    """Replace a CUDA-graph seq-len fill with the extend that was written.

    Capture stores ``GLM52_GRAPH_SEQ_LEN_CAPACITY`` in ``seq_lens``. A live
    short prefill then publishes that fill as ``max_seqlen_k``, and the
    indexer calls ``top_k_per_row_prefill`` on a row that only holds the
    extend. A real context, including one longer than the fill, is kept.
    Capture sets ``_atom_glm52_keep_graph_fill`` and is kept as well.
    """

    seq = np.asarray(seq_lens, dtype=np.int32).copy()
    extend = np.asarray(extend_lens, dtype=np.int32).reshape(-1)
    if seq.size == 0 or getattr(forward_batch, "_atom_glm52_keep_graph_fill", False):
        return seq
    fill = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    n = min(int(seq.size), int(extend.size))
    if n == 0:
        return seq
    bogus = seq[:n] == fill
    if not np.any(bogus):
        return seq
    prefix = get_extend_prefix_lens_cpu(forward_batch, n)
    if prefix is not None:
        prefix = np.asarray(prefix, dtype=np.int32).reshape(-1)[:n]
        use_prefix = (
            bogus
            & (prefix >= 0)
            & (prefix < fill)
            & (extend[:n] > 0)
            & (extend[:n] < fill)
        )
        seq[:n][use_prefix] = prefix[use_prefix] + extend[:n][use_prefix]
        bogus = seq[:n] == fill
    short = bogus & (extend[:n] > 0) & (extend[:n] < fill)
    seq[:n][short] = extend[:n][short]
    return seq


def _cap_context_to_written_tokens(
    context_lens: np.ndarray,
    forward_batch,
    req_to_token_pool,
) -> np.ndarray:
    """Do not walk ``req_to_token`` past the tokens this request has stored.

    Slot 0 is the dummy pad. A live row is a packed prefix of real cache
    locations followed by zeros. A CUDA-graph fill of 10240 keeps reading
    that tail, and those page ids are what the verify indexer faults on.
    """

    context = np.asarray(context_lens, dtype=np.int32).copy()
    if (
        context.size == 0
        or req_to_token_pool is None
        or getattr(forward_batch, "_atom_glm52_keep_graph_fill", False)
    ):
        return context
    table = getattr(req_to_token_pool, "req_to_token", None)
    indices = getattr(forward_batch, "req_pool_indices", None)
    if not torch.is_tensor(table) or not torch.is_tensor(indices):
        return context
    bs = int(context.size)
    if int(indices.numel()) < bs:
        return context
    limit = int(context.max(initial=0))
    if limit <= 0:
        return context
    limit = min(limit, int(table.shape[1]))
    rows = table[indices[:bs].to(device=table.device, dtype=torch.long), :limit]
    filled = rows != 0
    first_zero = filled.to(torch.int32).argmin(dim=1)
    counts = torch.where(
        filled.all(dim=1),
        torch.full_like(first_zero, limit),
        first_zero,
    )
    counts_np = counts.detach().cpu().numpy().astype(np.int32)
    # A claim at the graph fill is not a context. A written prefix is the
    # tokens prefill stored; an empty row has no such tokens (capture keeps
    # the fill only when ``_atom_glm52_keep_graph_fill`` is set). Walking the
    # fill on either row is the verify/draft indexer fault.
    capped = context.copy()
    fill = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    # A positive written prefix is the KV this row actually holds, even when
    # the claim is shorter than the graph fill. An empty row is only shortened
    # at the fill: capture stages a 4-token verify on a still-empty row.
    shrink = (counts_np < context) & ((counts_np > 0) | (context >= fill))
    capped[shrink] = np.maximum(counts_np[shrink], 1)
    if np.any(shrink):
        logger.warning(
            "GLM capped %d rows from max claimed %d to max written %d",
            int(np.sum(shrink)),
            int(context.max()),
            int(capped.max()),
        )
    return capped.astype(np.int32)


def clamp_published_indexer_context(metadata, forward_batch, req_to_token_pool) -> bool:
    """Shrink a published decode context that still carries the graph fill.

    Replay can hand the indexer the capture buffer after the host length was
    already copied in. The kernel reads this tensor, not the numpy that built
    it, so the clamp has to land here.
    """

    context = getattr(metadata, "context_lens", None)
    if not torch.is_tensor(context) or int(context.numel()) == 0:
        return True
    if getattr(forward_batch, "_atom_glm52_keep_graph_fill", False):
        return True
    try:
        claimed = (
            context.detach().to(dtype=torch.int32, device="cpu").reshape(-1).numpy()
        )
    except Exception as exc:
        # HIP refuses a host copy while the capture stream is recording.
        # AcceleratorError is not always a RuntimeError on this build.
        if "capturing" not in str(exc):
            raise
        return False
    query_len = int(getattr(metadata, "max_seqlen_q", 1) or 1)
    capped = _cap_context_to_host_seq(
        claimed, get_seq_lens_cpu(forward_batch, int(claimed.size)), query_len
    )
    if not np.array_equal(capped, claimed):
        logger.warning(
            "GLM host-capped indexer context from %s to %s",
            int(claimed.max(initial=0)),
            int(capped.max(initial=0)),
        )
    capped = _cap_context_to_written_tokens(capped, forward_batch, req_to_token_pool)
    fill = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    if int(capped.max(initial=0)) >= fill and not getattr(
        clamp_published_indexer_context, "_logged_fill", False
    ):
        clamp_published_indexer_context._logged_fill = True
        logger.warning(
            "GLM indexer context still at the graph fill: claimed_max=%s capped_max=%s rows=%s",
            int(claimed.max(initial=0)),
            int(capped.max(initial=0)),
            int(capped.size),
        )
    if np.array_equal(capped, claimed):
        return True
    context.copy_(
        torch.from_numpy(capped)
        .to(device=context.device, dtype=context.dtype)
        .reshape(context.shape)
    )
    if hasattr(metadata, "max_seqlen_k"):
        metadata.max_seqlen_k = int(max(int(capped.max(initial=1)), 1))
    return True


def mask_oob_indexer_pages(metadata, token_to_kv_pool) -> None:
    """Drop page ids and slots that are outside this rank's KV allocation.

    The verify indexer faults when a block-table entry is a token slot used as
    a page id, or a slot past ``token_to_kv_pool.size``. In-range ids are left
    alone.
    """

    if token_to_kv_pool is None:
        return
    n_slots = int(getattr(token_to_kv_pool, "size", 0) or 0)
    if n_slots <= 0:
        return
    page = max(1, int(getattr(token_to_kv_pool, "page_size", 1) or 1))
    n_pages = (n_slots + page - 1) // page
    block_tables = getattr(metadata, "block_tables", None)
    if torch.is_tensor(block_tables) and int(block_tables.numel()):
        oob = block_tables >= n_pages
        # A negative page is not a cache line either.
        oob = oob | (block_tables < 0)
        if bool(oob.any().item()):
            if not getattr(mask_oob_indexer_pages, "_logged", False):
                mask_oob_indexer_pages._logged = True
                logger.warning(
                    "GLM indexer masked %d block-table entries at or past %d pages",
                    int(oob.sum().item()),
                    n_pages,
                )
            block_tables.masked_fill_(oob, 0)
    slot_mapping = getattr(metadata, "slot_mapping", None)
    if torch.is_tensor(slot_mapping) and int(slot_mapping.numel()):
        bad_slot = slot_mapping >= n_slots
        if bool(bad_slot.any().item()):
            if not getattr(mask_oob_indexer_pages, "_logged_slot", False):
                mask_oob_indexer_pages._logged_slot = True
                logger.warning(
                    "GLM indexer masked %d slots at or past kv size %d",
                    int(bad_slot.sum().item()),
                    n_slots,
                )
            slot_mapping.masked_fill_(bad_slot, -1)


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

    committed = _drop_unconfirmed_graph_fill(
        get_seq_lens_cpu(forward_batch, bs), forward_batch
    )
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
                from_seq = int(effective[row])
                from_pos = int(pos_rows[idx]) + 1
                effective[row] = _prefer_real_length(from_seq, from_pos)
    return effective


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


def _make_decode_slot_mapping(
    out_cache_loc: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(out_cache_loc):
        return out_cache_loc[:batch_size].to(device=device, dtype=torch.int64)
    return torch.zeros(batch_size, dtype=torch.int64, device=device)


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


def _maybe_apply_pcp_prefill_reindex(
    md,
    *,
    sparse_counts: np.ndarray,
    total_tokens: int,
    topk: int,
    token_to_kv_pool,
    atom_config,
    dtype_q,
) -> None:
    """Align sparse-prefill metadata with the query rows owned by this PCP rank."""
    try:
        from atom.distributed.pcp_utils import (
            get_pcp_world_size,
            pcp_is_enabled,
            pcp_pad_dense,
            pcp_pad_len,
            pcp_round_robin_query_indices,
        )
    except ImportError:
        return

    if not pcp_is_enabled():
        return

    device = md.slot_mapping.device
    pcp_size = get_pcp_world_size()
    padded_total = pcp_pad_len(int(total_tokens), pcp_size)
    n_pad = padded_total - int(total_tokens)
    owned_q = pcp_round_robin_query_indices(padded_total, pcp_size).to(device)
    n_owned = int(owned_q.shape[0])

    md.cu_seqlen_ks = pcp_pad_dense(md.cu_seqlen_ks, n_pad)[owned_q].contiguous()
    md.cu_seqlen_ke = pcp_pad_dense(md.cu_seqlen_ke, n_pad)[owned_q].contiguous()
    md.batch_id_per_q_token = pcp_pad_dense(md.batch_id_per_q_token, n_pad)[
        owned_q
    ].contiguous()
    md.sparse_cu_seqlens_q = torch.arange(n_owned + 1, dtype=torch.int32, device=device)

    counts = torch.as_tensor(sparse_counts, dtype=torch.int64, device=device)
    owned_counts = pcp_pad_dense(counts, n_pad)[owned_q]
    owned_counts = torch.clamp(owned_counts, max=int(topk))
    sparse_kv_indptr = torch.zeros(n_owned + 1, dtype=torch.int32, device=device)
    sparse_kv_indptr[1:] = torch.cumsum(owned_counts, dim=0).to(torch.int32)
    md.sparse_kv_indptr = sparse_kv_indptr
    md.sparse_kv_last_page_lens = torch.ones(n_owned, dtype=torch.int32, device=device)

    sparse_work = make_mla_work_buffers(
        cu_seqlens_q=md.sparse_cu_seqlens_q,
        kv_indptr=md.sparse_kv_indptr,
        kv_last_page_lens=md.sparse_kv_last_page_lens,
        num_heads=local_num_attention_heads(atom_config),
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        page_size=attention_page_size(token_to_kv_pool),
    )
    for key, value in sparse_work.items():
        setattr(md, f"sparse_prefill_{key}", value)

    if int(total_tokens) > 0:
        owned_clamped = torch.clamp(owned_q, max=int(total_tokens) - 1)
        md.slot_mapping_owned = md.slot_mapping[owned_clamped].contiguous()
    else:
        md.slot_mapping_owned = md.slot_mapping[:0].contiguous()


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
# Bind SGLang KV pool views to ATOM GLM-5.2 sparse MLA modules.


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
# Multi-token MTP metadata shared by target_verify and draft_extend.


_INDEX_LOGITS_WORKSPACE = None


def _index_logits_workspace(device, width: int):
    """One decode logits plane, shared by every GLM indexer layer.

    Sized to the logits budget, not the live batch, and kept for the process.
    A CUDA graph records the plane's address, so a later bucket must not
    allocate a new one.
    """

    global _INDEX_LOGITS_WORKSPACE
    device = torch.device(device)
    current = _INDEX_LOGITS_WORKSPACE
    if current is not None and current[1] == int(width) and current[2] == device:
        return current[0]
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

    # ``for_rows`` caps the plane at the budget. Passing the width as the row
    # count asks for that cap instead of a one-row plane.
    workspace = LogitsWorkspace.for_rows(max(int(width), 1), int(width), device)
    _INDEX_LOGITS_WORKSPACE = (workspace, int(width), device)
    return workspace


def _publish_fp8_decode_index_rows(md, atom_config) -> None:
    """FP8 decode rows the native indexer scores after main's ``score_topk``.

    One request's ``context_lens`` / ``block_tables`` row, ``max_seqlen_q``
    query rows sharing it. DCP decode never reads these; it returns first.
    """

    context_lens = getattr(md, "context_lens", None)
    block_tables = getattr(md, "block_tables", None)
    if not torch.is_tensor(context_lens) or not torch.is_tensor(block_tables):
        return
    if int(context_lens.shape[0]) == 0:
        return
    from atom.model_ops.indexer.rows import fp8_rows

    next_n = int(getattr(md, "max_seqlen_q", 1) or 1)
    width = int(getattr(atom_config, "max_model_len", 0) or 0)
    if width <= 0:
        width = int(getattr(md, "max_seqlen_k", 1) or 1)
    md.index_rows = fp8_rows(
        context_lens,
        width,
        _index_logits_workspace(context_lens.device, width),
        block_tables,
        next_n=next_n,
    )


def _publish_fp8_prefill_index_rows(md, atom_config) -> None:
    """FP8 prefill rows against the requests' gathered keys.

    ``cu_seqlen_ke - cu_seqlen_ks`` is each query row's visible key count.
    Runs after PCP reindex so the published bounds are the ones attention uses.
    """

    starts = getattr(md, "cu_seqlen_ks", None)
    ends = getattr(md, "cu_seqlen_ke", None)
    if not torch.is_tensor(starts) or not torch.is_tensor(ends):
        return
    if int(starts.shape[0]) == 0:
        return
    from atom.distributed.dcp_utils import get_dcp_rank
    from atom.model_ops.indexer.rows import ConcatKeys, DcpSplit, concat_rows

    world = int(getattr(atom_config, "decode_context_parallel_size", 1) or 1)
    rank = get_dcp_rank() if world > 1 else 0
    row_ends = ends - starts
    md.index_rows = concat_rows(
        row_ends,
        ConcatKeys(
            block_tables=md.block_tables,
            cu_committed=md.cu_seqlens_k if md.has_cached else md.cu_seqlens_q,
            total=int(md.total_kv),
            row_starts=starts,
        ),
        DcpSplit.of(int(row_ends.shape[0]), rank, world),
    )


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
    capped_context = _cap_context_to_written_tokens(
        context_lens_np, forward_batch, req_to_token_pool
    )
    prefix_lens_np = np.maximum(
        prefix_lens_np - (context_lens_np - capped_context), 0
    ).astype(np.int32)
    context_lens_np = capped_context
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
    batch_id_per_q_token = torch.repeat_interleave(
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
        batch_id_per_q_token=batch_id_per_q_token,
        **work,
    )
    md.dtype_q = dtype_q
    md.sparse_kv_last_page_lens = sparse_kv_last_page_lens
    for key, value in sparse_mtp_work.items():
        setattr(md, key, value)
    _publish_fp8_decode_index_rows(md, atom_config)

    return md


# --- target_verify.py ---
# Target verify metadata — target pool, bs×K query rows.


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
    seq_lens = _drop_unconfirmed_graph_fill(
        get_seq_lens_cpu(forward_batch, bs), forward_batch
    )
    if use_positions:
        # Capture leaves positions at ``capacity - draft``. A shorter committed
        # seq_lens is the context prefill actually wrote.
        for row in range(int(prefix_lens.size)):
            from_pos = int(prefix_lens[row]) + int(draft_token_num)
            from_seq = int(seq_lens[row]) if row < int(seq_lens.size) else from_pos
            context = _prefer_real_length(from_pos, from_seq)
            prefix_lens[row] = max(0, context - int(draft_token_num))
        context_lens = prefix_lens + draft_token_num
    else:
        position_context_lens = prefix_lens + draft_token_num
        context_lens = np.array(
            [
                _prefer_real_length(int(seq_lens[row]), int(position_context_lens[row]))
                for row in range(bs)
            ],
            dtype=np.int32,
        )
        prefix_lens = context_lens - draft_token_num
    return prefix_lens, context_lens


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
# Draft forward decode metadata — draft pool, 1 tok/query, multi-step sub_step.


def get_draft_decode_sub_step(forward_batch) -> int:
    return int(getattr(forward_batch, DRAFT_SUB_STEP_ATTR, 0) or 0)


def set_draft_decode_sub_step(forward_batch, sub_step: int) -> None:
    setattr(forward_batch, DRAFT_SUB_STEP_ATTR, int(sub_step))


def clear_draft_decode_sub_step(forward_batch) -> None:
    if hasattr(forward_batch, DRAFT_SUB_STEP_ATTR):
        delattr(forward_batch, DRAFT_SUB_STEP_ATTR)


def is_draft_decode_metadata(forward_batch) -> bool:
    return hasattr(forward_batch, DRAFT_SUB_STEP_ATTR)


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
    context_lens_np = _cap_context_to_written_tokens(
        context_lens_np, forward_batch, req_to_token_pool
    )
    # The table gather adds ``sub_step + 1`` itself. Feed it the same committed
    # length the resolver just published, not the raw graph-fill seq_lens.
    committed_np = np.maximum(context_lens_np - int(sub_step) - 1, 0).astype(np.int32)
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
        page_size=1,
    )
    block_tables = build_draft_decode_token_table(
        forward_batch,
        req_to_token_pool,
        bs=bs,
        seq_lens_np=committed_np,
        sub_step=sub_step,
        num_steps=num_steps,
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
    batch_id_per_q_token = torch.arange(bs, dtype=torch.int32, device=device)

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

    slot_mapping = _make_decode_slot_mapping(
        getattr(forward_batch, "out_cache_loc", None),
        batch_size=bs,
        device=device,
    )

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
        batch_id_per_q_token=batch_id_per_q_token,
        **work,
    )
    md.dtype_q = dtype_q
    md.sparse_kv_last_page_lens = sparse_kv_last_page_lens
    for key, value in sparse_mtp_work.items():
        setattr(md, key, value)
    _publish_fp8_decode_index_rows(md, atom_config)

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
    seq_lens = None
    if spec_kv is not None:
        kv_indptr, kv_indices, bs = spec_kv
        raw_seq_lens = (
            (kv_indptr[1:] - kv_indptr[:-1]).detach().cpu().numpy().astype(np.int32)
        )
        seq_lens = _drop_unconfirmed_graph_fill(raw_seq_lens, forward_batch)
        seq_lens = _cap_context_to_written_tokens(
            seq_lens, forward_batch, req_to_token_pool
        )
        if not np.array_equal(seq_lens, raw_seq_lens):
            # The spec table still describes the graph fill. Rebuild from the
            # confirmed length so the indexer and the indptr agree.
            spec_kv = None
    if spec_kv is None:
        if seq_lens is None:
            seq_lens = _drop_unconfirmed_graph_fill(
                get_seq_lens_cpu(forward_batch, bs), forward_batch
            )
            seq_lens = _cap_context_to_written_tokens(
                seq_lens, forward_batch, req_to_token_pool
            )
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
    else:
        block_tables = build_token_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=None,
            page_size=page_size,
        )

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

    slot_mapping = _make_decode_slot_mapping(
        getattr(forward_batch, "out_cache_loc", None),
        batch_size=bs,
        device=device,
    )

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
    _publish_fp8_decode_index_rows(md, atom_config)
    return md


# --- draft_extend.py ---
# Draft extend metadata — draft pool, DRAFT_EXTEND_V2 bs×K fill after verify.


def resolve_draft_extend_lens(
    forward_batch,
    positions: torch.Tensor,
    bs: int,
    draft_token_num: int,
):
    """Return prefix and total KV lengths for DRAFT_EXTEND_V2."""
    draft_token_num = int(draft_token_num)
    seq_lens = _drop_unconfirmed_graph_fill(
        get_seq_lens_cpu(forward_batch, bs), forward_batch
    )
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

    # A capture-fill position must not hide a real committed seq_lens, and a
    # capture-fill seq_lens must not widen a real position. Either one makes
    # the draft indexer read past the KV that prefill wrote.
    for row in range(int(prefix_lens.size)):
        from_pos = int(prefix_lens[row]) + int(draft_token_num)
        from_seq = int(seq_lens[row]) if row < int(seq_lens.size) else from_pos
        context = _prefer_real_length(from_pos, from_seq)
        prefix_lens[row] = max(0, context - int(draft_token_num))
    context_lens = (prefix_lens + draft_token_num).astype(np.int32)
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
# Prefill metadata — target prefill and draft_extend_for_prefill path.


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
    seq_lens = real_prefill_seq_lens(seq_lens, extend_lens, forward_batch)
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

    md.run_sparse = md.max_seqlen_k > topk
    if md.run_sparse:
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
        md.batch_id_per_q_token = torch.repeat_interleave(
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
        _maybe_apply_pcp_prefill_reindex(
            md,
            sparse_counts=sparse_counts,
            total_tokens=total_tokens,
            topk=topk,
            token_to_kv_pool=token_to_kv_pool,
            atom_config=atom_config,
            dtype_q=dtype_q,
        )
        _publish_fp8_prefill_index_rows(md, atom_config)
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
    seq_lens = real_prefill_seq_lens(seq_lens, extend_lens, forward_batch)
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

    md.run_sparse = md.max_seqlen_k > topk
    if md.run_sparse:
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
        md.batch_id_per_q_token = torch.repeat_interleave(
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
        _maybe_apply_pcp_prefill_reindex(
            md,
            sparse_counts=sparse_counts,
            total_tokens=total_tokens,
            topk=topk,
            token_to_kv_pool=token_to_kv_pool,
            atom_config=atom_config,
            dtype_q=dtype_q,
        )
        _publish_fp8_prefill_index_rows(md, atom_config)
    else:
        ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=max(1, total_tokens),
            topk=topk,
            device=device,
        )

    return md


# --- decode_graph.py ---
# Target decode CUDA-graph metadata buffers.


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

    device = forward_batch.seq_lens.device
    bs = int(forward_batch.batch_size)
    seq_lens_np = _drop_unconfirmed_graph_fill(
        get_seq_lens_cpu(forward_batch, bs), forward_batch
    )
    if torch.is_tensor(positions) and int(positions.numel()) > 0:
        pos_rows = positions.detach().cpu().numpy().astype(np.int32).reshape(-1)
        for row in range(int(seq_lens_np.size)):
            if row < int(pos_rows.size):
                seq_lens_np[row] = _prefer_real_length(
                    int(seq_lens_np[row]), int(pos_rows[row]) + 1
                )
    seq_lens_np = _cap_context_to_written_tokens(
        seq_lens_np, forward_batch, req_to_token_pool
    )
    seq_lens = torch.from_numpy(np.ascontiguousarray(seq_lens_np)).to(
        device=device, dtype=torch.int32
    )
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
    _publish_fp8_decode_index_rows(md, atom_config)
    return md


# --- dispatcher.py ---
# Forward-mode router: target_verify / draft_extend / draft_decode / prefill.


def build_atom_glm52_attention_metadata_from_sglang(
    forward_batch,
    positions,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    if getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)():
        # Positions are the draft-token locations. seq_lens can still hold the
        # CUDA-graph fill (GLM52_GRAPH_SEQ_LEN_CAPACITY) on an uncaptured batch,
        # and max(seq_lens, positions) then walks req_to_token past the tokens
        # that were written. That is an HSA aperture in the verify indexer.
        return build_mtp_verify_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
            use_positions=True,
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
