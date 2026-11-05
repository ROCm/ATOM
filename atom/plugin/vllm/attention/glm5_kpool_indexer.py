# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash pooled (k-pool) sparse indexer for the vllm-atom backend.

The native pooled indexer lives in `atom/model_ops/glm5_next/indexer.py` and
reads ATOM's own forward context. This is the same algorithm driven by vLLM's
metadata instead: the compression, cache writes, pooled scoring, pooled top-k
and pool expansion are the *same* kernels (`atom/model_ops/glm5_next/kpool.py`),
so the two paths cannot drift numerically -- only the metadata plumbing differs.

Three things differ from the native path and account for all the extra code:

* vLLM batches prefill and decode together, where ATOM's scheduler runs one
  class per step. Rows are split by `num_decode_tokens`, exactly as the dense
  plugin indexer (`layer_sparse_mla.sparse_attn_indexer_plugin_mode`) does.
* The pooled cache is allocated by vLLM's block pool, one row per pool, which
  the indexer's `MLAAttentionSpec.tokens_per_state` declares.
* The per-request tail buffer is keyed on the KDA state slot, which lives in
  the KDA layer's metadata rather than in an ATOM forward context.
"""

from __future__ import annotations

import logging
import os

import torch
from aiter import dtypes
from aiter.ops.cache import (
    cp_gather_indexer_k_quant_cache,
    indexer_k_quant_and_cache,
)
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from atom.model_ops.glm5_next import kpool
from atom.model_ops.glm5_next.indexer import (
    _kpool_request_index,
    _kpool_write_completed_pools,
)
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")


def _kda_state_slots(attn_metadata_dict, kda_layer_name: str, num_reqs: int):
    """Per-request KDA state slot ids, which is what the tail buffer is keyed on.

    vLLM orders the non-spec batch decodes-first, the same order the indexer
    metadata uses, so one slice serves both classes.
    """
    kda_meta = attn_metadata_dict.get(kda_layer_name)
    slots = getattr(kda_meta, "non_spec_state_indices_tensor", None)
    if slots is None:
        return None
    return slots[:num_reqs].to(torch.int64)


def _pooled_cache_view(kv_cache: torch.Tensor, pool_rows: int) -> torch.Tensor:
    """[num_blocks, pool_rows, row_bytes], which the aiter kernels index by row."""
    if kv_cache.dim() == 3 and kv_cache.shape[1] == pool_rows:
        return kv_cache
    return kv_cache.view(-1, pool_rows, kv_cache.shape[-1])


# One pooled tail buffer per indexer layer, kept for the process. It lives here,
# not on the module, because the module's forward is traced: a tensor attribute
# read there is a graph input whose pointer the capture bakes in, while the op
# body is a graph break that reads this dict eagerly.
_TAIL_CACHES: dict[str, torch.Tensor] = {}


def allocate_tail_cache(
    k_cache_prefix: str,
    num_state_slots: int,
    index_kpool: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Allocate the per-state-slot buffer holding each request's open pool.

    Called when the KDA layer whose slots key it is bound to its state
    (`Glm5NextKDAAttentionVllm.bind_kv_cache`), because vLLM binds the KV cache
    BEFORE it warms up kernels and captures CUDA graphs. Allocated lazily from
    the first forward instead, it would be created inside a capture: the
    zero-fill would be recorded into the graph and rerun on every replay,
    silently dropping every request's open pool each step.

    Sized off the KDA state pool and keyed by its slot ids, rather than ridden
    into the KDA state page as a third plane: that page is shared with attention
    and its size decides the manager block size, so widening it moves numbers
    that are hard to see moving. A side buffer costs
    `num_state_slots * 2 * kpool * head_dim` bf16 per indexer layer -- tens of
    MB in total -- and follows a request only while that request keeps its slot,
    which is why prefix caching and speculative decoding are refused: both move
    mamba state between slots and neither migration knows this buffer exists.
    Rebinding (a KV cache re-initialization) replaces the buffer.
    """
    _TAIL_CACHES[k_cache_prefix] = torch.zeros(
        num_state_slots,
        2,
        index_kpool,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    logger.info(
        "[glm5 kpool] allocated tail buffer for %s: %d slots x %d x %d",
        k_cache_prefix,
        num_state_slots,
        index_kpool,
        head_dim,
    )


def _tail_cache(k_cache_prefix: str) -> torch.Tensor:
    cached = _TAIL_CACHES.get(k_cache_prefix)
    if cached is None:
        raise RuntimeError(
            f"pooled tail buffer for {k_cache_prefix!r} was never allocated; it "
            "is created when the KDA state is bound, and allocating it here "
            "could land inside a CUDA graph capture"
        )
    return cached


def sparse_attn_indexer_kpool_plugin_mode(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kda_layer_name: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    weights: torch.Tensor,
    compress_ape: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    head_dim: int,
    max_model_len: int,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    from vllm.forward_context import get_forward_context as get_vllm_forward_context
    from vllm.forward_context import is_forward_context_available

    from atom.plugin.vllm.attention.layer_sparse_mla import (
        _get_sparse_mla_metadata,
        triton_convert_req_index_to_global_index,
    )

    result = weights.to(dtype=torch.float32, copy=True)
    if not is_forward_context_available():
        return result
    attn_metadata_dict = get_vllm_forward_context().attn_metadata
    if attn_metadata_dict is None or k_cache_prefix not in attn_metadata_dict:
        # Profile / dummy run: the caches do not exist yet.
        return result
    indexer_meta = attn_metadata_dict[k_cache_prefix]
    if indexer_meta is None:
        return result
    sparse_meta = _get_sparse_mla_metadata(attn_metadata_dict, k_cache_prefix)
    if sparse_meta is None:
        raise RuntimeError(
            f"Sparse MLA metadata not found for indexer cache {k_cache_prefix!r}"
        )

    device = hidden_states.device
    pool_rows = int(indexer_meta.rows_per_block)
    if indexer_meta.tokens_per_state != index_kpool or pool_rows <= 0:
        raise RuntimeError(
            "pooled indexer needs the KV cache spec to carry "
            f"tokens_per_state={index_kpool}; the builder saw "
            f"tokens_per_state={indexer_meta.tokens_per_state}, "
            f"rows_per_block={pool_rows}"
        )
    kv_cache = _pooled_cache_view(kv_cache, pool_rows)
    select_k = topk_tokens // index_kpool
    # `hidden_states` is the PADDED batch under a captured graph; every count
    # that indexes metadata has to come from num_actual_tokens instead, or the
    # per-request expansion below runs off the end of query_start_loc.
    n_tokens = hidden_states.shape[0]
    n_actual = int(indexer_meta.num_actual_tokens)
    if n_actual == 0:
        return result
    num_reqs = indexer_meta.num_reqs
    state_slots = _kda_state_slots(attn_metadata_dict, kda_layer_name, num_reqs)
    tail_cache = _tail_cache(k_cache_prefix)
    if state_slots is None:
        raise RuntimeError(
            "pooled indexer needs the KDA state slots (tail buffer key); "
            f"layer {kda_layer_name!r} published none"
        )

    topk_indices = torch.full(
        (n_tokens, topk_out_width), -1, dtype=torch.int32, device=device
    )

    num_decode_tokens = indexer_meta.num_decode_tokens
    has_decode = indexer_meta.num_decodes > 0
    has_prefill = indexer_meta.num_prefills > 0

    if os.getenv("ATOM_GLM5_KPOOL_CHECK") == "1" and not (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    ):
        # Debug-only, and only here: every index below is consumed by a Triton
        # kernel, where an out-of-range slot is an illegal access reported at
        # some later launch rather than at its cause. Reading the extremes costs
        # a device sync, which CUDA-graph capture forbids -- hence the switch.
        lo = int(state_slots.min().item())
        hi = int(state_slots.max().item())
        blocks = kv_cache.shape[0]
        bt_hi = -1
        if has_decode and indexer_meta.decode is not None:
            bt_hi = int(indexer_meta.decode.block_table.max().item())
        logger.info(
            "[glm5 kpool] tail_slots=[%d,%d] tail_cache=%d pool_rows=%d "
            "cache_blocks=%d decode_bt_max=%d reqs=%d dec=%d pre=%d",
            lo,
            hi,
            tail_cache.shape[0],
            pool_rows,
            blocks,
            bt_hi,
            num_reqs,
            indexer_meta.num_decodes,
            indexer_meta.num_prefills,
        )
        if hi >= tail_cache.shape[0]:
            raise RuntimeError(
                f"pooled tail buffer holds {tail_cache.shape[0]} slots but the "
                f"KDA metadata handed out slot {hi}"
            )

    if has_decode:
        decode_meta = indexer_meta.decode
        bs = indexer_meta.num_decodes
        if num_decode_tokens != bs:
            raise NotImplementedError(
                "GLM-5.3-Flash kpool does not support speculative decode; "
                f"got {num_decode_tokens} decode tokens for {bs} requests"
            )
        dec_slots = state_slots[:bs]
        pos = positions[:bs].to(torch.int64)
        pooled = kpool.kpool_decode_stash_and_pool(
            tail_cache,
            k[:bs],
            gate_score[:bs],
            positions[:bs],
            dec_slots,
            compress_ape,
            index_kpool,
        )
        # The write is gated by the slot: -1 for a pool that has not closed,
        # which the aiter kernel skips. Same rule as the builder's compressed
        # slot mapping, computed here because only this call knows the pooled
        # rows it just produced.
        closes = ((pos % index_kpool) == (index_kpool - 1)) & (dec_slots >= 0)
        pool_ids = torch.where(closes, pos // index_kpool, torch.full_like(pos, -1))
        slots = kpool.pool_slot_mapping(
            decode_meta.block_table,
            pool_ids,
            torch.arange(bs, device=device, dtype=torch.int64),
            pool_rows,
        )
        indexer_k_quant_and_cache(
            pooled, kv_cache, slots, head_dim, scale_fmt, preshuffle=True
        )

        seq_lens = decode_meta.seq_lens[:bs]
        pool_ctx = (seq_lens.to(torch.int32) // index_kpool).contiguous()
        pool_max_len = -(-max_model_len // index_kpool)
        logits = torch.empty((bs, pool_max_len), dtype=torch.float32, device=device)
        deepgemm_fp8_paged_mqa_logits(
            q_fp8[:bs].view(bs, 1, q_fp8.shape[1], head_dim),
            kv_cache.unsqueeze(-2),
            weights[:bs],
            logits,
            pool_ctx,
            decode_meta.block_table,
            pool_max_len,
            KVBlockSize=pool_rows,
            Preshuffle=True,
        )
        pool_topk = torch.empty((bs, select_k), dtype=torch.int32, device=device)
        top_k_per_row_decode(
            logits,
            1,
            pool_ctx,
            pool_topk,
            bs,
            logits.stride(0),
            logits.stride(1),
            k=select_k,
            stable=stable_topk,
        )
        kpool.expand_pools_and_append_tail(
            pool_topk, seq_lens.to(torch.int32), index_kpool, out=topk_indices[:bs]
        )

    if has_prefill:
        # One pass per prefill chunk. The chunks partition the prefill requests
        # (each carries only its own `block_table` rows), so the write, the tail
        # seed and the scoring all have to be sliced the same way -- doing the
        # write once against the first chunk's table would address the wrong
        # blocks for every request in a later chunk.
        qsl = indexer_meta.query_start_loc
        for chunk in indexer_meta.prefill.chunks:
            tok_a, tok_b = chunk.token_start, chunk.token_end
            n_rows = tok_b - tok_a
            if n_rows <= 0:
                continue
            cu_q = qsl[chunk.reqs_start : chunk.reqs_end + 1] - tok_a
            req_idx = _kpool_request_index(cu_q, n_rows)
            slots = state_slots[chunk.reqs_start : chunk.reqs_end]
            rows_k = k[tok_a:tok_b]
            rows_gate = gate_score[tok_a:tok_b]
            rows_pos = positions[tok_a:tok_b]
            # A request resuming after a cached or chunked prefix starts at an
            # absolute position above zero. That offset is native ATOM's
            # `chunk_start`, and it is what tells the write which members of a
            # closing pool must come from the tail buffer rather than from this
            # batch's rows.
            chunk_start = rows_pos[cu_q[:-1].to(torch.int64)].to(torch.int64)

            _kpool_write_completed_pools(
                kv_cache,
                rows_k,
                rows_gate,
                rows_pos,
                chunk.block_table,
                req_idx,
                compress_ape,
                index_kpool,
                head_dim,
                scale_fmt,
                pool_rows,
                chunk_start=chunk_start,
                tail_cache=tail_cache,
                state_slot_idx_in=slots,
                state_slot_idx=slots,
            )
            kpool.kpool_seed_tail(
                tail_cache,
                rows_k,
                rows_gate,
                rows_pos,
                cu_q,
                slots,
                index_kpool,
            )

            # Pooled context per request: complete pools only. The still-open
            # tail is appended by `expand_pools_and_append_tail`, not scored.
            # Both counts come from the builder's CPU-side lengths -- reading
            # them off a device cumsum here would be a D2H sync, which CUDA
            # graph capture forbids.
            pool_cu = chunk.pool_cu_seq_lens
            total_pools = chunk.total_pools
            if pool_cu is None:
                raise RuntimeError(
                    "pooled prefill metadata is missing its pooled cumulative "
                    "lengths; the builder did not see tokens_per_state > 1"
                )
            if total_pools == 0:
                continue

            k_fp8 = torch.empty(
                (total_pools, head_dim), device=device, dtype=dtypes.fp8
            )
            k_scale = torch.empty((total_pools, 1), device=device, dtype=torch.float32)
            cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale.view(dtypes.fp8),
                chunk.block_table,
                pool_cu,
                preshuffle=True,
            )

            row_pos = rows_pos.to(torch.int64)
            pool_ks = pool_cu.to(torch.int64)[req_idx]
            pool_ke = (pool_ks + (row_pos + 1) // index_kpool).to(torch.int32)
            pool_ks = pool_ks.to(torch.int32)
            pool_topk = torch.empty(
                (n_rows, select_k), dtype=torch.int32, device=device
            )

            budget_bytes = envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB * 1024 * 1024
            if budget_bytes > 0 and budget_bytes // (total_pools * 4) < n_rows:
                budget_rows = budget_bytes // (total_pools * 4)
                row_chunk = (
                    (budget_rows // 128) * 128
                    if budget_rows >= 128
                    else 1 << (max(1, budget_rows).bit_length() - 1)
                )
            else:
                row_chunk = n_rows

            for start in range(0, n_rows, row_chunk):
                end = min(start + row_chunk, n_rows)
                row_ks = pool_ks[start:end]
                row_ke = pool_ke[start:end]
                logits = fp8_mqa_logits(
                    Q=q_fp8[tok_a + start : tok_a + end],
                    KV=k_fp8,
                    kv_scales=k_scale.squeeze(-1).contiguous(),
                    weights=weights[tok_a + start : tok_a + end],
                    cu_starts=row_ks,
                    cu_ends=row_ke,
                )
                top_k_per_row_prefill(
                    logits,
                    row_ks,
                    row_ke,
                    pool_topk[start:end],
                    None,
                    end - start,
                    logits.stride(0),
                    logits.stride(1),
                    k=select_k,
                    stable=stable_topk,
                )

            # aiter's prefill top-k reports a pool's place in the chunk-wide
            # gathered run; less its request's first pool it is the request's
            # own (the same correction `model_ops/indexer/score.py` applies).
            # The converter below takes request-LOCAL token indices, so the
            # expansion has to start from request-local pool ids.
            torch.where(
                pool_topk < 0,
                pool_topk,
                pool_topk - pool_ks.unsqueeze(1),
                out=pool_topk,
            )
            kpool.expand_pools_and_append_tail(
                pool_topk,
                (row_pos + 1).to(torch.int32),
                index_kpool,
                out=topk_indices[tok_a:tok_b],
            )

    # Local token indices -> global KV slots, the same conversion and buffer the
    # dense plugin indexer uses; only the row width differs.
    triton_convert_req_index_to_global_index(
        sparse_meta.batch_id_per_q_token.to(dtype=torch.int32),
        sparse_meta.block_table.to(dtype=torch.int32),
        topk_indices[: sparse_meta.num_actual_tokens].to(dtype=torch.int32),
        sparse_meta.paged_kv_indptr,
        sparse_kv_indices_buffer,
        BLOCK_SIZE=sparse_meta.block_size,
        NUM_TOPK_TOKENS=topk_out_width,
    )
    return result


def sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kda_layer_name: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    weights: torch.Tensor,
    compress_ape: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    head_dim: int,
    max_model_len: int,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    return torch.empty_like(weights, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool_plugin_mode",
    op_func=sparse_attn_indexer_kpool_plugin_mode,
    mutates_args=["sparse_kv_indices_buffer", "kv_cache"],
    fake_impl=sparse_attn_indexer_kpool_fake,
)
