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
* Each request's open pool (the tail) lives in a ring that vLLM allocates as
  its own KV cache group (`KpoolTailSpec`, one block per request), the layout
  vLLM's own GLM-5.3-Flash uses. The native kernels address a contiguous
  `[slot, 2, rows, head_dim]` buffer, so each step gathers the batch's rings
  into one and scatters them back.
"""

from __future__ import annotations

import logging

import torch
from aiter import dtypes
from aiter.ops.cache import (
    cp_gather_indexer_k_quant_cache,
    indexer_k_quant_and_cache,
)
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from vllm.models.glm5next.common.attention import Glm5NextTailCache
from vllm.v1.kv_cache_interface import KpoolTailSpec

from atom.model_ops.glm5_next import kpool
from atom.model_ops.glm5_next.geometry import speculative_kpool_history_size
from atom.model_ops.glm5_next.indexer import (
    _kpool_request_index,
    _kpool_write_completed_pools,
)
from atom.model_ops.glm5_next.speculative import (
    build_speculative_pool_candidates,
    update_speculative_kpool_history,
)
from atom.model_ops.sparse_indexer_chunk import sparse_indexer_row_chunk
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")


def tail_ring_rows(index_kpool: int, num_speculative_tokens: int) -> int:
    """Rows of each request's tail ring.

    One pool without speculation. With it, native ATOM's bound
    (`speculative_kpool_history_size`): the verification algorithm the plugin
    shares with native is only correct for a ring at least that deep, which is
    deeper than the `kpool + num_spec` vLLM sizes its own ring for.
    """
    return speculative_kpool_history_size(index_kpool, num_speculative_tokens or None)


class Glm5NextTailRingVllm(Glm5NextTailCache):
    """vLLM's GLM-5.3-Flash tail cache, with native ATOM's ring depth.

    vLLM allocates it as one block per request for the request's lifetime,
    never prefix-caches it, and frees it with the request, which is what lets
    the tail follow a request whatever happens to its KDA state slot, and gives
    an MTP draft layer (which has no KDA state) a tail of its own.
    """

    def get_kv_cache_spec(self, vllm_config) -> KpoolTailSpec:
        ring = tail_ring_rows(self._index_kpool, vllm_config.num_speculative_tokens)
        if self.cache_config.block_size % ring != 0:
            raise ValueError(
                f"GLM-5.3-Flash's tail ring ({ring} rows) must divide the KV "
                f"block size ({self.cache_config.block_size})"
            )
        return KpoolTailSpec(
            block_size=ring,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=ring,
        )


def _tail_ring_view(tail_kv_cache: torch.Tensor, ring: int, head_dim: int):
    """[num_blocks, 2 (K, gate), ring, head_dim], one block per request.

    vLLM pads each tail block to the indexer page and aliases it into the
    indexer layer's tensor, so the block stride is the padded page; only the
    inner three dims are contiguous.
    """
    if (
        tail_kv_cache.dim() != 4
        or tuple(tail_kv_cache.shape[1:]) != (2, ring, head_dim)
        or tuple(tail_kv_cache.stride()[1:]) != (ring * head_dim, head_dim, 1)
    ):
        raise RuntimeError(
            f"tail ring cache is {tuple(tail_kv_cache.shape)} with strides "
            f"{tuple(tail_kv_cache.stride())}; expected [blocks, 2, {ring}, "
            f"{head_dim}] with contiguous inner dims"
        )
    return tail_kv_cache


def _gather_tail_rings(
    tail_view: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: int,
    ring: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the batch's tail rings into one contiguous `[num_reqs, 2, ring, D]`.

    Returns the gathered rings, each request's slot in them (its batch index,
    or -1 for a request with no tokens or no ring, which every native kernel
    skips), and the block each one came from, for `_scatter_tail_rings`.

    The tail group's slot mapping is `block * ring + pos % ring` for every
    token, so a request's block is its first token's slot over `ring`.
    Device-side only: no host sync, so this records into a CUDA graph.
    """
    device = tail_view.device
    starts = query_start_loc[:num_reqs].to(torch.int64)
    has_tokens = query_start_loc[1 : num_reqs + 1] > query_start_loc[:num_reqs]
    first = starts.clamp(max=tail_slot_mapping.numel() - 1)
    tail_slots = tail_slot_mapping.to(torch.int64)[first]
    valid = has_tokens & (tail_slots >= 0)
    blocks = torch.where(valid, tail_slots // ring, torch.zeros_like(tail_slots))
    rings = tail_view.index_select(0, blocks)
    slots = torch.where(
        valid,
        torch.arange(num_reqs, device=device, dtype=torch.int64),
        torch.full_like(blocks, -1),
    )
    return rings, slots, blocks


def _scatter_tail_rings(
    tail_view: torch.Tensor, rings: torch.Tensor, blocks: torch.Tensor
) -> None:
    """Write the gathered rings back to their blocks.

    A request without a ring was gathered from block 0 and skipped by the
    kernels, so it writes block 0's own bytes back. Block 0 is vLLM's null
    block and never a request's, so no live ring shares that index.
    """
    tail_view.index_copy_(0, blocks, rings)


def _pooled_cache_view(kv_cache: torch.Tensor, pool_rows: int) -> torch.Tensor:
    """[num_blocks, pool_rows, row_bytes], which the aiter kernels index by row.

    vLLM already views the cache that way; a different row count means the
    builder's `rows_per_block` and vLLM's allocation disagree, and re-viewing
    the tensor would silently address the wrong rows.
    """
    if kv_cache.dim() != 3 or kv_cache.shape[1] != pool_rows:
        raise RuntimeError(
            f"pooled index cache is viewed as {tuple(kv_cache.shape)}, but the "
            f"indexer metadata pages it at {pool_rows} rows per block"
        )
    return kv_cache


def _prefill_chunk_topk(
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    rows_pos: torch.Tensor,
    req_idx: torch.Tensor,
    pool_cu: torch.Tensor,
    total_pools: int,
    block_table: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    select_k: int,
    stable_topk: bool,
    out: torch.Tensor,
) -> None:
    """Request-local token indices each row of one prefill chunk attends.

    Every row always gets its own open pool (the tail): with `total_pools == 0`
    -- no request in the chunk has closed a pool yet, e.g. a prompt shorter
    than `index_kpool` -- there is nothing to score, but the tail is still all
    the row may attend. Leaving the row at -1 instead would be silently wrong:
    the global-index conversion maps -1 to slot 0, and the plugin path always
    runs sparse MLA.
    """
    device = q_fp8.device
    n_rows = q_fp8.shape[0]
    row_pos = rows_pos.to(torch.int64)
    if total_pools == 0:
        pool_topk = torch.full((n_rows, select_k), -1, dtype=torch.int32, device=device)
    else:
        k_fp8 = torch.empty((total_pools, head_dim), device=device, dtype=dtypes.fp8)
        k_scale = torch.empty((total_pools, 1), device=device, dtype=torch.float32)
        cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),
            block_table,
            pool_cu,
            preshuffle=True,
        )
        pool_ks = pool_cu.to(torch.int64)[req_idx]
        pool_ke = (pool_ks + (row_pos + 1) // index_kpool).to(torch.int32)
        pool_ks = pool_ks.to(torch.int32)
        pool_topk = torch.empty((n_rows, select_k), dtype=torch.int32, device=device)
        # Same row batching as the dense plugin indexer, including the hard
        # 2 GiB bound on one [rows, total_pools] fp32 logits buffer.
        row_chunk = sparse_indexer_row_chunk(
            n_rows, total_pools, envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB
        )
        for start in range(0, n_rows, row_chunk):
            end = min(start + row_chunk, n_rows)
            row_ks = pool_ks[start:end]
            row_ke = pool_ke[start:end]
            logits = fp8_mqa_logits(
                Q=q_fp8[start:end],
                KV=k_fp8,
                kv_scales=k_scale.squeeze(-1).contiguous(),
                weights=weights[start:end],
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
        # gathered run; less its request's first pool it is the request's own
        # (the same correction `model_ops/indexer/score.py` applies). The
        # converter takes request-LOCAL token indices, so the expansion has to
        # start from request-local pool ids.
        torch.where(
            pool_topk < 0, pool_topk, pool_topk - pool_ks.unsqueeze(1), out=pool_topk
        )
    kpool.expand_pools_and_append_tail(
        pool_topk, (row_pos + 1).to(torch.int32), index_kpool, out=out
    )


def sparse_attn_indexer_kpool_plugin_mode(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    tail_prefix: str,
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_ring: int,
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
) -> None:
    from vllm.forward_context import get_forward_context as get_vllm_forward_context
    from vllm.forward_context import is_forward_context_available

    from atom.plugin.vllm.attention.layer_sparse_mla import (
        _get_sparse_mla_metadata,
        triton_convert_req_index_to_global_index,
    )

    if not is_forward_context_available():
        return
    attn_metadata_dict = get_vllm_forward_context().attn_metadata
    if attn_metadata_dict is None or k_cache_prefix not in attn_metadata_dict:
        # Profile / dummy run: the caches do not exist yet.
        return
    indexer_meta = attn_metadata_dict[k_cache_prefix]
    if indexer_meta is None:
        return
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
        return
    num_reqs = indexer_meta.num_reqs
    tail_meta = attn_metadata_dict.get(tail_prefix)
    if tail_meta is None:
        raise RuntimeError(
            f"pooled indexer {k_cache_prefix!r} has no metadata for its tail "
            f"ring {tail_prefix!r}"
        )
    tail_view = _tail_ring_view(tail_kv_cache, tail_ring, head_dim)
    tail_cache, state_slots, tail_blocks = _gather_tail_rings(
        tail_view,
        tail_meta.slot_mapping,
        indexer_meta.query_start_loc,
        num_reqs,
        tail_ring,
    )

    topk_indices = torch.full(
        (n_tokens, topk_out_width), -1, dtype=torch.int32, device=device
    )

    num_decode_tokens = indexer_meta.num_decode_tokens
    has_decode = indexer_meta.num_decodes > 0
    has_prefill = indexer_meta.num_prefills > 0

    if has_decode:
        decode_meta = indexer_meta.decode
        bs = indexer_meta.num_decodes
        # Under speculative decoding a request verifies its accepted token and
        # its drafts in one step, and the builder hands the indexer one decode
        # row per token, each with its own seq_len and its request's block
        # table. Rows, not requests, are what is scored below.
        rows = num_decode_tokens
        dec_slots = state_slots[:bs]
        pos = positions[:rows].to(torch.int64)
        if rows == bs:
            pooled = kpool.kpool_decode_stash_and_pool(
                tail_cache,
                k[:bs],
                gate_score[:bs],
                positions[:bs],
                dec_slots,
                compress_ape,
                index_kpool,
            )
            row_slots = dec_slots
        else:
            # Native ATOM's verification kernels: form the pool ending at each
            # row from this step's rows and the ring, then write the rows into
            # the ring. Rows of a rejected draft are overwritten by the next
            # verification before anything reads them, which the ring's depth
            # (`tail_ring_rows`) guarantees.
            cu_q = indexer_meta.query_start_loc[: bs + 1]
            pooled, row_req = build_speculative_pool_candidates(
                tail_cache,
                k[:rows],
                gate_score[:rows],
                positions[:rows],
                cu_q,
                dec_slots,
                compress_ape,
                index_kpool,
            )
            update_speculative_kpool_history(
                tail_cache,
                k[:rows],
                gate_score[:rows],
                positions[:rows],
                cu_q,
                dec_slots,
                dec_slots,
            )
            row_slots = dec_slots[row_req]
        # The write is gated by the slot: -1 for a pool that has not closed,
        # which the aiter kernel skips. Same rule as the builder's compressed
        # slot mapping, computed here because only this call knows the pooled
        # rows it just produced.
        closes = ((pos % index_kpool) == (index_kpool - 1)) & (row_slots >= 0)
        pool_ids = torch.where(closes, pos // index_kpool, torch.full_like(pos, -1))
        slots = kpool.pool_slot_mapping(
            decode_meta.block_table,
            pool_ids,
            torch.arange(rows, device=device, dtype=torch.int64),
            pool_rows,
        )
        indexer_k_quant_and_cache(
            pooled, kv_cache, slots, head_dim, scale_fmt, preshuffle=True
        )

        seq_lens = decode_meta.seq_lens[:rows]
        pool_ctx = (seq_lens.to(torch.int32) // index_kpool).contiguous()
        pool_max_len = -(-max_model_len // index_kpool)
        logits = torch.empty((rows, pool_max_len), dtype=torch.float32, device=device)
        deepgemm_fp8_paged_mqa_logits(
            q_fp8[:rows].view(rows, 1, q_fp8.shape[1], head_dim),
            kv_cache.unsqueeze(-2),
            weights[:rows],
            logits,
            pool_ctx,
            decode_meta.block_table,
            pool_max_len,
            KVBlockSize=pool_rows,
            Preshuffle=True,
        )
        pool_topk = torch.empty((rows, select_k), dtype=torch.int32, device=device)
        top_k_per_row_decode(
            logits,
            1,
            pool_ctx,
            pool_topk,
            rows,
            logits.stride(0),
            logits.stride(1),
            k=select_k,
            stable=stable_topk,
        )
        kpool.expand_pools_and_append_tail(
            pool_topk, seq_lens.to(torch.int32), index_kpool, out=topk_indices[:rows]
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
            _prefill_chunk_topk(
                kv_cache,
                q_fp8[tok_a:tok_b],
                weights[tok_a:tok_b],
                rows_pos,
                req_idx,
                pool_cu,
                total_pools,
                chunk.block_table,
                index_kpool,
                head_dim,
                select_k,
                stable_topk,
                out=topk_indices[tok_a:tok_b],
            )

    _scatter_tail_rings(tail_view, tail_cache, tail_blocks)

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


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool_plugin_mode",
    op_func=sparse_attn_indexer_kpool_plugin_mode,
    mutates_args=["sparse_kv_indices_buffer", "kv_cache", "tail_kv_cache"],
    fake_impl=lambda *_args, **_kwargs: None,
)
