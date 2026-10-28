# SPDX-License-Identifier: MIT
"""Model integration for M3's all-head, block-interleaved DCP cache."""

import torch

from atom.config import get_current_atom_config
from atom.distributed.dcp_utils import get_dcp_group, get_dcp_rank, get_dcp_world_size
from atom.model_ops.attention_mha import PagedAttentionImpl, SparseMHAPagedAttentionImpl
from atom.model_ops.dcp_ops import dcp_all_gather_query_heads
from atom.model_ops.minimax_m3.dcp_attention import attention, index_topk
from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows, prefill_rows


class M3DCPProjection:
    """Gather independently sharded projection fields in head order."""

    def __init__(self, tp_size, kv_heads):
        self.world = get_dcp_world_size()
        self.rank = get_dcp_rank()
        self.group = get_dcp_group() if self.world > 1 else None
        if self.world > 1:
            config = get_current_atom_config()
            if self.world != tp_size or kv_heads % tp_size:
                raise ValueError(
                    "M3 DCP currently requires DCP=TP with unreplicated KV heads"
                )
            if (
                config.kv_cache_block_size != 128
                or config.dcp_config.interleave_size != 128
            ):
                raise ValueError("M3 DCP requires block_size=interleave_size=128")
            if config.enable_tbo or config.enable_tbo_decode:
                raise ValueError("M3 DCP does not yet support TBO")
            from atom.utils import envs

            if envs.ATOM_MONO_ENABLE or config.dcp_config.indexer_dcp_only:
                raise ValueError(
                    "M3 DCP requires ATOM_MONO_ENABLE=0 and ATOM_M3_INDEXER_CP=0"
                )

    def gather(self, x, dim):
        if self.world == 1:
            return x
        return dcp_all_gather_query_heads(
            self.group, x.contiguous().view(x.shape[0], -1, dim)
        ).flatten(1)

    def local_output(self, x):
        if self.world == 1:
            return x
        return x.chunk(self.world, dim=-1)[self.rank].contiguous()


def _rows(md, prefill, total):
    if prefill:
        prefix = md.context_lens - (md.cu_seqlens_q[1:] - md.cu_seqlens_q[:-1])
        return prefill_rows(md.cu_seqlens_q, prefix, total)
    return decode_rows(md.context_lens, total)


class MiniMaxM3DCPAttentionImpl(PagedAttentionImpl):
    def dispatch_backend(self, fwd_ctx, q, k, v):
        return self._dcp_attention

    def _dcp_attention(self, q, k, v, kc, vc, ks, vs, fwd_ctx):
        md = fwd_ctx.attn_metadata
        req, lengths = _rows(md, fwd_ctx.context.is_prefill, q.shape[0])
        return attention(
            q,
            (kc, vc, ks, vs),
            md.block_tables,
            req,
            lengths,
            torch.zeros_like(lengths),
            world=get_dcp_world_size(),
            rank=get_dcp_rank(),
            group=get_dcp_group(),
            scale=self.scale,
        )


class MiniMaxM3DCPSparseAttentionImpl(SparseMHAPagedAttentionImpl):
    def dispatch_backend(self, fwd_ctx, q, k, v):
        return self._dcp_attention

    def _dcp_attention(self, q, k, v, kc, vc, ks, vs, fwd_ctx):
        sm = self._sparse_metadata(fwd_ctx)
        if fwd_ctx.context.is_prefill:
            md = sm.prefill
            req, lengths = prefill_rows(md.cu_seqlens_q, md.context_lens, q.shape[0])
        else:
            md = sm.decode
            req, lengths = decode_rows(md.seq_lens, q.shape[0])
        key = ("dcp", tuple(q.shape), tuple(md.block_table.shape))
        cached = self._load_cached_topk(sm, key)
        if cached is None:
            if self._index_q is None:
                raise RuntimeError("M3 DCP top-k reuse has no preceding index layer")
            selected = index_topk(
                self._index_q,
                self.index_cache,
                self.index_scale,
                md.block_table,
                req,
                lengths,
                get_dcp_group(),
                get_dcp_rank(),
                get_dcp_world_size(),
                self.topk,
                self.init_blocks,
                self.local_blocks,
                self.scale,
                max_seq_len=sm.max_seq_len,
            )
            self._store_cached_topk(sm, key, selected)
        else:
            selected = cached
        out = attention(
            q,
            (kc, vc, ks, vs),
            md.block_table,
            req,
            lengths,
            torch.zeros_like(lengths),
            world=get_dcp_world_size(),
            rank=get_dcp_rank(),
            group=get_dcp_group(),
            scale=self.scale,
            topk=selected,
            nvfp4=self.kv_nvfp4,
        )
        self._index_q = None
        self._index_q_cache_key_info = None
        return out
