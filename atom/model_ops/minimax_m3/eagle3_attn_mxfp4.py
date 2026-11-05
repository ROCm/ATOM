# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EAGLE3 draft attention over the MXFP4 KV pool (``--kv-cache-dtype fp4``).

The dense MiniMax-M3 impl (``MiniMaxM3DenseMxfp4PagedAttentionImpl``) for a layer
with no q/k norm. What differs, and only that:

* write: the AITER fused writer always RMS-norms, so it cannot write this layer.
  q and k are rotated by the layer's own ``rotary_emb`` (full-rotary NeoX, the call
  the bf16 / fp8 draft makes), then ``store_kv_mxfp4`` quantizes the bf16 K and the
  raw V into the pages. q stays bf16: it is un-normed, so a unit-scale e4m3 cast
  (no saturation in torch: > 448 is NaN) is unsafe, while ``pa_decode`` scales a
  bf16 q per head row itself.
* decode: ``plan_step_owner=None``. The draft shares the target's metadata object,
  and the draft steps change ``context_lens`` under it, so a once-per-owner plan
  would go stale (see ``dense_pa_mxfp4``).
* prefill: the full-KV gather is only reached through ``require_prefill_slots_written``,
  so a token whose K/V the writer skipped cannot be read back as an unwritten page.

The q_len > 1 fold, the page gather and the layout checks are the dense impl's.
"""

import torch

from atom.model_ops import attention_mha as am
from atom.model_ops.minimax_m3.store_mxfp4 import (
    require_prefill_slots_written,
    store_kv_mxfp4,
)
from atom.utils import envs
from atom.utils.decorators import mark_trace
from atom.utils.forward_context import ForwardContext


class Eagle3Mxfp4PagedAttentionImpl(am.MiniMaxM3DenseMxfp4PagedAttentionImpl):
    """EAGLE3's single norm-free attention layer on MXFP4 pages."""

    def _check_supported(self) -> None:
        from atom.distributed.dcp_utils import get_dcp_world_size
        from atom.distributed.pcp_utils import get_pcp_world_size
        from atom.model_ops.minimax_m3.mxfp4_pages import HEAD_DIM

        def refuse(why: str):
            raise NotImplementedError(
                f"layer {self.layer_num}: EAGLE3 MXFP4 attention {why}"
            )

        if self.kv_cache_dtype != "fp4":
            refuse(f"needs kv_cache_dtype fp4, got {self.kv_cache_dtype}")
        if self.head_dim != HEAD_DIM:
            refuse(f"needs head_dim {HEAD_DIM}, got {self.head_dim}")
        if self.sliding_window != -1 or self.sinks is not None:
            refuse("has no sliding window or sinks")
        if self.alibi_slopes is not None:
            refuse("has no alibi")
        if self.rotary_emb is None:
            refuse("needs the layer's rotary_emb: the writer does not rotate")
        if self.q_norm is not None or self.k_norm is not None:
            # this writer applies no norm; a normed layer is the dense M3 impl's
            refuse("is norm-free; a layer with q/k norm takes the dense M3 impl")
        if (
            self.num_heads % self.num_kv_heads
            or self.num_heads // self.num_kv_heads > 16
        ):
            refuse(
                "needs 1..16 query heads per kv head, got "
                f"{self.num_heads}/{self.num_kv_heads}"
            )
        if envs.ATOM_USE_UNIFIED_ATTN or self.use_flash_layout:
            refuse("does not run with ATOM_USE_UNIFIED_ATTN / flash layout")
        if get_dcp_world_size() > 1 or get_pcp_world_size() > 1:
            # a CP rank writes -1 slots for tokens it does not own, which the
            # prefill gather would read as unwritten pages
            refuse("does not run under decode / prefill context parallelism")

    @mark_trace(prefix="rope_cache", torch_compile=False)
    def rope_cache(self, q, k, v, qkv, position, fwd_ctx: ForwardContext):
        """RoPE on q and k, then K/V quantized into the pages (no norm).

        Returns the parent 7-tuple with the rotated bf16 ``q``; ``k`` / ``v`` are
        returned as given, the readers use the pages.
        """
        if position is None:
            raise ValueError(f"layer {self.layer_num}: RoPE needs positions")
        k_pages, v_pages = self._pages(fwd_ctx)
        # in place on the views of the qkv projection output, as the bf16 draft
        q, k = self.rotary_emb(position, q, k)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        if q.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"layer {self.layer_num}: EAGLE3 MXFP4 attention needs a bf16 qkv, "
                f"got {q.dtype}"
            )
        store_kv_mxfp4(
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            k_pages,
            v_pages,
            fwd_ctx.attn_metadata.slot_mapping,
        )
        # the pa_decode / flash readers take a dense q
        return q.contiguous(), k, v, k_pages, v_pages, None, None

    @mark_trace(prefix="prefill_attention", torch_compile=False)
    def prefill_attention(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        md = fwd_ctx.attn_metadata
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # the check needs the host; a captured prefill would skip it silently
            raise RuntimeError(
                f"layer {self.layer_num}: the EAGLE3 MXFP4 prefill gather cannot "
                "run in a captured graph"
            )
        require_prefill_slots_written(
            md.slot_mapping, md.cu_seqlens_q, q.shape[0], k_cache.shape[0]
        )
        return super().prefill_attention(
            q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
        )

    @mark_trace(prefix="paged_attention_mxfp4", torch_compile=False)
    def _dense_decode(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        md = fwd_ctx.attn_metadata
        query_len = max(1, int(md.max_seqlen_q))
        # q_len tokens a request, for the requests this step runs: a padded step
        # (graph bucket) carries more context_lens / block table rows than that
        num_seqs, ragged = divmod(q.shape[0], query_len)
        if ragged or num_seqs > md.context_lens.shape[0]:
            raise NotImplementedError(
                f"layer {self.layer_num}: EAGLE3 MXFP4 decode needs q_len "
                f"{query_len} tokens for each request, got {q.shape[0]} query "
                f"tokens over {md.context_lens.shape[0]} requests"
            )
        ctx, block_tables = am.dense_decode_rows(
            md.context_lens[:num_seqs], md.block_tables[:num_seqs], query_len
        )
        return am.dense_pa_mxfp4(
            q,
            k_cache,
            v_cache,
            ctx,
            block_tables,
            self.scale,
            plan_step_owner=None,  # the draft's lengths change under one metadata
        )
