"""vLLM-specific DeepSeek-V4 model.

This module reuses the native ATOM DeepSeek-V4 implementation
(:mod:`atom.models.deepseek_v4`) unchanged and only layers on the behaviour
that vLLM's graph-mode execution requires. The single vLLM-specific concern is
reconciling the padded CUDA-graph bucket width against the sparse-attention
metadata in the graph-break attention op (see ``DeepseekV4AttentionVllm``).

It follows the same construction-swap pattern as ``qwen3_next``: the
``DeepseekV4ForCausalLM`` subclass temporarily rebinds the module-global
``DeepseekV4Attention`` so the whole model tree is built with the vLLM attention
variant, then restores it.
"""

from typing import Optional

import torch

from atom.models import deepseek_v4 as deepseek_v4_base
from atom.models.deepseek_v4 import (
    DeepseekV4Attention as DeepseekV4AttentionBase,
    DeepseekV4ForCausalLM as DeepseekV4ForCausalLMBase,
    DeepseekV4Model as DeepseekV4ModelBase,
    Indexer as IndexerBase,
)
from atom.utils.forward_context import AttnState, get_forward_context


class IndexerVllm(IndexerBase):
    """DeepSeek-V4 sparse indexer with vLLM mixed-batch decode/prefill split.

    Native ATOM serves decode and prefill in *separate* forward passes, so its
    ``indexer_score_topk`` can route the whole batch by ``is_prefill``: pure
    decode -> fixed-shape paged ``_score_topk_decode``; pure prefill -> dense
    ``_score_topk_prefill``.

    vLLM continuous batching instead produces MIXED batches (chunked-prefill
    rows + running decode rows) that ATOM classifies as PREFILL (batch
    ``max_query_len`` > 1). Routing the decode rows through the dense
    ``fp8_mqa_logits`` prefill path makes its ``[total_tokens, total_committed]``
    logits sum EVERY running seq's committed K into ``total_committed`` —
    quadratic in batch size, which OOMs the GPU at high concurrency.

    Mirror native ATOM by splitting the mixed batch here: vLLM reorders decode
    rows (query_len <= 1[+num_spec]) to the FRONT (``reorder_batch_threshold``),
    so the split is a contiguous prefix. Decode rows take the fixed-shape paged
    path (bounded per-seq, no batch-sum); prefill rows take the dense path whose
    ``total_committed`` the bridge (``_populate_indexer``) now builds over the
    PREFILL sub-batch only. This override lives in the vLLM plugin so native
    ATOM behaviour is untouched.

    Only the PREFILL-classified branch changes. The pure-decode branch is
    byte-for-byte the native call, so the FULL-CUDAGraph-captured decode replay
    path is unaffected; the split runs eager (mixed batches are always
    PIECEWISE, never FULL-captured), introducing no new captured shapes or
    Python branches into any graph.
    """

    def indexer_score_topk(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        topk: int,
    ) -> torch.Tensor:
        fc = get_forward_context()
        indexer_meta = fc.attn_metadata.indexer_meta
        block_tables = fc.attn_metadata.block_tables  # [bs, max_blocks_per_seq]

        if not fc.context.is_prefill:
            # Pure-decode step (AttnState.DECODE): the CUDAGraph-captured
            # fixed-shape paged path. Byte-for-byte the native call.
            return self._score_topk_decode(
                q_fp8, weights, block_tables, indexer_meta, topk
            )  # [total_tokens, topk] int32

        # Prefill-classified step (may be MIXED under continuous batching).
        num_decode_tokens = int(indexer_meta.get("num_decode_tokens", 0))
        if num_decode_tokens == 0:
            # Pure prefill: `total_committed` already spans only prefill seqs.
            return self._score_topk_prefill(
                q_fp8, weights, block_tables, indexer_meta, topk
            )  # [total_tokens, topk] int32

        num_decodes = int(indexer_meta["num_decodes"])
        n_committed_per_seq = indexer_meta["n_committed_per_seq_gpu"]
        # Decode rows: paged fixed-shape logits (bounded per-seq, no batch-sum).
        decode_topk = self._score_topk_decode(
            q_fp8[:num_decode_tokens],
            weights[:num_decode_tokens],
            block_tables[:num_decodes],
            indexer_meta,
            topk,
            next_n=int(indexer_meta["decode_next_n"]),
            n_committed_per_seq=n_committed_per_seq[:num_decodes],
        )  # [num_decode_tokens, topk] int32
        # Prefill rows: dense logits now sized only by the PREFILL seqs'
        # committed K (the bridge builds the committed meta over the prefill
        # sub-batch), so `total_committed` no longer explodes.
        prefill_topk = self._score_topk_prefill(
            q_fp8[num_decode_tokens:],
            weights[num_decode_tokens:],
            block_tables[num_decodes:],
            indexer_meta,
            topk,
        )  # [num_prefill_tokens, topk] int32
        # Decode-first ordering matches the reordered batch row order.
        return torch.cat([decode_topk, prefill_topk], dim=0)

    def _score_topk_decode(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        block_tables: torch.Tensor,  # [bs, max_blocks_per_seq] int32
        indexer_meta: dict,
        topk: int,
        *,
        next_n: Optional[int] = None,
        n_committed_per_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Paged decode top-k, extended for the DECODE slice of a mixed batch.

        Native ATOM only ever calls this for a *pure* decode forward, where the
        step size is the batch-wide ``max_seqlen_q`` and the committed tensor is
        the whole ``indexer_meta["n_committed_per_seq_gpu"]``. Under vLLM
        continuous batching this also runs on the leading decode slice of a
        MIXED batch, where neither holds: ``max_seqlen_q`` is the prefill max
        (not the decode step size) and the committed tensor must be sliced to
        the decode sub-batch. So ``indexer_score_topk`` passes ``next_n`` and
        ``n_committed_per_seq`` explicitly for that case.

        When both are ``None`` (the pure-decode / spec-verify step) this is
        byte-for-byte the native call -- delegating to it keeps the
        FULL-CUDAGraph-captured decode replay path untouched. The mixed-batch
        slice is always eager (mixed batches are PIECEWISE, never FULL), so the
        extra Python branch introduces no new captured shapes. Living here keeps
        native ATOM's ``Indexer`` interface unchanged.
        """
        if next_n is None and n_committed_per_seq is None:
            return super()._score_topk_decode(
                q_fp8, weights, block_tables, indexer_meta, topk
            )
        total_tokens = q_fp8.size(0)
        n_committed_per_seq_gpu = (
            n_committed_per_seq
            if n_committed_per_seq is not None
            else indexer_meta["n_committed_per_seq_gpu"]
        )  # int32 [bs]
        if next_n is None:
            next_n = max(1, int(get_forward_context().attn_metadata.max_seqlen_q))
        bs = total_tokens // next_n
        # deepgemm requires Q in [bs, next_n, heads, head_dim], KV in
        # [num_blocks, block_size, n_head=1, hidden_dim+scale_dim] (4D).
        q_4d = q_fp8.view(bs, next_n, self.n_heads, self.head_dim)
        kv_cache_4d = self.kv_cache.unsqueeze(-2)
        # Logits column count. Native pure-decode allocates the model-max
        # `_max_model_len_idx` (= max_seq_len // compress_ratio, e.g. 262144 at
        # 1M context) because its captured shape must bound the longest
        # possible sequence. This eager mixed-batch slice is NOT captured, so
        # size it to this batch's actual max committed compressed-KV length
        # instead: the deepgemm kernel guards every store by
        # `col < max_model_len_arg` and only writes [0, n_committed) per row,
        # and `top_k_per_row_decode` only reads [0, n_committed) per row, so a
        # width >= max(n_committed) is exact. The model-max width would allocate
        # a ~GB fp32 transient PER CSA layer and OOM the GPU at high concurrency
        # (HSA_STATUS_ERROR_OUT_OF_RESOURCES). Round up to the kernel's ChunkK
        # (256) so chunked stores never touch a partial tail column, and cap at
        # the model max.
        _CHUNK_K = 256
        max_committed = int(indexer_meta.get("decode_max_committed", 0))
        logits_width = min(
            self._max_model_len_idx,
            max(_CHUNK_K, ((max_committed + _CHUNK_K - 1) // _CHUNK_K) * _CHUNK_K),
        )
        # Per-fwd write-once GPU scratch; `top_k_per_row_decode` bounds each row
        # by `n_committed_per_seq[batch]` so unwritten cols are never picked
        # (no `fill_(-inf)` needed).
        logits = torch.empty(
            total_tokens,
            logits_width,
            dtype=torch.float32,
            device=q_fp8.device,
        )
        deepseek_v4_base.deepgemm_fp8_paged_mqa_logits(
            q_4d,
            kv_cache_4d,
            weights,
            logits,
            n_committed_per_seq_gpu,  # int32, sized [bs] (staged in builder)
            block_tables,
            logits_width,  # max_model_len arg == buffer width (store guard)
            KVBlockSize=self.kv_cache.size(1),  # k1_csa = 32
            Preshuffle=True,
        )
        topk_local = torch.empty(
            total_tokens, self.index_topk, dtype=torch.int32, device=q_fp8.device
        )
        deepseek_v4_base.top_k_per_row_decode(
            logits,
            next_n,
            n_committed_per_seq_gpu,
            topk_local,
            total_tokens,
            logits.stride(0),
            logits.stride(1),
            k=topk,
        )
        return topk_local  # [total_tokens, index_topk] int32, raw seq-local


class DeepseekV4AttentionVllm(DeepseekV4AttentionBase):
    """DeepSeek-V4 attention with vLLM piecewise-CUDA-graph reconciliation.

    Under ``cudagraph_mode=FULL_AND_PIECEWISE`` vLLM captures/replays the dense
    regions of ATOM's torch.compiled graph at the padded bucket width, while the
    eager attention split op (a graph break) runs at that same padded width. So
    for a prefill/mixed batch whose bucket was captured, ``x`` / ``positions``
    (and every downstream per-token projection) arrive padded to ``T_pad``, but
    the sparse-attention metadata is built for the *real* token count (the
    bridge's prefill path sets ``batch_id_per_token`` to length == real tokens).

    There are two eager entry points, and ``DeepseekV4Attention.forward`` picks
    between them by cudagraph mode, so BOTH must reconcile the padding:

    * WIDE split (FULL / NONE): the whole attention is one eager op
      (``v4_attention_with_output``) that calls ``forward_impl`` — reconciled in
      the ``forward_impl`` override below.
    * NARROW split (PIECEWISE — the FULL_AND_PIECEWISE prefill/mixed path): the
      Q/KV/indexer projections run as a *captured* dense piece (``_attn_pre``)
      at the padded width, then the eager op (``v4_core_attention``) calls
      ``_attn_core`` DIRECTLY, never ``forward_impl`` — reconciled in the
      ``_attn_core`` override below. (This is the path exercised by the launch
      config; without the ``_attn_core`` slice the padded ``q`` reaches
      ``sparse_attn_v4_paged_prefill`` while ``kv_indptr_prefix`` is real-sized.)

    In both, slice every per-token input down to the real token count before the
    (unchanged) native attention so per-token Q rows match the ``kv_indptr``
    arrays — otherwise the paged-prefill kernel aborts with ``kv_indptr_prefix
    length must be N+1`` — then pad the output back to ``T_pad`` so the next
    captured dense region (and the op's ``empty_like`` fake-meta / piecewise
    output buffer, both keyed by the padded row count) see the full bucket width.

    Decode is fully captured (incl. these ops) with metadata already padded to
    the bucket, so it runs at the padded width and must NOT be sliced. The padded
    rows are never sampled (``logits_indices`` reference real positions only).
    """

    def forward_impl(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        fc = get_forward_context()
        # Dummy/bypass forwards short-circuit inside the native impl; defer to it
        # at full width (no metadata to reconcile against).
        if not fc.context.is_dummy_run:
            attn_md = fc.attn_metadata
            if attn_md is not None and attn_md.state is not AttnState.DECODE:
                num_in = x.size(0)
                bid = attn_md.batch_id_per_token
                num_real = bid.shape[0] if bid is not None else num_in
                if num_real < num_in:
                    out = super().forward_impl(x[:num_real], positions[:num_real])
                    return torch.nn.functional.pad(out, (0, 0, 0, num_in - num_real))
        return super().forward_impl(x, positions)

    def _attn_core(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        kv_pre: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor,
        positions: torch.Tensor,
        idx_q_fp8: Optional[torch.Tensor] = None,
        idx_weights: Optional[torch.Tensor] = None,
        compressor_already_launched: bool = False,
    ) -> torch.Tensor:
        # NARROW PIECEWISE entry (see class docstring): the ``v4_core_attention``
        # split op calls this DIRECTLY, so ``forward_impl``'s slice never runs.
        # ``_attn_pre`` projected every per-token tensor at the padded bucket
        # width, but the sparse-attention metadata (``batch_id_per_token`` /
        # ``kv_indptr_*``) is sized to the real token count. Clip all per-token
        # inputs to that real count, run the native core, then pad the output
        # back to the bucket width — the native core reads only real-sized
        # metadata, and the padded-width output keeps the piecewise output buffer
        # and the downstream graphed ``_attn_post`` piece at the captured shape.
        fc = get_forward_context()
        if not fc.context.is_dummy_run:
            attn_md = fc.attn_metadata
            if attn_md is not None and attn_md.state is not AttnState.DECODE:
                num_in = x.size(0)
                bid = attn_md.batch_id_per_token
                num_real = bid.shape[0] if bid is not None else num_in
                if num_real < num_in:

                    def _clip(t):
                        # Clip only the leading (token) dim, and only when it is
                        # the padded width — leaves per-seq / scalar tensors and
                        # any Nones untouched.
                        if (
                            isinstance(t, torch.Tensor)
                            and t.dim() >= 1
                            and t.size(0) == num_in
                        ):
                            return t[:num_real]
                        return t

                    o = super()._attn_core(
                        _clip(x),
                        _clip(q),
                        _clip(kv_pre),
                        _clip(qr),
                        _clip(qr_scale),
                        _clip(positions),
                        _clip(idx_q_fp8),
                        _clip(idx_weights),
                        compressor_already_launched=compressor_already_launched,
                    )
                    return torch.nn.functional.pad(o, (0, 0, 0, num_in - num_real))
        return super()._attn_core(
            x,
            q,
            kv_pre,
            qr,
            qr_scale,
            positions,
            idx_q_fp8,
            idx_weights,
            compressor_already_launched=compressor_already_launched,
        )


class DeepseekV4ModelVllm(DeepseekV4ModelBase):
    """DeepSeek-V4 model with a persistent MTP-draft hidden-state buffer.

    vLLM's DeepSeek-V4 MTP draft reads the target's pre-hc_head residual from
    Python *outside* the CUDAGraph (via the plugin's ``get_mtp_*`` hook). Under
    ``cudagraph_mode=FULL*`` the target model's Python ``forward`` body does not
    re-run on replay, so a plain Python stash of the return value would freeze
    the draft on the capture-time residual and draft acceptance collapses
    (~4%). Mirror vLLM's native DeepSeek-V4 ``_mtp_hidden_buffer``: allocate a
    stable-address buffer once (outside the graph pool) and refresh it every
    forward with an *in-graph* ``copy_`` that is captured and thus re-runs on
    every replay.

    This lives in the vLLM plugin only; native ATOM serving does not need it
    because its ModelRunner already routes the model output through a persistent
    ``forward_vars["outputs"]`` buffer that the drafter reads after replay.
    """

    def __init__(self, *, atom_config, args):
        super().__init__(atom_config=atom_config, args=args)
        hc_dim = self.hc_mult * self.args.dim
        sched_cfg = getattr(atom_config, "scheduler_config", None)
        max_num_batched_tokens = getattr(
            sched_cfg, "max_num_batched_tokens", None
        ) or getattr(atom_config, "max_num_batched_tokens", None)
        self._mtp_hidden_buffer = torch.empty(
            max_num_batched_tokens,
            hc_dim,
            dtype=self.embed.weight.dtype,
            device=self.embed.weight.device,
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = super().forward(input_ids, positions)
        # In-graph copy_: captured into the CUDAGraph so it refreshes the buffer
        # on every replay, keeping the MTP draft's input hidden states current.
        num_tokens = h.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(h.flatten(1))
        return h


class DeepseekV4ForCausalLM(DeepseekV4ForCausalLMBase):
    """Native DeepSeek-V4 model built with the vLLM attention + model variants.

    Temporarily rebinds the module-global ``DeepseekV4Attention`` /
    ``DeepseekV4Model`` to their vLLM subclasses while the base ``__init__``
    constructs the tree (each layer does ``self.attn = DeepseekV4Attention(...)``
    and the wrapper does ``self.model = DeepseekV4Model(...)`` via those
    globals), then restores them. Class attributes used by the plugin wrapper
    (``weights_mapper`` / ``weights_mapping`` / ``packed_modules_mapping`` /
    ``extra_output_dims``) are inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        original_attn_cls = deepseek_v4_base.DeepseekV4Attention
        original_model_cls = deepseek_v4_base.DeepseekV4Model
        original_indexer_cls = deepseek_v4_base.Indexer
        deepseek_v4_base.DeepseekV4Attention = DeepseekV4AttentionVllm
        deepseek_v4_base.DeepseekV4Model = DeepseekV4ModelVllm
        # `DeepseekV4Attention.__init__` builds `self.indexer = Indexer(...)`
        # via this module-global, so rebind it too to get the mixed-batch
        # split variant across the whole model tree.
        deepseek_v4_base.Indexer = IndexerVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v4_base.DeepseekV4Attention = original_attn_cls
            deepseek_v4_base.DeepseekV4Model = original_model_cls
            deepseek_v4_base.Indexer = original_indexer_cls
