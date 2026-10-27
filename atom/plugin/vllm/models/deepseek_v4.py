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

import dataclasses

import torch

from atom.models import deepseek_v4 as deepseek_v4_base

# isort: off
from atom.models.deepseek_v4 import (
    DeepseekV4Attention as DeepseekV4AttentionBase,
    DeepseekV4ForCausalLM as DeepseekV4ForCausalLMBase,
    DeepseekV4Model as DeepseekV4ModelBase,
    Indexer as IndexerBase,
)

# isort: on
from atom.model_ops.indexer import fp8_rows, score_topk
from atom.plugin.deepseek_v4_index_rows import v4_concat_rows, v4_paged_rows
from atom.plugin.vllm.deepseek_v4_bridge import ATOM_DEEPSEEK_V4_BLOCK_SIZE
from atom.utils.forward_context import AttnState, get_forward_context

# Compressor.forward and CSA translation read this module global at runtime, so
# it must remain aligned with the vLLM proxy page geometry after construction.
deepseek_v4_base._V4_BLOCK_SIZE = ATOM_DEEPSEEK_V4_BLOCK_SIZE


class IndexerVllm(IndexerBase):
    """DeepSeek-V4 sparse indexer with vLLM mixed-batch decode/prefill split.

    Native ATOM serves decode and prefill in *separate* forward passes, its
    builder publishing the step's rows (`attn_metadata.index_rows`). Under
    vLLM the bridge publishes the step's tensors instead, and continuous
    batching produces MIXED batches (chunked-prefill rows + running decode
    rows) that ATOM classifies as PREFILL. Scoring the decode rows as prefill
    rows would gather EVERY running seq's committed keys into the dense plane
    -- quadratic in batch size, which OOMs the GPU at high concurrency.

    So the batch is split here: vLLM reorders decode rows (query_len <=
    1[+num_spec]) to the FRONT, a contiguous prefix whose rows read their
    requests' keys paged, while the prefill rows score against the prefill
    sub-batch's committed keys, which the bridge (`_populate_indexer`) gathers
    over the prefill sequences only. A pure-decode step is the native layout
    (each row off its own table row), so the FULL-CUDAGraph decode path keeps
    its shapes; a mixed batch is always eager.
    """

    def indexer_score_topk(
        self,
        q_quant: torch.Tensor,  # [total_tokens, n_heads, head_dim] FP8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        q_scale: torch.Tensor | None,  # None: no bridge binds an FP4 cache
        topk: int,
    ) -> torch.Tensor:
        fc = get_forward_context()
        md = fc.attn_metadata
        cache = self.index_cache
        total = q_quant.size(0)
        if not fc.context.is_prefill:
            rows = v4_paged_rows(md, total, self._max_model_len_idx)
            return score_topk(q_quant, weights, cache, rows, topk=topk)
        meta = md.indexer_meta
        num_decode_tokens = int(meta.get("num_decode_tokens", 0))
        out = torch.empty(total, topk, dtype=torch.int32, device=q_quant.device)
        if num_decode_tokens:
            num_decodes = int(meta["num_decodes"])
            score_topk(
                q_quant[:num_decode_tokens],
                weights[:num_decode_tokens],
                cache,
                fp8_rows(
                    meta["n_committed_per_seq_gpu"][:num_decodes],
                    self._decode_slice_width(meta),
                    None,
                    md.block_tables[:num_decodes],
                    next_n=int(meta["decode_next_n"]),
                ),
                topk=topk,
                out=out[:num_decode_tokens],
            )
        # Under MTP a step of only 1-token decodes is not a uniform decode
        # (max_query_len < decode_q), and leaves no prefill rows: the dense
        # gather must never run on an empty slice, whose committed meta still
        # claims a row.
        if num_decode_tokens < total:
            score_topk(
                q_quant[num_decode_tokens:],
                weights[num_decode_tokens:],
                cache,
                v4_concat_rows(
                    meta, md.block_tables[int(meta.get("num_decodes", 0)) :]
                ),
                topk=topk,
                out=out[num_decode_tokens:],
            )
        return out

    def _decode_slice_width(self, meta: dict) -> int:
        """The plane of a mixed batch's decode rows: as wide as their longest
        committed run (a host int, no sync), rounded to the scorer's 256-column
        chunk and capped at the model's -- the model-max plane would be a ~GB
        transient per CSA layer at long context. Exact: a row writes and reads
        only `[0, n_committed)`."""
        chunk = 256
        longest = int(meta.get("decode_max_committed", 0))
        return min(self._max_model_len_idx, max(chunk, -(-longest // chunk) * chunk))


class DeepseekV4AttentionVllm(DeepseekV4AttentionBase):
    """DeepSeek-V4 attention with vLLM piecewise-CUDA-graph reconciliation.

    Under ``cudagraph_mode=FULL_AND_PIECEWISE`` vLLM captures/replays the dense
    regions of ATOM's torch.compiled graph at the padded bucket width, while the
    eager attention split op (a graph break) runs at that same padded width. So
    for a prefill/mixed batch whose bucket was captured, ``x`` / ``positions``
    (and every downstream per-token projection) arrive padded to ``T_pad``, but
    the sparse-attention metadata is built for the *real* token count (the
    bridge's prefill path sets ``batch_id_per_q_token`` to length == real tokens).

    There are two eager entry points, and ``DeepseekV4Attention.forward`` picks
    between them by cudagraph mode, so BOTH must reconcile the padding:

    * WIDE split (FULL / NONE): the whole attention is one eager op
      (``v4_attention_with_output``) that calls ``forward_impl`` — reconciled in
      the ``forward_impl`` override below.
    * NARROW split (PIECEWISE — the FULL_AND_PIECEWISE prefill/mixed path): the
      Q/KV/indexer projections run as a *captured* dense piece (``_attn_pre``)
      at the padded width and the attention runs as three ops, never through
      ``forward_impl`` — reconciled in the ``_sparse_attention`` override below,
      which is the one of the three that reads the real-sized metadata. (This is
      the path exercised by the launch config; without that clip the padded
      ``q`` reaches ``sparse_attn_v4_paged_2src`` while ``kv_indptr_prefix``
      is real-sized.)

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
                bid = attn_md.batch_id_per_q_token
                num_real = bid.shape[0] if bid is not None else num_in
                if num_real < num_in:
                    out = super().forward_impl(x[:num_real], positions[:num_real])
                    return torch.nn.functional.pad(out, (0, 0, 0, num_in - num_real))
        return super().forward_impl(x, positions)

    def _sparse_attention(
        self,
        qkn,
        positions: torch.Tensor,
        idx_q_quant: torch.Tensor | None = None,
        idx_weights: torch.Tensor | None = None,
        idx_q_scale: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        qr: torch.Tensor | None = None,
        qr_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NARROW entry (see class docstring). The pieces upstream produced every
        # per-token tensor at the padded bucket width, but the sparse-attention
        # metadata this half reads (`batch_id_per_q_token`, `kv_indptr_*`) is sized
        # to the real token count -- and this is the half that reads it, so this
        # is where the two have to be reconciled. Clip in, pad out: the padded
        # width is what the piecewise output buffer and the graphed `_attn_post`
        # piece downstream were captured at.
        #
        # Decode is fully captured with metadata already padded to the bucket, so
        # it runs at the padded width and must NOT be clipped.
        #
        # `x` is NOT clipped for the compressor in `_launch_compressors`: its plan is
        # built from the real lengths and addresses rows through it, so padded
        # rows are never read. Untested on this path -- if prefill misbehaves
        # here, that is the first thing to check.
        fc = get_forward_context()
        if not fc.context.is_dummy_run:
            attn_md = fc.attn_metadata
            if attn_md is not None and attn_md.state is not AttnState.DECODE:
                num_in = positions.size(0)
                bid = attn_md.batch_id_per_q_token
                num_real = bid.shape[0] if bid is not None else num_in
                if num_real < num_in:

                    def _clip(t):
                        # Leading (token) dim only, and only at the padded
                        # width -- leaves per-seq tensors and Nones alone.
                        if (
                            isinstance(t, torch.Tensor)
                            and t.dim() >= 1
                            and t.size(0) == num_in
                        ):
                            return t[:num_real]
                        return t

                    o = super()._sparse_attention(
                        dataclasses.replace(
                            qkn,
                            **{
                                f.name: _clip(getattr(qkn, f.name))
                                for f in dataclasses.fields(qkn)
                            },
                        ),
                        positions[:num_real],
                        _clip(idx_q_quant),
                        _clip(idx_weights),
                        _clip(idx_q_scale),
                        _clip(x),
                        _clip(qr),
                        _clip(qr_scale),
                    )
                    return torch.nn.functional.pad(o, (0, 0, 0, num_in - num_real))
        return super()._sparse_attention(
            qkn, positions, idx_q_quant, idx_weights, idx_q_scale, x, qr, qr_scale
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
