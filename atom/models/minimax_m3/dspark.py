# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 DSpark semi-autoregressive block drafter for ATOM.

The same algorithm as the Kimi-K3 draft (``kimi_k3_dspark.py``) -- the target's
aux hidden states are projected once into a context, every draft layer writes
its own K/V rows for that context into a paged cache, and one non-causal block
pass over ``[anchor, MASK, ...]`` drafts the whole block before a Markov head
samples it left to right -- on a MiniMax-M3 backbone instead of MLA. The
reference is NeMo AutoModel ``speculative/dspark/draft_minimax_m3.py``:

  * Attention is plain GQA (64 q / 4 kv heads, head_dim 128) with per-head
    Gemma RMSNorm on Q and K, partial neox RoPE on the first 64 lanes, and no
    norm or RoPE on V. The K/V of the context and of the block are projected
    separately and attended together, non-causally: every block position sees
    the whole context and the whole block.
  * Every norm is Gemma style, ``x * (1 + w)``; the MLP is dense SwiGLU-OAI.
    There is no indexer and no MoE -- both belong to the target.
  * The context is ``hidden_norm(fc(concat(aux)))`` and goes straight into each
    layer's K/V projection, never through ``input_layernorm``.

KV lives in a sibling pool of its own (an MHA draft cannot share the target's
latent pool the way K3's MLA draft does): ``draft_kv_builder`` resolves this
config to the MHA backend, which owns a pool laid out like the target's and
binds it onto each layer's ``KVCacheBinding``. Both writes -- context rows and the
block's own rows -- go through ``triton_fused_norm_rope_cache`` with an explicit
slot mapping, the kernel the target's dense layers use for the same norm, RoPE
and per-token fp8 cache format.

The block attention is ``B * T`` independent q_len=1 paged decode rows --
non-causal by construction -- over each request's ``[context | block]``,
through the per-row ASM / Gluon decode the target's sparse layers run and a
page-16 table ``prepare_paged_block`` builds outside the graph; the block's own K/V
are written into the cache first, except at positions past the request's
allocated pages. No host sync and fixed shapes, so the block pass is CUDA-graph
captured like the other drafts. The per-row decode takes one KV head per rank,
so the draft needs TP >= its KV head count (4) to draft; a P/D producer only
writes context K/V, so it may hold several (all four on a TP1 PP stage).

Checkpoint layout (``<draft>/dspark``, MXFP8 [1, 32] linears, BF16 rest):
  language_model.model.dspark.fc.weight                       [6144, 5 * 6144]
  language_model.model.dspark.{hidden_norm,final_norm}.weight [6144]
  language_model.model.dspark.layers.{0..4}.decoder_layer.
      {input_layernorm,post_attention_layernorm}.weight
      self_attn.{q,k,v,o}_proj.weight(_scale_inv)             -> qkv_proj
      self_attn.{q,k}_norm.weight                             [128]
      mlp.{gate,up,down}_proj.weight(_scale_inv)              -> gate_up_proj
  language_model.model.dspark.markov_head.markov_w{1,2}.weight [200064, 256]
  language_model.model.dspark.confidence_head.proj.{weight,bias} [1, 6400]
The draft ships no embedding or LM head; both are the target's.
"""

from typing import TYPE_CHECKING, ClassVar

import torch
from aiter import QuantType, dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
from aiter.rotary_embedding import get_rope
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.config import QuantizationConfig
from atom.kv_transfer.disaggregation import kv_config_has_producer
from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.model_ops.layernorm import (
    GemmaRMSNorm,
    fused_allreduce_gemma_rms_norm,
    fused_allreduce_gemma_rms_norm_mxfp8_quant,
)
from atom.model_ops.embed_head import VocabParallelEmbedding, masked_embedding
from atom.model_ops.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.minimax_m3.sparse_attn import (
    ASM_PAGE_SIZE,
    PAGES_PER_SPARSE_BLOCK,
    sparse_pa_per_row,
    to_page16_shuffle,
)
from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
    triton_fused_norm_rope_cache,
)
from atom.models.dspark_draft import (
    DSparkConfidenceHead,
    DSparkDraftModel,
    DSparkMarkovHead,
    linear_out,
)
from atom.models.minimax_m3.model import MiniMaxM3MLP
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

if TYPE_CHECKING:
    from atom.config import Config


def _draft_fused_ar() -> bool:
    """``ATOM_DSPARK_FUSED_AR``: the draft layers leave o_proj's and the MLP's
    outputs TP-partial and all-reduce them in the next norm, fused with the
    RMSNorm and, where the next linear reads MXFP8, its quant -- the target
    layers' fused all-reduce RMSNorm (MXFP8) path. Read at construction (the
    linears' ``reduce_results``) and in the forward alike."""
    return envs.ATOM_DSPARK_FUSED_AR and get_tensor_model_parallel_world_size() > 1


def _draft_fuse_rows(rows: int) -> bool:
    """ATOM_DSPARK_FUSED_AR at this many block rows: batches past
    ATOM_DSPARK_FUSED_AR_MAX_ROWS (default 56, eight requests of a 7-token
    block) keep the unfused all-reduce + norm launches. Decided per captured
    batch size (the draft runs eagerly or as a CUDA graph, not compiled)."""
    return rows <= envs.ATOM_DSPARK_FUSED_AR_MAX_ROWS


def _ar_norm(hidden, residual, norm, consumer):
    """All-reduce ``hidden``, add ``residual``, Gemma-RMSNorm; MXFP8-quantized
    for an MXFP8 ``consumer``. Returns ``(hidden, x_scale | None, residual)``.
    Past ATOM_DSPARK_FUSED_AR_MAX_ROWS: the launches the unfused path runs (the
    linear's all-reduce, then the norm; the consumer quantizes for itself)."""
    from atom.models.minimax_m3.model import _linear_consumes_mxfp8

    if not _draft_fuse_rows(hidden.shape[0]):
        hidden = tensor_model_parallel_all_reduce(hidden)
        hidden, residual = norm(hidden, residual)
        return hidden, None, residual
    if (
        consumer is not None
        and envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8
        and _linear_consumes_mxfp8(consumer)
    ):
        return fused_allreduce_gemma_rms_norm_mxfp8_quant(hidden, residual, norm)
    hidden, residual = fused_allreduce_gemma_rms_norm(hidden, residual, norm)
    return hidden, None, residual


def _draft_rope_theta(config) -> float:
    # transformers 5 moves rope_theta into rope_parameters on the M3 VL text
    # config class this draft loads as; the plain-dict target keeps it flat.
    params = getattr(config, "rope_parameters", None) or {}
    theta = params.get("rope_theta") or getattr(config, "rope_theta", None)
    if not theta:
        raise ValueError("MiniMax-M3 DSpark draft config names no rope_theta.")
    return float(theta)


def _draft_quant_config(config) -> QuantizationConfig:
    """The draft's own quantization, independent of the target's.

    The draft checkpoint is MXFP8 (``quant_method: mxfp8``, ``[1, 32]`` blocks)
    whatever the target is -- the M3.1 targets come as MXFP8 + NVFP4 experts,
    Quark MXFP4, or BF16 -- so the target's config, which ``_build_draft_model``
    copies onto the draft, cannot be trusted to describe these weights. The
    checkpoint's own block goes through the registered parser here; a draft
    config without one is BF16.
    """
    draft_qc = QuantizationConfig(config)
    spec = draft_qc.global_spec
    if draft_qc.quant_method and (spec.quant_type, spec.quant_dtype) != (
        QuantType.per_1x32,
        dtypes.fp8,
    ):
        raise ValueError(
            "MiniMax-M3 DSpark draft supports MXFP8 [1, 32] or BF16 weights, "
            f"got quant_method={draft_qc.quant_method!r} ({spec.quant_type}, "
            f"{spec.quant_dtype})."
        )
    return draft_qc


# Pinned staging slots for the pages each request holds; see `_alloc_block_rows`.
_ALLOC_RING = 4


def _page16_table(
    block_tables: torch.Tensor,  # [rb, W] page-128 ids
    out: torch.Tensor,  # [rb, 8 * W]
    sub_pages: torch.Tensor,  # [8] = arange(8)
) -> torch.Tensor:
    """Page-16 ids of a page-128 table: page p covers ``p * 8 + 0..7``.

    Through ``copy_`` of an expanded view: an ``out=`` of the ``[rb, W, 1]``
    product would be resized to that shape rather than broadcast over the
    eight sub-pages, leaving ``[8p0, 8p1+1, 8p2+2, ...]``.
    """
    rb, W = block_tables.shape
    out3 = out.view(rb, W, PAGES_PER_SPARSE_BLOCK)
    out3.copy_(block_tables.unsqueeze(-1).expand(rb, W, PAGES_PER_SPARSE_BLOCK))
    out3.mul_(PAGES_PER_SPARSE_BLOCK).add_(sub_pages)
    return out


def _window_rows(
    bt16_flat: torch.Tensor,  # [(rb + 1) * P] request tables, then P of slack
    ctx_eff: torch.Tensor,  # [rb] keys each request can read
    positions: torch.Tensor,  # [rb * T] each row's own position
    row_req: torch.Tensor,  # [rb * T] the request each row belongs to
    window: int,  # 0: attend the whole context
    real_rows: int,  # rows past this are padding
    start16: torch.Tensor,  # [rb * T] int64 scratch
    out_tables: torch.Tensor,  # [rb * T, P]
    out_lens: torch.Tensor,  # [rb * T]
) -> None:
    """Each block row's page-16 table and length under the sliding window.

    A row at position p attends the keys from ``p - window + 1`` on, as the
    reference does (the whole block is inside that window). The per-row
    decode starts a row at a page boundary, so the row starts at the page-16
    holding that key and also reads up to 15 keys before it. Its table is
    the P-wide window of its request's table from that page on, taken from
    the flat buffer; its length ends it within its own request's columns,
    so what that window runs into past them is never read.
    """
    P = out_tables.shape[1]
    if window:
        torch.sub(positions, window - 1, out=start16)
        start16.clamp_(min=0).div_(ASM_PAGE_SIZE, rounding_mode="floor")
        # Pad rows read one page.
        start16[real_rows:].zero_()
    else:
        start16.zero_()
    torch.index_select(ctx_eff, 0, row_req, out=out_lens)
    out_lens.sub_(start16, alpha=ASM_PAGE_SIZE)
    start16.add_(row_req, alpha=P)  # each row's first entry in the flat buffer
    torch.index_select(bt16_flat.unfold(0, P, 1), 0, start16, out=out_tables)


class KVCacheBinding(nn.Module):
    """What the draft KV pool binds one layer's cache onto, and nothing else.

    ``DraftKvBuilder`` recognizes a non-MLA attention module by these
    attributes and sets ``k_cache`` / ``v_cache`` (5-D SHUFFLE), the fp8
    per-token scales and ``max_model_len`` on it; the layer then reads and
    writes the cache itself.
    """

    base_attention = None
    use_mla = False
    # The page every access of the cache views it in (`_bound_cache`): a TP1
    # producer cuts each 128-token block per decode rank by these pages.
    page_tokens = ASM_PAGE_SIZE

    def __init__(self, layer_num: int) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.max_model_len = 0
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None


class M3DSparkAttention(nn.Module):
    """GQA attention for one draft layer, with a second entry point for the
    target-derived context rows."""

    def __init__(
        self,
        atom_config: "Config",
        config,
        quant_config: QuantizationConfig,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError(
                f"DSpark draft has {self.total_num_heads} attention heads, which "
                f"is not divisible by TP size {tp_size}."
            )
        # A P/D producer never runs the block pass
        # (`Drafter.context_kv_only`); it only writes context K/V.
        producer = kv_config_has_producer(atom_config.kv_transfer_config)
        if self.total_num_kv_heads > tp_size and not producer:
            raise ValueError(
                "The MiniMax-M3.1 DSpark draft decodes its block per row with one "
                f"KV head per rank, so it needs TP >= {self.total_num_kv_heads} "
                f"(its KV head count); got TP {tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.kv_cache_dtype = atom_config.kv_cache_dtype

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            # TP-partial under ATOM_DSPARK_FUSED_AR: the layer's next norm reduces.
            reduce_results=not _draft_fused_ar(),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(
            getattr(config, "rotary_dim", 0)
            or self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )
        # neox (half-split) rotation of the first `rotary_dim` lanes -- the
        # gpt_oss rotary the reference uses, and what the fused kernel applies.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_draft_rope_theta(config),
            rope_scaling=None,
        )
        self.kv = KVCacheBinding(layer_num)
        # 0: every block row attends the whole context (see `_window_rows`).
        self.sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        # ATOM_DSPARK_CTX_KV_ONLY, read once: the context write runs every step.
        self.context_kv_only = envs.ATOM_DSPARK_CTX_KV_ONLY

    def _context_kv_proj(self):
        """``qkv_proj`` narrowed to its K/V rows: a row view of the same weight
        and scales, built on first use (after the weights are loaded and laid
        out) and again if the weight's storage is ever rebound."""
        weight_ptr = self.qkv_proj.weight.data_ptr()
        cached = self.__dict__.get("_kv_rows")
        if cached is None or cached[0] != weight_ptr:
            # Kept out of the module tree: the view is not a parameter owner.
            cached = self.__dict__["_kv_rows"] = (
                weight_ptr,
                self.qkv_proj.make_row_view(self.q_size, 2 * self.kv_size),
            )
        return cached[1]

    def _bound_cache(self):
        """This layer's KV cache in the layout every access of it uses.

        The block reads through the page-16 SHUFFLE views the per-row decode
        kernels take, so the cache must also be WRITTEN through them: a page-128
        SHUFFLE view puts the 128 in-page rows inside the head-dim tiles, and
        reading that memory as eight 16-row pages scrambles it. Slot numbers are
        the same in both (page * 128 + off == (page * 8 + off // 16) * 16 +
        off % 16), only the view differs.
        """
        kv = self.kv
        if kv.k_cache.numel() == 0:
            return None
        return to_page16_shuffle(kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale)

    def _norm_rope_write(self, q, k, v, num_heads, positions, slot_mapping, cache):
        k_cache, v_cache, k_scale, v_scale = cache
        return triton_fused_norm_rope_cache(
            q,
            k,
            v,
            positions,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            slot_mapping=slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected + normed target hidden
        positions: torch.Tensor,  # [N]
        slot_mapping: torch.Tensor,  # [N] the target forward's slots
    ) -> None:
        """K/V rows for the target context: k_norm + RoPE on K, V as is.

        Only the K/V rows of the fused projection are computed under
        ATOM_DSPARK_CTX_KV_ONLY; otherwise the q third is computed and dropped.
        Either way the kernel writes K/V only when told it has zero query heads.
        """
        cache = self._bound_cache()
        if cache is None:
            return
        if self.context_kv_only:
            kv = linear_out(self._context_kv_proj()(ctx_hidden))
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            qkv = linear_out(self.qkv_proj(ctx_hidden))
            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        self._norm_rope_write(k, k, v, 0, positions, slot_mapping, cache)

    def write_context_kv_rows(self, k, v, positions, slot_mapping) -> None:
        """``write_context_kv`` from K/V already projected (``ContextKV``)."""
        cache = self._bound_cache()
        if cache is None:
            return
        self._norm_rope_write(k, k, v, 0, positions, slot_mapping, cache)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        x_scale: torch.Tensor | None = None,
    ):
        """The block pass: non-causal attention over ``[context | block]``.

        Write the block's K/V at the block slots the proposer masked to the
        request's allocated pages -- an unallocated column reads as page 0, a
        page another request owns (under prefix caching, a shared prefix) --
        then decode per row.
        """
        if self.num_kv_heads != 1:
            # Before any K/V is written: only a P/D producer, which never
            # drafts, is built with more than one KV head per rank.
            raise RuntimeError(
                f"The block pass decodes one KV head per rank; this rank holds "
                f"{self.num_kv_heads} of {self.total_num_kv_heads}"
            )
        qkv = linear_out(self.qkv_proj(hidden_states, x_scale=x_scale))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cache = self._bound_cache()
        if cache is None:
            # warmup_model() runs before the pool exists; the pass only sizes
            # activations then, so shapes are all that must be right.
            out = q.new_zeros(q.shape[0], self.q_size)
            return linear_out(self.o_proj(out))

        md = get_forward_context().attn_metadata
        # The block's K/V go into the cache at the block's slots (positions
        # past the allocated pages are -1, dropped), then every block position
        # decodes as its own q_len=1 row.
        q, _ = self._norm_rope_write(
            q, k, v, self.num_heads, positions, md.slot_mapping, cache
        )
        out = self._block_attention_paged(q, md, cache)
        return linear_out(self.o_proj(out))

    def _block_attention_paged(self, q: torch.Tensor, md, cache) -> torch.Tensor:
        """``[context | block]`` attention as ``B * T`` q_len=1 paged decode rows.

        A length-1 query sees its whole context, so per-row decode is the
        non-causal block attention by construction. The rows read the request's
        page-16 table and capped length that ``prepare_paged_block`` built
        (``dspark_bt16_rows`` / ``dspark_ctx_rows``, sliding window applied
        there), through the same per-row ASM / Gluon decode the target's sparse
        layers run -- no host sync, fixed shapes, so the block can be captured.
        The context is read once per block position, one KV head per rank.
        """
        k16, v16, ks16, vs16 = cache  # page-16 views, see _bound_cache
        rows = q.shape[0]
        q = q.view(rows, self.num_heads, self.head_dim)
        out = torch.empty_like(q)
        sparse_pa_per_row(
            q,
            k16,
            v16,
            md.dspark_bt16_rows,
            md.dspark_ctx_rows,
            num_kv_heads=self.num_kv_heads,
            sm_scale=self.scaling,
            output=out,
            k_scale=ks16,
            v_scale=vs16,
            # Every layer of the pass reads the same rows and lengths: one plan
            # refresh per pass (the target's decode layers do the same).
            plan_step_owner=md,
            # A row reads at most the window, the page-16 start rounding and
            # the block: the splits follow that instead of the legacy cap.
            max_context_length=(
                self.sliding_window + ASM_PAGE_SIZE + int(md.max_seqlen_q)
                if self.sliding_window
                else None
            ),
        )
        return out.view(rows, -1)


class M3DSparkDecoderLayer(nn.Module):
    """One draft layer: pre-norm attention + pre-norm dense SwiGLU-OAI MLP."""

    def __init__(
        self,
        atom_config: "Config",
        config,
        quant_config: QuantizationConfig,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = M3DSparkAttention(
            atom_config, config, quant_config, layer_num, prefix=f"{prefix}.self_attn"
        )
        self.mlp = MiniMaxM3MLP(
            config=config,
            intermediate_size=int(config.dspark_ffn_hidden_size),
            quant_config=quant_config,
            # TP-partial under ATOM_DSPARK_FUSED_AR: the next norm reduces.
            reduce_results=not _draft_fused_ar(),
            prefix=f"{prefix}.mlp",
            # The checkpoint says `swigluoai`, and the reference draft MLP is
            # SwiGLU-OAI whatever the field says; transformers'
            # MiniMaxM3VLTextConfig, which this config loads as, rewrites the
            # field to its `silu` default.
            hidden_act="swigluoai",
            # ATOM_DSPARK_SWIGLU_MXFP8: the target's SwiGLU + MXFP8 quant fusion
            # (bitwise its unfused pair) for the draft's MXFP8 down_proj too.
            fuse_swiglu_mxfp8=envs.ATOM_DSPARK_SWIGLU_MXFP8,
        )
        # ATOM_DSPARK_SWIGLU_MXFP8 as built (flag and arch); taken per call up to
        # ATOM_DSPARK_FUSED_AR_MAX_ROWS rows (`_forward_fused_ar`).
        self._swiglu_mxfp8 = self.mlp.fuse_swiglu_mxfp8
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def write_context_kv(self, ctx_hidden, positions) -> None:
        """Populate this layer's context rows at the TARGET forward's slots,
        which are still the installed metadata when the proposer calls this."""
        slot_mapping = get_forward_context().attn_metadata.slot_mapping[
            : ctx_hidden.shape[0]
        ]
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping)

    def forward(self, positions, hidden_states, residual=None):
        if _draft_fused_ar():
            return self._forward_fused_ar(positions, hidden_states, residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    def _forward_fused_ar(self, positions, hidden_states, residual):
        """``forward`` with each all-reduce fused into the norm after it
        (ATOM_DSPARK_FUSED_AR). The returned hidden is TP-partial; the next
        layer's input norm, or the model's final norm (``block_backbone``),
        reduces it."""
        x_scale = None
        if residual is None:
            # The embedding arrives reduced.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, x_scale, residual = _ar_norm(
                hidden_states, residual, self.input_layernorm, self.self_attn.qkv_proj
            )
        hidden_states = self.self_attn(positions, hidden_states, x_scale=x_scale)
        hidden_states, x_scale, residual = _ar_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm,
            self.mlp.gate_up_proj,
        )
        self.mlp.fuse_swiglu_mxfp8 = self._swiglu_mxfp8 and _draft_fuse_rows(
            hidden_states.shape[0]
        )
        return self.mlp(hidden_states, x_scale=x_scale), residual

    def forward_mono(self, positions, hidden_states, residual, normed, next_norm, next_mxfp8):
        """``_forward_fused_ar`` as two launches around the production paged attention where they serve the
        call (``dspark_draft_mono.py``): qkv + q/k norm + RoPE + K/V cache write, then o_proj -> all-reduce +
        norm -> MLP -> all-reduce + ``next_norm`` (the next layer's input norm, its MXFP8 output with
        ``next_mxfp8``, or the final norm). ``normed``: ``(x, x_scale, residual)`` when the previous layer's
        launch already all-reduced and normed this layer's input.

        Returns ``(hidden, residual, normed)``: ``normed`` is set when the launches ran (the next layer's input,
        or the final normed hidden); otherwise ``hidden`` is the TP-partial MLP output, as
        ``_forward_fused_ar`` returns it."""
        from atom.model_ops import dspark_draft_mono as mono
        from atom.model_ops.blockscale import quantize_fp8

        x_scale = None
        if normed is not None:
            hidden_states, x_scale, residual = normed
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, x_scale, residual = _ar_norm(
                hidden_states, residual, self.input_layernorm, self.self_attn.qkv_proj
            )
        attn = self.self_attn
        rows = hidden_states.shape[0]
        cache = attn._bound_cache()
        if mono.usable(self, rows, cache):
            md = get_forward_context().attn_metadata
            if x_scale is None:
                # what the qkv projection does with a bf16 input
                hidden_states, x_scale = quantize_fp8(hidden_states)
            q = mono.pre_launch(attn, hidden_states, x_scale, positions, md.slot_mapping, cache)
            if q is not None:
                o = None
                if mono.pa_usable(
                    getattr(self, "dspark_window", 0),
                    rows,
                    getattr(md, "dspark_bt16_rows", None),
                    getattr(md, "dspark_ctx_rows", None),
                ):
                    o = mono.pa_launch(
                        q, cache, md.dspark_bt16_rows, md.dspark_ctx_rows, attn.scaling
                    )
                if o is None:
                    o = attn._block_attention_paged(q.view(rows, -1), md, cache)
                o = o.view(rows, -1)
                if mono.POST_FUSES_OPROJ:
                    out = mono.post_launch(
                        o,
                        attn.o_proj,
                        self.mlp,
                        residual,
                        self.post_attention_layernorm,
                        next_norm,
                        next_mxfp8,
                    )
                    if out is not None:
                        return None, out[2], out
                hidden_states = linear_out(attn.o_proj(o))
                hidden_states, x_scale, residual = _ar_norm(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm,
                    self.mlp.gate_up_proj,
                )
                out = mono.mlp_launch(
                    hidden_states, x_scale, self.mlp, residual, next_norm, next_mxfp8
                )
                if out is not None:
                    return None, out[2], out
                self.mlp.fuse_swiglu_mxfp8 = self._swiglu_mxfp8 and _draft_fuse_rows(
                    hidden_states.shape[0]
                )
                return self.mlp(hidden_states, x_scale=x_scale), residual, None
        hidden_states = attn(positions, hidden_states, x_scale=x_scale)
        return self._mlp_half(hidden_states, residual)

    def _mlp_half(self, hidden_states, residual):
        hidden_states, x_scale, residual = _ar_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm,
            self.mlp.gate_up_proj,
        )
        self.mlp.fuse_swiglu_mxfp8 = self._swiglu_mxfp8 and _draft_fuse_rows(
            hidden_states.shape[0]
        )
        return self.mlp(hidden_states, x_scale=x_scale), residual, None


class MiniMaxM3DSpark(DSparkDraftModel):
    """Top-level standalone DSpark draft for MiniMax-M3.1 (GQA backbone +
    Markov head + confidence head)."""

    # Per-row MHA decode; its tables are built in `prepare_paged_block`.
    block_attention_is_mla = False

    weights_mapping: ClassVar[dict[str, str]] = {
        "language_model.model.dspark.": "",
        ".decoder_layer.": ".",
    }
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(self, atom_config: "Config", layer_offset: int = 0) -> None:
        super().__init__()
        if atom_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "The MiniMax-M3.1 DSpark draft does not support DCP."
            )
        config = atom_config.hf_config
        self.hf_config = config
        quant_config = _draft_quant_config(config)

        self.hidden_size = config.hidden_size
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.markov_rank = int(config.dspark_markov_rank)
        self.vocab_size = int(config.vocab_size)

        # The checkpoint keeps this projection BF16 (an ignored layer). The aux
        # hidden states arrive replicated. Replicated by default; under
        # ATOM_DSPARK_FUSED_AR each rank reduces its quarter of K and the context
        # norm all-reduces the partial sums (`project_context`), so a rank reads
        # a quarter of the weight; otherwise, under ATOM_DSPARK_FC_TP, each rank
        # computes 1/TP of the output columns and `project_context` all-gathers
        # them.
        self._fc_partial = _draft_fused_ar()
        self.fc_column_parallel = envs.ATOM_DSPARK_FC_TP and not self._fc_partial
        if self._fc_partial:
            fc_kind = RowParallelLinear
        elif self.fc_column_parallel:
            fc_kind = ColumnParallelLinear
        else:
            fc_kind = ReplicatedLinear
        self.fc = fc_kind(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix="fc",
            **({"reduce_results": False} if self._fc_partial else {}),
        )
        self.hidden_norm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                M3DSparkDecoderLayer(
                    atom_config,
                    config,
                    quant_config,
                    layer_offset + i,
                    prefix=f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(self.vocab_size, self.markov_rank)
        # 0 or absent: every block row attends the whole context.
        self.sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        for layer in self.layers:
            # the block attention launch bounds a row's keys by the window (dspark_draft_mono.pa_usable)
            layer.dspark_window = self.sliding_window
        self.confidence_head = (
            DSparkConfidenceHead(self.hidden_size, self.markov_rank, bias=True)
            if getattr(config, "enable_confidence_head", False)
            else None
        )
        # Whether `head_and_sample` computes the confidence head; the proposer
        # clears it when nothing consumes the confidence (no verify scheduler).
        self.emit_confidence = True

        # Bound by share_with_target(); the checkpoint ships neither.
        self.embed_tokens = None
        self.lm_head = None

        # `prepare_paged_block`'s buffers, sized here and allocated on its
        # first call.
        self.max_num_seqs = atom_config.max_num_seqs
        self.block_size = atom_config.kv_cache_block_size
        self._bt16 = None

    @property
    def first_attention_layer_num(self) -> int:
        return self.layers[0].self_attn.kv.layer_num

    # ---- weight-loading hooks ---------------------------------------------

    def share_with_target(self, target_base: nn.Module, loaded: set) -> None:
        target_vocab = target_base.model.embed_tokens.num_embeddings
        if target_vocab != self.vocab_size:
            raise ValueError(
                f"DSpark draft vocab {self.vocab_size} != target vocab "
                f"{target_vocab}. The draft shares the target's embedding and LM "
                "head, so the two must agree."
            )
        self.embed_tokens = target_base.model.embed_tokens
        self.lm_head = target_base.lm_head

    # ---- drafting entry points (called by the proposer) --------------------

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(concat(aux)))``, once per step for every layer.

        A target layer id is that layer's output, the residual stream before
        any final norm, which is what this checkpoint was trained on: feeding
        the last one through the target's final norm instead (the NeMo recipe's
        post-norm feature) dropped GSM8K acceptance from 51% to 39%.
        """
        if self._fc_partial:
            k, r = self.fc.input_size, self.fc.tp_rank
            part = linear_out(
                self.fc(aux_concat[:, r * k : (r + 1) * k].contiguous())
            )
            if _draft_fuse_rows(part.shape[0]):
                hidden, _ = fused_allreduce_gemma_rms_norm(
                    part, torch.zeros_like(part), self.hidden_norm
                )
                return hidden
            return self.hidden_norm(tensor_model_parallel_all_reduce(part))
        ctx = linear_out(self.fc(aux_concat))
        if self.fc_column_parallel:
            # This rank's output columns -> the whole row, in rank order (the
            # loader gives rank r rows r * hidden / TP on of the weight). RCCL:
            # a prefill chunk's rows exceed the custom all-gather's buffer, and
            # this runs eagerly after the target forward, never under capture.
            ctx = tensor_model_parallel_all_gather(ctx, use_custom=False, dim=-1)
        return self.hidden_norm(ctx)

    @property
    def context_layers(self):
        return self.layers

    def _context_kv(self):
        """Every layer's context K/V as one GEMM (dspark_context_kv.py), built
        on first use from the processed weights; None (per-layer path) unless
        the fused-QKV linears are preshuffled MXFP8 1x32."""
        ckv = getattr(self, "_ckv", False)
        if ckv is False:
            ckv = None
            lins = [layer.self_attn.qkv_proj for layer in self.layers]
            if all(
                getattr(lin, "native_a8_group_rows", None) == 1
                and getattr(lin.weight, "is_shuffled", False)
                for lin in lins
            ):
                from atom.model_ops.dspark_context_kv import ContextKV

                a = self.layers[0].self_attn
                ckv = ContextKV(lins, a.q_size, a.kv_size)
            self._ckv = ckv
        return ckv

    def _context_kv_write(self):
        """Every layer's context K/V into its cache in one launch
        (dspark_context_kv.ContextKVWrite), built once the pool is bound; None
        (a launch a layer) unless the layers' caches are one pool's alike,
        equally spaced views."""
        bound = self.layers[0].self_attn.kv.k_cache
        key = bound.data_ptr() if bound.numel() else None
        w = getattr(self, "_ckv_write", False)
        if w is False or self._ckv_write_key != key:
            from atom.model_ops.dspark_context_kv import ContextKVWrite

            attns = [layer.self_attn for layer in self.layers]
            caches = [a._bound_cache() for a in attns]
            if any(c is None for c in caches):
                return None  # the pool is not bound yet: decide later
            # Rebuilt whenever the pool is (re)bound: it holds layer 0's views.
            w = self._ckv_write = ContextKVWrite.build(attns, caches)
            self._ckv_write_key = key
        return w

    # The proposer may capture `write_context_kv` (with an explicit slot
    # mapping) in a CUDA graph for decode steps.
    context_kv_graphable = True

    def write_context_kv(self, aux_concat, positions, slot_mapping=None) -> None:
        """The base's absorb, with every layer's K/V projected in one GEMM
        (bit-identical per column to the per-layer fused-QKV GEMMs).
        ``slot_mapping``: the rows' cache slots; None reads the target
        forward's from the forward context."""
        from atom.utils.forward_context import get_forward_context

        if get_forward_context().context.is_dummy_run:
            return
        ctx_hidden = self.project_context(aux_concat)
        ckv = self._context_kv()
        kv = ckv(ctx_hidden) if ckv is not None else None
        if slot_mapping is None:
            slot_mapping = get_forward_context().attn_metadata.slot_mapping[
                : ctx_hidden.shape[0]
            ]
        if kv is None:
            for layer in self.layers:
                layer.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping)
            return
        writer = self._context_kv_write()
        if writer is not None:
            writer(kv, positions, slot_mapping)
            return
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            o = i * 2 * a.kv_size
            layer.self_attn.write_context_kv_rows(
                kv[:, o : o + a.kv_size],
                kv[:, o + a.kv_size : o + 2 * a.kv_size],
                positions,
                slot_mapping,
            )

    def _alloc_block_rows(self, P: int, T: int, device) -> None:
        """The first call is the warmup, ahead of any capture: these addresses
        are the ones every recording bakes in."""
        max_bs = self.max_num_seqs
        self._alloc = torch.zeros(max_bs, dtype=torch.int32, device=device)
        # Pinned staging, a ring of _ALLOC_RING: the host can run ahead of the
        # device, so a slot is rewritten only once its copy has read it, and
        # with a ring that copy is steps old -- the wait on its event returns
        # at once rather than syncing on the step just issued.
        self._alloc_host = torch.zeros(
            _ALLOC_RING, max_bs, dtype=torch.int32, pin_memory=True
        )
        # Recorded after each copy out of a slot; a fresh event has nothing to
        # wait for.
        self._alloc_copied = [torch.cuda.Event() for _ in range(_ALLOC_RING)]
        self._alloc_slot = 0
        # One row of slack past the last request: `_window_rows` reads every
        # row as a P-wide window of this buffer.
        self._bt16_flat = torch.zeros(
            (max_bs + 1) * P, dtype=torch.int32, device=device
        )
        self._bt16 = self._bt16_flat[: max_bs * P].view(max_bs, P)
        self._bt16_rows = torch.zeros(max_bs * T, P, dtype=torch.int32, device=device)
        self._ctx_rows = torch.zeros(max_bs * T, dtype=torch.int32, device=device)
        self._row_req = torch.arange(
            max_bs, dtype=torch.int64, device=device
        ).repeat_interleave(T)
        self._sub_pages = torch.arange(
            PAGES_PER_SPARSE_BLOCK, dtype=torch.int32, device=device
        )
        self._start16 = torch.zeros(max_bs * T, dtype=torch.int64, device=device)
        self._ctx_eff = torch.zeros(max_bs, dtype=torch.int32, device=device)

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] page ids (page-128)
        block_positions: torch.Tensor,  # [running_bs, T]
        slots: torch.Tensor,  # [running_bs, T] -- the block's KV write slots
        ctx_lens: torch.Tensor,  # [running_bs] anchor + 1 + T
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
    ) -> None:
        """Per-row paged decode tables for the block.

        The block is attended as ``running_bs * T`` independent q_len=1 decode
        rows -- non-causal by construction -- each over its request's
        ``[context | block]`` through a page-16 block table, the layout the
        target's sparse decode kernels read. Everything lands in buffers sized at
        ``max_num_seqs`` and is only sliced, so a captured block replays against
        this step's numbers.

        Block positions can run past the pages the scheduler has allocated for
        the request (it allocates through the verify window, the block reaches
        up to T past the accepted anchor). A table column that was never
        allocated reads as page 0, which is some other request's page -- under
        prefix caching, a prefix every request shares. Such positions are
        neither written (slot -1) nor read (the row's context is capped at what
        is allocated): their draft rows simply miss a few of their own keys.
        """
        running_bs, T = block_positions.shape
        alloc = self._stage_allocated_pages(
            block_tables, T, allocated_pages, scheduled_bs, running_bs
        )
        alloc.mul_(self.block_size)

        slots.masked_fill_(block_positions >= alloc.view(running_bs, 1), -1)
        # The rows' own lengths; `ctx_lens` is left as the proposer built it.
        ctx_eff = self._ctx_eff[:running_bs]
        torch.minimum(ctx_lens, alloc, out=ctx_eff)

        bt16 = _page16_table(block_tables, self._bt16[:running_bs], self._sub_pages)
        if running_bs > scheduled_bs:
            # A pad row reads one page: page 0.
            bt16[scheduled_bs:, :PAGES_PER_SPARSE_BLOCK].copy_(self._sub_pages)

        rows = running_bs * T
        _window_rows(
            self._bt16_flat,
            ctx_eff,
            block_positions.view(-1),
            self._row_req[:rows],
            self.sliding_window,
            scheduled_bs * T,
            self._start16[:rows],
            self._bt16_rows[:rows],
            self._ctx_rows[:rows],
        )
        self._publish_paged_block(attn_metadata, rows)

    def prepare_paged_block_fused(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] page ids (page-128)
        anchor_positions: torch.Tensor,  # [running_bs] int64
        offsets: torch.Tensor,  # [T] int64, 1..T
        block_positions: torch.Tensor,  # [running_bs, T] out
        slots: torch.Tensor,  # [running_bs, T] out -- the block's KV write slots
        ctx_lens: torch.Tensor,  # [running_bs] out, anchor + 1 + T
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
        *,
        pad_ctx_len: int,
        block_size: int,
        stage=None,
    ) -> bool:
        """The proposer's block positions, slots and context lengths (no DCP)
        and `prepare_paged_block`, in ONE launch (`dspark_block_meta`): the
        same numbers as those ~30 small launches. False when it does not
        apply; the caller then builds them as before."""
        if block_size != self.block_size or not block_tables.is_cuda:
            return False
        from atom.model_ops.dspark_block_meta import PACKED_MAX_ROWS, block_meta

        running_bs, T = block_positions.shape
        W = block_tables.shape[1]
        if running_bs <= PACKED_MAX_ROWS and W < 1 << 16:
            # A few requests: their page counts go in as kernel scalars, no
            # staging copy (a count is at most W).
            self._ensure_block_rows(block_tables, T)
            n = min(len(allocated_pages), scheduled_bs)
            alloc = [max(int(p), 1) for p in allocated_pages[:n]]
            alloc += [1] * (running_bs - n)
        else:
            alloc = self._stage_allocated_pages(
                block_tables, T, allocated_pages, scheduled_bs, running_bs
            )
        rows = running_bs * T
        block_meta(
            anchor_positions,
            offsets,
            block_tables,
            alloc,
            block_positions,
            slots.view(-1),
            ctx_lens,
            self._ctx_eff[:running_bs],
            self._bt16_flat,
            self._bt16_rows[:rows],
            self._ctx_rows[:rows],
            block_size=block_size,
            sub_pages=PAGES_PER_SPARSE_BLOCK,
            window_page=ASM_PAGE_SIZE,
            window=self.sliding_window,
            scheduled_bs=scheduled_bs,
            pad_ctx_len=pad_ctx_len,
            stage=stage,
        )
        self._publish_paged_block(attn_metadata, rows)
        return True

    def _ensure_block_rows(self, block_tables, T) -> None:
        P = PAGES_PER_SPARSE_BLOCK * block_tables.shape[1]
        if self._bt16 is None or self._bt16.shape[1] != P:
            self._alloc_block_rows(P, T, block_tables.device)

    def _stage_allocated_pages(
        self, block_tables, T, allocated_pages, scheduled_bs, running_bs
    ) -> torch.Tensor:
        """Pages each request holds, host -> ``self._alloc[:running_bs]``.

        At least one page: the warmup batch and the pad tail address page 0 by
        convention."""
        self._ensure_block_rows(block_tables, T)
        k = self._alloc_slot
        self._alloc_slot = (k + 1) % _ALLOC_RING
        self._alloc_copied[k].synchronize()
        pages = self._alloc_host[k].numpy()[:running_bs]
        n = min(len(allocated_pages), scheduled_bs)
        pages[:n] = allocated_pages[:n]
        pages[n:] = 1
        pages.clip(min=1, out=pages)
        alloc = self._alloc[:running_bs]
        alloc.copy_(self._alloc_host[k, :running_bs], non_blocking=True)
        self._alloc_copied[k].record()
        return alloc

    def _publish_paged_block(self, attn_metadata, rows) -> None:
        attn_metadata.dspark_bt16_rows = self._bt16_rows[:rows]
        attn_metadata.dspark_ctx_rows = self._ctx_rows[:rows]
        if self.markov_head.fused_sample:
            from atom.model_ops.dspark_markov_chain import poll_errors

            poll_errors()
        from atom.model_ops import dspark_draft_mono

        dspark_draft_mono.poll_errors()

    def forward_spec(self, input_ids, positions, num_draft):
        return self.head_and_sample(
            self.block_backbone(input_ids, positions, num_draft),
            input_ids,
            num_draft,
        )

    def block_backbone(
        self,
        input_ids: torch.Tensor,  # [B]   verified anchor token per request
        positions: torch.Tensor,  # [B*T] block absolute positions
        num_draft: int,
    ) -> torch.Tensor:
        """Embed ``[anchor, MASK, ...]``, run the layers, final norm.

        Returns the post-final-norm hidden states ``[B*T, hidden]``. No
        embedding scale: the reference embeds with the target's table as is.
        """
        bs = input_ids.shape[0]
        draft_ids = input_ids.new_full((bs, num_draft), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden, residual = self._embed_block(draft_ids.view(-1))
        if _draft_fused_ar() and _draft_fuse_rows(hidden.shape[0]):
            # Each layer as two launches around the paged attention where they serve (dspark_draft_mono.py).
            from atom.models.minimax_m3.model import _linear_consumes_mxfp8

            normed = None
            n = len(self.layers)
            for i, layer in enumerate(self.layers):
                last = i + 1 == n
                nxt = self.final_norm if last else self.layers[i + 1]
                next_mxfp8 = (
                    not last
                    and envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8
                    and _linear_consumes_mxfp8(nxt.self_attn.qkv_proj)
                )
                hidden, residual, normed = layer.forward_mono(
                    positions,
                    hidden,
                    residual,
                    normed,
                    nxt if last else nxt.input_layernorm,
                    next_mxfp8,
                )
            if normed is not None:
                return normed[0]
            hidden, _ = fused_allreduce_gemma_rms_norm(hidden, residual, self.final_norm)
            return hidden
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        if _draft_fused_ar() and _draft_fuse_rows(hidden.shape[0]):
            hidden, _ = fused_allreduce_gemma_rms_norm(
                hidden, residual, self.final_norm
            )
        elif _draft_fused_ar():
            # Past ATOM_DSPARK_FUSED_AR_MAX_ROWS: the down_proj's all-reduce,
            # unfused, as the original path runs it.
            hidden = tensor_model_parallel_all_reduce(hidden)
            hidden, _ = self.final_norm(hidden, residual)
        else:
            hidden, _ = self.final_norm(hidden, residual)
        return hidden

    def _embed_block(self, ids: torch.Tensor):
        """The block's embedding and the residual layer 0 starts from.

        Under ATOM_DSPARK_FUSED_AR at a fused row count, the vocabulary-parallel
        lookup stays TP-partial with a zero residual: layer 0's input norm then
        all-reduces it with the norm and the MXFP8 quant in one launch, as every
        later layer's input norm does, instead of the embedding's own all-reduce
        and an unfused norm. Exactly one rank holds each row (the others add
        zeros), so the reduced embedding is the same value either way."""
        embed = self.embed_tokens
        if (
            _draft_fused_ar()
            and _draft_fuse_rows(ids.shape[0])
            and isinstance(embed, VocabParallelEmbedding)
            and embed.tp_size > 1
        ):
            hidden = masked_embedding(
                ids, embed.weight, embed.vocab_start_idx, embed.vocab_end_idx
            )
            return hidden, torch.zeros_like(hidden)
        return embed(ids), None

    def _sharded_markov_sample(self) -> bool:
        """``ATOM_DSPARK_SHARDED_MARKOV``: sample the block off this rank's
        vocab shard of the LM head. The whole-vocab path all-gathers the
        ``[B * T, V]`` logits for a sampler that only wants each position's
        argmax; here the ranks exchange ``(max, id)`` per position (T gathers of
        ``[B, 2]``), as the DeepSeek-V4.1 DSpark draft does. Same ids."""
        head, markov = self.lm_head, self.markov_head
        return (
            envs.ATOM_DSPARK_SHARDED_MARKOV
            and head.tp_size > 1
            and head.bias is None
            and markov.fused_sample
            and markov.markov_w1.weight.dtype in (torch.bfloat16, torch.float16)
            and markov.markov_w2.weight.dtype is markov.markov_w1.weight.dtype
        )

    def head_and_sample(self, hidden, anchor_ids, num_draft):
        bs = anchor_ids.shape[0]
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        if self._sharded_markov_sample():
            # Imported here, as sample_next does: the op imports Triton.
            from atom.model_ops.dspark_markov_sample import dspark_markov_argmax_tp

            head, markov = self.lm_head, self.markov_head
            # [B, T, V / tp]: this rank's vocab shard, ids vocab_start_idx ..
            base_logits = tgemm.mm(hidden, head.weight).view(bs, num_draft, -1)
            w1 = markov.markov_w1.weight
            markov_stack = w1.new_empty(bs, num_draft, w1.shape[1])
            for k in range(num_draft):
                dspark_markov_argmax_tp(
                    base_logits[:, k],
                    out_ids[:, k],
                    w1,
                    markov.markov_w2.weight,
                    out_ids[:, k + 1],
                    head.vocab_start_idx,
                    get_tp_group(),
                    use_custom=envs.ATOM_USE_CUSTOM_ALL_GATHER,
                    embed_out=markov_stack[:, k],
                )
        else:
            base_logits = self.lm_head(hidden).view(bs, num_draft, -1)
            markov_stack = None
            if self.markov_head.fused_sample:
                # One request: all K steps in one launch, bit-identical to the
                # two-stage fused path below (dspark_markov_chain.py); None where
                # its guard does not hold.
                from atom.model_ops.dspark_markov_chain import markov_chain_bs1

                embeds = markov_chain_bs1(
                    base_logits,
                    out_ids,
                    self.markov_head.markov_w1.weight,
                    self.markov_head.markov_w2.weight,
                )
                if embeds is not None:
                    markov_stack = embeds.unsqueeze(0)
        if markov_stack is None:
            markov_embeds = []
            for k in range(num_draft):
                # Greedy: sampling temperature is the target verify's business.
                markov_embeds.append(
                    self.markov_head.sample_next(
                        out_ids[:, k], base_logits[:, k], out_ids[:, k + 1]
                    )
                )
            markov_stack = torch.stack(markov_embeds, dim=1)
        confidence = None
        if self.confidence_head is not None and self.emit_confidence:
            confidence = self.confidence_head(
                hidden.view(bs, num_draft, -1), markov_stack
            )
        return out_ids[:, 1:], confidence
