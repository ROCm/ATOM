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
  * The context is ``hidden_norm(fc(concat(taps)))`` and goes straight into each
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
import triton
import triton.language as tl
from aiter import QuantType, dtypes
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.rotary_embedding import get_rope
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.config import QuantizationConfig
from atom.distributed.ulysses_sp import sp_is_enabled
from atom.kv_transfer.disaggregation import kv_config_has_producer
from atom.model_ops.blockscale import native_quant_linear
from atom.model_ops.dspark_markov_argmax import markov_weights_unsupported
from atom.model_ops.dspark_markov_stage1 import (
    BLOCK_K,
    BLOCK_V,
    markov_launch_geometry,
    markov_logits_tile_of,
)
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from atom.model_ops.minimax_m3.sparse_attn import (
    ASM_PAGE_SIZE,
    PAGES_PER_SPARSE_BLOCK,
    sparse_pa_per_row,
    to_page16_shuffle,
)
from atom.model_ops.tile_argmax import argmax_lowest_id
from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
    triton_fused_norm_rope_cache,
)
from atom.models.dspark_draft import (
    AllocationCap,
    DSparkConfidenceHead,
    DSparkDraftModel,
    DSparkGQADecoderLayer,
    KVCacheBinding,
    draft_quant_config,
    linear_out,
)
from atom.models.minimax_m3.draft_fused import gemma_rmsnorm_mxfp8
from atom.models.minimax_m3.model import MiniMaxM3MLP
from atom.models.minimax_m3.packed_argmax import (
    EMPTY,
    EMPTY_K,
    pack_max_id,
    ranks_best,
    unpack_id,
)
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

if TYPE_CHECKING:
    from atom.config import Config


def _draft_rope_theta(config) -> float:
    # transformers 5 moves rope_theta into rope_parameters on the M3 VL text
    # config class this draft loads as; the plain-dict target keeps it flat.
    params = getattr(config, "rope_parameters", None) or {}
    theta = params.get("rope_theta") or getattr(config, "rope_theta", None)
    if not theta:
        raise ValueError("MiniMax-M3 DSpark draft config names no rope_theta.")
    return float(theta)


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


# Request counts, table widths and ids vary per step: no per-value variants
# (Triton would otherwise specialize on ==1 and divisibility by 16 and compile
# a new one mid-serving or mid-capture).
@triton.jit(do_not_specialize=["bt_stride", "vocab_max", "scheduled_bs", "W"])
def _block_meta_kernel(
    anchor_ptr,  # [running_bs] int64, each row's anchor position; STAGE: written
    bt_ptr,  # [running_bs, W] int32 page-128 ids, row stride bt_stride
    bt_stride,
    alloc_ptr,  # [running_bs] int32, pages each row holds (pad rows: read as 1)
    next_ids_ptr,  # STAGE: [scheduled_bs] each row's verified anchor token
    positions_ptr,  # STAGE: the target step's positions
    rows_ptr,  # STAGE: [scheduled_bs] each anchor's row in them
    ids_out_ptr,  # STAGE: [running_bs] the block's anchor ids (0 on pad rows)
    vocab_max,
    pos_ptr,  # [running_bs, T] int64 out: the block's positions
    slot_ptr,  # [running_bs, T] int64 out: its KV slots, -1 dropped
    ctx_ptr,  # [running_bs] int32 out: anchor + 1 + T (PAD_CTX on pad rows)
    rows_bt_ptr,  # [running_bs * T, PW] int32 out: each row's page-16 window table
    rows_len_ptr,  # [running_bs * T] int32 out: each row's key count from its table start
    scheduled_bs,
    W,
    T: tl.constexpr,
    TP: tl.constexpr,
    ENTRY: tl.constexpr,
    WINDOW: tl.constexpr,
    PW: tl.constexpr,
    PWP: tl.constexpr,
    PAD_CTX: tl.constexpr,
    STAGE: tl.constexpr,
):
    """One request row of the block's metadata, everything the generic torch
    path (``DSparkProposer._build_paged_block_metadata`` + ``prepare_paged_block``)
    builds in ~30 launches: positions, slots (dropped past the table, past the
    allocation and on pad rows), the context length, and each block row's
    page-16 table from the page holding its window's first key, with its key
    count. The tables hold only the ``PW`` pages a window can span rather than
    the whole context's; the per-row decode reads a row's table up to its key
    count, so the read is the same. ``STAGE``: also stages the row's anchor id
    and position (``_stage_anchors_kernel``) before using them."""
    i = tl.program_id(0)
    real = i < scheduled_bs
    if STAGE:
        token = tl.load(next_ids_ptr + i, mask=real, other=0)
        tl.store(
            ids_out_ptr + i, tl.minimum(tl.maximum(token, 0), vocab_max).to(tl.int32)
        )
        row = tl.load(rows_ptr + i, mask=real, other=0)
        anchor = tl.load(positions_ptr + row, mask=real, other=0).to(tl.int64)
        tl.store(anchor_ptr + i, anchor)
    else:
        anchor = tl.load(anchor_ptr + i)
    alloc_tok = (
        tl.where(real, tl.maximum(tl.load(alloc_ptr + i), 1), 1).to(tl.int64) * ENTRY
    )
    j = tl.arange(0, TP)
    jm = j < T
    pos = anchor + 1 + j
    tl.store(pos_ptr + i * T + j, pos, mask=jm)
    page = pos // ENTRY
    in_table = page < W
    bt_row = bt_ptr + i.to(tl.int64) * bt_stride
    pg = tl.load(bt_row + tl.minimum(page, W - 1), mask=jm, other=0).to(tl.int64)
    slot = pg * ENTRY + pos % ENTRY
    slot = tl.where(in_table & real & (pos < alloc_tok), slot, -1)
    tl.store(slot_ptr + i * T + j, slot, mask=jm)
    ctx = tl.where(real, anchor + 1 + T, PAD_CTX)
    tl.store(ctx_ptr + i, ctx.to(tl.int32))
    ctx_eff = tl.minimum(ctx, alloc_tok)
    start16 = tl.maximum(pos - (WINDOW - 1), 0) // 16
    start16 = tl.where(real, start16, 0)
    tl.store(rows_len_ptr + i * T + j, (ctx_eff - start16 * 16).to(tl.int32), mask=jm)
    e = tl.arange(0, PWP)
    p16 = start16[:, None] + e[None, :]
    p128 = p16 // 8
    em = jm[:, None] & (e[None, :] < PW)
    pgs = tl.load(bt_row + p128, mask=em & (p128 < W), other=0)
    entry = tl.where(real, pgs * 8 + (p16 % 8), e[None, :] % 8)
    out = (i * T + j)[:, None].to(tl.int64) * PW + e[None, :]
    tl.store(rows_bt_ptr + out, entry.to(tl.int32), mask=em)


@triton.jit(do_not_specialize=["vocab_start", "vocab_end", "noise_id"])
def _block_embed_kernel(
    anchor_ptr,  # [B] anchor ids
    weight_ptr,  # [V / tp, D] this rank's rows of the table
    out_ptr,  # [B * T, D]
    vocab_start,
    vocab_end,
    noise_id,
    stride_w,
    D,
    T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    tok = tl.where(r % T == 0, tl.load(anchor_ptr + r // T), noise_id)
    in_range = (tok >= vocab_start) & (tok < vocab_end)
    emb = tl.load(
        weight_ptr + (tok - vocab_start).to(tl.int64) * stride_w + cols,
        mask=in_range & (cols < D),
        other=0.0,
    )
    tl.store(out_ptr + r.to(tl.int64) * D + cols, emb, mask=cols < D)


@triton.jit
def _ranks_best_id(gathered_ptr, offs_row, row_mask, num_rows, W: tl.constexpr):
    """The id of the largest packed value over the W ranks' rows ``[W, B]``."""
    return unpack_id(ranks_best(gathered_ptr, offs_row, row_mask, num_rows, W, 1))


@triton.jit
def _markov_step_atomic(
    base_ptr,  # this position's base logits, this rank's vocab shard, rows strided
    prev_ids_ptr,  # FIRST: [B] the anchors
    gathered_ptr,  # not FIRST: [W, B] int64, every rank's packed best of the previous position
    prev_packed_ptr,  # not FIRST: [B] int64, this rank's packed best of the previous position (reset here)
    out_prev_ptr,  # not FIRST: the previous position's column of the [B, T] ids
    w1_ptr,
    w2_ptr,
    packed_ptr,  # [B] int64: this position's best on this rank, atomic_max (reset by the next launch)
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_prev,
    stride_out,
    stride_w1_vocab,
    stride_w2_vocab,
    vocab_offset,
    FIRST: tl.constexpr,
    W: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One block position of the sharded greedy Markov block in one launch: the
    previous position's cross-rank winner (tile 0 also writes it into the
    previous column and resets this rank's previous packed best), the bias
    GEMV of `markov_logits_tile_of`, and each tile's (max, lowest id) folded
    into the row's packed best by ``atomic_max``: the reduction launch of
    `dspark_markov_argmax` without the launch. The same values and tie rule,
    so the same ids."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    if FIRST:
        prev = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)
    else:
        prev = _ranks_best_id(gathered_ptr, offs_row, row_mask, num_rows, W)
        if tile == 0:
            tl.store(out_prev_ptr + offs_row * stride_out, prev, mask=row_mask)
            tl.store(
                prev_packed_ptr + offs_row,
                tl.full([BLOCK_ROW], EMPTY_K, tl.int64),
                mask=row_mask,
            )
    vals = markov_logits_tile_of(
        prev, base_ptr, w1_ptr, w2_ptr, base_ptr, tile, offs_row, offs_v, row_mask,
        v_mask, rank, stride_base_row, stride_w1_vocab, 0, stride_w2_vocab,
        vocab_offset, BLOCK_ROW, BLOCK_V, BLOCK_K, False,
    )  # fmt: skip
    tile_max, tile_idx = argmax_lowest_id(
        vals, offs_v[None, :], v_mask[None, :], 2147483647
    )
    hit = tile_idx != 2147483647
    packed = pack_max_id(tile_max, tile_idx + vocab_offset)
    tl.atomic_max(packed_ptr + offs_row, packed, mask=row_mask & hit, sem="relaxed")


@triton.jit
def _markov_last_atomic(
    gathered_ptr,
    packed_ptr,
    out_ptr,
    num_rows,
    stride_out,
    W: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """The last position's cross-rank winner into its column; resets this rank's
    packed best for the next block."""
    offs_row = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = offs_row < num_rows
    best_id = _ranks_best_id(gathered_ptr, offs_row, row_mask, num_rows, W)
    tl.store(out_ptr + offs_row * stride_out, best_id, mask=row_mask)
    tl.store(
        packed_ptr + offs_row, tl.full([BLOCK_ROW], EMPTY_K, tl.int64), mask=row_mask
    )


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
            reduce_results=True,
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
        self.kv = KVCacheBinding(layer_num, page_tokens=ASM_PAGE_SIZE)

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
        if not kv.bound:
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

        The q third of the fused projection is computed and dropped; the kernel
        writes K/V only when told it has zero query heads.
        """
        cache = self._bound_cache()
        if cache is None:
            return
        qkv = linear_out(self.qkv_proj(ctx_hidden))
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
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
        then decode per row. ``x_scale``: ``hidden_states`` is already MXFP8.
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
        )
        return out.view(rows, -1)


def _is_native_mxfp8(lin) -> bool:
    """``lin`` runs the native MXFP8 1x32 path on a quantized input (not a
    weight dequantized to bf16, not another weight format)."""
    weight = getattr(lin, "weight", None)
    return (
        getattr(lin, "native_a8_group_rows", None) == 1
        and not getattr(lin, "runs_bf16", False)
        and weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and getattr(lin, "weight_scale", None) is not None
    )


class M3DSparkDecoderLayer(DSparkGQADecoderLayer):
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
            reduce_results=True,
            prefix=f"{prefix}.mlp",
            # The checkpoint says `swigluoai`, and the reference draft MLP is
            # SwiGLU-OAI whatever the field says; transformers'
            # MiniMaxM3VLTextConfig, which this config loads as, rewrites the
            # field to its `silu` default.
            hidden_act="swigluoai",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, hidden_states, residual=None):
        # Both norms feed a native MXFP8 linear (qkv_proj, gate_up_proj): write
        # its quantized input directly instead of bf16 + a quantize pass.
        # Checked per call: a linear's mode can change as the weights load.
        if not all(
            _is_native_mxfp8(lin)
            for lin in (self.self_attn.qkv_proj, self.mlp.gate_up_proj)
        ):
            return super().forward(positions, hidden_states, residual)
        norm = self.input_layernorm
        if residual is None:
            residual = hidden_states
            xq, xs, _ = gemma_rmsnorm_mxfp8(
                hidden_states, norm.weight, norm.variance_epsilon
            )
        else:
            xq, xs, residual = gemma_rmsnorm_mxfp8(
                hidden_states, norm.weight, norm.variance_epsilon, residual
            )
        hidden_states = self.self_attn(positions, xq, x_scale=xs)
        norm = self.post_attention_layernorm
        xq, xs, residual = gemma_rmsnorm_mxfp8(
            hidden_states, norm.weight, norm.variance_epsilon, residual
        )
        return self.mlp(xq, x_scale=xs), residual


class MiniMaxM3DSpark(DSparkDraftModel):
    """Top-level standalone DSpark draft for MiniMax-M3.1 (GQA backbone +
    Markov head + confidence head).

    The block is the base ``block_backbone``, with no embedding scale: the
    reference embeds with the target's table as is.
    """

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
        # The draft checkpoint is MXFP8 (``[1, 32]`` blocks) whatever the
        # target is -- the M3.1 targets come as MXFP8 + NVFP4 experts, Quark
        # MXFP4, or BF16.
        quant_config = draft_quant_config(
            config,
            draft="MiniMax-M3 DSpark",
            accepted={(QuantType.per_1x32, dtypes.fp8)},
        )

        self.hidden_size = config.hidden_size
        self.vocab_size = int(config.vocab_size)

        # Replicated: the aux hidden states arrive replicated, and the
        # checkpoint keeps this projection BF16 (an ignored layer).
        self.fc = ReplicatedLinear(
            self.hidden_size * len(config.dspark_target_layer_ids),
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix="fc",
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
        self.init_block_fields(config)
        # 0 or absent: every block row attends the whole context.
        self.sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        self.confidence_head = (
            DSparkConfidenceHead(self.hidden_size, self.markov_head.rank, bias=True)
            if getattr(config, "enable_confidence_head", False)
            else None
        )
        # Only a confidence-scheduled verify reads the head (the proposer drops
        # it otherwise), so the block skips its GEMM.
        self.emit_confidence = self.confidence_head is not None and bool(
            atom_config.dspark.confidence_schedule
        )

        # Bound by share_with_target(); the checkpoint ships neither.
        self.embed_tokens = None
        self.lm_head = None

        # `prepare_paged_block`'s buffers, sized here and allocated on its
        # first call.
        self.max_num_seqs = atom_config.max_num_seqs
        self._bt16 = None
        # `_context_kv_weight`, built on first use (after the weights load)
        self._ctx_kv = None
        # `_markov_atomic`'s per-position packed bests, allocated with the
        # block rows
        self._mk_packed = None

    def _context_kv_weight(self):
        """The K/V rows of every context layer's qkv_proj as one MXFP8 weight
        ``[layers * 2 * kv, hidden]`` (and its E8M0 scales): one GEMM gives
        every layer's context K/V, the q rows each layer's projection would
        compute and drop are not read. The bytes are the layers' own, copied
        (``make_row_view`` slices whole 16-row shuffle tiles). None where a
        layer's projection is not native MXFP8 1x32. The copy is made once,
        after the weights load; weights are not updated in place afterwards."""
        if self._ctx_kv is None:
            if torch.cuda.is_current_stream_capturing():
                # built only eagerly: a recording would put it in its private pool
                return None
            attns = [layer.self_attn for layer in self.context_layers]
            projs = [a.qkv_proj for a in attns]
            a0 = attns[0]
            if not all(_is_native_mxfp8(p) for p in projs) or any(
                p.bias is not None or getattr(p, "is_output_padded", False)
                for p in projs
            ):
                self._ctx_kv = ()
            else:
                try:
                    views = [p.make_row_view(a0.q_size, 2 * a0.kv_size) for p in projs]
                except ValueError:  # offsets not tile-aligned at this TP size
                    self._ctx_kv = ()
                else:
                    self._ctx_kv = (
                        torch.cat([v.weight.data for v in views], dim=0),
                        torch.cat([v.weight_scale.data for v in views], dim=0),
                        bool(getattr(views[0].weight, "is_shuffled", False)),
                    )
        return self._ctx_kv or None

    def write_context_kv(
        self, taps, positions, slot_mapping=None, pcp_num_tokens=None
    ) -> None:
        """The context K/V of every layer from one GEMM over one quantized
        ``hidden_norm(fc(taps))`` (see ``_context_kv_weight``)."""
        kv_weight = None
        if (
            pcp_num_tokens is None
            and not get_forward_context().context.is_dummy_run
            and self.layers[0].self_attn.kv.bound
        ):
            kv_weight = self._context_kv_weight()
        if kv_weight is None:
            return super().write_context_kv(
                taps, positions, slot_mapping, pcp_num_tokens
            )
        weight, scale, shuffled = kv_weight
        norm = self.hidden_norm
        xq, xs, _ = gemma_rmsnorm_mxfp8(
            linear_out(self.fc(self.concat_taps(taps))),
            norm.weight,
            norm.variance_epsilon,
        )
        kv = native_quant_linear(
            xq,
            weight,
            scale,
            x_scale=xs,
            weight_group_rows=1,
            weight_preshuffled=shuffled,
        )
        for i, layer in enumerate(self.context_layers):
            attn = layer.self_attn
            k = kv[:, (2 * i) * attn.kv_size : (2 * i + 1) * attn.kv_size]
            v = kv[:, (2 * i + 1) * attn.kv_size : (2 * i + 2) * attn.kv_size]
            attn._norm_rope_write(
                k, k, v, 0, positions, slot_mapping, attn._bound_cache()
            )

    # ---- drafting entry points (called by the proposer) --------------------

    def _alloc_block_rows(self, P: int, T: int, device) -> None:
        """The first call is the warmup, ahead of any capture: these addresses
        are the ones every recording bakes in."""
        max_bs = self.max_num_seqs
        self._alloc_cap = AllocationCap(max_bs, device)
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
        self._alloc_markov_packed(T, device)

    def _alloc_markov_packed(self, num_draft: int, device) -> None:
        self._mk_packed = torch.full(
            (num_draft, self.max_num_seqs), EMPTY, dtype=torch.int64, device=device
        )

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] page ids (page-128)
        block_positions: torch.Tensor,  # [running_bs, T]
        slots: torch.Tensor,  # [running_bs, T] -- the block's KV write slots
        ctx_lens: torch.Tensor,  # [running_bs] anchor + 1 + T
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
        entry_tokens: int,  # tokens one page id spans, the slots' unit
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
        P = PAGES_PER_SPARSE_BLOCK * block_tables.shape[1]
        if self._bt16 is None or self._bt16.shape[1] != P:
            self._alloc_block_rows(P, T, block_tables.device)

        alloc = self._alloc_cap.apply(
            allocated_pages, scheduled_bs, block_positions, slots, entry_tokens
        )
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
        attn_metadata.dspark_bt16_rows = self._bt16_rows[:rows]
        attn_metadata.dspark_ctx_rows = self._ctx_rows[:rows]

    def embed_block(self, input_ids, num_draft):
        """The base block embedding, each row's id chosen in the lookup kernel
        (the anchor at column 0, MASK past it); the same masked lookup and
        all-reduce as the target's ``VocabParallelEmbedding``."""
        embed = self.embed_tokens
        if not hasattr(embed, "vocab_start_idx"):
            return super().embed_block(input_ids, num_draft)
        weight = embed.weight
        assert weight.stride(1) == 1
        rows, dim = input_ids.shape[0] * num_draft, weight.shape[1]
        out = torch.empty(rows, dim, dtype=weight.dtype, device=weight.device)
        block_d = 1024
        _block_embed_kernel[(rows, triton.cdiv(dim, block_d))](
            input_ids, weight, out, embed.vocab_start_idx,
            embed.vocab_start_idx + weight.shape[0], self.noise_token_id,
            weight.stride(0), dim, T=num_draft, BLOCK_D=block_d,
        )  # fmt: skip
        if embed.tp_size > 1:
            out = get_tp_group().all_reduce(out, ca_fp8_quant=False)
        return out

    def fused_block_metadata_ok(self) -> bool:
        """Whether `build_block_metadata` serves this draft: a sliding window
        bounds the pages a block row reads."""
        return self.sliding_window > 0

    def build_block_metadata(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] page-128 ids
        anchor_positions: torch.Tensor,  # [running_bs] int64
        block_positions: torch.Tensor,  # [running_bs, T] int64 out
        slots: torch.Tensor,  # [running_bs, T] int64 out
        ctx_lens: torch.Tensor,  # [running_bs] int32 out
        allocated_pages,  # [>= scheduled_bs] host, or a device int32 tensor
        scheduled_bs: int,
        entry_tokens: int,
        pad_ctx_len: int,
        stage=None,  # (next_token_ids, positions, rows, ids_out): stage the anchors too
    ) -> None:
        """The block's metadata in one launch (`_block_meta_kernel`): what the
        proposer's generic path and `prepare_paged_block` build, with compact
        window tables. Buffers are allocated at ``max_num_seqs`` on the first
        call (the warmup, ahead of any capture) and only sliced after.
        ``allocated_pages`` as a device tensor (published with the step's
        inputs) skips the host-to-device copy; ``stage`` writes the anchor ids
        and positions in the same launch."""
        running_bs, T = block_positions.shape
        W = block_tables.shape[1]
        pw = triton.cdiv(self.sliding_window + T + 16, 16) + 1
        pw = triton.cdiv(pw, 8) * 8
        if getattr(self, "_meta_key", None) != (T, pw):
            max_bs, device = self.max_num_seqs, block_tables.device
            self._meta_key = (T, pw)
            self._meta_cap = None
            self._meta_rows_bt = torch.zeros(
                max_bs * T, pw, dtype=torch.int32, device=device
            )
            self._meta_rows_len = torch.zeros(
                max_bs * T, dtype=torch.int32, device=device
            )
        if isinstance(allocated_pages, torch.Tensor):
            alloc = allocated_pages
        else:
            if self._meta_cap is None:
                self._meta_cap = AllocationCap(self.max_num_seqs, block_tables.device)
            alloc = self._meta_cap.upload(allocated_pages, scheduled_bs, running_bs)
        rows = running_bs * T
        assert rows <= self._meta_rows_bt.shape[0], (running_bs, T)
        rows_bt = self._meta_rows_bt[:rows]
        rows_len = self._meta_rows_len[:rows]
        if stage is None:
            next_ids = positions = anchor_rows = ids_out = anchor_positions
        else:
            next_ids, positions, anchor_rows, ids_out = stage
        _block_meta_kernel[(running_bs,)](
            anchor_positions, block_tables, block_tables.stride(0), alloc,
            next_ids, positions, anchor_rows, ids_out, int(self.vocab_size) - 1,
            block_positions, slots, ctx_lens, rows_bt, rows_len,
            scheduled_bs, W,
            T=T, TP=triton.next_power_of_2(T), ENTRY=entry_tokens,
            WINDOW=self.sliding_window, PW=pw, PWP=triton.next_power_of_2(pw),
            PAD_CTX=pad_ctx_len, STAGE=stage is not None,
            num_warps=4,  # [T, PW] tiles of one request row; more warps only idle
        )  # fmt: skip
        attn_metadata.dspark_bt16_rows = rows_bt
        attn_metadata.dspark_ctx_rows = rows_len

    def _sharded_markov_sample(self) -> bool:
        """Whether the block is sampled off this rank's vocab shard: the ranks
        exchange each position's ``(max, id)`` instead of all-gathering the
        ``[B * T, V]`` logits, and each reads a quarter of ``W2`` per position
        (see ``DeepseekV41DSpark._sharded_markov_sample``). Not with the
        confidence head on, which reads each position's ``W1[prev]`` row: that
        block keeps the all-gathered path."""
        head, markov = self.lm_head, self.markov_head
        return (
            head is not None
            and not self.emit_confidence
            and head.tp_size > 1
            and head.bias is None
            # `_head_and_sample_sharded` multiplies the raw shard, so the head
            # must be the plain TP-sharded one: an even, unpadded split of the
            # vocab and no sequence-parallel logits path.
            and not sp_is_enabled()
            and head.num_embeddings % head.tp_size == 0
            and head.weight.shape[0] == head.num_embeddings // head.tp_size
            and markov.fused_sample
            and markov_weights_unsupported(
                markov.markov_w1.weight, markov.markov_w2.weight
            )
            is None
        )

    def _head_and_sample_sharded(self, hidden, anchor_ids, num_draft):
        """``head_and_sample`` with the LM head left sharded; the same ids as
        the whole-vocab op."""
        bs = anchor_ids.shape[0]
        head, markov = self.lm_head, self.markov_head
        base_logits = tgemm.mm(hidden, head.weight).view(bs, num_draft, -1)
        tp_group = get_tp_group()
        w1, w2 = markov.markov_w1.weight, markov.markov_w2.weight
        out_ids = anchor_ids.new_empty(bs, num_draft)
        self._markov_atomic(base_logits, anchor_ids, w1, w2, out_ids, head, tp_group)
        return out_ids, None

    def _markov_atomic(self, base_logits, anchor_ids, w1, w2, out_ids, head, tp_group):
        """The sharded greedy Markov block, one launch and one all-gather per
        position (`_markov_step_atomic`) and one closing launch. Each position's
        packed best lives in a persistent ``[T, max_num_seqs]`` int64 buffer,
        at its reset value between blocks (each is reset by the launch after the
        all-gather that read it), so no launch clears it."""
        bs, num_draft, v_local = base_logits.shape
        world = tp_group.world_size
        rank = w2.shape[1]
        geometry = markov_launch_geometry(bs, v_local)
        # Sized with the block buffers ahead of graph capture; never regrown
        # here, a captured graph holds its address. A forward that runs before
        # the block buffers exist (the profile pass) sizes it on first use.
        if self._mk_packed is None:
            self._alloc_markov_packed(num_draft, base_logits.device)
        assert num_draft <= self._mk_packed.shape[0]
        assert bs <= self._mk_packed.shape[1]
        gathered = None
        for k in range(num_draft):
            base_k = base_logits[:, k]
            packed = self._mk_packed[k, :bs]
            prev_packed = self._mk_packed[max(k - 1, 0), :bs]
            _markov_step_atomic[(geometry.v_tiles, geometry.row_tiles)](
                base_k, anchor_ids, packed if gathered is None else gathered,
                prev_packed, out_ids[:, max(k - 1, 0)], w1, w2, packed,
                bs, v_local, rank, base_k.stride(0), anchor_ids.stride(0),
                out_ids.stride(0), w1.stride(0), w2.stride(0), head.vocab_start_idx,
                FIRST=k == 0, W=world, BLOCK_ROW=geometry.block_row,
                BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K, num_warps=geometry.num_warps,
                num_stages=2,
            )  # fmt: skip
            gathered = (
                tp_group.all_gather(
                    packed.view(torch.float32).view(bs, 2),
                    dim=0,
                    use_custom=envs.ATOM_USE_CUSTOM_ALL_GATHER,
                )
                .view(torch.int64)
                .view(-1)
            )
        block_row = min(1024, triton.next_power_of_2(bs))
        _markov_last_atomic[(triton.cdiv(bs, block_row),)](
            gathered, self._mk_packed[num_draft - 1, :bs], out_ids[:, num_draft - 1],
            bs, out_ids.stride(0), W=world, BLOCK_ROW=block_row, num_warps=1,
        )  # fmt: skip

    def head_and_sample(self, hidden, anchor_ids, num_draft):
        if self._sharded_markov_sample():
            return self._head_and_sample_sharded(hidden, anchor_ids, num_draft)
        bs = anchor_ids.shape[0]
        base_logits = self.lm_head(hidden).view(bs, num_draft, -1)
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        markov_embeds = []
        for k in range(num_draft):
            # Greedy: sampling temperature is the target verify's business.
            markov_embeds.append(
                self.markov_head.sample_next(
                    out_ids[:, k], base_logits[:, k], out_ids[:, k + 1]
                )
            )
        confidence = None
        if self.emit_confidence:
            confidence = self.confidence_head(
                hidden.view(bs, num_draft, -1), torch.stack(markov_embeds, dim=1)
            )
        return out_ids[:, 1:], confidence
