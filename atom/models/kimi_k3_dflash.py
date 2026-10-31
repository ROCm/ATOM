# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 DFlash draft: the SpecForge/SGLang DSpark speculator.

`RadixArk/Kimi-K3-DSpark`, trained by SpecForge on hidden states from a live
SGLang Kimi-K3 target. Same algorithm as the TorchSpec draft in
`kimi_k3_dspark.py` -- block-parallel draft, low-rank Markov head, confidence
head -- on a different backbone: 5 Qwen3-style GQA layers instead of MLA.

Checkpoint layout (62 tensors, BF16, no biases)::

    fc.weight                                   [7168, 35840]  (= 5 x 7168)
    hidden_norm.weight                          [7168]
    norm.weight                                 [7168]
    layers.{0..4}.self_attn.q_proj.weight       [4096, 7168]
    layers.{0..4}.self_attn.k_proj.weight       [1024, 7168]
    layers.{0..4}.self_attn.v_proj.weight       [1024, 7168]
    layers.{0..4}.self_attn.{q,k}_norm.weight   [64]           per-head QK-norm
    layers.{0..4}.self_attn.o_proj.weight       [7168, 4096]
    layers.{0..4}.mlp.{gate,up}_proj.weight     [14336, 7168]
    layers.{0..4}.mlp.down_proj.weight          [7168, 14336]
    layers.{0..4}.{input,post_attention}_layernorm.weight  [7168]
    markov_head.markov_w{1,2}.weight            [163840, 256]
    confidence_head.proj.{weight,bias}          [1, 7424] / [1]  TRAINING ONLY

No `embed_tokens`, no `lm_head` -- both shared from the target, as the MLA draft
does. The fusion width is ``len(target_layer_ids)`` (5), NOT the config's
``num_target_layers`` (93, which is the TARGET's depth); `fc.weight`'s 35840
confirms it.

DUAL-SOURCE ATTENTION
---------------------
Per layer, the reference (`dflash.py` in the checkpoint) computes::

    q = q_norm(q_proj(noise))                   # q-norm BEFORE rope
    k = k_norm(cat([k_proj(ctx), k_proj(noise)]))   # SAME weights both sources
    v =        cat([v_proj(ctx), v_proj(noise)])    # v is NOT normed
    q, k = rope(q, k)                           # rope AFTER norm
    cache.update(k, v)                          # cache holds post-rope K
    attention is NON-CAUSAL over the whole concatenation

`k_norm` is an RMSNorm over `head_dim`, i.e. row-wise, so
``norm(cat(a, b)) == cat(norm(a), norm(b))`` exactly. That identity is what lets
the context half be normed and cached at `write_context_kv` time and the noise
half be normed in the block pass, and still equal the reference's single concat.

The norms run HERE, in the model, not inside `Attention`: the op's non-fused
path applies rope first and the norms second (`attention_mha.py`), which is the
wrong order for this draft.

NON-CAUSALITY, AND WHY THIS BYPASSES `PagedAttentionImpl`
--------------------------------------------------------
`PagedAttentionImpl.dispatch_backend` is causal at EVERY arm: `causal=True` is a
literal in all four runners and `attn_metadata.causal` is read only by the MLA
impl, so the proposer setting it False is inert here. Routing this block through
the op would silently apply a causal mask to an attention the reference defines
as bidirectional -- no error, no fault, just lost acceptance length.

So the block pass does its own write and its own read:

  write  `aiter.reshape_and_cache_flash` -- the block WRITES the KV it attends,
         before the read: the next step's prefix is this step's block, and the
         read below takes the block's own rows out of the pool with the prefix.
  read   aiter's `unified_attention` on the FOLDED block -- see
         `_attend_non_causal` and `model_ops/attentions/dspark_draft_attn.py`.

THE READ
--------
`unified_attention` is the kernel `PagedAttentionImpl.paged_attention_unified`
already decodes through, with the same per-tensor fp8 descale. It asserts
`causal`, and `pa_decode_gluon` derives `is_causal = query_length > 1`, so no
aiter paged-GQA entry point is non-causal at `qlen > 1`. A `qlen == 1` query is,
by construction: the causal mask admits every key below `seqused_k`. So the
block is FOLDED -- each request becomes ONE query row whose heads are its T
block rows times its G query heads per KV head, `[bs, h_kv * T * G, d]` --
and read as an all-decode batch over `[0, anchor + 1 + T)`. All `T * G` rows of
a KV head share one KV tile, and the all-decode launch is the split-KV grid, so
a long context is spread over segments rather than walked by one program.

The kernel reads the 4-D FLASH layout `[entries, entry_tokens, h_kv, head_dim]`
(`MhaKvPool.fa_views`) through `block_table` at any page size; SHUFFLE's x-pack
would need `shuffled_kv_cache` and a power-of-two page. It widens block ids to
int64 before scaling by the page stride, which this pool needs: an entry is
`block_size * dcp` tokens, so `entry * stride` passes 2**31 well inside the
pool sizes ATOM builds.

NO DCP, AND WHY THE ENTRY IS WIDER THAN A BLOCK
----------------------------------------------
SGLang's draft never joins the dcp group: `models/qwen3_5.py:1112` reads
`dcp_size = 1 if is_nextn else attn_dcp_size` under the comment "Drafts are
TP-sharded and do not replicate KV under DCP", and `models/dflash.py:146-171`
shards q, k and v by `tp_size` flat without ever consulting `attn_dcp_size`.
This module mirrors that exactly: the TARGET stays at DCP8, this draft is
plain TP, `num_heads == 64 // tp` and `num_kv_heads == 16 // tp`, and there is
no LSE merge anywhere because the local read already IS the global softmax.

What that costs is a pool geometry, not an attention: the draft rides the
TARGET's block ids, and under DCP8 one of those ids addresses `block_size`
LOCAL tokens, i.e. `block_size * dcp_size` GLOBAL positions. So the draft's
pool is built with `blocks_per_entry = dcp_size` and everything addressing it
-- the block pass's `slot_mapping`, the context write's, and the read's
`block_table` -- is in that virtual block. Per-rank bytes are unchanged: 8x the
tokens against 8x fewer KV heads.

fp8, AND THE ONE SCALE
----------------------
The pool is fp8_e4m3fn when the target's is fp8 (SGLang serves this checkpoint
with `--speculative-draft-kv-cache-dtype fp8_e4m3`; ATOM's draft pool follows
the target's `--kv_cache_dtype`). Both ends take ONE PER-TENSOR scale and
neither can take anything else: `aiter.reshape_and_cache_flash` dereferences
`*k_scale` once per launch, and `unified_attention` loads its `k_descale` /
`v_descale` once per program. SGLang stores nothing else either -- it refuses,
"Only support per-tensor scaling factor for fp8 KV cache"
(`srt/layers/quantization/kv_cache.py:76`) -- and defaults the value to 1.0
when the checkpoint carries no scale, which this one does not (all 62 tensors
BF16, no `quantization_config`). So the pool declares NO per-token scale planes
(`mha_kv_scale_fields(per_tensor_scales=True)`), and the binder hands every
layer one shared one-element 1.0 tensor (`DraftKvBuilder`). A draft that did
ship a calibrated scalar scale would need a loader.

The pool's fp8 is the platform's, `aiter.utility.dtypes.d_dtypes["fp8"]`, as
the target's is: `float8_e4m3fn` on gfx950, `float8_e4m3fnuz` on gfx942.
"""

import logging
from typing import ClassVar

import torch
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.models.dspark_draft import (
    DSparkDraftModel,
    DSparkMarkovHead,
    KVCacheBinding,
    linear_out,
)
from atom.models.qwen3 import Qwen3MLP

logger = logging.getLogger("atom")


class KimiK3DFlashAttention(nn.Module):
    """One DFlash GQA layer's attention, with a second entry for context rows.

    `forward` is the block pass over the noise stream. `write_context_kv` is the
    target-derived half: it runs the same k/v projections over the projected aux
    context and scatters the result into this layer's rows of the draft's own KV
    pool, so the block pass then attends context and noise in one call.
    """

    def __init__(
        self,
        config,
        layer_num: int,
        block_attn,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # PLAIN TP, no DCP anywhere in this module. The draft does not join the
        # dcp group -- SGLang's does not either (`models/qwen3_5.py:1112`,
        # "Drafts are TP-sharded and do not replicate KV under DCP"; DFlash's
        # own attention shards by `tp_size` flat, `models/dflash.py:146-171`) --
        # so there is ONE head count, `o_proj` consumes the same slice the read
        # produces, and no LSE merge exists to get wrong.
        #
        # What used to be here was the opposite trade: KV heads REPLICATED
        # across the DCP group at `tp_eff = tp // dcp` so the token round-robin
        # could be the only shard. It is numerically fine and it is not what
        # SGLang runs; it also forces every read through a kernel that returns
        # an LSE for a cross-rank merge. The token axis is
        # unsharded instead, and the pool pays for it in `blocks_per_entry`
        # rather than in bytes (see the module docstring).
        # NOTHING in this module reads `get_dcp_world_size()` any more, which
        # is the point: `models/dflash.py` does not read `attn_dcp_size` either.
        # The target's DCP survives only as the draft POOL's `blocks_per_entry`
        # (`draft_kv_builder`) and as the virtual block the PROPOSER addresses
        # it in (`DSparkProposer._draft_entry_tokens`) -- one number, decided
        # where the pool is built, never re-derived here.
        tp_full = get_tensor_model_parallel_world_size()
        if self.total_num_heads % tp_full != 0:
            raise ValueError(
                f"DFlash draft has {self.total_num_heads} query heads, which is "
                f"not divisible by TP size {tp_full}; `o_proj` cannot be split."
            )
        self.num_heads = self.total_num_heads // tp_full
        # The GQA-tolerant form, not a bare divisibility assert: below TP the kv
        # heads replicate rather than shard, and QKVParallelLinear already does
        # that. With 16 kv heads this only bites past TP16. Mirrors the same
        # pair of checks in `models/dflash.py:161-171`, raised rather than
        # asserted so `python -O` keeps them.
        if self.total_num_kv_heads >= tp_full:
            sharded = self.total_num_kv_heads % tp_full == 0
        else:
            sharded = tp_full % self.total_num_kv_heads == 0
        if not sharded:
            raise ValueError(
                f"DFlash draft has {self.total_num_kv_heads} KV heads, which "
                f"neither divide nor are divided by TP size {tp_full}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_full)

        # The block's read, one instance shared by every layer
        # (`KimiK3DFlash.__init__`).
        self.block_attn = block_attn
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # No `override_tp_size` on EITHER projection. Both are sharded by the
        # one TP group, and `o_proj`'s RowParallel input column shard is
        # exactly the head slice the read above produces -- which is the whole
        # point of taking the draft out of DCP.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK-norm, Qwen3 style: RMSNorm over head_dim is row-wise, so a
        # flat [N, q_size] tensor is normalized in head_dim-wide groups without
        # an explicit reshape.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # The DRAFT'S OWN rope, never the target's. This checkpoint is YaRN x16
        # from 65536 at theta 10000; Kimi-K3 is x32 from 32768 at theta 50000.
        # Inheriting the target's loads clean and only costs acceptance length.
        rope_params = dict(getattr(config, "rope_parameters", None) or {})
        rope_theta = rope_params.get("rope_theta") or getattr(
            config, "rope_theta", 10000.0
        )
        rope_scaling = None
        if rope_params.get("rope_type") == "yarn" or rope_params.get(
            "factor", 1.0
        ) not in (
            1.0,
            None,
        ):
            rope_scaling = dict(rope_params)
            rope_scaling.setdefault(
                "original_max_position_embeddings", config.max_position_embeddings
            )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        # What the draft pool binds this layer's cache and scales onto
        # (`DraftKvBuilder.build_kv_cache_tensor`).
        self.kv = KVCacheBinding(layer_num)

    def _qkv(self, hidden_states: torch.Tensor):
        qkv = linear_out(self.qkv_proj(hidden_states))
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """One block pass: write this block's own K/V, then attend non-causally.

        Deliberately NOT the paged `Attention` op. `PagedAttentionImpl.dispatch_backend`
        is causal at every arm -- `causal=True` is a literal in all four runners
        and `attn_metadata.causal` is read only by the MLA impl -- so routing
        the block through it would silently apply a causal mask to an attention
        the reference defines as bidirectional. That is a numerical bug with no
        error and no fault, visible only as lost acceptance length.
        """
        from atom.utils.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata

        q, k, v = self._qkv(hidden_states)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Norm, THEN rope -- the reference's order, and the reason the norms are
        # not handed to the op (its non-fused path ropes first).
        q, k = self.rotary_emb(positions, q, k)

        n = q.shape[0]
        k_cache, v_cache = self._caches()
        if k_cache is None:
            # warmup_model() runs BEFORE allocate_kv_cache(), so there is no
            # paged state to write or read. The block forward still runs -- that
            # pass doubles as memory profiling, and skipping it would leave the
            # draft's activations out of the KV budget -- but its attention has
            # nothing to attend, so the output is shaped and left at zero.
            # One head count, because the read produces exactly what `o_proj`
            # consumes -- there is no merge between them any more.
            o = q.new_zeros((n, self.num_heads * self.head_dim))
            return linear_out(self.o_proj(o))

        # The write goes FIRST: the read takes the block's own T rows out of
        # the pool with the prefix, and the next block's prefix is these rows.
        self._write_block_kv(k, v, attn_metadata.slot_mapping[:n], k_cache, v_cache)
        o = self._attend_non_causal(q, k, v, attn_metadata, k_cache, v_cache)
        return linear_out(self.o_proj(o))

    def _write_block_kv(self, k, v, slot_mapping, k_cache, v_cache) -> None:
        """Scatter the block's own K/V into this layer's rows of the pool.

        Unlike V4's rolling window, this block WRITES the KV it attends: the T
        draft rows join the context rows already there, and the single read
        below then sees `[0, anchor] ++ T` -- the reference's one attention over
        `cat([k_ctx, k_noise])`.
        """
        import aiter

        aiter.reshape_and_cache_flash(
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            k_cache,
            v_cache,
            slot_mapping,
            # `reshape_and_cache_flash`'s vocabulary: "fp8" divides by the
            # scale and casts, "auto" stores the source dtype unchanged.
            "fp8" if k_cache.dtype.itemsize == 1 else "auto",
            self.kv.k_scale,
            self.kv.v_scale,
        )

    def _attend_non_causal(self, q, k, v, attn_metadata, k_cache, v_cache):
        """The block's one non-causal read: aiter `unified_attention`, folded.

        Keys are `[0, context_lens[b])` of the pool, the T block rows included
        since `_write_block_kv` already landed them; `block_tables` is
        entry-granular, which is exactly the page `k_cache` (`fa_views`) is.
        See `model_ops/attentions/dspark_draft_attn.py` for the fold.

        NO LSE AND NO MERGE, by construction -- this draft is not in the dcp
        group, so there is one head count and the local read already is the
        global softmax over this rank's TP head slice.
        """
        bs = attn_metadata.block_tables.shape[0]
        o = self.block_attn(
            q.view(-1, self.num_heads, self.head_dim),
            k_cache,
            v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens[:bs],
            self.scaling,
            self.kv.k_scale,
            self.kv.v_scale,
        )
        return o.reshape(-1, self.num_heads * self.head_dim)

    def _caches(self):
        """This layer's bound pool rows in FLASH layout, or `(None, None)`.

        Unbound is a real state, not an error: `warmup_model()` runs before
        `allocate_kv_cache()`, so the binding still holds its empty init tensor.
        """
        if self.kv.k_cache.numel() == 0:
            return None, None
        return self.kv.k_cache, self.kv.v_cache

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected aux context
        positions: torch.Tensor,  # [N] absolute positions
        slot_mapping: torch.Tensor,  # [N] destination slots in the draft pool
    ) -> None:
        """Project the target-derived context and cache it as this layer's K/V.

        The whole fused QKV GEMM runs and the q columns are dropped. Not
        bookkeeping that could be avoided by slicing the weight: with
        per-output-channel quant and preshuffled rows, a row slice of the merged
        weight is not that shard's weight. The MLA draft does the same.
        """
        _, k, v = self._qkv(ctx_hidden)
        # Only k is normed -- v is not, matching the reference. Row-wise RMSNorm
        # makes norming this half separately identical to norming the concat,
        # which is what lets the context live in the pool at all.
        k = self.k_norm(k)
        # Rope at the token's ABSOLUTE position, before the write: that is what
        # makes a cached row valid on every later step without being rewritten.
        k = self.rotary_emb(positions, k)
        k_cache, v_cache = self._caches()
        if k_cache is None:
            return
        self._write_block_kv(k, v, slot_mapping, k_cache, v_cache)


class KimiK3DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_num: int,
        block_attn,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = KimiK3DFlashAttention(
            config,
            layer_num,
            block_attn,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        # SpecForge builds this draft on a plain Qwen3 config; its MLP is Qwen3's.
        self.mlp = Qwen3MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Residual add fused into the norm, as in Qwen3DecoderLayer.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    def write_context_kv(self, ctx_hidden: torch.Tensor, positions: torch.Tensor):
        """The context rows bypass `input_layernorm`.

        The reference feeds `hidden_norm(fc(aux))` straight into k_proj/v_proj;
        only the noise stream goes through the layer's input norm.
        """
        from atom.utils.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        # The DRAFT's OWN slot mapping, published by
        # `DSparkProposer._publish_draft_ctx_slots` for exactly this call. NOT
        # `attn_metadata.slot_mapping`, which is the TARGET's: under DCP that
        # one is `-1` for the 7 of every 8 tokens this rank does not own, and
        # `block_table[p // vblock] * block_size + local_offset` for the one it
        # does (`dcp_ops.py:1059-1072`). This draft is not in the dcp group, so
        # it stores EVERY token, at `block_table[p // vblock] * vblock + p %
        # vblock`. The two coincide at DCP1 and only there.
        #
        # Falling back to the target's is deliberate rather than an assert: the
        # proposer publishes this only for the paged-GQA flavor, and at DCP1
        # the two mappings are the same tensor's worth of numbers anyway.
        slot_mapping = getattr(attn_metadata, "draft_ctx_slot_mapping", None)
        if slot_mapping is None:
            slot_mapping = attn_metadata.slot_mapping
        rows = ctx_hidden.shape[0]
        if slot_mapping.shape[0] < rows:
            raise RuntimeError(
                f"DFlash context write has {rows} rows but only "
                f"{slot_mapping.shape[0]} draft slots"
            )
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping[:rows])


class KimiK3DFlash(DSparkDraftModel):
    """The SpecForge DFlash draft for Kimi-K3.

    Registered as ``K3DFlashModel``: the checkpoint ships
    ``architectures: ["DSparkDraftModel"]``, which SpecForge stamps on every
    DFlash draft and which collides with this module's own base class, so
    `SpeculativeConfig.hf_config_override` renames it.
    """

    # GQA, read through `unified_attention` straight off `block_tables` /
    # `context_lens`: no MLA decode plan and no CSR to build, and nothing for
    # `prepare_paged_block` to add.
    block_attention_is_mla = False

    packed_modules_mapping: ClassVar[dict[str, tuple]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # confidence_head predicts a per-position accept probability and is used at
    # training time (and by SGLang for adaptive block length); ATOM runs a fixed
    # block, so it is never read. embed_tokens / lm_head are the target's.
    skip_weight_prefixes: ClassVar[list[str]] = [
        "confidence_head.",
        "embed_tokens.",
        "lm_head.",
    ]

    def __init__(self, atom_config, layer_offset: int = 0):
        super().__init__()
        config = atom_config.hf_config
        quant_config = getattr(atom_config, "quant_config", None)

        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = config.hidden_size
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = list(config.dspark_target_layer_ids)

        # Aux fusion: [N, hidden * len(target_layer_ids)] -> [N, hidden].
        # Replicated on purpose: the aux hidden states arrive replicated (they
        # are the target's residual stream), so a row-parallel split would buy
        # memory at the cost of an all-reduce every drafting step.
        self.fc = ReplicatedLinear(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            prefix="fc",
        )
        self.hidden_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # One block read for every layer, built here -- on the engine's
        # device, ahead of any capture -- and sized for the engine's widest
        # batch and longest context.
        from atom.model_ops.attentions.dspark_draft_attn import (
            DSparkDraftUnifiedAttention,
        )

        block_attn = DSparkDraftUnifiedAttention(
            torch.get_default_device(),
            max_bs=int(atom_config.max_num_seqs),
            max_model_len=int(atom_config.max_model_len),
        )
        self.layers = nn.ModuleList(
            [
                KimiK3DFlashDecoderLayer(
                    config,
                    layer_offset + i,
                    block_attn,
                    quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.vocab_size, int(config.dspark_markov_rank)
        )

        # Bound by `share_with_target`.
        self.embed_tokens = None
        self.lm_head = None

    # -- the target feature this draft was trained on ----------------------

    def target_aux_capture_spec(self, layer_ids, hidden_size, target_model):
        """Tap the target's AGGREGATE stream, not its running prefix.

        Kimi-K3 does not carry a plain pre-norm residual stream. It carries a
        BANK of per-``attn_res_block_size`` snapshots plus a RUNNING PREFIX,
        recombined at each attn-residual site by a softmax mixture over the
        bank plus the prefix (`AttnRes`). ATOM's generic aux tap reconstructs
        the prefix alone (`KimiDecoderLayer.aux_hidden_state` sums the layer's
        deferred addends into it) and discards the bank; SGLang's dspark aux tap
        returns the MIXTURE (`aggregate_stream`).

        This checkpoint (`RadixArk/Kimi-K3-DSpark`) was trained by SpecForge on
        hidden states captured from a live SGLang Kimi-K3 target, so the
        mixture is the feature it saw. Measured on GSM8K n=1319 at k=5: feeding
        the prefix scores 55.61% acceptance, feeding the aggregate 62.04%
        (+6.43 pp), accuracy unchanged within noise.

        This is a PER-DRAFT contract, not a target-side or global one. The same
        aggregate makes the vLLM/HF-trained MLA draft (`KimiK3DSpark`,
        `Inferact/Kimi-K3-DSpark`) about 2.6 pp WORSE -- that one wants the raw
        prefix, and gets it by not implementing this method at all and falling
        through to `DSparkProposer._extract_layer_hidden`. Which feature a draft
        is fed therefore follows from which draft class is loaded, with nothing
        to configure and no way for the two to disagree.

        The mixture is ATOM's own (`apply_attn_res`), not a transcription of
        SGLang's: the scoring weight is the NEXT target layer's
        `self_attention_attn_res.score_weight` (= `self_attention_res_norm.weight *
        self_attention_res_proj.weight`, folded at load) and the kernel is the
        one the target itself runs, called with ``out_norm_weight=None`` so the
        result comes back un-normed -- exactly SGLang's "softmax mixture, no
        output norm". `kimi_k3.py` is `@support_torch_compile` and is not
        touched; the whole reconstruction lives here, in the draft that wants it.
        """
        # Local: `drafter` imports the model loader, so a module-level import
        # here would close a cycle. Both are resolved once, at arm time.
        from atom.spec_decode.drafter import AuxCaptureSpec, _resolve_decoder_layers
        from atom.spec_decode.dspark_proposer import DSparkProposer

        # The prefix half is still the generic reconstruction -- it is a
        # property of the TARGET's layer protocol, and the aggregate is a
        # mixture OF it, not a replacement for it.
        base_extract = DSparkProposer._extract_layer_hidden
        layers = _resolve_decoder_layers(target_model)

        # lid -> (num_valid_blocks, the AttnRes that scores the mixture).
        # Keyed by the layer's OWN `layer_idx`, which is what the extract can
        # read back off the module it is handed: a module int attribute, which
        # Dynamo constant-folds inside the target's traced region, where an
        # `id()`-keyed lookup would graph-break.
        taps: dict[int, tuple[int, nn.Module]] = {}
        for lid in layer_ids:
            if lid < 0 or lid + 1 >= len(layers):
                continue  # the embedding tap, and the last layer (no successor)
            if getattr(layers[lid], "layer_idx", lid) != lid:
                continue  # not a stack we can index; take the prefix
            res = getattr(layers[lid + 1], "self_attention_attn_res", None)
            if res is None or not getattr(res, "enabled", False):
                continue
            block_size = getattr(res, "block_size", None)
            if not block_size:
                continue
            # CEILING division, matching SGLang's `prev_valid_blocks`: the bank
            # holds one snapshot per closed block below this layer.
            nvb = -(-(lid + 1) // block_size)
            if nvb > 0:
                taps[lid] = (nvb, res)

        if not taps:
            # A target without attention residuals (or a stack we could not
            # index) has no aggregate to form; the prefix IS the whole stream.
            logger.warning(
                "DFlash draft found no Kimi-K3 attn-residual sites above its "
                "taps %s; feeding the plain residual stream, which is NOT the "
                "feature this checkpoint was trained on.",
                layer_ids,
            )
            return AuxCaptureSpec(layer_ids, hidden_size, base_extract)

        logger.info("DFlash aggregate tap armed on target layers %s", sorted(taps))

        def extract(output, block):
            prefix = base_extract(output, block)
            tap = taps.get(getattr(block, "layer_idx", -1))
            if prefix is None or tap is None:
                return prefix
            nvb, res = tap
            # `KimiDecoderLayer.forward` returns (prefix_sum, *pendings, bank).
            bank = output[-1] if isinstance(output, tuple) else None
            if bank is None or bank.shape[1] < nvb:
                # Fewer closed blocks than this layer implies: nothing to mix
                # against, so the prefix already is the whole stream.
                return prefix
            score_weight = res.score_weight
            if score_weight is None:  # loader hook did not run (plugin hosts)
                res.process_weights_after_loading()
                score_weight = res.score_weight
            from atom.model_ops.kimi_k3 import apply_attn_res

            # `out_norm_weight=None` -> the mix comes back un-normed. The base
            # hook then copies it in place into the same fixed capture buffer it
            # would have filled with `prefix` -- same buffer, same shape, so
            # cudagraph capture is unaffected.
            mixed, _ = apply_attn_res(
                prefix,
                bank[:, :nvb, :],
                score_weight,
                res.eps,
                None,
                None,
                res.out_eps,
            )
            return mixed

        return AuxCaptureSpec(layer_ids, hidden_size, extract)

    # -- DSparkDraftModel contract ----------------------------------------

    def write_context_kv(self, aux_concat: torch.Tensor, positions: torch.Tensor):
        # Under TR_HYBRID the runner hands over `running_tokens` rows of target
        # hidden state but only the scheduled rows' `positions`; the padded tail
        # has no slot to land in, so it is dropped before the projection.
        super().write_context_kv(aux_concat[: positions.shape[0]], positions)

    @property
    def first_attention_layer_num(self) -> int:
        return self.layers[0].self_attn.kv.layer_num

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(linear_out(self.fc(aux_concat)))

    @property
    def context_layers(self):
        """Every draft layer holds context rows; `layers` is already in order."""
        return self.layers

    def block_backbone(self, input_ids, positions, num_draft: int):
        """One block: [anchor, MASK, ...] of width T in, T hidden rows out.

        `positions` is [B*T] absolute block positions -- the standalone-draft
        shape `DSparkProposer._block_positions` produces.
        """
        bs = input_ids.shape[0]
        draft_ids = input_ids.new_full((bs, num_draft), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = self.embed_tokens(draft_ids.view(-1))
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden

    def head_and_sample(self, out, anchor_ids: torch.Tensor, num_draft: int):
        base_logits = self.lm_head(out).view(anchor_ids.shape[0], num_draft, -1)
        return self._sample_block(base_logits, anchor_ids), None

    def forward_spec(self, input_ids, positions, num_draft=None):
        out = self.block_backbone(input_ids, positions, num_draft)
        return self.head_and_sample(out, input_ids, num_draft)

    def _sample_block(self, base_logits: torch.Tensor, anchor_ids: torch.Tensor):
        """Markov-biased left-to-right sampling over the computed block.

            logits_k <- U_k + B(x_{k-1}, .);  x_k <- argmax(logits_k)

        Only this loop is sequential, and it is T rank-256 matmuls rather than T
        backbone passes -- which is the point of DSpark.
        """
        bs, num_draft, _ = base_logits.shape
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        for k in range(num_draft):
            # Greedy: temperature/sampling is applied by the target's verify.
            # The column IS the op's destination -- `sample_next` writes ids
            # into `out`, and returns the markov embedding, not the ids -- so
            # no id is written twice and a strided view costs no copy.
            self.markov_head.sample_next(
                out_ids[:, k], base_logits[:, k], out_ids[:, k + 1]
            )
        return out_ids[:, 1:]

    def share_with_target(self, target_base, loaded) -> None:
        """Bind the target's embedding and unembedding.

        The checkpoint ships neither; the draft drafts in the target's token
        space, so a vocab disagreement is a load-time error, not a runtime one.
        """
        # `target_base.model.embed_tokens`, not `target_base.embed_tokens`:
        # ModelRunner unwraps the multimodal `language_model` wrapper but
        # hands over the KimiLinearForCausalLM, whose embedding lives one
        # level down on `.model`.
        target_vocab = int(target_base.model.embed_tokens.num_embeddings)
        if target_vocab != self.vocab_size:
            raise ValueError(
                f"DFlash draft vocab {self.vocab_size} != target vocab "
                f"{target_vocab}; the draft samples in the target's token space."
            )
        self.embed_tokens = target_base.model.embed_tokens
        self.lm_head = target_base.lm_head
