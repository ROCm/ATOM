# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 DSpark GQA draft: the SpecForge/SGLang DSpark speculator.

`RadixArk/Kimi-K3-DSpark` (a SpecForge DFlash checkpoint): the same
block-parallel draft and Markov head as `kimi_k3_dspark.py`, on 5 Qwen3-style
GQA layers instead of MLA. As there, the checkpoint's confidence head is not
loaded, so the draft returns no confidence. `embed_tokens` and `lm_head` are
shared from the target. The fusion width is ``len(target_layer_ids)`` (5), not
the config's ``num_target_layers`` (the target's depth).

Each layer attends non-causally over the cached context K/V followed by the
block's own K/V, with q/k RMS-normed per head BEFORE rope. The norms run here,
not in `Attention`, whose non-fused path ropes first. `k_norm` is row-wise, so
the context half can be normed when it is cached and the block half in the
block pass.

`PagedAttentionImpl` is causal on every arm, so the block pass writes with
`aiter.reshape_and_cache_flash` and reads with `unified_attention` on the
folded block: one `qlen == 1` row per request carrying its `T * G` query rows,
which the causal mask admits over every key (see `_attend_non_causal`). The
read widens block ids to int64 before the page stride, which an entry this
wide needs.

The draft stays out of the DCP group: it is plain TP and needs no LSE merge.
It still rides the target's block ids, and under DCP one id addresses
`block_size * dcp` global positions, so its pool is entry-granular
(`blocks_per_entry = dcp`, read through `MhaKvPool.fa_views`) and every slot
and block table addressing it is in that unit.

An fp8 pool takes one per-tensor scale: `reshape_and_cache_flash` and
`unified_attention` both load a single scalar per launch. The checkpoint ships
no KV scales, so `kv_pool_spec` declares the unit scale, every layer binds one
shared 1.0, and the pool declares no per-token scale planes.
"""

import logging
from typing import ClassVar

import aiter
import torch
from aiter.rotary_embedding import get_rope
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.attentions.dspark_block_attn import DSparkGQABlockAttention
from atom.model_ops.layernorm import DualRMSNorm, RMSNorm
from atom.model_ops.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.models.dspark_draft import (
    AllocationCap,
    DSparkDraftModel,
    DSparkGQADecoderLayer,
    KVCacheBinding,
    draft_quant_config,
    draft_rope_kwargs,
    linear_out,
)
from atom.models.qwen3 import Qwen3MLP
from atom.spec_decode.draft_kv import DraftKvPoolSpec, draft_kv_dtype
from atom.spec_decode.drafter import AuxCaptureSpec
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _head_dim(config) -> int:
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


class KimiK3DSparkGQAAttention(nn.Module):
    """One DSpark GQA draft layer's attention, with a second entry for context rows.

    `forward` is the block pass over the noise stream. `write_context_kv` is the
    target-derived half: it runs the same k/v projections over the projected aux
    context and scatters the result into this layer's rows of the draft's own KV
    pool, so the block pass then attends context and noise in one call.
    """

    def __init__(
        self,
        config,
        layer_num: int,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = _head_dim(config)

        # Plain TP: the draft is outside the target's DCP group, so `o_proj`'s
        # input shard is exactly the head slice the read produces.
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK-norm; the context write norms k alone.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.qk_norm = DualRMSNorm(
            self.q_norm,
            self.k_norm,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            prefix=f"{prefix}.qk_norm",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            **draft_rope_kwargs(config, is_neox_style=True),
        )

        self.kv = KVCacheBinding(layer_num)
        self.block_attn = None

    def use_block_read(self, block_attn) -> None:
        """Read the block through `block_attn`, and bind only a pool it was
        built for."""
        self.block_attn = block_attn
        self.kv.check_cache = block_attn.check_cache

    def _qkv(self, hidden_states: torch.Tensor):
        qkv = linear_out(self.qkv_proj(hidden_states))
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """One block pass: write this block's own K/V, then attend non-causally.

        Not the paged `Attention` op, which is causal on every arm and would
        mask silently.
        """
        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata

        q, k, v = self._qkv(hidden_states)
        q, k = self.qk_norm(q, k)
        # Norm, then rope: the reference's order.
        q, k = self.rotary_emb(positions, q, k)

        n = q.shape[0]
        if not self.kv.bound:
            # Warmup runs before the pool is allocated, and still has to run
            # the rest of the block so profiling counts its activations.
            o = q.new_zeros((n, self.num_heads * self.head_dim))
            return linear_out(self.o_proj(o))
        k_cache, v_cache = self.kv.k_cache, self.kv.v_cache

        # The write goes FIRST: the read takes the block's own T rows out of
        # the pool with the prefix, and the next block's prefix is these rows.
        self._write_kv(k, v, attn_metadata.slot_mapping[:n], k_cache, v_cache)
        o = self._attend_non_causal(q, attn_metadata, k_cache, v_cache)
        return linear_out(self.o_proj(o))

    def _write_kv(self, k, v, slot_mapping, k_cache, v_cache) -> None:
        """Scatter K/V into this layer's rows of the pool: the block's own rows
        on the block pass, the context rows on `write_context_kv`.
        """
        if k_cache.dtype.itemsize == 1:
            # The pool's scale is 1.0 (`kv_pool_spec`), and a plain cast turns
            # an out-of-range value into NaN, which a context row -- written
            # once -- would carry for the request's life. Saturate, as the read
            # does for q.
            fmax = torch.finfo(k_cache.dtype).max
            k = k.clamp(-fmax, fmax)
            v = v.clamp(-fmax, fmax)
        aiter.reshape_and_cache_flash(
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            k_cache,
            v_cache,
            slot_mapping,
            "fp8" if k_cache.dtype.itemsize == 1 else "auto",
            self.kv.k_scale,
            self.kv.v_scale,
        )

    def _attend_non_causal(self, q, attn_metadata, k_cache, v_cache):
        """The block's one non-causal read: aiter `unified_attention`, folded.

        Keys are `[0, context_lens[b])` of the pool, the T block rows included;
        `block_tables` is entry-granular, as the `fa_views` page is. No LSE
        merge: outside the DCP group the local read is the whole softmax.
        """
        bs = attn_metadata.block_tables.shape[0]
        o = self.block_attn(
            q.view(-1, self.num_heads, self.head_dim),
            k_cache,
            v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens[:bs],
            self.kv.k_scale,
            self.kv.v_scale,
        )
        return o.reshape(-1, self.num_heads * self.head_dim)

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected aux context
        positions: torch.Tensor,  # [N] absolute positions
        slot_mapping: torch.Tensor,  # [N] destination slots in the draft pool
    ) -> None:
        """Project the target-derived context and cache it as this layer's K/V.

        Only the k and v rows of the fused QKV weight run: the context rows
        never need a query. The draft is BF16 (`draft_quant_config`), whose
        merged weight is the shards' rows laid end to end, so the row slice is a
        contiguous view of exactly the k and v shards.
        """
        kv = tgemm.mm(ctx_hidden, self.qkv_proj.weight[self.q_size :], None)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        # v is not normed, matching the reference.
        k = self.k_norm(k)
        # Absolute positions, so a cached row stays valid on every later step.
        k = self.rotary_emb(positions, k)
        self._write_kv(k, v, slot_mapping, self.kv.k_cache, self.kv.v_cache)


class KimiK3DSparkGQADecoderLayer(DSparkGQADecoderLayer):
    def __init__(
        self,
        config,
        layer_num: int,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = KimiK3DSparkGQAAttention(
            config,
            layer_num,
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


def _check_rope_covers(max_model_len: int, rope_rows: int) -> None:
    """Refuse a `max_model_len` past the draft's rope cache.

    Both passes clamp their positions to the cache's last row. Only rows past
    `max_model_len` reach the clamp: verify rows up to `mtp_k` past the
    request's last token, and block rows up to T past its anchor. The block
    rewrites those rows before it reads them, or the request has finished.
    """
    if max_model_len > rope_rows:
        raise ValueError(
            f"max_model_len {max_model_len} exceeds the DSpark GQA draft's rope "
            f"cache of {rope_rows} positions; lower --max-model-len to at "
            f"most {rope_rows}"
        )


class KimiK3DSparkGQA(DSparkDraftModel):
    """The Kimi-K3 DSpark GQA draft (the RadixArk SpecForge DFlash checkpoint).

    Registered as ``K3DSparkGQAModel``, which `SpeculativeConfig.hf_config_override`
    renames the checkpoint's generic ``DSparkDraftModel`` to.
    """

    block_attention_is_mla = False
    kv_pool_spec = DraftKvPoolSpec()
    # The base `_sample_block` hands the binding to the head's `sample_block`,
    # so `DSparkProposer` may arm lossless speculative sampling here.
    supports_lossless_draft = True

    packed_modules_mapping: ClassVar[dict[str, tuple]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # The checkpoint's final norm is `norm`; the base `block_backbone` reads
    # `final_norm`. Prefix-anchored: `hidden_norm` and the layers' norms keep
    # their names.
    weights_mapper = WeightsMapper(orig_to_new_prefix={"norm.": "final_norm."})

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
        # BF16 only: `--online_quant_config` would otherwise re-quantize every
        # draft linear its exclude list misses.
        quant_config = draft_quant_config(config, draft="Kimi-K3 DSpark GQA")

        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = config.hidden_size

        # Aux fusion: [N, hidden * len(target_layer_ids)] -> [N, hidden].
        # Replicated: the aux hidden states arrive replicated, and a
        # row-parallel split would cost an all-reduce every drafting step.
        self.fc = ReplicatedLinear(
            self.hidden_size * len(config.dspark_target_layer_ids),
            self.hidden_size,
            bias=False,
            prefix="fc",
        )
        self.hidden_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                KimiK3DSparkGQADecoderLayer(
                    config,
                    layer_offset + i,
                    quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        # The last row of the rope cache. YaRN sizes it from
        # `original_max_position_embeddings * factor`, which equals the target's
        # default `max_model_len`, while block positions reach T past it.
        self._rope_max_row = (
            int(self.layers[0].self_attn.rotary_emb.cos_cache.shape[0]) - 1
        )
        _check_rope_covers(int(atom_config.max_model_len), self._rope_max_row + 1)
        self.init_block_fields(config)

        # Bound by `share_with_target`.
        self.embed_tokens = None
        self.lm_head = None

        # `prepare_paged_block`'s per-request allocation.
        self._alloc_cap = AllocationCap(
            int(atom_config.max_num_seqs), torch.get_default_device()
        )

        # One block read for every layer, built before any capture, for the
        # pool `draft_kv_builder` builds from the same two calls, and the
        # per-rank heads the layers project. The block is `mtp_k` rows wide,
        # which is `num_speculative_tokens`.
        attn = self.layers[0].self_attn
        block_attn = DSparkGQABlockAttention(
            torch.get_default_device(),
            max_bs=int(atom_config.max_num_seqs),
            max_model_len=int(atom_config.max_model_len),
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            block_rows=int(atom_config.speculative_config.num_speculative_tokens),
            head_size=attn.head_dim,
            block_size=int(self.kv_pool_spec.entry_tokens(atom_config)),
            kv_dtype=draft_kv_dtype(atom_config),
            softmax_scale=attn.head_dim**-0.5,
        )
        for layer in self.layers:
            layer.self_attn.use_block_read(block_attn)

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables,
        block_positions,  # [running_bs, T]
        slots,  # [running_bs, T] the block's KV write slots
        ctx_lens,  # [running_bs] anchor + 1 + T, read as `context_lens`
        allocated_pages,  # [>= scheduled_bs] host, blocks each request holds
        scheduled_bs: int,
        entry_tokens: int,  # tokens one block id spans, the slots' unit
    ) -> None:
        """Keep the block inside the blocks each request was allocated.

        The block writes up to T positions past the anchor, which can run past
        the scheduler's allocation; that table column reads as block 0, another
        request's entry. Such positions are neither written (slot -1) nor read
        (the context is capped at the allocation), as MiniMax-M3.1's draft does.
        """
        alloc = self._alloc_cap.apply(
            allocated_pages, scheduled_bs, block_positions, slots, entry_tokens
        )
        torch.minimum(ctx_lens, alloc, out=ctx_lens)

    # -- the target feature this draft was trained on ----------------------

    def target_aux_capture_spec(self, layer_ids, hidden_size, target_layers):
        """Tap the target's AGGREGATE stream, not its running prefix.

        Kimi-K3 does not carry a plain pre-norm residual stream. It carries a
        BANK of per-``attn_res_block_size`` snapshots plus a RUNNING PREFIX,
        recombined at each attn-residual site by a softmax mixture over the
        bank plus the prefix (`AttnRes`). ATOM's generic aux tap reconstructs
        the prefix alone (`KimiDecoderLayer.aux_hidden_state`) and discards the
        bank.

        This checkpoint was trained on the mixture the NEXT target layer forms,
        so that is the feature it has to see; the target layer reconstructs it
        (`KimiDecoderLayer.aggregate_hidden_state`). It is a per-draft
        contract: the MLA draft (`KimiK3DSpark`) wants the raw prefix, and gets
        it by not implementing this method.
        """
        # lid -> the layer that scores lid's aggregate. Keyed by the layer's OWN
        # `layer_idx`, which is what the extract can read back off the module it
        # is handed: a module int attribute, which Dynamo constant-folds inside
        # the target's traced region, where an `id()`-keyed lookup would
        # graph-break.
        taps: dict[int, nn.Module] = {}
        for lid in layer_ids:
            if lid == -1:
                continue  # the embedding tap: no mixture below the first layer
            # Anything else that cannot form the aggregate would leave this tap
            # on the prefix, a feature the checkpoint was not trained on.
            if not 0 <= lid < len(target_layers) - 1:
                raise ValueError(
                    f"DSpark GQA aggregate tap on target layer {lid}: needs a "
                    f"successor layer in a {len(target_layers)}-layer stack"
                )
            # The checkpoint arch is generic (see `hf_config_override`), so the
            # target is checked here, where its layers are known.
            if not hasattr(target_layers[lid], "check_aggregate_source"):
                raise ValueError(
                    "The Kimi-K3 DSpark GQA draft serves a Kimi-K3 target only; "
                    f"target layer {lid} is {type(target_layers[lid]).__name__}"
                )
            if target_layers[lid].layer_idx != lid:
                raise ValueError(
                    f"DSpark GQA aggregate tap on target layer {lid}: the stack's "
                    f"layer {lid} has layer_idx {target_layers[lid].layer_idx}"
                )
            target_layers[lid].check_aggregate_source(target_layers[lid + 1])
            taps[lid] = target_layers[lid + 1]

        if taps:
            logger.info(
                "DSpark GQA aggregate tap armed on target layers %s", sorted(taps)
            )

        def extract(output, block):
            # The base hook copies the mix in place into the same fixed capture
            # buffer the prefix would have filled, so cudagraph capture is
            # unaffected.
            return block.aggregate_hidden_state(output, taps[block.layer_idx])

        return AuxCaptureSpec(layer_ids, hidden_size, extract)

    def write_context_kv(self, taps, positions, slot_mapping=None) -> None:
        # Verify rows reach `mtp_k` past the request's last token, so past the
        # rope cache near `max_model_len` (`_check_rope_covers`). Out of place:
        # `positions` is the target's.
        super().write_context_kv(
            taps, positions.clamp(max=self._rope_max_row), slot_mapping
        )

    # -- DSparkDraftModel contract ----------------------------------------

    def rope_positions(self, positions):
        # Block positions run T past `max_model_len`, so they pass the cache in
        # the last T of a max-length context, whose drafts cannot be verified
        # anyway (`_check_rope_covers`).
        return positions.clamp(max=self._rope_max_row)
