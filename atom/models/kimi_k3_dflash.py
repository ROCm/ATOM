# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 DFlash draft: the SpecForge/SGLang DSpark speculator.

`RadixArk/Kimi-K3-DSpark`: the same block-parallel draft, Markov head and
confidence head as `kimi_k3_dspark.py`, on 5 Qwen3-style GQA layers instead of
MLA. `embed_tokens` and `lm_head` are shared from the target. The fusion width
is ``len(target_layer_ids)`` (5), not the config's ``num_target_layers`` (the
target's depth).

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
no KV scales, so every layer binds one shared 1.0, and the pool declares no
per-token scale planes.
"""

import logging
from typing import ClassVar

import aiter
import torch
from aiter.rotary_embedding import get_rope
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.config import QuantizationConfig
from atom.model_ops.attentions.dflash_draft_attn import DFlashDraftUnifiedAttention
from atom.model_ops.kimi_k3 import apply_attn_res
from atom.model_ops.layernorm import DualRMSNorm, RMSNorm
from atom.model_ops.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.models.dspark_draft import (
    AllocationCap,
    DSparkDraftModel,
    DSparkMarkovHead,
    KVCacheBinding,
    draft_rope_scaling,
    linear_out,
)
from atom.models.qwen3 import Qwen3MLP
from atom.spec_decode.draft_kv import DraftKvPoolSpec
from atom.spec_decode.drafter import AuxCaptureSpec, _resolve_decoder_layers
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.utils.forward_context import get_forward_context

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

        self.block_attn = block_attn

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
        self.scaling = self.head_dim**-0.5
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
            self.head_dim, rotary_dim=self.head_dim, **_draft_rope_kwargs(config)
        )

        self.kv = KVCacheBinding(layer_num)

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
        k_cache, v_cache = self._caches()
        if k_cache is None:
            # Warmup runs before the pool is allocated, and still has to run
            # the rest of the block so profiling counts its activations.
            o = q.new_zeros((n, self.num_heads * self.head_dim))
            return linear_out(self.o_proj(o))

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
            # The pool's scale is 1.0 (`DraftKvBuilder`), and a plain cast turns
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

        Only the k and v rows of the fused QKV weight run: the context rows
        never need a query. The draft is BF16 (`_draft_quant_config`), whose
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

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """The context rows bypass `input_layernorm`.

        The reference feeds `hidden_norm(fc(aux))` straight into k_proj/v_proj;
        only the noise stream goes through the layer's input norm.
        """
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping)


def _draft_rope_kwargs(config) -> dict:
    """`get_rope` arguments from the DRAFT's own config, never the target's.

    The two ropes differ, and the target's loads clean, so nothing would catch
    the draft inheriting it.
    """
    rope_theta, rope_scaling = draft_rope_scaling(config)
    return {
        "max_position": config.max_position_embeddings,
        "base": rope_theta,
        "rope_scaling": rope_scaling,
        "is_neox_style": True,
    }


def _check_rope_covers(max_model_len: int, rope_rows: int) -> None:
    """Refuse a `max_model_len` past the draft's rope cache.

    The context write ropes its positions unclamped, and rows past the cache
    read out of bounds rather than fail.
    """
    if max_model_len > rope_rows:
        raise ValueError(
            f"max_model_len {max_model_len} exceeds the DFlash draft's rope "
            f"cache of {rope_rows} positions; lower --max-model-len to at "
            f"most {rope_rows}"
        )


def _draft_quant_config(config) -> QuantizationConfig:
    """The draft's own quantization: none.

    The atom config copied onto the draft carries the TARGET's offline and
    online quantization, which do not describe this checkpoint (BF16, no
    `quantization_config`). Taken as is, `--online_quant_config` would
    re-quantize every draft linear its exclude list misses, and the draft would
    run as a model it was not validated as.
    """
    draft_qc = QuantizationConfig(config)
    if draft_qc.quant_method:
        raise ValueError(
            "the Kimi-K3 DFlash draft supports BF16 weights only, got "
            f"quant_method={draft_qc.quant_method!r}"
        )
    return draft_qc


class KimiK3DFlash(DSparkDraftModel):
    """The SpecForge DFlash draft for Kimi-K3.

    Registered as ``K3DFlashModel``: the checkpoint ships
    ``architectures: ["DSparkDraftModel"]``, which SpecForge stamps on every
    DFlash draft and which collides with this module's own base class, so
    `SpeculativeConfig.hf_config_override` renames it.
    """

    block_attention_is_mla = False
    kv_pool_spec = DraftKvPoolSpec(flash_layout=True, in_dcp_group=False)

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
        quant_config = _draft_quant_config(config)

        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = config.hidden_size
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = list(config.dspark_target_layer_ids)

        # Aux fusion: [N, hidden * len(target_layer_ids)] -> [N, hidden].
        # Replicated: the aux hidden states arrive replicated, and a
        # row-parallel split would cost an all-reduce every drafting step.
        self.fc = ReplicatedLinear(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            prefix="fc",
        )
        self.hidden_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # One block read for every layer, built before any capture.
        block_attn = DFlashDraftUnifiedAttention(
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
        # The last row of the rope cache. YaRN sizes it from
        # `original_max_position_embeddings * factor`, which equals the target's
        # default `max_model_len`, while block positions reach T past it.
        self._rope_max_row = (
            int(self.layers[0].self_attn.rotary_emb.cos_cache.shape[0]) - 1
        )
        _check_rope_covers(int(atom_config.max_model_len), self._rope_max_row + 1)
        self.markov_head = DSparkMarkovHead(
            self.vocab_size, int(config.dspark_markov_rank)
        )

        # Bound by `share_with_target`.
        self.embed_tokens = None
        self.lm_head = None

        # `prepare_paged_block`'s per-request allocation.
        self._alloc_cap = AllocationCap(
            int(atom_config.max_num_seqs), torch.get_default_device()
        )

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables,
        block_positions,  # [running_bs, T]
        slots,  # [running_bs, T] the block's KV write slots
        ctx_lens,  # [running_bs] anchor + 1 + T, read as `context_lens`
        allocated_pages,  # [>= scheduled_bs] host, blocks each request holds
        scheduled_bs: int,
    ) -> None:
        """Keep the block inside the blocks each request was allocated.

        The block writes up to T positions past the anchor, which can run past
        the scheduler's allocation; that table column reads as block 0, another
        request's entry. Such positions are neither written (slot -1) nor read
        (the context is capped at the allocation), as MiniMax-M3.1's draft does.
        """
        # One block id spans one pool entry; the bound FLASH page is that wide.
        alloc = self._alloc_cap.apply(
            allocated_pages,
            scheduled_bs,
            block_positions,
            slots,
            self.layers[0].self_attn.kv.k_cache.shape[1],
        )
        torch.minimum(ctx_lens, alloc, out=ctx_lens)

    # -- the target feature this draft was trained on ----------------------

    def target_aux_capture_spec(self, layer_ids, hidden_size, target_model):
        """Tap the target's AGGREGATE stream, not its running prefix.

        Kimi-K3 does not carry a plain pre-norm residual stream. It carries a
        BANK of per-``attn_res_block_size`` snapshots plus a RUNNING PREFIX,
        recombined at each attn-residual site by a softmax mixture over the
        bank plus the prefix (`AttnRes`). ATOM's generic aux tap reconstructs
        the prefix alone (`KimiDecoderLayer.aux_hidden_state` sums the layer's
        deferred addends into it) and discards the bank.

        This checkpoint was trained on the mixture, so that is the feature it
        has to see. It is a per-draft contract: the MLA draft (`KimiK3DSpark`)
        wants the raw prefix, and gets it by not implementing this method.

        The mixture is ATOM's own `apply_attn_res`, scored by the NEXT target
        layer's `self_attention_attn_res.score_weight` and called with
        ``out_norm_weight=None`` so it comes back un-normed. The target model
        is not touched; the reconstruction lives in this draft.
        """
        base_extract = DSparkProposer._extract_layer_hidden
        layers = _resolve_decoder_layers(target_model)

        # lid -> (num_valid_blocks, the AttnRes that scores the mixture).
        # Keyed by the layer's OWN `layer_idx`, which is what the extract can
        # read back off the module it is handed: a module int attribute, which
        # Dynamo constant-folds inside the target's traced region, where an
        # `id()`-keyed lookup would graph-break.
        taps: dict[int, tuple[int, nn.Module]] = {}
        for lid in layer_ids:
            if lid == -1:
                continue  # the embedding tap: no mixture below the first layer
            # Anything else that cannot form the aggregate would leave this tap
            # on the prefix, a feature the checkpoint was not trained on.
            if not 0 <= lid < len(layers) - 1:
                raise ValueError(
                    f"DFlash aggregate tap on target layer {lid}: needs a "
                    f"successor layer in a {len(layers)}-layer stack"
                )
            if getattr(layers[lid], "layer_idx", lid) != lid:
                raise ValueError(
                    f"DFlash aggregate tap on target layer {lid}: the stack's "
                    f"layer {lid} has layer_idx {layers[lid].layer_idx}"
                )
            res = getattr(layers[lid + 1], "self_attention_attn_res", None)
            if res is None or not getattr(res, "enabled", False):
                raise ValueError(
                    f"DFlash aggregate tap on target layer {lid}: target layer "
                    f"{lid + 1} has no enabled attn-residual"
                )
            block_size = getattr(res, "block_size", None)
            if not block_size:
                raise ValueError(
                    f"DFlash aggregate tap on target layer {lid}: target layer "
                    f"{lid + 1}'s attn-residual has no block size"
                )
            # Folded here, once after load, never inside the traced and
            # captured target forward the extract runs in.
            if res.score_weight is None:
                res.process_weights_after_loading()
            if res.score_weight is None:
                raise RuntimeError(
                    f"DFlash aggregate tap: target layer {lid + 1}'s "
                    "attn-residual has no score weight"
                )
            # Ceiling division: the bank holds one snapshot per closed block
            # below this layer.
            taps[lid] = (-(-(lid + 1) // block_size), res)

        if not taps:
            # Only the embedding tap: the prefix IS the whole stream there.
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
                # A layout change in the target's return would otherwise drop
                # the draft back to the prefix feature with no error.
                raise RuntimeError(
                    f"DFlash aggregate tap on target layer {block.layer_idx}: "
                    f"expected a bank of >= {nvb} blocks, got "
                    f"{None if bank is None else tuple(bank.shape)}"
                )
            # `out_norm_weight=None` -> the mix comes back un-normed. The base
            # hook then copies it in place into the same fixed capture buffer it
            # would have filled with `prefix` -- same buffer, same shape, so
            # cudagraph capture is unaffected.
            mixed, _ = apply_attn_res(
                prefix,
                bank[:, :nvb, :],
                res.score_weight,
                res.eps,
                None,
                None,
                res.out_eps,
            )
            return mixed

        return AuxCaptureSpec(layer_ids, hidden_size, extract)

    # -- DSparkDraftModel contract ----------------------------------------

    def write_context_kv(
        self,
        aux_concat: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None = None,
    ):
        # Under TR_HYBRID the runner hands over `running_tokens` rows of target
        # hidden state but only the scheduled rows' `positions`; the padded tail
        # has no slot to land in, so it is dropped before the projection.
        super().write_context_kv(
            aux_concat[: positions.shape[0]], positions, slot_mapping
        )

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
        # Block positions run T past `max_model_len`, so they pass the cache in
        # the last T of a max-length context, whose drafts cannot be verified
        # anyway. Context positions stay below it, which `_check_rope_covers`
        # holds inside the cache.
        positions = positions.clamp(max=self._rope_max_row)
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
