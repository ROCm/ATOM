"""Qwen3.8-Flash-Next full-attention layer: QSA sparse paged GQA with an index side branch.

Port of `qwen3_8_flash_next/nvidia/qsa.py:Qwen3_8FlashNextQSAAttention`.

12 of Qwen3.8-Flash-Next's 48 layers are full attention, and every one of them replaces
dense attention with Query-Sparse Attention: a 4-head MQA "indexer" scores
mean-pooled groups of 4 past keys, keeps the best 512 groups, and the real
24-head GQA reads only the ~2051 token positions those groups expand to. Below
`indexer_budget` tokens of context the selection covers everything visible, so
the result is exactly dense attention; above it, it is the model's intended
sparse approximation.

Three paged caches per layer, all riding the SAME block table as the main pool:

  * `k_cache` / `v_cache` -- the ordinary BF16 K/V, one row per token;
  * `raw_key_cache` -- the indexer's key BEFORE normalization, one row per
    token. It has to be cached raw because a group's mean must be taken over
    raw keys, and a group can straddle a prefill chunk boundary;
  * `compressed_key_cache` -- the pooled, normalized, rotated group key, one
    row per COMPLETE group, i.e. `block_size / compress_ratio` rows per block.

Everything is addressed by the flat slot index ATOM already computes for the
main pool, so no second block allocator is involved.

RoPE is mRoPE (`mrope_section [11, 11, 10]`, interleaved). Text requests hand
it three identical position rows, which makes it identical to 1D RoPE, so the
same code path serves both. Image and video requests hand it three genuinely
different rows, and then the compressed key -- which is rotated at the
position of its group's FIRST token -- can no longer recover that position
arithmetically once the group is behind the current chunk. That is what
`rope_position_cache` is for: the per-token 3-axis positions ride alongside the
raw index keys so pooling can read them back.
"""

from types import SimpleNamespace

import aiter
import torch
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.model_ops.layernorm import DualRMSNorm, GemmaRMSNorm
from atom.model_ops.linear import (
    QKVGParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.qwen4_exp.ops.gated import (
    is_per_token_fp8_linear,
    sigmoid_mul,
    sigmoid_mul_quant,
)
from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_apply_mrope,
    qsa_cache_formats,
    qsa_cache_row_layout,
    qsa_compress_groups,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
    qsa_store_rows,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context


def build_qwen4_exp_rope(config, head_size: int):
    """mRoPE over the leading `rotary_dim` channels of `head_size`.

    Both the attention heads (256) and the indexer heads (128) rotate the same
    64 leading dimensions, so the two instances share an identical cos/sin
    cache and differ only in the head size they reshape by. Built as mRoPE
    even for text-only serving: with three equal position rows it reduces
    exactly to 1D RoPE, so one path covers both.
    """
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_theta = float(rope_parameters.get("rope_theta", 10000.0))
    partial = float(rope_parameters.get("partial_rotary_factor", 1.0))
    rotary_dim = int(int(config.head_dim) * partial)
    # `get_rope` only reaches its mRoPE branch when the scaling dict names both
    # a rope type and the sections; anything else must go through as None or it
    # trips over the missing keys.
    scaling = (
        dict(rope_parameters)
        if rope_parameters.get("mrope_section")
        and (rope_parameters.get("rope_type") or rope_parameters.get("type"))
        else None
    )
    return get_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=int(config.max_position_embeddings),
        base=rope_theta,
        is_neox_style=True,
        rope_scaling=scaling,
    )


def canonical_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Per-token positions as `[tokens, 1, 3]` int64, for the position cache."""
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    elif positions.shape[0] == 1:
        positions = positions.expand(3, -1)
    return positions.transpose(0, 1).unsqueeze(1).to(torch.int64)


class Qwen4ExpIndexer(nn.Module):
    """Replicated index Q/K projection plus the QSA selection contract."""

    def __init__(
        self,
        config,
        rotary_emb: nn.Module | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        if self.index_kv_heads != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if (
            self.compress_ratio <= 0
            or self.token_topk <= 0
            or self.token_topk % self.compress_ratio
        ):
            raise ValueError(
                "QSA budget must be positive and divisible by its compression ratio"
            )
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=eps)

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized+rotated index Q, and RAW index K for later pooling."""
        otype = hidden_states.dtype
        qk = self.index_qk_proj(hidden_states, otype=otype)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim))
        q, _ = qsa_apply_mrope(
            self.rotary_emb,
            positions,
            q.view(-1, self.index_n_heads, self.index_head_dim),
        )
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and rotate it at its group's first-token position."""
        keys = compressed_keys.reshape(-1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            # `[groups, 3]` -> the `[3, groups]` row layout mRoPE expects.
            positions = first_rope_positions.transpose(0, 1)
        else:
            positions = first_rope_positions[:, 0]
        keys = self.k_layernorm(keys).view(-1, 1, self.index_head_dim)
        keys, _ = qsa_apply_mrope(self.rotary_emb, positions, keys)
        return keys


class Qwen4ExpAttention(nn.Module):
    """Full attention with QSA selection, owning its three paged caches."""

    # Marker read by `Qwen4ExpMetadataBuilder.build_kv_cache_tensor`: this
    # layer does not go through ATOM's `Attention` wrapper, so the binder
    # cannot recognize it by `base_attention`.
    is_qsa_attention = True

    def __init__(
        self,
        config,
        atom_config,
        quant_config=None,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.prefix = prefix
        self.layer_num = layer_num
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_heads % tp_size:
            raise ValueError(f"TP={tp_size} must divide {self.total_num_heads} q heads")
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError(f"TP={tp_size} must divide the KV heads")
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            if tp_size % self.total_num_kv_heads:
                raise ValueError("TP size must be a multiple of the KV head count")
            self.num_kv_heads = 1
        self.head_dim = int(config.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # The checkpoint's q_proj is [24 heads x 2*head_dim] with q and the
        # sigmoid output gate INTERLEAVED per head. QKVGParallelLinear
        # de-interleaves at load into a contiguous [Gate, Q, K, V].
        self.qkv_proj = QKVGParallelLinear(
            int(config.hidden_size),
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            int(config.hidden_size),
            bias=False,
            # The decoder layer all-reduces once for the whole sub-layer.
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Both norms in one launch. It reads `add_unit_offset` off the norms
        # themselves, so the Gemma `(1 + w)` these carry survives the fusion.
        self.qk_norm = DualRMSNorm(
            self.q_norm,
            self.k_norm,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            prefix=f"{prefix}.qk_norm",
        )

        self.rotary_emb = build_qwen4_exp_rope(config, self.head_dim)
        self.indexer = Qwen4ExpIndexer(
            config,
            rotary_emb=build_qwen4_exp_rope(config, int(config.indexer_head_dim)),
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

        # `reshape_and_cache_flash` takes a K and a V scale even when it is
        # not quantizing. The pool is bf16, so both are a fixed 1.0 -- a
        # buffer rather than a per-step tensor so the decode graph captures
        # one address.
        self.register_buffer(
            "_unit_scale", torch.ones(1, dtype=torch.float32), persistent=False
        )

        # Selection scratch is forward-local; sequential layers can reuse the
        # allocator/graph pool instead of retaining one full workspace each.
        # Bound by the metadata builder once the pool is sized.
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.raw_key_cache: torch.Tensor | None = None
        self.compressed_key_cache: torch.Tensor | None = None
        # Only allocated for multimodal serving; None means group positions are
        # derived arithmetically, which is exact while all three mRoPE rows
        # hold the linear position (i.e. text).
        self.rope_position_cache: torch.Tensor | None = None
        # Scale planes of the quantized caches; None while a cache is BF16.
        self.k_scale_cache: torch.Tensor | None = None
        self.v_scale_cache: torch.Tensor | None = None
        self.compressed_key_scale: torch.Tensor | None = None
        # Plugin hosts allocate and bind BF16 caches of their own.
        self.kv_format, self.index_format = (
            ("bf16", "bf16") if is_plugin_mode() else qsa_cache_formats(atom_config)
        )
        # The MTP draft layer sits past the target stack. It keeps BF16 caches
        # under quantized target caches, in a row space of its own: an FP8
        # draft KV costs ~3.5% of the real acceptance length, a BF16 one none.
        self.qsa_row_space = "qsa"
        if (
            layer_num >= int(config.num_hidden_layers)
            and envs.ATOM_QSA_DRAFT_KV_BF16
            and (self.kv_format, self.index_format) != ("bf16", "bf16")
        ):
            self.kv_format = self.index_format = "bf16"
            self.qsa_row_space = "qsa_draft"
        self.profile_block_size = getattr(atom_config, "kv_cache_block_size", 64)

    def bind_caches(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        raw_key_cache: torch.Tensor,
        compressed_key_cache: torch.Tensor,
        rope_position_cache: torch.Tensor | None = None,
        *,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
        compressed_key_scale: torch.Tensor | None = None,
    ) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.raw_key_cache = raw_key_cache
        self.compressed_key_cache = compressed_key_cache
        self.rope_position_cache = rope_position_cache
        self.k_scale_cache = k_scale_cache
        self.v_scale_cache = v_scale_cache
        self.compressed_key_scale = compressed_key_scale

    def _store_kv(self, key, value, slots) -> None:
        if self.k_scale_cache is None:
            aiter.reshape_and_cache_flash(
                key,
                value,
                self.k_cache,
                self.v_cache,
                slots,
                "auto",
                self._unit_scale,
                self._unit_scale,
            )
            return
        qsa_store_quantized_rows(
            self.k_cache,
            self.k_scale_cache,
            slots,
            key,
            self.v_cache,
            self.v_scale_cache,
            value,
        )

    def _attend(self, query, selected, block_tables, token_to_req, decode_reqs):
        return qsa_sparse_paged_gqa(
            query,
            self.k_cache,
            self.v_cache,
            selected,
            block_tables,
            token_to_req,
            softmax_scale=self.scaling,
            num_decode_requests=decode_reqs,
            k_scale=self.k_scale_cache,
            v_scale=self.v_scale_cache,
        )

    def _select_tokens(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        qsa,
    ) -> torch.Tensor:
        """Update both index caches, then return the per-token selection."""
        index_q, raw_key = self.indexer.project_qk(hidden_states, positions)
        qsa_store_rows(self.raw_key_cache, qsa.slot_mapping, raw_key)
        if self.rope_position_cache is not None:
            qsa_store_rows(
                self.rope_position_cache,
                qsa.slot_mapping,
                canonical_rope_positions(positions[..., : hidden_states.shape[0]]),
            )

        pooled, first_positions = qsa_compress_groups(
            self.raw_key_cache,
            qsa.block_tables,
            qsa.token_to_req,
            qsa.logical_positions,
            qsa.compressed_slot_mapping,
            self.indexer.compress_ratio,
            position_cache=self.rope_position_cache,
            # Draft RoPE is one ahead of its logical cache slots.
            rope_position_offset=int(get_forward_context().context.is_draft),
        )
        normalized = self.indexer.normalize_compressed_keys(pooled, first_positions)
        if self.compressed_key_scale is None:
            qsa_store_rows(
                self.compressed_key_cache, qsa.compressed_slot_mapping, normalized
            )
        else:
            qsa_store_quantized_rows(
                self.compressed_key_cache,
                self.compressed_key_scale,
                qsa.compressed_slot_mapping,
                normalized,
            )

        return qsa_select_paged_tokens(
            index_q,
            self.compressed_key_cache,
            qsa.block_tables,
            qsa.token_to_req,
            qsa.logical_positions,
            qsa.seq_lens,
            self.indexer.token_topk,
            self.indexer.compress_ratio,
            max_seq_len=qsa.max_seq_len,
            allow_flydsl=get_forward_context().context.is_prefill,
            compressed_k_scale=self.compressed_key_scale,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """`quantized` is an optional `(fp8, scale)` copy of `hidden_states`
        that an FP8 `qkv_proj` takes instead of quantizing it again; the
        indexer still reads the BF16 `hidden_states`."""
        num_tokens = hidden_states.shape[0]
        qkv = (
            self.qkv_proj(hidden_states)
            if quantized is None
            else self.qkv_proj(quantized[0], x_scale=quantized[1])
        )
        gate, q, k, v = qkv.split(
            [self.q_size, self.q_size, self.kv_size, self.kv_size], -1
        )
        # Normalization and cache writes accept packed token strides without copies.
        q, k = self.qk_norm(q, k)
        query, key = qsa_apply_mrope(
            self.rotary_emb,
            positions,
            q.view(num_tokens, self.num_heads, self.head_dim),
            k.view(num_tokens, self.num_kv_heads, self.head_dim),
        )
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        forward_context = get_forward_context()
        qsa = forward_context.attn_metadata.qsa_metadata
        if qsa is None:
            attn_out = self._profile_attention(
                query, key, value, hidden_states, positions
            )
        else:
            qsa = qsa.for_tokens(num_tokens)
            self._store_kv(key, value, qsa.slot_mapping)
            selected = self._select_tokens(hidden_states, positions, qsa)
            attn_out = self._attend(
                query,
                selected,
                qsa.block_tables,
                qsa.token_to_req,
                (
                    None
                    if forward_context.context.is_prefill
                    else qsa.block_tables.shape[0]
                ),
            )

        if is_per_token_fp8_linear(self.o_proj):
            gated, gated_scale = sigmoid_mul_quant(
                attn_out.reshape(num_tokens, -1), gate, self.o_proj.params_dtype
            )
            return self.o_proj(gated, x_scale=gated_scale)
        gated = sigmoid_mul(attn_out.reshape(num_tokens, -1), gate)
        return self.o_proj(gated)

    def _profile_attention(self, query, key, value, hidden_states, positions):
        """Exercise real indexer, cache and sparse workspaces before pool sizing.

        A synthetic request has local pages; no persistent cache may survive
        this pass. This deliberately uses the same operators as serving.
        """
        tokens = query.shape[0]
        block = self.profile_block_size
        ratio = self.indexer.compress_ratio
        pages = max(1, (tokens + block - 1) // block)
        device = query.device
        logical = torch.arange(tokens, device=device, dtype=torch.int64)
        metadata = SimpleNamespace(
            slot_mapping=logical,
            compressed_slot_mapping=torch.where(
                (logical + 1) % ratio == 0, logical // ratio, -1
            ),
            logical_positions=logical,
            token_to_req=torch.zeros(tokens, device=device, dtype=torch.int32),
            block_tables=torch.arange(pages, device=device, dtype=torch.int32)[None],
            seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
            max_seq_len=tokens,
        )
        names = (
            "k_cache",
            "v_cache",
            "raw_key_cache",
            "compressed_key_cache",
            "rope_position_cache",
        )
        scale_names = ("k_scale_cache", "v_scale_cache", "compressed_key_scale")
        saved = [getattr(self, name) for name in names]
        saved_scales = {name: getattr(self, name) for name in scale_names}

        def paged(rows, heads, fmt, head_dim):
            dim, dtype, scale_tail, scale_dtype = qsa_cache_row_layout(fmt, head_dim)
            values = torch.zeros((pages, rows, heads, dim), device=device, dtype=dtype)
            if scale_tail is None:
                return values, None
            return values, torch.ones(
                (pages, rows, heads, *scale_tail), device=device, dtype=scale_dtype
            )

        k_cache, k_scale = paged(
            block, self.num_kv_heads, self.kv_format, self.head_dim
        )
        v_cache, v_scale = paged(
            block, self.num_kv_heads, self.kv_format, self.head_dim
        )
        compressed, compressed_scale = paged(
            block // ratio, 1, self.index_format, self.indexer.index_head_dim
        )
        self.bind_caches(
            k_cache,
            v_cache,
            query.new_zeros((pages, block, 1, self.indexer.index_head_dim)),
            compressed,
            torch.zeros((pages, block, 1, 3), device=device, dtype=torch.int64),
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
            compressed_key_scale=compressed_scale,
        )
        try:
            self._store_kv(key, value, logical)
            selected = self._select_tokens(hidden_states, positions, metadata)
            return self._attend(
                query, selected, metadata.block_tables, metadata.token_to_req, None
            )
        finally:
            self.bind_caches(*saved, **saved_scales)
