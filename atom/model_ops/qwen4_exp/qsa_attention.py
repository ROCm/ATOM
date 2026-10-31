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

import functools
import logging
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
    qsa_expand_block_indices,
    qsa_select_paged_groups,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
    qsa_store_rows,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger(__name__)

# Selection widths the aiter FlyDSL kernel accepts: 4T + 3 for T complete
# four-token groups (T = 256, 512, 768, 1024), then the 0..3 causal tail.
_FLYDSL_WIDTHS = (1027, 2051, 3075, 4099)


@functools.cache
def _flydsl_arch_supported() -> bool:
    # aiter's entry raises on anything but gfx942 instead of falling back.
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx942"


@functools.cache
def _flydsl_attention_op():
    # Imported on first use so the flag-off path needs no FlyDSL QSA in aiter.
    # aiter.ops.qsa is aiter's entry for consumers (ROCm/aiter#6576).
    from aiter.ops.qsa import qsa_sparse_prefill

    return qsa_sparse_prefill


def qsa_rank_heads(
    num_attention_heads: int, num_key_value_heads: int, tp_size: int
) -> tuple[int, int]:
    """Per-rank (query heads, KV heads) under TP. With fewer KV heads than
    ranks each rank keeps one replicated KV head."""
    if num_attention_heads % tp_size:
        raise ValueError(f"TP={tp_size} must divide {num_attention_heads} q heads")
    if num_key_value_heads >= tp_size:
        if num_key_value_heads % tp_size:
            raise ValueError(f"TP={tp_size} must divide the KV heads")
        return num_attention_heads // tp_size, num_key_value_heads // tp_size
    if tp_size % num_key_value_heads:
        raise ValueError("TP size must be a multiple of the KV head count")
    return num_attention_heads // tp_size, 1


def flydsl_fallback_reason(
    compress_ratio: int,
    token_topk: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    kv_bf16: bool,
) -> str | None:
    """Why QSA layers with this indexer, per-rank head shape and KV format
    cannot take the aiter FlyDSL attention, or None if they can. The kernel
    expects min(v // 4, T) blocks for a row with v visible tokens (width
    4T + 3); another budget or ratio selects a different number, so padding
    its selection would give wrong output. The one authority for the layer's
    fallback and the SGLang bridge's layout build; the per-call conditions
    (host layout, selection byte span, graph capture, query dtype) stay with
    the layer. `ATOM_QSA_FLYDSL_ATTN` is checked by the callers."""
    if not kv_bf16:
        return "the KV cache is not BF16"
    if compress_ratio != 4 or token_topk + 3 not in _FLYDSL_WIDTHS:
        return f"indexer compress ratio {compress_ratio} / budget {token_topk}"
    if num_kv_heads != 1 or num_heads not in (12, 6, 3) or head_dim != 256:
        return (
            f"per-rank head shape {num_heads} q / {num_kv_heads} kv heads "
            f"x {head_dim} (needs 12, 6 or 3 q / 1 kv x 256)"
        )
    if not _flydsl_arch_supported():
        return "the GPU is not gfx942"
    return None


@functools.cache
def log_flydsl_fallback(reason: str) -> None:
    """Warn once per distinct reason, so a server started with the flag shows
    whether, and why, QSA attention stays on the native kernel."""
    logger.warning(
        "ATOM_QSA_FLYDSL_ATTN=1, but QSA attention uses the native kernel: %s",
        reason,
    )


@functools.cache
def _log_flydsl_active() -> None:
    logger.info("QSA attention runs eager prefill on the aiter FlyDSL kernel")


def _flydsl_call_fallback_reason(query, selected, width, qsa) -> str | None:
    """Why this one laid-out prefill call cannot take the FlyDSL attention, or None."""
    if width is None:
        return f"selection wider than {_FLYDSL_WIDTHS[-1]} columns"
    # aiter rejects a padded selection spanning 2**31 bytes or more.
    if query.shape[0] * width * selected.element_size() >= 2**31:
        return "padded selection spans 2**31 bytes or more"
    if torch.cuda.is_current_stream_capturing():
        return "graph capture"
    if query.dtype != torch.bfloat16:
        return f"query dtype {query.dtype}"
    if sum(qsa.host_lens[0]) != query.shape[0]:
        return "host query lengths do not cover the query rows"
    return None


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
        self.num_heads, self.num_kv_heads = qsa_rank_heads(
            self.total_num_heads, self.total_num_kv_heads, tp_size
        )
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
        rows=None,
    ) -> torch.Tensor:
        """Update both index caches, then return the per-token selection.

        The indexer is replicated, so with `rows` (the caller's token sharding:
        `start`, `chunk`, `num_tokens` and a row `gather`) each rank scores and
        selects only its own query rows and the selected groups are
        all-gathered before the (cheap) expansion. Every row's selection
        depends only on its own query, so the result is the same.
        """
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

        allow_flydsl = get_forward_context().context.is_prefill
        if rows is None:
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
                allow_flydsl=allow_flydsl,
                compressed_k_scale=self.compressed_key_scale,
            )
        local = slice(rows.start, min(rows.start + rows.chunk, rows.num_tokens))
        groups, width = qsa_select_paged_groups(
            index_q[local],
            self.compressed_key_cache,
            qsa.block_tables,
            qsa.token_to_req[local],
            qsa.logical_positions[local],
            qsa.seq_lens,
            self.indexer.token_topk,
            self.indexer.compress_ratio,
            max_seq_len=qsa.max_seq_len,
            allow_flydsl=allow_flydsl,
            compressed_k_scale=self.compressed_key_scale,
        )
        if groups.shape[0] < rows.chunk:
            groups = nn.functional.pad(groups, (0, 0, 0, rows.chunk - groups.shape[0]))
        return qsa_expand_block_indices(
            rows.gather(groups),
            qsa.logical_positions,
            qsa.seq_lens,
            qsa.token_to_req,
            self.indexer.compress_ratio,
            self.indexer.token_topk,
            output_width=width,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
        rows=None,
    ) -> torch.Tensor:
        """`quantized` is an optional `(fp8, scale)` copy of `hidden_states`
        that an FP8 `qkv_proj` takes instead of quantizing it again; the
        indexer still reads the BF16 `hidden_states`. `rows`: the decoder's
        token sharding, see `_select_tokens`."""
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
            selected = self._select_tokens(hidden_states, positions, qsa, rows)
            attn_out = None
            if forward_context.context.is_prefill:
                attn_out = self._flydsl_attention(query, key, value, selected, qsa)
            if attn_out is None:
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

    def _flydsl_attention(self, query, key, value, selected, qsa):
        """aiter FlyDSL sparse attention for an eager prefill, or None for the
        native kernel. With the flag on, the first FlyDSL call and each
        distinct fallback reason are logged once."""
        if not envs.ATOM_QSA_FLYDSL_ATTN:
            return None
        width = next((w for w in _FLYDSL_WIDTHS if w >= selected.shape[1]), None)
        # The prefixed path gathers raw cache rows; the kernel takes BF16 only.
        reason = flydsl_fallback_reason(
            self.indexer.compress_ratio,
            self.indexer.token_topk,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            kv_bf16=self.k_scale_cache is None,
        )
        if reason is None and qsa.host_lens is None:
            # Not laid out for FlyDSL (target verify, warmup, ATOM's own
            # runner); the SGLang bridge logs its own reasons for skipping.
            return None
        reason = reason or _flydsl_call_fallback_reason(query, selected, width, qsa)
        if reason is not None:
            log_flydsl_fallback(reason)
            return None
        _log_flydsl_active()
        query_lens, seq_lens = qsa.host_lens
        if qsa.context_slots is None:
            k, v = key.contiguous(), value.contiguous()
        else:
            k = self.k_cache.flatten(0, 1).index_select(0, qsa.context_slots)
            v = self.v_cache.flatten(0, 1).index_select(0, qsa.context_slots)
        # The expansion drops columns no row can reach (short contexts select
        # every complete block, then the tail, left-aligned), so padding with -1
        # keeps each row's layout. The kernel does not validate it: a malformed
        # selection gives wrong output.
        indices = torch.nn.functional.pad(
            selected, (0, width - selected.shape[1]), value=-1
        )
        return _flydsl_attention_op()(
            query.contiguous(),
            k,
            v,
            indices,
            query_lens=query_lens,
            prefix_lens=tuple(s - q for q, s in zip(query_lens, seq_lens)),
            softmax_scale=self.scaling,
        )

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
