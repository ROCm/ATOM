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

import torch
import torch.nn.functional as F
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import QKVGParallelLinear, RowParallelLinear
from atom.model_ops.qwen3_8_flash_next.indexer import Qwen3_8FlashNextIndexer
from atom.model_ops.qwen3_8_flash_next.kernels.qsa_cache_ops import (
    qsa_compress_groups,
    qsa_store_rows,
)
from atom.model_ops.qwen3_8_flash_next.qsa_ops import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)
from atom.utils.forward_context import get_forward_context


def build_qwen3_8_flash_next_rope(config, head_size: int):
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


class Qwen3_8FlashNextAttention(nn.Module):
    """Full attention with QSA selection, owning its three paged caches."""

    # Marker read by `Qwen3_8FlashNextMetadataBuilder.build_kv_cache_tensor`: this
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

        self.rotary_emb = build_qwen3_8_flash_next_rope(config, self.head_dim)
        self.indexer = Qwen3_8FlashNextIndexer(
            config,
            rotary_emb=build_qwen3_8_flash_next_rope(config, int(config.indexer_head_dim)),
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

        max_tokens = atom_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(max_tokens, self.indexer.output_width, dtype=torch.int32),
            persistent=False,
        )
        # Bound by the metadata builder once the pool is sized.
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.raw_key_cache: torch.Tensor | None = None
        self.compressed_key_cache: torch.Tensor | None = None
        # Only allocated for multimodal serving; None means group positions are
        # derived arithmetically, which is exact while all three mRoPE rows
        # hold the linear position (i.e. text).
        self.rope_position_cache: torch.Tensor | None = None

    def bind_caches(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        raw_key_cache: torch.Tensor,
        compressed_key_cache: torch.Tensor,
        rope_position_cache: torch.Tensor | None = None,
    ) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.raw_key_cache = raw_key_cache
        self.compressed_key_cache = compressed_key_cache
        self.rope_position_cache = rope_position_cache

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
        )
        normalized = self.indexer.normalize_compressed_keys(pooled, first_positions)
        qsa_store_rows(
            self.compressed_key_cache, qsa.compressed_slot_mapping, normalized
        )

        num_tokens = hidden_states.shape[0]
        return qsa_select_paged_tokens(
            index_q,
            self.compressed_key_cache,
            qsa.block_tables,
            qsa.token_to_req,
            qsa.logical_positions,
            qsa.seq_lens,
            self.indexer.token_topk,
            self.indexer.compress_ratio,
            out=self.topk_indices_buffer[:num_tokens],
            max_seq_len=qsa.max_seq_len,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        gate, q, k, v = torch.split(
            qkv, [self.q_size, self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # `split` hands back last-dim views; the norm and RoPE kernels below
        # both reshape, which a strided slice cannot do.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q = self.q_norm(q.view(num_tokens, self.num_heads, self.head_dim)).view(
            num_tokens, self.q_size
        )
        k = self.k_norm(k.view(num_tokens, self.num_kv_heads, self.head_dim)).view(
            num_tokens, self.kv_size
        )
        q, k = self.rotary_emb(positions, q, k)

        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        qsa = get_forward_context().attn_metadata.qsa_metadata
        if qsa is None:
            # Warmup / profiling runs with no allocated pool: shape-correct
            # zeros, never a served token.
            attn_out = torch.zeros_like(query)
        else:
            qsa_store_rows(self.k_cache, qsa.slot_mapping, key)
            qsa_store_rows(self.v_cache, qsa.slot_mapping, value)
            selected = self._select_tokens(hidden_states, positions, qsa)
            attn_out = qsa_sparse_paged_gqa(
                query,
                self.k_cache,
                self.v_cache,
                selected,
                qsa.block_tables,
                qsa.token_to_req,
                softmax_scale=self.scaling,
            )

        gated = attn_out.reshape(num_tokens, -1) * F.sigmoid(gate)
        return self.o_proj(gated)
