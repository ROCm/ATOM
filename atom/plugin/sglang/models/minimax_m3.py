from __future__ import annotations

from types import MethodType

import torch

from atom.model_ops.layernorm import fused_qk_norm
from atom.models.minimax_m3 import MiniMaxM3Attention, MiniMaxM3SparseAttention
from atom.plugin.sglang.attention_backend.minimax_m3_sparse import (
    minimax_m3_sparse_attention_for_sglang,
)


def _gemma_qk_norm_for_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm,
    k_norm,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = fused_qk_norm(
        q.view(-1, num_q_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        q_norm.weight,
        k_norm.weight,
        q_norm.variance_epsilon,
        add_unit_offset=True,
    )
    return (
        q.view(-1, num_q_heads * head_dim),
        k.view(-1, num_kv_heads * head_dim),
    )


def _patch_minimax_m3_dense_attention_for_sglang(module: MiniMaxM3Attention) -> None:
    inner_attn = getattr(getattr(module, "attn", None), "attn", None)
    if inner_attn is not None:
        setattr(inner_attn, "_atom_minimax_m3_dense_mha", True)


def _sparse_forward_for_sglang(
    self: MiniMaxM3SparseAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states, x_scale=hidden_states_scale)
    if isinstance(qkv, tuple):
        qkv = qkv[0]

    q, k, v, index_q, index_k = qkv.split(
        [
            self.q_size,
            self.kv_size,
            self.kv_size,
            self.index_q_size,
            self.idx_head_dim,
        ],
        dim=-1,
    )
    q, k = _gemma_qk_norm_for_sglang(
        q,
        k,
        self.q_norm,
        self.k_norm,
        self.num_heads,
        self.num_kv_heads,
        self.head_dim,
    )
    q, k = self.rotary_emb(positions, q, k)

    index_q, index_k = _gemma_qk_norm_for_sglang(
        index_q,
        index_k,
        self.index_q_norm,
        self.index_k_norm,
        self.num_idx_heads,
        1,
        self.idx_head_dim,
    )
    index_q, index_k = self.index_rotary_emb(positions, index_q, index_k)

    attn_output = minimax_m3_sparse_attention_for_sglang(
        q,
        k,
        v,
        index_q,
        index_k,
        self,
    )
    return self.o_proj(attn_output)


def _patch_minimax_m3_sparse_attention_for_sglang(
    module: MiniMaxM3SparseAttention,
) -> None:
    if getattr(module, "_atom_sglang_minimax_m3_sparse_patched", False):
        return
    # SGLang's token_to_kv_pool APIs are keyed by layer_id.  The native ATOM
    # layer uses layer_num, so expose both names for the plugin helper.
    module.layer_id = module.layer_num
    module.forward = MethodType(_sparse_forward_for_sglang, module)
    module._atom_sglang_minimax_m3_sparse_patched = True


def setup_minimax_m3_for_sglang(model) -> None:
    """Patch MiniMax-M3 modules for SGLang plugin mode."""

    for module in model.modules():
        if isinstance(module, MiniMaxM3Attention):
            _patch_minimax_m3_dense_attention_for_sglang(module)
        elif isinstance(module, MiniMaxM3SparseAttention):
            _patch_minimax_m3_sparse_attention_for_sglang(module)
