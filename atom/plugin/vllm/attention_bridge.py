"""Minimal vLLM plugin bridge for Qwen3.5 legacy PagedAttention path.

Routes compiled unified_attention calls to vLLM native Attention (with optional
RoPE in the bridge) instead of the isolated AttentionForVllmMHA metadata path.
"""

from __future__ import annotations

import os

import torch

from atom.config import get_current_atom_config

_disable_vllm_plugin_attention = os.getenv(
    "ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "0"
).lower() in ("1", "true", "yes")


def unified_attention_with_output_base_for_plugin_mode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    current_atom_config = get_current_atom_config()
    static_forward_context = (
        current_atom_config.compilation_config.static_forward_context
    )

    if use_mla:
        kv_c_normed = k
        k_pe = v
        self = static_forward_context[layer_name]
        q = self.q_proj(q, q_scale)
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        if _disable_vllm_plugin_attention:
            k_pe = k_pe.unsqueeze(1)
            if self.rotary_emb is not None:
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )
        output = self.attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.num_heads * self.v_head_dim),
        )
        return self.o_proj(output)

    self = static_forward_context[layer_name]
    if current_atom_config.plugin_config.vllm_use_atom_attention:
        output = self.attn(q, k, v)
    else:
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
    return output
