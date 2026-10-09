"""Qwen3.5-35B-A3B vLLM MHA path for FULL-cudagraph correctness on vLLM 0.27+.

Routes Qwen3.5-35B-A3B hybrid models through a legacy PagedAttention-style wrapper
around vLLM native Attention (via attention_bridge) instead of the isolated
AttentionForVllmMHA metadata path. Larger Qwen3.5 variants are unaffected.
Applied via plugin registration only.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.config import get_current_atom_config
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.base_attention import BaseAttention
from atom.plugin.vllm.attention_bridge import (
    unified_attention_with_output_base_for_plugin_mode,
)
from atom.plugin.vllm.qwen35_plugin_scope import is_qwen35_35b_a3b_vllm_plugin_model

_QWEN35_ATTENTION_PATCH_APPLIED = False


class Qwen35VllmPagedAttention(BaseAttention):
    """vLLM-native Attention wrapper for Qwen3.5 MHA under ATOM plugin compile."""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes: list[float] | None = None,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: MLAModules | None = None,
        sinks: nn.Parameter | None = None,
        per_layer_sliding_window: int | None = None,
        rotary_emb: nn.Module | None = None,
        prefix: str | None = None,
        q_norm: nn.Module | None = None,
        k_norm: nn.Module | None = None,
        impl_cls: type | None = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=use_mla,
            mla_modules=mla_modules,
            sinks=sinks,
            per_layer_sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            prefix=prefix,
            **kwargs,
        )

        self.use_mla = use_mla
        self.rotary_emb = mla_modules.rotary_emb if use_mla else rotary_emb

        try:
            from vllm.attention.layer import Attention, AttentionType, MLAAttention
        except ImportError:
            from vllm.model_executor.layers.attention import Attention, MLAAttention
            from vllm.v1.attention.backend import AttentionType

        atom_config = get_current_atom_config()
        assert atom_config is not None, "atom_config is required for vLLM plugin"

        cache_config = atom_config.plugin_config.vllm_cache_config
        quant_config = atom_config.plugin_config.vllm_quant_config

        # vLLM 0.27 RocmAttentionImpl rejects legacy ATOM impl kwargs
        # (rotary_emb/q_norm). RoPE is applied in the plugin bridge instead.
        extra_impl_args: dict = {}

        if use_mla:
            self.num_heads = num_heads
            self.v_head_dim = mla_modules.v_head_dim
            self.qk_head_dim = mla_modules.qk_head_dim
            self.qk_nope_head_dim = mla_modules.qk_nope_head_dim
            self.q_proj = mla_modules.q_proj
            self.o_proj = mla_modules.o_proj

            self.attn = MLAAttention(
                num_heads=num_heads,
                scale=scale,
                qk_nope_head_dim=mla_modules.qk_nope_head_dim,
                qk_rope_head_dim=mla_modules.qk_rope_head_dim,
                v_head_dim=mla_modules.v_head_dim,
                q_lora_rank=mla_modules.q_lora_rank,
                kv_lora_rank=mla_modules.kv_lora_rank,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                kv_b_proj=mla_modules.kv_b_proj,
                use_sparse=mla_modules.indexer is not None,
                indexer=mla_modules.indexer,
                **extra_impl_args,
            )
        else:
            self.attn = Attention(
                num_heads=num_heads,
                head_size=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=alibi_slopes,
                cache_config=cache_config,
                quant_config=quant_config,
                logits_soft_cap=None,
                per_layer_sliding_window=per_layer_sliding_window,
                prefix=f"{prefix}",
                attn_type=AttentionType.DECODER,
                kv_sharing_target_layer_name=None,
                **extra_impl_args,
            )

        compilation_config = atom_config.compilation_config
        self.layer_name = prefix
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        if (
            self.use_mla
            and "positions" not in compilation_config.static_forward_context
        ):
            max_num_tokens = (
                atom_config.plugin_config.vllm_scheduler_config.max_num_batched_tokens
            )
            compilation_config.static_forward_context["positions"] = torch.zeros(
                max_num_tokens, dtype=torch.int64, device="cuda"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: torch.Tensor | None = None,
        qkv: torch.Tensor = None,
        **kwargs,
    ):
        return unified_attention_with_output_base_for_plugin_mode(
            query,
            q_scale,
            key,
            value,
            positions,
            layer_name=self.layer_name,
            use_mla=self.use_mla,
            qkv=qkv,
        )


def apply_qwen35_vllm_attention_patch() -> None:
    """Monkeypatch Attention construction for Qwen3.5 vLLM plugin models."""
    global _QWEN35_ATTENTION_PATCH_APPLIED
    if _QWEN35_ATTENTION_PATCH_APPLIED:
        return

    import atom.model_ops.base_attention as base_attention_module

    original_new = base_attention_module.Attention.__new__

    def qwen35_attention_new(cls, *args, **kwargs):
        from atom.plugin.prepare import is_vllm

        if is_vllm() and is_qwen35_35b_a3b_vllm_plugin_model():
            return Qwen35VllmPagedAttention(*args, **kwargs)
        return original_new(cls, *args, **kwargs)

    base_attention_module.Attention.__new__ = qwen35_attention_new
    _QWEN35_ATTENTION_PATCH_APPLIED = True
