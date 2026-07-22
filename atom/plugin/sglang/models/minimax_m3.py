from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch

from atom.model_ops.base_attention import BaseAttention
from atom.models.minimax_m3 import MiniMaxM3Attention, MiniMaxM3SparseAttention


@contextmanager
def minimax_m3_native_sparse_attention_construction():
    """Construct MiniMax-M3 attention layers with ATOM's native impls."""

    import atom.models.minimax_m3 as minimax_m3

    previous = minimax_m3.Attention

    def _build_minimax_m3_attention(*args: Any, **kwargs: Any):
        return SGLangATOMMiniMaxM3Attention(*args, **kwargs)

    minimax_m3.Attention = _build_minimax_m3_attention
    try:
        yield
    finally:
        minimax_m3.Attention = previous


class SGLangATOMMiniMaxM3Attention(BaseAttention):
    """Use ATOM native MiniMax-M3 attention under SGLang plugin runtime."""

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
        mla_modules=None,
        sinks=None,
        per_layer_sliding_window=None,
        rotary_emb=None,
        prefix: str | None = None,
        q_norm=None,
        k_norm=None,
        impl_cls=None,
        **kwargs,
    ) -> None:
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
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )

        from atom.config import get_current_atom_config
        from atom.model_ops.attention_mha import PagedAttentionImpl

        atom_config = get_current_atom_config()
        impl_cls = impl_cls or PagedAttentionImpl
        atom_kv_cache_dtype = (
            "fp8" if str(kv_cache_dtype).startswith("fp8") else kv_cache_dtype
        )
        self.use_mla = use_mla
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = atom_kv_cache_dtype
        self.layer_num = layer_num
        self.base_attention = None
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.impl = impl_cls(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=atom_kv_cache_dtype,
            layer_num=layer_num,
            mla_modules=mla_modules,
            sinks=sinks,
            sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            dtype=atom_config.torch_dtype,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )
        self.layer_name = prefix if prefix is not None else f"MHA_{layer_num}"
        static_context = atom_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer: {self.layer_name}")
        static_context[self.layer_name] = self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: torch.Tensor | None = None,
        qkv: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return torch.ops.aiter.unified_attention_with_output_base(
            query,
            q_scale,
            key,
            value,
            positions,
            self.layer_name,
            self.use_mla,
            qkv,
        )


def _patch_minimax_m3_dense_attention_for_sglang(module: MiniMaxM3Attention) -> None:
    if not isinstance(getattr(module, "attn", None), SGLangATOMMiniMaxM3Attention):
        raise RuntimeError(
            "MiniMax-M3 SGLang dense setup expected native ATOM attention. "
            "Ensure the MiniMax-M3 construction context is installed before "
            "model initialization."
        )


def _sparse_forward_native_for_sglang(
    self: MiniMaxM3SparseAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states, x_scale=hidden_states_scale)
    if isinstance(qkv, tuple):
        qkv = qkv[0]
    q, k, v, _, _ = qkv.split(
        [
            self.q_size,
            self.kv_size,
            self.kv_size,
            self.index_q_size,
            self.idx_head_dim,
        ],
        dim=-1,
    )
    attn_output = self.attn(q, k, v, positions, qkv=qkv)
    return self.o_proj(attn_output)


def _patch_minimax_m3_sparse_attention_for_sglang(
    module: MiniMaxM3SparseAttention,
) -> None:
    if getattr(module, "_atom_sglang_minimax_m3_sparse_patched", False):
        return
    # SGLang's token_to_kv_pool APIs are keyed by layer_id.  The native ATOM
    # layer uses layer_num, so expose both names for the plugin helper.
    module.layer_id = module.layer_num
    impl = getattr(getattr(module, "attn", None), "impl", None)
    if not getattr(impl, "is_indexed_sparse_attention", False):
        raise RuntimeError(
            "MiniMax-M3 SGLang native sparse setup expected "
            "SparseMHAPagedAttentionImpl. Ensure the MiniMax-M3 construction "
            "context is installed before model initialization."
        )
    module.forward = MethodType(_sparse_forward_native_for_sglang, module)
    module._atom_sglang_minimax_m3_sparse_patched = True


def setup_minimax_m3_for_sglang(model) -> None:
    """Patch MiniMax-M3 modules for SGLang plugin mode."""

    for module in model.modules():
        if isinstance(module, MiniMaxM3Attention):
            _patch_minimax_m3_dense_attention_for_sglang(module)
        elif isinstance(module, MiniMaxM3SparseAttention):
            _patch_minimax_m3_sparse_attention_for_sglang(module)
