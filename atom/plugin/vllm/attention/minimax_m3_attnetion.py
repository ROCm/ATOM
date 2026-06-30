# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 dense attention adapter backed by vLLM Attention.

This module keeps the ATOM ``MiniMaxM3Attention`` layer intact: qkv/o
projections, per-head QK norms, RoPE objects, and checkpoint weight names stay
owned by ``atom.models.minimax_m3``.  Only the dense attention backend is
replaced with vLLM's ``Attention`` implementation.
"""

from typing import Optional

import aiter
import torch
from aiter import dtypes
from torch import nn

from atom.config import get_current_atom_config
from atom.plugin.vllm.attention.backend import (
    MiniMaxM3SparseAttentionBackend,
    SparseMHAIndexerBackend,
)
from atom.plugin.vllm.attention.layer_common import (
    _register_vllm_static_forward_context,
)
from atom.utils import mark_spliting_op
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.attention import Attention as VllmAttention


def minimax_m3_sparse_attention_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    output_hidden_size: int,
) -> torch.Tensor:
    del positions, layer_name
    return qkv.new_empty((qkv.shape[0], output_hidden_size))


@mark_spliting_op(
    is_custom=True,
    gen_fake=minimax_m3_sparse_attention_fake,
    mutates_args=[],
)
def minimax_m3_sparse_attention(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    output_hidden_size: int,
) -> torch.Tensor:
    from vllm.forward_context import get_forward_context

    layer = get_forward_context().no_compile_layers[layer_name]
    output = qkv.new_empty((qkv.shape[0], output_hidden_size))
    return layer._forward_with_output(qkv, positions, output)


class MiniMaxM3SparseIndexerCache(nn.Module, AttentionLayerBase):
    """Key-only index cache owned by MiniMax-M3 sparse attention."""

    def __init__(
        self,
        *,
        layer_name: str,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> None:
        from vllm.v1.attention.backend import AttentionType
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        super().__init__()
        atom_config = get_current_atom_config()
        vllm_config = atom_config.plugin_config.vllm_config
        self.layer_name = layer_name
        self.prefix = layer_name
        self.attn_type = AttentionType.DECODER
        self.attn_backend = SparseMHAIndexerBackend
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.num_kv_heads = 1
        self.head_size = head_dim
        self.head_size_v = head_dim
        self.sliding_window = -1
        self.kv_cache = torch.tensor([])
        _register_vllm_static_forward_context(self)

    @property
    def impl(self):
        return self

    def get_attn_backend(self):
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


AttentionLayerBase.register(MiniMaxM3SparseIndexerCache)


class MiniMaxM3SparseAttentionForVllm(nn.Module, AttentionLayerBase):
    """MiniMax-M3 sparse attention backend for ATOM models under vLLM.

    This intentionally depends only on the generic ATOM vLLM attention stack
    under ``atom.plugin.vllm.attention``. Do not depend on model-local MiniMax-M3
    backend modules here: that model directory is not part of the long-term ATOM
    backend surface.
    """

    is_indexed_sparse_attention = True

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]] = None,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        use_mla: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        prefix: Optional[str] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        cache_config=None,
        quant_config=None,
        index_q_norm: Optional[nn.Module] = None,
        index_k_norm: Optional[nn.Module] = None,
        index_rotary_emb: Optional[nn.Module] = None,
        index_q_size: int = 0,
        index_head_dim: int = 0,
        topk: int = 0,
        init_blocks: int = 0,
        local_blocks: int = 0,
        skip_index_topk: bool = False,
        sparse_layer_ordinal: int = -1,
        impl_cls=None,
        **kwargs,
    ) -> None:
        super().__init__()
        del alibi_slopes, use_mla, quant_config, impl_cls, kwargs
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        atom_config = get_current_atom_config()
        if atom_config is None or atom_config.plugin_config is None:
            raise RuntimeError("atom_config with vLLM plugin_config is required")

        # ATOM's MiniMax-M3 sparse layer historically passes CacheConfig through
        # the kv_cache_dtype argument name used by atom.model_ops.base_attention.
        if cache_config is None and hasattr(kv_cache_dtype, "cache_dtype"):
            cache_config = kv_cache_dtype
        cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else kv_cache_dtype
        )
        self.layer_name = prefix if prefix is not None else f"M3_SPARSE_{layer_num}"
        self.attn_backend = MiniMaxM3SparseAttentionBackend
        self.kv_cache_dtype = cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            cache_dtype, atom_config.plugin_config.vllm_config.model_config
        )
        self.kv_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.kv_scale = torch.tensor(1.0, dtype=torch.float32)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_size = head_dim
        self.head_size_v = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.layer_num = layer_num
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.index_q_norm = index_q_norm
        self.index_k_norm = index_k_norm
        self.index_rotary_emb = (
            index_rotary_emb if index_rotary_emb is not None else rotary_emb
        )
        self.index_q_size = index_q_size
        self.index_head_dim = index_head_dim
        self.num_idx_heads = num_kv_heads
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.skip_index_topk = skip_index_topk
        self.sparse_layer_ordinal = sparse_layer_ordinal
        self._cached_topk: tuple | None = None
        self._cached_topk_key: tuple | None = None

        if self.head_dim != 128:
            raise ValueError("MiniMax-M3 sparse attention requires head_dim == 128.")
        if index_q_norm is None or index_k_norm is None:
            raise ValueError("MiniMax-M3 sparse attention requires index norms.")
        if index_head_dim <= 0 or index_q_size <= 0 or topk <= 0:
            raise ValueError(
                "MiniMax-M3 sparse attention requires index dimensions/topk."
            )

        self.index_cache_layer = MiniMaxM3SparseIndexerCache(
            layer_name=f"{self.layer_name}.index_cache",
            head_dim=index_head_dim,
            kv_cache_dtype="auto",
        )
        _register_vllm_static_forward_context(self)

    @property
    def impl(self):
        return self

    def get_attn_backend(self):
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    @staticmethod
    def _main_metadata():
        metadata = get_forward_context().attn_metadata
        return metadata

    def _metadata_for_layer(self):
        metadata = self._main_metadata()
        if not isinstance(metadata, dict):
            return metadata, metadata
        return metadata.get(self.layer_name), metadata.get(
            self.index_cache_layer.layer_name
        )

    def _ensure_fp8_scales(self, kv_cache: torch.Tensor):
        if self.kv_cache_dtype != "fp8":
            return None, None
        _kv, num_blocks, block_size, num_kv_heads, _head_dim = kv_cache.shape
        max_kv_tokens = num_blocks * block_size
        expected_shape = (num_kv_heads, max_kv_tokens)
        if (
            self.k_scale is None
            or self.v_scale is None
            or self.k_scale.shape != expected_shape
            or self.k_scale.device != kv_cache.device
        ):
            self.kv_scale = torch.zeros(
                2,
                num_kv_heads,
                max_kv_tokens,
                dtype=dtypes.fp32,
                device=kv_cache.device,
            )
            self.k_scale = self.kv_scale[0]
            self.v_scale = self.kv_scale[1]
        return self.k_scale, self.v_scale

    def _insert_qkv_and_index(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        main_metadata,
        index_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from atom.models.minimax_m3 import _minimax_m3_cos_sin_cache

        if self.kv_cache.numel() == 0 or self.index_cache_layer.kv_cache.numel() == 0:
            num_tokens = qkv.shape[0]
            return (
                qkv.new_zeros((num_tokens, self.q_size)),
                qkv.new_zeros((num_tokens, self.index_q_size)),
            )

        qkv = qkv.contiguous()
        num_tokens = qkv.shape[0]
        q_out = qkv.new_empty((num_tokens, self.q_size))
        index_q = qkv.new_empty((num_tokens, self.index_q_size))
        k_scale, v_scale = self._ensure_fp8_scales(self.kv_cache)
        k_cache, v_cache = self.kv_cache.unbind(0)
        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)
        kv_cache_dtype = self.kv_cache_dtype if self.kv_cache_dtype == "fp8" else "auto"

        aiter.fused_qknorm_idxrqknorm(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            _minimax_m3_cos_sin_cache(self.rotary_emb, qkv),
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            slot_mapping=main_metadata.slot_mapping,
            kv_cache_k=k_cache,
            kv_cache_v=v_cache,
            index_cache=self.index_cache_layer.kv_cache,
            block_size=k_cache.shape[1],
            q_out=q_out,
            index_q_out=index_q,
            index_slot_mapping=index_metadata.slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale if self.kv_cache_dtype == "fp8" else None,
            v_scale=v_scale if self.kv_cache_dtype == "fp8" else None,
            asm_layout=False,
        )
        return q_out, index_q

    def _topk_cache_key(self, phase: str, index_q: torch.Tensor, metadata) -> tuple:
        return (
            phase,
            tuple(index_q.shape),
            index_q.dtype,
            index_q.device,
            tuple(metadata.block_table.shape),
            tuple(metadata.seq_lens.shape),
            self.topk,
            self.init_blocks,
            self.local_blocks,
        )

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        index_q: torch.Tensor,
        output: torch.Tensor,
        main_metadata,
        index_metadata,
    ) -> torch.Tensor:
        from atom.model_ops.minimax_m3.index_topk import (
            minimax_m3_index_topk,
            minimax_m3_index_topk_decode,
        )
        from atom.model_ops.minimax_m3.sparse_attn import (
            minimax_m3_sparse_attn,
            minimax_m3_sparse_attn_decode,
        )

        q = query.view(-1, self.num_heads, self.head_dim)
        out = output.view(-1, self.num_heads, self.head_dim)
        kv_cache = self.kv_cache.permute(1, 0, 2, 3, 4)
        k_scale = self.k_scale if self.kv_cache_dtype == "fp8" else None
        v_scale = self.v_scale if self.kv_cache_dtype == "fp8" else None
        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            kv_cache = kv_cache.view(target_dtype)

        num_decode_tokens = getattr(main_metadata, "num_decode_tokens", 0)
        num_prefill_tokens = getattr(main_metadata, "num_prefill_tokens", 0)
        if num_decode_tokens > 0 and main_metadata.decode is not None:
            decode_md = main_metadata.decode
            index_decode_md = (
                index_metadata.decode if index_metadata is not None else decode_md
            )
            max_query_len = max(1, int(getattr(decode_md, "max_query_len", 1) or 1))
            key = self._topk_cache_key(
                "decode", index_q[:num_decode_tokens], index_decode_md
            )
            cached = (
                self._cached_topk
                if self.skip_index_topk and self._cached_topk_key == key
                else None
            )
            if cached is None:
                topk_idx = minimax_m3_index_topk_decode(
                    index_q[:num_decode_tokens].view(
                        -1, self.num_idx_heads, self.index_head_dim
                    ),
                    self.index_cache_layer.kv_cache,
                    index_decode_md.block_table,
                    index_decode_md.seq_lens,
                    getattr(index_metadata, "max_seq_len", main_metadata.max_seq_len),
                    self.topk,
                    self.init_blocks,
                    self.local_blocks,
                    self.num_kv_heads,
                    self.scale,
                    emit_sparse_block_table=False,
                    max_query_len=max_query_len,
                )
                if self.skip_index_topk:
                    self._cached_topk_key = key
                    self._cached_topk = topk_idx
            else:
                topk_idx = cached
            minimax_m3_sparse_attn_decode(
                q[:num_decode_tokens],
                kv_cache,
                topk_idx,
                decode_md.block_table,
                decode_md.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:num_decode_tokens],
                k_scale=k_scale,
                v_scale=v_scale,
            )

        if num_prefill_tokens > 0 and main_metadata.prefill is not None:
            start = num_decode_tokens
            stop = start + num_prefill_tokens
            prefill_md = main_metadata.prefill
            index_prefill_md = (
                index_metadata.prefill if index_metadata is not None else prefill_md
            )
            topk_idx = minimax_m3_index_topk(
                index_q[start:stop].view(-1, self.num_idx_heads, self.index_head_dim),
                self.index_cache_layer.kv_cache,
                index_prefill_md.block_table,
                index_prefill_md.cu_seqlens_q,
                index_prefill_md.seq_lens,
                index_prefill_md.context_lens,
                index_prefill_md.max_query_len,
                index_prefill_md.max_seq_len,
                self.topk,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                self.scale,
                emit_sparse_block_table=False,
            )
            minimax_m3_sparse_attn(
                q[start:stop],
                kv_cache,
                topk_idx,
                prefill_md.block_table,
                prefill_md.cu_seqlens_q,
                prefill_md.seq_lens,
                prefill_md.context_lens,
                prefill_md.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[start:stop],
                k_scale=k_scale,
                v_scale=v_scale,
            )
        return output

    def _forward_with_output(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        main_metadata, index_metadata = self._metadata_for_layer()
        num_tokens = qkv.shape[0]
        if output is None:
            output = qkv.new_empty((num_tokens, self.q_size))
        if main_metadata is None or positions is None:
            return output.fill_(0)
        actual_tokens = min(
            getattr(main_metadata, "num_actual_tokens", num_tokens), num_tokens
        )
        if actual_tokens < num_tokens:
            output[actual_tokens:].zero_()
        q_actual, index_q = self._insert_qkv_and_index(
            qkv[:actual_tokens],
            positions[:actual_tokens],
            main_metadata,
            index_metadata if index_metadata is not None else main_metadata,
        )
        output[:actual_tokens] = self._run_sparse_attention(
            q_actual,
            index_q,
            output[:actual_tokens],
            main_metadata,
            index_metadata if index_metadata is not None else main_metadata,
        )
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del query, key, value, q_scale, kwargs
        if qkv is None:
            raise ValueError("MiniMax-M3 sparse vLLM attention requires packed qkv.")
        if positions is None:
            raise ValueError("positions is required for MiniMax-M3 sparse attention.")
        return torch.ops.aiter.minimax_m3_sparse_attention(
            qkv,
            positions,
            self.layer_name,
            self.q_size,
        )


class MiniMaxM3DenseAttentionForVllm(nn.Module):
    """Drop-in dense attention backend for ATOM MiniMax-M3 under vLLM.

    The constructor mirrors ``atom.model_ops.base_attention.Attention`` for the
    arguments used by ``atom.models.minimax_m3.MiniMaxM3Attention``.  Forward
    accepts raw q/k/v plus the packed qkv tensor, applies MiniMax-M3 QK norm and
    partial RoPE, then delegates KV-cache update and attention dispatch to vLLM.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]] = None,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        use_mla: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        prefix: Optional[str] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        cache_config=None,
        quant_config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        del alibi_slopes, kv_cache_dtype, layer_num, use_mla, quant_config, kwargs

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm

        if cache_config is None:
            atom_config = get_current_atom_config()
            if atom_config is not None and atom_config.plugin_config is not None:
                cache_config = atom_config.plugin_config.vllm_cache_config

        # Projection quantization belongs to the outer ATOM MiniMaxM3Attention
        # module. vLLM Attention's quant_config is only for its KV-cache path, so
        # ignore the ATOM projection quant_config passed through this adapter.
        self.attn = VllmAttention(
            num_heads,
            head_dim,
            scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=prefix,
        )

    @property
    def layer_name(self):
        return self.attn.layer_name

    def get_attn_backend(self):
        return self.attn.get_attn_backend()

    def process_weights_after_loading(
        self, act_dtype: torch.dtype = torch.bfloat16
    ) -> None:
        process = getattr(self.attn, "process_weights_after_loading", None)
        if process is not None:
            process(act_dtype)

    def _qk_norm_rope(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from atom.models.minimax_m3 import _minimax_m3_cos_sin_cache

        qkv = qkv.contiguous()
        aiter.fused_qknorm_idxrqknorm(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            _minimax_m3_cos_sin_cache(self.rotary_emb, qkv),
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            num_index_heads=0,
        )
        return tuple(
            tensor.contiguous()
            for tensor in qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del query, key, value, q_scale, kwargs
        query, key, value = self._qk_norm_rope(qkv, positions)
        return self.attn(query, key, value)
