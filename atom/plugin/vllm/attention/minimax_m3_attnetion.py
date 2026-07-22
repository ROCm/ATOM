# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 attention adapters for ATOM vLLM plugin mode.

This module keeps the ATOM ``MiniMaxM3Attention`` layer intact: qkv/o
projections, per-head QK norms, RoPE objects, and checkpoint weight names stay
owned by ``atom.models.minimax_m3``. Sparse layers use the MiniMax-M3-specific
runtime below; dense layers use vLLM's Triton custom-op backend after applying
MiniMax-M3's q/k norm + RoPE transform.
"""

from typing import Optional

import aiter
import torch
from aiter import dtypes
from torch import nn

from atom.config import get_current_atom_config
from atom.model_ops.minimax_m3.sparse_attn import (
    ASM_PAGE_SIZE,
    PAGES_PER_SPARSE_BLOCK,
    SPARSE_BLOCK_SIZE,
)
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

_MINIMAX_M3_TOPK_CACHE_STATE: dict = {}


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

        block_size = vllm_config.cache_config.block_size
        if block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 sparse index block size must be {SPARSE_BLOCK_SIZE}."
            )

        return MLAAttentionSpec(
            block_size=block_size,
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
        del (
            alibi_slopes,
            use_mla,
            quant_config,
            index_rotary_emb,
            impl_cls,
            kwargs,
        )
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
        if cache_config is not None:
            block_size = getattr(cache_config, "block_size", SPARSE_BLOCK_SIZE)
            if block_size != SPARSE_BLOCK_SIZE:
                raise ValueError(
                    f"MiniMax-M3 sparse block size must be {SPARSE_BLOCK_SIZE}."
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
        self.index_q_size = index_q_size
        self.index_head_dim = index_head_dim
        self.num_idx_heads = num_kv_heads
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.skip_index_topk = skip_index_topk
        self.sparse_layer_ordinal = sparse_layer_ordinal

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
            kv_cache_dtype=(
                cache_dtype if str(cache_dtype).startswith("fp8") else "auto"
            ),
        )
        _register_vllm_static_forward_context(self)

    @property
    def impl(self):
        return self

    def get_attn_backend(self):
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        block_size = vllm_config.cache_config.block_size
        if block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 sparse block size must be {SPARSE_BLOCK_SIZE}."
            )

        return FullAttentionSpec(
            block_size=block_size,
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

    def _validate_bound_sparse_state(self, main_metadata, index_metadata) -> None:
        if main_metadata is None:
            raise ValueError("MiniMax-M3 sparse attention metadata is required.")
        if index_metadata is None:
            raise ValueError("MiniMax-M3 sparse index metadata is required.")
        if self.kv_cache.numel() == 0 or self.index_cache_layer.kv_cache.numel() == 0:
            # vLLM profiling calls can run before cache binding; the caller
            # handles this by returning zero outputs.
            return

        if self.kv_cache.ndim != 5:
            raise ValueError(
                "MiniMax-M3 sparse KV cache must have shape "
                "[2, num_blocks, block_size, num_kv_heads, head_dim]."
            )
        if self.kv_cache.shape[0] != 2:
            raise ValueError("MiniMax-M3 sparse KV cache must store K and V.")
        if self.kv_cache.shape[2] != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 sparse KV block size must be {SPARSE_BLOCK_SIZE}."
            )
        if self.kv_cache.shape[3] != self.num_kv_heads:
            raise ValueError("MiniMax-M3 sparse KV cache head count mismatch.")
        if self.kv_cache.shape[4] != self.head_dim:
            raise ValueError("MiniMax-M3 sparse KV cache head dim mismatch.")

        if self.index_cache_layer.kv_cache.ndim != 3:
            raise ValueError(
                "MiniMax-M3 sparse index cache must have shape "
                "[num_blocks, block_size, index_head_dim]."
            )
        if self.index_cache_layer.kv_cache.shape[1] != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 index cache block size must be {SPARSE_BLOCK_SIZE}."
            )
        if self.index_cache_layer.kv_cache.shape[2] != self.index_head_dim:
            raise ValueError("MiniMax-M3 index cache head dim mismatch.")

    def _ensure_fp8_scales(self, kv_cache: torch.Tensor):
        if self.kv_cache_dtype != "fp8":
            return None, None
        _kv, num_blocks, block_size, num_kv_heads, _head_dim = kv_cache.shape
        expected_shape = (num_blocks, num_kv_heads, block_size)
        if (
            self.k_scale is None
            or self.v_scale is None
            or self.k_scale.shape != expected_shape
            or self.k_scale.device != kv_cache.device
        ):
            self.kv_scale = torch.zeros(
                2,
                num_blocks,
                num_kv_heads,
                block_size,
                dtype=dtypes.fp32,
                device=kv_cache.device,
            )
            self.k_scale = self.kv_scale[0]
            self.v_scale = self.kv_scale[1]
        return self.k_scale, self.v_scale

    def _page16_shuffle_cache_for_sparse_kernel(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, object, object]:
        _kv, num_blocks, block_size, num_kv_heads, head_dim = self.kv_cache.shape
        if block_size != SPARSE_BLOCK_SIZE:
            raise ValueError("MiniMax-M3 sparse cache must use page size 128.")
        k_cache, v_cache = self.kv_cache.unbind(0)
        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)
        x = 16 // k_cache.element_size()
        num_phys16 = num_blocks * PAGES_PER_SPARSE_BLOCK
        k_cache = k_cache.view(
            num_phys16,
            num_kv_heads,
            head_dim // x,
            ASM_PAGE_SIZE,
            x,
        )
        v_cache = v_cache.view(
            num_phys16,
            num_kv_heads,
            ASM_PAGE_SIZE // x,
            head_dim,
            x,
        )
        if self.kv_cache_dtype == "fp8":
            k_scale = self.k_scale.view(num_phys16, num_kv_heads, ASM_PAGE_SIZE)
            v_scale = self.v_scale.view(num_phys16, num_kv_heads, ASM_PAGE_SIZE)
        else:
            k_scale = v_scale = None
        return k_cache, v_cache, k_scale, v_scale

    def _insert_qkv_and_index(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        main_metadata,
        index_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object, object]:
        from atom.models.minimax_m3 import _minimax_m3_cos_sin_cache

        if self.kv_cache.numel() == 0 or self.index_cache_layer.kv_cache.numel() == 0:
            num_tokens = qkv.shape[0]
            return (
                qkv.new_zeros((num_tokens, self.q_size)),
                qkv.new_zeros((num_tokens, self.index_q_size)),
                self.kv_cache,
                self.kv_cache,
                None,
                None,
            )

        qkv = qkv.contiguous()
        num_tokens = qkv.shape[0]
        q_out = qkv.new_empty((num_tokens, self.q_size))
        index_q = qkv.new_empty((num_tokens, self.index_q_size))
        self._ensure_fp8_scales(self.kv_cache)
        k_cache, v_cache, k_scale, v_scale = (
            self._page16_shuffle_cache_for_sparse_kernel()
        )
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
            block_size=k_cache.shape[3],
            q_out=q_out,
            index_q_out=index_q,
            index_slot_mapping=index_metadata.slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale if self.kv_cache_dtype == "fp8" else None,
            v_scale=v_scale if self.kv_cache_dtype == "fp8" else None,
            asm_layout=True,
        )
        return q_out, index_q, k_cache, v_cache, k_scale, v_scale

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

    @staticmethod
    def _topk_cache_state():
        return _MINIMAX_M3_TOPK_CACHE_STATE

    def _load_cached_topk(self, key: tuple):
        if not self.skip_index_topk:
            return None
        phase = key[0]
        entry = self._topk_cache_state().get(phase)
        if entry is not None and entry.get("key") == key:
            return entry["value"]
        raise RuntimeError("MiniMax-M3 topk cache miss on a skip-index layer")
        return None

    def _store_cached_topk(self, key: tuple, topk_idx) -> None:
        phase = key[0]
        self._topk_cache_state()[phase] = {
            "key": key,
            "value": topk_idx,
            "layer_num": self.layer_num,
            "sparse_layer_ordinal": self.sparse_layer_ordinal,
        }

    def _decode_topk(
        self,
        index_q: torch.Tensor,
        main_metadata,
        index_metadata,
    ):
        from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

        num_decode_tokens = main_metadata.num_decode_tokens
        decode_md = main_metadata.decode
        index_decode_md = (
            index_metadata.decode if index_metadata is not None else decode_md
        )
        max_query_len = max(1, int(getattr(decode_md, "max_query_len", 1) or 1))
        key = self._topk_cache_key(
            "decode", index_q[:num_decode_tokens], index_decode_md
        )
        cached = self._load_cached_topk(key)
        if cached is not None:
            return cached
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
            emit_sparse_block_table=True,
            max_query_len=max_query_len,
        )
        self._store_cached_topk(key, topk_idx)
        return topk_idx

    def _prefill_topk(
        self,
        index_q: torch.Tensor,
        start: int,
        stop: int,
        main_metadata,
        index_metadata,
    ):
        from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk

        prefill_md = main_metadata.prefill
        index_prefill_md = (
            index_metadata.prefill if index_metadata is not None else prefill_md
        )
        key = self._topk_cache_key("prefill", index_q[start:stop], index_prefill_md)
        cached = self._load_cached_topk(key)
        if cached is not None:
            return cached
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
            emit_sparse_block_table=True,
        )
        self._store_cached_topk(key, topk_idx)
        return topk_idx

    def _run_decode_sparse_attention(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale,
        v_scale,
        main_metadata,
        index_metadata,
    ) -> None:
        from atom.model_ops.minimax_m3.sparse_attn import (
            minimax_m3_sparse_attn_decode_asm,
        )

        num_decode_tokens = getattr(main_metadata, "num_decode_tokens", 0)
        if num_decode_tokens <= 0 or main_metadata.decode is None:
            return
        topk_idx, sparse_bt, sparse_ctx = self._decode_topk(
            index_q, main_metadata, index_metadata
        )
        decode_md = main_metadata.decode
        minimax_m3_sparse_attn_decode_asm(
            q[:num_decode_tokens],
            k_cache,
            v_cache,
            topk_idx,
            decode_md.block_table,
            decode_md.seq_lens,
            self.num_kv_heads,
            self.scale,
            out[:num_decode_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
        )

    def _run_prefill_sparse_attention(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale,
        v_scale,
        main_metadata,
        index_metadata,
    ) -> None:
        from atom.model_ops.minimax_m3.sparse_attn import (
            minimax_m3_sparse_attn_prefill_asm,
        )

        num_decode_tokens = getattr(main_metadata, "num_decode_tokens", 0)
        num_prefill_tokens = getattr(main_metadata, "num_prefill_tokens", 0)
        if num_prefill_tokens <= 0 or main_metadata.prefill is None:
            return
        start = num_decode_tokens
        stop = start + num_prefill_tokens
        topk_idx, sparse_bt, sparse_ctx = self._prefill_topk(
            index_q, start, stop, main_metadata, index_metadata
        )
        prefill_md = main_metadata.prefill
        minimax_m3_sparse_attn_prefill_asm(
            q[start:stop],
            k_cache,
            v_cache,
            topk_idx,
            prefill_md.block_table,
            None,
            None,
            prefill_md.qo_indptr,
            self.num_kv_heads,
            self.scale,
            out[start:stop],
            k_scale=k_scale,
            v_scale=v_scale,
            cu_seqlens_q=prefill_md.cu_seqlens_q,
            prefix_lens=prefill_md.context_lens,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
        )

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        index_q: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale,
        v_scale,
        main_metadata,
        index_metadata,
    ) -> torch.Tensor:
        q = query.view(-1, self.num_heads, self.head_dim)
        out = output.view(-1, self.num_heads, self.head_dim)
        self._run_decode_sparse_attention(
            q,
            index_q,
            out,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            main_metadata,
            index_metadata,
        )
        self._run_prefill_sparse_attention(
            q,
            index_q,
            out,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            main_metadata,
            index_metadata,
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
        index_metadata = index_metadata if index_metadata is not None else main_metadata
        self._validate_bound_sparse_state(main_metadata, index_metadata)
        if self.kv_cache.numel() == 0 or self.index_cache_layer.kv_cache.numel() == 0:
            return output.fill_(0)
        q_actual, index_q, k_cache, v_cache, k_scale, v_scale = (
            self._insert_qkv_and_index(
                qkv[:actual_tokens],
                positions[:actual_tokens],
                main_metadata,
                index_metadata,
            )
        )
        output[:actual_tokens] = self._run_sparse_attention(
            q_actual,
            index_q,
            output[:actual_tokens],
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            main_metadata,
            index_metadata,
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


class MiniMaxM3DenseAttentionForVllm(nn.Module, AttentionLayerBase):
    """MiniMax-M3 dense attention using vLLM's Triton backend contract."""

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
        del use_mla, cache_config, quant_config, kwargs
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionBackend,
            TritonAttentionImpl,
        )

        atom_config = get_current_atom_config()
        if atom_config is None or atom_config.plugin_config is None:
            raise RuntimeError("atom_config with vLLM plugin_config is required")
        vllm_config = atom_config.plugin_config.vllm_config
        cache_config = atom_config.plugin_config.vllm_cache_config
        cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else kv_cache_dtype
        )
        self.layer_name = prefix if prefix is not None else f"M3_DENSE_{layer_num}"
        self.attn_type = AttentionType.DECODER
        self.attn_backend = TritonAttentionBackend
        self.kv_cache_dtype = cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            cache_dtype, vllm_config.model_config
        )
        self.calculate_kv_scales = (
            cache_config.calculate_kv_scales if cache_config is not None else False
        )
        self.quant_config = None
        self.kv_cache = torch.tensor([])
        self.has_sink = False
        self.dtype = torch.get_default_dtype()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_size = head_dim
        self.head_size_v = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.impl = TritonAttentionImpl(
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            alibi_slopes,
            None,  # sliding_window
            cache_dtype,
            None,  # logits_soft_cap
            self.attn_type,
            None,  # kv_sharing_target_layer_name
        )
        from vllm.model_executor.layers.attention.attention import _init_kv_cache_quant

        _init_kv_cache_quant(self, None, self.layer_name)
        _register_vllm_static_forward_context(self)

    @property
    def layer_name(self):
        return self._layer_name

    @layer_name.setter
    def layer_name(self, value):
        self._layer_name = value

    def get_attn_backend(self):
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode

        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def process_weights_after_loading(
        self, act_dtype: torch.dtype = torch.bfloat16
    ) -> None:
        from vllm.model_executor.layers.attention.attention import (
            set_default_quant_scales,
        )

        self.impl.process_weights_after_loading(act_dtype)
        set_default_quant_scales(self, register_buffer=False)

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
        if qkv is None:
            raise ValueError("MiniMax-M3 dense vLLM attention requires packed qkv.")
        if positions is None:
            raise ValueError("positions is required for MiniMax-M3 dense attention.")
        query, key, value = self._qk_norm_rope(qkv, positions)
        if self.calculate_kv_scales and key is not None and value is not None:
            from vllm.model_executor.layers.attention.attention import (
                _encode_layer_name,
            )

            torch.ops.vllm.maybe_calc_kv_scales(
                query, key, value, _encode_layer_name(self.layer_name)
            )
            self.calculate_kv_scales = False

        output_shape = torch.Size((query.shape[0], self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=query.dtype, device=query.device)
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size_v)
        output = output.view(-1, self.num_heads, self.head_size_v)

        from vllm.model_executor.layers.attention.attention import _encode_layer_name

        encoded = _encode_layer_name(self.layer_name)
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(key, value, encoded)
        torch.ops.vllm.unified_attention_with_output(
            query,
            key,
            value,
            output,
            encoded,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        return output.view(-1, self.num_heads * self.head_size_v)
