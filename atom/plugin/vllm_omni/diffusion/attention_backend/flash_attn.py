# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


class AiterFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "AITER_DIFFUSION_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AiterDiffusionFlashAttentionImpl"]:
        return AiterDiffusionFlashAttentionImpl


class AiterDiffusionFlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale

    @staticmethod
    def _unwrap_flash_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        # FA3 may return (out, lse), FA2 returns out
        return out[0] if isinstance(out, tuple) else out

    def _forward_varlen_masked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from aiter import flash_attn_varlen_func 
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            _pad_input,
            _unpad_input,
            _upad_input,
        )

        assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
        query_length = query.size(1)
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query, key, value, attention_mask, query_length, _unpad_input
        )

        out_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **{
                "causal": self.causal,
                "softmax_scale": self.softmax_scale,
            },
        )
        out_unpad = self._unwrap_flash_output(out_unpad)
        return _pad_input(out_unpad, indices_q, query.size(0), query_length)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        from aiter import flash_attn_func

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        out = flash_attn_func(
            query,
            key,
            value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return self._unwrap_flash_output(out)
