# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlashAttention backend for MLA prefill (ROCm).

This backend calls ``aiter.flash_attn_varlen_func`` directly, which natively
supports different q/k and v head dims (qk headdim 192, v headdim 128) without
padding V, and dispatches to the fast ``aiter::fmha_fwd_`` kernel on
gfx942/gfx950 (fp16/bf16).

With ``VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1`` on gfx950, attention instead runs
AITER's FlyDSL FP8 kernel on per-tensor dynamically quantized Q/K/V.
"""

import os
from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)


def _quant_fp8_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x8, scale = ops.scaled_fp8_quant(x.reshape(-1, x.shape[-1]))
    return x8.view(x.shape), scale


class AiterFlashAttnPrefillBackend(MLAPrefillBackend):
    """AITER FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_mi3xx

        return on_mi3xx()

    @classmethod
    def is_available(cls) -> bool:
        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.is_enabled()

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.fp8_attn_func = None
        if os.environ.get("VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8", "0") == "1":
            from aiter.ops.flydsl import (
                flydsl_flash_attn_fp8_func,
                flydsl_flash_attn_fp8_supported,
            )

            device = torch.device("cuda", torch.cuda.current_device())
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            if flydsl_flash_attn_fp8_supported(
                device, num_heads, num_heads, qk_head_dim, v_head_dim
            ):
                self.fp8_attn_func = flydsl_flash_attn_fp8_func
                logger.info_once("MLA prefill uses FlyDSL FP8 attention")
            else:
                logger.warning_once(
                    "VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1 but FlyDSL FP8 attention "
                    "does not support %s heads, qk_dim=%d, v_dim=%d on %s; "
                    "using BF16 attention.",
                    num_heads,
                    qk_head_dim,
                    v_head_dim,
                    device,
                )

    def _fp8_attn(self, q, k, v, **kwargs):
        q8, q_descale = _quant_fp8_per_tensor(q)
        k8, k_descale = _quant_fp8_per_tensor(k)
        v8, v_descale = _quant_fp8_per_tensor(v)
        assert self.fp8_attn_func is not None
        return self.fp8_attn_func(
            q8,
            k8,
            v8,
            softmax_scale=self.scale,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            **kwargs,
        )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, (
            "AiterFlashAttnPrefillBackend does not support fused quantized output."
        )
        if self.fp8_attn_func is not None:
            return self._fp8_attn(
                q,
                k,
                v,
                cu_seqlens_q=self._prefill_metadata.query_start_loc,
                cu_seqlens_kv=self._prefill_metadata.query_start_loc,
                max_seqlen_q=self._prefill_metadata.max_query_len,
                max_seqlen_kv=self._prefill_metadata.max_query_len,
                cross_seqlen=False,
                causal=True,
                return_lse=return_softmax_lse,
            )
        result = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_lse=return_softmax_lse,
            out=out,
        )

        # aiter returns the bare output tensor when return_lse is False, and
        # (out, softmax_lse) when it is True.
        if return_softmax_lse:
            return result[0], result[1]
        return result

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out is None, (
            "AiterFlashAttnPrefillBackend does not report supports_out(), so it "
            "is never given a context-chunk `out` to write into."
        )
        if self.fp8_attn_func is not None:
            return self._fp8_attn(
                q,
                k,
                v,
                cu_seqlens_q=chunk.query_start_loc,
                cu_seqlens_kv=chunk.cu_seq_lens,
                max_seqlen_q=chunk.max_query_len,
                max_seqlen_kv=chunk.max_seq_len,
                cross_seqlen=True,
                causal=False,
                return_lse=True,
            )
        out, lse = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
            return_lse=True,
        )
        return out, lse
