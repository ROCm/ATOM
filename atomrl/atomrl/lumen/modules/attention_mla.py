# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Multi-Latent Attention (MLA) module using Lumen kernels.

MLA uses *different* head dimensions for K and V:
  - K head dim = ``kv_channels + qk_rope_head_dim``
  - V head dim = ``kv_channels``

Delegates the actual QKV computation to Lumen's attention ops.
"""

import math

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from megatron.training import get_args

from atomrl.lumen.modules.parallel_linear import _use_sdma_from_args
from atomrl.lumen.ops.attention.attention import attention, attention_fp8_quant

__all__ = ["LumenDotProductAttentionMLA"]


def _sbhd_to_bshd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(1, 0, 2, 3).contiguous()


def _bshd_to_sbhd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(1, 0, 2, 3).contiguous()


class LumenDotProductAttentionMLA(MegatronModule):
    """Dot-product attention with Multi-Latent Attention (MLA) support.

    Handles the case where K and V have different head dimensions, as
    required by DeepSeek-V2 / V3 style models.  When K and V head dims
    are equal, this degenerates to standard MHA/GQA.

    Handles MLA-style attention for Megatron-Core layer specs.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        world_size = getattr(config, "tensor_model_parallel_size", 1)
        self.num_attention_heads_per_partition = divide(
            config.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            config.num_query_groups, world_size
        )

        if k_channels is not None and v_channels is not None:
            self.k_head_dim = k_channels
            self.v_head_dim = v_channels
        else:
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
            self.k_head_dim = config.kv_channels + qk_rope_head_dim
            self.v_head_dim = config.kv_channels

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.k_head_dim)
        else:
            self.softmax_scale = softmax_scale

        if getattr(config, "apply_query_key_layer_scaling", False):
            self.softmax_scale /= self.layer_number

        self.dropout_p = (
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

        self.cp_size = getattr(config, "context_parallel_size", 1)
        self.cp_comm_type = cp_comm_type

        args = get_args()
        self.backend = getattr(args, "lumen_attn_backend", "aiter_csrc")
        self.fp8_quant_type = getattr(args, "lumen_fp8_quant_type", "blockwise")

        fp8_attn = getattr(args, "lumen_fp8_attn", "none")
        self.fp8_dpa = fp8_attn in ("dpa", "mha")
        self.fp8_mha = fp8_attn == "mha"

        self.scale_manager = None

        self.block_m_fwd = getattr(args, "mxfp8_block_m_fwd", 128)
        self.block_n_fwd = getattr(args, "mxfp8_block_n_fwd", 128)
        self.block_m_dq_bwd = getattr(args, "mxfp8_block_m_dq_bwd", 128)
        self.block_n_dq_bwd = getattr(args, "mxfp8_block_n_dq_bwd", 128)
        self.block_m_dkv_bwd = getattr(args, "mxfp8_block_m_dkv_bwd", 128)
        self.block_n_dkv_bwd = getattr(args, "mxfp8_block_n_dkv_bwd", 128)
        self.mxfp8_quant_block_size = getattr(args, "mxfp8_quant_block_size", 128)
        self.grad_quant_type = getattr(args, "grad_quant_type", None)

    @property
    def _is_fp8_backend(self):
        return self.backend in ("aiter_triton_fp8", "aiter_csrc_fp8", "aiter_asm_fp8")

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """
        Args:
            query:  [sq, b, np, k_head_dim]
            key:    [sk, b, ng, k_head_dim]
            value:  [sk, b, ng, v_head_dim]
        Returns:
            context: [sq, b, np * v_head_dim]
        """
        q = _sbhd_to_bshd(query)
        k = _sbhd_to_bshd(key)
        v = _sbhd_to_bshd(value)

        if self.k_head_dim != self.v_head_dim:
            pad_size = self.k_head_dim - self.v_head_dim
            v = F.pad(v, (0, pad_size))

        causal = self.attn_mask_type == AttnMaskType.causal
        dropout_p = self.dropout_p if self.training else 0.0

        cp_param_bundle = None
        if self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            cp_comm_type = self.cp_comm_type or getattr(
                get_args(), "lumen_cp_comm_type", "a2a"
            )
            cp_param_bundle = {
                "cp_group": cp_group,
                "cp_comm_type": cp_comm_type,
                "use_sdma": _use_sdma_from_args(),
            }

        if self._is_fp8_backend:
            out = attention_fp8_quant(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
                backend_type=self.backend,
                quant_type=self.fp8_quant_type,
                cp_param_bundle=cp_param_bundle,
                block_m_fwd=self.block_m_fwd,
                block_n_fwd=self.block_n_fwd,
                block_m_dq_bwd=self.block_m_dq_bwd,
                block_n_dq_bwd=self.block_n_dq_bwd,
                block_m_dkv_bwd=self.block_m_dkv_bwd,
                block_n_dkv_bwd=self.block_n_dkv_bwd,
                quant_block_size=self.mxfp8_quant_block_size,
                grad_quant_type=self.grad_quant_type,
                fp8_mha=self.fp8_mha,
                scale_manager=self.scale_manager,
            )
        else:
            out = attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
                backend_type=self.backend,
                cp_param_bundle=cp_param_bundle,
                grad_quant_type=self.grad_quant_type,
            )

        if self.k_head_dim != self.v_head_dim:
            out = out[..., : self.v_head_dim]

        context = _bshd_to_sbhd(out)
        return context.reshape(context.shape[0], context.shape[1], -1)
