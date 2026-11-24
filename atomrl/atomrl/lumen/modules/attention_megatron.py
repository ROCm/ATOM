# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Megatron-compatible attention module using Lumen kernels.

This module provides :class:`LumenDotProductAttention`, an ``nn.Module``
for Megatron-Core compatible dot-product attention.

It bridges the Megatron ``[s, b, h, d]`` tensor layout to the Transformer
Light ``[b, s, h, d]`` layout and delegates the actual Q·K^T·V computation
(both forward **and** backward) to the TL public API:

* :func:`~atomrl.lumen.ops.attention.attention`
  – dispatches to AITER or Triton backends
* :func:`~atomrl.lumen.ops.attention.attention_fp8_quant`
  – dispatches to Triton FP8 (blockwise / MXFP8) backends

Each of these functions internally uses a ``torch.autograd.Function`` with
explicit ``forward()`` and ``backward()`` that call the optimised TL / AITER
kernels, so **no** part of the attention forward or backward falls back to
generic PyTorch ops.

Prerequisites:
    pip install megatron-lm   # or: pip install git+https://github.com/ROCm/Megatron-LM.git
"""

import math

import torch
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from megatron.training import get_args

from atomrl.lumen.modules.parallel_linear import _use_sdma_from_args
from atomrl.lumen.ops.attention.attention import (
    attention,
    attention_fp8_quant,
)

__all__ = [
    "LumenDotProductAttention",
]

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _sbhd_to_bshd(t: torch.Tensor, materialize: bool = True) -> torch.Tensor:
    """Megatron [s, b, h, d] -> TL [b, s, h, d]

    The permuted view keeps ``stride(-1) == 1``, which is the only layout
    constraint the CK flash-attention entry point imposes -- it reads the batch,
    sequence and head strides from the tensor. Materialising the permute is
    therefore only needed for backends that assert contiguity (Triton FP8) or
    that reinterpret the buffer directly (context-parallel all-to-all).
    """
    view = t.permute(1, 0, 2, 3)
    return view.contiguous() if materialize else view


def _bshd_to_sbhd(t: torch.Tensor) -> torch.Tensor:
    """TL [b, s, h, d] -> Megatron [s, b, h, d]

    Backends that honour ``seq_major_out`` already wrote the values in Megatron's
    order, which leaves the permute alone contiguous and the copy a no-op.
    """
    return t.permute(1, 0, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------


class LumenDotProductAttention(MegatronModule):
    """Lumen dot-product attention for Megatron-Core.

    Converts the Megatron ``[s, b, h, d]`` layout to ``[b, s, h, d]``,
    calls the Lumen :func:`attention` /
    :func:`attention_fp8_quant` API (which use ``torch.autograd.Function``
    with explicit forward **and** backward kernels), and converts the
    output back.

    Backends (``--lumen-attn-backend``):
        User-facing choices: ``auto``, ``triton``, ``csrc``, ``asm``.
        Combined with ``--lumen-fp8-attn`` (``none``/``dpa``/``mha``), these
        resolve to concrete kernels: ``aiter_csrc``, ``aiter_triton``,
        ``aiter_triton_fp8``, ``aiter_csrc_fp8``, ``aiter_asm_fp8``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        **kwargs,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type

        projection_size = config.kv_channels * config.num_attention_heads
        world_size = (
            config.tensor_model_parallel_size
            if hasattr(config, "tensor_model_parallel_size")
            else 1
        )
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(
            projection_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            config.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            config.num_query_groups, world_size
        )

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        self.dropout_p = (
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

        self.cp_size = config.context_parallel_size
        self.cp_comm_type = cp_comm_type

        args = get_args()
        self.backend = getattr(args, "lumen_attn_backend", "aiter_csrc")
        self.fp8_quant_type = getattr(args, "lumen_fp8_quant_type", "blockwise")

        fp8_attn = getattr(args, "lumen_fp8_attn", "none")
        self.fp8_dpa = fp8_attn in ("dpa", "mha")
        self.fp8_mha = fp8_attn == "mha"

        # Shared scale manager for FP8 MHA (set externally by enable_fp8_for_parallel_linear)
        self.scale_manager = None

        # Fine-grained MXFP8 block configuration (per-dimension)
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
            query:  [sq, b, np, hn]
            key:    [sk, b, ng, hn]
            value:  [sk, b, ng, hn]
        Returns:
            context: [sq, b, hp]   (hp = np * hn)
        """
        materialize = self.backend != "aiter_csrc" or self.cp_size > 1
        q = _sbhd_to_bshd(query, materialize)
        k = _sbhd_to_bshd(key, materialize)
        v = _sbhd_to_bshd(value, materialize)

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
                # The result feeds Megatron's [s, b, ...] residual stream, so
                # ask the kernel to write it that way and skip the transpose.
                seq_major_out=cp_param_bundle is None,
            )

        # out: [b, sq, np, hn] -> [sq, b, np*hn]
        context = _bshd_to_sbhd(out)
        return context.reshape(context.shape[0], context.shape[1], -1)
