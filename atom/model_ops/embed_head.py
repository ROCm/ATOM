# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.utils import atom_parameter
from atom.model_ops.linear import (
    _fp8_per_tensor_linear_triton,
    _get_triton_fp8_gemm,
)
from atom.plugin import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import ForwardContext, get_forward_context
from aiter.tuned_gemm import tgemm


@triton.jit
def _masked_embedding_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    vocab_start_idx,
    vocab_end_idx,
    stride_w_row,
    stride_out_row,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    if pid_row >= N:
        return

    token_id = tl.load(x_ptr + pid_row)
    in_range = (token_id >= vocab_start_idx) & (token_id < vocab_end_idx)
    local_idx = token_id - vocab_start_idx

    col_start = pid_col * BLOCK_D
    cols = col_start + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    emb = tl.load(
        weight_ptr + local_idx * stride_w_row + cols,
        mask=in_range & col_mask,
        other=0.0,
    )

    tl.store(out_ptr + pid_row * stride_out_row + cols, emb, mask=col_mask)


def _masked_embedding_launcher(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    N = x.numel()
    D = weight.shape[1]
    BLOCK_D = 1024
    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N, triton.cdiv(D, BLOCK_D))
    _masked_embedding_kernel[grid](
        x,
        weight,
        out,
        vocab_start_idx,
        vocab_end_idx,
        weight.stride(0),
        out.stride(0),
        N,
        D,
        BLOCK_D=BLOCK_D,
    )
    return out


def _masked_embedding_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    return torch.empty(
        x.numel(),
        weight.shape[1],
        dtype=weight.dtype,
        device=weight.device,
    )


@torch_compile_guard(gen_fake=_masked_embedding_fake)
def masked_embedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    return _masked_embedding_launcher(x, weight, vocab_start_idx, vocab_end_idx)


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = atom_parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim),
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # Torch compile will make logical_and, mask, embedding in a fused triton kernel, but make accuracy issue in MTP.
        if self.tp_size > 1:
            y = masked_embedding(
                x, self.weight, self.vocab_start_idx, self.vocab_end_idx
            )
            y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        else:
            y = F.embedding(x, self.weight)
        return y
        # if self.tp_size > 1:
        #     mask = torch.logical_and(x >= self.vocab_start_idx, x < self.vocab_end_idx)
        #     # mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        #     x = mask * (x - self.vocab_start_idx)
        # y = F.embedding(x, self.weight)
        # if self.tp_size > 1:
        #     y.masked_fill_(~mask.unsqueeze(1), 0)
        #     y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        # return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = atom_parameter(
                torch.empty(self.num_embeddings_per_partition),
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        self._fp8_lm_head_weight = None
        self._fp8_lm_head_scale = None
        self._fp8_lm_head_src_ptr = None

    def _get_fp8_lm_head_weight(self):
        src_ptr = self.weight.data_ptr()
        if (
            self._fp8_lm_head_weight is not None
            and self._fp8_lm_head_scale is not None
            and self._fp8_lm_head_src_ptr == src_ptr
        ):
            return self._fp8_lm_head_weight, self._fp8_lm_head_scale

        weight = self.weight.detach()
        num_rows, hidden_size = weight.shape
        weight_q = torch.empty_like(weight, dtype=torch.uint8)
        weight_scale = torch.empty(
            (num_rows, 1), dtype=torch.float32, device=weight.device
        )

        # Chunking avoids a transient full FP32 copy of the 131k x 4096 lm_head.
        chunk_rows = 4096
        for start in range(0, num_rows, chunk_rows):
            end = min(start + chunk_rows, num_rows)
            block = weight[start:end].float()
            scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
            weight_scale[start:end].copy_(scale)
            weight_q[start:end].copy_(
                (block / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            )

        self._fp8_lm_head_weight = weight_q
        self._fp8_lm_head_scale = weight_scale
        self._fp8_lm_head_src_ptr = src_ptr
        return weight_q, weight_scale

    def _use_fp8_lm_head(self, x: torch.Tensor) -> bool:
        """Whether this forward should route through the aiter triton gemm_a8w8
        FP8 lm_head path. Pure capability check on the inputs and the env var
        - no arch detection. The decision to actually quantize is deferred to
        _get_fp8_lm_head_weight() (lazy + cached)."""
        return (
            envs.ATOM_LM_HEAD_FP8
            and x.is_cuda
            and x.dim() == 2
            and self.weight.dim() == 2
            and self.weight.dtype == torch.bfloat16
        )

    def forward(self, x: torch.Tensor):
        if not is_plugin_mode():
            forward_context: ForwardContext = get_forward_context()
            context = forward_context.context
            attn_metadata = forward_context.attn_metadata
            # context = get_context()
            if context.is_prefill and not context.is_draft:
                last_indices = attn_metadata.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
        if self._use_fp8_lm_head(x):
            triton_gemm = _get_triton_fp8_gemm()
            if triton_gemm is None:
                logits = tgemm.mm(x, self.weight, self.bias)
            else:
                weight_q, weight_scale = self._get_fp8_lm_head_weight()
                logits = _fp8_per_tensor_linear_triton(
                    triton_gemm,
                    x,
                    weight_q,
                    weight_scale,
                    self.bias,
                    x.dtype,
                    None,
                )
        else:
            logits = tgemm.mm(x, self.weight, self.bias)
        if self.tp_size > 1:
            use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER
            logits = tensor_model_parallel_all_gather(logits, use_custom=use_custom)
            # all_logits = (
            #     [torch.empty_like(logits) for _ in range(self.tp_size)]
            #     if self.tp_rank == 0
            #     else None
            # )
            # dist.gather(logits, all_logits, 0)
            # logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
