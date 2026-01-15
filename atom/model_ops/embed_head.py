# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.dist.parallel_state import get_tp_group
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.utils.forward_context import ForwardContext, get_forward_context
from atom.plugin import is_plugin_mode


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
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
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
        if self.tp_size > 1:
            mask = torch.logical_and(x >= self.vocab_start_idx, x < self.vocab_end_idx)
            # mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y.masked_fill_(~mask.unsqueeze(1), 0)
            y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        return y


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
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        if not is_plugin_mode():
            forward_context: ForwardContext = get_forward_context()
            context = forward_context.context
            attn_metadata = forward_context.attn_metadata
            # context = get_context()
            if context.is_prefill and not context.is_draft:
                last_indices = attn_metadata.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
        logits = tgemm.mm(x, self.weight, self.bias)
        if self.tp_size > 1:
            logits = tensor_model_parallel_all_gather(logits)
            # all_logits = (
            #     [torch.empty_like(logits) for _ in range(self.tp_size)]
            #     if self.tp_rank == 0
            #     else None
            # )
            # dist.gather(logits, all_logits, 0)
            # logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
