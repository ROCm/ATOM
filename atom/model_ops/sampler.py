# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from aiter import mixed_sample, mixed_sample_outer_exponential
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.topk import topk
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(
        self,
        logits: torch.Tensor,  # (token_num, vocab_size)
        temperatures: torch.Tensor,  # (token_num,)
    ) -> torch.Tensor:  # (token_num,)
        sampled_tokens = torch.empty(
            logits.size(0), dtype=torch.int, device=logits.device
        )
        exponential = (
            torch.empty((1, logits.shape[-1]), dtype=torch.float, device=logits.device)
            .exponential_(1)
            .expand(*logits.shape)
        )
        mixed_sample_outer_exponential(
            sampled_tokens, logits, exponential, temperatures, eps=self.eps
        )
        return sampled_tokens
        logits = logits.float()
        return torch.where(
            temperatures == 0, self.greedy_sample(logits), self.random_sample(logits)
        ).to(torch.int)

    def greedy_sample(
        self, logits: torch.Tensor  # (token_num, vocab_size)
    ) -> torch.Tensor:  # (token_num,)
        _, sampled_tokens = topk(logits, 1)
        return sampled_tokens.view(-1)

    def random_sample(
        self, logits: torch.Tensor  # (token_num, vocab_size)
    ) -> torch.Tensor:  # (token_num,)
        probs = softmax(logits)
        logits = probs.div_(torch.empty_like(probs).exponential_(1) + self.eps)
        _, sampled_tokens = topk(logits, 1)
        return sampled_tokens.view(-1)
