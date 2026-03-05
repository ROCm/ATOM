# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache

import torch
from aiter import mixed_sample_outer_exponential
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.topk import topk
from torch import nn


@lru_cache(maxsize=1)
def get_per_token_exponential(vocab_size: int, device) -> torch.Tensor:
    """Returns a tensor of shape (1, vocab_size) filled with exponential random values.
    This is key to deterministic inference, as it ensures that the same random values are used for each token across different runs.
    """
    return torch.empty((1, vocab_size), dtype=torch.float, device=device).exponential_(
        1
    )


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(
        self,
        logits: torch.Tensor,  # (token_num, vocab_size)
        temperatures: torch.Tensor,  # (token_num,)
    ) -> torch.Tensor:  # (token_num,)
        token_num, vocab_size = logits.shape
        sampled_tokens = torch.empty(token_num, dtype=torch.int, device=logits.device)
        exponential = get_per_token_exponential(vocab_size, logits.device).expand(
            token_num, vocab_size
        )
        mixed_sample_outer_exponential(
            sampled_tokens, logits, exponential, temperatures, eps=self.eps
        )
        return sampled_tokens

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
