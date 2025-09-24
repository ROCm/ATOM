import torch
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
        logits = logits.float()
        return torch.where(
            temperatures == 0, self.greedy_sample(logits), self.random_sample(logits)
        )

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
