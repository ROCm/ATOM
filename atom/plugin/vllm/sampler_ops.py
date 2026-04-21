import torch
import aiter


def greedy_sample_tokens(logits: torch.Tensor) -> torch.Tensor:
    """Plugin-local greedy sampler wrapper that routes to AITER."""

    if not logits.is_contiguous():
        logits = logits.contiguous()

    sampled_tokens = torch.empty(
        logits.size(0), dtype=torch.int32, device=logits.device
    )
    aiter.greedy_sample(sampled_tokens, logits)
    return sampled_tokens
