# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import warnings

import torch
from aiter import mixed_sample_outer_exponential
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.topk import topk
from torch import nn

# Try to import aiter top-k/top-p sampling ops
try:
    import aiter.ops.sampling  # noqa: F401

    aiter_ops = torch.ops.aiter
    AITER_TOPK_TOPP_AVAILABLE = True
except ImportError:
    AITER_TOPK_TOPP_AVAILABLE = False
    warnings.warn(
        "aiter.ops.sampling not available. Top-k/top-p sampling will use "
        "experimental native PyTorch implementation as fallback.",
        UserWarning,
        stacklevel=1,
    )

# Track whether we've already warned about native sampling being used
_NATIVE_SAMPLING_WARNING_ISSUED = False


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(
        self,
        logits: torch.Tensor,  # (num_tokens, vocab_size)
        temperatures: torch.Tensor,  # (num_tokens,)
        top_ks: torch.Tensor | None = None,  # (num_tokens,) int32, -1 means disabled
        top_ps: torch.Tensor | None = None,  # (num_tokens,) float32, 1.0 means disabled
    ) -> torch.Tensor:  # (num_tokens,)
        """
        Sample tokens from logits using temperature or top-k top-p filtering.

        Args:
            logits: Raw logits from model (num_tokens, vocab_size)
            temperatures: Temperature for each token (num_tokens,)
            top_ks: Top-k value per token, -1 means disabled (num_tokens,)
            top_ps: Top-p value per token, 1.0 means disabled (num_tokens,)

        Returns:
            Sampled token IDs (num_tokens,)
        """
        # No Top-K Top-P parameters, perform temperature-based sampling
        if not self._needs_filtering(top_ks, top_ps):
            return self._temperature_sample(logits, temperatures)

        # Apply top-k/top-p filtering
        return self._topk_topp_sample(logits, temperatures, top_ks, top_ps)

    def _needs_filtering(
        self,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> bool:
        """Check if any request needs top-k or top-p filtering."""
        if top_ks is None and top_ps is None:
            return False

        needs_topk = top_ks is not None and (top_ks != -1).any()
        needs_topp = top_ps is not None and (top_ps < 1.0).any()

        return needs_topk or needs_topp

    def _temperature_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """Temperature-based Gumbel-max sampling."""
        sampled_tokens = torch.empty(
            logits.size(0), dtype=torch.int, device=logits.device
        )
        exponential = (
            torch.empty(
                (1, logits.shape[-1]), dtype=torch.float, device=logits.device
            )
            .exponential_(1)
            .expand(*logits.shape)
        )
        mixed_sample_outer_exponential(
            sampled_tokens, logits, exponential, temperatures, eps=self.eps
        )
        return sampled_tokens

    def _topk_topp_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> torch.Tensor:
        """Top-K/Top-P sampling with temperature scaling."""
        # Fast path: if ALL requests are greedy (temperature=0), just do argmax
        # This avoids the overhead of softmax and top-k/top-p filtering
        all_greedy = (temperatures == 0).all()
        if all_greedy:
            return logits.argmax(dim=-1).to(torch.int)

        # Apply temperature scaling
        # Clamp to avoid division by zero; temperature=0 handled separately as greedy
        scaled_logits = logits / temperatures.unsqueeze(-1).clamp(min=self.eps)
        probs = scaled_logits.softmax(dim=-1, dtype=torch.float32).contiguous()

        # Determine which filtering is needed
        has_topk = top_ks is not None and (top_ks != -1).any()
        has_topp = top_ps is not None and (top_ps < 1.0).any()

        if AITER_TOPK_TOPP_AVAILABLE:
            return self._aiter_sample(
                probs, top_ks, top_ps, has_topk, has_topp, temperatures
            )
        else:
            return self._native_sample(probs, top_ks, top_ps, temperatures)

    def _to_tensor_scalar(self, x: torch.Tensor):
        """Convert to (tensor, scalar) tuple for aiter ops.

        If tensor has size 1 (uniform value optimization from model_runner),
        extract the scalar value for more efficient aiter kernel dispatch.
        """
        if x is None:
            return (None, 0)
        if x.numel() == 1:  # Uniform value - use scalar for efficiency
            return (None, x[0].item())
        return (x, 0)

    def _aiter_sample(
        self,
        probs: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        has_topk: bool,
        has_topp: bool,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """Use aiter optimized ops for top-k/top-p sampling."""
        # Convert to tensor/scalar format for aiter
        k_tensor, k_scalar = self._to_tensor_scalar(top_ks)
        p_tensor, p_scalar = self._to_tensor_scalar(top_ps)

        if has_topk and has_topp:
            # Joint k+p path
            next_tokens = aiter_ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                k_tensor,
                k_scalar,
                p_tensor,
                p_scalar,
                deterministic=True,
            )
        elif has_topp:
            # Top-p only
            next_tokens = aiter_ops.top_p_sampling_from_probs(
                probs, None, p_tensor, p_scalar, deterministic=True
            )
        elif has_topk:
            # Top-k only: renormalize and multinomial
            renorm_probs = aiter_ops.top_k_renorm_probs(probs, k_tensor, k_scalar)
            next_tokens = torch.multinomial(renorm_probs, num_samples=1)
        else:
            # Neither - just multinomial from probs
            next_tokens = torch.multinomial(probs, num_samples=1)

        # Handle greedy sampling (temperature=0)
        greedy_mask = temperatures == 0
        if greedy_mask.any():
            next_tokens[greedy_mask] = probs[greedy_mask].argmax(dim=-1).unsqueeze(-1)

        return next_tokens.view(-1).to(torch.int)

    def _native_sample(
        self,
        probs: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        EXPERIMENTAL: Native PyTorch fallback for top-k/top-p sampling.

        This implementation has not been thoroughly tested and may produce
        different results compared to the optimized aiter implementation.
        Use aiter.ops.sampling for production workloads.
        """
        global _NATIVE_SAMPLING_WARNING_ISSUED
        if not _NATIVE_SAMPLING_WARNING_ISSUED:
            warnings.warn(
                "Using experimental native top-k/top-p sampling. "
                "Install aiter.ops.sampling for optimized performance.",
                UserWarning,
                stacklevel=2,
            )
            _NATIVE_SAMPLING_WARNING_ISSUED = True

        batch_size, vocab_size = probs.shape
        device = probs.device

        # Sort probs descending
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Top-p mask: keep tokens until cumsum exceeds top_p
        # The mask keeps tokens where cumsum - current_prob <= top_p
        # (i.e., before we exceed the threshold)
        if top_ps is not None:
            topp_mask = (cumsum_probs - sorted_probs) <= top_ps.unsqueeze(-1)
        else:
            topp_mask = torch.ones_like(sorted_probs, dtype=torch.bool)

        # Top-k mask: keep first k tokens
        if top_ks is not None:
            indices = torch.arange(vocab_size, device=device).unsqueeze(0)
            effective_k = torch.where(top_ks == -1, vocab_size, top_ks)
            topk_mask = indices < effective_k.unsqueeze(-1)
        else:
            topk_mask = torch.ones_like(sorted_probs, dtype=torch.bool)

        # Combined filtering
        mask = topp_mask & topk_mask
        mask[:, 0] = True  # Always keep at least one token

        filtered_probs = sorted_probs * mask.float()
        filtered_probs = filtered_probs / filtered_probs.sum(
            dim=-1, keepdim=True
        ).clamp(min=self.eps)

        # Sample and map back to original indices
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        next_tokens = sorted_indices.gather(1, sampled_idx.unsqueeze(-1)).squeeze(-1)

        # Handle greedy (temperature=0)
        greedy_mask = temperatures == 0
        if greedy_mask.any():
            next_tokens[greedy_mask] = probs[greedy_mask].argmax(dim=-1)

        return next_tokens.to(torch.int)

    # Legacy methods kept for reference
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
