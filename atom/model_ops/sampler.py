# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import warnings
from functools import lru_cache

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

# Epsilon value for numerical stability and to prevent division by 0
SAMPLER_EPS = 1e-10


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
        self.eps = SAMPLER_EPS

    def forward(
        self,
        logits: torch.Tensor,  # (num_tokens, vocab_size)
        temperatures: torch.Tensor,  # (num_tokens,)
        top_ks: torch.Tensor | None = None,  # (num_tokens,) int32, -1 means disabled
        top_ps: torch.Tensor | None = None,  # (num_tokens,) float32, 1.0 means disabled
        all_greedy: bool = False,  # True if all temperatures are 0 (checked on CPU)
        needs_independent_noise: bool = False,
    ) -> torch.Tensor:  # (num_tokens,)
        """
        Sample tokens from logits using temperature or top-k top-p filtering.

        Args:
            logits: Raw logits from model (num_tokens, vocab_size)
            temperatures: Temperature for each token (num_tokens,), pre-clamped to eps
            top_ks: Top-k value per token, -1 means disabled (num_tokens,)
            top_ps: Top-p value per token, 1.0 means disabled (num_tokens,)
            all_greedy: True if all requests use greedy sampling (checked on CPU)
            needs_independent_noise: True when the batch contains fan-out
                siblings (SamplingParams.n>1). Forces fresh per-row random
                exponential noise so sibling sequences with identical logits
                do not produce identical tokens. Adds an O(bs * vocab_size)
                allocation, so only enabled when actually needed.

        Returns:
            Sampled token IDs (num_tokens,)
        """
        # No Top-K Top-P parameters, perform temperature-based sampling
        if not self._needs_filtering(top_ks, top_ps):
            return self._temperature_sample(
                logits, temperatures, needs_independent_noise=needs_independent_noise
            )

        # Apply top-k/top-p filtering
        return self._topk_topp_sample(
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise=needs_independent_noise,
        )

    def _needs_filtering(
        self,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
    ) -> bool:
        """Check if any request needs top-k or top-p filtering.

        This check is O(1) - the actual filtering check is done on CPU in
        model_runner.prepare_sample(), which passes None if no filtering needed.
        """
        return top_ks is not None or top_ps is not None

    def _temperature_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        needs_independent_noise: bool = False,
    ) -> torch.Tensor:
        """Temperature-based Gumbel-max sampling.

        When ``needs_independent_noise`` is True the per-row exponential noise
        tensor is freshly drawn with shape ``(num_tokens, vocab_size)`` so that
        fan-out siblings produced by ``SamplingParams.n > 1`` diverge instead
        of collapsing onto the same token when they share logits. Otherwise we
        keep the cached ``(1, vocab_size)`` row broadcasted across the batch,
        which preserves the existing run-to-run determinism optimization.
        """
        num_tokens, vocab_size = logits.shape
        sampled_tokens = torch.empty(num_tokens, dtype=torch.int, device=logits.device)
        if needs_independent_noise:
            exponential = torch.empty(
                (num_tokens, vocab_size), dtype=torch.float, device=logits.device
            ).exponential_(1)
        else:
            exponential = get_per_token_exponential(vocab_size, logits.device).expand(
                num_tokens, vocab_size
            )
        mixed_sample_outer_exponential(
            sampled_tokens, logits, exponential, temperatures, eps=self.eps
        )
        return sampled_tokens

    def _topk_topp_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
        all_greedy: bool,
        needs_independent_noise: bool = False,
    ) -> torch.Tensor:
        """Top-K/Top-P sampling with temperature scaling.

        The top-k/top-p paths below use ``torch.multinomial`` (native fallback)
        or aiter's sampling ops that draw from torch's default RNG stream per
        batch row, so fan-out siblings already diverge naturally and the
        ``needs_independent_noise`` flag is accepted purely for API symmetry.
        """
        # Accepted but unused here; see docstring.
        del needs_independent_noise
        # Fast path: if ALL requests are greedy (temperature=0), just do argmax
        # This avoids the overhead of softmax and top-k/top-p filtering
        if all_greedy:
            return logits.argmax(dim=-1).to(torch.int)

        # Apply temperature scaling
        # Temperatures are pre-clamped to eps in model_runner.prepare_sample()
        scaled_logits = logits / temperatures.unsqueeze(-1)
        probs = scaled_logits.softmax(dim=-1, dtype=torch.float32).contiguous()

        # model_runner.prepare_sample passes None if filtering not needed for that type
        has_topk = top_ks is not None
        has_topp = top_ps is not None

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
