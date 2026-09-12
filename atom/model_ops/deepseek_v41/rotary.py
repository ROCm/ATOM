# SPDX-License-Identifier: MIT
"""Interleaved CSA2 RoPE with separate window and compressed YaRN frequencies."""

import math

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position,
        *,
        base,
        original_length=0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
    ):
        super().__init__()
        # Keep frequency construction in FP32, including on a BF16 model context.
        frequencies = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        if original_length > 0:

            def corrected_dim(rotations):
                return (
                    dim
                    * math.log(original_length / (rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = max(math.floor(corrected_dim(beta_fast)), 0)
            high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
            ramp = (
                (torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)
            ).clamp(0, 1)
            frequencies = frequencies / factor * ramp + frequencies * (1 - ramp)
        angles = torch.outer(torch.arange(max_position), frequencies)
        self.register_buffer(
            "frequencies",
            torch.polar(torch.ones_like(angles), angles),
            persistent=False,
        )

    def forward(self, x, positions, *, inverse=False):
        """Rotate the final RoPE dimensions in place, preserving the NoPE prefix."""
        freqs = self.frequencies[positions]
        dim = freqs.shape[-1] * 2
        tail = x[..., -dim:]
        pairs = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
        if inverse:
            freqs = freqs.conj()
        shape = [1, positions.numel()] + [1] * (pairs.ndim - 3) + [dim // 2]
        tail.copy_(torch.view_as_real(pairs * freqs.view(shape)).flatten(-2))
        return x
