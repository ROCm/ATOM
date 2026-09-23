# SPDX-License-Identifier: MIT
"""Interleaved CSA2 RoPE with separate window and compressed YaRN frequencies."""

import math

import torch
from aiter import (
    rope_cached_positions_2c_fwd_inplace,
    rope_cached_positions_fwd_inplace,
)
from torch import nn

from atom.model_ops.v4_kernels.inverse_rope import inverse_rope_inplace


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
        frequencies = torch.polar(torch.ones_like(angles), angles)
        # The cached AITER API expects contiguous real caches. Keep FP32 YaRN
        # frequencies, including for BF16 activations, as on the original path.
        self.register_buffer(
            "cos_cache", frequencies.real.contiguous(), persistent=False
        )
        self.register_buffer(
            "sin_cache", frequencies.imag.contiguous(), persistent=False
        )

    @property
    def frequencies(self):
        """Complex view for CPU/reference callers; GPU execution uses the caches."""
        return torch.complex(self.cos_cache, self.sin_cache)

    @property
    def rope_dim(self):
        """Trailing width this rotates; a row's NoPE prefix is what is left."""
        return self.cos_cache.shape[-1] * 2

    def forward(self, x, positions, *, inverse=False):
        """Rotate the final RoPE dimensions in place, preserving the NoPE prefix."""
        if x.is_cuda:
            return self._rotate_cuda(x, positions, inverse=inverse)
        freqs = self.frequencies[positions]
        dim = freqs.shape[-1] * 2
        tail = x[..., -dim:]
        pairs = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
        if inverse:
            freqs = freqs.conj()
        shape = [1, positions.numel()] + [1] * (pairs.ndim - 3) + [dim // 2]
        tail.copy_(torch.view_as_real(pairs * freqs.view(shape)).flatten(-2))
        return x

    def pair(self, x, y, positions):
        """Rotate two tensors that share a position line, in one launch.

        Their head counts may differ, which is what the two-channel entry is
        for. Forward only: `inverse` has one channel.
        """
        if not (x.is_cuda and y.is_cuda) or not x.numel() or not y.numel():
            return self.forward(x, positions), self.forward(y, positions)
        values_x, values_y = x.contiguous(), y.contiguous()
        flat_positions, cos, sin, dim, rows = self._cached_args(x, positions)
        rope_cached_positions_2c_fwd_inplace(
            values_x[..., -dim:].view(1, rows, -1, dim),
            values_y[..., -dim:].view(1, rows, -1, dim),
            cos,
            sin,
            flat_positions.to(torch.int64).view(1, -1),
            1,  # GPT-J interleaved pairs, as in V4.
            reuse_freqs_front_part=True,
            nope_first=False,
        )
        if values_x is not x:
            x.copy_(values_x)
        if values_y is not y:
            y.copy_(values_y)
        return x, y

    def _cached_args(self, x, positions):
        """Position line, frequency views, rotate width and folded row count."""
        batch, length = x.shape[:2]
        flat_positions = (
            positions.repeat(batch) if batch > 1 else positions.contiguous()
        )
        # A length-one strided view is contiguous to PyTorch, but the cached
        # RoPE ABI still requires a literal unit position stride.
        if flat_positions.stride(0) != 1:
            flat_positions = flat_positions.clone(memory_format=torch.contiguous_format)
        return (
            flat_positions,
            self.cos_cache[:, None, None, :],
            self.sin_cache[:, None, None, :],
            self.rope_dim,
            batch * length,
        )

    def _rotate_cuda(self, x, positions, *, inverse):
        if x.numel() == 0:
            return x
        # Contiguous model outputs use a view. Copy back only for strided callers
        # so the public rotation remains in place for batched chunk views.
        values = x.contiguous()
        flat_positions, cos, sin, dim, rows = self._cached_args(x, positions)
        if inverse:
            inverse_rope_inplace(
                values.view(rows, -1, x.shape[-1]),
                cos,
                sin,
                flat_positions,
                dim,
            )
        else:
            rope_cached_positions_fwd_inplace(
                values[..., -dim:].view(1, rows, -1, dim),
                cos,
                sin,
                flat_positions.to(torch.int64).view(1, -1),
                1,  # GPT-J interleaved pairs, as in V4.
                reuse_freqs_front_part=True,
                nope_first=False,
            )
        if values is not x:
            x.copy_(values)
        return x
