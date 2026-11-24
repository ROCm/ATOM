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


def rope_frequencies(
    dim, *, base, original_length=0, factor=1.0, beta_fast=32, beta_slow=1
):
    """Each rotated pair's frequency, FP32 [dim // 2]; YaRN when ``original_length`` > 0."""
    # Keep frequency construction in FP32, including on a BF16 model context.
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
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
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        frequencies = frequencies / factor * ramp + frequencies * (1 - ramp)
    return frequencies


def fp32_tables(rope, *, widened=False):
    """``rope``'s ``(cos, sin)`` as contiguous fp32 ``[positions, pairs]``
    tensors (``pairs`` the tables' last dim, ``rope_dim // 2``), or None: what
    a kernel that indexes ``table[position * pairs + pair]`` in fp32 can be
    handed.

    Tables that are fp32 already come back as 2-D views of their own memory,
    whatever singleton dims sit between the position and the pair (V4 keeps
    ``[positions, 1, 1, pairs]``); every kernel of this seam reads them by
    that flat index. BF16 tables -- this model's -- have no fp32 memory of
    their own; ``widened`` hands out the twins `build_fp32_tables` registered
    (None before it ran). A caller may only ask for them where the launch it
    stands in for widens each BF16 entry and then computes in fp32 exactly as
    the fp32-table launch does; that is a property of the compiled reference
    kernel, not of the table.
    """
    cos, sin = rope.cos_cache, rope.sin_cache
    if (
        cos.dim() < 2
        or cos.shape != sin.shape
        or cos.dtype != sin.dtype
        or any(n != 1 for n in cos.shape[1:-1])
        or not (cos.is_contiguous() and sin.is_contiguous())
    ):
        return None
    pairs = cos.shape[-1]
    if cos.dtype == torch.float32:
        return cos.view(-1, pairs), sin.view(-1, pairs)
    if not widened:
        return None
    twins = getattr(rope, "cos_fp32", None), getattr(rope, "sin_fp32", None)
    if any(
        t is None
        or t.dtype != torch.float32
        or t.shape != (cos.shape[0], pairs)
        or not t.is_contiguous()
        for t in twins
    ):
        return None
    return twins


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
        frequencies = rope_frequencies(
            dim,
            base=base,
            original_length=original_length,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        angles = torch.outer(torch.arange(max_position), frequencies)
        frequencies = torch.polar(torch.ones_like(angles), angles)
        # The cached AITER API expects contiguous real caches. BF16, as every
        # other model's RoPE tables (V4's included).
        self.register_buffer(
            "cos_cache",
            frequencies.real.to(torch.bfloat16).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            frequencies.imag.to(torch.bfloat16).contiguous(),
            persistent=False,
        )

    def build_fp32_tables(self):
        """Register ``cos_fp32`` / ``sin_fp32``: the caches widened to fp32,
        value for value, for a kernel that reads an fp32 table and must
        compute what a kernel widening the BF16 one computes (`fp32_tables`).

        Call while the model is built: a tensor first made inside a graph
        capture would live in the capture's pool. Twice the caches' bytes.
        """
        if "cos_fp32" not in self._buffers:
            for name in ("cos", "sin"):
                table = getattr(self, f"{name}_cache").float().contiguous()
                self.register_buffer(f"{name}_fp32", table, persistent=False)
        return self.cos_fp32, self.sin_fp32

    @property
    def frequencies(self):
        """Complex view for CPU/reference callers; GPU execution uses the caches."""
        return torch.complex(self.cos_cache.float(), self.sin_cache.float())

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
