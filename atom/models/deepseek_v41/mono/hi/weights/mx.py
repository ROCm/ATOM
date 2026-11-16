# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 / MXFP8 codecs (OCP MX, group 32, E8M0 scales), in torch.

The packed weights and the references quantize by one rule, so they agree bit for bit:

- scale = the smallest power of two >= amax / FMAX (FMAX = 6 for E2M1, 448 for
  E4M3), as an E8M0 code clipped to [1, 254]; an all-zero group gets code 127;
- elements are divided by the scale and rounded to nearest, ties to even,
  saturating at FMAX.

Weights are packed two E2M1 codes per byte, even element in the low nibble.
"""

import torch

GROUP = 32
FP4_MAX = 6.0
FP8_MAX = 448.0
ZERO_CODE = 127  # the E8M0 code of 1.0, used for all-zero groups and padding

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# midpoints between consecutive E2M1 magnitudes
_FP4_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _groups(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point() or x.shape[-1] % GROUP:
        raise ValueError(
            f"MX quantization needs a float tensor with K % {GROUP} == 0, got {x.dtype} {tuple(x.shape)}"
        )
    x = x.float()
    if not bool(torch.isfinite(x).all()):
        raise ValueError("MX quantization rejects NaN / inf")
    return x.reshape(*x.shape[:-1], -1, GROUP)


def scale_codes(amax: torch.Tensor, fmax: float) -> torch.Tensor:
    """E8M0 code of the smallest power of two >= amax / fmax (code 127 for 0)."""
    # float64 keeps amax / fmax exact enough that a ratio landing exactly on a
    # power of two is not pushed across it by rounding
    mantissa, exponent = torch.frexp(amax.double() / fmax)
    exponent = exponent - (mantissa == 0.5).int()  # exact powers of two stay put
    codes = (exponent + 127).clamp(1, 254)
    return torch.where(amax == 0, ZERO_CODE, codes).to(torch.uint8)


def code_values(codes: torch.Tensor) -> torch.Tensor:
    """The fp32 power of two each E8M0 code stands for."""
    return torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes.int() - 127)


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [..., K] -> (packed uint8 [..., K/2], scale codes uint8 [..., K/32])."""
    g = _groups(x)
    codes = scale_codes(g.abs().amax(-1), FP4_MAX)
    y = g / code_values(codes).unsqueeze(-1)
    bounds = y.new_tensor(_FP4_BOUNDS)
    mag = y.abs().clamp(max=FP4_MAX).contiguous()
    idx = torch.bucketize(mag, bounds)  # a tie picks the lower magnitude...
    tie = mag == bounds[idx.clamp(max=len(_FP4_BOUNDS) - 1)]
    idx = idx + (tie & (idx % 2 == 1)).long()  # ...so round odd codes up to even
    e2m1 = (idx | (torch.signbit(y).long() << 3)).to(torch.uint8).reshape(x.shape)
    packed = e2m1[..., 0::2] | (e2m1[..., 1::2] << 4)
    return packed.contiguous(), codes.contiguous()


def dequantize_mxfp4(packed: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_mxfp4`, fp32 [..., K]."""
    k = packed.shape[-1] * 2
    e2m1 = torch.stack((packed & 15, packed >> 4), -1).reshape(*packed.shape[:-1], k)
    table = torch.tensor(
        _FP4_VALUES + tuple(-v for v in _FP4_VALUES), device=packed.device
    )
    values = table[e2m1.long()].reshape(*packed.shape[:-1], -1, GROUP)
    return (values * code_values(codes).unsqueeze(-1)).reshape(*packed.shape[:-1], k)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [..., K] -> (float8_e4m3fn [..., K], scale codes uint8 [..., K/32])."""
    g = _groups(x)
    codes = scale_codes(g.abs().amax(-1), FP8_MAX)
    y = (g / code_values(codes).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return y.reshape(x.shape).to(torch.float8_e4m3fn), codes.contiguous()


def dequantize_mxfp8(values: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    g = values.float().reshape(*values.shape[:-1], -1, GROUP)
    return (g * code_values(codes).unsqueeze(-1)).reshape(values.shape)


# ---------------------------------------------------------------- packed layouts
# The aiter layouts Atom's native MXFP4 path uses (shuffle_weight(layout=(16, 16))
# and shuffle_scale). Blocks that want other layouts declare their own packers.


# ---------------------------------------------------------------- MXFP8 weights


class MXFP8Weight:
    """One MXFP8 linear for ``Linear(weight_format="mxfp8")``: E4M3 values ``packed`` [N, K]
    and E8M0 ``codes`` [N, K/32] (the attribute names ``Linear.pack`` reads)."""

    fmt = "mxfp8"

    def __init__(self, packed: torch.Tensor, codes: torch.Tensor):
        if packed.dtype != torch.float8_e4m3fn or codes.dtype != torch.uint8:
            raise ValueError(
                f"MXFP8Weight needs float8_e4m3fn values and uint8 codes, got {packed.dtype} {codes.dtype}"
            )
        if codes.shape != (packed.shape[0], packed.shape[1] // GROUP):
            raise ValueError(
                f"MXFP8Weight codes {tuple(codes.shape)} do not match values {tuple(packed.shape)}"
            )
        self.packed, self.codes = packed, codes

    @classmethod
    def quantize(cls, w: torch.Tensor) -> "MXFP8Weight":
        return cls(*quantize_mxfp8(w))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.packed.shape)

    def dequantize(self) -> torch.Tensor:
        return dequantize_mxfp8(self.packed, self.codes)

    def to(self, device) -> "MXFP8Weight":
        return MXFP8Weight(self.packed.to(device), self.codes.to(device))


def fp8_block_to_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> MXFP8Weight:
    """An FP8 block-scaled weight (E4M3 ``weight`` [N, K], E8M0 ``scale`` [ceil(N/bn), ceil(K/bk)],
    e.g. DeepSeek's 32 x 32 or 128 x 128 blocks) as MXFP8, exactly: the values are unchanged and
    each row's group-32 code is its block's scale (bk a multiple of 32)."""
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"fp8_block_to_mxfp8 needs float8_e4m3fn weights, got {weight.dtype}"
        )
    n, k = weight.shape
    codes = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    if codes.dtype != torch.uint8:
        raise ValueError(f"fp8_block_to_mxfp8 needs E8M0 scales, got {scale.dtype}")
    bn, bk = -(-n // codes.shape[0]), -(-k // codes.shape[1])
    if k % GROUP or bk % GROUP:
        raise ValueError(
            f"fp8_block_to_mxfp8: K {k} and the K block {bk} must be multiples of {GROUP}"
        )
    per_row = codes.repeat_interleave(bn, 0)[:n]
    return MXFP8Weight(
        weight, per_row.repeat_interleave(bk // GROUP, 1)[:, : k // GROUP].contiguous()
    )
