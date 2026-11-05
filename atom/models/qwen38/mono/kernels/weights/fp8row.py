# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 per-output-channel weights (E4M3 values, one fp32 scale a row) as ``Linear(weight_format="mxfp8",
row_scale=True)`` reads them.

The row scales are not powers of two, so no E8M0 group code can carry them: the values are packed as MXFP8 with every
group code ``ZERO_CODE`` (scale 1), which keeps the stored E4M3 bytes unchanged, and the row scale multiplies the fp32
accumulator. ``weight[n, k] = E4M3(packed[n, k]) * row_scale[n]`` is the checkpoint's own dequantization (Quark's,
ATOM's); the kernel scales the row's fp32 sum instead of each product (a summation-order difference).
"""

import torch

GROUP = 32  # the MX group along K
ZERO_CODE = 127  # the E8M0 code of 1.0


class FP8RowWeight:
    """E4M3 ``packed`` [N, K], unit E8M0 ``codes`` [N, K / 32] (``Linear.pack``'s MXFP8 attributes) and an fp32
    ``row_scale`` [N]."""

    fmt = "mxfp8"

    def __init__(
        self, packed: torch.Tensor, codes: torch.Tensor, row_scale: torch.Tensor
    ):
        if packed.dtype != torch.float8_e4m3fn or codes.dtype != torch.uint8:
            raise ValueError(
                f"FP8RowWeight needs float8_e4m3fn values and uint8 codes, got {packed.dtype} {codes.dtype}"
            )
        if codes.shape != (packed.shape[0], packed.shape[1] // GROUP):
            raise ValueError(
                f"FP8RowWeight codes {tuple(codes.shape)} do not match values {tuple(packed.shape)}"
            )
        self.packed, self.codes = packed, codes
        if row_scale.dtype != torch.float32 or tuple(row_scale.shape) != (
            packed.shape[0],
        ):
            raise ValueError(f"FP8RowWeight needs an fp32 row scale [{packed.shape[0]}], got "
                             f"{row_scale.dtype} {tuple(row_scale.shape)}")  # fmt: skip
        if bool((codes != ZERO_CODE).any()):
            raise ValueError(
                "FP8RowWeight: every group code must be ZERO_CODE (the row scale carries the scale)"
            )
        self.row_scale = row_scale

    @classmethod
    def from_checkpoint(
        cls, weight: torch.Tensor, row_scale: torch.Tensor
    ) -> "FP8RowWeight":
        """The checkpoint's E4M3 ``weight`` [N, K] and fp32 ``row_scale`` [N] (or [N, 1]), values unchanged."""
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"FP8RowWeight needs float8_e4m3fn weights, got {weight.dtype}"
            )
        n, k = weight.shape
        if k % GROUP:
            raise ValueError(f"FP8RowWeight: K {k} must be a multiple of {GROUP}")
        if row_scale.numel() != n:
            raise ValueError(
                f"FP8RowWeight: {row_scale.numel()} row scales for {n} rows"
            )
        codes = torch.full(
            (n, k // GROUP), ZERO_CODE, dtype=torch.uint8, device=weight.device
        )
        return cls(
            weight.contiguous(),
            codes,
            row_scale.reshape(n).to(torch.float32).contiguous(),
        )

    def dequantize(self) -> torch.Tensor:
        """The checkpoint's dequantized weight, fp32 [N, K] (one fp32 product an element, as Quark and ATOM)."""
        return self.packed.float() * self.row_scale.unsqueeze(1)
