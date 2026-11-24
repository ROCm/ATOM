# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The routed experts' MXFP4 values in the layout the W4A4 kernels read: a pure permutation of the checkpoint's bytes,
within a 16-row block aiter's ``shuffle_weight`` (16, 16). The scales go through
``prefill_h6144_i768.Mxfp4PrefillScales.from_row_major``.

  W13  [E, I/16, 2 (gate, up), K/128, klane 4, nlane 16, 16 B]   K = H
  W2   [E, H/16, I/128, klane 4, nlane 16, 16 B]
"""

import torch

TILE_K = 128  # K per 1 KB block of 16 columns


def _shuffle_values(packed: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] u8 -> [E, N/16, K/128, klane 4, nlane 16, 16 B]."""
    e, n, half_k = packed.shape
    k = half_k * 2
    v = packed.view(e, n // 16, 16, k // TILE_K, 4, 16)  # e nb nl kt kl j
    return v.permute(0, 1, 3, 4, 2, 5).contiguous()


def w13_layout(packed: torch.Tensor) -> torch.Tensor:
    """Gate / up e2m1 codes ``[E, 2I, H/2]`` (gate rows first) -> W13 (module docstring): per gate / up half, then the
    halves interleaved per 16 rows."""
    e, n2, half_k = packed.shape
    inter = n2 // 2
    assert n2 % 32 == 0 and (half_k * 2) % 256 == 0, packed.shape
    w = _shuffle_values(packed.reshape(e * 2, inter, half_k))
    return w.view(e, 2, inter // 16, *w.shape[2:]).transpose(1, 2).contiguous()


def w2_layout(packed: torch.Tensor) -> torch.Tensor:
    """Down e2m1 codes ``[E, H, I/2]`` -> W2 (module docstring)."""
    e, h, half_k = packed.shape
    assert h % 16 == 0 and (half_k * 2) % 256 == 0, packed.shape
    return _shuffle_values(packed)


def router_line_layout(weight: torch.Tensor) -> torch.Tensor:
    """FP32 [128,6144] -> [192,8,2,4,16,4] for contiguous router loads.

    A wave's 64 lanes read 16 bytes each: eight complete 128-byte lines.
    This pure permutation runs once at weight loading, before graph capture.
    """
    assert weight.shape == (128, 6144) and weight.dtype == torch.float32
    return weight.reshape(8, 16, 192, 4, 2, 4).permute(2, 0, 4, 3, 1, 5).contiguous()
