# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Undoing the two AITER weight layouts, so a loaded tensor can be read back.

The mono kernels read every large weight where the model already keeps it. Two
of K3's do not survive that rule: the latent MoE's down- and up-projections are
MXFP4 in the loaded model and the kernels' latent GEMVs are bf16, so those two
are dequantized once at bind time (``weights.py``). Dequantizing needs the
*unshuffled* bytes, and ATOM stores them in AITER's layouts:

* ``aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))`` -- 16-row by
  32-column tiles, applied to the packed FP4 bytes;
* ``aiter.ops.shuffle.shuffle_scale`` (``fp4_utils.e8m0_shuffle``) -- the E8M0
  scale plane, which also pads the group count up to a multiple of 8.

Each function here is the exact inverse of its AITER counterpart, derived from
that function's own view / permute, and
``tests/models/test_kimi_k3_mono_layouts.py`` holds it to a round trip through
the real AITER one.
"""

import torch

# aiter's shuffle_weight(layout=(16, 16)): 16 rows (N) by IK * 2 = 32 columns
TILE_N, TILE_K = 16, 32
# ... and the element group inside a tile row: 16 bytes
TILE_BYTES = 16
# shuffle_scale pads a plane up to these before permuting it (E8M0 1.0 in the
# pad, so a kernel reading a whole tile sees ones rather than NaNs)
SCALE_GROUP_ALIGN = 8
SCALE_ROW_ALIGN = 256


def unshuffle_weight(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``shuffle_weight(x, layout=(16, 16))`` on a 2-D weight.

    The forward is
    ``view(-1, N // 16, 16, K // 32, 32 // E, E).permute(0, 1, 3, 4, 2, 5)``
    with ``E = 16 // element_size``; this applies the inverse permutation.
    """
    dtype = x.dtype
    if dtype != torch.uint8:
        x = x.view(torch.uint8)
    if x.dim() != 2:
        raise ValueError(f"unshuffle_weight wants a 2-D weight, got {tuple(x.shape)}")
    n, k = x.shape
    if n % TILE_N or k % TILE_K:
        raise ValueError(f"weight {n}x{k} is not {TILE_N}x{TILE_K}-tiled")
    inner = TILE_K // TILE_BYTES
    out = x.reshape(1, n // TILE_N, k // TILE_K, inner, TILE_N, TILE_BYTES)
    out = out.permute(0, 1, 4, 2, 3, 5).contiguous()
    return out.reshape(n, k).view(dtype)


def unshuffle_scale(x: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    """Inverse of ``e8m0_shuffle`` / ``shuffle_scale`` for a ``[rows, groups]``
    E8M0 plane, dropping the rows and columns the forward padded in.

    The forward pads to (256, 8) multiples, then
    ``view(R // 32, 2, 16, G // 8, 2, 4).permute(0, 3, 5, 2, 4, 1)`` -- the same
    permutation ``atom.mono.device.mx.scale_index`` indexes a byte of.
    """
    dtype = x.dtype
    flat = x.reshape(-1).view(torch.uint8)
    pad_rows = -(-rows // SCALE_ROW_ALIGN) * SCALE_ROW_ALIGN
    pad_groups = -(-groups // SCALE_GROUP_ALIGN) * SCALE_GROUP_ALIGN
    if flat.numel() != pad_rows * pad_groups:
        raise ValueError(
            f"scale plane has {flat.numel()} bytes, expected "
            f"{pad_rows} x {pad_groups}"
        )
    out = flat.reshape(pad_rows // 32, pad_groups // 8, 4, 16, 2, 2)
    out = out.permute(0, 5, 3, 1, 4, 2).contiguous()
    out = out.reshape(pad_rows, pad_groups)[:rows, :groups].contiguous()
    return out.view(dtype)
