# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Zero-padding a tensor-parallel dimension at load, GPU-free.

The preshuffled MXFP4 GEMMs cannot handle a K that is not a multiple of 256
(``atom.model_ops.linear.MXFP4_PRESHUFFLE_K_ALIGN``). Kimi-K3's dense MLP has a
per-rank intermediate of 33792 / TP8 = 4224, so ``KimiMLP`` pads it the way MoE
pads its intermediate. These helpers re-lay each checkpoint tensor so the stock
tensor-parallel loaders place it into the padded parameters unchanged.

Torch only -- no AITER -- so the layout logic is covered by CPU CI.
"""

import torch
from torch import nn


def padded_local_size(real_local: int, align: int) -> int:
    """Round a per-rank size up to a multiple of ``align``."""
    return (real_local + align - 1) // align * align


def pad_tp_chunks(
    t: torch.Tensor, dim: int, tp_size: int, real: int, padded: int, fill: float
) -> torch.Tensor:
    """Re-lay a checkpoint tensor for a per-rank size padded from `real` to `padded`.

    `t` is the full (unsharded) tensor; along `dim` it holds `tp_size` equal rank
    chunks. Each chunk is extended by the same fraction the per-rank intermediate
    grows by -- which covers weights in elements, fp4x2 bytes (half) and e8m0
    scale groups (1/32) alike -- and the result is laid out so the stock
    tensor-parallel loader, chunking by `tp_size`, lands each real chunk at the
    front of its padded rank slice.
    """
    chunk = t.shape[dim] // tp_size
    extra, rem = divmod((padded - real) * chunk, real)
    assert t.shape[dim] == chunk * tp_size and rem == 0, (
        f"cannot pad dim {dim} of {tuple(t.shape)} from {real} to {padded} "
        f"per rank at tp={tp_size}"
    )
    pad_shape = list(t.shape)
    pad_shape[dim] = extra
    pad = torch.full(pad_shape, fill, dtype=t.dtype, device=t.device)
    parts = []
    for c in t.split(chunk, dim=dim):
        parts += [c, pad]
    return torch.cat(parts, dim=dim)


def pad_param_loads(
    linear: nn.Module, dim: int, tp_size: int, real: int, padded: int
) -> None:
    """Pad every checkpoint tensor this linear loads (see `pad_tp_chunks`).

    Padded weight elements are zero, so they contribute nothing whatever scale
    sits beside them; padded scales are 1.0 (e8m0 127) only so they are finite.
    """
    for name in ("weight", "weight_scale"):
        param = getattr(linear, name, None)
        if param is None:
            continue
        fill = 0
        if name == "weight_scale":
            fill = 127 if param.dtype.itemsize == 1 else 1.0
        inner = param.weight_loader

        def loader(p, loaded, *args, _inner=inner, _fill=fill):
            return _inner(
                p, pad_tp_chunks(loaded, dim, tp_size, real, padded, _fill), *args
            )

        param.weight_loader = loader
