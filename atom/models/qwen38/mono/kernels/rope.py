# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU RoPE table construction for Mono weight binding."""

import torch

BF16 = torch.bfloat16


def rope_cos_sin(
    positions: torch.Tensor,
    rotary_dim: int,
    theta: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [n, rotary_dim] (bf16) for text positions, on ``positions``' device.

    Qwen3.5's interleaved MRoPE gives a text token the same position on all three axes,
    which makes it plain RoPE over the first ``rotary_dim`` dims.

    **The contract is this computation on the GPU** (``device``, default ``positions``' device,
    must then be CUDA): ``inv = 1 / theta ** (arange(0, rotary_dim, 2) / rotary_dim)`` and ``freq = fp32(position) *
    inv`` in fp32, ``cos`` / ``sin`` in fp32 on the GPU, each rounded once to bf16. That is how Atom / aiter build their
    cos / sin cache (``RotaryEmbedding`` under ``torch.set_default_device("cuda")``): bit-exact against Atom's cache at
    every position below 262144. The same formula on the CPU differs (its ``pow`` and ``cos`` round differently), so the
    table must be built on the GPU. CPU callers may pass ``device="cuda"`` (the result comes back on ``positions``'
    device).
    """
    dev = positions.device if device is None else torch.device(device)
    inv = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=dev).float() / rotary_dim)
    )
    freq = positions.to(dev).float()[:, None] * inv
    freq = torch.cat((freq, freq), -1)
    return freq.cos().to(BF16).to(positions.device), freq.sin().to(BF16).to(
        positions.device
    )
