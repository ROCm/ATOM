# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One decoder layer's weights as the model package loads them (``mono.weights.load_layer``).

``linears``: name (the HF module name, e.g. ``"mlp.shared_expert.down_proj"``) -> an ``MXLinear`` (packed E2M1 bytes
and E8M0 codes, HF row order: group quantization runs along K, so a block's packer may stack or reorder rows), an
``FP8RowWeight`` or a bf16 tensor. ``retained``: tensors kept as stored (norm gammas, conv taps, gate parameters).
Blocks read the names the recipe gives them (``Block.pack``).
"""

from dataclasses import dataclass

import torch


@dataclass
class MXLinear:
    """One MXFP4 linear: packed E2M1 bytes [N, K/2] and E8M0 codes [N, K/32]."""

    packed: torch.Tensor
    codes: torch.Tensor


@dataclass
class QuantizedLayer:
    index: int
    kind: str  # the layer's program kind ("delta" | "qsa" | "mtp")
    linears: dict
    retained: dict[str, torch.Tensor]
