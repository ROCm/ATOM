# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Phase stamps of the mono layer kernel inside the server (debug only).

``ATOM_MONO_TIMELINE=<prefix>``, meant for ``--enforce-eager`` (a graph replay
runs no Python, so nothing would be saved): the layer kernels are built with their
timeline stamps, every sparse layer writes its own buffer, and past the warm-up
steps of each decode token count a few steps are saved as
``<prefix>_r<rank>_s<S>_<i>.pt`` -- int64 [layer][CTA][stamp], ``s_memrealtime``
ticks (100 MHz), a stamp never reached left 0.
"""

import collections

import torch

from atom.models.minimax_m3.mono.config import BLOCKS
from atom.models.minimax_m3.mono.kernels.post_attn import TL_POINTS

WARMUP_STEPS = 20  # steps of a token count skipped before any is saved
SAVED_STEPS = 5  # steps of a token count saved


class LayerTimeline:
    def __init__(self, prefix: str, n_layers: int, rank: int, dev) -> None:
        self.prefix = prefix
        self.rank = rank
        self.stamps = torch.zeros(
            n_layers, BLOCKS, TL_POINTS, dtype=torch.int64, device=dev
        )
        self.steps = collections.Counter()

    def ptr(self, layer: int) -> int:
        return self.stamps[layer].data_ptr()

    def step_done(self, tokens: int) -> None:
        seen = self.steps[tokens]
        self.steps[tokens] += 1
        if WARMUP_STEPS <= seen < WARMUP_STEPS + SAVED_STEPS:
            torch.save(
                self.stamps.cpu(),
                f"{self.prefix}_r{self.rank}_s{tokens}_{seen - WARMUP_STEPS}.pt",
            )
