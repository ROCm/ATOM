# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-call context for one fused QKV+attention invocation."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AttnFusionCtx:
    attn: Any
    positions: torch.Tensor
    fwd_ctx: Any
    world: int
    real_tokens: int  # global tokens, excluding SP divisibility padding
    padded_tokens: int  # local rows * world

    @classmethod
    def from_forward_context(cls, attn, positions, local_rows, world):
        from atom.utils.forward_context import get_forward_context

        return cls(
            attn=attn,
            positions=positions,
            fwd_ctx=get_forward_context(),
            world=world,
            real_tokens=positions.shape[-1],
            padded_tokens=local_rows * world,
        )

    @property
    def is_dummy_run(self) -> bool:
        return bool(self.fwd_ctx.context.is_dummy_run)

    def prologue_state(self):
        return self.attn.impl.gemm_fusion_prologue_state(self.fwd_ctx)
