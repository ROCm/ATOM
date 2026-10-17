# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-call context for one fused QKV+attention invocation."""

from dataclasses import dataclass, field
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

    def prologue_state(self) -> dict:
        return self.attn.impl.gemm_fusion_prologue_state(self.fwd_ctx)

    def alloc_outputs(self, spec_prefix_kinds, site, device, dtype) -> dict:
        """Outputs of the covered prefix. Rows are the padded global count."""
        impl = self.attn.impl
        t = self.padded_tokens
        q_w = impl.num_heads * impl.head_dim
        kv_w = impl.num_kv_heads * impl.head_dim
        outs: dict = {}
        if "head_norm_rope_cache" in spec_prefix_kinds:
            outs["q"] = torch.empty((t, q_w), dtype=dtype, device=device)
            if not site.sparse:  # sparse backends read the cache, not k/v
                outs["k"] = torch.empty((t, kv_w), dtype=dtype, device=device)
                outs["v"] = torch.empty((t, kv_w), dtype=dtype, device=device)
            if site.sparse and not site.skip_index:
                outs["index_q"] = torch.empty(
                    (t, impl.index_q_size), dtype=dtype, device=device
                )
        else:
            outs["packed"] = torch.empty(
                (t, site.packed_width(impl)), dtype=dtype, device=device
            )
        return outs
