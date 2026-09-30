# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host wrapper for the single-launch Kimi-K3 decode MonoKernel."""

from __future__ import annotations

import torch

from atom.model_ops.monokernel.config import ConvStateLayout
from atom.model_ops.monokernel.k3.prepared import KimiK3PreparedWeights
from atom.model_ops.monokernel.k3.staged import _KimiK3KdaStagedPath
from atom.model_ops.monokernel.weights import LayerWeights


class KimiK3MonoKernel(_KimiK3KdaStagedPath):
    """Run KDA, AttnRes, latent-MoE, TP reductions, and residual update in one launch."""

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        layer_idx: int,
        rank: int,
        npes: int = 8,
        group=None,
        reduce_group=None,
        mtp: bool = False,
        conv_state_layout: ConvStateLayout = ConvStateLayout.CHANNEL_MAJOR,
        conv_state_rows: int = 3,
        state_dtype: torch.dtype = torch.float32,
        prepared_weights: KimiK3PreparedWeights | None = None,
    ) -> None:
        super().__init__(
            weights,
            samples,
            layer_idx=layer_idx,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            fuse_attn_res=True,
            fuse_router=True,
            fuse_shared_experts=True,
            reduce_backend="symmetric",
            mtp=mtp,
            conv_state_layout=conv_state_layout,
            conv_state_rows=conv_state_rows,
            state_dtype=state_dtype,
            prepared_weights=prepared_weights,
            prepared_backend="mono",
        )
        self.attention.configure_monokernel(layer_idx, fuse_moe=True)

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        """Run one complete Kimi-K3 decode layer."""

        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != self.S
            or block_residual.shape[2] != self.config.hidden
        ):
            raise ValueError(
                "block_residual must have shape "
                f"[{self.S}, blocks, {self.config.hidden}], got {tuple(block_residual.shape)}"
            )
        if block_residual.shape[1] <= self.block_write_idx:
            raise ValueError(
                f"block_residual needs index {self.block_write_idx}, " f"got {block_residual.shape[1]} blocks"
            )

        target = self.output if x_out is None else x_out
        self.attention.forward(
            prefix_sum,
            state_indices,
            conv_state,
            recurrent_state,
            num_accepted_tokens=num_accepted_tokens,
            x_out=self.attention_delta,
            block_residual=block_residual,
            pre_updated=self.pre_updated,
            pre_output=self.pre_attn,
            updated_prefix=self.updated_prefix,
            moe_input=self.moe_input,
            quantized_moe_input=self.latent_projection.activation,
            quantized_moe_scale=self.latent_projection.activation_scale,
            monokernel_output=target,
            moe_symmetric=self.symmetric_allreduce.peer_buffer.local_address,
            moe_peers=self.symmetric_allreduce.peer_buffer.addresses,
            layer=epoch_layer,
            advance=False,
        )
        if advance:
            self.advance_step()
        return target


__all__ = ["KimiK3MonoKernel"]
