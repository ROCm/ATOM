# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Kimi-K3 C1 fused MoE tail."""

from __future__ import annotations

import torch

from atom.model_ops.monokernel.config import KIMI_K3_CONFIG
from atom.model_ops.monokernel.k3.moe import kimi_k3_mxfp4_gemm1, kimi_k3_mxfp4_gemm2
from atom.model_ops.monokernel.k3.prepared import KimiK3PreparedTailWeights
from atom.model_ops.monokernel.k3.router_projection import FusedRouterProjection
from atom.model_ops.monokernel.k3.tail import FusedKimiK3Tail
from atom.model_ops.monokernel.mxfp8_linear import Mxfp8Linear
from atom.model_ops.monokernel.symmetric_allreduce import SymmetricBf16Allreduce
from atom.model_ops.monokernel.weights import LayerWeights

_TP_SIZE = 8
_ROUTING_TILE_M = 16


class _KimiK3FusedTail:
    """Run the measured C1 router, latent MoE, shared experts, and TP reductions."""

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        layer_idx: int,
        rank: int,
        npes: int,
        group,
        prepared_weights: KimiK3PreparedTailWeights,
    ) -> None:
        config = weights.config
        if config != KIMI_K3_CONFIG:
            raise ValueError("the Kimi-K3 fused tail requires Kimi-K3 weights")
        if samples != 8:
            raise ValueError(f"the Kimi-K3 fused tail requires C1/S8, got S{samples}")
        if npes != _TP_SIZE:
            raise ValueError(f"the Kimi-K3 fused tail requires TP8, got TP{npes}")
        if weights.rank != rank or weights.npes != npes:
            raise ValueError(
                f"weight shard is rank {weights.rank}/TP{weights.npes}, "
                f"requested rank {rank}/TP{npes}"
            )
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        if config.routed_hidden is None or config.shared_inter is None:
            raise ValueError("Kimi-K3 latent-MoE dimensions are missing")

        required = {"bias", "g_latent"}
        missing = sorted(required.difference(weights.t))
        if missing:
            raise ValueError(f"missing Kimi-K3 tail weights: {', '.join(missing)}")
        prepared_weights.validate_source(weights, "tail")

        self.config = config
        self.samples = samples
        self.layer_idx = layer_idx
        self.rank = rank
        self.npes = npes
        self.t = weights.t
        self.hidden_shard = config.hidden // npes
        self.step = torch.zeros(
            1, dtype=torch.int32, device=prepared_weights.w_router.device
        )

        self.w_router = prepared_weights.w_router
        self.w_ug = prepared_weights.w_ug
        self.s_ug = prepared_weights.s_ug
        self.w_dn = prepared_weights.w_dn
        self.s_dn = prepared_weights.s_dn
        self.latent_projection = Mxfp8Linear.from_prepared(
            prepared_weights.latent_down, samples
        )
        self.w_shared_up = prepared_weights.shared_up.weight
        self.s_shared_up = prepared_weights.shared_up.scale
        self.w_shared_down = prepared_weights.shared_down.weight
        self.s_shared_down = prepared_weights.shared_down.scale
        self.w_latent_up = prepared_weights.latent_up.weight
        self.s_latent_up = prepared_weights.latent_up.scale

        device = prepared_weights.w_router.device
        max_sorted = samples * config.top_k * _ROUTING_TILE_M
        max_blocks = (max_sorted + _ROUTING_TILE_M - 1) // _ROUTING_TILE_M
        self.max_sorted = max_sorted
        self.sorted_token_ids = torch.empty(
            max_sorted, dtype=torch.int32, device=device
        )
        self.sorted_weights = torch.empty(
            max_sorted, dtype=torch.float32, device=device
        )
        self.sorted_expert_ids = torch.empty(
            max_blocks, dtype=torch.int32, device=device
        )
        self.num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
        self.inter_sorted = torch.empty(
            max_sorted, config.inter, dtype=torch.bfloat16, device=device
        )
        self.router_scores = torch.empty(
            samples, config.n_experts, dtype=torch.float32, device=device
        )
        self.topk_ids = torch.empty(
            samples, config.top_k, dtype=torch.int32, device=device
        )
        self.topk_weights = torch.empty(
            samples, config.top_k, dtype=torch.float32, device=device
        )
        self.router_score_mailbox = torch.zeros(
            samples * config.n_experts * 2,
            dtype=torch.int32,
            device=device,
        )
        self.latent = torch.empty(
            samples, config.routed_hidden, dtype=torch.bfloat16, device=device
        )
        self.routed_partial = torch.empty_like(self.latent)
        self.routed_reduced = torch.empty_like(self.latent)
        self.latent_norm = torch.empty_like(self.latent)
        self.shared_up = torch.empty(
            samples, 2 * config.shared_inter, dtype=torch.bfloat16, device=device
        )
        self.shared_mid = torch.empty(
            samples, config.shared_inter, dtype=torch.bfloat16, device=device
        )
        self.shared_partial = torch.empty(
            samples, config.hidden, dtype=torch.bfloat16, device=device
        )
        self.tail = torch.empty(
            samples, self.hidden_shard, dtype=torch.bfloat16, device=device
        )
        self.final_partial = torch.empty_like(self.shared_partial)
        self.moe_delta = torch.empty_like(self.shared_partial)
        self.output = torch.empty_like(self.shared_partial)

        self.router_projection = FusedRouterProjection(
            config.hidden,
            config.n_experts,
            config.top_k,
            samples,
            samples * config.routed_hidden,
            config.routed_hidden,
            2 * config.shared_inter,
            config.situ_beta,
            config.situ_linear_beta,
        )
        self.symmetric_allreduce = SymmetricBf16Allreduce(
            (self.routed_partial.numel(), self.final_partial.numel()),
            rank=rank,
            npes=npes,
            group=group,
            rmsnorm_width=config.routed_hidden,
        )
        self.fused_tail = FusedKimiK3Tail(
            samples,
            config.hidden,
            config.routed_hidden,
            config.shared_inter,
            rank,
            npes,
            self.symmetric_allreduce.max_pairs,
        )

    def _route(self, hidden_states: torch.Tensor, epoch_layer: int) -> None:
        self.router_projection(
            hidden_states,
            self.latent_projection.activation,
            self.latent_projection.activation_scale,
            self.w_router,
            self.latent_projection.weight,
            self.latent_projection.scale,
            self.w_shared_up,
            self.s_shared_up,
            self.t["bias"],
            self.router_score_mailbox,
            self.router_scores,
            self.topk_ids,
            self.topk_weights,
            self.sorted_token_ids,
            self.sorted_weights,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.routed_partial,
            self.latent,
            self.shared_up,
            self.shared_mid,
            self.step,
            epoch_layer,
        )

    def _moe(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        output: torch.Tensor,
        epoch_layer: int,
    ) -> torch.Tensor:
        self._route(hidden_states, epoch_layer)
        kimi_k3_mxfp4_gemm1(
            self.latent,
            self.w_ug,
            self.s_ug,
            sorted_expert_ids=self.sorted_expert_ids,
            num_valid_ids=self.num_valid_ids,
            sorted_token_ids=self.sorted_token_ids,
            output=self.inter_sorted,
            samples=self.samples,
            situ_beta=self.config.situ_beta,
            situ_linear_beta=self.config.situ_linear_beta,
        )
        kimi_k3_mxfp4_gemm2(
            self.inter_sorted,
            self.w_dn,
            self.s_dn,
            sorted_expert_ids=self.sorted_expert_ids,
            num_valid_ids=self.num_valid_ids,
            sorted_token_ids=self.sorted_token_ids,
            sorted_weights=self.sorted_weights,
            output=self.routed_partial,
            samples=self.samples,
            max_sorted=self.max_sorted,
        )
        return self.fused_tail(
            self.routed_partial,
            self.routed_reduced,
            self.t["g_latent"],
            self.symmetric_allreduce.rmsnorm_scratch,
            self.shared_mid,
            self.latent_norm,
            self.w_shared_down,
            self.s_shared_down,
            self.w_latent_up,
            self.s_latent_up,
            residual,
            self.shared_partial,
            self.tail,
            self.final_partial,
            self.moe_delta,
            output,
            self.symmetric_allreduce.peer_buffer.local_address,
            self.symmetric_allreduce.peer_buffer.addresses,
            self.step,
            epoch_layer,
        )

    def forward_from_moe_input(
        self,
        updated_prefix: torch.Tensor,
        moe_input: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        epoch_layer: int = 0,
    ) -> torch.Tensor:
        expected = (self.samples, self.config.hidden)
        if updated_prefix.shape != expected or moe_input.shape != expected:
            raise ValueError(f"updated_prefix and moe_input must have shape {expected}")
        self.latent_projection.quantize_input(moe_input)
        target = self.output if x_out is None else x_out
        self._moe(moe_input, updated_prefix, target, epoch_layer)
        self.step.add_(1)
        return target

    def close(self) -> None:
        self.symmetric_allreduce.close()
