# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Immutable, sample-independent Kimi-K3 kernel weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.model_ops.monokernel.config import KIMI_K3_CONFIG
from atom.model_ops.monokernel.formats import quantize_mxfp8
from atom.model_ops.monokernel.packing import (
    pack_bf16,
    pack_mxfp8_scale,
    pack_mxfp8_weight,
)
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    prepare_aiter_mxfp4_expert_storage,
)

MONOKERNEL_INPUT_ROWS = 6400


@dataclass(frozen=True)
class PreparedMxfp8Weight:
    rows: int
    cols: int
    weight: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class KimiK3PreparedTailWeights:
    source: LayerWeights
    backend: str
    rank: int
    npes: int
    source_ptrs: tuple[tuple[str, int], ...]
    w_router: torch.Tensor
    w_ug: torch.Tensor
    s_ug: torch.Tensor
    w_dn: torch.Tensor
    s_dn: torch.Tensor
    latent_down: PreparedMxfp8Weight
    shared_up: PreparedMxfp8Weight
    shared_down: PreparedMxfp8Weight
    latent_up: PreparedMxfp8Weight

    def validate_source(self, weights: LayerWeights, backend: str | None = None) -> None:
        if backend is not None and backend != self.backend:
            raise ValueError(
                f"prepared Kimi backend is {self.backend!r}, requested {backend!r}"
            )
        if weights is not self.source:
            raise ValueError("prepared Kimi weights belong to a different LayerWeights")
        if (
            weights.config != KIMI_K3_CONFIG
            or weights.rank != self.rank
            or weights.npes != self.npes
        ):
            raise ValueError("prepared Kimi weight geometry changed")
        current = tuple(
            (name, weights.t[name].data_ptr()) for name, _ in self.source_ptrs
        )
        if current != self.source_ptrs:
            raise ValueError("prepared Kimi source storage changed")


@dataclass(frozen=True)
class KimiK3PreparedWeights(KimiK3PreparedTailWeights):
    w_kda_in_packed: torch.Tensor | None
    w_kda_o_packed: torch.Tensor | None


def _prepare_mxfp8(weight: torch.Tensor) -> PreparedMxfp8Weight:
    quantized, scale = quantize_mxfp8(weight)
    return PreparedMxfp8Weight(
        rows=weight.shape[0],
        cols=weight.shape[1],
        weight=pack_mxfp8_weight(quantized),
        scale=pack_mxfp8_scale(scale),
    )


def _prepare_tail_kwargs(weights: LayerWeights, backend: str) -> dict:
    if weights.config != KIMI_K3_CONFIG:
        raise ValueError(f"expected Kimi-K3 weights, got {weights.config.name!r}")
    tensors = weights.t
    required = {
        "w_r",
        "w_latent_down",
        "w_shared_ug",
        "w_shared_dn",
        "w_latent_up",
        "w_ug",
        "s_ug",
        "w_dn",
        "s_dn",
    }
    missing = sorted(required.difference(tensors))
    if missing:
        raise ValueError(f"missing Kimi tail weights: {', '.join(missing)}")
    w_ug, s_ug, w_dn, s_dn = prepare_aiter_mxfp4_expert_storage(weights)
    source_ptrs = tuple(
        (name, tensor.data_ptr()) for name, tensor in sorted(tensors.items())
    )
    return {
        "source": weights,
        "backend": backend,
        "rank": weights.rank,
        "npes": weights.npes,
        "source_ptrs": source_ptrs,
        "w_router": pack_bf16(tensors["w_r"]),
        "w_ug": w_ug,
        "s_ug": s_ug,
        "w_dn": w_dn,
        "s_dn": s_dn,
        "latent_down": _prepare_mxfp8(tensors["w_latent_down"]),
        "shared_up": _prepare_mxfp8(tensors["w_shared_ug"]),
        "shared_down": _prepare_mxfp8(tensors["w_shared_dn"]),
        "latent_up": _prepare_mxfp8(tensors["w_latent_up"]),
    }


def prepare_kimi_k3_tail_weights(
    weights: LayerWeights,
) -> KimiK3PreparedTailWeights:
    return KimiK3PreparedTailWeights(**_prepare_tail_kwargs(weights, "tail"))


def prepare_kimi_k3_weights(
    weights: LayerWeights,
    backend: str,
    *,
    mtp: bool = False,
) -> KimiK3PreparedWeights:
    if weights.config != KIMI_K3_CONFIG:
        raise ValueError(f"expected Kimi-K3 weights, got {weights.config.name!r}")
    if backend not in {"staged", "mono"}:
        raise ValueError(f"unsupported Kimi prepared backend {backend!r}")
    tensors = weights.t
    config = weights.config
    projection = config.local_heads * config.v_dim
    fused_width = 4 * projection + config.local_heads + config.v_dim
    if tensors["w_kda_in"].shape != (fused_width, config.hidden):
        raise ValueError(f"w_kda_in has shape {tuple(tensors['w_kda_in'].shape)}")
    if tensors["w_kda_o"].shape != (config.hidden, projection):
        raise ValueError(f"w_kda_o has shape {tuple(tensors['w_kda_o'].shape)}")

    w_kda_in_packed = None
    w_kda_o_packed = None
    if backend == "mono" or mtp:
        monokernel_input = torch.zeros(
            MONOKERNEL_INPUT_ROWS,
            config.hidden,
            dtype=torch.bfloat16,
            device=tensors["w_kda_in"].device,
        )
        monokernel_input[:fused_width].copy_(tensors["w_kda_in"])
        w_kda_in_packed = pack_bf16(monokernel_input)
        w_kda_o_packed = pack_bf16(tensors["w_kda_o"])
    return KimiK3PreparedWeights(
        **_prepare_tail_kwargs(weights, backend),
        w_kda_in_packed=w_kda_in_packed,
        w_kda_o_packed=w_kda_o_packed,
    )


__all__ = [
    "KimiK3PreparedTailWeights",
    "KimiK3PreparedWeights",
    "MONOKERNEL_INPUT_ROWS",
    "PreparedMxfp8Weight",
    "prepare_kimi_k3_tail_weights",
    "prepare_kimi_k3_weights",
]
