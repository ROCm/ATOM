# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The router's gate in EPLB's slot order, on the layers EPLB places.

The MoE kernel takes a routed expert's id for its slot: rank r holds slots
r E / ep .. (``kernels.moe.is_local``). Without redundant experts EPLB's
placement is a permutation of the experts over those slots, so a gate whose
row p (and score bias p) is the expert slot p holds routes to slots directly,
the kernel unchanged. Each layer's copy follows its EPLB commits."""

from dataclasses import replace

import torch

from atom.model_ops.eplb import (
    add_eplb_commit_listener,
    get_live_expert_location_metadata,
)
from atom.models.deepseek_v41.mono.weights import LayerWeights

GATE = ("ffn.gate.weight", "ffn.gate.e_score_correction_bias")


def bind_eplb_gates(layers: list[LayerWeights]) -> list[LayerWeights]:
    """``layers``, each one EPLB places reading its gate in slot order; as
    given without EPLB."""
    meta = get_live_expert_location_metadata()
    if meta is None:
        return layers
    order = meta.physical_to_logical_map
    gates: dict[int, tuple[dict, dict]] = {}
    bound = []
    for layer in layers:
        if layer.layer_id >= meta.num_layers:
            bound.append(layer)
            continue
        src = {name: layer[name] for name in GATE}
        dst = {name: torch.empty_like(t) for name, t in src.items()}
        gates[layer.layer_id] = (src, dst)
        bound.append(replace(layer, tensors={**layer.tensors, **dst}))

    def reorder(layer_id: int) -> None:
        src, dst = gates[layer_id]
        for name in GATE:
            torch.index_select(src[name], 0, order[layer_id], out=dst[name])

    for layer_id in gates:
        reorder(layer_id)
    add_eplb_commit_listener(reorder)
    return bound
