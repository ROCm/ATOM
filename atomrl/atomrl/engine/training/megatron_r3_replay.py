# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Install Megatron-Core ``RouterReplay`` from vLLM rollout expert ids.

Used by :class:`MegatronNativeEngine` for R3 (rollout → logprob/update). The
payload is hard-assignment indices, not router logits.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from atomrl.core.protocol import DataProto


def routes_from_batch(batch: DataProto) -> Tensor | list[Any]:
    """Return ragged ``rollout_routing`` or dense ``rollout_routed_experts``."""
    routes = batch.ragged.get("rollout_routing")
    if routes is not None:
        if len(routes) != batch.batch_size:
            raise ValueError(
                "rollout_routing row count mismatch: "
                f"got {len(routes)}, expected {batch.batch_size}"
            )
        if any(row is None for row in routes):
            raise RuntimeError(
                "moe.r3.enabled=true but rollout_routing contains missing rows."
            )
        return routes
    routes = batch.tensors.get("rollout_routed_experts")
    if routes is None:
        raise RuntimeError(
            "moe.r3.enabled=true but neither ragged rollout_routing nor "
            "rollout_routed_experts is present. "
            "R3 must not silently fall back to training-time routing."
        )
    if routes.ndim != 4:
        raise ValueError(
            "rollout_routed_experts must have shape [batch, seq_len-1, "
            f"num_layers, top_k], got {tuple(routes.shape)}"
        )
    return routes


def extract_row_routes(
    routes: Tensor | list[Any],
    row: int,
    start: int,
    length: int,
) -> Tensor:
    """Return the ``length - 1`` rollout routes for one real token row."""
    expected = length - 1
    if isinstance(routes, Tensor):
        extracted = routes[row, start : start + expected]
    else:
        extracted = torch.as_tensor(routes[row])
    if extracted.ndim != 3:
        raise ValueError(
            "one-row R3 routes must have shape [tokens, layers, top_k], "
            f"got {tuple(extracted.shape)}"
        )
    if extracted.shape[0] != expected:
        raise ValueError(
            f"R3 route length mismatch for row {row}: "
            f"got {extracted.shape[0]}, expected {expected}"
        )
    return extracted


def local_router_layer_indices(model: Any) -> list[int]:
    """Global 0-based transformer-layer index for each local ``RouterReplay``.

    Returned in ``RouterReplay.global_router_replay_instances`` order (the order
    ``set_replay_data`` expects), so the caller can slice the rollout route
    tensor's layer axis to exactly the MoE layers this rank owns.

    Megatron sets each router's ``layer_number`` (1-indexed) with the pipeline
    offset already applied (``get_transformer_layer_offset``) and attaches the
    ``RouterReplay`` instance to the same router, so the mapping is read straight
    off the constructed model rather than recomputed.
    """
    from megatron.core.transformer.moe.router_replay import RouterReplay

    instances = list(RouterReplay.global_router_replay_instances)
    if not instances:
        raise RuntimeError(
            "moe.r3.enabled=true but Megatron created no RouterReplay instances; "
            "set moe_enable_routing_replay on TransformerConfig."
        )
    by_id: dict[int, int] = {}
    for module in model.modules():
        replay = getattr(module, "router_replay", None)
        layer_number = getattr(module, "layer_number", None)
        if replay is None or layer_number is None:
            continue
        by_id[id(replay)] = int(layer_number) - 1
    try:
        return [by_id[id(inst)] for inst in instances]
    except KeyError as exc:
        raise RuntimeError(
            "R3 could not map a RouterReplay instance to a model layer_number. "
            "The module tree does not expose router_replay and layer_number on "
            "the same router, or global instances are stale (clear them before "
            "building the model)."
        ) from exc


def _route_meta(routes: Tensor | list[Any]) -> tuple[int, int, torch.dtype]:
    if isinstance(routes, Tensor):
        return int(routes.shape[2]), int(routes.shape[3]), routes.dtype
    sample = torch.as_tensor(routes[0])
    return int(sample.shape[1]), int(sample.shape[2]), sample.dtype


def _cp_local_indices(
    length: int,
    *,
    cp_size: int,
    cp_rank: int,
    contiguous: bool,
    align: int,
) -> Tensor:
    """Token indices this rank feeds the router (matches ``_pp_forward_model``)."""
    if cp_size <= 1:
        return torch.arange(length)
    if contiguous:
        chunk = -(-length // cp_size)
        chunk = -(-chunk // align) * align
        padded = cp_size * chunk
        start = cp_rank * chunk
        return torch.arange(start, start + chunk).clamp(max=padded - 1)
    chunk = (length + 2 * cp_size - 1) // (2 * cp_size)
    padded = 2 * cp_size * chunk
    first = cp_rank * chunk
    second = (2 * cp_size - cp_rank - 1) * chunk
    return torch.cat(
        [
            torch.arange(first, first + chunk),
            torch.arange(second, second + chunk),
        ]
    )


def _gather_token_routes(full: Tensor, idx: Tensor) -> Tensor:
    """``full`` is ``[seq, layers, topk]``; pad then gather along token 0."""
    n = int(full.shape[0])
    max_i = int(idx.max().item()) if idx.numel() else -1
    if max_i >= n:
        pad = full[-1:].expand(max_i + 1 - n, -1, -1)
        full = torch.cat([full, pad], dim=0)
    return full.index_select(0, idx.to(device=full.device))


def pack_microbatch_routes(
    routes: Tensor | list[Any] | None,
    rows: list[tuple[int, int, int]],
    packed_len: int,
    *,
    num_experts: int,
    layer_indices: list[int],
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_contiguous: bool = False,
    align: int = 1,
    tp_size: int = 1,
    tp_rank: int = 0,
    sequence_parallel: bool = False,
) -> list[Tensor]:
    """Build per-local-layer ``[tokens, topk]`` int64 tensors for ``set_replay_data``.

    ``layer_indices`` gives the global 0-based transformer-layer index for each
    local ``RouterReplay`` instance (see :func:`local_router_layer_indices`), in
    ``set_replay_data`` order. ``rows`` empty means an EP dummy microbatch: filler
    ids for ``packed_len``.
    """
    if packed_len <= 0:
        raise ValueError(f"R3 packed_len must be positive, got {packed_len}")
    if routes is None:
        raise ValueError("R3 packing requires rollout route metadata")

    if not rows:
        num_layers, topk, dtype = _route_meta(routes)
        filler = torch.arange(topk, dtype=dtype)
        replay = filler.view(1, 1, topk).expand(packed_len, num_layers, topk).clone()
    else:
        num_layers, topk, dtype = _route_meta(routes)
        filler = torch.arange(topk, dtype=dtype)
        pieces: list[Tensor] = []
        for row, start, length in rows:
            real = extract_row_routes(routes, row, start, length)
            last = filler.view(1, 1, topk).expand(1, num_layers, topk)
            full = torch.cat([real, last], dim=0)
            idx = _cp_local_indices(
                length,
                cp_size=cp_size,
                cp_rank=cp_rank,
                contiguous=cp_contiguous,
                align=align,
            )
            pieces.append(_gather_token_routes(full, idx))
        replay = torch.cat(pieces, dim=0)
        if replay.shape[0] < packed_len:
            pad = filler.view(1, 1, topk).expand(
                packed_len - replay.shape[0],
                num_layers,
                topk,
            )
            replay = torch.cat([replay, pad], dim=0)
        if replay.shape[0] != packed_len:
            raise ValueError(
                f"R3 packed token mismatch: routes={replay.shape[0]}, packed={packed_len}"
            )

    if sequence_parallel and tp_size > 1:
        if replay.shape[0] % tp_size:
            raise ValueError(
                f"R3 token count {replay.shape[0]} is not divisible by TP={tp_size}"
            )
        shard = replay.shape[0] // tp_size
        replay = replay[tp_rank * shard : (tp_rank + 1) * shard]

    total_layers = int(replay.shape[1])
    if any(idx < 0 or idx >= total_layers for idx in layer_indices):
        raise ValueError(
            f"R3 local layer index out of range for a {total_layers}-layer route "
            f"payload: {layer_indices}"
        )
    replay = replay.index_select(
        1, torch.as_tensor(layer_indices, dtype=torch.long, device=replay.device)
    )
    if (replay < 0).any() or (replay >= num_experts).any():
        bad = int(replay[(replay < 0) | (replay >= num_experts)][0].item())
        raise ValueError(
            f"R3 expert id {bad} is outside global range [0, {num_experts})"
        )
    return [
        replay[:, layer, :].to(dtype=torch.int64).contiguous()
        for layer in range(replay.shape[1])
    ]


def clear_stale_instances() -> None:
    from megatron.core.transformer.moe.router_replay import RouterReplay

    clear = getattr(RouterReplay, "clear_global_router_replay_instances", None)
    if not callable(clear):
        raise TypeError(
            "R3 requires RouterReplay.clear_global_router_replay_instances()."
        )
    clear()


def clear_replay() -> None:
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()


def install_replay(layer_tensors: list[Tensor], *, append: bool) -> None:
    """Load indices into Megatron ``RouterReplay`` and set ``REPLAY_FORWARD``."""
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    instances = list(RouterReplay.global_router_replay_instances)
    if not instances:
        raise RuntimeError(
            "moe.r3.enabled=true but Megatron created no RouterReplay instances; "
            "set moe_enable_routing_replay on TransformerConfig."
        )
    if len(layer_tensors) != len(instances):
        raise ValueError(
            f"R3 has {len(layer_tensors)} local layers but this stage has "
            f"{len(instances)} RouterReplay instances."
        )
    # ``set_target_indices`` both overwrites ``target_topk_idx`` (current forward)
    # and appends to ``replay_backward_list``. Clearing between update microbatches
    # would drop the FIFO that recomputation needs; logprob is forward-only so
    # each microbatch can replace the previous payload.
    if not append:
        RouterReplay.clear_global_indices()
    RouterReplay.set_replay_data(layer_tensors)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
