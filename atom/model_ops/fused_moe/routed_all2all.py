# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-torch routing helpers for the native MoE all-to-all backend.

The helpers intentionally do not import AITER or distributed state.  Besides
keeping the routing policy independently testable on CPU, this makes the
contract between routing and transport explicit:

* ``topk_ids`` are physical IDs in dispatch space.
* physical experts are laid out in contiguous, equally sized rank shards.
* a token is sent once per destination rank, even if several of its selected
  experts live on that rank.
* each destination copy retains valid global expert IDs; AITER's binary
  ``expert_mask`` filters the non-local routes during MoE sorting.

The transport layer exchanges the returned rows in reverse and uses
``token_indices`` to sum contributions from different expert-owner ranks back
into the source token order.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutedDispatchPlan:
    """Destination-major payload and the inverse combine map for one rank."""

    token_indices: torch.Tensor
    send_counts: torch.Tensor
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class RoutedPayloadLayout:
    """Byte widths and dtypes needed to decode one packed dispatch row."""

    hidden_dim: int
    topk: int
    hidden_dtype: torch.dtype
    ids_dtype: torch.dtype
    weights_dtype: torch.dtype
    hidden_bytes: int
    ids_offset: int
    ids_bytes: int
    weights_offset: int
    weights_bytes: int
    row_bytes: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_routed_dispatch_plan(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    world_size: int,
) -> RoutedDispatchPlan:
    """Pack local tokens into destination-rank-major routed rows.

    Each output row represents one ``(token, destination rank)`` pair.  If two
    top-k experts for a token live on the same rank, the hidden state appears
    only once and both expert columns remain enabled in that row.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be rank 2, got shape={tuple(hidden_states.shape)}"
        )
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must both be rank 2")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights must have the same shape, got "
            f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError(
            "hidden_states and top-k tensors must have the same token count, got "
            f"{hidden_states.shape[0]} and {topk_ids.shape[0]}"
        )
    if not (hidden_states.device == topk_ids.device == topk_weights.device):
        raise ValueError("hidden_states and top-k tensors must be on the same device")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")
    if num_local_experts <= 0:
        raise ValueError("num_local_experts must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    num_tokens, topk = topk_ids.shape
    num_experts = num_local_experts * world_size
    valid = topk_ids >= 0
    invalid = valid & (topk_ids >= num_experts)
    if bool(invalid.any().item()):
        bad_id = int(topk_ids[invalid][0].item())
        raise ValueError(
            f"physical expert id {bad_id} is outside dispatch space [0, {num_experts})"
        )

    safe_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
    owners = torch.div(safe_ids, num_local_experts, rounding_mode="floor").to(
        torch.int64
    )

    # scatter_add handles duplicate owners deterministically: two experts on
    # one rank still create one token row for that destination.
    destination_hits = torch.zeros(
        (num_tokens, world_size),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    if topk:
        destination_hits.scatter_add_(1, owners, valid.to(torch.int32))
    destination_mask = destination_hits > 0

    # nonzero on [destination, token] returns destination-major rows, exactly
    # the layout expected by all_to_all_single input_split_sizes.
    pairs = destination_mask.transpose(0, 1).contiguous().nonzero(as_tuple=False)
    token_indices = pairs[:, 1]
    send_counts = destination_mask.sum(dim=0, dtype=torch.int64)

    # AITER's EP sorting kernels index expert_mask with every top-k ID before
    # filtering non-local routes, so -1 sentinels are not safe here. Preserve
    # the complete global routing row for parity with the gather/scatter path,
    # and sanitize only genuinely empty input slots to an in-range zero-weight
    # route. The destination's binary expert_mask performs the ownership
    # filtering inside fused_moe.
    selected_ids = safe_ids.index_select(0, token_indices)
    selected_weights = topk_weights.index_select(0, token_indices)
    selected_valid = valid.index_select(0, token_indices)
    dispatch_weights = torch.where(
        selected_valid, selected_weights, torch.zeros_like(selected_weights)
    )
    dispatch_hidden = hidden_states.index_select(0, token_indices)

    return RoutedDispatchPlan(
        token_indices=token_indices,
        send_counts=send_counts,
        hidden_states=dispatch_hidden.contiguous(),
        topk_ids=selected_ids.contiguous(),
        topk_weights=dispatch_weights.contiguous(),
    )


def combine_routed_rows(
    returned_rows: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum destination-rank contributions back into source token order."""
    if returned_rows.shape[0] != token_indices.numel():
        raise ValueError(
            "returned row count must match token_indices, got "
            f"{returned_rows.shape[0]} and {token_indices.numel()}"
        )
    expected_shape = (num_tokens, *returned_rows.shape[1:])
    if output is None:
        output = returned_rows.new_zeros(expected_shape)
    else:
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"output shape must be {expected_shape}, got {tuple(output.shape)}"
            )
        output.zero_()
    if token_indices.numel():
        output.index_add_(0, token_indices, returned_rows)
    return output


def pack_routed_payload(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, RoutedPayloadLayout]:
    """Pack mixed-dtype routed fields into one uint8 all-to-all payload."""
    rows, hidden_dim = hidden_states.shape
    if topk_ids.shape[0] != rows or topk_weights.shape != topk_ids.shape:
        raise ValueError("routed payload fields must have matching row counts")
    topk = topk_ids.shape[1]
    hidden_bytes = hidden_dim * hidden_states.element_size()
    ids_bytes = topk * topk_ids.element_size()
    weights_bytes = topk * topk_weights.element_size()
    ids_offset = _align_up(hidden_bytes, topk_ids.element_size())
    weights_offset = _align_up(ids_offset + ids_bytes, topk_weights.element_size())
    row_alignment = max(
        hidden_states.element_size(),
        topk_ids.element_size(),
        topk_weights.element_size(),
    )
    row_bytes = _align_up(weights_offset + weights_bytes, row_alignment)
    payload = torch.zeros(
        (rows, row_bytes), dtype=torch.uint8, device=hidden_states.device
    )
    payload[:, :hidden_bytes].copy_(
        hidden_states.contiguous().view(torch.uint8).reshape(rows, hidden_bytes)
    )
    payload[:, ids_offset : ids_offset + ids_bytes].copy_(
        topk_ids.contiguous().view(torch.uint8).reshape(rows, ids_bytes)
    )
    payload[:, weights_offset : weights_offset + weights_bytes].copy_(
        topk_weights.contiguous().view(torch.uint8).reshape(rows, weights_bytes)
    )
    return payload, RoutedPayloadLayout(
        hidden_dim=hidden_dim,
        topk=topk,
        hidden_dtype=hidden_states.dtype,
        ids_dtype=topk_ids.dtype,
        weights_dtype=topk_weights.dtype,
        hidden_bytes=hidden_bytes,
        ids_offset=ids_offset,
        ids_bytes=ids_bytes,
        weights_offset=weights_offset,
        weights_bytes=weights_bytes,
        row_bytes=row_bytes,
    )


def unpack_routed_payload(
    payload: torch.Tensor,
    layout: RoutedPayloadLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse :func:`pack_routed_payload` into contiguous typed tensors."""
    expected_width = layout.row_bytes
    if payload.dtype != torch.uint8 or payload.ndim != 2:
        raise ValueError("routed payload must be a rank-2 uint8 tensor")
    if payload.shape[1] != expected_width:
        raise ValueError(
            f"routed payload width must be {expected_width}, got {payload.shape[1]}"
        )

    hidden_states = (
        payload[:, : layout.hidden_bytes]
        .contiguous()
        .view(layout.hidden_dtype)
        .reshape(payload.shape[0], layout.hidden_dim)
    )
    topk_ids = (
        payload[:, layout.ids_offset : layout.ids_offset + layout.ids_bytes]
        .contiguous()
        .view(layout.ids_dtype)
        .reshape(payload.shape[0], layout.topk)
    )
    topk_weights = (
        payload[
            :,
            layout.weights_offset : layout.weights_offset + layout.weights_bytes,
        ]
        .contiguous()
        .view(layout.weights_dtype)
        .reshape(payload.shape[0], layout.topk)
    )
    return hidden_states, topk_ids, topk_weights
