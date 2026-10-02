# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The one check that backend-published PAGE views are what they claim to be.

Both LMCache MP registrations -- PAGE-only and native state -- hand the same
block-major views to the server, so they share this validation rather than
keeping two copies that drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import logging

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class PageView:
    """One validated block-major PAGE view and the region it aliases."""

    index: int
    view: torch.Tensor
    region: Any
    unit_bytes: int


def validate_page_views(
    transfer_tensors: Any,
    *,
    num_blocks: int,
    block_size: int | None = None,
) -> list[PageView]:
    """Validate ``block_tensor_views`` against ``block_regions``.

    Each view must be ``[num_blocks, physical_slots, opaque_width]``, tightly
    block-major (one unit per block, contiguous inside it), exactly cover and
    start at its region, be forward-indexed, and share one device with the
    rest. ``block_size``, when given, also requires the physical slots of a
    block to divide it.
    """
    if transfer_tensors is None:
        raise ValueError("lmcache_mp requires KVTransferTensors")
    if type(num_blocks) is not int or num_blocks <= 0:
        raise ValueError("lmcache_mp num_blocks must be a positive integer")
    regions = list(getattr(transfer_tensors, "block_regions", None) or [])
    views = list(getattr(transfer_tensors, "block_tensor_views", None) or [])
    if not regions or len(views) != len(regions):
        raise ValueError(
            "lmcache_mp requires one block_tensor_view per block region: "
            f"views={len(views)} regions={len(regions)}"
        )

    return _validate_views(
        views,
        regions,
        num_blocks=num_blocks,
        block_size=block_size,
        label="PAGE",
        index_offset=0,
    )


def _validate_views(
    views: list[Any],
    regions: list[Any],
    *,
    num_blocks: int,
    block_size: int | None,
    label: str,
    index_offset: int,
) -> list[PageView]:
    """The per-view geometry check both block-major registrations share."""
    validated: list[PageView] = []
    devices: set[torch.device] = set()
    for offset, (view, region) in enumerate(zip(views, regions, strict=True)):
        index = index_offset + offset
        name = f"LMCache MP {label} view {index}"
        if not isinstance(view, torch.Tensor):
            raise TypeError(f"{name} is not a Tensor")
        if view.ndim != 3 or int(view.shape[0]) != num_blocks:
            raise ValueError(
                f"{name} has shape={tuple(view.shape)}, "
                f"expected [{num_blocks}, physical_slots, opaque_width]"
            )
        if view.numel() == 0 or not view[0].is_contiguous():
            raise ValueError(f"{name} must be non-empty and contiguous")
        if block_size is not None and block_size % int(view.shape[1]):
            raise ValueError(f"{name} physical slots must divide block size")
        unit_bytes = int(region.unit_bytes)
        actual_unit_bytes = view[0].numel() * view.element_size()
        if (
            unit_bytes <= 0
            or actual_unit_bytes != unit_bytes
            or view.stride(0) * view.element_size() != unit_bytes
            or int(region.total_bytes) != num_blocks * unit_bytes
        ):
            raise ValueError(
                f"{name} byte geometry mismatch: unit={actual_unit_bytes}/"
                f"{unit_bytes} stride={view.stride(0) * view.element_size()} "
                f"total={region.total_bytes}/{num_blocks * unit_bytes}"
            )
        if view.data_ptr() != int(region.base_addr):
            raise ValueError(f"{name} does not alias its declared region")
        if bool(getattr(region, "reverse_indexed", False)):
            raise ValueError(f"lmcache_mp {label} regions cannot be reverse-indexed")
        devices.add(view.device)
        validated.append(PageView(index, view, region, unit_bytes))
    if len(devices) != 1:
        raise ValueError(f"lmcache_mp {label} views must share one device")
    return validated


@dataclass(frozen=True)
class _RecurrentViews:
    """One recurrent group's registration identity, after validation."""

    # Positions in the flat registration order -- what LMCache calls
    # ``layer_indices`` -- so this group's tensors can be named without
    # depending on where the PAGE planes ended.
    tensor_indices: tuple[int, ...]
    num_blocks: int
    tokens_per_block: int
    bytes_per_block: int


@dataclass(frozen=True)
class _CacheViews:
    """Validated block-major tensors and their copy-kernel group indices."""

    tensors: dict[str, torch.Tensor]
    layer_groups: tuple[tuple[int, ...], ...]
    bytes_per_block: int
    # Recurrent groups in ordinal order, registered after the PAGE planes.
    # Empty on an attention-only layout.
    recurrent: tuple[_RecurrentViews, ...] = ()


def _token_major_view(
    view: torch.Tensor,
    tokens_per_block: int,
    *,
    label: str,
) -> torch.Tensor:
    """Re-expose an opaque ``[num_units, 1, unit_bytes]`` PAGE view as
    ``[num_units, tokens_per_block, bytes_per_token]``.

    Zero-copy and byte-identical -- only the way the extent is split changes.

    WHY this is not cosmetic.  LMCache rebuilds the chunk shape from the
    registration TWICE, with two different formulas, and they only agree when
    the engine's detected ``block_size`` is the real tokens-per-block:

      client  (``v1/multiprocess/transfer_context/worker_transfer.py``):
                shape = [num_layers, blocks_in_chunk * block_size, hidden]
      server  (``v1/multiprocess/modules/engine_driven_transfer.py``):
                shape = [num_layers, ctx.chunk_size, hidden]

    ``blocks_in_chunk`` is ``tokens_per_chunk // tokens_per_block`` and
    ``ctx.chunk_size`` is the chunk's token count, so the two are equal exactly
    when ``block_size == tokens_per_block``.  A ``[N, 1, unit_bytes]`` view --
    which is what ``page_region`` publishes, and what ATOM used to hand over --
    makes LMCache detect ``block_size = 1``.  Measured on K3 (2026-10-01,
    ATOM_ENGDRV_SHAPE_PROBE): detected ``(block_size=1, num_layers=29,
    hidden=884736)``, ``tokens_per_block=1536``, so the client computed
    ``shape[1] = 1`` and the server computed ``shape[1] = 1536``.  In
    ``engine_driven`` mode the server's number wins, and every stored object
    became 29 x 884736 x 1536 = 39,409,680,384 B -- the whole tier in one
    object instead of the correct 25,657,344 B chunk.  The tier then held 4
    objects, served 0 external hits, and the mode looked like it had no
    benefit, when in fact it had never stored a usable chunk.

    Splitting the extent restores agreement: detected ``block_size = 1536``,
    ``hidden = 576``, and both formulas give 29 x 1536 x 576 = 25,657,344 B.

    WHY only ``engine_driven`` (``expose_token_axis``).  The two transfer modes
    use the registered shape for different things, and only one of them is
    served by splitting it:

      engine_driven   the server uses the shape to SIZE the object, so the two
                      formulas must agree.
      lmcache_driven  the server's gather/scatter kernels use the shape to
                      ADDRESS the pages.  Telling them each block holds 1536
                      slots, while ATOM keeps handing them whole-block ids,
                      makes them index out of bounds.

    Measured 2026-10-01, two arms, both orderings (so the slot is not the
    explanation): with the split applied unconditionally, every lmcache_driven
    arm died ~6 min after a clean registration with "TimeoutError: RPC call to
    sample_tokens timed out" -- the worker hung on the GPU and never returned.
    engine_driven ran the same shape to completion (47.52 per-user tok/s p50,
    1056 objects of exactly 25,657,344 B, 14,510,592 external hits), which is
    what rules out the shape being wrong in itself.
    """

    if tokens_per_block <= 0:
        raise ValueError(f"{label}: tokens_per_block must be positive")
    if int(view.shape[1]) != 1 or tokens_per_block == 1:
        # Already carries a token axis (or there is only one token per block),
        # so LMCache's two formulas already agree.  Leave it alone: the view is
        # the backend's declaration of its own geometry, not ours to restate.
        return view
    unit_bytes = int(view.shape[-1])
    if unit_bytes % tokens_per_block:
        # Cannot express the token axis without reinterpreting bytes across
        # token boundaries.  Say so loudly rather than silently shipping the
        # shape that makes the two LMCache formulas disagree.
        logger.warning(
            "%s: %d bytes per block is not divisible by %d tokens per block; "
            "publishing the opaque [N, 1, unit_bytes] view.  LMCache will "
            "detect block_size=1, and in engine_driven transfer mode the "
            "server will size every object %d x too large.",
            label,
            unit_bytes,
            tokens_per_block,
            tokens_per_block,
        )
        return view
    return view.view(
        int(view.shape[0]), tokens_per_block, unit_bytes // tokens_per_block
    )


def _build_cache_views(
    transfer_tensors: Any,
    *,
    num_blocks: int,
    tokens_per_block: int,
    expose_token_axis: bool = False,
) -> _CacheViews:
    """Validate backend-published PAGE views without inspecting model internals."""

    if transfer_tensors is None:
        raise ValueError("lmcache_mp requires KVTransferTensors")

    stateful_fields = {
        "num_slots": getattr(transfer_tensors, "num_slots", 0),
        "slot_regions": getattr(transfer_tensors, "slot_regions", None),
        "swa_block_regions": getattr(transfer_tensors, "swa_block_regions", None),
        "staging_region": getattr(transfer_tensors, "staging_region", None),
        "gather_slot": getattr(transfer_tensors, "gather_slot", None),
        "scatter_slot": getattr(transfer_tensors, "scatter_slot", None),
        "expected_full_slot_region_count": getattr(
            transfer_tensors, "expected_full_slot_region_count", None
        ),
    }
    populated_stateful_fields = [
        name for name, value in stateful_fields.items() if value
    ]
    if populated_stateful_fields:
        raise NotImplementedError(
            "lmcache_mp supports PAGE-only layouts; stateful SLOT data was "
            f"published through {', '.join(populated_stateful_fields)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    indices_by_layout: dict[tuple[torch.dtype, tuple[int, ...]], list[int]] = {}
    bytes_per_block = 0
    for page in validate_page_views(transfer_tensors, num_blocks=num_blocks):
        index, view, region = page.index, page.view, page.region
        # LMCache receives these tensors as opaque PAGE storage, not numerical
        # values.  Publish a zero-copy byte view so every transfer path copies
        # the exact bit pattern.  This is especially important on ROCm, where
        # LMCache's Python raw-pointer fallback cannot express FP8 through the
        # CUDA array interface and would otherwise reconstruct the destination
        # as uint8 while keeping the staging object as FP8.
        byte_view = view.view(torch.uint8)
        if expose_token_axis:
            byte_view = _token_major_view(
                byte_view,
                tokens_per_block,
                label=f"lmcache_mp page plane {index}",
            )
        role = str(getattr(region, "semantic_role", None) or f"plane_{index}")
        tensors[f"page.{index}.{role}"] = byte_view
        layout = (byte_view.dtype, tuple(int(dim) for dim in byte_view.shape[1:]))
        indices_by_layout.setdefault(layout, []).append(index)
        bytes_per_block += page.unit_bytes

    recurrent = _build_recurrent_views(
        transfer_tensors,
        tensors=tensors,
        first_index=len(tensors),
        expose_token_axis=expose_token_axis,
    )

    return _CacheViews(
        tensors=tensors,
        layer_groups=tuple(tuple(indices) for indices in indices_by_layout.values()),
        bytes_per_block=bytes_per_block,
        recurrent=recurrent,
    )


def _build_recurrent_views(
    transfer_tensors: Any,
    *,
    tensors: dict[str, torch.Tensor],
    first_index: int,
    expose_token_axis: bool = False,
) -> tuple[_RecurrentViews, ...]:
    """Validate and add the recurrent groups to the flat registration order.

    They are appended, never interleaved, so a layout that grows or loses a
    recurrent group does not renumber the PAGE planes -- the plane order is the
    key space, and renumbering it would make every object already in the tier
    address the wrong bytes.
    """
    groups = tuple(getattr(transfer_tensors, "recurrent_page_groups", None) or ())
    if not groups:
        return ()

    built: list[_RecurrentViews] = []
    index = first_index
    for ordinal, group in enumerate(groups):
        pages = list(group.pages)
        if not pages:
            raise ValueError(
                f"lmcache_mp recurrent group {ordinal} published no planes"
            )
        num_blocks = int(group.num_blocks)
        tokens_per_block = int(group.tokens_per_block)
        if num_blocks <= 0 or tokens_per_block <= 0:
            raise ValueError(
                f"lmcache_mp recurrent group {ordinal} has num_blocks="
                f"{num_blocks} tokens_per_block={tokens_per_block}; both must "
                "be positive"
            )
        validated = _validate_views(
            [page.view for page in pages],
            [page.region for page in pages],
            num_blocks=num_blocks,
            block_size=None,
            label=f"recurrent[{ordinal}]",
            index_offset=index,
        )
        indices = []
        for page in validated:
            role = str(
                getattr(page.region, "semantic_role", None) or f"plane_{page.index}"
            )
            recurrent_view = page.view
            if expose_token_axis:
                recurrent_view = _token_major_view(
                    recurrent_view,
                    tokens_per_block,
                    label=f"lmcache_mp recurrent[{ordinal}] plane {page.index}",
                )
            tensors[f"recurrent.{ordinal}.{page.index}.{role}"] = recurrent_view
            indices.append(page.index)
        index += len(validated)
        built.append(
            _RecurrentViews(
                tensor_indices=tuple(indices),
                num_blocks=num_blocks,
                tokens_per_block=tokens_per_block,
                bytes_per_block=sum(page.unit_bytes for page in validated),
            )
        )
    return tuple(built)


__all__ = ["PageView", "validate_page_views"]
