# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Complete PAGE-only byte-codec geometry validation and construction."""

from __future__ import annotations

from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.kv_transfer.offload.capabilities import replays_state
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


def pure_page_codec(
    transfer_tensors: KVTransferTensors | None, num_blocks: int | None = None
) -> DSV4PageSlotCodec:
    """Construct the complete PAGE codec; reject unusable geometry/staging.

    M3 and V4.1 replay both store PAGE bytes and no SLOT state, so they share
    this constructor. The device comes from the published tensor views: a
    region without one has no device to stage on.
    """
    regions = getattr(transfer_tensors, "block_regions", None)
    if not regions:
        raise ValueError("PAGE offload requires non-empty block_regions")
    if getattr(transfer_tensors, "slot_regions", ()):
        raise ValueError("PAGE offload requires empty SLOT regions")
    return _page_codec(transfer_tensors, num_blocks)


def replay_page_codec(
    transfer_tensors: KVTransferTensors | None, num_blocks: int | None = None
) -> DSV4PageSlotCodec:
    """Encode PAGEs while the backend rebuilds state after a prefix restore.

    A composite P/D connector may also publish the live request SLOT for its
    remote consumer. That SLOT belongs to P/D, not to this reusable prefix.
    Select PAGEs only after verifying the backend's explicit replay contract;
    leave the original descriptor intact for the other connector.
    """
    if not replays_state(transfer_tensors):
        raise ValueError("PAGE replay offload requires the state replay contract")
    if (
        transfer_tensors.paged_state_checkpoint_spec is not None
        or transfer_tensors.execute_paged_state_copies is not None
        or getattr(transfer_tensors.state_backend.state_transfer(), "copies", False)
    ):
        raise ValueError("PAGE replay offload cannot carry native state checkpoints")
    return _page_codec(transfer_tensors, num_blocks)


def _page_codec(
    transfer_tensors: KVTransferTensors | None, num_blocks: int | None
) -> DSV4PageSlotCodec:
    regions = getattr(transfer_tensors, "block_regions", None)
    if not regions:
        raise ValueError("PAGE offload requires non-empty block_regions")
    if num_blocks is None:
        num_blocks = getattr(transfer_tensors, "num_blocks", None)
    if num_blocks is None or int(num_blocks) <= 0:
        raise ValueError("PAGE offload requires a positive num_blocks value")
    views = getattr(transfer_tensors, "block_tensor_views", ())
    if len(views) != len(regions):
        raise ValueError("PAGE offload requires a tensor view for every PAGE region")
    codec = DSV4PageSlotCodec(
        page_regions=regions,
        slot_regions=(),
        num_blocks=int(num_blocks),
        num_slots=0,
        device=views[0].device,
    )
    if not codec.has_fused_chunk_major_staging:
        raise RuntimeError(
            "PAGE offload requires the Triton fused chunk-major PAGE staging kernels"
        )
    return codec
