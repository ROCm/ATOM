# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Complete-PAGE codec construction shared by M3 offload connectors."""

from __future__ import annotations

from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


def build_m3_page_codec(
    transfer_tensors: KVTransferTensors | None, num_blocks: int | None = None
) -> DSV4PageSlotCodec:
    """Pack all published PAGE regions, including the NSA index and draft KV.

    M3's per-layer kv_caches includes target and draft K/V/scales, but only
    block_regions includes the NSA index cache. Preserve publication order;
    changes to that order or region set require a Store M3_CODEC_ID bump.
    """
    pages = getattr(transfer_tensors, "block_regions", ())
    if not pages or getattr(transfer_tensors, "slot_regions", ()):
        raise ValueError("M3 offload requires complete PAGE regions and no SLOT state")
    blocks = (
        num_blocks
        if num_blocks is not None
        else getattr(transfer_tensors, "num_blocks", None)
    )
    if blocks is None:
        raise ValueError("M3 offload PAGE regions require a num_blocks value")
    views = transfer_tensors.block_tensor_views
    if len(views) != len(pages):
        raise ValueError("M3 offload requires a tensor view for every PAGE region")
    codec = DSV4PageSlotCodec(
        page_regions=pages,
        slot_regions=(),
        num_blocks=blocks,
        num_slots=0,
        device=views[0].device,
    )
    if not codec.has_fused_chunk_major_staging:
        raise RuntimeError(
            "M3 offload requires the Triton fused chunk-major PAGE staging "
            "kernels (DSV4PageSlotCodec reported them unavailable)"
        )
    return codec
