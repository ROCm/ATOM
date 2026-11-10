# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Complete PAGE-only byte-codec geometry validation and construction."""

from __future__ import annotations

import torch

from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


def pure_page_codec(
    transfer_tensors: KVTransferTensors | None, num_blocks: int | None = None
) -> DSV4PageSlotCodec:
    """Construct the complete PAGE codec; reject unusable geometry/staging."""
    regions = getattr(transfer_tensors, "block_regions", None)
    if not regions:
        raise ValueError("PAGE offload requires non-empty block_regions")
    if getattr(transfer_tensors, "slot_regions", ()):
        raise ValueError("PAGE offload requires empty SLOT regions")
    if num_blocks is None:
        num_blocks = getattr(transfer_tensors, "num_blocks", None)
    if num_blocks is None or int(num_blocks) <= 0:
        raise ValueError("PAGE offload requires a positive num_blocks value")
    codec = DSV4PageSlotCodec(
        page_regions=regions,
        slot_regions=(),
        num_blocks=int(num_blocks),
        num_slots=0,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if not codec.has_fused_chunk_major_staging:
        raise RuntimeError(
            "PAGE offload requires the Triton fused chunk-major PAGE staging kernels"
        )
    return codec
