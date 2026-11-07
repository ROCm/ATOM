# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared replay capability and PAGE-only byte-codec construction."""

from __future__ import annotations

from typing import Any

import torch

from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


def replays_state(transfer_tensors: Any) -> bool:
    """Whether the backend rebuilds its per-request state at every hit.

    A `replay` state transfer keeps no checkpoint: a prefix hit resumes with a
    fresh state slot and the forward recomputes the hash block below it
    (`BlockManager.replay_start`), so a PAGE-only load is resumable as is.
    """

    builder = getattr(transfer_tensors, "state_backend", None)
    state_transfer = getattr(builder, "state_transfer", None)
    return callable(state_transfer) and bool(
        getattr(state_transfer(), "replays", False)
    )


def pure_page_codec(regions, num_blocks):
    """Construct the complete PAGE codec; reject unusable geometry/staging."""
    if not regions:
        raise ValueError("PAGE offload requires non-empty block_regions")
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
