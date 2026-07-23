# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 offload profile: compressed(BLOCK) + swa(SWA) + csa_state(STAGING).

This is the hybrid-layout profile for DeepSeek-V4. It reproduces exactly the
component set, ordering, geometry fingerprint, and 128-aligned cadence that the
standalone DSV4 connector used, so a unit built via this profile is byte-
identical to the pre-refactor DSV4 offload unit.
"""

from __future__ import annotations

from atom.kv_transfer.offload.hybrid.kv_bundle import BundleGeometry
from atom.kv_transfer.offload.hybrid.profiles.base import (
    ComponentSpec,
    HybridProfile,
    RegionKind,
    SaveCadence,
)

DSV4_BLOCK_SIZE = 128  # DSV4 attention block_size (deepseek_v4_attn.py); constant
CHECKPOINT_ALIGN = 128  # SWA window / HCA compression block; boundaries align here


def build_dsv4_geometry(config, *, world: int, rank: int) -> BundleGeometry:
    hf = config.hf_config
    ratios = tuple(int(r) for r in getattr(hf, "compress_ratios", ()) or ())
    window = int(getattr(hf, "sliding_window", CHECKPOINT_ALIGN) or CHECKPOINT_ALIGN)
    fields = {
        "num_layers": int(getattr(hf, "num_hidden_layers", 0) or 0),
        "head_dim": int(getattr(hf, "head_dim", 0) or 0),
        "window_size": window,
        "block_size": DSV4_BLOCK_SIZE,
        "swa_block_size": DSV4_BLOCK_SIZE,
        "k1_csa": DSV4_BLOCK_SIZE // 4,
        "k2_hca": max(1, DSV4_BLOCK_SIZE // 128),
        "kv_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
        "compress_ratios": ",".join(str(r) for r in ratios),
    }
    return BundleGeometry.build(
        model_tag=str(getattr(config, "model", "deepseek-v4")),
        tp_size=int(world),
        tp_rank=int(rank),
        fields=fields,
    )


def build_dsv4_profile(config, *, world: int, rank: int) -> HybridProfile:
    return HybridProfile(
        model_tag="deepseek-v4",
        components=(
            ComponentSpec("compressed", RegionKind.BLOCK),
            ComponentSpec("swa", RegionKind.SWA),
            ComponentSpec("csa_state", RegionKind.STAGING),
        ),
        cadence=SaveCadence(align=CHECKPOINT_ALIGN, min_len=CHECKPOINT_ALIGN),
        block_size=DSV4_BLOCK_SIZE,
        build_geometry=lambda: build_dsv4_geometry(config, world=world, rank=rank),
    )
