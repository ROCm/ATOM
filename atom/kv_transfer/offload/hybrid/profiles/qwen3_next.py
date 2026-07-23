# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next (GDN hybrid KV) offload profile: full_kv(BLOCK) + gdn_state(SLOT).

Qwen3-Next interleaves full-attention layers (paged KV — token-major, block
addressable) with Gated DeltaNet linear-attention layers (per-request recurrent
state: conv_state + temporal_state — no token span). The offload unit therefore
has two components:

* ``full_kv``   (BLOCK) — the full-attention layers' KV, keyed by block_table.
* ``gdn_state`` (SLOT)  — the GDN layers' recurrent state, one per-request slot.
                          temporal_state is a full-history accumulator, so it
                          MUST be snapshotted (cannot be recomputed from a tail).

MODEL-SIDE CONTRACT (not yet implemented — see gdn_attn.py):
  ``get_kv_transfer_tensors()`` for a Qwen3-Next runner must populate:
    * ``block_regions``  — full-attention layers' KV regions (as MLA/MHA does)
    * ``slot_regions``   — GDN conv+temporal state, one fixed-size slot per req
    * ``gather_slot`` / ``scatter_slot`` — snapshot the live GDN state slot into
      an offload pool slot before save / drain it back on load (mirrors DSV4's
      staging path, since the compute slot is overwritten by the next decode).
  Until that lands, this profile is exercised only by CPU tests with mock
  transfer tensors.
"""

from __future__ import annotations

from atom.kv_transfer.offload.hybrid.kv_bundle import BundleGeometry
from atom.kv_transfer.offload.hybrid.profiles.base import (
    ComponentSpec,
    HybridProfile,
    RegionKind,
    SaveCadence,
)


def _hf(config):
    return config.hf_config


def build_qwen3_next_geometry(config, *, world: int, rank: int) -> BundleGeometry:
    hf = _hf(config)
    block_size = int(getattr(config, "kv_cache_block_size", 0) or 0)
    fields = {
        "num_hidden_layers": int(getattr(hf, "num_hidden_layers", 0) or 0),
        "head_dim": int(getattr(hf, "head_dim", 0) or 0),
        "block_size": block_size,
        "kv_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
        # GDN state layout — must match for the SLOT bytes to be interpretable.
        "linear_num_key_heads": int(getattr(hf, "linear_num_key_heads", 0) or 0),
        "linear_num_value_heads": int(getattr(hf, "linear_num_value_heads", 0) or 0),
        "linear_key_head_dim": int(getattr(hf, "linear_key_head_dim", 0) or 0),
        "linear_value_head_dim": int(getattr(hf, "linear_value_head_dim", 0) or 0),
        "linear_conv_kernel_dim": int(getattr(hf, "linear_conv_kernel_dim", 0) or 0),
        # Which layers are linear vs full — affects both components.
        "layer_types": ",".join(str(t) for t in (getattr(hf, "layer_types", None) or [])),
    }
    return BundleGeometry.build(
        model_tag=str(getattr(config, "model", "qwen3-next")),
        tp_size=int(world),
        tp_rank=int(rank),
        fields=fields,
    )


def build_qwen3_next_profile(config, *, world: int, rank: int) -> HybridProfile:
    block_size = int(getattr(config, "kv_cache_block_size", 0) or 0)
    if block_size <= 0:
        raise ValueError("qwen3_next profile: kv_cache_block_size must be > 0")
    return HybridProfile(
        model_tag="qwen3-next",
        components=(
            ComponentSpec("full_kv", RegionKind.BLOCK),
            ComponentSpec("gdn_state", RegionKind.SLOT),
        ),
        # No 128-SWA constraint (unlike DSV4); checkpoints align to the KV block
        # so full-attention blocks are complete. GDN state can snapshot anywhere.
        cadence=SaveCadence(align=block_size, min_len=block_size),
        block_size=block_size,
        build_geometry=lambda: build_qwen3_next_geometry(config, world=world, rank=rank),
    )
