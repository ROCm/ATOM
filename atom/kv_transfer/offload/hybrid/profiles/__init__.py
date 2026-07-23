# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Offload *profiles* for the hybrid (opaque terminal-unit) layout family.

A profile declares, for one model, the ordered list of unit components and the
region kind each is sourced from (block / swa / slot / staging), plus the save
cadence and geometry fingerprint. Adding a new offload target (e.g. Qwen3-Next
hybrid KV) means writing a profile here — not a new connector or unit format.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")


def select_profile(config, *, world: int, rank: int):
    """Pick the hybrid-family profile for *config* (worker + scheduler agree).

    Discriminates on ``hf_config`` alone (same signal the dispatch shell uses):
    ``compress_ratios`` => DeepSeek-V4; ``linear_num_key_heads`` (GDN) =>
    Qwen3-Next. An explicit ``kv_transfer_config["offload_profile"]`` overrides.
    """
    kvc = getattr(config, "kv_transfer_config", {}) or {}
    hf = getattr(config, "hf_config", None)
    override = kvc.get("offload_profile")

    def _dsv4():
        from atom.kv_transfer.offload.hybrid.profiles.dsv4 import build_dsv4_profile

        return build_dsv4_profile(config, world=world, rank=rank)

    def _qwen3_next():
        from atom.kv_transfer.offload.hybrid.profiles.qwen3_next import (
            build_qwen3_next_profile,
        )

        return build_qwen3_next_profile(config, world=world, rank=rank)

    builders = {"deepseek-v4": _dsv4, "qwen3-next": _qwen3_next}
    if override is not None:
        if override in builders:
            return builders[override]()
        logger.warning(
            "offload: unknown offload_profile=%r; falling back to auto", override
        )

    if hf is not None and (getattr(hf, "compress_ratios", None)):
        return _dsv4()
    if hf is not None and getattr(hf, "linear_num_key_heads", None):
        return _qwen3_next()
    raise ValueError(
        "offload: no hybrid profile matches this model "
        "(need hf_config.compress_ratios for DeepSeek-V4 or "
        "hf_config.linear_num_key_heads for Qwen3-Next); "
        "set kv_transfer_config['offload_profile'] to override"
    )
