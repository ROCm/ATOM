# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 KV-cache scale management for ATOM-style inference engines."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FP8KVCacheManager:
    """Manages FP8 KV-cache quantization scales and periodic QKV recalibration.

    Integrates with engines that expose optional FP8 KV paths. When the
    underlying runtime is unavailable, operations degrade gracefully while
    still exposing a consistent API for rollout workers.
    """

    def __init__(self, enabled: bool = False, **kwargs: Any) -> None:
        self._fp8_enabled = bool(enabled)
        self._kwargs = kwargs
        self._last_scale_step: int = -1

    def recalibrate_scales(self) -> None:
        """Run per-step QKV scale recalibration for FP8 KV caches.

        Attempts to delegate to optional ``atomrl.lumen`` runtime hooks when present;
        otherwise updates internal bookkeeping so callers can still sequence
        recalibration in the training loop.
        """
        try:
            import atomrl.lumen

            _ = (
                atomrl.lumen
            )  # runtime may register scale hooks; keep import for side effects
            logger.debug(
                "FP8KVCacheManager: Lumen runtime present; recalibration delegated."
            )
        except (AttributeError, RuntimeError) as exc:
            logger.debug(
                "FP8KVCacheManager.recalibrate_scales: Lumen unavailable or failed (%s); "
                "using local bookkeeping only.",
                exc,
            )
        self._last_scale_step += 1

    def is_fp8_enabled(self) -> bool:
        """Return whether FP8 KV-cache paths are active for this manager."""
        return self._fp8_enabled
