# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 SP QKV prologue through the routed ``sp_qkv_fused`` provider.

Enablement is the aiter provider rail (``--aiter-provider-enable``); this
module only installs the attention sites and owns the provider warm-up, which
the aiter interface leaves to the framework.  With no provider enabled the
function returns None and the model keeps #3084's completely unfused path.
"""

import logging

logger = logging.getLogger("atom")


class GemmFusionSites:
    """The installed sites plus the tiny lifecycle #3084's runner calls.

    Plan/bucket/resources machinery is gone (the provider owns them); these
    hooks keep ModelRunner's existing prepare/before_step/clean_point/close
    ordering without a no-op object per feature.
    """

    def __init__(self, sites, config):
        self.sites = sites
        self.config = config

    def prepare(self, kv_cache_data):
        """The (re)allocated KV pool is now final: reset before any fused call.

        The provider binds address-stable KV/output pointers at first prepare;
        a dummy/compile-only run or an earlier pool must never be bound.  Drop
        the provider session and every site's per-M gate verdict, mark the sites
        KV-ready, then re-warm against the final pool.  Called from both the
        allocating and the IPC-import cache paths, before warmup/capture.
        """
        from atom.aiter_provider import shutdown_aiter_providers

        try:
            from .site import _sp_group

            group = _sp_group()
        except Exception:  # noqa: BLE001 - teardown is best effort
            group = None
        logger.info(
            "provider session: reset for final KV pool (%d sites); re-warming",
            len(self.sites),
        )
        shutdown_aiter_providers(self.config, group=group)
        for site in self.sites:
            site.mark_kv_ready()
        _warmup(self.config, self.sites)

    def before_step(self):
        pass

    def clean_point(self):
        pass

    def close(self):
        from atom.aiter_provider import shutdown_aiter_providers

        try:
            from .site import _sp_group

            group = _sp_group()
        except Exception:  # noqa: BLE001 - teardown is best effort
            group = None
        shutdown_aiter_providers(self.config, group=group)


def _warmup(config, sites):
    """Framework-owned provider warmup: the interface has no lifecycle hook."""
    from atom.aiter_provider import warmup_aiter_providers

    try:
        from .site import _sp_group

        group = _sp_group()
    except Exception:  # noqa: BLE001 - no aiter interface/device: provider declines
        group = None
    payload = None
    for site in sites:
        try:
            payload = site.warmup_payload()
            break
        except Exception:
            logger.warning(
                "gemm_fusion %s: provider warmup payload failed",
                site.name,
                exc_info=True,
            )
    warmed = payload is not None and warmup_aiter_providers(
        config, payload, group=group
    )
    if not warmed:
        logger.warning(
            "gemm_fusion: provider warmup did not succeed; every routed call "
            "will fall back to the non-fused path"
        )
    else:
        logger.info("provider session: created (warmup) for %d sites", len(sites))
    return warmed


def initialize_gemm_fusion(model, config):
    """After weight processing, before warmup and capture.

    Returns the installed sites, or None when the aiter provider rail is off
    (byte-identical to #3084 with gemm-fusion off).
    """
    if not getattr(config, "aiter_provider_enable", ()):
        return None
    from .site import install_minimax_m3

    sites = install_minimax_m3(model, config)
    if not sites:
        logger.warning("gemm_fusion: no MiniMax-M3 attention layer can be served")
        return None
    _warmup(config, sites)
    return GemmFusionSites(sites, config)
