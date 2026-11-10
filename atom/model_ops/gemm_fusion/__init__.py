# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 SP QKV prologue through the routed ``sp_qkv_fused`` provider.

Enablement is the aiter provider rail (``--aiter-provider-enable``); this
module only installs the attention sites and owns the provider warm-up, which
the aiter interface leaves to the framework.  With no provider enabled the
function returns None and the model keeps #3084's completely unfused path.
"""

import logging

import torch

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

    def prepare_capture(self, local_m):
        """Pre-bind the provider for one capture M class before capture starts.

        The capture trace must not allocate a new provider binding: build the
        capture-time payload (final layer KV state, weight/norm/cos/sin objects,
        the site's shared output set for this M) and call the provider's
        prepare-only API once per (site, mode, M). Missing API/False leaves that
        cell on the non-fused path in capture (the eager path is unchanged).
        """
        from atom.aiter_provider import prepare_aiter_providers
        from atom.utils.forward_context import get_forward_context

        try:
            fwd_ctx = get_forward_context()
        except Exception:  # noqa: BLE001 - no capture context: nothing to do
            return
        try:
            device = torch.cuda.current_device()
        except Exception:  # noqa: BLE001 - CPU/test process
            device = None
        for site in self.sites:
            if not site.provider_enabled or not site.kv_ready:
                continue
            key = (site.mode, int(local_m))
            if key in site._capture_prepared or key in site._capture_declined:
                continue
            try:
                gate_args, gate_kwargs = site.capture_gate_args(local_m, fwd_ctx)
                served = site._serves(
                    local_m, gate_args, gate_kwargs, capture_gate=True
                )
                if not served:
                    site._capture_declined.add(key)
                    logger.info(
                        "provider capture %s mode=%s m=%d: fallback " "(gate declined)",
                        site.name,
                        site.mode,
                        local_m,
                    )
                    continue
                payload = site.capture_prepare_payload(local_m, fwd_ctx)
                ok = prepare_aiter_providers(
                    self.config,
                    payload,
                    group=site._group_value(),
                    device=device,
                )
            except Exception:
                logger.warning(
                    "provider capture %s mode=%s m=%d: prepare failed",
                    site.name,
                    site.mode,
                    local_m,
                    exc_info=True,
                )
                ok = False
            if ok:
                site._capture_prepared.add(key)
                logger.info(
                    "provider capture %s mode=%s m=%d: fused (prepare)",
                    site.name,
                    site.mode,
                    local_m,
                )
            else:
                site._capture_declined.add(key)
                logger.warning(
                    "provider capture %s mode=%s m=%d: fallback (prepare unavailable)",
                    site.name,
                    site.mode,
                    local_m,
                )

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
