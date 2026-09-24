# SPDX-License-Identifier: MIT
"""Refresh Flash PLE after hybrid GDN out_graph on decode CUDA-graph replay.

HybridLinearAttnBackend runs ``[full_attn, linear]`` in that order. Flash QSA
prepare hooks the full-attn child, which is *before* GDN copies
``mamba_cache_indices`` into the static buffer. PLE refresh must run after
both children so replay does not keep capture-time state indices.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_flash_decode_graph_ple_patch() -> None:
    HybridLinearAttnBackend = None
    for mod_name in (
        "sglang.srt.layers.attention.hybrid_linear_attn_backend",
        "sglang.srt.layers.attention.hybrid_attn_backend",
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            HybridLinearAttnBackend = getattr(mod, "HybridLinearAttnBackend", None)
            if HybridLinearAttnBackend is not None:
                break
        except (ImportError, AttributeError) as exc:
            logger.debug("HybridLinearAttnBackend import failed (%s): %s", mod_name, exc)
            continue
    if HybridLinearAttnBackend is None:
        logger.debug("HybridLinearAttnBackend unavailable; skip Flash PLE patch")
        return

    if getattr(HybridLinearAttnBackend, "_atom_flash_ple_out_graph_patch", False):
        return

    original = HybridLinearAttnBackend.init_forward_metadata_out_graph

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        original(self, forward_batch, in_capture=in_capture)
        try:
            from atom.plugin.sglang.qwen3_8_flash_next_bridge import (
                refresh_flash_decode_graph_ple,
            )

            refresh_flash_decode_graph_ple(forward_batch, in_capture=in_capture)
        except Exception as exc:
            logger.warning(
                "refresh_flash_decode_graph_ple failed (in_capture=%s): %s",
                in_capture,
                exc,
                exc_info=True,
            )

    HybridLinearAttnBackend.init_forward_metadata_out_graph = (
        init_forward_metadata_out_graph
    )
    HybridLinearAttnBackend._atom_flash_ple_out_graph_patch = True
    logger.info(
        "Patched HybridLinearAttnBackend out_graph to refresh Flash PLE after GDN"
    )
