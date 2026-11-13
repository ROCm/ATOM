# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 BF16 sparse decode through plugin_poc's FlyDSL kernel.

The kernel is imported straight from the plugin_poc package
(``ops.sparse_mla.runtime``), not through aiter: aiter has no contract for this
op. So nothing here goes through ``aiter.plugin`` -- no default-deny
``configure()``, no ``AITER_PLUGINS=0``, no decision records, no versioned
contract. ``ATOM_V4_DECODE_FLYDSL`` is the only switch, and ATOM has to track
the package's ``covers_v4_decode`` / ``sparse_attn_v4_paged_decode`` API itself.
"""

import logging

import torch

logger = logging.getLogger("atom")

# The runtime module, or False once importing it failed (warned once).
_RUNTIME = None
# Shapes that have run outside graph capture in this process. FlyDSL compiles on
# first launch and a capture cannot compile, so capture serves only these.
_WARM: set[tuple] = set()


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        try:
            from ops.sparse_mla import runtime
        except Exception:  # absent or broken package: keep Triton
            logger.warning(
                "ATOM_V4_DECODE_FLYDSL=1 but plugin_poc's ops.sparse_mla is not "
                "importable; V4 BF16 decode keeps the Triton path",
                exc_info=True,
            )
            _RUNTIME = False
        else:
            _RUNTIME = runtime
    return _RUNTIME or None


def _key(q, unified_kv, kv_indices, softmax_scale) -> tuple:
    """What decides the compiled kernels a call uses: the runtime picks its tile
    and split count from these host-side sizes."""
    return (
        q.device.index,
        str(unified_kv.dtype),
        tuple(unified_kv.shape[1:]),
        int(q.shape[0]),
        int(q.shape[1]),
        int(kv_indices.numel()),
        float(softmax_scale),
    )


def flydsl_paged_decode(
    q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, kv_scales=None
):
    """The FlyDSL result for a call it covers, or None to keep ATOM's path."""
    runtime = _runtime()
    if runtime is None:
        return None
    if not runtime.covers_v4_decode(
        q, unified_kv, kv_indices, kv_indptr, None, kv_scales
    ):
        return None
    key = _key(q, unified_kv, kv_indices, softmax_scale)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _WARM:
        return None
    out = runtime.sparse_attn_v4_paged_decode(
        q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale
    )
    if not capturing:
        _WARM.add(key)
    return out
