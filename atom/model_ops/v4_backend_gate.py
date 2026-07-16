# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 attention backend selector — gates the legacy vs. new backend per layer.

Env:
  ATOM_V4_BACKEND          legacy|new   (default legacy)
  ATOM_V4_BACKEND_LAYERS   csv int      empty=apply backend setting to all layers,
                                        otherwise only listed layer ids use the new backend

The new backend (`atom.model_ops.v4_attention_backend.V4AttentionBackend`)
removes per-seq Python dispatch and `.item()` syncs from the V4 forward path,
unblocking CUDAGraph capture. Legacy is kept available during migration so
each layer can be A/B verified for byte-equivalence via dump-bisect before
the legacy path is removed.

Usage in V4 model code:

    from atom.model_ops.v4_backend_gate import use_new_v4_backend
    if use_new_v4_backend(self.layer_id):
        # call backend
    else:
        # legacy per-seq loop
"""

from functools import lru_cache

from atom.utils import envs


@lru_cache(maxsize=1)
def _enabled_layers() -> frozenset[int] | None:
    """Parse ATOM_V4_BACKEND_LAYERS; None means "apply ATOM_V4_BACKEND to all layers"."""
    raw = envs.ATOM_V4_BACKEND_LAYERS
    if not raw:
        return None
    return frozenset(int(s.strip()) for s in raw.split(",") if s.strip())


def use_new_v4_backend(layer_id: int) -> bool:
    """Return True if this layer should route through V4AttentionBackend."""
    layers = _enabled_layers()
    if layers is not None:
        # Layer list overrides ATOM_V4_BACKEND: listed layers use new, others legacy.
        return layer_id in layers
    return envs.ATOM_V4_BACKEND == "new"
