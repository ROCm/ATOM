# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which V4.1 forwards take the mono path: ``install_mono_decode`` (``models.registry.mono_decode_installers``) wraps
the loaded model with the mono decode ``selection`` picks for the deployment (gfx1250's single launch; on gfx950 the
TileRT-path step, the hi monokernels or ATOM's own mono decode, ``gfx950.k1k2.dispatch``), or leaves it unwrapped.
"""

from torch import nn

from atom.models.deepseek_v41.mono import selection

# ATOM's own mono decode and its runners, imported with this module as before the split
from atom.models.deepseek_v41.mono.gfx950.k1k2 import dispatch as _k1k2  # noqa: F401


def install_mono_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    """``model`` wrapped for mono decode (and ``drafter``'s DSpark draft model
    for its backbone), or ``model`` itself when this deployment is one mono
    does not serve (``selection.install``)."""
    return selection.install(model, atom_config, drafter, metadata_builder)
