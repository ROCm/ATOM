# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Wan2.1. So far only its video VAE, with every conv dispatched through one
conv3d entry point that prefers AITER's kernels; the DiT, text encoder and
pipeline are not ported yet, so nothing is registered in ``registry.py``."""

from atom.diffusion.models.wan.conv_dispatch import (
    CONV3D_BACKENDS,
    conv3d,
    conv3d_backend_override,
    conv3d_call_counts,
    reset_conv3d_call_counts,
    select_conv3d_backend,
)
from atom.diffusion.models.wan.vae import (
    WanVAE,
    WanVAEConfig,
    convert_original_wan_vae_state_dict,
    load_wan_vae,
)

__all__ = [
    "CONV3D_BACKENDS",
    "WanVAE",
    "WanVAEConfig",
    "conv3d",
    "conv3d_backend_override",
    "conv3d_call_counts",
    "convert_original_wan_vae_state_dict",
    "load_wan_vae",
    "reset_conv3d_call_counts",
    "select_conv3d_backend",
]
