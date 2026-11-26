# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek V4.1 decode as one FlyDSL launch a step on gfx1250 (TP4): the 40 target layers, the DSpark draft's three
stages and the step tail in a single persistent kernel.

Switched like the gfx950 mono decode: on (``ATOM_MONO_ENABLE``, default 1) on a gfx1250 device for the deployment it is
written for (``config.config_refusal``)."""

import torch

from atom.models.deepseek_v41.mono.gfx1250.config import config_refusal
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.utils import envs


def enabled() -> bool:
    """``ATOM_MONO_ENABLE`` on a gfx1250 device."""
    if not envs.ATOM_MONO_ENABLE or not torch.cuda.is_available():
        return False
    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
            ":"
        )[0]
        == "gfx1250"
    )


def refusal(atom_config) -> str | None:
    """Why this deployment cannot run the single-launch step, or None."""
    return config_refusal(atom_config, Dims)
