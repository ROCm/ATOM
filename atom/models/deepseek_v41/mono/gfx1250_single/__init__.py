# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek V4.1 decode as one FlyDSL launch a step on gfx1250 (TP4): the 40 target layers, the DSpark draft's three
stages and the step tail in a single persistent kernel.

``enabled()`` decides it before the model is built (``ATOM_MONO_SINGLE=1`` on a gfx1250 device), so the MoE layers can
hand their routed experts to ``planes`` before the loader shuffles them in place."""

import torch

from atom.utils import envs


def enabled() -> bool:
    if not envs.ATOM_MONO_SINGLE or not torch.cuda.is_available():
        return False
    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
            ":"
        )[0]
        == "gfx1250"
    )
