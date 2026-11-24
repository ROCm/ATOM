# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Whether a deployment takes the shared mono LM head, from what needs no
FlyDSL: ``ATOM_MONO_ENABLE``, the arch, the TP size and the TP group's peer
memory. The runner asks it before it loads the launch (``head``), so a
deployment the head does not serve never imports its kernels;
``head.MonoLmHead.refusal`` adds what the launch's own shapes decide."""

from aiter.jit.utils.chip_info import get_gfx

from atom.config import get_current_atom_config
from atom.mono.runtime.deployment import peer_memory_refusals
from atom.utils import envs

# the architecture and TP sizes the launch is built and measured for
ARCH = "gfx950"
SERVED_TPS = (2, 4, 8)


def deployment_refusal(tp: int) -> str | None:
    """Why this deployment's TP group of ``tp`` does not take the launch,
    None where it does: the same on every rank of the group."""
    if not envs.ATOM_MONO_ENABLE:
        return "ATOM_MONO_ENABLE is off"
    if get_gfx() != ARCH:
        return f"built for {ARCH}"
    if tp not in SERVED_TPS:
        return f"TP {tp}: built for {SERVED_TPS}"
    for holds, why in peer_memory_refusals(get_current_atom_config()):
        if not holds:
            return why
    return None
