# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step-gap hooks Kimi-K3's decode installer applies once, after the
weights are loaded and before the first warmup: every rank reaches it at the
same point, outside any graph capture."""


def install(model, atom_config, drafter) -> None:
    """``ATOM_KIMI_MONO_STEP_LEAN``: the drafter's block metadata in one
    launch (``block_meta``), where the mirror covers the drafter.
    ``ATOM_KIMI_MONO_EMBED_LOCAL``: the whole embedding table bound on every
    TP rank (``embed_local``; collective)."""
    from atom.utils import envs

    if envs.ATOM_KIMI_MONO_STEP_LEAN and drafter is not None:
        from atom.models.kimi_k3.step_gap import block_meta

        block_meta.install(drafter)
    if envs.ATOM_KIMI_MONO_EMBED_LOCAL:
        from atom.models.kimi_k3.step_gap import embed_local

        embed_local.install(model)
