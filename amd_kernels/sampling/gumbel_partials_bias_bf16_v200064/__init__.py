# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gumbel block partials of a DSpark Markov draft step (base + bias rows, bf16 cache row, 196 blocks of 1024)."""

def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


from .host import blocks_per_wg, gumbel_partials_bias, warmup_gumbel_partials_bias
from .reference import NB, V

__all__ = ["NB", "V", "available", "blocks_per_wg", "gumbel_partials_bias", "warmup_gumbel_partials_bias"]
