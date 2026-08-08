# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Distributed primitives specific to diffusion (Ulysses sequence parallelism)."""

from atom.diffusion.distributed.ulysses import UlyssesGroup

__all__ = ["UlyssesGroup"]
