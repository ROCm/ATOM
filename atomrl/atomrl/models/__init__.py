# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Draft model architectures for speculative decoding distillation."""

from __future__ import annotations

from atomrl.models.dflash import DFlashModel
from atomrl.models.eagle3 import Eagle3Model

__all__ = [
    "DFlashModel",
    "Eagle3Model",
]
