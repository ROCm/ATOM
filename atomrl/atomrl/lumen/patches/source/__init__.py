# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""On-disk Megatron source patches (PatchPhase.SOURCE)."""

from atomrl.lumen.patches.source import dsv4 as dsv4
from atomrl.lumen.patches.source import llama as llama
from atomrl.lumen.patches.source import (
    llama_lora as llama_lora,
)
from atomrl.lumen.patches.source import (
    rocm as rocm,
)

__all__ = ["dsv4", "llama", "llama_lora", "rocm"]
