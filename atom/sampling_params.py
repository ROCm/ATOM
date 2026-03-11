# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1  # -1 means disabled (keep all tokens)
    top_p: float = 1.0  # 1.0 means disabled (keep all tokens)
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_strings: Optional[list[str]] = None

    def __post_init__(self):
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
