# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    request_id: int  # Sequence ID
    output_tokens: List[int]  # Newly generated tokens since last callback
    finished: bool  # Whether the sequence is finished
    finish_reason: Optional[str] = (
        None  # Reason for finishing (eos, max_tokens, stop_sequence, etc.)
    )
