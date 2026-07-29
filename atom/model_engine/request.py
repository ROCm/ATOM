# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any


@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    request_id: int
    output_tokens: list[int]
    finished: bool
    finish_reason: str | None = None
    kv_transfer_params_output: dict[str, Any] | None = None
    num_cached_tokens: int = 0
