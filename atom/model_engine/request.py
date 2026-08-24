# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    request_id: int
    output_tokens: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    kv_transfer_params_output: Optional[Dict[str, Any]] = None
    num_cached_tokens: int = 0
    # Which stop string ended the request, for the client to read back. Only
    # ever set by the frontend stop-string check -- the engine core decides
    # token-level stops and reports those through `finish_reason` alone.
    stop_reason: Optional[str] = None
    # How far the accumulated text should be cut, or -1 to leave it. Set with
    # `stop_reason`, and honoured by every consumer that builds text.
    stop_truncate_to: int = -1
