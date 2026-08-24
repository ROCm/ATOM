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
    # How far the accumulated text should be cut, or -1 to leave it. Set by
    # the frontend stop-string check, and honoured by every consumer that
    # builds text. Which stop string matched is deliberately not reported:
    # `finish_reason` already says a stop sequence fired, and OpenAI's schema
    # has no field for the identity -- vLLM keeps it off its OpenAI endpoint
    # for the same reason.
    stop_truncate_to: int = -1
