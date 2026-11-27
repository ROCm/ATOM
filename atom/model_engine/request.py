# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

# `kv_transfer_params_output` key of the token-less, unfinished output that
# tells a P/D early-handoff prefill's front end its request was scheduled
# (ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE).
PD_EARLY_SCHEDULED_KEY = "pd_early_scheduled"


@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    request_id: int
    output_tokens: list[int]
    finished: bool
    finish_reason: str | None = None
    kv_transfer_params_output: dict[str, Any] | None = None
    num_cached_tokens: int = 0
    logprobs: list[Any] | None = None
