# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    request_id: int
    output_tokens: list[int]
    finished: bool
    finish_reason: str | None = None
    kv_transfer_params_output: dict[str, Any] | None = None
    num_cached_tokens: int = 0
    # `enable_return_routed_experts` only, and only on the final output (None
    # on every streamed one): read-only int16 [seq_len - 1, num_hidden_layers,
    # top_k], -1 on dense layers. Surfaced through `generate()`'s dict, not
    # the OpenAI HTTP response.
    routed_experts: np.ndarray | None = None
