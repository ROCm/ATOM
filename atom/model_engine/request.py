# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

# A finish reason that starts so says the server ended the request itself, and
# why after it: the client is answered an error, not a completion cut short.
SERVER_ABORT_REASON = "server_abort: "


def server_abort(finish_reason) -> str | None:
    """Why the server ended the request itself; None: it did not."""
    if isinstance(finish_reason, str) and finish_reason.startswith(SERVER_ABORT_REASON):
        return finish_reason[len(SERVER_ABORT_REASON) :]
    return None


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
