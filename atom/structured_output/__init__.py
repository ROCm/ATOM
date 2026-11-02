# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Engine-level structured output; optional backends load only when requested."""

from .manager import (
    StructuredOutput,
    prepare_worker_grammar,
    structured_stop_token_ids,
    tokenizer_fingerprint,
)

__all__ = [
    "StructuredOutput",
    "prepare_worker_grammar",
    "structured_stop_token_ids",
    "tokenizer_fingerprint",
]
