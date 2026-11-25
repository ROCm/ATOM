# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight structured-output protocol; execution dependencies load on demand."""

import logging
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec


@dataclass(frozen=True)
class StructuredRequest:
    """One-time grammar configuration followed by acknowledged token deltas."""

    schema: tuple[str, str | None, str] | None
    start: int
    tokens: tuple[int, ...]

    @classmethod
    def capture(cls, seq):
        end = len(seq.output_tokens) - seq.num_placeholder_tokens
        start = seq.structured_cursor
        if end < start:
            raise ValueError("Structured committed-token cursor moved backwards")
        schema = (
            None
            if seq.structured_started
            else (
                seq.json_schema,
                seq.structured_tool_format,
                seq.structured_reasoning_end,
            )
        )
        update = cls(schema, start, tuple(seq.output_tokens[start:end]))
        seq.structured_cursor = end
        seq.structured_started = True
        return update


# The llguidance release the backend is written against, as pinned in
# pyproject.toml's dependencies (a test keeps the two equal).
LLGUIDANCE_VERSION = "1.9.1"

UNAVAILABLE = (
    f"structured output needs llguidance=={LLGUIDANCE_VERSION} and google-re2, "
    "which ATOM installs; this environment is missing one or has another llguidance "
    "(the vLLM image pins <1.8)"
)


@cache
def available():
    """Whether llguidance (the release the backend is written against),
    jsonschema and RE2 are installed."""
    if not all(
        find_spec(name) is not None for name in ("llguidance", "jsonschema", "re2")
    ):
        return False
    try:
        installed = version("llguidance")
    except PackageNotFoundError:
        installed = "without package metadata"
    if installed != LLGUIDANCE_VERSION:
        # The vLLM image pins llguidance <1.8 for vLLM's own backend.
        logging.getLogger("atom").warning(
            "llguidance %s is installed; ATOM's structured output needs %s and is "
            "off",
            installed,
            LLGUIDANCE_VERSION,
        )
        return False
    return True


def is_structured(request):
    """A JSON Schema answer, strict tool arguments, or both constrain decoding."""
    return (
        getattr(request, "json_schema", None) is not None
        or getattr(request, "structured_tool_format", None) is not None
    )


__all__ = [
    "LLGUIDANCE_VERSION",
    "UNAVAILABLE",
    "StructuredOutput",
    "StructuredRequest",
    "available",
    "is_structured",
    "prepare_worker_grammar",
    "structured_stop_token_ids",
    "tokenizer_fingerprint",
]


def __getattr__(name):
    if name in __all__ and name not in ("StructuredRequest", "available"):
        return getattr(import_module(".manager", __name__), name)
    raise AttributeError(name)
