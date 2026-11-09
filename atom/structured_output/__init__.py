# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight structured-output protocol; execution dependencies load on demand."""

from dataclasses import dataclass
from importlib import import_module


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


def is_structured(request):
    """A JSON Schema answer, strict tool arguments, or both constrain decoding."""
    return (
        getattr(request, "json_schema", None) is not None
        or getattr(request, "structured_tool_format", None) is not None
    )


__all__ = [
    "StructuredOutput",
    "StructuredRequest",
    "is_structured",
    "prepare_worker_grammar",
    "structured_stop_token_ids",
    "tokenizer_fingerprint",
]


def __getattr__(name):
    if name in __all__ and name != "StructuredRequest":
        return getattr(import_module(".manager", __name__), name)
    raise AttributeError(name)
