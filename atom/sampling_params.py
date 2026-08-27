# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass

#: Upper bound on ``temperature`` the OpenAI-compatible API accepts. Enforced by
#: the API layer (``entrypoints/openai/chat_request.py``), not here: the engine
#: stays permissive so offline callers keep working.
MAX_TEMPERATURE = 2.0


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1  # -1 means disabled (keep all tokens)
    top_p: float = 1.0  # 1.0 means disabled (keep all tokens)
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_strings: list[str] | None = None
    # Number of independently sampled completions to return for a single
    # prompt. n == 1 preserves the historical single-sequence behavior.
    # n > 1 causes the engine to fan out N sibling sequences sharing the
    # same prompt; each uses independent random noise at the sampler so
    # outputs diverge when temperature > 0.
    n: int = 1
    logprobs: bool | int | None = None
    # Derives each position's random draw from (seed, position) instead of the
    # engine's shared noise. Determinism is best-effort: identical logits are
    # also required, and batched decode is not bitwise reproducible.
    seed: int | None = None

    def __post_init__(self):
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.seed is not None:
            # bool is an int subclass and would silently seed with 0 or 1.
            if not isinstance(self.seed, int) or isinstance(self.seed, bool):
                raise ValueError("seed must be an integer or None")
            # torch.Generator.manual_seed takes a signed 64-bit value.
            if not -(2**63) <= self.seed < 2**63:
                raise ValueError("seed must fit in a signed 64-bit integer")
