# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1  # -1 (or 0) means disabled (keep all tokens)
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
    top_logprobs: int = 0
    json_schema: str | None = None
    structured_tool_format: str | None = None
    # Generated reasoning tokens are unconstrained until this marker closes.
    structured_reasoning_end: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.top_logprobs) is not int or not 0 <= self.top_logprobs <= 20:
            raise ValueError("top_logprobs must be an integer between 0 and 20")
        if self.top_logprobs and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs")
        # 0 also means disabled, as in vLLM/SGLang; MLPerf Endpoints workloads send it
        if self.top_k == 0:
            self.top_k = -1
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or 0 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")
