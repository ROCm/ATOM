# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1  # -1 means disabled (keep all tokens)
    top_p: float = 1.0  # 1.0 means disabled (keep all tokens)
    max_tokens: int = 64
    # Minimum number of visible generation steps before EOS/single-token stop
    # tokens may be sampled. Matches vLLM/OpenAI-compatible serving semantics
    # and prevents a stochastic first-token EOS from becoming an empty reply.
    min_tokens: int = 0
    ignore_eos: bool = False
    # Stop strings, matched on the *detokenized* text by whoever holds the
    # tokenizer -- see `atom.model_engine.stop_strings`. Not encoded to token
    # ids: a client's spelling of a stop string need not tokenize the way the
    # model emits it, and when it does not, token matching just never fires.
    stop_strings: list[str] | None = None
    # Token ids that end the request, on top of the server's own
    # `config.stop_token_ids`. Decided in the scheduler, which needs no
    # tokenizer to compare ids.
    stop_token_ids: list[int] | None = None
    # Whether a matched stop string stays in the returned text. False is what
    # OpenAI, vLLM and TGI all default to.
    include_stop_str_in_output: bool = False
    # Number of independently sampled completions to return for a single
    # prompt. n == 1 preserves the historical single-sequence behavior.
    # n > 1 causes the engine to fan out N sibling sequences sharing the
    # same prompt; each uses independent random noise at the sampler so
    # outputs diverge when temperature > 0.
    n: int = 1
    logprobs: bool | int | None = None

    def __post_init__(self):
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        # A floor of 0 is no floor at all, so it contradicts no cap -- not
        # even the degenerate `max_tokens <= 0` that asks for no output. Only
        # a real floor is measured against the ceiling.
        if self.min_tokens > 0 and self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.stop_token_ids is not None and any(
            not isinstance(t, int) or isinstance(t, bool) or t < 0
            for t in self.stop_token_ids
        ):
            raise ValueError("stop_token_ids must be non-negative ints")
        if isinstance(self.stop_strings, str):
            # A bare string is a single stop, not an iterable of one-character
            # ones. Normalizing here means no consumer has to ask.
            self.stop_strings = [self.stop_strings]
