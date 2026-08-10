# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = -1
DEFAULT_TOP_P = 1.0


class _GenerationConfig(Protocol):
    def to_diff_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SamplingDefaults:
    """Effective model defaults for sampling-related request fields."""

    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P

    @classmethod
    def from_generation_config(
        cls, generation_config: _GenerationConfig | None
    ) -> "SamplingDefaults":
        """Apply non-default values from a model ``generation_config.json``.

        ``GenerationConfig`` materializes Transformers defaults even when a
        field is absent from the model file. Its diff excludes those implicit
        values, notably ``top_k=50``, so ATOM's neutral defaults remain intact.
        """
        if generation_config is None:
            return cls()

        config_diff = generation_config.to_diff_dict()
        values = {
            name: config_diff[name]
            for name in ("temperature", "top_k", "top_p")
            if config_diff.get(name) is not None
        }
        return cls(**values)

    def with_overrides(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> "SamplingDefaults":
        """Return defaults overridden by values explicitly sent by a client."""
        return type(self)(
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            top_p=self.top_p if top_p is None else top_p,
        )


@dataclass
class SamplingParams:
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K  # -1 means disabled (keep all tokens)
    top_p: float = DEFAULT_TOP_P  # 1.0 means disabled (keep all tokens)
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

    def __post_init__(self):
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")
