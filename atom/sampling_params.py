# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable
from dataclasses import dataclass, replace

UNCONSTRAINED = -1
"""The channel state of a sequence whose request requires no channel."""


@dataclass(frozen=True)
class RequiredChannel:
    """A channel the output must open before it may end.

    A tool-call format builds one for ``tool_choice="required"``. Kimi-K3
    closes its response channel with ``after`` and then either opens the tools
    channel or closes the message; a required call must do the former. The
    sampler holds the output to it on the GPU, on the target logits:

    - until ``opening`` is complete, no id in ``end_ids`` can be sampled;
    - once the output ends with ``after``, the next ``len(opening)`` tokens
      are ``opening``.

    A sequence's state is how far its output has got. ``0 .. len(after) - 1``
    is how much of ``after`` it currently ends with; ``len(after) + i`` means
    the next token is ``opening[i]``. Once ``opening`` is written the channel
    is open, nothing is constrained any more, and the state becomes
    ``UNCONSTRAINED``: the sequence drops out of its batch's channel, so a
    batch whose requests have all opened theirs costs nothing. The state
    counts generated tokens only, so a sequence starts at 0 whatever its
    prompt ends with.
    """

    after: tuple[int, ...]
    opening: tuple[int, ...]
    end_ids: tuple[int, ...]

    def __post_init__(self):
        if not self.after or not self.opening:
            raise ValueError("a required channel needs both `after` and `opening`")
        # On a mismatch `step` restarts the match at 1 or 0. That is the whole
        # KMP failure function only while the first token of `after` does not
        # recur in it.
        if self.after[0] in self.after[1:]:
            raise ValueError(
                f"`after` may not repeat its first token, got {self.after}"
            )

    @property
    def open_state(self) -> int:
        """Where writing ``opening`` would take the state; ``step`` returns
        ``UNCONSTRAINED`` instead, so no sequence is ever in it."""
        return len(self.after) + len(self.opening)

    @property
    def table(self) -> tuple[int, ...]:
        """``after + opening + end_ids``, the layout the GPU reads.

        Below ``open_state`` a state indexes the token it waits for (while
        matching ``after``) or forces (while opening)."""
        return self.after + self.opening + self.end_ids

    def step(self, state: int, token_id: int) -> int:
        """The state after ``token_id``. The GPU kernels make the same moves."""
        if state < 0:
            return state
        if state < len(self.after):
            if token_id == self.after[state]:
                return state + 1
            return 1 if token_id == self.after[0] else 0
        # Forced tokens advance the state whatever they are: a row that saw a
        # different one is a rejected draft, whose state nobody reads.
        state += 1
        return state if state < self.open_state else UNCONSTRAINED

    def advance(self, state: int, token_ids: Iterable[int]) -> int:
        for token_id in token_ids:
            state = self.step(state, int(token_id))
        return state

    def ending_on(self, stop_ids: Iterable[int | None]) -> "RequiredChannel":
        """This channel, also holding back ``stop_ids``.

        The format names the ids that end its own message; the engine may end
        a sequence on more (its EOS and ``stop_token_ids``), and any it does
        not hold back would end the message before the channel opens."""
        end_ids = set(self.end_ids)
        end_ids.update(i for i in stop_ids if i is not None and i >= 0)
        if len(end_ids) == len(self.end_ids):
            return self
        return replace(self, end_ids=tuple(sorted(end_ids)))


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
    # Set by the server for `tool_choice="required"` when the tool-call format
    # defines one (Kimi-K3 only); None leaves the output unconstrained.
    required_channel: RequiredChannel | None = None

    def __post_init__(self):
        # 0 also means disabled, as in vLLM/SGLang; MLPerf Endpoints workloads send it
        if self.top_k == 0:
            self.top_k = -1
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or 0 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")
