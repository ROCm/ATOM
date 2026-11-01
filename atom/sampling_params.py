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

    - until the channel is open, no id in ``end_ids`` can be sampled;
    - once the output ends with ``after``, the next ``len(opening)`` tokens
      are ``opening``.

    ``opening[:opens_at]`` is what opens the channel itself (K3:
    ``<|open|>tools<|sep|>``; the rest of ``opening`` starts the call). The
    model writing it on its own, without ``after`` first, opens the channel
    too. Otherwise a model that took any other way into the channel would
    have its end ids banned until ``max_tokens``.

    A sequence's state is how far its output has got:

    - ``0 .. len(after) - 1``: how much of ``after`` it ends with (0 also
      when it ends with nothing either pattern starts with);
    - ``len(after) + i``: the next token is ``opening[i]``, forced;
    - ``len(after) + len(opening) + k - 1``: it ends with ``opening[:k]``,
      written by the model itself, for ``k`` in ``1 .. opens_at - 1``;
    - ``UNCONSTRAINED`` once the channel is open, by either way: nothing is
      held back any more, and the sequence drops out of its batch's channel,
      so a batch whose requests have all opened theirs costs nothing.

    The state counts generated tokens only, so a sequence starts at 0
    whatever its prompt ends with.
    """

    after: tuple[int, ...]
    opening: tuple[int, ...]
    opens_at: int
    end_ids: tuple[int, ...]

    def __post_init__(self):
        if not self.after or not self.opening:
            raise ValueError("a required channel needs both `after` and `opening`")
        if not 1 <= self.opens_at <= len(self.opening):
            raise ValueError(
                f"`opens_at` must be in 1..{len(self.opening)}, got {self.opens_at}"
            )
        # `step` follows `after` and the channel's own start at once and, on a
        # mismatch, restarts at either one's first token or at nothing. That is
        # the whole failure function only while the two first tokens differ
        # and neither recurs inside either pattern.
        own = self.opening[: self.opens_at]
        firsts = {self.after[0], own[0]}
        if len(firsts) < 2 or firsts & set(self.after[1:] + own[1:]):
            raise ValueError(
                f"`after` {self.after} and `opening[:opens_at]` {own} must start "
                "with different ids, neither recurring inside either"
            )
        # A banned id the channel needs written could never be written: the
        # request would never open it and run to `max_tokens`.
        clash = set(self.end_ids) & set(self.after + self.opening)
        if clash:
            raise ValueError(
                f"end ids {sorted(clash)} are part of the channel's own ids"
            )

    @property
    def num_states(self) -> int:
        """Every state a sequence held to this channel can be in is in
        ``range(num_states)``; an open one is ``UNCONSTRAINED``."""
        return len(self.after) + len(self.opening) + self.opens_at - 1

    @property
    def table(self) -> tuple[int, ...]:
        """``after + opening + end_ids``, the layout the GPU reads.

        A state indexes the token it waits for (``after[s]``, or
        ``opening[k]`` at ``s - len(opening) + 1`` past the forced ones) or
        forces (``opening[s - len(after)]``)."""
        return self.after + self.opening + self.end_ids

    def step(self, state: int, token_id: int) -> int:
        """The state after ``token_id``. The GPU kernels make the same moves."""
        if state < 0:
            return state
        forced_end = len(self.after) + len(self.opening)
        if len(self.after) <= state < forced_end:
            # Forced tokens advance the state whatever they are: a row that saw
            # a different one is a rejected draft, whose state nobody reads.
            state += 1
            return state if state < forced_end else UNCONSTRAINED
        if state < len(self.after):
            if token_id == self.after[state]:
                return state + 1
        else:
            own = state - forced_end + 1  # it ends with `opening[:own]`
            if token_id == self.opening[own]:
                return UNCONSTRAINED if own + 1 == self.opens_at else state + 1
        if token_id == self.after[0]:
            return 1
        if token_id == self.opening[0]:
            return UNCONSTRAINED if self.opens_at == 1 else forced_end
        return 0

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
    json_schema: str | None = None
    structured_tool_format: str | None = None
    # Generated reasoning tokens are unconstrained until this marker closes.
    structured_reasoning_end: tuple[int, ...] = ()
    # Frontend default may shrink after tokenization; explicit client caps may not.
    max_tokens_is_default: bool = False

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
