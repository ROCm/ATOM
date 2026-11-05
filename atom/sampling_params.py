# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

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


@dataclass(frozen=True)
class TokenConstraint:
    """A hard token-ID rule, not a probability preference.

    Small static ID lists can compile once to an allowed-token bitmask. Dynamic
    grammars fill that representation directly; GPU channel state machines apply
    the same allow/deny semantics without transferring lists to the host.
    """

    mode: Literal["allow", "deny"]
    token_ids: tuple[int, ...]
    until: int | None = None  # generated tokens; None means the whole request

    def validate(self):
        if self.mode not in ("allow", "deny"):
            raise ValueError("Token constraint must be allow or deny")
        if any(type(i) is not int or i < 0 for i in self.token_ids):
            raise ValueError("Token IDs must be nonnegative integers")
        if self.until is not None and (type(self.until) is not int or self.until < 0):
            raise ValueError("until must be a nonnegative generated-token count")
        if self.mode == "allow" and not self.token_ids:
            raise ValueError("An allow rule cannot be empty")

    @classmethod
    def allow(cls, token_ids, *, until=None):
        return cls("allow", tuple(token_ids), until)

    @classmethod
    def deny(cls, token_ids, *, until=None):
        return cls("deny", tuple(token_ids), until)

    @classmethod
    def force(cls, token_id):
        return cls.allow((token_id,))

    def to_bitmask(self, vocab_size):
        """Compile on the CPU outside the per-token sampling hot path."""
        if self.mode not in ("allow", "deny") or vocab_size <= 0:
            raise ValueError("Invalid token constraint mode or vocabulary size")
        if any(
            type(token) is not int or not 0 <= token < vocab_size
            for token in self.token_ids
        ):
            raise ValueError("Constraint token ID is outside the vocabulary")
        if (self.mode == "allow" and not self.token_ids) or (
            self.mode == "deny" and len(set(self.token_ids)) == vocab_size
        ):
            raise ValueError("Token constraint must leave at least one allowed token")
        import torch

        mask = torch.full(
            (1, (vocab_size + 31) // 32),
            -1 if self.mode == "deny" else 0,
            dtype=torch.int32,
        )
        for token in self.token_ids:
            value = int(mask[0, token // 32])
            bit = 1 << (token % 32)
            value = (value | bit if self.mode == "allow" else value & ~bit) & 0xFFFFFFFF
            mask[0, token // 32] = value if value < 2**31 else value - 2**32
        return mask


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
    top_logprobs: int = 0
    json_schema: str | None = None
    structured_tool_format: str | None = None
    # Generated reasoning tokens are unconstrained until this marker closes.
    structured_reasoning_end: str = ""
    # Frontend default may shrink after tokenization; explicit client caps may not.
    max_tokens_is_default: bool = False
    min_tokens: int = 0
    token_constraints: tuple[TokenConstraint, ...] = ()

    def __post_init__(self):
        if type(self.top_logprobs) is not int or not 0 <= self.top_logprobs <= 20:
            raise ValueError("top_logprobs must be an integer between 0 and 20")
        if self.top_logprobs and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs")
        if type(self.min_tokens) is not int or (
            self.min_tokens < 0
            or (self.min_tokens > 0 and self.min_tokens > self.max_tokens)
        ):
            raise ValueError("min_tokens must be an integer in [0, max_tokens]")
        for rule in self.token_constraints:
            if isinstance(rule, TokenConstraint):
                rule.validate()
        if any(
            not isinstance(rule, TokenConstraint) for rule in self.token_constraints
        ):
            raise ValueError("token_constraints must contain TokenConstraint rules")
        # 0 also means disabled, as in vLLM/SGLang; MLPerf Endpoints workloads send it
        if self.top_k == 0:
            self.top_k = -1
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or 0 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")

    def validate_logprob_options(
        self, config, *, kv_transfer=None, multimodal=False, synthetic=False
    ):
        """Execution admission shared by HTTP and direct engine callers."""
        if not self.logprobs:
            return
        if getattr(config, "enable_rapidserve", False):
            raise ValueError("logprobs is unavailable with RapidServe")
        if kv_transfer or getattr(config, "kv_transfer_config", None):
            raise ValueError("logprobs is unavailable with P/D transfer")
        if multimodal:
            raise ValueError("Chat logprobs currently supports text input only")
        spec = getattr(config, "speculative_config", None)
        if synthetic or getattr(spec, "synthetic_acceptance_rates", None) is not None:
            raise ValueError("logprobs is unavailable with synthetic forced acceptance")
