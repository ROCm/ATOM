# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`add_request` and `SamplingParams.n > 1`.

`InputOutputProcessor.preprocess` returns one Sequence and rejects n > 1,
pointing the caller at `preprocess_fanout` -- its own guard says so.
`add_request` called `preprocess` anyway, so every offline n > 1 request
raised `ValueError` before a token was generated. Only the n == 1 path works
today, which is why nothing caught it.

The guard under test is the production method, attached unbound: a stand-in
that re-implemented "raise above n = 1" could not tell us whether the real one
still does.
"""

from types import SimpleNamespace

import pytest

from atom.model_engine.llm_engine import InputOutputProcessor, LLMEngine
from atom.sampling_params import SamplingParams


class _IOProcessorDouble:
    """The real `preprocess`, over a `preprocess_fanout` that records calls."""

    preprocess = InputOutputProcessor.preprocess

    def __init__(self):
        self.fanout_calls = []

    def preprocess_fanout(self, prompt_or_tokens, sampling_params, **kwargs):
        self.fanout_calls.append((prompt_or_tokens, sampling_params, kwargs))
        n = max(1, int(getattr(sampling_params, "n", 1)))
        return [
            SimpleNamespace(prompt=prompt_or_tokens, choice_index=i) for i in range(n)
        ]


def _engine():
    io = _IOProcessorDouble()
    submitted = []
    engine = SimpleNamespace(
        io_processor=io,
        core_mgr=SimpleNamespace(add_request=submitted.append),
    )
    return engine, io, submitted


def test_n_greater_than_one_reaches_the_scheduler():
    engine, _io, submitted = _engine()

    LLMEngine.add_request(engine, ["hello"], SamplingParams(n=3))

    assert [seq.choice_index for seq in submitted[0]] == [0, 1, 2]


def test_n_equals_one_is_unchanged():
    engine, _io, submitted = _engine()

    LLMEngine.add_request(engine, ["hello"], SamplingParams(n=1))

    assert len(submitted[0]) == 1


def test_the_siblings_of_every_prompt_are_submitted_together():
    """One `core_mgr.add_request`, all prompts' siblings flattened into it."""
    engine, _io, submitted = _engine()

    LLMEngine.add_request(
        engine,
        ["a", "b"],
        [SamplingParams(n=2), SamplingParams(n=3)],
    )

    assert len(submitted) == 1
    assert [seq.prompt for seq in submitted[0]] == ["a", "a", "b", "b", "b"]


def test_the_request_id_is_passed_as_the_parent():
    """Siblings derive their ids from it, so it cannot go in as `request_id`."""
    engine, io, _submitted = _engine()

    LLMEngine.add_request(engine, ["hello"], SamplingParams(n=2), request_ids=["req-7"])

    assert io.fanout_calls[0][2]["parent_request_id"] == "req-7"


def test_preprocess_still_refuses_n_greater_than_one():
    """The contract `add_request` was breaking.

    If this ever stops raising, the fan-out in `add_request` is no longer load
    bearing -- and the reason for these tests has moved.
    """
    io = _IOProcessorDouble()

    with pytest.raises(ValueError, match="preprocess_fanout"):
        io.preprocess("hello", SamplingParams(n=2))

    assert io.preprocess("hello", SamplingParams(n=1)).choice_index == 0
