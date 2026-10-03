# SPDX-License-Identifier: MIT
"""V4.1 prefill chunks must end where a CSA2 state image can exist.

A STATE image exists only where the frontier lands. vLLM spends one token
budget across the step, so a prefill sharing a batch with decodes is scheduled
a few tokens short of the budget and then steps over the interval grid for the
rest of the prompt -- measured at concurrency 8 as `boundary_passed` 1,315
against `sweep_offered` 106.

The clip is exercised through the patch's own wrapper so that what is tested
is what `register_model` installs.
"""

from types import SimpleNamespace

import pytest

INTERVAL = 4096


class FakeScheduler:
    """Only what the wrapper reads: a connector carrying a planner."""

    def __init__(self, interval=INTERVAL):
        self.connector = SimpleNamespace(
            _v41_planner=SimpleNamespace(state_interval=interval)
        )


def wrapper():
    """vLLM's `_reserve_prefill_lookahead`, wrapped, with upstream as identity."""
    pytest.importorskip("vllm")
    from vllm.v1.core.sched.scheduler import Scheduler

    from atom.plugin.vllm.scheduler import apply_vllm_v41_prefill_alignment_patch

    original = Scheduler._reserve_prefill_lookahead
    Scheduler._reserve_prefill_lookahead = (
        lambda self, request, num_computed_tokens, num_new_tokens: num_new_tokens
    )
    try:
        apply_vllm_v41_prefill_alignment_patch()
        return Scheduler._reserve_prefill_lookahead
    finally:
        Scheduler._reserve_prefill_lookahead = original


def request(prompt=65536):
    return SimpleNamespace(num_prompt_tokens=prompt, num_tokens=prompt)


def test_a_chunk_is_clipped_back_to_the_interval():
    """8,186 tokens of budget must stop at 4,096, not run to 8,186."""
    assert wrapper()(FakeScheduler(), request(), 0, 8186) == INTERVAL


def test_a_chunk_already_on_a_boundary_is_untouched():
    assert wrapper()(FakeScheduler(), request(), 0, INTERVAL) == INTERVAL


def test_a_chunk_too_small_to_reach_a_boundary_is_left_alone():
    """Clipping cannot extend a chunk, and must never schedule zero tokens.

    With the budget at or below the interval the first boundary is already out
    of reach; crossing it and letting `boundary_passed` say so beats stalling
    the request.
    """
    assert wrapper()(FakeScheduler(), request(), 0, 4090) == 4090


def test_the_last_chunk_of_a_prompt_is_exempt():
    """Decode walks the frontier to the next boundary one token at a time."""
    w = wrapper()
    assert w(FakeScheduler(), request(prompt=5000), 4096, 904) == 904


def test_a_mid_prompt_chunk_clips_from_its_own_start():
    assert wrapper()(FakeScheduler(), request(), INTERVAL, 8000) == INTERVAL


def test_a_model_with_no_planner_is_untouched():
    plain = SimpleNamespace(connector=None)
    assert wrapper()(plain, request(), 0, 8186) == 8186
