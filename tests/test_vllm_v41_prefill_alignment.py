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
BLOCK = 256


class FakeScheduler:
    """Only what the wrapper reads: a connector carrying a planner."""

    def __init__(self, interval=INTERVAL, block=BLOCK):
        self.connector = SimpleNamespace(
            _v41_planner=SimpleNamespace(state_interval=interval, hash_block_size=block)
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


def test_a_chunk_short_of_the_interval_still_lands_on_a_block():
    """The boundary grid is a minimum spacing, so a block boundary will do.

    This used to run the chunk out to 4,090 and record a crossing. The planner
    now accepts any block-aligned frontier far enough from the last one, so
    stopping at 3,840 gives it something to land on at no cost: the tokens
    between go in the next chunk either way.
    """
    assert wrapper()(FakeScheduler(), request(), 0, 4090) == 3840


def test_a_chunk_too_small_for_even_one_block_is_left_alone():
    """Clipping cannot extend a chunk, and must never schedule zero tokens."""
    assert wrapper()(FakeScheduler(), request(), 0, 200) == 200


def test_the_last_chunk_stops_on_the_prompt_anchor():
    """The one boundary below the prompt that nothing else will reach.

    Boundaries ABOVE the prompt are reached anyway -- decode walks the frontier
    one token at a time -- so the last chunk used to be exempt entirely. The
    anchor is different: it is the last block-aligned point at or before the
    prompt end, and once the frontier passes it nothing returns to it. A prompt
    of 5,000 tokens anchors at 4,864.
    """
    w = wrapper()
    assert w(FakeScheduler(), request(prompt=5000), 4096, 904) == 4864 - 4096


def test_a_last_chunk_already_past_the_anchor_is_left_alone():
    """Clipping must not extend, and a chunk starting past the anchor cannot
    reach it: the remaining tokens are the unaligned tail of the prompt."""
    w = wrapper()
    assert w(FakeScheduler(), request(prompt=5000), 4864, 136) == 136


def test_a_block_aligned_prompt_anchors_at_its_own_end():
    """No clip needed: the prompt end IS the anchor, and the chunk ends there."""
    w = wrapper()
    assert w(FakeScheduler(), request(prompt=4096 + 768), 4096, 768) == 768


def test_a_mid_prompt_chunk_clips_from_its_own_start():
    assert wrapper()(FakeScheduler(), request(), INTERVAL, 8000) == INTERVAL


def test_a_model_with_no_planner_is_untouched():
    plain = SimpleNamespace(connector=None)
    assert wrapper()(plain, request(), 0, 8186) == 8186
