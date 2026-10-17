# SPDX-License-Identifier: MIT
"""V4.1 prefill chunks must end where a CSA2 state checkpoint can exist.

A STATE checkpoint exists only where the frontier lands. vLLM spends one token
budget across the step, so a prefill sharing a batch with decodes is scheduled
a few tokens short of the budget and lands off the PAGE grid for the rest of
the prompt -- and a tier save that ends there has no checkpoint to carry.

The clip is exercised through the patch's own wrapper so that what is tested
is what `register_model` installs.
"""

from types import SimpleNamespace

import pytest

BLOCK = 256


class FakeScheduler:
    """Only what the wrapper reads: a connector carrying a planner."""

    def __init__(self, block=BLOCK):
        self.connector = SimpleNamespace(
            _v41_planner=SimpleNamespace(boundary_grid=block)
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


def test_a_chunk_is_clipped_back_to_a_page_edge():
    """4,090 tokens of budget stop at 3,840, the last PAGE edge they reach."""
    assert wrapper()(FakeScheduler(), request(), 0, 4090) == 3840


def test_a_chunk_already_on_a_page_edge_is_untouched():
    assert wrapper()(FakeScheduler(), request(), 0, 4096) == 4096


def test_a_chunk_too_small_for_even_one_block_is_left_alone():
    """Clipping cannot extend a chunk, and must never schedule zero tokens."""
    assert wrapper()(FakeScheduler(), request(), 0, 200) == 200


def test_the_last_chunk_takes_the_anchor_by_default():
    """The prompt-end anchor: the last PAGE edge at or before the prompt end.

    5,000 tokens anchors at 4,864, so the last chunk stops 136 short of the
    prompt end and the request owns a boundary there -- the one a later
    request with the same prompt asks about.
    """
    w = wrapper()
    assert w(FakeScheduler(), request(prompt=5000), 4096, 904) == 4864 - 4096


def test_the_anchor_clip_can_still_be_turned_off(monkeypatch):
    """Kept as a switch for bisecting a prefill-shape problem, not as a
    default anyone should need."""
    monkeypatch.setenv("ATOM_V41_PROMPT_ANCHOR_CLIP", "0")
    w = wrapper()
    assert w(FakeScheduler(), request(prompt=5000), 4096, 904) == 904


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
    assert wrapper()(FakeScheduler(), request(), 4096, 8000) == 12032 - 4096


def test_a_model_with_no_planner_is_untouched():
    plain = SimpleNamespace(connector=None)
    assert wrapper()(plain, request(), 0, 8186) == 8186
