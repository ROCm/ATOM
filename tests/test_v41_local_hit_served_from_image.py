"""A local prefix hit survives `cap_local_hit` when images can restore it.

The refusal this covers is correct without images -- the capped point is a
position nobody stored state at -- and wrong with them, because an image is
written at every block boundary vLLM checkpoints. Both directions are
asserted: a mechanism that returns the hit unconditionally would pass the
first test alone.
"""

import pytest

from atom.plugin.vllm.kv_transfer import v41_state


class _Req:
    request_id = "r0"
    block_hashes = ()


def _leg():
    leg = v41_state.V41BoundaryPlanner.__new__(v41_state.V41BoundaryPlanner)
    leg._local_caps = {}
    leg._local_cap_withdrawn = set()
    leg._counters = {
        "local_cap_declined": 0,
        "local_cap_image": 0,
        "local_cap_kept": 0,
        "local_cap_calls": 0,
    }
    leg.hash_block_size = 256
    return leg


def test_a_hit_is_refused_when_no_image_can_back_it(monkeypatch):
    monkeypatch.setattr(v41_state, "_images_can_serve", lambda: False)
    monkeypatch.setattr(v41_state, "_LOCAL_PREFIX_HITS_ENABLED", False)
    leg = _leg()
    assert leg.cap_local_hit(_Req(), 9472) == 0
    assert leg._counters["local_cap_declined"] == 1
    assert leg._counters["local_cap_image"] == 0


def test_a_hit_is_capped_to_the_checkpoint_an_image_sits_at(monkeypatch):
    monkeypatch.setattr(v41_state, "_images_can_serve", lambda: True)
    monkeypatch.setattr(v41_state, "_IMAGE_LOCAL_HITS_ENABLED", True)
    monkeypatch.setattr(v41_state, "_LOCAL_PREFIX_HITS_ENABLED", False)
    leg = _leg()
    # vLLM writes one image per prefill, at `(seq_len - 1) // 256 * 256`.
    # For an offered hit of 9472 that is 9216 -- one block below, which is
    # the whole point: at 9472 there is no image and the engine dies.
    assert leg.cap_local_hit(_Req(), 9472) == 9216
    assert leg._counters["local_cap_image"] == 1
    assert leg._counters["local_cap_declined"] == 0


def test_a_hit_inside_the_first_block_is_refused(monkeypatch):
    """Below one block there is no checkpoint to cap to, and capping to 0
    is not a hit."""
    monkeypatch.setattr(v41_state, "_images_can_serve", lambda: True)
    monkeypatch.setattr(v41_state, "_IMAGE_LOCAL_HITS_ENABLED", True)
    leg = _leg()
    assert leg.cap_local_hit(_Req(), 200) == 0
    assert leg._counters["local_cap_declined"] == 1


def test_wired_images_alone_do_not_open_the_gate(monkeypatch):
    """Storing and restoring images is a separate claim from letting one
    back a local hit. Serving a hit no image happens to sit at is a dead
    engine, so the second gate is opted into on its own."""
    monkeypatch.setattr(v41_state, "_images_can_serve", lambda: True)
    monkeypatch.setattr(v41_state, "_IMAGE_LOCAL_HITS_ENABLED", False)
    monkeypatch.setattr(v41_state, "_LOCAL_PREFIX_HITS_ENABLED", False)
    leg = _leg()
    assert leg.cap_local_hit(_Req(), 9472) == 0
    assert leg._counters["local_cap_image"] == 0
    assert leg._counters["local_cap_declined"] == 1


def test_no_hit_is_still_no_hit(monkeypatch):
    """Images do not invent a hit where vLLM found none."""
    monkeypatch.setattr(v41_state, "_images_can_serve", lambda: True)
    leg = _leg()
    assert leg.cap_local_hit(_Req(), 0) == 0
    assert leg._counters["local_cap_image"] == 0
