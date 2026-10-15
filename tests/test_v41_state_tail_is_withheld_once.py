"""The STATE tail has exactly one owner.

Two mechanisms withhold the same bytes: the original reserve patch, which
subtracts `v41_proxy_state_reserve_blocks` from the pool, and the PAGE-backed
checkpoint path, whose `V41StateCheckpointSizing.usable_blocks` withholds the slot area
before handing the rest out. Both running deducts it twice, and the symptom is
a pool quietly smaller than it was priced at -- no error, just fewer blocks
than the recipe says and a preemption rate nobody can explain.

So the reserve patch defers when images are on. These tests pin that it
defers only then, because deferring unconditionally would leave the tail
unreserved on every run that has images off -- the opposite failure, and a
louder one (the carve runs off the end of the tensor).
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm import deepseek_v41_state_reserve_patch as reserve_mod


@pytest.fixture
def blocks(monkeypatch):
    """`v41_proxy_state_reserve_blocks` stubbed to a recognisable count."""
    import atom.plugin.vllm.deepseek_v41_bridge as bridge

    monkeypatch.setattr(bridge, "v41_proxy_state_reserve_blocks", lambda cfg: 37)
    return 37


def images(monkeypatch, on: bool):
    import atom.plugin.vllm.paged_state_checkpoint as mech

    monkeypatch.setattr(
        mech, "adapter_for_config", lambda cfg: object() if on else None
    )
    monkeypatch.setattr(mech, "state_checkpoints_on", lambda adapter, cfg: on)


def test_the_patch_owns_the_tail_when_images_are_off(monkeypatch, blocks):
    images(monkeypatch, False)
    assert reserve_mod.deepseek_v41_state_reserve_blocks(SimpleNamespace()) == blocks


def test_the_patch_defers_when_images_are_on(monkeypatch, blocks):
    images(monkeypatch, True)
    assert reserve_mod.deepseek_v41_state_reserve_blocks(SimpleNamespace()) == 0


def test_an_unimportable_mechanism_means_the_patch_still_owns_it(monkeypatch, blocks):
    """This module runs before ATOM's attention stack on some configs. The
    mechanism being absent is 'images are off', not 'skip the reservation'."""

    def boom(cfg):
        raise ImportError("not available here")

    import atom.plugin.vllm.paged_state_checkpoint as mech

    monkeypatch.setattr(mech, "adapter_for_config", boom)
    assert reserve_mod.deepseek_v41_state_reserve_blocks(SimpleNamespace()) == blocks


def test_the_deferral_is_decided_where_both_are_visible():
    """Not in the image path and not in the bridge: in the one function that
    can see both answers, so the choice cannot be made twice."""
    import inspect

    src = inspect.getsource(reserve_mod.deepseek_v41_state_reserve_blocks)
    assert "_images_own_the_tail" in src


def test_register_installs_both_and_the_image_path_after_the_patch():
    """Order matters for readability, not correctness -- but the image install
    must be present at all, or the adapter never exists in EngineCore and the
    pool is sized without it."""
    import inspect

    from atom.plugin.vllm import register

    src = inspect.getsource(register)
    assert "apply_vllm_v41_prefix_install" in src
    assert src.index("apply_vllm_v41_state_reserve_patch") < src.index(
        "apply_vllm_v41_prefix_install"
    )
