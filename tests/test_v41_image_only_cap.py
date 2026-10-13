"""Without a connector, the images still cap the local hit.

Every arm that had images also had a tier, and the tier reached the
boundary first: 11 state loads completed, 0 images restored. So the restore
direction has never run on a live engine, and the only way to make it run
is an arm with no tier -- which needs this cap, since the one that existed
hung off the connector's planner.
"""

from atom.plugin.vllm import scheduler as sched


def test_vllms_answer_is_left_alone_when_images_are_wired(monkeypatch):
    """vLLM has already reduced the hit to a boundary the state group can
    serve -- its coordinator takes the minimum across groups, and the image
    group only offers boundaries it holds a cached checkpoint block for.
    Measured: `MambaManager(MambaSpec) groups=[1] max=1280 -> hit=1280`."""
    monkeypatch.setattr(
        "atom.plugin.vllm.v41_state_image.images_can_serve", lambda: True
    )
    for hit in (1280, 1024, 6912, 6900, 1):
        assert sched._image_only_cap(hit, 256) == hit


def test_the_checkpoint_rule_is_not_applied_to_an_already_snapped_hit():
    """The regression this replaces, stated as arithmetic.

    `get_mamba_prefill_checkpoint_position` is a rule about a RAW prefix
    length. Applied to a hit vLLM has already snapped to a boundary it
    returns one block lower -- a position nothing ever wrote, which is
    where `needs state at 1024, found 3` came from. The mismatch was
    manufactured by that call, not found in the data.
    """
    from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position

    assert get_mamba_prefill_checkpoint_position(1280, 256, False) == 1024
    assert sched._image_only_cap(1280, 256) == 1280


def test_vllms_answer_is_left_alone_without_images(monkeypatch):
    """A no-op for every model that has none -- this runs for all of them."""
    monkeypatch.setattr(
        "atom.plugin.vllm.v41_state_image.images_can_serve", lambda: False
    )
    assert sched._image_only_cap(6912, 256) == 6912


def test_an_unimportable_image_module_leaves_the_hit_alone(monkeypatch):
    import builtins

    real = builtins.__import__

    def boom(name, *a, **kw):
        if "v41_state_image" in name:
            raise ImportError("nope")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert sched._image_only_cap(6912, 256) == 6912
