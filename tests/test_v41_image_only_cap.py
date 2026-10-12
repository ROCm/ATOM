"""Without a connector, the images still cap the local hit.

Every arm that had images also had a tier, and the tier reached the
boundary first: 11 state loads completed, 0 images restored. So the restore
direction has never run on a live engine, and the only way to make it run
is an arm with no tier -- which needs this cap, since the one that existed
hung off the connector's planner.
"""

from atom.plugin.vllm import scheduler as sched


def test_the_hit_is_capped_to_the_checkpoint_when_images_are_wired(monkeypatch):
    monkeypatch.setattr(
        "atom.plugin.vllm.v41_state_image.images_can_serve", lambda: True
    )
    assert sched._image_only_cap(6912, 256) == 6656
    assert sched._image_only_cap(9472, 256) == 9216


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
