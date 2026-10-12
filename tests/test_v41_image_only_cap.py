"""Without a connector, the images still cap the local hit.

Every arm that had images also had a tier, and the tier reached the
boundary first: 11 state loads completed, 0 images restored. So the restore
direction has never run on a live engine, and the only way to make it run
is an arm with no tier -- which needs this cap, since the one that existed
hung off the connector's planner.
"""

from atom.plugin.vllm import scheduler as sched


def test_a_hit_that_is_not_the_checkpoint_is_refused(monkeypatch):
    """vLLM writes one image per prefill, at that prompt's own last block
    boundary. A shared-prefix hit lands elsewhere: measured without a tier,
    a 1362-token prompt's image sat at 1280 while the hit arrived at 1024,
    and the engine died rather than reusing nothing."""
    monkeypatch.setattr(
        "atom.plugin.vllm.v41_state_image.images_can_serve", lambda: True
    )
    # Not block-aligned, so nothing was written there.
    assert sched._image_only_cap(6900, 256) == 0
    assert sched._image_only_cap(1000, 256) == 0


def test_a_block_aligned_hit_is_not_kept_on_that_ground(monkeypatch):
    """Accepting these was tried, on the grounds that the store writes an
    image at every step end landing on a boundary. A step end almost never
    does: `image_boundary_unaligned=2087`, `image_boundary_marks=0`, and
    the engine died on the same hit again."""
    monkeypatch.setattr(
        "atom.plugin.vllm.v41_state_image.images_can_serve", lambda: True
    )
    assert sched._image_only_cap(1024, 256) == 0


def test_a_hit_that_is_a_checkpoint_is_kept(monkeypatch):
    """`get_mamba_prefill_checkpoint_position` is the identity exactly on
    the positions an image can sit at."""
    monkeypatch.setattr(
        "atom.plugin.vllm.v41_state_image.images_can_serve", lambda: True
    )
    from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position

    for hit in (1024, 6656, 9216):
        assert get_mamba_prefill_checkpoint_position(hit + 1, 256, False) == hit


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
