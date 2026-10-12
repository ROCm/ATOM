"""Every name the live path calls must exist and be callable.

Twice now an edit has removed a function this module's own callers depend
on while the whole unit suite stayed green -- once `_V41ImageLayer` and its
backend, once `restore_stale_rows` and the block-table accessor with it,
because the deletion spanned them. Both times the engine found it, at the
cost of a boot: `NameError` and `name 'restore_stale_rows' is not defined`.

The sizing and copier tests cannot catch this; they never reach these
names. This does, for the price of one import.
"""

import inspect

from atom.plugin.vllm import v41_state_image as mod

#: Called from outside this module -- the cache's restore hook, the bridge,
#: the prefix-caching gates, the bind-time wiring, the counters.
LIVE = [
    "images_registered",
    "images_can_serve",
    "register_image_layers",
    "image_layer_class",
    "build_runtime",
    "install_runtime",
    "runtime",
    "install_group_index_patch",
    "image_group_index",
    "image_block_table",
    "image_read_column",
    "restore_stale_rows",
    "restore_declines",
    "group_lookup_failures",
    "publish_image_halves",
    "take_store_marks",
]


def test_every_live_name_is_present_and_callable():
    missing = [n for n in LIVE if not callable(getattr(mod, n, None))]
    assert not missing, f"gone from the module its callers import: {missing}"


def test_the_layer_class_still_builds():
    """It is built lazily, so its absence shows up only when something asks."""
    cls = mod.image_layer_class()
    assert inspect.isclass(cls)
    assert hasattr(cls, "get_attn_backend")
    backend = cls.get_attn_backend(cls.__new__(cls))
    assert backend.is_ssm() is True
    assert backend.get_builder_cls() is not None


def test_the_restore_hook_the_cache_installs_is_this_function():
    """The cache calls whatever `install_runtime` handed it. If that name
    ever points somewhere else, this is where it shows."""
    src = inspect.getsource(mod.install_runtime)
    assert "install_image_restore_hook(restore_stale_rows)" in src


def test_the_whole_batch_restore_is_gone_and_stays_gone():
    """It restored every row of every step from blocks nothing had written,
    erasing state a prefill had just built. Reconnecting it looks like a
    one-line convenience, so the function refuses rather than being absent
    -- an absent name invites a fresh one with the same shape."""
    import pytest

    with pytest.raises(NotImplementedError):
        mod.offer_restore_half(0, slots=[1], blocks=[2])
