"""The checkpoint-image group is not a recurrent cache to hand to the tier.

It is declared as a `MambaSpec` so vLLM allocates and prefix-caches blocks
for it, which is enough for `find_mamba_groups` to pick it up. Left in, the
recurrent leg built `KdaStore`s for it and fed them to the V4.1 leg, which
expects its own shape: `AttributeError: 'KdaStore' object has no attribute
'req_id'` on the first request, on every rank, engine down.
"""

from types import SimpleNamespace

from atom.plugin.vllm.kv_transfer.connector import _without_v41_image_groups
from atom.plugin.vllm.v41_state_image import expected_layer_names, sizing_from_geometry


def _cfg(*layer_name_lists):
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=list(names)) for names in layer_name_lists
        ]
    )


def _image_names(n=8):
    class _S:
        units = n

    return list(expected_layer_names(_S()))[:n]


def test_an_all_image_group_is_dropped():
    names = _image_names()
    cfg = _cfg(["model.layers.0.self_attn"], names)
    kept = _without_v41_image_groups([(1, "spec")], cfg)
    assert kept == []


def test_a_real_recurrent_group_is_kept():
    cfg = _cfg(["model.layers.0.self_attn"], ["model.layers.3.linear_attn"])
    kept = _without_v41_image_groups([(1, "spec")], cfg)
    assert kept == [(1, "spec")]


def test_a_mixed_group_is_kept():
    """Dropping a group that also holds real recurrent layers would silently
    stop offloading them. Only an all-image group is ours."""
    names = _image_names(2) + ["model.layers.3.linear_attn"]
    cfg = _cfg(["model.layers.0.self_attn"], names)
    kept = _without_v41_image_groups([(1, "spec")], cfg)
    assert kept == [(1, "spec")]


def test_no_config_leaves_the_groups_alone():
    kept = _without_v41_image_groups([(1, "spec")], SimpleNamespace())
    assert kept == [(1, "spec")]
