"""The V4.1 side of the PAGE-backed checkpoint mechanism.

Sizing is checked against the real DeepSeek-V4.1-Flash geometry rather than a
made-up one, because the numbers it has to agree with are not free parameters:
`StateCopies.__init__` refuses a spec whose `page_unit_bytes` / `slot_bytes` /
`image_bytes` differ from the allocated cache, so a sizing that is internally
consistent but wrong is a startup failure, not a slow path.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm import deepseek_v41_state_checkpoint as adapter_mod

BLOCK = 256


def geometry():
    """A V4.1 geometry of the shipped shape.

    Built directly rather than through `build_v41_pool_geometry`, which pulls
    the attention stack and its GPU driver in. The dimensions are the shipped
    checkpoint's (40 layers, head_dim 512, index_dim 128, window 128, DSpark
    5), but the owner derivation is this file's, so **no test below pins a
    byte count** -- they check the relations the adapter has to preserve, not
    values that would make them change-detectors for a config edit.
    """
    geo = pytest.importorskip("atom.model_ops.attentions.pool_layout.v41_pool_geometry")
    return geo.V41PoolGeometry(
        layers=40,
        owners=((2, 2), (8, 2), (14, 2), (20, 2)),
        block_size=BLOCK,
        window_size=128,
        head_dim=512,
        index_dim=128,
        history_size=3,
        packed=False,
        speculative_tokens=5,
        layer_ratios=tuple([0, 0] + [2] * 18 + [1] * 20),
        index_topk=512,
    )


def sizing():
    geo = geometry()
    page_unit = int(geo.paged_bytes)
    page = -(-page_unit // BLOCK) * BLOCK
    return adapter_mod._with_geometry(
        adapter_mod.V41StateCheckpointSizing(
            num_slots=64,
            max_num_seqs=64,
            page_unit_bytes=page_unit,
            page_bytes=page,
            slot_bytes=int(geo.state_bytes),
            image_bytes=int(geo.checkpoint_image_bytes),
            layout_id=str(geo.layout_id),
        ),
        geo,
    )


def test_the_spec_matches_the_geometry_state_copies_will_check_it_against():
    """`StateCopies` compares exactly these three against the live cache."""
    geo, s = geometry(), sizing()
    spec = s.spec
    assert spec.page_unit_bytes == int(geo.paged_bytes)
    assert spec.slot_bytes == int(geo.state_bytes)
    assert spec.image_bytes == int(geo.checkpoint_image_bytes)
    assert spec.layout_id == str(geo.layout_id)


def test_an_image_is_several_whole_page_units():
    """`k` is what sizes the image layer list, so it must be the ceiling the
    spec computes and not a separate rounding."""
    s = sizing()
    assert s.k == -(-s.image_bytes // s.page_unit_bytes)
    assert s.k >= 1
    assert len(adapter_mod.deepseek_v41_state_checkpoint_layer_names(s.k)) == s.k


def test_the_image_is_a_strict_part_of_a_slot():
    """The compressor rings are `in_checkpoint=False`, so the image is smaller
    than the slot. Equal would mean that exclusion stopped applying."""
    s = sizing()
    assert 0 < s.image_bytes < s.slot_bytes


def test_a_page_unit_is_exactly_one_proxy_block():
    """The simplification this adapter is built on: V4's unit needs eleven
    blocks of another size, V4.1's is the block."""
    geo, s = geometry(), sizing()
    assert s.page_unit_bytes == int(geo.paged_bytes)
    assert s.page_bytes >= s.page_unit_bytes
    assert s.page_bytes % BLOCK == 0


def test_usable_blocks_withholds_the_slot_tail():
    s = sizing()
    total = 4096
    usable = s.usable_blocks(total)
    assert 0 < usable < total
    withheld = (total - usable) * s.page_bytes
    assert withheld >= s.num_slots * s.slot_bytes


def test_usable_blocks_is_a_pure_function_of_the_block_count():
    """EngineCore sets the scheduler's count and every worker carves; they
    never compare notes, so the same input must give the same answer."""
    s = sizing()
    assert s.usable_blocks(4096) == s.usable_blocks(4096)
    assert s.usable_blocks(8192) >= s.usable_blocks(4096)


def test_usable_blocks_never_goes_negative_on_a_tiny_pool():
    s = sizing()
    assert s.usable_blocks(1) == 0
    assert s.usable_blocks(0) == 0


# --- the decision this adapter makes differently from V4 ----------------


@pytest.mark.parametrize(
    "method, expected",
    [
        ("dspark", True),
        (None, True),
        ("mtp", False),
        ("eagle3", False),
        ("ngram", False),
    ],
)
def test_only_dspark_and_no_drafter_keep_their_state_in_the_image(method, expected):
    config = SimpleNamespace(
        speculative_config=None if method is None else SimpleNamespace(method=method)
    )
    assert adapter_mod.v41_drafter_in_target_slot(config) is expected


def test_no_speculative_config_at_all_is_accepted():
    assert adapter_mod.v41_drafter_in_target_slot(SimpleNamespace()) is True


# --- startup refusals ----------------------------------------------------


def config(**overrides):
    cache = {
        "mamba_cache_mode": "align",
        "prefix_match_unit": None,
        "prefix_cache_retention_interval": 0,
        "enable_mamba_shared_prefix_checkpoint": False,
    }
    cache.update(overrides)
    return SimpleNamespace(cache_config=SimpleNamespace(**cache))


def test_the_shipped_configuration_is_accepted():
    adapter_mod.register_v41_state_checkpoint_adapter().check_state_checkpoints_supported(
        config()
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"mamba_cache_mode": "none"}, "mamba_cache_mode"),
        ({"mamba_cache_mode": "all"}, "mamba_cache_mode"),
        ({"prefix_match_unit": 32}, "prefix_match_unit"),
        ({"prefix_cache_retention_interval": 4096}, "prefix_cache_retention_interval"),
        ({"enable_mamba_shared_prefix_checkpoint": True}, "shared_prefix_checkpoint"),
    ],
)
def test_each_unsupported_setting_is_refused(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        adapter_mod.register_v41_state_checkpoint_adapter().check_state_checkpoints_supported(
            config(**overrides)
        )


def test_a_matching_prefix_match_unit_is_fine():
    adapter_mod.register_v41_state_checkpoint_adapter().check_state_checkpoints_supported(
        config(prefix_match_unit=BLOCK)
    )


def test_every_problem_is_reported_not_just_the_first():
    with pytest.raises(ValueError) as caught:
        adapter_mod.register_v41_state_checkpoint_adapter().check_state_checkpoints_supported(
            config(mamba_cache_mode="none", prefix_match_unit=32)
        )
    assert "mamba_cache_mode" in str(caught.value)
    assert "prefix_match_unit" in str(caught.value)


def test_a_missing_attribute_is_a_version_difference_not_a_refusal():
    adapter_mod.register_v41_state_checkpoint_adapter().check_state_checkpoints_supported(
        SimpleNamespace(cache_config=SimpleNamespace())
    )


# --- registration --------------------------------------------------------


def test_the_adapter_is_registered_once_however_often_it_is_asked():
    mech = pytest.importorskip("atom.plugin.vllm.paged_state_checkpoint")
    first = adapter_mod.register_v41_state_checkpoint_adapter()
    before = len(mech._adapters)
    assert adapter_mod.register_v41_state_checkpoint_adapter() is first
    assert len(mech._adapters) == before


def test_the_store_sites_go_on_this_vllms_runner_not_the_hardcoded_one():
    """The vendored mechanism installs on the V1 runner; V4.1 is on V2. The
    overlay must enumerate the classes instead of importing one."""
    import inspect

    src = inspect.getsource(adapter_mod.install_v41_store_sites)
    # The docstring names the V1 runner to explain why this exists, so strip
    # it before looking: a string being MENTIONED is not it being used, and a
    # check that cannot tell those apart would be satisfied by deleting the
    # explanation.
    body = src.replace(adapter_mod.install_v41_store_sites.__doc__ or "", "")
    assert "gpu_model_runner_classes()" in body
    assert "vllm.v1.worker.gpu_model_runner" not in body
