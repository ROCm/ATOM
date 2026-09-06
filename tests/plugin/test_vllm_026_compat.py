from types import SimpleNamespace

from atom.plugin.vllm.deepseek_v4_prefix_patch import (
    _kv_cache_config_has_v4_proxy,
    _kv_cache_config_needs_non_immediate_reuse,
    _roll_back_prefix_hit,
)
from atom.plugin.vllm.deepseek_v4_profiling_patch import _mark_v4_proxy_cache_mode
from atom.plugin.vllm.spec_decode_patch import _make_atom_compatible_eagle3_type


class _FakeKVCacheManager:
    """A manager with no ``coordinator``, so every group takes the fallback."""

    @staticmethod
    def create_kv_cache_blocks(groups):
        return SimpleNamespace(blocks=groups)


class _FakeTwoGroupKVCacheManager(_FakeKVCacheManager):
    """Target at block 128 plus a DSpark draft group at block 64."""

    coordinator = SimpleNamespace(
        single_type_managers=(
            SimpleNamespace(block_size=128),
            SimpleNamespace(block_size=64),
        )
    )


def test_v4_prefix_cache_drop_preserves_vllm_026_boundary():
    manager = _FakeKVCacheManager()
    computed_blocks = SimpleNamespace(
        blocks=(
            (0, 1, 2, 3),
            (4, 5, 6, 7),
        )
    )

    new_blocks, num_tokens, shared_prefix_boundary = _roll_back_prefix_hit(
        manager,
        computed_blocks,
        512,  # 4 blocks x 128
        384,
        rollback_tokens=256,
    )

    # No coordinator -> both groups convert with ATOM_DEEPSEEK_V4_BLOCK_SIZE.
    # That fallback is the only reader of the constant, so this also pins the
    # import that used to sit in the installer and raised NameError here.
    assert new_blocks.blocks == ([0, 1], [4, 5])
    assert num_tokens == 256
    assert shared_prefix_boundary == 384


def test_v4_prefix_cache_drop_converts_per_group_block_size():
    manager = _FakeTwoGroupKVCacheManager()
    computed_blocks = SimpleNamespace(
        blocks=(
            tuple(range(8)),  # 8 x 128 = 1024 tokens
            tuple(range(100, 116)),  # 16 x 64 = 1024 tokens
        )
    )

    new_blocks, num_tokens, _ = _roll_back_prefix_hit(
        manager,
        computed_blocks,
        1024,
        384,
        rollback_tokens=256,
    )

    # 768 tokens kept in both groups: 6 blocks at 128, 12 at 64. Dropping a
    # fixed block *count* would have charged 512 tokens to the block-64 group.
    assert new_blocks.blocks == (list(range(6)), list(range(100, 112)))
    assert num_tokens == 768


def test_v4_prefix_cache_empty_hit_keeps_vllm_026_result():
    manager = _FakeKVCacheManager()
    computed_blocks = SimpleNamespace(blocks=((),))

    result = _roll_back_prefix_hit(
        manager,
        computed_blocks,
        0,
        0,
        rollback_tokens=256,
    )

    assert result == (computed_blocks, 0, 0)


def test_v4_profile_cache_mode_marks_only_proxy_layers():
    proxy = SimpleNamespace(
        _atom_v4_proxy_layer=True,
        _atom_v4_profiling_kv_cache=False,
    )
    unrelated = SimpleNamespace(_atom_v4_profiling_kv_cache=False)
    context = {"proxy": proxy, "unrelated": unrelated}

    _mark_v4_proxy_cache_mode(context, True)

    assert proxy._atom_v4_profiling_kv_cache is True
    assert unrelated._atom_v4_profiling_kv_cache is False

    _mark_v4_proxy_cache_mode(context, False)
    assert proxy._atom_v4_profiling_kv_cache is False


def test_v4_proxy_kv_cache_detection_ignores_unrelated_layers():
    unrelated = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["model.layers.0.attn"])],
        has_mamba_layers=False,
    )
    proxy = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.atom_deepseek_v4_proxy"])
        ],
        has_mamba_layers=False,
    )
    assert not _kv_cache_config_has_v4_proxy(unrelated)
    assert _kv_cache_config_has_v4_proxy(proxy)
    assert not _kv_cache_config_needs_non_immediate_reuse(unrelated)
    assert _kv_cache_config_needs_non_immediate_reuse(proxy)

    mamba = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["model.layers.0.linear_attn"])],
        has_mamba_layers=True,
    )
    assert _kv_cache_config_needs_non_immediate_reuse(mamba)


def test_eagle3_type_proxy_is_valid_isinstance_guard():
    class NativeEagle3:
        pass

    class AtomModelBase:
        pass

    proxy = _make_atom_compatible_eagle3_type(NativeEagle3, AtomModelBase)
    guard = (proxy, str)

    assert isinstance(proxy, type)
    assert isinstance(NativeEagle3(), guard)
    assert isinstance(AtomModelBase(), guard)
    assert not isinstance(object(), guard)
