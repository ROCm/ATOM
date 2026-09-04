from types import SimpleNamespace

import torch

from atom.plugin.sglang.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    ATOMDeepSeekV4ProxyKVPool,
    _proxy_pool_geometry,
    _resolve_v4_pool_geometry,
)


def test_proxy_geometry_matches_per_layer_cache_views():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        num_req_slots=2,
        swa_size=256,
        c4_size=64,
        c128_size=3,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.bfloat16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        indexer_head_dim=8,
        layer_num=3,
        compression_ratios=[0, 4, 128],
        device="cpu",
    )
    geometry = _proxy_pool_geometry(pool)

    assert geometry.window_params(0).ring_start == 0
    for layer, ratio, compressed in (
        (1, 4, pool.views["csa_main"][0]),
        (2, 128, pool.views["hca_main"][0]),
    ):
        unified = pool.views["unified"][layer]
        window = pool.views["swa"][layer]
        ring_start = pool.num_blocks * (ATOM_DEEPSEEK_V4_BLOCK_SIZE // ratio)

        assert geometry.window_params(ratio).ring_start == ring_start
        assert compressed.data_ptr() == unified.data_ptr()
        assert window.data_ptr() == unified[ring_start].data_ptr()


def test_proxy_metadata_uses_per_layer_csa_block_stride():
    pool = SimpleNamespace(
        num_blocks=3,
        swa_cache_size=128,
        _atom_v4_geometry=None,
    )
    metadata = SimpleNamespace()

    geometry = _resolve_v4_pool_geometry(metadata, pool)

    assert metadata.pool_geometry is geometry
    assert metadata.envelope_rows == ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
    assert geometry.envelope_rows == ATOM_DEEPSEEK_V4_BLOCK_SIZE // 128
