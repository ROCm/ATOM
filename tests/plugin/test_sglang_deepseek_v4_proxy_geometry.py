from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
)
from atom.plugin.sglang.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    ATOMDeepSeekV4ProxyKVPool,
    _geometry_serves_ratio,
    _proxy_pool_geometry,
    _resolve_v4_pool_geometry,
)

SGLANG_BRIDGE = (
    Path(__file__).parents[2] / "atom/plugin/sglang/deepseek_v4_bridge.py"
).read_text()


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

    assert geometry.classes == (DENSE_RATIO, CSA_RATIO, HCA_RATIO)
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
        stage_ratios=[DENSE_RATIO, CSA_RATIO, HCA_RATIO],
        _atom_v4_geometry=None,
    )
    metadata = SimpleNamespace()

    geometry = _resolve_v4_pool_geometry(metadata, pool)

    assert metadata.pool_geometry is geometry
    assert metadata.envelope_rows == geometry.block_rows(CSA_RATIO)
    assert geometry.envelope_rows == geometry.block_rows(HCA_RATIO)
    assert metadata.envelope_rows == ATOM_DEEPSEEK_V4_BLOCK_SIZE // CSA_RATIO
    assert geometry.envelope_rows == ATOM_DEEPSEEK_V4_BLOCK_SIZE // HCA_RATIO


def test_proxy_geometry_omits_absent_stage_ratios():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        num_req_slots=2,
        swa_size=256,
        c4_size=0,
        c128_size=3,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.bfloat16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        indexer_head_dim=8,
        layer_num=2,
        compression_ratios=[DENSE_RATIO, HCA_RATIO],
        device="cpu",
    )
    geometry = _proxy_pool_geometry(pool)

    assert geometry.classes == (DENSE_RATIO, HCA_RATIO)
    assert _geometry_serves_ratio(geometry, DENSE_RATIO)
    assert _geometry_serves_ratio(geometry, HCA_RATIO)
    assert not _geometry_serves_ratio(geometry, CSA_RATIO)
    with pytest.raises(KeyError):
        geometry.window_params(CSA_RATIO)


def test_sglang_decode_graph_pads_csa_visibility_to_t_pad():
    assert "visible_np = np.zeros(t_pad, dtype=np.int32)" in SGLANG_BRIDGE
    assert "visible_np[:total] = visible_csa(pos_np).astype(np.int32)" in SGLANG_BRIDGE


def test_sglang_graph_buffers_keep_distinct_state_slot_addresses():
    assert "self.state_slot_in = i32(s)" in SGLANG_BRIDGE
    assert "self.state_slot_out = i32(s)" in SGLANG_BRIDGE
    assert (
        "md.state_slot_in = out.clone() if state_slot_in is None else state_slot_in"
        in SGLANG_BRIDGE
    )
    assert (
        "md.state_slot_out = bufs.stage(bufs.state_slot_out, slot_arr, n)"
        in SGLANG_BRIDGE
    )
    assert (
        "md.state_slot_in = bufs.stage(bufs.state_slot_in, slot_arr, n)"
        in SGLANG_BRIDGE
    )
