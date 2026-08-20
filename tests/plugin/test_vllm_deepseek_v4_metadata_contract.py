# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

BRIDGE_SOURCE = (
    Path(__file__).parents[2] / "atom/plugin/vllm/deepseek_v4_bridge.py"
).read_text()
OPS_SOURCE = (
    Path(__file__).parents[2] / "atom/plugin/vllm/deepseek_v4_ops.py"
).read_text()


def test_vllm_decode_buffers_keep_distinct_state_input_and_output_addresses():
    assert "self.state_slot_in = i32(S)" in BRIDGE_SOURCE
    assert "self.state_slot_out = i32(S)" in BRIDGE_SOURCE
    assert "bufs.stage(bufs.state_slot_in, physical_slot_arr)" in BRIDGE_SOURCE
    assert "bufs.stage(bufs.state_slot_out, physical_slot_arr)" in BRIDGE_SOURCE


def test_vllm_eager_metadata_satisfies_split_state_slot_contract():
    assert "md = AttentionMetaData_DSV4(" in BRIDGE_SOURCE
    assert (
        "md.state_slot_out = torch.from_numpy(physical_slot_arr).to(device)"
        in BRIDGE_SOURCE
    )
    assert "md.state_slot_in = md.state_slot_out.clone()" in BRIDGE_SOURCE
    assert "md.state_slot_mapping = md.state_slot_out" in BRIDGE_SOURCE


def test_vllm_proxy_embeds_compressor_state_in_unified_slot_planes():
    assert "arena_rows=arena_rows" in BRIDGE_SOURCE
    assert "StateArena(" in BRIDGE_SOURCE
    assert "geometry.slot_positions" in BRIDGE_SOURCE
    assert 'kv_state=arena.view("csa_main_kv")[csa_i]' in BRIDGE_SOURCE
    assert 'kv_state=arena.view("hca_main_kv")[hca_i]' in BRIDGE_SOURCE


def test_vllm_proxy_aligns_embedded_state_after_packed_kv_offset():
    assert "total += ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT - 1" in BRIDGE_SOURCE
    assert "alignment_pad = (" in BRIDGE_SOURCE
    assert ") % ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT" in BRIDGE_SOURCE
    assert "raw = raw[alignment_pad:]" in BRIDGE_SOURCE


def test_vllm_decode_hca_indices_use_unified_geometry_rows():
    assert "envelope_rows=md.pool_geometry.envelope_rows" in BRIDGE_SOURCE
    assert (
        "tl.store(hca_indices_ptr + base + k, bt * envelope_rows, mask=mask)"
        in OPS_SOURCE
    )
