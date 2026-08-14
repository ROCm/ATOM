# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

BRIDGE_SOURCE = (
    Path(__file__).parents[2] / "atom/plugin/vllm/deepseek_v4_bridge.py"
).read_text()


def test_vllm_decode_buffers_keep_distinct_state_input_and_output_addresses():
    assert "self.state_slot_in = i32(S)" in BRIDGE_SOURCE
    assert "self.state_slot_out = i32(S)" in BRIDGE_SOURCE
    assert "bufs.stage(bufs.state_slot_in, slot_arr)" in BRIDGE_SOURCE
    assert "bufs.stage(bufs.state_slot_out, slot_arr)" in BRIDGE_SOURCE


def test_vllm_eager_metadata_satisfies_split_state_slot_contract():
    assert "md = AttentionMetaData_DSV4(" in BRIDGE_SOURCE
    assert "md.state_slot_out = torch.from_numpy(slot_arr).to(device)" in BRIDGE_SOURCE
    assert "md.state_slot_in = md.state_slot_out.clone()" in BRIDGE_SOURCE
    assert "md.state_slot_mapping = md.state_slot_out" in BRIDGE_SOURCE


def test_vllm_decode_metadata_supplies_fused_swa_destination_rows():
    assert "self.swa_dest_row = i32(T)" in BRIDGE_SOURCE
    assert "bufs.stage(bufs.swa_dest_row, dest_np)" in BRIDGE_SOURCE
    assert "md.swa_dest_rows = _shared_vllm_swa_dest_rows(dest_gpu)" in BRIDGE_SOURCE
    assert "self.n_committed_per_token = i32(T)" in BRIDGE_SOURCE
    assert "md.n_committed_per_token = bufs.stage(" in BRIDGE_SOURCE
    assert "self.block_tables_per_token = i32(T, self.max_blocks)" in BRIDGE_SOURCE
    assert "md.block_tables_per_token = block_rows" in BRIDGE_SOURCE


def test_vllm_cache_binding_populates_current_swa_plane_contract():
    assert "attn.swa_plane = attn.swa_kv" in BRIDGE_SOURCE
    assert "attn.swa_window = swa_window" in BRIDGE_SOURCE
    assert "attn.swa_plane_rope = attn.swa_kv_rope" in BRIDGE_SOURCE


def test_vllm_prefill_uses_proxy_layout_specific_index_kernel():
    assert "write_v4_prefill_indices_fused" in BRIDGE_SOURCE
    assert "cache_size=cs" not in BRIDGE_SOURCE
