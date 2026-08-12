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
