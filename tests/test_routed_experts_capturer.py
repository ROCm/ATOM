# SPDX-License-Identifier: MIT
"""Slot-indexed MoE route capture: scatter, gather, CUDA-graph pad slots."""

from dataclasses import fields

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atom.config import Config
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.model_ops.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    capture_bytes_per_kv_block,
    capture_pad_row_bytes,
    check_return_routed_experts,
    kv_slots_from_block_table,
    maybe_capture_routed_experts,
    topk_ids_from_triton_routing,
    trim_routed_experts,
)


@pytest.fixture(autouse=True)
def _reset_capturer():
    RoutedExpertsCapturer.reset()
    yield
    RoutedExpertsCapturer.reset()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_config_flag_defaults_off():
    names = {f.name for f in fields(Config)}
    assert "enable_return_routed_experts" in names
    field = next(f for f in fields(Config) if f.name == "enable_return_routed_experts")
    assert field.default is False


def test_dcp_pcp_pp_fail_closed():
    check_return_routed_experts(1, 1, 1)
    with pytest.raises(ValueError, match="decode_context_parallel_size"):
        check_return_routed_experts(2, 1)
    with pytest.raises(ValueError, match="prefill_context_parallel_size"):
        check_return_routed_experts(1, 2)
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        check_return_routed_experts(1, 1, 2)
    with pytest.raises(ValueError, match="KV transfer"):
        check_return_routed_experts(1, 1, 1, kv_transfer_config={"kv_connector": "moriio"})
    with pytest.raises(ValueError, match="RapidServe"):
        check_return_routed_experts(1, 1, 1, enable_rapidserve=True)
    check_return_routed_experts(1, 1, 1, kv_transfer_config={})


def test_rapidserve_decode_skip_is_rejected():
    """Decode skips KV alloc / capturer init; RapidServeModelRunner must refuse."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "atom"
        / "model_engine"
        / "model_runner.py"
    ).read_text()
    rapid = src.split("class RapidServeModelRunner", 1)[1]
    assert "def _refuse_routed_experts_capture" in rapid
    assert rapid.count("self._refuse_routed_experts_capture()") >= 2
    with pytest.raises(ValueError, match="never initializes"):
        check_return_routed_experts(1, 1, 1, enable_rapidserve=True)


def test_flag_off_leaves_field_absent():
    seq = Sequence([1, 2, 3], 3)
    assert seq.routed_experts is None
    ro = RequestOutput(request_id=0, output_tokens=[1], finished=True)
    assert ro.routed_experts is None


def test_interleaved_two_request_scatter_gather():
    """Two requests share one select_experts-style batch; gather by block table."""
    device = _device()
    block_size = 16
    capturer = RoutedExpertsCapturer.init(
        num_slots=64, num_layers=3, top_k=2, device=device
    )
    # Request A: blocks [0], tokens at slots 0,1 then later 2
    # Request B: blocks [1], tokens at slots 16,17
    # Interleaved batch 1: A0, B0, A1, B1
    slots_b1 = torch.tensor([0, 16, 1, 17], device=device)
    ids_b1 = torch.tensor(
        [[10, 11], [20, 21], [12, 13], [22, 23]],
        dtype=torch.int32,
        device=device,
    )
    capturer.capture(0, ids_b1, slot_mapping=slots_b1)
    capturer.capture(1, ids_b1 + 100, slot_mapping=slots_b1)

    # Later step: only A token 2, B idle
    slots_b2 = torch.tensor([2], device=device)
    ids_b2 = torch.tensor([[14, 15]], dtype=torch.int32, device=device)
    capturer.capture(0, ids_b2, slot_mapping=slots_b2)
    capturer.capture(1, ids_b2 + 100, slot_mapping=slots_b2)

    exported = capturer.export_batch(
        req_ids=[7, 9],
        block_tables=[[0], [1]],
        num_tokens_list=[3, 2],
        block_size=block_size,
    )
    a = exported[7]
    b = exported[9]
    assert a.shape == (3, 3, 2)
    assert b.shape == (2, 3, 2)
    assert a.dtype == np.int16
    np.testing.assert_array_equal(a[:, 0, :], [[10, 11], [12, 13], [14, 15]])
    np.testing.assert_array_equal(a[:, 1, :], [[110, 111], [112, 113], [114, 115]])
    np.testing.assert_array_equal(b[:, 0, :], [[20, 21], [22, 23]])


def test_length_contract_prompt_plus_completion_minus_one():
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=32, num_layers=1, top_k=1, device=device
    )
    prompt_len, completion_len = 4, 2
    # Routes exist for every forwarded token: all prompt tokens plus every
    # completion token except the last sampled id (never forwarded).
    n_routes = prompt_len + completion_len - 1
    slots = torch.arange(n_routes, device=device)
    ids = torch.arange(n_routes, device=device, dtype=torch.int32).unsqueeze(-1)
    capturer.capture(0, ids, slot_mapping=slots)
    exported = capturer.export_batch(
        [1], [[0]], [n_routes], block_size=16
    )[1]
    assert exported.shape[0] == n_routes
    assert exported.shape[0] == prompt_len + completion_len - 1


def test_graph_dummy_does_not_clobber_slot_zero():
    """Trailing -1 must not restore the pre-write value of physical slot 0."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=2, device=device
    )
    capturer.buffer[:, 0, :] = -7
    slots = torch.tensor([0, -1], device=device)
    ids = torch.tensor([[9, 8], [1, 2]], dtype=torch.int32, device=device)
    capturer.capture(0, ids, slot_mapping=slots)
    assert capturer.buffer[0, 0].tolist() == [9, 8]
    assert capturer.buffer[capturer._pad_slot, 0].tolist() == [1, 2]


def test_graph_dummy_negative_slots_ignored():
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=2, device=device
    )
    capturer.buffer[:, 0, :] = -7
    slots = torch.tensor([-1, -1, 3], device=device)
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device=device)
    capturer.capture(0, ids, slot_mapping=slots)
    # Pad rows must not clobber slot 0.
    got0 = capturer.buffer[0, 0].tolist()
    assert got0 == [-7, -7]
    got3 = capturer.buffer[3, 0].tolist()
    assert got3 == [5, 6]


def test_prefix_hit_reuses_physical_slots():
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=32, num_layers=1, top_k=2, device=device
    )
    slots = torch.tensor([0, 1, 2], device=device)
    ids = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32, device=device)
    capturer.capture(0, ids, slot_mapping=slots)
    # Request B prefix-hits A's blocks; gather the same physical slots.
    a = capturer.export_batch([1], [[0]], [3], block_size=16)[1]
    b = capturer.export_batch([2], [[0]], [3], block_size=16)[2]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:, 0, :], [[7, 8], [9, 10], [11, 12]])


def test_maybe_capture_skips_when_uninitialized():
    class _Layer:
        moe_capture_layer_id = 0

    maybe_capture_routed_experts(
        _Layer(), torch.zeros((2, 2), dtype=torch.int32)
    )
    assert RoutedExpertsCapturer.get() is None


def test_kv_slots_from_block_table():
    slots = kv_slots_from_block_table([4, 9], num_tokens=18, block_size=16)
    assert slots.tolist() == list(range(4 * 16, 4 * 16 + 16)) + [
        9 * 16,
        9 * 16 + 1,
    ]


def test_capture_page_bytes_are_budgeted_per_block():
    assert capture_bytes_per_kv_block(16, 58, 8) == 16 * 58 * 8 * 4
    assert capture_pad_row_bytes(58, 8) == 58 * 8 * 4
    from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
        PAGED_CLASS,
        page_pool,
        plan_pools,
    )

    kv = page_pool(1024)
    capture = page_pool(capture_bytes_per_kv_block(16, 2, 2))
    plan = plan_pools([kv, capture], available_bytes=1024 * 10 + 256, max_num_seqs=1)
    assert plan.entry_bytes[PAGED_CLASS] == 1024 + capture_bytes_per_kv_block(16, 2, 2)


def test_trim_routed_experts_before_request_output():
    routes = np.arange(6, dtype=np.int16).reshape(6, 1, 1)
    trimmed = trim_routed_experts(routes, num_tokens=4)
    assert trimmed.shape[0] == 3
    ro = RequestOutput(
        request_id=0,
        output_tokens=[1],
        finished=True,
        routed_experts=trimmed,
    )
    assert ro.routed_experts.shape[0] == 3


def test_triton_routing_ids_match_packed_histogram():
    """Reconstruct token-major ids from the same gather/hist Triton consumes."""

    class _Expt:
        token_offs_raw = torch.tensor([0, 0, 1, 1, 3, 5, 6], dtype=torch.int32)

    class _Routing:
        expt_data = _Expt()
        expt_hist = torch.tensor([0, 1, 0, 2, 2, 1], dtype=torch.int32)

    gather = torch.tensor([0, 2, 4, 1, 5, 3], dtype=torch.int32)
    ids = topk_ids_from_triton_routing(_Routing(), gather, num_tokens=3, topk=2)
    assert ids.tolist() == [[1, 4], [3, 5], [3, 4]]
