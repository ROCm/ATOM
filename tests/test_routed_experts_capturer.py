# SPDX-License-Identifier: MIT
"""Slot-indexed MoE route capture: scatter, gather, CUDA-graph pad slots."""

from dataclasses import fields

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atom.config import Config
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.model_ops.fused_moe import routed_experts_capturer as cap_mod
from atom.model_ops.fused_moe.routed_experts_capturer import (
    UNCAPTURED,
    RoutedExpertsCapturer,
    RoutedExpertsPatch,
    capture_bytes_per_kv_block,
    capture_pad_row_bytes,
    check_return_routed_experts,
    maybe_capture_routed_experts,
    prefix_slots,
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


def _segments(patch: RoutedExpertsPatch) -> list[tuple[int, int, np.ndarray]]:
    off = patch.offsets.tolist()
    return [
        (int(r), int(s), patch.rows[off[j] : off[j + 1]])
        for j, (r, s) in enumerate(zip(patch.req_ids, patch.starts))
    ]


def _store_take(capturer, slots, req_ids, starts, lens, **kw):
    return capturer.store_step(slots, req_ids, starts, lens, **kw).patch()


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
        check_return_routed_experts(
            1, 1, 1, kv_transfer_config={"kv_connector": "moriio"}
        )
    with pytest.raises(ValueError, match="RapidServe"):
        check_return_routed_experts(1, 1, 1, enable_rapidserve=True)
    with pytest.raises(ValueError, match="enable_dp_attention"):
        check_return_routed_experts(1, 1, 1, enable_dp_attention=True)
    check_return_routed_experts(1, 1, 1, kv_transfer_config={})
    check_return_routed_experts(1, 1, 1, enable_dp_attention=False)


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


def test_two_request_batch_order_segments():
    """A step's rows come back in batch token order, one segment per request."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=64, num_layers=3, top_k=2, device=device
    )
    # Request A on block 0, request B on block 1, two prefill tokens each.
    slots = torch.tensor([0, 1, 16, 17], device=device)
    ids = torch.tensor(
        [[10, 11], [12, 13], [20, 21], [22, 23]], dtype=torch.int32, device=device
    )
    capturer.capture(0, ids, slot_mapping=slots)
    capturer.capture(1, ids + 100, slot_mapping=slots)
    patch = _store_take(capturer, slots, [7, 9], [0, 0], [2, 2])
    assert patch.rows.dtype == np.int16
    assert patch.rows.shape == (4, 3, 2)
    (ra, sa, a), (rb, sb, b) = _segments(patch)
    assert (ra, sa, rb, sb) == (7, 0, 9, 0)
    np.testing.assert_array_equal(a[:, 0, :], [[10, 11], [12, 13]])
    np.testing.assert_array_equal(a[:, 1, :], [[110, 111], [112, 113]])
    np.testing.assert_array_equal(b[:, 0, :], [[20, 21], [22, 23]])

    # Next step decodes A alone at position 2.
    slots2 = torch.tensor([2], device=device)
    ids2 = torch.tensor([[14, 15]], dtype=torch.int32, device=device)
    capturer.capture(0, ids2, slot_mapping=slots2)
    patch2 = _store_take(capturer, slots2, [7], [2], [1])
    [(r, s, rows)] = _segments(patch2)
    assert (r, s) == (7, 2)
    np.testing.assert_array_equal(rows[:, 0, :], [[14, 15]])


def test_patches_rebuild_request_history():
    seq = Sequence([1, 2, 3], 3)
    seq.id = 7
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=32, num_layers=1, top_k=1, device=device
    )
    for start, n in ((0, 3), (3, 1), (4, 1)):
        slots = torch.arange(start, start + n, device=device)
        ids = slots.to(torch.int32).unsqueeze(-1) + 40
        capturer.capture(0, ids, slot_mapping=slots)
        patch = _store_take(capturer, slots, [7], [start], [n])
        for _r, s, rows in _segments(patch):
            seq.apply_routed_expert_patch(s, rows)
    assert seq.routed_experts[:, 0, 0].tolist() == [40, 41, 42, 43, 44]


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
    patch = _store_take(capturer, slots, [1], [0], [n_routes])
    assert patch.rows.shape[0] == prompt_len + completion_len - 1


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
    assert capturer.buffer[capturer.num_slots, 0].tolist() == [1, 2]


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


def test_prefix_hit_segment_reads_cached_slots():
    """B prefix-hits A's block: its cached rows come first, then its new token."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=32, num_layers=1, top_k=2, device=device
    )
    slots = torch.tensor([0, 1, 2], device=device)
    ids = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32, device=device)
    capturer.capture(0, ids, slot_mapping=slots)
    a = _store_take(capturer, slots, [1], [0], [3])

    b_slots = torch.tensor([3], device=device)
    capturer.capture(
        0,
        torch.tensor([[5, 6]], dtype=torch.int32, device=device),
        slot_mapping=b_slots,
    )
    prefix = (
        np.array([2]),
        np.array([0]),
        np.array([3]),
        prefix_slots([0, 1], 0, 3, 16),
    )
    b = _store_take(capturer, b_slots, [2], [3], [1], prefix=prefix)
    (r0, s0, cached), (r1, s1, new) = _segments(b)
    assert (r0, s0, r1, s1) == (2, 0, 2, 3)
    np.testing.assert_array_equal(cached, a.rows)
    np.testing.assert_array_equal(new[:, 0, :], [[5, 6]])


def test_maybe_capture_skips_when_uninitialized():
    class _Layer:
        moe_capture_layer_id = 0

    maybe_capture_routed_experts(_Layer(), torch.zeros((2, 2), dtype=torch.int32))
    assert RoutedExpertsCapturer.get() is None


def test_moe_outside_the_target_model_never_captures(monkeypatch):
    """A drafter's MoE has a `layer_id` but no `moe_capture_layer_id`."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=2, top_k=2, device=device
    )
    monkeypatch.setattr(
        cap_mod, "current_slot_mapping", lambda: torch.tensor([0], device=device)
    )

    class _DraftMoE:
        layer_id = 0

    maybe_capture_routed_experts(
        _DraftMoE(), torch.tensor([[5, 6]], dtype=torch.int32, device=device)
    )
    assert (capturer.buffer == UNCAPTURED).all()


def test_prefix_slots_match_gpu_formula():
    table = [4, 9, 2, 7]
    pos = torch.arange(50, dtype=torch.long)
    blocks = torch.tensor(table, dtype=torch.long)
    full = (blocks[pos // 16] * 16 + (pos % 16)).tolist()
    for lo, hi in ((0, 18), (5, 16), (15, 17), (16, 50), (33, 34)):
        assert prefix_slots(table, lo, hi, 16).tolist() == full[lo:hi]


def test_store_step_drops_graph_pad_tail():
    """Graph-padded slot_mapping: only the step's real tokens are returned."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=2, device=device
    )
    slots = torch.tensor([0, -1, -1], device=device)
    capturer.capture(
        0,
        torch.tensor([[9, 8], [1, 2], [3, 4]], dtype=torch.int32, device=device),
        slot_mapping=slots,
    )
    patch = _store_take(capturer, slots, [1], [0], [1])
    np.testing.assert_array_equal(patch.rows[:, 0, :], [[9, 8]])


def test_uncaptured_rows_are_minus_one():
    """Layers or slots no MoE layer wrote read back as UNCAPTURED, not expert 0."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=2, top_k=2, device=device
    )
    slots = torch.tensor([0, 1], device=device)
    capturer.capture(
        0, torch.tensor([[7, 8], [9, 10]], dtype=torch.int32, device=device), slots
    )
    patch = _store_take(capturer, slots, [1], [0], [2])
    np.testing.assert_array_equal(patch.rows[:, 0, :], [[7, 8], [9, 10]])
    assert (patch.rows[:, 1, :] == UNCAPTURED).all()


def test_short_slot_mapping_pads_uncaptured():
    """A metadata builder exposing fewer slots than tokens yields -1 rows."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=1, device=device
    )
    slots = torch.tensor([2], device=device)
    capturer.capture(0, torch.tensor([[5]], dtype=torch.int32, device=device), slots)
    patch = _store_take(capturer, slots, [1, 2], [0, 0], [2, 1])
    assert patch.rows[:, 0, 0].tolist() == [5, UNCAPTURED, UNCAPTURED]
    assert patch.offsets.tolist() == [0, 2, 3]


def test_empty_step_stores_nothing():
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=2, device="cpu"
    )
    assert capturer.store_step(torch.tensor([], dtype=torch.long), [], [], []) is None
    assert capturer.store_step(None, [1], [0], [1]) is None


def test_stored_steps_are_independent():
    """A later store neither replaces nor reorders an earlier one."""
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=1, device="cpu"
    )
    capturer.capture(0, torch.tensor([[4]], dtype=torch.int32), torch.tensor([1]))
    a = capturer.store_step(torch.tensor([1]), [1], [0], [1])
    capturer.capture(0, torch.tensor([[6]], dtype=torch.int32), torch.tensor([2]))
    b = capturer.store_step(torch.tensor([2]), [2], [0], [1])
    assert b.patch().req_ids.tolist() == [2]
    assert a.patch().req_ids.tolist() == [1]
    assert a.patch().rows[:, 0, 0].tolist() == [4]
    assert b.patch().rows[:, 0, 0].tolist() == [6]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for copy stream")
def test_async_store_snapshots_slots_before_staging_reuse():
    """The next forward rewrites slot_mapping in place on the compute stream;
    a stalled copy stream must still return this step's rows."""
    device = torch.device("cuda")
    capturer = RoutedExpertsCapturer.init(
        num_slots=8, num_layers=1, top_k=2, device=device
    )
    slots = torch.tensor([1], device=device)
    capturer.capture(
        0, torch.tensor([[7, 8]], dtype=torch.int32, device=device), slot_mapping=slots
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000_000)
    pending = capturer.store_step(slots, [1], [0], [1], stream=stream)
    slots.fill_(2)
    capturer.capture(
        0, torch.tensor([[3, 4]], dtype=torch.int32, device=device), slot_mapping=slots
    )
    np.testing.assert_array_equal(pending.patch().rows[:, 0, :], [[7, 8]])


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


def test_lazy_wrapper_fused_moe_is_detected(monkeypatch):
    """Plugin LazyMoEWrapper instances are not isinstance(FusedMoE).

    Do not import ``atom.model_ops.moe``: FusedMoE pulls AITER, which needs a
    GPU (rocminfo) even to load the class.
    """
    import sys
    import types

    from torch import nn

    from atom.model_ops.fused_moe.routed_experts_capturer import is_fused_moe_module

    class InnerFusedMoE(nn.Module):
        pass

    class LazyMoEWrapper(InnerFusedMoE):
        def __new__(cls, *args, **kwargs):
            return InnerFusedMoE()

    fake_moe = types.ModuleType("atom.model_ops.moe")
    fake_moe.FusedMoE = LazyMoEWrapper
    monkeypatch.setitem(sys.modules, "atom.model_ops.moe", fake_moe)

    module = InnerFusedMoE()
    assert is_fused_moe_module(module)
    assert not isinstance(module, LazyMoEWrapper)
    assert not is_fused_moe_module(nn.Identity())


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


def test_in_place_slot_rewrite_between_forwards():
    """The staging slot buffer is rewritten in place between forwards; each
    forward's layers write and read that forward's slots."""
    device = _device()
    capturer = RoutedExpertsCapturer.init(
        num_slots=32, num_layers=3, top_k=2, device=device
    )
    staging = torch.tensor([0, 1, -1], device=device)
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device=device)
    for layer in range(3):
        capturer.capture(layer, ids + 10 * layer, slot_mapping=staging)
    first = _store_take(capturer, staging, [0], [0], [2])
    staging.copy_(torch.tensor([2, 3, -1], device=device))
    for layer in range(3):
        capturer.capture(layer, ids + 100 + 10 * layer, slot_mapping=staging)
    second = _store_take(capturer, staging, [0], [2], [2])
    for layer in range(3):
        np.testing.assert_array_equal(
            first.rows[:, layer], (ids[:2] + 10 * layer).cpu()
        )
        np.testing.assert_array_equal(
            second.rows[:, layer], (ids[:2] + 100 + 10 * layer).cpu()
        )
    # Pad slot (-1) lands in the sacrificial row, never a real slot.
    assert (capturer.buffer[4:32] == UNCAPTURED).all()


def test_apply_patch_extends_none_and_appends():
    seq = Sequence([1, 2, 3], 3)
    assert seq.routed_experts is None
    first = np.arange(4, dtype=np.int16).reshape(2, 1, 2)
    seq.apply_routed_expert_patch(0, first)
    np.testing.assert_array_equal(seq.routed_experts, first)
    second = np.arange(4, 6, dtype=np.int16).reshape(1, 1, 2)
    seq.apply_routed_expert_patch(2, second)
    assert seq.routed_expert_rows == 3
    np.testing.assert_array_equal(
        seq.routed_experts, np.concatenate([first, second], axis=0)
    )


def test_apply_patch_gap_fills_uncaptured():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.zeros((2, 1, 1), dtype=np.int16))
    seq.apply_routed_expert_patch(4, np.full((1, 1, 1), 5, dtype=np.int16))
    assert seq.routed_experts[:, 0, 0].tolist() == [0, 0, -1, -1, 5]


def test_apply_patch_replaces_suffix():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.arange(6, dtype=np.int16).reshape(3, 1, 2))
    replacement = np.array([[[9, 8]]], dtype=np.int16)
    seq.apply_routed_expert_patch(1, replacement)
    assert seq.routed_expert_rows == 2
    np.testing.assert_array_equal(seq.routed_experts[1:], replacement)


def test_truncate_routed_experts_drops_tail():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.arange(8, dtype=np.int16).reshape(4, 1, 2))
    seq.truncate_routed_experts(2)
    assert seq.routed_experts.shape[0] == 2
    seq.truncate_routed_experts(0)
    assert seq.routed_experts is None


def test_apply_patch_deferred_overlap_keeps_prefix():
    """PP=1 deferred attach reports start = have-1; overwrite last row, keep prefix."""
    seq = Sequence([1, 2, 3], 3)
    first = np.arange(6, dtype=np.int16).reshape(3, 1, 2)
    seq.apply_routed_expert_patch(0, first)
    nxt = np.array([[[90, 91], [92, 93]]], dtype=np.int16).reshape(2, 1, 2)
    seq.apply_routed_expert_patch(seq.routed_expert_rows - 1, nxt)
    assert seq.routed_expert_rows == 4
    np.testing.assert_array_equal(seq.routed_experts[:2], first[:2])
    np.testing.assert_array_equal(seq.routed_experts[2:], nxt)


def test_apply_patch_after_empty_full_array_grows():
    seq = Sequence([1, 2, 3], 3)
    seq.routed_experts = np.zeros((0, 2, 2), dtype=np.int16)
    rows = np.arange(12, dtype=np.int16).reshape(3, 2, 2)
    seq.apply_routed_expert_patch(0, rows)
    np.testing.assert_array_equal(seq.routed_experts, rows)


def test_routed_experts_setter_does_not_alias_caller_array():
    seq = Sequence([1, 2, 3], 3)
    full = np.arange(12, dtype=np.int16).reshape(3, 2, 2)
    seq.routed_experts = full
    seq.apply_routed_expert_patch(2, np.full((1, 2, 2), 99, dtype=np.int16))
    np.testing.assert_array_equal(full[2], [[8, 9], [10, 11]])
    np.testing.assert_array_equal(seq.routed_experts[2], [[99, 99], [99, 99]])


def test_apply_patch_grows_capacity_across_many_decode_steps():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.zeros((8, 2, 2), dtype=np.int16))
    for i in range(8, 64):
        seq.apply_routed_expert_patch(
            seq.routed_expert_rows - 1,
            np.full((2, 2, 2), i, dtype=np.int16),
        )
    assert seq.routed_expert_rows == 64
    out = seq.routed_experts
    assert out.shape == (64, 2, 2)
    np.testing.assert_array_equal(out[-2:], np.full((2, 2, 2), 63, dtype=np.int16))
    assert seq._routed_expert_buf.shape[0] >= 64


def _grown_seq(rows: int) -> Sequence:
    seq = Sequence([1, 2, 3], 3)
    for i in range(rows):
        seq.apply_routed_expert_patch(i, np.full((1, 4, 2), i, dtype=np.int16))
    return seq


def test_finished_routes_are_read_without_copying():
    seq = _grown_seq(9)
    seq.finish_routed_experts(7)
    first, second = seq.routed_experts, seq.routed_experts
    assert first is second
    assert first.shape == (7, 4, 2)
    assert not first.flags.writeable


def test_finished_routes_pickle_without_the_spare_capacity():
    import pickle

    seq = _grown_seq(9)
    assert seq._routed_expert_buf.shape[0] > 9
    live = seq.routed_experts.nbytes
    seq.finish_routed_experts(9)
    restored = pickle.loads(pickle.dumps(seq))
    assert restored._routed_expert_buf.nbytes == live
    np.testing.assert_array_equal(restored.routed_experts, seq.routed_experts)


def test_late_patch_after_finish_is_dropped():
    """A deferred step scheduled before the finish still reports its routes."""
    seq = _grown_seq(5)
    seq.finish_routed_experts(4)
    seq.apply_routed_expert_patch(4, np.full((2, 4, 2), 99, dtype=np.int16))
    assert seq.routed_expert_rows == 4
    assert (seq.routed_experts != 99).all()


def test_running_routes_are_still_copies():
    seq = _grown_seq(3)
    seq.routed_experts[0] = 99
    assert seq.routed_experts[0, 0, 0] == 0


def test_incremental_decode_payload_smaller_than_full_history():
    import pickle

    layers, top_k, ctx, batch = 48, 8, 512, 8
    full = {i: np.zeros((ctx, layers, top_k), dtype=np.int16) for i in range(batch)}
    patch = RoutedExpertsPatch(
        np.arange(batch),
        np.full(batch, ctx - 1),
        np.arange(batch + 1),
        np.zeros((batch, layers, top_k), dtype=np.int16),
    )
    full_size = len(pickle.dumps(full))
    patch_size = len(pickle.dumps(patch))
    assert patch_size * 20 < full_size
