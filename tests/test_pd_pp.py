# SPDX-License-Identifier: MIT
# PD-disaggregation + pipeline-parallel unit tests (GPU-free).

import os
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure aiter.dist.parallel_state exposes symbols mooncake_connector needs.
_ps = sys.modules.get("aiter.dist.parallel_state")
if _ps is not None:
    for _fn in ("get_dp_group", "get_tp_group"):
        if not hasattr(_ps, _fn):
            setattr(_ps, _fn, MagicMock())
else:
    _aiter_pkg = types.ModuleType("aiter")
    _aiter_pkg.__path__ = []
    sys.modules.setdefault("aiter", _aiter_pkg)
    _dist = types.ModuleType("aiter.dist")
    _dist.__path__ = []
    sys.modules.setdefault("aiter.dist", _dist)
    _ps_stub = types.ModuleType("aiter.dist.parallel_state")
    for _fn in ("get_dp_group", "get_tp_group"):
        setattr(_ps_stub, _fn, MagicMock())
    sys.modules.setdefault("aiter.dist.parallel_state", _ps_stub)

from atom.kv_transfer.disaggregation.port_offset import (
    consumer_region_indices,
    side_channel_port_offset,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata

# ---------------------------------------------------------------------------
# pp-aware side-channel port offset
# ---------------------------------------------------------------------------


def test_port_offset_pp1_matches_legacy():
    # pp_rank=0, pp_size=1 must reproduce the old dp_rank*tp_size + tp_rank.
    for dp_size in (1, 2, 4):
        for tp_size in (1, 2, 8):
            for dp_rank in range(dp_size):
                for tp_rank in range(tp_size):
                    legacy = dp_rank * tp_size + tp_rank
                    assert side_channel_port_offset(dp_rank, tp_rank, tp_size) == legacy
                    assert (
                        side_channel_port_offset(
                            dp_rank, tp_rank, tp_size, 0, 1, dp_size
                        )
                        == legacy
                    )


def test_port_offset_unique_across_pp_dp_tp():
    pp_size, dp_size, tp_size = 4, 2, 2
    seen = {}
    for pp_rank in range(pp_size):
        for dp_rank in range(dp_size):
            for tp_rank in range(tp_size):
                off = side_channel_port_offset(
                    dp_rank, tp_rank, tp_size, pp_rank, pp_size, dp_size
                )
                key = (pp_rank, dp_rank, tp_rank)
                assert off not in seen, f"collision {key} vs {seen.get(off)}"
                seen[off] = key
    # Dense packing: offsets fill [0, pp*dp*tp).
    assert sorted(seen) == list(range(pp_size * dp_size * tp_size))


def test_port_offset_pp4_tp1_no_collision():
    offs = [side_channel_port_offset(0, 0, 1, pp_rank, 4, 1) for pp_rank in range(4)]
    assert offs == [0, 1, 2, 3]


def test_consumer_targets_every_producer_stage_port():
    # The ports a consumer computes for stages 0..pp-1 must equal the ports each
    # producer stage binds, or a stage never receives its write_request.
    base = 6301
    args = {
        "remote_dp_rank": 0,
        "remote_tp_rank": 0,
        "remote_tp_size": 1,
        "remote_dp_size": 1,
    }
    pp_size = 4
    consumer_ports = {
        base
        + side_channel_port_offset(
            args["remote_dp_rank"],
            args["remote_tp_rank"],
            args["remote_tp_size"],
            stage,
            pp_size,
            args["remote_dp_size"],
        )
        for stage in range(pp_size)
    }
    assert consumer_ports == {base + i for i in range(pp_size)}
    assert len(consumer_ports) == pp_size


# ---------------------------------------------------------------------------
# remote_pp_size plumbing
# ---------------------------------------------------------------------------


def test_build_req_meta_reads_remote_pp_size():
    meta = ConnectorMetadata._build_req_meta(
        req_id="r0",
        local_block_ids=[0, 1],
        kv_transfer_params={
            "remote_block_ids": [5, 6],
            "remote_engine_id": "eng",
            "remote_host": "10.0.0.1",
            "remote_port": 41000,
            "remote_handshake_port": 6301,
            "tp_size": 1,
            "remote_pp_size": 4,
        },
    )
    assert meta.remote_pp_size == 4


def test_build_req_meta_keeps_remote_and_local_state_slots_distinct():
    meta = ConnectorMetadata._build_req_meta(
        req_id="r0",
        local_block_ids=[0],
        kv_transfer_params={
            "remote_block_ids": [5],
            "remote_host": "h",
            "remote_handshake_port": 6301,
            "tp_size": 1,
            "local_slot_index": 11,
            "remote_slot_index": 7,
        },
    )
    assert meta.local_slot_index == 11
    assert meta.remote_slot_index == 7


def test_build_req_meta_defaults_pp_size_one():
    meta = ConnectorMetadata._build_req_meta(
        req_id="r0",
        local_block_ids=[0],
        kv_transfer_params={
            "remote_block_ids": [5],
            "remote_host": "h",
            "remote_handshake_port": 6301,
            "tp_size": 1,
        },
    )
    assert meta.remote_pp_size == 1


def test_producer_advertises_remote_pp_size():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = object.__new__(mc.MooncakeConnectorScheduler)
    sched.pp_size = 4
    sched.tp_size = 1
    sched.dp_rank = 0
    sched.engine_id = "eng"
    sched.host_ip = "10.0.0.1"
    sched.handshake_port = 40000
    sched.base_handshake_port = 6301
    sched.is_producer = True

    seq = SimpleNamespace(
        output_tokens=[7],
        spec_token_ids=None,
        block_table=[1, 2, 3],
        id=99,
        state_slot=7,
        state_slots=[],
        kv_transfer_params_output=None,
    )
    mc.MooncakeConnectorScheduler.request_finished(sched, seq)
    assert seq.kv_transfer_params_output["remote_pp_size"] == 4
    assert seq.kv_transfer_params_output["remote_block_ids"] == [1, 2, 3]
    assert seq.kv_transfer_params_output["remote_slot_index"] == 7


@pytest.mark.parametrize(
    "producer_slots",
    [
        # Compatibility form emitted by older producers.
        {"local_slot_index": 7},
        # The explicit source slot must win over the legacy field.
        {"local_slot_index": 3, "remote_slot_index": 7},
    ],
)
def test_consumer_preserves_producer_final_slot_before_setting_destination(
    producer_slots,
):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = object.__new__(mc.MooncakeConnectorScheduler)
    sched.is_producer = False
    sched._reqs_need_recv = {}
    sched._reqs_need_save = {}
    sched.transfer_id_to_request_id = {}
    sched.request_id_to_transfer_id = {}

    seq = SimpleNamespace(
        id="decode-1",
        state_slot=11,
        block_table=[1, 2],
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "prefill-1",
            **producer_slots,
        },
    )
    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["remote_slot_index"] == 7
    assert seq.kv_transfer_params["local_slot_index"] == 11
    meta = sched.build_connector_meta().reqs_to_recv["decode-1"]
    assert meta.remote_slot_index == 7
    assert meta.local_slot_index == 11


# ---------------------------------------------------------------------------
# Mooncake transport selection
# ---------------------------------------------------------------------------


def test_mooncake_tcp_disables_rdma_device_even_when_configured():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    assert mc._select_ib_device("tcp", "rdma0", None) == ""
    assert mc._select_ib_device(" TCP ", "ionic_0", None) == ""


def test_mooncake_tcp_forces_transfer_engine_transport(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    mc._configure_mooncake_transport(" TCP ")
    assert os.environ["MC_FORCE_TCP"] == "true"


def test_mooncake_rdma_preserves_explicit_device():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    assert mc._select_ib_device("rdma", "ionic_3", None) == "ionic_3"


def test_mooncake_rdma_auto_selects_from_physical_gpu(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"auto{idx}")
    assert mc._select_ib_device("rdma", "", 5) == "auto5"


def test_mooncake_rdma_requires_gpu_index_without_explicit_device():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    with pytest.raises(ValueError, match="physical GPU index"):
        mc._select_ib_device("rdma", "", None)


# ---------------------------------------------------------------------------
# Producer per-layer region mapping (consumer_region_indices)
# ---------------------------------------------------------------------------


def _starts(partitions):
    """Global start layer of each stage, given a per-stage layer-count list."""
    starts, acc = [], 0
    for p in partitions:
        starts.append(acc)
        acc += p
    return starts


def test_region_map_identity_when_pp1():
    assert consumer_region_indices(156, 78, 0, 78, 1) == list(range(156))


def test_region_map_identity_when_empty():
    assert consumer_region_indices(0, 0, 5, 78, 4) == []


def test_region_map_group_major_single_group():
    # 1 region/layer, stage of 20 layers @ global start 18 → consumer 18..37.
    assert consumer_region_indices(20, 20, 18, 78, 4) == list(range(18, 38))


def test_region_map_group_major_two_groups_mla():
    # MLA: 2 groups [kv, index], stage=20 layers @ start 18, N=78.
    got = consumer_region_indices(40, 20, 18, 78, 4)
    assert got[:20] == list(range(18, 38))
    assert got[20:] == list(range(78 + 18, 78 + 38))


def test_region_map_undefined_when_not_multiple():
    assert consumer_region_indices(41, 20, 18, 78, 4) is None


def test_region_map_stages_tile_consumer_no_overlap():
    # GLM-5.2: 78 layers, PP4 partition [18,20,20,20], 2 groups (kv + index).
    partitions, num_hidden, groups = [18, 20, 20, 20], 78, 2
    covered = []
    for start, n_local in zip(_starts(partitions), partitions):
        covered.extend(
            consumer_region_indices(n_local * groups, n_local, start, num_hidden, 4)
        )
    total = num_hidden * groups
    assert sorted(covered) == list(range(total))
    assert len(covered) == len(set(covered))  # no overlap


def test_region_map_group_major_beats_naive_offset():
    # Regression guard: a naive additive offset (start_layer*groups + i) would
    # misroute group-major layouts. Stage1's index-group region 0 (local idx 20)
    # must land in the consumer's index group (>=78), not at 36+20=56 (kv group).
    cmap = consumer_region_indices(40, 20, 18, 78, 4)
    assert cmap[20] == 78 + 18
    assert cmap[20] != 56


# ---------------------------------------------------------------------------
# Consumer write-done completion counting + nonce validation
# ---------------------------------------------------------------------------


def _make_connector(**overrides):
    """Real MooncakeConnector with only the fields _record_write_done touches.

    Bypasses __init__ (RDMA/ZMQ). An empty _release_targets makes the real
    _send_release a no-op on completion, so the genuine method is exercised.
    """
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = object.__new__(mc.MooncakeConnector)
    conn._completion_lock = threading.Lock()
    conn._fence_lock = threading.Lock()
    conn._pending_recv_expected = {}
    conn._pending_recv_stages = {}
    conn._pending_recv_nonce = {}
    conn._pending_recv = set()
    conn._pending_recv_blocks = {}
    conn._pending_recv_slots = {}
    conn._blocks_pending_fence = []
    conn.done_recving = set()
    conn._scatter_slot = None
    conn._release_targets = {}
    for k, v in overrides.items():
        setattr(conn, k, v)
    return conn


def _make_transfer_connector(prefill_data, captured):
    conn = _make_connector()
    conn.pp_size = 1
    conn.pp_rank = 0
    conn.tp_size = 1
    conn.transfer_engine = object()
    conn._transfer_refcount_lock = threading.Lock()
    conn._transfer_refcount = {}
    conn._completed_prefills_lock = threading.Lock()
    conn._completed_prefills = {}
    conn.done_sending = set()
    conn._wait_for_prefill_data = lambda _transfer_id: dict(prefill_data)
    conn._execute_block_slot_transfer = (
        lambda request, target, src, dst, data, req_id: captured.append(
            (request, target, src, dst, data, req_id)
        )
        or True
    )
    conn._send_write_done = lambda *_args: None
    return conn


def _stateful_write_request(**overrides):
    request = {
        "request_id": "decode-1",
        "transfer_id": "prefill-1",
        "consumer_host": "10.0.0.2",
        "consumer_rpc_port": 42000,
        "dst_block_ids": [9],
        "notify_host": "10.0.0.2",
        "notify_port": 42001,
        "consumer_tp_size": 1,
        "write_nonce": 1,
        "has_slot_regions": True,
        "src_slot_index": 7,
        "src_swa_block_ids": [7],
        "dst_slot_index": 11,
    }
    request.update(overrides)
    return request


def test_tp_transfer_uses_final_source_slot_after_checkpoint_move():
    captured = []
    conn = _make_transfer_connector(
        {"block_ids": [3], "slot_index": 3, "swa_block_ids": [3]}, captured
    )

    conn._execute_transfer(_stateful_write_request())

    assert len(captured) == 1
    transferred = captured[0][4]
    assert transferred["slot_index"] == 7
    assert transferred["swa_block_ids"] == [7]


@pytest.mark.parametrize("invalid_slot", [-1, None, "invalid"])
def test_stateful_transfer_without_final_source_slot_fails_closed(
    caplog, invalid_slot
):
    captured = []
    conn = _make_transfer_connector(
        {"block_ids": [3], "slot_index": 3, "swa_block_ids": [3]}, captured
    )

    conn._execute_transfer(_stateful_write_request(src_slot_index=invalid_slot))

    assert captured == []
    assert "missing the producer's final source slot" in caplog.text


def test_write_done_correct_nonce_accepted():
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_nonce["r1"] = 4242
    assert conn._record_write_done("r1", 0, 0, 4242)
    assert "r1" in conn.done_recving


def test_write_done_wrong_nonce_rejected():
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_nonce["r1"] = 12345
    assert not conn._record_write_done("r1", 0, 0, 99999)
    assert "r1" not in conn.done_recving
    assert "r1" in conn._pending_recv_expected


def test_write_done_zero_nonce_skips_validation():
    """Old producers that don't send a nonce (default 0) are accepted."""
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_nonce["r1"] = 0
    assert conn._record_write_done("r1", 0, 0, 0)
    assert "r1" in conn.done_recving


def test_write_done_missing_nonce_from_old_producer_rejected():
    """Consumer expects a nonce but the producer sends 0 → rejected."""
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_nonce["r1"] = 42
    assert not conn._record_write_done("r1", 0, 0, 0)
    assert "r1" not in conn.done_recving


def test_write_done_nonce_cleaned_up_on_completion():
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_nonce["r1"] = 777
    conn._record_write_done("r1", 0, 0, 777)
    assert "r1" not in conn._pending_recv_nonce


def test_write_done_pp_only_dedup():
    """PP4, TP symmetric: 4 distinct pp_ranks needed to finalize."""
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 4
    for pp in range(3):
        assert not conn._record_write_done("r1", pp, 0, 0)
    assert conn._record_write_done("r1", 3, 0, 0)
    assert "r1" in conn.done_recving


def test_write_done_duplicate_pp_rank_ignored():
    """Same pp_rank resent (reliability) is not double-counted."""
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 2
    assert not conn._record_write_done("r1", 0, 0, 0)
    assert not conn._record_write_done("r1", 0, 0, 0)
    assert conn._record_write_done("r1", 1, 0, 0)


def test_write_done_tp_asymmetric_dedup():
    """PP2 x TP fan-in 2: 4 distinct (pp, tp) pairs needed."""
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 4
    assert not conn._record_write_done("r1", 0, 0, 0)
    assert not conn._record_write_done("r1", 0, 1, 0)
    assert not conn._record_write_done("r1", 1, 0, 0)
    assert conn._record_write_done("r1", 1, 1, 0)
    assert "r1" in conn.done_recving


def test_write_done_unknown_request_ignored():
    conn = _make_connector()
    assert not conn._record_write_done("unknown", 0, 0, 0)


def test_write_done_late_duplicate_after_completion_ignored():
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    assert conn._record_write_done("r1", 0, 0, 0)
    assert not conn._record_write_done("r1", 0, 0, 0)


def test_write_done_blocks_fenced_on_completion():
    conn = _make_connector()
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_blocks["r1"] = [10, 20, 30]
    conn._record_write_done("r1", 0, 0, 0)
    assert conn._blocks_pending_fence == [10, 20, 30]
