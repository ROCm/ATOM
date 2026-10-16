# SPDX-License-Identifier: MIT
# PD-disaggregation + pipeline-parallel unit tests (GPU-free).

import logging
import os
import subprocess
import sys
import threading
import time
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure aiter.dist.parallel_state exposes symbols mooncake_connector needs.
# The top-level name is taken back down below.
_ps = sys.modules.get("aiter.dist.parallel_state")
_stubbed: list[str] = []
if _ps is not None:
    for _fn in ("get_dp_group", "get_tp_group"):
        if not hasattr(_ps, _fn):
            setattr(_ps, _fn, MagicMock())
else:
    _aiter_pkg = types.ModuleType("aiter")
    _aiter_pkg.__path__ = []
    _dist = types.ModuleType("aiter.dist")
    _dist.__path__ = []
    _ps_stub = types.ModuleType("aiter.dist.parallel_state")
    for _fn in ("get_dp_group", "get_tp_group"):
        setattr(_ps_stub, _fn, MagicMock())
    for _name, _mod in (
        ("aiter", _aiter_pkg),
        ("aiter.dist", _dist),
        ("aiter.dist.parallel_state", _ps_stub),
    ):
        if _name not in sys.modules:
            sys.modules[_name] = _mod
            _stubbed.append(_name)

from atom.kv_transfer.disaggregation.port_offset import (
    consumer_region_indices,
    side_channel_port_offset,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata

# Drop the top-level name now that the imports above are bound. The submodules
# stay -- the connectors reach for them lazily, at call time -- but `aiter`
# itself has no reader here, and leaving it is what made
# `pytest.importorskip("aiter")` succeed in every module collected after this
# one: the skip did not fire, and the real `from aiter import ...` inside the
# module under test then raised ImportError during collection, which takes the
# whole run down instead of skipping one file.
if "aiter" in _stubbed:
    del sys.modules["aiter"]


@pytest.fixture(autouse=True)
def _report_failed_receives_at_once(monkeypatch):
    """Report a failed receive at once; the hold has tests of its own."""
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")


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
# remote topology metadata plumbing
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
    sched.hash_block_size = 64
    sched.block_size = 64
    sched.dcp_size = 1
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
    assert seq.kv_transfer_params_output["hash_block_size"] == 64
    assert seq.kv_transfer_params_output["block_size"] == 64
    assert seq.kv_transfer_params_output["dcp_size"] == 1
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
    sched = _mooncake_consumer_scheduler(mc)

    # Nothing locally cached, so the incremental-transfer branch resolves to a
    # full transfer and leaves the slot bookkeeping under test alone.
    seq = SimpleNamespace(
        id="decode-1",
        state_slot=11,
        block_table=[1, 2],
        has_per_req_cache=False,
        num_cached_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "prefill-1",
            "hash_block_size": 64,
            **producer_slots,
        },
    )
    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["remote_slot_index"] == 7
    assert seq.kv_transfer_params["local_slot_index"] == 11
    meta = sched.build_connector_meta().reqs_to_recv["decode-1"]
    assert meta.remote_slot_index == 7
    assert meta.local_slot_index == 11


def _mooncake_consumer_scheduler(mc, block_size=64, dcp_size=1):
    sched = object.__new__(mc.MooncakeConnectorScheduler)
    sched.is_producer = False
    sched.block_size = block_size
    sched.dcp_size = dcp_size
    sched.hash_block_size = block_size * dcp_size
    sched.request_id_to_transfer_id = {}
    sched.transfer_id_to_request_id = {}
    sched._reqs_need_recv = {}
    sched._reqs_need_save = {}
    return sched


def _remote_prefill_seq(remote_hash_block_size):
    params = {"do_remote_prefill": True}
    if remote_hash_block_size is not None:
        params["hash_block_size"] = remote_hash_block_size
    return SimpleNamespace(
        id=99,
        kv_transfer_params=params,
        block_table=[1, 2, 3],
        per_req_cache_group=-1,
        has_per_req_cache=False,
        num_cached_tokens=128,
    )


def test_matching_block_size_enables_incremental_transfer():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc)
    seq = _remote_prefill_seq(remote_hash_block_size=64)

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 2
    assert seq.kv_transfer_params["src_block_skip_factor"] == 1


def test_dcp_consumer_stays_incremental_against_a_non_dcp_producer():
    # CPP prefill (dcp=1, 16-token blocks) -> DCP decode (dcp=4). The consumer
    # addresses 64-token virtual blocks, so the two sides' hash_block_size
    # differ by exactly dcp_size; the block slicing applies that factor itself.
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc, block_size=16, dcp_size=4)
    seq = _remote_prefill_seq(remote_hash_block_size=16)

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 2
    assert seq.kv_transfer_params["src_block_skip_factor"] == 4


def test_symmetric_dcp_keeps_incremental_transfer():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc, block_size=16, dcp_size=4)
    seq = _remote_prefill_seq(remote_hash_block_size=64)
    seq.kv_transfer_params["block_size"] = 16
    seq.kv_transfer_params["dcp_size"] = 4

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 2
    assert seq.kv_transfer_params["src_block_skip_factor"] == 1


def test_mismatched_producer_block_size_disables_incremental_transfer():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc, block_size=16, dcp_size=4)
    seq = _remote_prefill_seq(remote_hash_block_size=16)
    seq.kv_transfer_params["block_size"] = 8
    seq.kv_transfer_params["dcp_size"] = 2

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 0
    assert seq.kv_transfer_params["src_block_skip_factor"] == 1


@pytest.mark.parametrize("remote_hash_block_size", [32, None])
def test_mismatched_or_missing_block_size_forces_full_transfer(
    remote_hash_block_size, caplog
):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc)
    seq = _remote_prefill_seq(remote_hash_block_size)

    with caplog.at_level(logging.WARNING, logger="atom"):
        sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 0
    assert seq.kv_transfer_params["src_block_skip_factor"] == 1
    assert "falling back to full transfer" in caplog.text


# ---------------------------------------------------------------------------
# Mooncake transport selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize(
    "protocol,configured_device,expected_device",
    [
        ("rdma", "", "rdma2"),
        ("rdma", "ionic_4,ionic_0", "ionic_4,ionic_0"),
        ("tcp", "rdma2", ""),
    ],
)
def test_mooncake_control_address_is_independent_of_rdma_device(
    monkeypatch, role, protocol, configured_device, expected_device
):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    host_ip = "10.19.0.140"
    monkeypatch.setenv("ATOM_HOST_IP", host_ip)
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE", raising=False)
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(mc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mc, "_ib_device_exists", lambda name: True)
    monkeypatch.setattr(
        mc, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=8)
    )
    monkeypatch.setattr(
        mc, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )

    # Reproduce a host with a separate RoCE network. The old constructor
    # replaced ATOM_HOST_IP with this HCA address, breaking TCP handshakes.
    original_listdir = os.listdir
    monkeypatch.setattr(
        os,
        "listdir",
        lambda path: (
            ["tw-eth2"]
            if str(path).startswith("/sys/class/infiniband/")
            else original_listdir(path)
        ),
    )
    original_check_output = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kwargs: (
            "2: tw-eth2 inet 10.103.40.121/31 scope global tw-eth2\n"
            if cmd[:4] == ["ip", "-o", "-4", "addr"]
            else original_check_output(cmd, **kwargs)
        ),
    )

    engine = MagicMock()
    engine.initialize.return_value = 0
    engine.get_rpc_port.return_value = 16578
    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(mc, "TransferEngine", lambda: engine, raising=False)
    monkeypatch.setattr(mc, "get_open_port", lambda: 41000)
    monkeypatch.setattr(mc.zmq, "Context", MagicMock())
    monkeypatch.setattr(mc, "ThreadPoolExecutor", MagicMock())
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_rank=0),
        pipeline_parallel_size=1,
        hf_config=SimpleNamespace(num_hidden_layers=2),
        kv_cache_block_size=16,
        decode_context_parallel_size=1,
        dcp_config=SimpleNamespace(interleave_size=1),
        kv_transfer_config={
            "kv_role": role,
            "protocol": protocol,
            "ib_device": configured_device,
        },
    )
    conn = mc.MooncakeConnector(config)

    engine.initialize.assert_called_once_with(
        host_ip, "P2PHANDSHAKE", protocol, expected_device
    )
    assert conn.engine_id == f"{host_ip}:16578"
    assert conn.request_address == f"{host_ip}:8000"
    assert conn.ib_devices == (expected_device.split(",") if expected_device else [])

    if role == "kv_consumer":
        # This test exercises the wire payload without registering GPU memory.
        conn._notification_port = 42000
        # Check the actual wire payload used for the RDMA target and ZMQ
        # write-done notification, not only the engine's bootstrap address.
        conn._send_on_socket = MagicMock()
        meta = ConnectorMetadata._build_req_meta(
            req_id="r0",
            local_block_ids=[0],
            kv_transfer_params={
                "remote_block_ids": [1],
                "remote_host": "10.19.0.113",
                "remote_handshake_port": 6301,
                "tp_size": 8,
                "transfer_id": 9,
            },
        )
        conn.start_load_kv(
            SimpleNamespace(
                request_id_to_transfer_id={"r0": 9}, reqs_to_recv={"r0": meta}
            )
        )
        addr, (_, payload) = conn._send_on_socket.call_args.args
        request = mc.msgpack.loads(payload)
        assert addr == "tcp://10.19.0.113:6301"
        assert request["consumer_host"] == host_ip
        assert request["consumer_rpc_port"] == 16578
        assert request["notify_host"] == host_ip
        assert request["notify_port"] == 42000


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


def test_mooncake_rdma_normalizes_explicit_device_list():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    assert (
        mc._select_ib_device("rdma", " ionic_4, ionic_0,ionic_4 ", None)
        == "ionic_4,ionic_0"
    )


def test_mooncake_rdma_auto_selects_from_physical_gpu(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"auto{idx}")
    assert mc._select_ib_device("rdma", "", 5) == "auto5"


def test_mooncake_rdma_registers_all_alternate_hcas_for_upper_rail_gpu(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"ionic_{idx}")
    monkeypatch.setattr(mc, "_ib_device_exists", lambda _device: True)

    assert mc._select_ib_devices(
        "rdma",
        "",
        4,
        enable_alternate_hca=True,
        hca_count=8,
    ) == [
        "ionic_4",
        *(f"ionic_{idx}" for idx in range(4)),
        *(f"ionic_{idx}" for idx in range(5, 8)),
    ]


def test_mooncake_rdma_registers_all_alternate_hcas_for_lower_rail_gpu(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"ionic_{idx}")
    monkeypatch.setattr(mc, "_ib_device_exists", lambda _device: True)

    assert mc._select_ib_devices(
        "rdma",
        "",
        3,
        enable_alternate_hca=True,
        hca_count=8,
    ) == [
        "ionic_3",
        "ionic_0",
        "ionic_1",
        "ionic_2",
        *(f"ionic_{idx}" for idx in range(4, 8)),
    ]


def test_mooncake_rdma_skips_missing_alternate_hcas(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"ionic_{idx}")
    monkeypatch.setattr(
        mc,
        "_ib_device_exists",
        lambda device: device in {"ionic_3", "ionic_0", "ionic_7"},
    )

    assert mc._select_ib_devices(
        "rdma",
        "",
        3,
        enable_alternate_hca=True,
        hca_count=8,
    ) == ["ionic_3", "ionic_0", "ionic_7"]


def test_mooncake_rdma_rejects_nonpositive_hca_count(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"ionic_{idx}")

    with pytest.raises(ValueError, match="ib_hca_count"):
        mc._select_ib_devices(
            "rdma",
            "",
            4,
            enable_alternate_hca=True,
            hca_count=0,
        )


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
    assert consumer_region_indices(156, 78, 0, 156, 1) == list(range(156))


def test_region_map_identity_when_empty():
    assert consumer_region_indices(0, 0, 5, 156, 4) == []


def test_region_map_group_major_single_group():
    # 1 region/layer, stage of 20 layers @ global start 18 → consumer 18..37.
    assert consumer_region_indices(20, 20, 18, 78, 4) == list(range(18, 38))


def test_region_map_group_major_two_groups_mla():
    # MLA: 2 groups [kv, index], stage=20 layers @ start 18, consumer has 156.
    got = consumer_region_indices(40, 20, 18, 156, 4)
    assert got[:20] == list(range(18, 38))
    assert got[20:] == list(range(78 + 18, 78 + 38))


def test_region_map_undefined_when_not_multiple():
    assert consumer_region_indices(41, 20, 18, 156, 4) is None


def test_region_map_undefined_when_groups_uneven():
    # 2 local groups against a consumer list that is not 2 whole groups.
    assert consumer_region_indices(40, 20, 18, 157, 4) is None


def test_region_map_stages_tile_consumer_no_overlap():
    # Uniform MLA layout: 78 layers, PP4 partition [18,20,20,20],
    # 2 complete groups (kv + per-layer index).
    partitions, num_hidden, groups = [18, 20, 20, 20], 78, 2
    covered = []
    for start, n_local in zip(_starts(partitions), partitions):
        covered.extend(
            consumer_region_indices(
                n_local * groups, n_local, start, num_hidden * groups, 4
            )
        )
    total = num_hidden * groups
    assert sorted(covered) == list(range(total))
    assert len(covered) == len(set(covered))  # no overlap


def test_region_map_stages_tile_consumer_with_mtp_layer():
    # Last PP stage binds the draft KV layer, making its group one entry
    # wider. Stride derived from consumer region count keeps all stages aligned.
    partitions, num_hidden, groups = [20, 20, 20, 18], 78, 2
    consumer_layers = num_hidden + 1  # + MTP layer 78
    covered = []
    for stage, (start, n_local) in enumerate(zip(_starts(partitions), partitions)):
        if stage == len(partitions) - 1:
            n_local += 1
        covered.extend(
            consumer_region_indices(
                n_local * groups, n_local, start, consumer_layers * groups, 4
            )
        )
    assert sorted(covered) == list(range(consumer_layers * groups))
    assert len(covered) == len(set(covered))


def test_region_map_undefined_when_producer_drafts_and_consumer_does_not():
    # Prefill with --method mtp against a consumer without it: the last stage's
    # 19th layer has nowhere to land. Must refuse rather than alias onto the
    # consumer's next group.
    assert consumer_region_indices(38, 19, 60, 78 * 2, 4) is None


def test_region_map_consumer_mtp_layer_left_unwritten_is_fine():
    # Benign: the producer simply never writes the consumer's MTP layer.
    got = consumer_region_indices(36, 18, 60, 79 * 2, 4)
    assert got[:18] == list(range(60, 78))
    assert got[18:] == list(range(79 + 60, 79 + 78))


def test_region_map_group_major_beats_naive_offset():
    # Regression guard: a naive additive offset (start_layer*groups + i) would
    # misroute group-major layouts. Stage1's index-group region 0 (local idx 20)
    # must land in the consumer's index group (>=78), not at 36+20=56 (kv group).
    cmap = consumer_region_indices(40, 20, 18, 156, 4)
    assert cmap[20] == 78 + 18
    assert cmap[20] != 56


def test_explicit_region_map_supports_compact_index_group():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = object.__new__(mc.MooncakeConnector)
    # PP stage 1 owns target layers [18, 38). In this synthetic IndexShare
    # schedule only global layers 18, 22, 26, 30, and 34 own index caches.
    explicit = list(range(18, 38)) + [83, 84, 85, 86, 87]

    assert (
        conn._consumer_region_map(
            len(explicit), len(explicit), explicit_indices=explicit
        )
        == explicit
    )


def test_compact_index_region_maps_tile_consumer_without_overlap():
    partitions = [18, 20, 20, 20]
    num_hidden = 78
    full_layer_ids = tuple(range(0, num_hidden, 4))
    full_layer_slots = {layer_id: slot for slot, layer_id in enumerate(full_layer_ids)}
    covered = []

    for start, n_local in zip(_starts(partitions), partitions):
        local_layers = tuple(range(start, start + n_local))
        local_full_layers = tuple(
            layer_id for layer_id in local_layers if layer_id in full_layer_slots
        )
        explicit = list(local_layers) + [
            num_hidden + full_layer_slots[layer_id] for layer_id in local_full_layers
        ]
        covered.extend(explicit)

    expected = list(range(num_hidden + len(full_layer_ids)))
    assert sorted(covered) == expected
    assert len(covered) == len(set(covered))


def test_explicit_region_map_rejects_length_mismatch():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = object.__new__(mc.MooncakeConnector)

    with pytest.raises(ValueError, match="length does not match"):
        conn._consumer_region_map(3, 2, explicit_indices=[0, 1])


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
    conn._pending_recv_failed = set()
    conn._pending_recv_nonce = {}
    conn._pending_recv_acks = {}
    conn._pending_recv_stall_watch = {}
    conn._held_failures = {}
    conn._pending_recv = set()
    conn._pending_recv_blocks = {}
    conn._pending_recv_slots = {}
    conn._blocks_pending_fence = []
    conn.done_recving = set()
    conn.failed_recving = set()
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
    conn.dcp_size = 1
    conn.transfer_engine = object()
    conn._transfer_refcount_lock = threading.Lock()
    conn._transfer_refcount = {}
    conn._completed_prefills_lock = threading.Lock()
    conn._completed_prefills = {}
    conn._kv_cache_ready_events = {}
    conn.done_sending = set()
    conn._wait_for_prefill_data = lambda _transfer_id: dict(prefill_data)
    conn._execute_block_slot_transfer = (
        lambda request, target, src, dst, data, req_id, *, engine=None: captured.append(
            (request, target, src, dst, data, req_id, engine)
        )
        or True
    )
    conn._send_write_done = lambda *_args, **_kwargs: None
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
    # The slot path takes the engine `_execute_transfer` picked, so #2276's
    # matched-rail routing reaches it too.
    assert captured[0][6] is conn.transfer_engine


@pytest.mark.parametrize("invalid_slot", [-1, None, "invalid"])
def test_stateful_transfer_without_final_source_slot_fails_closed(caplog, invalid_slot):
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


def test_failed_write_done_returns_the_staging_row_to_the_pool():
    """A rejected transfer must not leak the staging row it reserved.

    The consumer reserves a row before asking the producer to write. When the
    producer reports failure the bytes never landed, so the scatter is skipped
    on purpose -- but the row still has to go back, or _staging_slots.acquire()
    blocks forever once repeated failures drain the pool.
    """
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    scattered = []
    conn = _make_connector(
        _staging_slots=mc.SlotPool(),
        _scatter_slot=lambda *args: scattered.append(args),
    )
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv_slots["r1"] = (7, 3)

    assert conn._record_write_done("r1", 0, 0, 0, success=False)

    assert conn._staging_slots.free == [3], "staging row was not returned to the pool"
    assert not scattered, "a failed transfer must not scatter its staging row"
    assert "r1" not in conn._pending_recv_slots
    assert "r1" in conn.failed_recving


def test_failed_write_done_without_a_staging_row_is_a_noop():
    """Block-only transfers reserve no row; the failure path must not crash."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = _make_connector(_staging_slots=mc.SlotPool())
    conn._pending_recv_expected["r1"] = 1
    assert conn._record_write_done("r1", 0, 0, 0, success=False)
    assert conn._staging_slots.free == []
    assert "r1" in conn.failed_recving


def test_failed_write_done_waits_for_every_producer_rank():
    """A failure is published, and stage-0 released, only with the last rank.

    Producers are never cancelled: a stage still running may write the
    request's pages after the scheduler reused them for the recompute.
    """
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sent = []
    conn = _make_connector(
        _release_targets={"r1": ("stage0", "t1", 1)},
        _send_on_socket=lambda addr, parts, repeat=1: sent.append((addr, parts[0])),
    )
    conn._pending_recv_expected["r1"] = 2
    assert not conn._record_write_done("r1", 0, 0, 0, success=False)
    assert not conn._record_write_done("r1", 0, 0, 0, success=False)  # resend
    assert conn.failed_recving == set() and sent == []
    assert conn._record_write_done("r1", 1, 0, 0)
    assert conn.failed_recving == {"r1"} and conn.done_recving == set()
    assert sent == [("stage0", mc.MSG_RELEASE)]
    assert not conn._pending_recv_failed


def test_silent_producer_ranks_fail_the_receive_after_their_window():
    """An acknowledged rank silent past its write window has stopped writing.

    Without a bound, one failed stage plus one dead stage kept the request,
    its decode blocks and stage-0's page table forever.
    """
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sent = []
    conn = _make_connector(
        _release_targets={"old": ("stage0", "t1", 1)},
        _send_on_socket=lambda addr, parts, repeat=1: sent.append((addr, parts[0])),
        _mla_landing=None,
    )
    for req_id in ("old", "new"):
        conn._pending_recv_expected[req_id] = 2
        conn._pending_recv.add(req_id)
    assert not conn._record_write_done("old", 1, 0, 0, success=False)
    ack = {"request_id": "old", "pp_rank": 0, "tp_rank": 0, "write_window_s": 60}
    conn._record_write_ack(ack)
    conn._record_write_ack({**ack, "request_id": "new"})
    conn._record_write_ack({**ack, "request_id": "new", "pp_rank": 1})
    conn._expire_pending_recvs()
    assert conn.failed_recving == set()  # the windows are still open
    conn._pending_recv_acks["old"][(0, 0)] = time.monotonic() - 1
    conn._expire_pending_recvs()
    assert conn.failed_recving == {"old"} and conn.done_recving == set()
    assert sent == [("stage0", mc.MSG_RELEASE)]
    assert list(conn._pending_recv_acks) == ["new"]
    assert conn._pending_recv == {"new"}
    assert not conn._pending_recv_failed and "old" not in conn._pending_recv_stages
    # The dead stage's write-done, should it ever come, changes nothing.
    assert not conn._record_write_done("old", 0, 0, 0)
    assert conn.done_recving == set()


def test_a_rank_that_never_acknowledged_keeps_the_receive_open():
    """It may still get the request and write, however long that takes."""
    pytest.importorskip("atom.kv_transfer.disaggregation.mooncake.mooncake_connector")
    conn = _make_connector(_mla_landing=None)
    conn._pending_recv_expected["r1"] = 2
    conn._pending_recv.add("r1")
    assert not conn._record_write_done("r1", 1, 0, 0, success=False)
    conn._record_write_ack(
        {"request_id": "r1", "pp_rank": 1, "tp_rank": 0, "write_window_s": 1}
    )
    conn._pending_recv_acks["r1"][(1, 0)] = time.monotonic() - 1
    conn._expire_pending_recvs()
    assert conn.failed_recving == set()


@pytest.mark.parametrize("window", [None, -1, 0, float("inf"), True, "60"])
def test_an_ack_without_a_usable_window_starts_no_deadline(window):
    pytest.importorskip("atom.kv_transfer.disaggregation.mooncake.mooncake_connector")
    conn = _make_connector(_mla_landing=None)
    conn._pending_recv_expected["r1"] = 1
    conn._record_write_ack(
        {"request_id": "r1", "pp_rank": 0, "tp_rank": 0, "write_window_s": window}
    )
    assert conn._pending_recv_acks == {}


@pytest.mark.parametrize(
    "raw, seconds",
    [(None, 30.0), ("60", 60.0), ("2", 5.0), ("abc", 5.0), (" 45s", 45.0)],
)
def test_the_write_window_follows_mooncakes_transfer_timeout(monkeypatch, raw, seconds):
    """Read the way Mooncake reads MC_TRANSFER_TIMEOUT (atoi, 5 s floor)."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    if raw is None:
        monkeypatch.delenv("MC_TRANSFER_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("MC_TRANSFER_TIMEOUT", raw)
    assert mc._mc_transfer_timeout_s() == seconds
    assert mc._write_window_s() == (
        mc.PD_TRANSFER_DEADLINE_S + seconds + mc.PD_WRITE_WINDOW_SLACK_S
    )


def test_no_rdma_attempt_starts_past_the_transfer_deadline():
    """A stage stops writing at its deadline, inside the window it acknowledged."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = object.__new__(mc.MooncakeConnector)
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 0
    args = ("consumer:1", [1], [2], [3], "r", "block")
    assert conn._rdma_write(*args, engine=engine, deadline=time.monotonic() + 60)
    assert conn._rdma_write(*args, engine=engine, deadline=time.monotonic() - 1) is None
    assert engine.batch_transfer_sync_write.call_count == 1


@pytest.mark.parametrize("timed_out", [True, False])
def test_a_failed_rdma_write_is_not_retried(monkeypatch, caplog, timed_out):
    """Mooncake returns from a timed-out sync write without cancelling it."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "_mc_transfer_timeout_s", lambda: 0.0 if timed_out else 1e9)
    conn = object.__new__(mc.MooncakeConnector)
    engine = MagicMock()
    engine.batch_transfer_sync_write.side_effect = [-1, 0]
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert (
            conn._rdma_write("consumer:1", [1], [2], [3], "r", "block", engine=engine)
            is False
        )
    assert engine.batch_transfer_sync_write.call_count == 1
    (failed,) = [r for r in caplog.records if "[PRODUCER]" in r.getMessage()]
    assert ("may still land" in failed.getMessage()) == timed_out


def test_a_failed_receive_is_reported_only_after_its_hold(monkeypatch):
    """Its pages and staging row stay held while a failed write may land."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "60")
    conn = _make_connector(_staging_slots=mc.SlotPool(), _mla_landing=None)
    conn._pending_recv_expected["r1"] = 1
    conn._pending_recv.add("r1")
    conn._pending_recv_slots["r1"] = (7, 3)

    assert conn._record_write_done("r1", 0, 0, 0, success=False)
    conn._report_held_failures()
    assert conn.failed_recving == set() and conn._staging_slots.free == []
    assert "r1" in conn._pending_recv

    now = time.monotonic()
    monkeypatch.setattr(mc.time, "monotonic", lambda: now + 61)
    conn._report_held_failures()
    assert conn.failed_recving == {"r1"} and conn._staging_slots.free == [3]
    assert conn._held_failures == {} and "r1" not in conn._pending_recv


@pytest.mark.parametrize("acked", [True, False])
def test_a_stalled_receive_is_logged_once(monkeypatch, caplog, acked):
    """As an error when a rank never acknowledged it: it waits without a bound."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = _make_connector(_mla_landing=None)
    conn._pending_recv_expected["r1"] = 2
    conn._pending_recv_stall_watch["r1"] = time.monotonic()
    assert not conn._record_write_done("r1", 0, 0, 0, success=False)
    if acked:
        conn._pending_recv_acks["r1"] = {(1, 0): time.monotonic() + 600}
    conn._expire_pending_recvs()  # too early to log

    now = time.monotonic()
    monkeypatch.setattr(mc.time, "monotonic", lambda: now + 31)
    with caplog.at_level(logging.WARNING, logger="atom"):
        conn._expire_pending_recvs()
        conn._expire_pending_recvs()
    (record,) = [r for r in caplog.records if "has waited" in r.getMessage()]
    assert "1 of 2 reported" in record.getMessage()
    assert ("never acknowledged" in record.getMessage()) == (not acked)
    assert record.levelno == (logging.WARNING if acked else logging.ERROR)
    assert conn.failed_recving == set()


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


# ---------------------------------------------------------------------------
# PP head: request-less batches still carry KV connector metadata
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, requests):
        self.requests = list(requests)


def _fake_batch(req_ids, meta):
    return SimpleNamespace(req_ids=list(req_ids), connector_meta_output=meta)


def _pp_engine_core_cls():
    # The aiter stubs above shadow the real package; fill in the submodules the
    # engine-core import chain reaches for.
    for name, attrs in (
        ("aiter.dist.shm_broadcast", ("MessageQueue",)),
        ("aiter.ops.communication", ("set_custom_all_reduce",)),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for attr in attrs:
                setattr(mod, attr, MagicMock())
            sys.modules[name] = mod
    from atom.model_engine.pp_engine_core import PPEngineCoreProc

    return PPEngineCoreProc


def _fake_head(batch):
    PPEngineCoreProc = _pp_engine_core_cls()

    head = SimpleNamespace(
        _in_flight=deque(),
        _defer_prefix_hash=False,
        pp_size=4,
        kv_transfer_enabled=True,
        output_queue=MagicMock(),
        runner_mgr=MagicMock(),
        pp_transport=MagicMock(),
        scheduler=MagicMock(),
        _poll_kv_transfer_progress=MagicMock(),
        _dispatch_idle_offload_work=MagicMock(),
        # Real throttle, mocked dispatch: `_pp_head_step` runs on a loop that
        # never sleeps, so it must go through `_advance_idle_kv_transfer`.
        _next_idle_kv_drain=0.0,
    )
    head.scheduler.schedule.side_effect = [(batch, {}), None]
    head.scheduler.take_rejected.return_value = None
    head._advance_idle_kv_transfer = PPEngineCoreProc._advance_idle_kv_transfer.__get__(
        head
    )
    head._dispatch_connector_only_batch = (
        PPEngineCoreProc._dispatch_connector_only_batch.__get__(head)
    )
    PPEngineCoreProc._pp_head_step(head)
    return head


def _dispatched_metas(head):
    return [
        c.args[1]
        for c in head.runner_mgr.call_func.call_args_list
        if c.args and c.args[0] == "process_kvconnector_output"
    ]


def test_pp_head_dispatches_meta_of_request_less_batch():
    """All sequences parked on offload loads -> no batch, but the metadata that
    starts those loads must still reach the workers, or the head deadlocks."""
    meta = _FakeMeta(["load-r1"])
    head = _fake_head(_fake_batch([], meta))

    assert _dispatched_metas(head) == [meta]
    head.pp_transport.send_metadata.assert_called_once()
    assert head.pp_transport.send_metadata.call_args.args[0].req_ids == []
    head.runner_mgr.call_func.assert_any_call("flush_pp_send", wait_out=True)


def test_pp_head_skips_empty_meta_of_request_less_batch():
    """An idle head must not broadcast metadata that carries no work."""
    for meta in (None, _FakeMeta([])):
        head = _fake_head(_fake_batch([], meta))
        assert _dispatched_metas(head) == []
        head.pp_transport.send_metadata.assert_not_called()


def test_pp_head_dispatches_idle_offload_without_a_batch():
    """`Scheduler.is_finished()` stays false while blocks are deferred, so the
    loop re-enters `_pp_head_step` and never reaches its own idle-drain branch.
    The leftover suffix saves must be dispatched here or never at all."""
    head = _fake_head(None)
    head._dispatch_idle_offload_work.assert_called_once()


def test_pp_head_idle_dispatch_shares_the_drain_throttle(monkeypatch):
    """The loop has no sleep; a direct dispatch would rebuild connector
    metadata on every turn."""
    PPEngineCoreProc = _pp_engine_core_cls()
    from atom.model_engine import engine_core

    # CI can spend longer than the 1 ms drain interval between calls. Control
    # the clock so this tests the throttle rather than the runner's speed.
    now = 100.0
    monkeypatch.setattr(engine_core, "time", SimpleNamespace(monotonic=lambda: now))
    head = _fake_head(None)
    head._dispatch_idle_offload_work.assert_called_once()
    assert head._next_idle_kv_drain == now + engine_core.KV_IDLE_DRAIN_INTERVAL_S

    now += engine_core.KV_IDLE_DRAIN_INTERVAL_S / 2
    head.scheduler.schedule.side_effect = [None]
    PPEngineCoreProc._pp_head_step(head)

    head._dispatch_idle_offload_work.assert_called_once()  # still once: throttled

    now = head._next_idle_kv_drain
    head.scheduler.schedule.side_effect = [None]
    PPEngineCoreProc._pp_head_step(head)

    assert head._dispatch_idle_offload_work.call_count == 2
    assert head._next_idle_kv_drain == now + engine_core.KV_IDLE_DRAIN_INTERVAL_S


def test_pp_head_forwards_normal_batch_with_meta():
    """A batch with requests keeps the original path: dispatch, send, forward."""
    meta = _FakeMeta(["load-r1"])
    batch = _fake_batch([7], meta)
    batch.produces_output = lambda: False
    head = _fake_head(batch)

    assert _dispatched_metas(head) == [meta]
    head.pp_transport.send_metadata.assert_called_once_with(batch)
    head.runner_mgr.call_func.assert_any_call("forward", batch, wait_out=True)
    head._dispatch_idle_offload_work.assert_called_once()  # Next schedule is None.


def test_pp_downstream_skips_forward_for_request_less_batch():
    """Downstream must apply the metadata but run no forward for it."""
    PPEngineCoreProc = _pp_engine_core_cls()

    meta = _FakeMeta(["load-r1"])
    stage = SimpleNamespace(
        kv_transfer_enabled=True,
        is_last=True,
        runner_mgr=MagicMock(),
        pp_transport=MagicMock(),
        scheduler=MagicMock(),
        utility_handler=MagicMock(),
        utility_queue=MagicMock(),
        _is_idle_rl_weights_offloaded=lambda: False,
        _poll_and_send_kv_status=MagicMock(),
    )
    # One pass over the loop body, then shut down.
    stage.pull_and_process_input_queue = MagicMock(side_effect=[False, True])
    stage.pp_transport.recv_metadata.return_value = _fake_batch([], meta)

    PPEngineCoreProc._downstream_busy_loop(stage)

    stage.runner_mgr.call_func.assert_any_call("process_kvconnector_output", meta)
    forwards = [
        c for c in stage.runner_mgr.call_func.call_args_list if c.args[0] == "forward"
    ]
    assert forwards == []
    stage.pp_transport.send_tokens.assert_not_called()


def test_dcp_block_descriptors_are_streamed_in_bounded_batches():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE

    connector = object.__new__(mc.MooncakeConnector)
    connector.dcp_size = 1
    connector.block_size = 64
    connector.kv_caches_base_addr = [1_000_000, 2_000_000]
    bytes_per_block = connector.block_size * 576
    connector._per_block_bytes_list = [bytes_per_block, bytes_per_block]
    connector._block_region_roles = [MLA_KV_ROLE, MLA_KV_ROLE]
    connector._block_region_consumer_indices = None
    connector._consumer_region_map = lambda *_args, **_kwargs: [0, 1]

    batch_sizes = []
    transferred_bytes = 0

    def record_batch(
        _target,
        src_addrs,
        dst_addrs,
        sizes,
        _req_id,
        _label,
        *,
        engine=None,
        deadline=None,
    ):
        assert engine is None
        nonlocal transferred_bytes
        assert len(src_addrs) == len(dst_addrs) == len(sizes)
        assert len(src_addrs) <= connector._MAX_RDMA_ENTRIES_PER_BATCH
        batch_sizes.append(len(src_addrs))
        transferred_bytes += sum(sizes)
        return True

    connector._rdma_write = record_batch
    dst_block_ids = list(range(100, 140))
    src_block_ids = list(range(len(dst_block_ids) * 8))
    request_data = {
        "consumer_base_addrs": [3_000_000, 4_000_000],
        "consumer_num_layers": 2,
        "consumer_region_roles": [MLA_KV_ROLE, MLA_KV_ROLE],
        "consumer_block_bpb": [bytes_per_block, bytes_per_block],
        "consumer_dcp_size": 8,
        "consumer_dcp_rank": 0,
        "consumer_dcp_interleave": 1,
    }

    assert mc.MooncakeConnector._execute_block_transfer(
        connector,
        request_data,
        "consumer:1234",
        src_block_ids,
        dst_block_ids,
        "req-1",
    )
    descriptors_per_region = len(dst_block_ids) * connector.block_size
    assert batch_sizes == [4096, 1024]
    assert sum(batch_sizes) == 2 * descriptors_per_region
    assert transferred_bytes == 2 * descriptors_per_region * 576


def test_dcp_index_staging_waits_for_request_ready_event():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    connector = object.__new__(mc.MooncakeConnector)
    connector.dcp_size = 1
    connector.block_size = 16
    connector.kv_caches_base_addr = [1_000_000, 2_000_000]
    mla_block_bytes = connector.block_size * 576
    index_block_bytes = connector.block_size * 144
    connector._per_block_bytes_list = [mla_block_bytes, index_block_bytes]
    connector._block_region_roles = [MLA_KV_ROLE, INDEX_CACHE_ROLE]
    connector._block_region_consumer_indices = None
    connector._consumer_region_map = lambda *_args, **_kwargs: [0, 1]
    connector._index_staging_chunk_pages = 256
    gather_indices = object()
    connector._prepare_sharded_index = MagicMock(return_value=gather_indices)
    connector._gather_sharded_index = MagicMock()
    staging_stream = MagicMock()
    connector._send_worker_stream = MagicMock(return_value=staging_stream)
    connector._execute_staged_index_layer_chunk = MagicMock(return_value=True)
    connector._rdma_write = MagicMock(return_value=True)
    ready_event = object()
    request_data = {
        "consumer_base_addrs": [3_000_000, 4_000_000],
        "consumer_num_layers": 2,
        "consumer_region_roles": [MLA_KV_ROLE, INDEX_CACHE_ROLE],
        "consumer_block_bpb": [mla_block_bytes, index_block_bytes],
        "consumer_dcp_size": 2,
        "consumer_dcp_rank": 0,
        "consumer_dcp_interleave": 1,
    }

    assert mc.MooncakeConnector._execute_block_transfer(
        connector,
        request_data,
        "consumer:1234",
        [0, 1],
        [10],
        "req-1",
        ready_event,
        engine=ready_event,
    )
    staging_stream.wait_event.assert_called_once_with(ready_event)
    connector._execute_staged_index_layer_chunk.assert_called_once_with(
        "consumer:1234",
        1,
        4_000_000,
        index_block_bytes,
        [10],
        "req-1",
        gather_indices,
        engine=ready_event,
        deadline=None,
    )
    assert all(
        call.kwargs["engine"] is ready_event
        for call in connector._rdma_write.call_args_list
    )


def test_dcp_index_staging_rejects_missing_request_ready_event():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    connector = object.__new__(mc.MooncakeConnector)
    connector.dcp_size = 1
    connector.block_size = 16
    connector.kv_caches_base_addr = [1_000_000, 2_000_000]
    connector._per_block_bytes_list = [16 * 576, 16 * 144]
    connector._block_region_roles = [MLA_KV_ROLE, INDEX_CACHE_ROLE]
    connector._block_region_consumer_indices = None
    connector._consumer_region_map = lambda *_args, **_kwargs: [0, 1]
    connector._index_staging_chunk_pages = 256
    connector._prepare_sharded_index = MagicMock(return_value=object())
    connector._gather_sharded_index = MagicMock()
    connector._send_worker_stream = MagicMock()
    connector._rdma_write = MagicMock(return_value=True)
    request_data = {
        "consumer_base_addrs": [3_000_000, 4_000_000],
        "consumer_num_layers": 2,
        "consumer_region_roles": [MLA_KV_ROLE, INDEX_CACHE_ROLE],
        "consumer_block_bpb": [16 * 576, 16 * 144],
        "consumer_dcp_size": 2,
        "consumer_dcp_rank": 0,
        "consumer_dcp_interleave": 1,
    }

    with pytest.raises(RuntimeError, match="ready event"):
        mc.MooncakeConnector._execute_block_transfer(
            connector,
            request_data,
            "consumer:1234",
            [0, 1],
            [10],
            "req-1",
        )


def test_mooncake_records_one_ready_event_for_a_prefill_batch(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector = object.__new__(mc.MooncakeConnector)
    connector.is_producer = True
    connector._index_staging_pool_size = 1
    connector._cuda_device = 3
    connector._kv_cache_ready_events = {}
    connector._kv_cache_ready_batches = deque()
    connector._kv_ready_done_event = None
    connector._kv_ready_expired = {}
    connector._completed_prefills_lock = threading.Lock()
    ready_event = MagicMock()
    producer_stream = object()
    monkeypatch.setattr(mc.torch.cuda, "Event", MagicMock(return_value=ready_event))
    monkeypatch.setattr(
        mc.torch.cuda,
        "current_stream",
        MagicMock(return_value=producer_stream),
    )

    connector.record_kv_cache_ready([11, 12])

    ready_event.record.assert_called_once_with(producer_stream)
    assert connector._kv_cache_ready_events == {11: ready_event, 12: ready_event}


def _ready_event_producer(mc, monkeypatch):
    connector = object.__new__(mc.MooncakeConnector)
    connector.is_producer = True
    connector._index_staging_pool_size = 1
    connector._cuda_device = 0
    connector._kv_cache_ready_events = {}
    connector._kv_cache_ready_batches = deque()
    connector._kv_ready_done_event = None
    connector._kv_ready_expired = {}
    connector._completed_prefills_lock = threading.Lock()
    monkeypatch.setattr(mc.torch.cuda, "Event", MagicMock)
    monkeypatch.setattr(mc.torch.cuda, "current_stream", lambda _d: None)
    clock = [1000.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: clock[0])
    return connector, clock


def test_ready_events_no_transfer_takes_are_freed_after_their_ttl(monkeypatch):
    """Without the TTL, every final prefill with no finished transfer leaked
    its event: requests without P/D, aborts, unpulled requests, failures.

    A transfer that still comes later finds a long-complete stand-in, so a
    DSA index transfer goes ahead instead of failing on a missing event.
    """
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector, clock = _ready_event_producer(mc, monkeypatch)
    connector.record_kv_cache_ready(["a", "b"])
    first = connector._kv_cache_ready_events["a"]
    clock[0] += 10
    connector.record_kv_cache_ready(["b"])  # recorded again, by a later batch
    clock[0] += mc.KV_CACHE_READY_EVENT_TTL_S - 5
    connector.record_kv_cache_ready(["c"])
    # The first batch aged out: "a" holds the shared stand-in (that batch's
    # own event, complete long ago); "b" keeps its later event.
    done = connector._kv_ready_done_event
    assert done is first
    assert connector._get_kv_cache_ready_event("a") is done
    assert connector._kv_cache_ready_events["b"] is not first
    assert list(connector._kv_ready_expired) == ["a"]
    assert len(connector._kv_cache_ready_batches) == 2


def test_ready_events_age_out_while_prefill_is_idle(monkeypatch):
    """get_finished runs every step, so no new prefill batch is needed."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector, clock = _ready_event_producer(mc, monkeypatch)
    for name, value in vars(_make_connector()).items():
        setattr(connector, name, value)
    connector.done_sending = set()
    connector.record_kv_cache_ready(["a"])
    clock[0] += mc.KV_CACHE_READY_EVENT_TTL_S + 1
    connector.get_finished()
    assert not connector._kv_cache_ready_batches
    assert list(connector._kv_ready_expired) == ["a"]


def test_requests_whose_event_aged_out_are_remembered_up_to_a_cap(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc, "KV_CACHE_READY_EXPIRED_MAX", 2)
    connector, clock = _ready_event_producer(mc, monkeypatch)
    connector.record_kv_cache_ready(["a", "b", "c"])
    clock[0] += mc.KV_CACHE_READY_EVENT_TTL_S + 1
    connector.record_kv_cache_ready(["d"])
    assert list(connector._kv_ready_expired) == ["b", "c"]
    assert connector._get_kv_cache_ready_event("a") is None
    assert connector._get_kv_cache_ready_event("c") is connector._kv_ready_done_event


# ---------------------------------------------------------------------------
# Matched-rail engines for independent P/D GPU ranks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol,devices,rails,message",
    [
        ("tcp", [], ["ionic_0"], "protocol=rdma"),
        ("rdma", ["ionic_0", "ionic_1"], ["ionic_0", "ionic_1"], "single primary"),
        ("rdma", ["ionic_0"], ["ionic_1"], "including the primary"),
        ("rdma", ["ionic_0"], ["ionic_0", "missing"], "existing local HCAs"),
    ],
)
def test_matched_rails_reject_invalid_transport_configuration(
    monkeypatch, protocol, devices, rails, message
):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    monkeypatch.setattr(mc, "_ib_device_exists", lambda name: name.startswith("ionic_"))
    with pytest.raises(ValueError, match=message):
        mc._validate_matched_rails(protocol, devices, rails)


def test_matched_rails_disabled_preserves_tcp_and_multi_hca(monkeypatch):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    mc._validate_matched_rails("tcp", [], [])
    mc._validate_matched_rails("rdma", ["ionic_0", "ionic_1"], [])
    monkeypatch.setattr(mc, "_ib_device_exists", lambda _: True)
    mc._validate_matched_rails("rdma", ["ionic_2"], ["ionic_2", "ionic_6"])


@pytest.fixture
def matched_rail_sysfs(tmp_path, monkeypatch):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    monkeypatch.setattr(mc, "_IB_SYSFS_ROOT", tmp_path)

    def add(name, *states):
        device = tmp_path / name
        device.mkdir(exist_ok=True)
        for index, state in enumerate(states, 1):
            port = device / "ports" / str(index)
            port.mkdir(parents=True)
            if state is not None:
                (port / "state").write_text(state)
        return device

    return mc, add


@pytest.mark.parametrize("prefix", ["ionic_", "rdma", "mlx5_"])
@pytest.mark.parametrize("value", ["auto", " AUTO "])
def test_matched_rails_auto_discovers_only_active_primary_family(
    matched_rail_sysfs, prefix, value
):
    mc, add = matched_rail_sysfs
    for suffix in (10, 2, 0):
        add(f"{prefix}{suffix}", "4: ACTIVE\n")
    add(f"{prefix}3", "1: DOWN\n")
    add(f"{prefix}4", None)  # A missing state file is not an active port.
    add(f"{prefix}5")  # No ports exposed.
    add(f"{prefix}6", "2: INIT\n", "4: ACTIVE\n")
    add(f"{prefix}7", "4: ACTIVE\n", "4: ACTIVE\n")  # Include a device once.
    add(f"{prefix}8_extra", "4: ACTIVE\n")
    add("other_0", "4: ACTIVE\n")

    assert mc._resolve_matched_rails("rdma", [f"{prefix}2"], value) == [
        f"{prefix}{i}" for i in (0, 2, 6, 7, 10)
    ]


def test_matched_rails_auto_excludes_unrelated_active_nic(matched_rail_sysfs):
    mc, add = matched_rail_sysfs
    for i in range(8):
        add(f"ionic_{i}", "4: ACTIVE\n")
    add("mlx5_0", "4: ACTIVE\n")
    assert mc._resolve_matched_rails("rdma", ["ionic_2"], "auto") == [
        f"ionic_{i}" for i in range(8)
    ]


@pytest.mark.parametrize("state", [None, "1: DOWN\n", "invalid"])
def test_matched_rails_auto_rejects_inactive_primary(matched_rail_sysfs, state):
    mc, add = matched_rail_sysfs
    add("ionic_2", state)
    add("ionic_6", "4: ACTIVE\n")
    with pytest.raises(ValueError, match="Primary HCA.*no readable ACTIVE"):
        mc._resolve_matched_rails("rdma", ["ionic_2"], "auto")


def test_matched_rails_auto_rejects_missing_primary(matched_rail_sysfs):
    mc, add = matched_rail_sysfs
    add("ionic_6", "4: ACTIVE\n")
    with pytest.raises(ValueError, match="Primary HCA.*no readable ACTIVE"):
        mc._resolve_matched_rails("rdma", ["ionic_2"], "auto")


def test_matched_rails_auto_reports_hidden_sysfs(matched_rail_sysfs, tmp_path):
    mc, _ = matched_rail_sysfs
    mc._IB_SYSFS_ROOT = tmp_path / "not-mounted"
    with pytest.raises(ValueError, match="Cannot discover RDMA HCAs"):
        mc._resolve_matched_rails("rdma", ["ionic_2"], "auto")


def test_matched_rails_auto_requires_numbered_names(matched_rail_sysfs):
    mc, add = matched_rail_sysfs
    add("custom_hca", "4: ACTIVE\n")
    with pytest.raises(ValueError, match="explicit HCA list"):
        mc._resolve_matched_rails("rdma", ["custom_hca"], "auto")
    assert mc._resolve_matched_rails("rdma", ["custom_hca"], "custom_hca") == [
        "custom_hca"
    ]


@pytest.mark.parametrize(
    "protocol,devices,message",
    [
        ("tcp", [], "protocol=rdma"),
        ("rdma", [], "single primary"),
        ("rdma", ["ionic_0", "ionic_1"], "single primary"),
    ],
)
def test_matched_rails_auto_validates_before_discovery(
    monkeypatch, protocol, devices, message
):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    discovery = MagicMock(side_effect=AssertionError("must not discover"))
    monkeypatch.setattr(mc, "_discover_active_matched_rails", discovery)
    with pytest.raises(ValueError, match=message):
        mc._resolve_matched_rails(protocol, devices, "auto")
    discovery.assert_not_called()


def test_matched_rails_explicit_list_and_unset_do_not_discover(matched_rail_sysfs):
    mc, add = matched_rail_sysfs
    add("ionic_2", "4: ACTIVE\n")
    add("ionic_6", "1: DOWN\n")
    # Preserve explicit-list behavior; only auto filters link state.
    assert mc._resolve_matched_rails(
        "rdma", ["ionic_2"], " ionic_6, ionic_2,ionic_6, "
    ) == ["ionic_6", "ionic_2"]
    mc._IB_SYSFS_ROOT = mc._IB_SYSFS_ROOT / "not-mounted"
    assert mc._resolve_matched_rails("tcp", [], " ") == []
    assert mc._resolve_matched_rails("rdma", ["ionic_2", "ionic_6"], "") == []


def _matched_rail_producer():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    conn = object.__new__(mc.MooncakeConnector)
    conn.dp_rank = 2
    conn.pp_rank = 0
    conn.pp_size = conn.tp_size = 1
    conn._completed_prefills = {}
    conn._kv_cache_ready_events = {}
    conn._completed_prefills_lock = threading.Lock()
    conn._transfer_refcount_lock = threading.Lock()
    conn._completion_lock = threading.Lock()
    conn._transfer_refcount = {}
    conn.done_sending = set()
    conn._wait_for_prefill_data = lambda _: {"block_ids": [1], "slot_index": -1}
    conn._get_kv_cache_ready_event = lambda _: None
    conn._notify_transfer_result = MagicMock()
    conn.transfer_engine = MagicMock()
    conn.transfer_engine.batch_transfer_sync_write.return_value = 0
    conn.transfer_engine.get_first_buffer_address.return_value = 1
    conn._rail_pool = None
    return conn


def _matched_rail_request(name, device):
    return {
        "request_id": name,
        "transfer_id": name,
        "consumer_host": device,
        "consumer_rpc_port": 1234,
        "consumer_ib_device": device,
        "consumer_dp_rank": 6,
        "dst_block_ids": [2],
    }


@pytest.mark.parametrize("has_slot_data", [False, True])
@pytest.mark.parametrize("matched", [False, True])
def test_concurrent_pd_requests_keep_their_selected_engine(has_slot_data, matched):
    from concurrent.futures import ThreadPoolExecutor

    conn = _matched_rail_producer()
    extra = MagicMock()
    extra.batch_transfer_sync_write.return_value = 0
    extra.get_first_buffer_address.return_value = 1
    engines = {"ionic_2": conn.transfer_engine, "ionic_6": extra}
    if matched:
        conn._rail_pool = SimpleNamespace(get=engines.__getitem__)
    barrier = threading.Barrier(2)

    def transfer(data, target, *_args, engine=None):
        # Both selections must finish before either write. Mutating the
        # connector's shared engine here would send a request on the wrong rail.
        barrier.wait(timeout=10)
        return conn._rdma_write(
            target, [100], [200], [64], data["request_id"], "test", engine=engine
        )

    conn._execute_block_transfer = transfer
    conn._execute_block_slot_transfer = transfer
    requests = [_matched_rail_request(f"req-{device}", device) for device in engines]
    for request in requests:
        request["has_slot_regions"] = has_slot_data
        if has_slot_data:
            # A stateful transfer must carry the producer's final source slot
            # (#2154); _execute_transfer rejects the request before it ever
            # reaches the rail selection this test exercises.
            request["src_slot_index"] = 5
            request["dst_slot_index"] = 6
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(conn._execute_transfer, requests))

    assert conn.done_sending == {request["request_id"] for request in requests}
    assert all(
        call.kwargs["success"] for call in conn._notify_transfer_result.call_args_list
    )
    assert conn.transfer_engine is engines["ionic_2"]
    for device, engine in engines.items():
        calls = engine.batch_transfer_sync_write.call_args_list
        expected = (
            [f"{device}:1234"]
            if matched
            else (["ionic_2:1234", "ionic_6:1234"] if device == "ionic_2" else [])
        )
        assert sorted(call.args[0] for call in calls) == expected


def test_missing_consumer_rail_notifies_failure_without_writing():
    from atom.kv_transfer.disaggregation.mooncake.rail_engine_pool import RailEnginePool

    conn = _matched_rail_producer()
    factory = MagicMock()
    conn._rail_pool = RailEnginePool(
        factory,
        conn.transfer_engine,
        "ionic_2",
        ["ionic_2"],
        lambda device: "127.0.0.1",
    )
    conn._rail_pool.set_regions([100], [64])
    conn._execute_block_transfer = MagicMock()
    request = _matched_rail_request("old-consumer", None)
    request.pop("consumer_ib_device")
    conn._execute_transfer(request)
    conn._notify_transfer_result.assert_called_once_with(request, success=False)
    conn._execute_block_transfer.assert_not_called()
    conn.transfer_engine.batch_transfer_sync_write.assert_not_called()
    factory.assert_not_called()
    assert not conn.done_sending


@pytest.mark.parametrize("succeeds", [False, True])
def test_rdma_chunks_use_selected_engine(succeeds):
    conn = _matched_rail_producer()
    conn._MAX_RDMA_ENTRIES_PER_BATCH = 2
    selected = MagicMock()
    selected.batch_transfer_sync_write.side_effect = [0, 0] if succeeds else [-1]
    assert (
        conn._rdma_write(
            "consumer:1234",
            [1, 2, 3],
            [4, 5, 6],
            [64, 64, 64],
            "request",
            "block",
            engine=selected,
        )
        is succeeds
    )
    calls = selected.batch_transfer_sync_write.call_args_list
    assert calls[0].args == ("consumer:1234", [1, 2], [4, 5], [64, 64])
    # A failed chunk is final: its writes may still land, a retry would hide them.
    assert [c.args for c in calls[1:]] == (
        [("consumer:1234", [3], [6], [64])] if succeeds else []
    )
    conn.transfer_engine.batch_transfer_sync_write.assert_not_called()


def test_staged_index_write_preserves_selected_engine(monkeypatch):
    from contextlib import nullcontext

    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    conn = _matched_rail_producer()
    conn._index_staging_slots = MagicMock()
    conn._index_staging_slots.acquire.return_value = 3
    staging_stream = SimpleNamespace(synchronize=MagicMock())
    conn._send_worker_stream = lambda: staging_stream
    conn._gather_sharded_index = lambda *_args: (10000, 2)
    conn._rdma_write = MagicMock(return_value=True)
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _: nullcontext())
    selected = object()
    assert conn._execute_staged_index_layer_chunk(
        "consumer:1234", 0, 20000, 64, [4, 5], "request", object(), engine=selected
    )
    conn._rdma_write.assert_called_once_with(
        "consumer:1234",
        [10000],
        [20256],
        [128],
        "request",
        "staged-index",
        engine=selected,
        deadline=None,
    )
    staging_stream.synchronize.assert_called_once()
    conn._index_staging_slots.release.assert_called_once_with(3)
