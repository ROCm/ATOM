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

from atom.utils import mooncake_hca

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
        num_prompt_tokens=130,
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


@pytest.mark.parametrize("dcp_size", [1, 2])
@pytest.mark.parametrize(
    "extra_tokens, prompt_blocks",
    [(0, [1, 2]), (1, [1, 2, 3])],
    ids=["fills-last-block", "one-token-into-a-block"],
)
def test_producer_publishes_only_the_prompt_blocks(
    dcp_size, extra_tokens, prompt_blocks
):
    """With deferred output a finished prefill is scheduled for one more step,
    and a prompt that fills its last block gets a block for the first token
    there; MTP lookahead adds blocks too. The consumer allocates the prompt's
    blocks only."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = object.__new__(mc.MooncakeConnectorScheduler)
    sched.pp_size = 1
    sched.tp_size = 1
    sched.hash_block_size = 16
    sched.block_size = 16
    sched.dcp_size = dcp_size
    sched.dp_rank = 0
    sched.engine_id = "eng"
    sched.host_ip = "10.0.0.1"
    sched.handshake_port = 40000
    sched.base_handshake_port = 6301
    sched.is_producer = True
    sched._awaiting_send = set()
    sched._reqs_need_save = {}

    seq = SimpleNamespace(
        kv_transfer_params={"do_remote_decode": True},
        output_tokens=[7],
        spec_token_ids=None,
        num_prompt_tokens=2 * 16 * dcp_size + extra_tokens,
        block_table=[1, 2, 3, 4],
        id=99,
        state_slot=-1,
        state_slots=[],
        kv_transfer_params_output=None,
    )
    mc.MooncakeConnectorScheduler.request_finished(sched, seq)
    assert seq.kv_transfer_params_output["remote_block_ids"] == prompt_blocks
    assert sched._reqs_need_save[seq.id][1] == prompt_blocks


@pytest.mark.parametrize("is_producer", [True, False], ids=["producer", "consumer"])
def test_finished_block_ids_are_lists_of_their_own(is_producer):
    """`seq.block_table` is a live `BlockTable`, an `array`. What
    `request_finished` publishes or queues must be a list of its own: a slice
    of the table is an `array`, and the table itself follows later changes."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.model_engine.sequence import new_block_table

    sched = object.__new__(mc.MooncakeConnectorScheduler)
    sched.pp_size = 1
    sched.tp_size = 1
    sched.hash_block_size = 16
    sched.block_size = 16
    sched.dcp_size = 1
    sched.dp_rank = 0
    sched.engine_id = "eng"
    sched.host_ip = "10.0.0.1"
    sched.handshake_port = 40000
    sched.base_handshake_port = 6301
    sched.is_producer = is_producer
    sched._awaiting_send = set()
    sched._reqs_need_save = {}
    sched.request_id_to_transfer_id = {}

    seq = SimpleNamespace(
        kv_transfer_params={"do_remote_decode": is_producer},
        output_tokens=[7],
        spec_token_ids=None,
        num_prompt_tokens=2 * 16,
        block_table=new_block_table([1, 2, 3]),
        id=99,
        state_slot=-1,
        state_slots=[],
        kv_transfer_params_output=None,
    )
    mc.MooncakeConnectorScheduler.request_finished(sched, seq)
    finished = [seq.kv_transfer_params_output["remote_block_ids"]]
    if is_producer:
        finished.append(sched._reqs_need_save[seq.id][1])
    seq.block_table[0] = 9
    seq.block_table.append(4)

    for block_ids in finished:
        assert type(block_ids) is list
        assert block_ids == ([1, 2] if is_producer else [1, 2, 3])


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
    sched._producer_admission_undo = {}
    sched._pending_chunked_admissions = {}
    sched._chunked_waiting = {}
    sched._chunked_ready = set()
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
        state_slot=-1,
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


def test_per_request_state_keeps_the_dcp_block_ratio_on_a_full_transfer():
    # Kimi-K3 from a PP prefill (dcp=1) to a DCP decode. Its KDA state rules
    # out a block-only delta, so the transfer is full -- but the producer still
    # pairs dcp_size source blocks with each destination block, and rejects
    # the transfer outright if told the ratio is one.
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc, block_size=16, dcp_size=4)
    seq = _remote_prefill_seq(remote_hash_block_size=16)
    seq.kv_transfer_params["block_size"] = 16
    seq.kv_transfer_params["dcp_size"] = 1
    seq.has_per_req_cache = True

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 0
    assert seq.kv_transfer_params["src_block_skip_factor"] == 4


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
    monkeypatch.setattr(mc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mc, "gpu_pci_bdf", {0: "0000:65:00.0"}.__getitem__)
    monkeypatch.setattr(
        mooncake_hca, "rail_rdma_device", {"0000:65:00.0": "rdma2"}.__getitem__
    )
    monkeypatch.setattr(mooncake_hca, "ib_device_exists", lambda name: True)
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
        # Drive the write-request worker by hand: no thread, one round per call.
        conn._write_thread = object()
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
        # Nothing goes out on the engine thread: the request is posted by the
        # write-request worker, and only after the device has drained.
        conn._send_on_socket.assert_not_called()
        assert len(conn._write_queue) == 1
        order = []
        conn._send_on_socket.side_effect = lambda *a: order.append("send")
        monkeypatch.setattr(
            mc.torch.cuda, "synchronize", lambda *a: order.append("synchronize")
        )
        assert conn._post_queued_write_requests()
        assert order == ["synchronize", "send"]
        assert conn._write_request_error is None
        addr, (_, payload) = conn._send_on_socket.call_args.args
        request = mc.msgpack.loads(payload)
        assert addr == "tcp://10.19.0.113:6301"
        assert request["consumer_host"] == host_ip
        assert request["consumer_rpc_port"] == 16578
        assert request["notify_host"] == host_ip
        assert request["notify_port"] == 42000

        # A post that fails off-thread surfaces on the engine thread.
        conn._send_on_socket.side_effect = OSError("unreachable")
        conn._write_queue.append(("tcp://10.19.0.113:6301", [b"x"]))
        assert not conn._post_queued_write_requests()
        with pytest.raises(RuntimeError, match="could not post a write request"):
            conn.get_finished()


def _one_row_staging_consumer(monkeypatch, send):
    """A real consumer with one staging row whose producer runs `send`."""
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    monkeypatch.setenv("ATOM_HOST_IP", "10.19.0.140")
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE", raising=False)
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    monkeypatch.setattr(mc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mc, "gpu_pci_bdf", {0: "0000:05:00.0"}.__getitem__)
    monkeypatch.setattr(
        mooncake_hca, "rail_rdma_device", {"0000:05:00.0": "rdma0"}.__getitem__
    )
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(mc.torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(mooncake_hca, "ib_device_exists", lambda name: True)
    monkeypatch.setattr(
        mc, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        mc, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    engine = MagicMock()
    engine.initialize.return_value = 0
    engine.get_rpc_port.return_value = 16578
    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(mc, "TransferEngine", lambda: engine, raising=False)
    monkeypatch.setattr(mc, "get_open_port", lambda: 41000)
    monkeypatch.setattr(mc.zmq, "Context", MagicMock())
    conn = mc.MooncakeConnector(
        SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_rank=0),
            pipeline_parallel_size=1,
            hf_config=SimpleNamespace(num_hidden_layers=2),
            kv_cache_block_size=16,
            decode_context_parallel_size=1,
            dcp_config=SimpleNamespace(interleave_size=1),
            kv_transfer_config={"kv_role": "kv_consumer", "protocol": "rdma"},
        )
    )
    conn._notification_port = 42000
    conn._has_slot_regions = True
    conn._staging_pool_size = 1
    conn._staging_slots = mc.SlotPool(1)
    conn._staging_base_addr = 0
    conn._staging_slot_bytes = 64
    conn._send_on_socket = lambda addr, parts, repeat=1: send(conn, addr)
    conn._verify_producer = lambda addr: None
    return conn


def _receive_two(conn):
    """Run `start_load_kv` for two receives off-thread; return what it did."""

    def meta(i):
        return ConnectorMetadata._build_req_meta(
            req_id=f"r{i}",
            local_block_ids=[i],
            kv_transfer_params={
                "remote_block_ids": [i],
                "remote_host": "10.19.0.113",
                "remote_handshake_port": 6301,
                "tp_size": 1,
                "transfer_id": i,
            },
        )

    outcome = {}
    done = threading.Event()

    def run():
        try:
            conn.start_load_kv(
                SimpleNamespace(
                    request_id_to_transfer_id={"r0": 0, "r1": 1},
                    reqs_to_recv={"r0": meta(0), "r1": meta(1)},
                )
            )
        except RuntimeError as e:
            outcome["error"] = e
        done.set()

    threading.Thread(target=run, daemon=True).start()
    outcome["returned"] = done.wait(10)
    return outcome


def test_consumer_posts_each_receive_before_taking_the_next_staging_row(
    monkeypatch,
):
    """With one staging row, the second receive waits in `acquire()` until
    the first transfer frees the row, which needs its write request posted
    first. Posting only after the whole loop would deadlock."""
    posted = []
    both_posted = threading.Event()

    def send(conn, addr):
        # The transfer completes as soon as it is posted, returning the row.
        posted.append(addr)
        conn._staging_slots.release(0)
        if len(posted) == 2:
            both_posted.set()

    conn = _one_row_staging_consumer(monkeypatch, send)
    try:
        outcome = _receive_two(conn)
        assert outcome["returned"], "second receive never got a staging row"
        assert "error" not in outcome
        # Returning means the second request was queued, not yet posted.
        assert both_posted.wait(5), "second write request was never posted"
        assert len(posted) == 2
    finally:
        conn.close()
        if conn._write_thread is not None:
            conn._write_thread.join(5)


def test_a_failed_post_wakes_a_receive_waiting_for_a_staging_row(monkeypatch):
    """If the first request is never posted its row never comes back; the
    second receive must raise instead of blocking the engine thread."""

    def send(conn, addr):
        raise OSError("producer unreachable")

    conn = _one_row_staging_consumer(monkeypatch, send)
    outcome = _receive_two(conn)
    assert outcome["returned"], "engine thread stayed blocked on the staging row"
    assert isinstance(outcome.get("error"), RuntimeError)
    # The row stays held: a partly posted transfer may still write it.
    assert conn._staging_slots.free == []
    with pytest.raises(RuntimeError, match="could not post a write request"):
        conn.get_finished()
    # The worker stopped at the failure; nothing outlives the test.
    conn._write_thread.join(5)
    assert not conn._write_thread.is_alive()


def _queue_only_consumer():
    """A consumer whose write-request worker is driven by hand."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = object.__new__(mc.MooncakeConnector)
    conn.is_producer = False
    conn._cuda_device = 0
    conn._write_queue = []
    conn._write_cv = threading.Condition()
    conn._write_closed = False
    conn._write_thread = object()
    conn._write_request_error = None
    conn._verified_producers = set()
    conn._has_slot_regions = False
    conn._staging_slots = mc.SlotPool()
    return mc, conn


def test_one_synchronize_posts_every_request_queued_before_it(monkeypatch):
    """A burst of receives shares one device synchronize instead of paying
    one each, in order, behind the requests ahead of it."""
    mc, conn = _queue_only_consumer()
    events = []
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: events.append("sync"))
    conn._send_on_socket = lambda addr, parts: events.append(parts[0])
    for i in range(4):
        conn._enqueue_write_requests([("tcp://p:1", [f"r{i}".encode()])])
    assert conn._post_queued_write_requests()
    assert events == ["sync", b"r0", b"r1", b"r2", b"r3"]


def test_close_drops_queued_requests_and_stops_the_worker(monkeypatch):
    mc, conn = _queue_only_consumer()
    sent = []
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: None)
    conn._send_on_socket = lambda addr, parts: sent.append(parts)
    conn._enqueue_write_requests([("tcp://p:1", [b"r0"])])
    conn.close()
    assert not conn._post_queued_write_requests()
    conn._enqueue_write_requests([("tcp://p:1", [b"r1"])])
    assert sent == [] and conn._write_queue == []


def test_nothing_is_posted_after_close_returns(monkeypatch):
    """A batch the worker took before `close()` -- here, while it waits for the
    device -- must not go out afterwards: teardown releases the memory the
    producer would write into."""
    mc, conn = _queue_only_consumer()
    syncing, release = threading.Event(), threading.Event()

    def blocked_sync(*_):
        syncing.set()
        release.wait(5)

    monkeypatch.setattr(mc.torch.cuda, "synchronize", blocked_sync)
    sent = []
    conn._send_on_socket = lambda addr, parts: sent.append(parts)
    conn._enqueue_write_requests([("tcp://p:1", [b"r0"]), ("tcp://p:1", [b"r1"])])
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(more=conn._post_queued_write_requests()),
        daemon=True,
    )
    worker.start()
    try:
        assert syncing.wait(5)
        conn.close()
    finally:
        release.set()
        worker.join(5)
    assert sent == []
    assert result == {"more": False}


def test_consumer_refuses_a_producer_with_other_slot_sizes(monkeypatch):
    """Checked once, on first contact, before any write request goes out."""
    mc, conn = _queue_only_consumer()
    conn._has_slot_regions = True
    conn._slot_regions = [(0, 4096)]
    conn._decoder = mc.msgspec.msgpack.Decoder(mc.MooncakeAgentMetadata)
    reply = mc.msgspec.msgpack.encode(
        mc.MooncakeAgentMetadata(
            engine_id="p", rpc_port=1, has_slot_regions=True, slot_bps=[8192]
        )
    )
    sock = MagicMock()
    sock.poll.return_value = 1
    sock.recv_multipart.return_value = [b"", reply]
    conn.zmq_context = MagicMock()
    conn.zmq_context.socket.return_value = sock
    sent = []
    conn._send_on_socket = lambda addr, parts: sent.append(parts)
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: None)
    conn._enqueue_write_requests([("tcp://p:1", [b"r0"])])
    assert not conn._post_queued_write_requests()
    assert sent == []
    assert "state slots of [8192] B" in str(conn._write_request_error)
    sock.send_multipart.assert_called_once_with([mc.MSG_GET_META])

    # Matching sizes pass, and the producer is asked only once.
    _, conn2 = _queue_only_consumer()
    conn2.__dict__.update(
        _has_slot_regions=True,
        _slot_regions=[(0, 8192)],
        _decoder=conn._decoder,
        zmq_context=conn.zmq_context,
        _send_on_socket=lambda addr, parts: sent.append(parts),
    )
    for i in range(2):
        conn2._enqueue_write_requests([("tcp://p:1", [b"r%d" % i])])
        assert conn2._post_queued_write_requests()
    assert sent == [[b"r0"], [b"r1"]]
    assert sock.send_multipart.call_count == 2


def test_mooncake_tcp_disables_rdma_device_even_when_configured():
    assert mooncake_hca.ib_device_filter("tcp", "rdma0", None) == ""
    assert mooncake_hca.ib_device_filter(" TCP ", "ionic_0", None) == ""


def test_mooncake_tcp_forces_transfer_engine_transport(monkeypatch):
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    mooncake_hca.configure_transport(" TCP ")
    assert os.environ["MC_FORCE_TCP"] == "true"


def test_mooncake_rdma_preserves_explicit_device():
    assert mooncake_hca.ib_device_filter("rdma", "ionic_3", None) == "ionic_3"


def test_mooncake_rdma_normalizes_explicit_device_list():
    assert (
        mooncake_hca.ib_device_filter("rdma", " ionic_4, ionic_0,ionic_4 ", None)
        == "ionic_4,ionic_0"
    )


def test_mooncake_rdma_auto_selects_from_gpu_pci_address(monkeypatch):
    monkeypatch.setattr(
        mooncake_hca, "auto_select_ib_device", lambda bdf: f"auto-{bdf}"
    )
    assert (
        mooncake_hca.ib_device_filter("rdma", "", "0000:85:00.0") == "auto-0000:85:00.0"
    )


def test_mooncake_rdma_registers_all_alternate_hcas_for_upper_rail_gpu(monkeypatch):
    monkeypatch.setattr(
        mooncake_hca, "auto_select_ib_device", {"0000:85:00.0": "ionic_4"}.__getitem__
    )
    monkeypatch.setattr(mooncake_hca, "numbered_ib_device", lambda idx: f"ionic_{idx}")
    monkeypatch.setattr(mooncake_hca, "ib_device_exists", lambda _device: True)

    assert mooncake_hca.select_ib_devices(
        "rdma",
        "",
        "0000:85:00.0",
        enable_alternate_hca=True,
        hca_count=8,
    ) == [
        "ionic_4",
        *(f"ionic_{idx}" for idx in range(4)),
        *(f"ionic_{idx}" for idx in range(5, 8)),
    ]


def test_mooncake_rdma_registers_all_alternate_hcas_for_lower_rail_gpu(monkeypatch):
    monkeypatch.setattr(
        mooncake_hca, "auto_select_ib_device", {"0000:75:00.0": "ionic_3"}.__getitem__
    )
    monkeypatch.setattr(mooncake_hca, "numbered_ib_device", lambda idx: f"ionic_{idx}")
    monkeypatch.setattr(mooncake_hca, "ib_device_exists", lambda _device: True)

    assert mooncake_hca.select_ib_devices(
        "rdma",
        "",
        "0000:75:00.0",
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
    monkeypatch.setattr(
        mooncake_hca, "auto_select_ib_device", {"0000:75:00.0": "ionic_3"}.__getitem__
    )
    monkeypatch.setattr(mooncake_hca, "numbered_ib_device", lambda idx: f"ionic_{idx}")
    monkeypatch.setattr(
        mooncake_hca,
        "ib_device_exists",
        lambda device: device in {"ionic_3", "ionic_0", "ionic_7"},
    )

    assert mooncake_hca.select_ib_devices(
        "rdma",
        "",
        "0000:75:00.0",
        enable_alternate_hca=True,
        hca_count=8,
    ) == ["ionic_3", "ionic_0", "ionic_7"]


def test_mooncake_rdma_rejects_nonpositive_hca_count(monkeypatch):
    monkeypatch.setattr(
        mooncake_hca, "auto_select_ib_device", {"0000:85:00.0": "ionic_4"}.__getitem__
    )
    monkeypatch.setattr(mooncake_hca, "numbered_ib_device", lambda idx: f"ionic_{idx}")

    with pytest.raises(ValueError, match="ib_hca_count"):
        mooncake_hca.select_ib_devices(
            "rdma",
            "",
            "0000:85:00.0",
            enable_alternate_hca=True,
            hca_count=0,
        )


def test_mooncake_rdma_requires_gpu_address_without_explicit_device():
    with pytest.raises(ValueError, match="GPU's PCI address"):
        mooncake_hca.ib_device_filter("rdma", "", None)


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
    conn._pending_handoffs = {}
    conn._pending_chunked_requests = []
    conn._chunked_prefills = {}
    conn._admission_results = set()
    conn._chunked_admissions = {}
    conn._chunked_local_ids = {}
    conn._received_handoffs = {}
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
        lambda request, target, src, dst, data, req_id, *, engine=None, **kw: (
            captured.append((request, target, src, dst, data, req_id, engine, kw))
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


def test_slot_transfer_gets_the_prefill_kv_ready_event():
    """The staged index gather waits on this event; `_execute_transfer` must
    hand the slot path the one recorded for the prefill."""
    captured = []
    conn = _make_transfer_connector(
        {"block_ids": [3], "slot_index": 7, "swa_block_ids": [7]}, captured
    )
    ready = object()
    conn._kv_cache_ready_events["prefill-1"] = ready

    conn._execute_transfer(_stateful_write_request())

    assert len(captured) == 1
    assert captured[0][7]["kv_cache_ready_event"] is ready


@pytest.mark.parametrize("invalid_slot", [-1, None, "invalid"])
def test_stateful_transfer_without_final_source_slot_fails_closed(caplog, invalid_slot):
    captured = []
    conn = _make_transfer_connector(
        {"block_ids": [3], "slot_index": 3, "swa_block_ids": [3]}, captured
    )

    conn._execute_transfer(_stateful_write_request(src_slot_index=invalid_slot))

    assert captured == []
    assert "missing the producer's final source slot" in caplog.text


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_only_a_pp_stage_past_0_drops_its_producer_entry_once_served(pp_rank):
    captured = []
    conn = _make_transfer_connector({}, captured)
    conn.pp_size = 2
    conn.pp_rank = pp_rank
    conn._completed_prefills["prefill-1"] = {"block_ids": [3], "slot_index": 7}

    conn._execute_transfer(_stateful_write_request(src_block_ids=[3]))

    assert len(captured) == 1
    # Stage 0 keeps its entry for the release, which only it receives.
    assert ("prefill-1" in conn._completed_prefills) == (pp_rank == 0)
    assert conn.done_sending == set()


@pytest.mark.parametrize(
    ("producer_dcp", "consumer_dcp", "sent"),
    [(1, 2, True), (1, 8, True), (2, 4, False)],
    ids=["dcp1-to-dcp2", "dcp1-to-dcp8", "dcp2-to-dcp4"],
)
def test_slot_transfer_refuses_only_a_sharded_producer_of_another_dcp(
    caplog, producer_dcp, consumer_dcp, sent
):
    captured = []
    conn = _make_transfer_connector({}, captured)
    conn.pp_size = 2
    conn.dcp_size = producer_dcp

    conn._execute_transfer(
        _stateful_write_request(
            src_block_ids=list(range(3, 3 + consumer_dcp)),
            consumer_dcp_size=consumer_dcp,
        )
    )

    assert len(captured) == int(sent)
    assert ("does not support asymmetric DCP" in caplog.text) != sent


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
        _landing=None,
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
    conn = _make_connector(_landing=None)
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
    conn = _make_connector(_landing=None)
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
    conn = _make_connector(_staging_slots=mc.SlotPool(), _landing=None)
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
    conn = _make_connector(_landing=None)
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


def _fake_batch(req_ids, meta, *, pp_completion_ack=False):
    return SimpleNamespace(
        req_ids=list(req_ids),
        connector_meta_output=meta,
        pp_completion_ack=pp_completion_ack,
    )


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


def _fake_head(batch, *, launches=1, pp_size=4, pp_max_inflight=4):
    PPEngineCoreProc = _pp_engine_core_cls()

    head = SimpleNamespace(
        _in_flight=deque(),
        _defer_prefix_hash=False,
        pp_size=pp_size,
        pp_max_inflight=pp_max_inflight,
        _stages_push_load_terminals=False,
        kv_transfer_enabled=True,
        output_queue=MagicMock(),
        runner_mgr=MagicMock(),
        pp_transport=MagicMock(),
        scheduler=MagicMock(),
        _poll_kv_transfer_progress=MagicMock(),
        _dispatch_idle_offload_work=MagicMock(),
        _poll_dynamic_chunking_calibration=MagicMock(),
        # Real throttle, mocked dispatch: `_pp_head_step` runs on a loop that
        # never sleeps, so it must go through `_advance_idle_kv_transfer`.
        _next_idle_kv_drain=0.0,
    )
    head.scheduler.schedule.side_effect = [(batch, {})] * launches + [None]
    head.scheduler.take_rejected.return_value = None
    head.scheduler.dynamic_chunking_active = False
    head.pp_transport.recv_completion.return_value = None
    head._advance_idle_kv_transfer = PPEngineCoreProc._advance_idle_kv_transfer.__get__(
        head
    )
    head._dispatch_connector_only_batch = (
        PPEngineCoreProc._dispatch_connector_only_batch.__get__(head)
    )
    head._queue_or_flush_prefill_hashes = (
        PPEngineCoreProc._queue_or_flush_prefill_hashes.__get__(head)
    )
    head._collect_pp_completions = PPEngineCoreProc._collect_pp_completions.__get__(
        head
    )
    PPEngineCoreProc._pp_head_step(head)
    return head


def _dispatched_metas(head):
    return [
        c.args[1]
        for c in head.runner_mgr.call_func.call_args_list
        if c.args and c.args[0] == "process_kvconnector_output"
    ]


def test_the_head_fills_its_inflight_cap_past_the_stage_count():
    """Middle chunks count against the cap as output batches do, and a cap
    above the stage count keeps the head issuing until it is reached."""
    chunk = SimpleNamespace(
        req_ids=["r1"], connector_meta_output=None, produces_output=lambda: False
    )
    head = _fake_head(chunk, launches=8, pp_size=2, pp_max_inflight=4)
    assert len(head._in_flight) == 4
    assert head.scheduler.schedule.call_count == 4


@pytest.mark.parametrize("env, cap", [("0", 2), ("16", 16), ("-1", None)])
def test_pp_max_inflight_defaults_to_the_stage_count(monkeypatch, env, cap):
    pp_engine_core = sys.modules[_pp_engine_core_cls().__module__]
    monkeypatch.setenv("ATOM_PP_MAX_INFLIGHT", env)
    if cap is None:
        with pytest.raises(ValueError, match="must be >= 0"):
            pp_engine_core.pp_max_inflight(2)
    else:
        assert pp_engine_core.pp_max_inflight(2) == cap


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
        _is_engine_held=lambda: False,
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
    stage.pp_transport.send_completion.assert_not_called()


def _downstream_stage(**overrides):
    stage = SimpleNamespace(
        kv_transfer_enabled=False,
        is_last=True,
        runner_mgr=MagicMock(),
        pp_transport=MagicMock(),
        scheduler=MagicMock(),
        utility_handler=MagicMock(),
        utility_queue=MagicMock(),
        _is_engine_held=lambda: False,
        _poll_and_send_kv_status=MagicMock(),
    )
    for key, value in overrides.items():
        setattr(stage, key, value)
    return stage


def _run_last_stage(batch):
    PPEngineCoreProc = _pp_engine_core_cls()
    stage = _downstream_stage()
    stage.pull_and_process_input_queue = MagicMock(side_effect=[False, True])
    stage.pp_transport.recv_metadata.return_value = batch
    stage.runner_mgr.call_func.return_value = "fwd-out"
    PPEngineCoreProc._downstream_busy_loop(stage)
    return stage


@pytest.mark.parametrize("weights_offloaded", [False, True])
def test_a_held_stage_still_reports_its_kv_transfers(weights_offloaded):
    """The head waits for every stage's report before a KV change. A stage
    fenced by a weight sync stopped reporting, so the change waited out its
    deadline and refused a sync that had already written. With its weights,
    and so its KV pool, released, the stage runs nothing."""
    PPEngineCoreProc = _pp_engine_core_cls()
    stage = _downstream_stage(
        kv_transfer_enabled=True,
        _is_engine_held=lambda: True,
        _is_rl_weights_offloaded=weights_offloaded,
    )
    stage.pull_and_process_input_queue = MagicMock(side_effect=[False, True])

    PPEngineCoreProc._downstream_busy_loop(stage)

    stage.pp_transport.recv_metadata.assert_not_called()
    # The last call is the report every stage makes as it shuts down.
    held_reports = stage._poll_and_send_kv_status.call_count - 1
    assert held_reports == (0 if weights_offloaded else 1)


def test_pp_downstream_skips_the_middle_chunk_ack_by_default():
    """A middle chunk retires on the last stage without a completion message."""
    batch = _fake_batch([7], None)
    batch.produces_output = lambda: False

    stage = _run_last_stage(batch)

    stage.pp_transport.send_completion.assert_not_called()


def test_pp_downstream_acks_a_middle_chunk_the_head_asked_for():
    batch = _fake_batch([7], None, pp_completion_ack=True)
    batch.produces_output = lambda: False

    stage = _run_last_stage(batch)

    stage.pp_transport.send_completion.assert_called_once_with([7])


def test_pp_downstream_sends_output_whatever_the_ack_flag():
    batch = _fake_batch([7], None)
    batch.produces_output = lambda: True

    stage = _run_last_stage(batch)

    stage.pp_transport.send_completion.assert_called_once_with([7], "fwd-out")


def _head_for_batch(batch, *, dynamic_chunking_active=False):
    PPEngineCoreProc = _pp_engine_core_cls()
    head = SimpleNamespace(
        _in_flight=deque(),
        _defer_prefix_hash=False,
        pp_size=2,
        pp_max_inflight=2,
        _stages_push_load_terminals=False,
        kv_transfer_enabled=False,
        output_queue=MagicMock(),
        runner_mgr=MagicMock(),
        pp_transport=MagicMock(),
        scheduler=MagicMock(),
        _poll_kv_transfer_progress=MagicMock(),
        _dispatch_idle_offload_work=MagicMock(),
        _poll_dynamic_chunking_calibration=MagicMock(),
        _advance_idle_kv_transfer=MagicMock(),
        _next_idle_kv_drain=0.0,
    )
    head.scheduler.schedule.side_effect = [(batch, {7: SimpleNamespace(id=7)}), None]
    head.scheduler.take_rejected.return_value = None
    head.scheduler.dynamic_chunking_active = dynamic_chunking_active
    head._dispatch_connector_only_batch = MagicMock()
    head._queue_or_flush_prefill_hashes = (
        PPEngineCoreProc._queue_or_flush_prefill_hashes.__get__(head)
    )
    head._collect_pp_completions = PPEngineCoreProc._collect_pp_completions.__get__(
        head
    )
    return PPEngineCoreProc, head


@pytest.mark.parametrize("dynamic_chunking_active", [False, True])
def test_pp_head_waits_for_the_ack_of_a_middle_chunk(dynamic_chunking_active):
    # Retired unacked, the chunk could still be in stage 1's queue when the
    # head took its pipeline for empty and let a pause or a KV change go on.
    batch = _fake_batch([7], None)
    batch.produces_output = lambda: False
    PPEngineCoreProc, head = _head_for_batch(
        batch, dynamic_chunking_active=dynamic_chunking_active
    )
    head.pp_transport.recv_completion.return_value = None

    PPEngineCoreProc._pp_head_step(head)

    # Set before the metadata goes out, so the last stage reads the same flag.
    assert head.pp_transport.send_metadata.call_args.args[0].pp_completion_ack is True
    assert len(head._in_flight) == 1
    head.scheduler.release_pp_inflight.assert_not_called()

    head.pp_transport.recv_completion.return_value = ([7], None)
    PPEngineCoreProc._collect_pp_completions(head, 0)

    assert len(head._in_flight) == 0
    head.scheduler.release_pp_inflight.assert_called_once_with(batch)


def test_pp_head_rejects_a_final_chunk_whose_tokens_are_out_of_order():
    batch = _fake_batch([7], None)
    batch.produces_output = lambda: True
    PPEngineCoreProc, head = _head_for_batch(batch)
    head.pp_transport.recv_completion.return_value = (
        [7],
        SimpleNamespace(req_ids=[8]),
    )

    with pytest.raises(AssertionError, match="PP token ordering violated"):
        PPEngineCoreProc._pp_head_step(head)


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
    connector._block_unit_specs = None
    connector._block_head_major_runs = ()
    connector._unit_layout = None
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


# ---------------------------------------------------------------------------
# Stateful P/D from an unsharded PP stage to a DCP x TP decode (Kimi-K3:
# PP8xTP1 prefill, TP8 DCP8 decode). The stub NIC copies bytes within one
# address space, so each case checks what lands, not just what was asked for.
# ---------------------------------------------------------------------------

_TOKEN_BYTES = 8
_BLOCK = 4
_BPB = _BLOCK * _TOKEN_BYTES
_CONV_ROWS = 2
_CONV_RUN = 4  # one of q, k, v in a conv row, at producer TP
_CONV_BPS = _CONV_ROWS * 3 * _CONV_RUN
_SSM_BPS = 16
_CURSOR_BPS = 4
_SRC_SLOTS = 8
# Producer regions, then the consumer's: two MLA layers, three KDA layers and
# ReplaySSM's cursor.
_MLA_SRC = 0
_CONV_SRC = _MLA_SRC + 4 * _BPB
_SSM_SRC = _CONV_SRC + _SRC_SLOTS * _CONV_BPS
_CURSOR_SRC = _SSM_SRC + _SRC_SLOTS * _SSM_BPS
_DST_MLA = [4096 + i * 4 * _BPB for i in range(2)]
_DST_SLOT = [8192 + i * 1024 for i in range(7)]


class _ByteNic:
    def __init__(self):
        np = pytest.importorskip("numpy")
        self.mem = np.zeros(16384, dtype=np.uint8)

    def write(
        self, _target, src, dst, sizes, _req_id, _label, *, engine=None, deadline=None
    ):
        for s, d, n in zip(src, dst, sizes):
            self.mem[d : d + n] = self.mem[s : s + n].copy()
        return True


def _pp_stage_connector(mc, nic, *, tp_size=1, tp_rank=0, ssm_segments=True):
    from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE

    connector = object.__new__(mc.MooncakeConnector)
    connector.dcp_size = 1
    connector.block_size = _BLOCK
    connector.tp_size = tp_size
    connector.tp_rank = tp_rank
    connector.pp_size = 2
    connector._num_local_layers = 2
    connector._start_layer = 4
    connector._fp4_index_layout = False
    connector._block_unit_specs = None
    connector._block_regions = [(_MLA_SRC, _BPB)]
    connector._per_block_bytes_list = [_BPB]
    connector._block_region_roles = [MLA_KV_ROLE]
    connector._block_head_major_runs = [None]
    # This stage's one MLA layer is the peer's second, its one KDA layer the
    # peer's third (conv region 2, SSM region 5), then the shared cursor.
    connector._block_region_consumer_indices = [1]
    connector._swa_block_regions = []
    connector._slot_regions = [
        (_CONV_SRC, _CONV_BPS),
        (_SSM_SRC, _SSM_BPS),
        (_CURSOR_SRC, _CURSOR_BPS),
    ]
    connector._slot_region_consumer_indices = [2, 5, 6]
    connector._slot_region_tp_segments = [
        (_CONV_RUN,) * (3 * _CONV_ROWS),
        (_SSM_BPS,) if ssm_segments else None,
        None,
    ]
    connector._slot_region_tp_replicated = [False, False, True]
    connector._gather_slot = None
    connector._rdma_write = nic.write
    return connector


def _dcp_tp_request(consumer_tp_size=2, consumer_tp_rank=1, ratio=2):
    from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE

    return {
        "consumer_block_base_addrs": _DST_MLA,
        "consumer_block_bpb": [_BPB, _BPB],
        "consumer_region_roles": [MLA_KV_ROLE, MLA_KV_ROLE],
        "consumer_slot_base_addrs": _DST_SLOT,
        "consumer_slot_bps": [_CONV_BPS // ratio] * 3
        + [_SSM_BPS // ratio] * 3
        + [_CURSOR_BPS],
        "dst_slot_index": 3,
        "consumer_dcp_size": 2,
        "consumer_dcp_rank": 1,
        "consumer_dcp_interleave": 1,
        "consumer_tp_size": consumer_tp_size,
        "consumer_tp_rank": consumer_tp_rank,
    }


def _run_slot_transfer(mc, connector, request_data):
    return mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        request_data,
        "consumer:1234",
        [0, 1, 2, 3],
        [1, 2],
        {"slot_index": 5},
        "req-1",
    )


def test_pp_stage_relayouts_kv_and_splits_kda_state_for_a_dcp_tp_consumer():
    np = pytest.importorskip("numpy")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    nic = _ByteNic()
    # Token t of the prompt (block t // 4) is eight bytes of t + 1; every
    # state byte is distinct.
    for t in range(4 * _BLOCK):
        nic.mem[_MLA_SRC + t * _TOKEN_BYTES : _MLA_SRC + (t + 1) * _TOKEN_BYTES] = t + 1
    nic.mem[_CONV_SRC : _CURSOR_SRC + _SRC_SLOTS * _CURSOR_BPS] = np.arange(
        _CURSOR_SRC + _SRC_SLOTS * _CURSOR_BPS - _CONV_SRC, dtype=np.uint8
    )
    before = nic.mem.copy()

    assert _run_slot_transfer(mc, _pp_stage_connector(mc, nic), _dcp_tp_request())

    conv = before[_CONV_SRC + 5 * _CONV_BPS : _CONV_SRC + 6 * _CONV_BPS]
    ssm = before[_SSM_SRC + 5 * _SSM_BPS : _SSM_SRC + 6 * _SSM_BPS]
    cursor = before[_CURSOR_SRC + 5 * _CURSOR_BPS : _CURSOR_SRC + 6 * _CURSOR_BPS]
    landed = [
        # DCP rank 1 of 2 at interleave 1 owns the odd tokens, packed in order
        # into destination blocks 1 and 2 of the peer's second MLA layer.
        (
            _DST_MLA[1] + _BPB,
            np.repeat(np.arange(2, 17, 2, dtype=np.uint8), _TOKEN_BYTES),
        ),
        # TP rank 1 of 2 holds the second half of every q, k and v run in each
        # conv row and the second half of the SSM heads; the cursor lands whole.
        (
            _DST_SLOT[2] + 3 * (_CONV_BPS // 2),
            conv.reshape(_CONV_ROWS, 3, 2, _CONV_RUN // 2)[:, :, 1].ravel(),
        ),
        (_DST_SLOT[5] + 3 * (_SSM_BPS // 2), ssm[_SSM_BPS // 2 :]),
        (_DST_SLOT[6] + 3 * _CURSOR_BPS, cursor),
    ]
    written = np.zeros_like(nic.mem, dtype=bool)
    for dst, want in landed:
        assert (nic.mem[dst : dst + want.size] == want).all()
        written[dst : dst + want.size] = True
    assert (nic.mem[~written] == before[~written]).all()


class _SlotView:
    """One slot region's units inside the byte NIC's flat memory."""

    def __init__(self, mem, base, unit):
        self.mem, self.base, self.unit = mem, base, unit

    def __getitem__(self, slot):
        start = self.base + slot * self.unit
        return self.mem[start : start + self.unit]


def test_staged_state_matches_the_direct_shard(_no_cuda_stream):
    """Gathering the TP shard writes the same bytes as the direct descriptors,
    in one descriptor per region instead of one per conv run."""
    np = pytest.importorskip("numpy")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import SlotPool

    nic = _ByteNic()
    for t in range(4 * _BLOCK):
        nic.mem[_MLA_SRC + t * _TOKEN_BYTES : _MLA_SRC + (t + 1) * _TOKEN_BYTES] = t + 1
    nic.mem[_CONV_SRC : _CURSOR_SRC + _SRC_SLOTS * _CURSOR_BPS] = np.arange(
        _CURSOR_SRC + _SRC_SLOTS * _CURSOR_BPS - _CONV_SRC, dtype=np.uint8
    )
    before = nic.mem.copy()
    connector = _pp_stage_connector(mc, nic)
    base = 15000
    connector._state_staging = nic.mem[base : base + 64].reshape(1, 64)
    connector._state_staging_base = base
    connector._state_staging_slot_bytes = 64
    connector._state_staging_slots = SlotPool(1)
    connector._state_region_views = [
        _SlotView(nic.mem, _CONV_SRC, _CONV_BPS),
        _SlotView(nic.mem, _SSM_SRC, _SSM_BPS),
        _SlotView(nic.mem, _CURSOR_SRC, _CURSOR_BPS),
    ]
    connector._send_worker_stream = MagicMock()
    labels = []
    nic_write = nic.write

    def write(*args, **kwargs):
        labels.append(args[5])
        return nic_write(*args, **kwargs)

    connector._rdma_write = write

    assert mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        _dcp_tp_request(),
        "consumer:1234",
        [0, 1, 2, 3],
        [1, 2],
        {"slot_index": 5},
        "req-1",
        kv_cache_ready_event=object(),
    )

    conv = before[_CONV_SRC + 5 * _CONV_BPS : _CONV_SRC + 6 * _CONV_BPS]
    ssm = before[_SSM_SRC + 5 * _SSM_BPS : _SSM_SRC + 6 * _SSM_BPS]
    cursor = before[_CURSOR_SRC + 5 * _CURSOR_BPS : _CURSOR_SRC + 6 * _CURSOR_BPS]
    landed = [
        (
            _DST_SLOT[2] + 3 * (_CONV_BPS // 2),
            conv.reshape(_CONV_ROWS, 3, 2, _CONV_RUN // 2)[:, :, 1].ravel(),
        ),
        (_DST_SLOT[5] + 3 * (_SSM_BPS // 2), ssm[_SSM_BPS // 2 :]),
        (_DST_SLOT[6] + 3 * _CURSOR_BPS, cursor),
    ]
    for dst, want in landed:
        assert (nic.mem[dst : dst + want.size] == want).all()
    assert labels.count("staged-state") == 1
    connector._send_worker_stream().wait_event.assert_called_once()


def test_a_state_shard_larger_than_the_staging_slot_stays_direct(_no_cuda_stream):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import SlotPool

    nic = _ByteNic()
    connector = _pp_stage_connector(mc, nic)
    connector._state_staging = nic.mem[15000:15004].reshape(1, 4)
    connector._state_staging_base = 15000
    connector._state_staging_slot_bytes = 4
    connector._state_staging_slots = SlotPool(1)
    connector._state_region_views = [
        _SlotView(nic.mem, _CONV_SRC, _CONV_BPS),
        _SlotView(nic.mem, _SSM_SRC, _SSM_BPS),
        _SlotView(nic.mem, _CURSOR_SRC, _CURSOR_BPS),
    ]
    labels = []
    nic_write = nic.write

    def write(*args, **kwargs):
        labels.append(args[5])
        return nic_write(*args, **kwargs)

    connector._rdma_write = write
    connector._send_worker_stream = MagicMock()
    assert mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        _dcp_tp_request(),
        "consumer:1234",
        [0, 1, 2, 3],
        [1, 2],
        {"slot_index": 5},
        "req-1",
        kv_cache_ready_event=object(),
    )
    assert "staged-state" not in labels


def _arm_state_staging(nic, connector, *, slot_bytes=64):
    from atom.kv_transfer.disaggregation.mooncake.landing import LandingCredits
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import SlotPool

    base = 15000
    connector._state_staging = nic.mem[base : base + slot_bytes].reshape(1, slot_bytes)
    connector._state_staging_base = base
    connector._state_staging_slot_bytes = slot_bytes
    connector._state_staging_slots = SlotPool(1)
    connector._state_region_views = [
        _SlotView(nic.mem, _CONV_SRC, _CONV_BPS),
        _SlotView(nic.mem, _SSM_SRC, _SSM_BPS),
        _SlotView(nic.mem, _CURSOR_SRC, _CURSOR_BPS),
    ]
    connector._send_worker_stream = MagicMock()
    connector._state_landing_credits = LandingCredits()
    labels = []
    nic_write = nic.write

    def write(*args, **kwargs):
        labels.append(args[5])
        return nic_write(*args, **kwargs)

    connector._rdma_write = write
    return labels


def test_one_state_slot_lands_and_a_failed_write_quarantines_its_credit(
    _no_cuda_stream,
):
    """One slot is enough for state. A failed write keeps that credit out."""
    np = pytest.importorskip("numpy")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.state_gather import KDA_SLOT_LAYOUT

    nic = _ByteNic()
    nic.mem[_CONV_SRC : _CURSOR_SRC + _SRC_SLOTS * _CURSOR_BPS] = np.arange(
        _CURSOR_SRC + _SRC_SLOTS * _CURSOR_BPS - _CONV_SRC, dtype=np.uint8
    )
    before = nic.mem.copy()
    connector = _pp_stage_connector(mc, nic)
    connector.pp_rank = 0
    labels = _arm_state_staging(nic, connector)
    connector._state_landing_credits.sync("consumer:1234", {"epoch": 7, "slots": [0]})
    sent = []
    connector._send_on_socket = lambda addr, parts, repeat=1: sent.append(parts)
    request = _dcp_tp_request()
    request.update(
        notify_host="decode",
        notify_port=9,
        state_landing={
            "layout": KDA_SLOT_LAYOUT,
            "epoch": 7,
            "base": 15200,
            "slot_bytes": 64,
            "slots": [0],
        },
    )
    assert mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        request,
        "consumer:1234",
        [0, 1, 2, 3],
        [1, 2],
        {"slot_index": 5},
        "req-1",
        kv_cache_ready_event=object(),
    )
    conv = before[_CONV_SRC + 5 * _CONV_BPS : _CONV_SRC + 6 * _CONV_BPS]
    packed = np.concatenate(
        [
            conv.reshape(_CONV_ROWS, 3, 2, _CONV_RUN // 2)[:, :, 1].ravel(),
            before[_SSM_SRC + 5 * _SSM_BPS : _SSM_SRC + 6 * _SSM_BPS][_SSM_BPS // 2 :],
            before[_CURSOR_SRC + 5 * _CURSOR_BPS : _CURSOR_SRC + 6 * _CURSOR_BPS],
        ]
    )
    assert (nic.mem[15200 : 15200 + packed.size] == packed).all()
    # The decode slot itself is untouched: scatter, not this write, fills it.
    assert not nic.mem[_DST_SLOT[2] : _DST_SLOT[2] + _CONV_BPS // 2].any()
    assert labels.count("landed-state") == 1
    assert request["_state_landed_slots"] == [0]
    assert sent[0][0] == b"state_landing_ready"
    items = __import__("msgpack").loads(sent[0][1])["items"]
    assert items == [[2, 0, 1, 0], [5, 0, 1, 12], [6, 0, 1, 20]]

    # The decode returns the credit. The next write fails, and that credit
    # stays out of the free list instead of being handed on.
    connector._state_landing_credits.release("consumer:1234", 7, [0])

    def fail_landed(*args, **kwargs):
        return args[5] != "landed-state"

    connector._rdma_write = fail_landed
    assert not mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        request,
        "consumer:1234",
        [0, 1, 2, 3],
        [1, 2],
        {"slot_index": 5},
        "req-1",
        kv_cache_ready_event=object(),
    )
    held = connector._state_landing_credits._targets["consumer:1234"]
    assert 0 in held.quarantined
    assert held.free == []


def test_a_non_dcp_consumer_keeps_the_direct_state_writes(_no_cuda_stream):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    nic = _ByteNic()
    connector = _pp_stage_connector(mc, nic)
    labels = _arm_state_staging(nic, connector)
    request = _dcp_tp_request()
    request["consumer_dcp_size"] = 1
    request["consumer_dcp_rank"] = 0
    # Without DCP the MLA pages go whole; only the state decision matters.
    assert (
        mc.MooncakeConnector._try_staged_state(
            connector,
            request,
            "consumer:1234",
            5,
            3,
            [2, 5, 6],
            connector._slot_region_tp_segments,
            connector._slot_region_tp_replicated,
            True,
            2,
            1,
            _DST_SLOT,
            [_CONV_BPS // 2] * 3 + [_SSM_BPS // 2] * 3 + [_CURSOR_BPS],
            "req-1",
            object(),
        )
        is None
    )
    assert labels == []


def test_state_landing_is_swept_and_acked_with_the_mla_one():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector = object.__new__(mc.MooncakeConnector)
    calls = []

    class _Receiver:
        def sweep(self):
            calls.append("sweep")

        def stage_acked(self, *args):
            calls.append(("ack", args))
            return True

    connector._state_landing = _Receiver()
    connector._landing = None
    connector._write_request_error = None
    connector.is_producer = False

    class _Stop(Exception):
        pass

    def stop():
        raise _Stop

    # get_finished sweeps both receivers before anything else it does.
    connector._expire_pending_recvs = stop
    with pytest.raises(_Stop):
        mc.MooncakeConnector.get_finished(connector)
    assert calls[0] == "sweep"
    mc.MooncakeConnector._record_write_ack(
        connector, {"request_id": "r", "write_nonce": 1, "write_window_s": 5}
    )
    assert calls[1][0] == "ack"


def test_two_landing_receivers_finish_a_request_once():
    import threading

    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector = object.__new__(mc.MooncakeConnector)
    connector._completion_lock = threading.Lock()
    connector._landing_gates = {"r": {"left": 2, "failed": False}}
    done = []
    connector._complete_recv = lambda req, failed: done.append((req, failed))
    mc.MooncakeConnector._landing_finished(connector, "r", False)
    assert done == []
    mc.MooncakeConnector._landing_finished(connector, "r", True)
    assert done == [("r", True)]
    assert "r" not in connector._landing_gates
    # One receiver, the MLA-only case, completes on its own finish.
    mc.MooncakeConnector._landing_finished(connector, "solo", False)
    assert done == [("r", True), ("solo", False)]


# ---------------------------------------------------------------------------
# Kimi-K3.1's MLA rows on the slot path. A PP stage gathers a DCP rank's rows
# into staging and writes or lands them, beside its index pages and KDA state.
# The per-token direct path the staging replaces is the byte reference.
# ---------------------------------------------------------------------------

_K_BLOCK = 16
_K_ROW = 8  # one MLA row
_K_IDX_ROW = 4  # one index row
_K_MLA_PAGE = _K_BLOCK * _K_ROW
_K_IDX_PAGE = _K_BLOCK * _K_IDX_ROW
_K_SRC_PAGES = 6
# This stage: a target MLA row, a DSpark draft row and one index layer, after
# `_pp_stage_connector`'s KDA slots. Five source pages are 80 tokens, so each
# of two DCP ranks owns 40: two full destination pages and half a third.
_K_MLA_SRC = (1024, 2048)
_K_IDX_SRC = 3072
_K_SRC_BLOCKS = (4, 1, 5, 2, 0)
# The PP=1 peer: three MLA rows, two index layers and the KDA slots.
_K_DST_MLA = [8192 + i * 1024 for i in range(3)]
_K_DST_IDX = [11264 + i * 512 for i in range(2)]
_K_DST_SLOT = [16384 + i * 1024 for i in range(7)]
_K_DST_BLOCKS = (6, 2, 3)
_K_IDX_STAGING = 24576
_K_KV_STAGING = 32768


class _PtrNic(_ByteNic):
    """A `_ByteNic` that also takes real addresses, which a torch staging slot
    or landing pool names its bytes by: an address past ``mem`` is a pointer.
    Counts descriptors per write label."""

    def __init__(self):
        np = pytest.importorskip("numpy")
        self.mem = np.random.default_rng(31).integers(0, 256, 1 << 16, dtype=np.uint8)
        self.descriptors = {}

    def ptr(self, addr):
        return self.mem.ctypes.data + addr if addr < self.mem.size else addr

    def tensor(self, start, stop):
        torch = pytest.importorskip("torch")
        return torch.from_numpy(self.mem[start:stop])

    def write(
        self, _target, src, dst, sizes, _req_id, label, *, engine=None, deadline=None
    ):
        import ctypes

        for s, d, n in zip(src, dst, sizes):
            ctypes.memmove(self.ptr(int(d)), self.ptr(int(s)), int(n))
        self.descriptors[label] = self.descriptors.get(label, 0) + len(src)
        return True

    def compared(self):
        """Sources, then every destination a transfer writes."""
        np = pytest.importorskip("numpy")
        return np.concatenate(
            [self.mem[:4096], self.mem[8192:12288], self.mem[16384:23552]]
        )


def _kimi_stage(mc, nic, *, slot_bytes=None):
    """A Kimi-K3.1 PP stage on ``nic``: `_pp_stage_connector`'s KDA slots, two
    MLA rows and one index layer at a 16-token block. ``slot_bytes`` arms MLA
    staging with two slots of that size; without it the rows go per token."""
    from atom.kv_transfer.disaggregation.mooncake.landing import LandingCredits
    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    connector = _pp_stage_connector(mc, nic)
    connector.pp_rank = 1
    connector.block_size = _K_BLOCK
    bases = [*_K_MLA_SRC, _K_IDX_SRC]
    widths = [_K_MLA_PAGE, _K_MLA_PAGE, _K_IDX_PAGE]
    connector._block_regions = list(zip(bases, widths))
    connector._per_block_bytes_list = widths
    connector._block_region_roles = [MLA_KV_ROLE, MLA_KV_ROLE, INDEX_CACHE_ROLE]
    connector._block_head_major_runs = [None] * 3
    connector._block_region_consumer_indices = [1, 2, 4]
    connector._send_worker_stream = MagicMock()
    connector._landing_credits = LandingCredits()
    # The backend packs a rank's index rows into preshuffled 16-row tiles; this
    # stand-in packs them row by row, enough to see them reach the peer.
    connector._index_staging_chunk_pages = 256
    connector._index_staging_slots = mc.SlotPool(1)
    connector._prepare_sharded_index = lambda plan: plan.source_token_per_dst_token()

    def gather(region, tokens, _pool_idx):
        base = connector._block_regions[region][0]
        for j, token in enumerate(tokens):
            src = base + int(token) * _K_IDX_ROW
            dst = _K_IDX_STAGING + j * _K_IDX_ROW
            nic.mem[dst : dst + _K_IDX_ROW] = nic.mem[src : src + _K_IDX_ROW]
        return _K_IDX_STAGING, len(tokens) // _K_BLOCK

    connector._gather_sharded_index = gather
    if slot_bytes is not None:
        connector._kv_staging = nic.tensor(
            _K_KV_STAGING, _K_KV_STAGING + 2 * slot_bytes
        ).view(2, slot_bytes)
        connector._kv_staging_slots = mc.SlotPool(2)
        connector._kv_source_views = {
            i: nic.tensor(base, base + _K_SRC_PAGES * _K_MLA_PAGE)
            for i, base in enumerate(_K_MLA_SRC)
        }
    return connector


def _kimi_request(dcp_rank, *, dcp=2):
    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    request = _dcp_tp_request()
    request.update(
        request_id="req-1",
        write_nonce=5,
        notify_host="decode",
        notify_port=9,
        consumer_block_base_addrs=_K_DST_MLA + _K_DST_IDX,
        consumer_block_bpb=[_K_MLA_PAGE] * 3 + [_K_IDX_PAGE] * 2,
        consumer_region_roles=[MLA_KV_ROLE] * 3 + [INDEX_CACHE_ROLE] * 2,
        consumer_slot_base_addrs=_K_DST_SLOT,
        consumer_dcp_size=dcp,
        consumer_dcp_rank=dcp_rank,
    )
    return request


def _kimi_transfer(mc, connector, request, dst_blocks=_K_DST_BLOCKS, ready=True):
    return mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        request,
        "consumer:1234",
        list(_K_SRC_BLOCKS),
        list(dst_blocks),
        {"slot_index": 5},
        "req-1",
        kv_cache_ready_event=object() if ready else None,
    )


def _kimi_oracle(mc, dcp_rank, *, dcp=2, dst_blocks=_K_DST_BLOCKS):
    """Today's path: no MLA staging, so the rows go one token per descriptor."""
    nic = _PtrNic()
    assert _kimi_transfer(
        mc, _kimi_stage(mc, nic), _kimi_request(dcp_rank, dcp=dcp), dst_blocks
    )
    assert "staged-kv" not in nic.descriptors
    return nic


def _assert_rank_rows(nic, before, dcp_rank):
    """Row j of a DCP rank is global token ``j * 2 + rank``, at token j % 16
    of destination page j // 16, worked out apart from either plan."""
    np = pytest.importorskip("numpy")
    owned = np.arange(dcp_rank, len(_K_SRC_BLOCKS) * _K_BLOCK, 2)
    tokens = np.asarray(_K_SRC_BLOCKS)[owned // _K_BLOCK] * _K_BLOCK + owned % _K_BLOCK
    for src_base, dst_base, row in (
        (_K_MLA_SRC[0], _K_DST_MLA[1], _K_ROW),
        (_K_MLA_SRC[1], _K_DST_MLA[2], _K_ROW),
        (_K_IDX_SRC, _K_DST_IDX[1], _K_IDX_ROW),
    ):
        for j, token in enumerate(tokens):
            page = _K_DST_BLOCKS[j // _K_BLOCK]
            dst = dst_base + (page * _K_BLOCK + j % _K_BLOCK) * row
            src = src_base + int(token) * row
            assert (nic.mem[dst : dst + row] == before[src : src + row]).all()


@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_kimi_staged_mla_rows_match_the_per_token_path(_no_cuda_stream, dcp_rank):
    """One slot holds both rows' 40 tokens, so each goes out as one descriptor
    per run of adjacent destination pages instead of one per token."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    oracle = _kimi_oracle(mc, dcp_rank)
    nic = _PtrNic()
    before = nic.mem.copy()
    connector = _kimi_stage(mc, nic, slot_bytes=1024)

    assert _kimi_transfer(mc, connector, _kimi_request(dcp_rank))

    assert (nic.compared() == oracle.compared()).all()
    _assert_rank_rows(nic, before, dcp_rank)
    # Two rows x 40 tokens before; pages 6 and 2-3 of each row now.
    assert oracle.descriptors["block"] == 80
    assert nic.descriptors["staged-kv"] == 4
    assert "block" not in nic.descriptors
    # Index pages and KDA state still take their own paths.
    assert nic.descriptors["staged-index"] == oracle.descriptors["staged-index"]
    assert nic.descriptors["slot"] == oracle.descriptors["slot"]
    # The rows and the index pages both wait for this prefill's KV.
    assert connector._send_worker_stream().wait_event.call_count == 2
    assert sorted(connector._kv_staging_slots.free) == [0, 1]


def _kimi_receiver(nic, *, bases, widths, regions, slot_bytes, credit_pool=""):
    """A CPU decode-side receiver whose scatter is a memmove into ``nic``."""
    import ctypes

    from atom.kv_transfer.disaggregation.landing_scatter import (
        segment_table_capacity,
    )
    from atom.kv_transfer.disaggregation.mooncake.landing import LandingReceiver

    sent, finished = [], []
    recv = LandingReceiver(
        device=None,
        pool_slots=8,
        slot_bytes=slot_bytes,
        units_per_page=_K_BLOCK if not credit_pool else 1,
        consumer_key="consumer:1234",
        region_bases=bases,
        region_block_bytes=widths,
        region_indices=regions,
        send=lambda addr, parts: sent.append((addr, parts)),
        finish=lambda req, failed: finished.append((req, failed)),
        table_segments=segment_table_capacity(1024),
        credit_pool=credit_pool,
    )

    def copy(src, dst, nbytes, _events):
        for s, d, n in zip(src, dst, nbytes):
            ctypes.memmove(
                nic.ptr(recv._dst_origin + int(d)),
                recv.pool.data_ptr() + int(s),
                int(n),
            )

    recv._copy_segments = copy
    recv.sent, recv.finished = sent, finished
    return recv


def _run_listener(monkeypatch, mc, listener, messages):
    """Run a connector's listener loop over ``(type, payload)`` messages."""
    from contextlib import nullcontext

    class _Stop(Exception):
        pass

    class _Socket:
        def poll(self, timeout):
            if not messages:
                raise _Stop
            return mc.zmq.POLLIN

        def getsockopt(self, _option):
            return b"tcp://0.0.0.0:1"

        def recv_multipart(self):
            if not messages:
                raise _Stop
            return [b"peer", *messages.pop(0)]

    conn = listener.__self__
    conn._completed_prefills_lock = threading.Lock()
    conn._pending_chunked_requests = []
    conn._chunked_admissions = {}
    conn._chunked_local_ids = {}
    conn._listener_bound = threading.Event()
    conn._listener_error = None
    monkeypatch.setattr(mc, "zmq_socket_ctx", lambda *_a, **_k: nullcontext(_Socket()))
    with pytest.raises(_Stop):
        listener()


@pytest.mark.parametrize("offered", ["both", "mla", "state"])
@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_kimi_mla_rows_land_beside_the_kda_state(
    _no_cuda_stream, monkeypatch, dcp_rank, offered
):
    """The decode scatters both landings into the per-token path's bytes, the
    request completes once after both, and each credit goes back to its pool.

    With one landing offered, the other half goes out staged and the request
    completes on the one receiver that tracks it.
    """
    msgpack = pytest.importorskip("msgpack")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.state_gather import KDA_SLOT_LAYOUT

    oracle = _kimi_oracle(mc, dcp_rank)
    nic = _PtrNic()
    # Two 256-byte slots: 40 rows x 2 rows x 8 B pack into three, so they land.
    connector = _kimi_stage(mc, nic, slot_bytes=2 * _K_MLA_PAGE)
    _arm_state_staging(nic, connector)
    connector._rdma_write = nic.write
    sent = []
    connector._send_on_socket = lambda _addr, parts, repeat=1: sent.extend(
        [parts] * repeat
    )

    mla = _kimi_receiver(
        nic,
        bases=_K_DST_MLA + _K_DST_IDX,
        widths=[_K_MLA_PAGE] * 3 + [_K_IDX_PAGE] * 2,
        regions=[0, 1, 2],
        slot_bytes=2 * _K_MLA_PAGE,
    )
    state = _kimi_receiver(
        nic,
        bases=_K_DST_SLOT,
        widths=[_CONV_BPS // 2] * 3 + [_SSM_BPS // 2] * 3 + [_CURSOR_BPS],
        regions=list(range(7)),
        slot_bytes=64,
        credit_pool="state",
    )
    consumer = object.__new__(mc.MooncakeConnector)
    consumer._landing, consumer._state_landing = mla, state
    consumer._completion_lock = threading.Lock()
    consumer._pending_handoffs = {}
    consumer._landing_gates = {}
    done = []
    consumer._complete_recv = lambda req, failed: done.append((req, failed))
    for recv in (mla, state):
        recv._finish = consumer._landing_finished

    request = _kimi_request(dcp_rank)
    began = 0
    if offered in ("both", "mla"):
        request["mla_landing"] = mla.advertise("stage1", 2)
        connector._landing_credits.sync("consumer:1234", request["mla_landing"])
        mla.begin("req-1", 5, list(_K_DST_BLOCKS), {1: "stage1"})
        began += 1
    if offered in ("both", "state"):
        request["state_landing"] = {
            **state.advertise("stage1", 2),
            "layout": KDA_SLOT_LAYOUT,
        }
        connector._state_landing_credits.sync("consumer:1234", request["state_landing"])
        state.begin("req-1", 5, [3], {1: "stage1"})
        began += 1
    if began > 1:  # as start_load_kv arms it
        consumer._landing_gates["req-1"] = {"left": began, "failed": False}

    assert _kimi_transfer(mc, connector, request)
    mc.MooncakeConnector._notify_transfer_result(connector, request, success=True)
    kinds = [parts[0] for parts in sent]
    assert kinds.count(mc.MSG_LANDING_READY) == (3 if offered != "state" else 0)
    assert kinds.count(mc.MSG_STATE_LANDING_READY) == (1 if offered != "mla" else 0)
    assert kinds.count(mc.MSG_WRITE_DONE) == 3  # sent three times, counted once
    # One descriptor per landed slot; staged, the three slots take five.
    if offered == "state":
        assert nic.descriptors["staged-kv"] == 5 and "landed-kv" not in nic.descriptors
    else:
        assert nic.descriptors["landed-kv"] == 3 and "staged-kv" not in nic.descriptors

    _run_listener(monkeypatch, mc, consumer._notification_listener, sent)
    assert done == []  # nothing scattered yet
    for recv in (mla, state):
        recv._process(recv._drain(block=False))
    assert done == [("req-1", False)]
    assert consumer._landing_gates == {}
    assert (nic.compared() == oracle.compared()).all()

    # The returned credits reach the pool that lent them; each pool's epoch
    # is its own, so a credit read against the other pool is dropped.
    credits = [parts for _addr, parts in mla.sent + state.sent]
    assert {msgpack.loads(p).get("pool") for _kind, p in credits} == {
        "both": {None, "state"},
        "mla": {None},
        "state": {"state"},
    }[offered]
    connector._side_channel_port = 1
    _run_listener(monkeypatch, mc, connector._write_listener, credits)
    for credits_of, landing in (
        (connector._landing_credits, request.get("mla_landing")),
        (connector._state_landing_credits, request.get("state_landing")),
    ):
        if landing is None:
            continue
        held = credits_of._targets["consumer:1234"]
        assert held.in_flight == set()
        assert sorted(held.free) == sorted(landing["slots"])


@pytest.mark.parametrize("spent", [None, "mla", "state"])
def test_kimi_decode_offers_its_landings_and_completes_after_both(monkeypatch, spent):
    """`start_load_kv` gives each prefill stage a partition of every pool with
    slots left and tracks the request in each pool it offered. A request in
    both completes once, after the second receiver settles."""
    msgpack = pytest.importorskip("msgpack")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.staging_config import landing_spec
    from atom.kv_transfer.disaggregation.state_gather import KDA_SLOT_LAYOUT
    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    monkeypatch.delenv("ATOM_PD_MLA_LANDING", raising=False)
    monkeypatch.delenv("ATOM_GLM5_KPOOL", raising=False)
    nic = _PtrNic()
    mla = _kimi_receiver(
        nic,
        bases=_K_DST_MLA + _K_DST_IDX,
        widths=[_K_MLA_PAGE] * 3 + [_K_IDX_PAGE] * 2,
        regions=[0, 1, 2],
        slot_bytes=2 * _K_MLA_PAGE,
    )
    state = _kimi_receiver(
        nic,
        bases=_K_DST_SLOT,
        widths=[_CONV_BPS // 2] * 3 + [_SSM_BPS // 2] * 3 + [_CURSOR_BPS],
        regions=list(range(7)),
        slot_bytes=64,
        credit_pool="state",
    )
    pools = {"mla": mla, "state": state}
    if spent is not None:
        # An earlier prefill instance took every slot of that pool.
        pools[spent]._unassigned.clear()

    consumer = object.__new__(mc.MooncakeConnector)
    consumer.is_producer = False
    consumer.tp_size, consumer.tp_rank, consumer.dp_rank = 1, 0, 0
    consumer.dcp_size, consumer.dcp_rank, consumer.dcp_interleave_size = 2, 1, 1
    consumer.local_ip, consumer.rpc_port, consumer.ib_device = "decode", 1, None
    consumer._notification_port = 9
    consumer._num_local_layers = 3
    consumer._block_regions = list(
        zip(_K_DST_MLA + _K_DST_IDX, [_K_MLA_PAGE] * 3 + [_K_IDX_PAGE] * 2)
    )
    consumer._block_region_roles = [MLA_KV_ROLE] * 3 + [INDEX_CACHE_ROLE] * 2
    consumer._unit_layout = None
    consumer._has_slot_regions = True
    consumer._slot_regions = [(base, 64) for base in _K_DST_SLOT]
    consumer._swa_block_regions = []
    consumer._staging_pool_size = consumer._staging_slot_bytes = 0
    consumer._fp4_index_layout = False
    consumer._landing_spec = landing_spec(
        SimpleNamespace(
            kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_consumer"},
            decode_context_parallel_size=2,
            kv_cache_block_size=_K_BLOCK,
            hf_config=SimpleNamespace(model_type="kimi_linear", kv_lora_rank=512),
        )
    )
    assert consumer._landing_spec is not None
    consumer._landing, consumer._state_landing = mla, state
    consumer._completion_lock = threading.Lock()
    consumer._landing_gates = {}
    consumer._pending_handoffs = {}
    consumer._pending_recv_expected = {}
    consumer._pending_recv_nonce = {}
    consumer._pending_recv_stall_watch = {}
    consumer._pending_recv = set()
    consumer._pending_recv_blocks = {}
    consumer._pending_recv_slots = {}
    consumer._release_targets = {}
    queued = []
    consumer._enqueue_write_requests = queued.extend
    done = []
    consumer._complete_recv = lambda req, failed: done.append((req, failed))
    for recv in (mla, state):
        recv._finish = consumer._landing_finished

    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(
        "req-1",
        list(_K_DST_BLOCKS),
        {
            "remote_block_ids": list(_K_SRC_BLOCKS),
            "remote_host": "prefill",
            "remote_handshake_port": 6301,
            "remote_pp_size": 2,
            "tp_size": 1,
            "transfer_id": 7,
            "local_slot_index": 3,
            "remote_slot_index": 5,
            "src_block_skip_factor": 2,
        },
    )
    mc.MooncakeConnector.start_load_kv(consumer, meta)

    assert [parts[0] for _addr, parts in queued] == [mc.MSG_WRITE_REQUEST] * 2
    requests = [msgpack.loads(parts[1]) for _addr, parts in queued]
    offered = [name for name in pools if name != spent]
    for stage, request in enumerate(requests):
        assert request["has_slot_regions"] and request["dst_slot_index"] == 3
        # Eight slots per pool over two stages.
        share = list(range(4 * stage, 4 * stage + 4))
        for name, key in (("mla", "mla_landing"), ("state", "state_landing")):
            if name not in offered:
                assert key not in request
                continue
            assert request[key]["slots"] == share
            assert request[key]["epoch"] == pools[name].epoch
        assert "layout" not in request.get("mla_landing", {})
        if "state" in offered:
            assert request["state_landing"]["layout"] == KDA_SLOT_LAYOUT
    gate = {"left": 2, "failed": False}
    assert consumer._landing_gates == ({"req-1": gate} if spent is None else {})

    # Neither stage landed a slot: each write-done ends its stage in every
    # receiver that tracks the request, and the last one completes it.
    nonce = requests[0]["write_nonce"]
    for stage in range(2):
        assert done == []
        mc.MooncakeConnector._record_write_done(
            consumer, "req-1", stage, 0, nonce, landed_slots=[], state_landed_slots=[]
        )
    assert done == [("req-1", False)]
    assert consumer._landing_gates == {}


def test_landing_credits_go_back_to_the_pool_that_lent_them(monkeypatch):
    msgpack = pytest.importorskip("msgpack")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.mooncake.landing import LandingCredits

    connector = object.__new__(mc.MooncakeConnector)
    connector._side_channel_port = 1
    connector._landing_credits = LandingCredits()
    connector._state_landing_credits = LandingCredits()
    # One decode rank lends both pools under one key, each with its own epoch.
    connector._landing_credits.sync("d:1", {"epoch": 7, "slots": [0]})
    connector._state_landing_credits.sync("d:1", {"epoch": 9, "slots": [0]})
    assert connector._landing_credits.acquire("d:1", 7, 0) == 0
    assert connector._state_landing_credits.acquire("d:1", 9, 0) == 0

    def credit(epoch, pool=None):
        body = {"consumer": "d:1", "epoch": epoch, "slots": [0]}
        if pool is not None:
            body["pool"] = pool
        return (mc.MSG_LANDING_CREDIT, msgpack.dumps(body))

    mla = connector._landing_credits._targets["d:1"]
    state = connector._state_landing_credits._targets["d:1"]
    # Misrouted: the state pool's credit without its tag, and the reverse.
    _run_listener(
        monkeypatch,
        mc,
        connector._write_listener,
        [credit(9), credit(7, "state")],
    )
    assert mla.in_flight == {0} and state.in_flight == {0}
    _run_listener(monkeypatch, mc, connector._write_listener, [credit(7)])
    assert mla.free == [0] and state.in_flight == {0}
    _run_listener(monkeypatch, mc, connector._write_listener, [credit(9, "state")])
    assert state.free == [0]


def test_kimi_mla_rows_needing_one_landing_slot_stay_staged(_no_cuda_stream):
    """Below LANDING_MIN_SLOTS the rows go staged: no credit, no READY, and the
    write-done lists none landed rather than "cannot land"."""
    msgpack = pytest.importorskip("msgpack")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    oracle = _kimi_oracle(mc, 1)
    nic = _PtrNic()
    connector = _kimi_stage(mc, nic, slot_bytes=1024)
    sent = []
    connector._send_on_socket = lambda _addr, parts, repeat=1: sent.append(parts)
    request = _kimi_request(1)
    request["mla_landing"] = {"epoch": 3, "base": 1 << 40, "slot_bytes": 1024}
    request["mla_landing"]["slots"] = [0, 1, 2, 3]
    connector._landing_credits.sync("consumer:1234", request["mla_landing"])

    assert _kimi_transfer(mc, connector, request)
    mc.MooncakeConnector._notify_transfer_result(connector, request, success=True)

    assert (nic.compared() == oracle.compared()).all()
    assert nic.descriptors["staged-kv"] == 4 and "landed-kv" not in nic.descriptors
    [(kind, payload)] = sent
    assert kind == mc.MSG_WRITE_DONE
    assert msgpack.loads(payload)["landed_slots"] == []
    assert sorted(connector._landing_credits._targets["consumer:1234"].free) == [
        0,
        1,
        2,
        3,
    ]


def test_a_failed_landed_kimi_mla_write_quarantines_only_its_credit(
    _no_cuda_stream,
):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    nic = _PtrNic()
    connector = _kimi_stage(mc, nic, slot_bytes=2 * _K_MLA_PAGE)
    _arm_state_staging(nic, connector)
    sent = []
    connector._send_on_socket = lambda _addr, parts, repeat=1: sent.append(parts)
    connector._rdma_write = lambda *args, **kwargs: (
        args[5] != "landed-kv" and nic.write(*args, **kwargs)
    )
    request = _kimi_request(0)
    request["mla_landing"] = {
        "epoch": 3,
        "base": 1 << 40,
        "slot_bytes": 2 * _K_MLA_PAGE,
        "slots": [0, 1, 2, 3],
    }
    connector._landing_credits.sync("consumer:1234", request["mla_landing"])
    connector._state_landing_credits.sync("consumer:1234", {"epoch": 4, "slots": [0]})

    assert not _kimi_transfer(mc, connector, request)

    held = connector._landing_credits._targets["consumer:1234"]
    assert len(held.quarantined) == 1 and not held.in_flight
    assert len(held.free) == 3
    assert not request.get("_landed_slots")
    assert sent == []  # no READY for a slot that may hold a partial write
    # The state never started, so its pool lent nothing.
    assert connector._state_landing_credits._targets["consumer:1234"].free == [0]
    assert sorted(connector._kv_staging_slots.free) == [0, 1]


def test_kimi_mla_rows_to_a_non_dcp_decode_go_whole(_no_cuda_stream):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    dst_blocks = (6, 2, 3, 7, 0)
    oracle = _kimi_oracle(mc, 0, dcp=1, dst_blocks=dst_blocks)
    nic = _PtrNic()
    connector = _kimi_stage(mc, nic, slot_bytes=1024)

    assert _kimi_transfer(mc, connector, _kimi_request(0, dcp=1), dst_blocks)

    assert (nic.compared() == oracle.compared()).all()
    # Three regions x five whole pages, nothing staged or per token.
    assert nic.descriptors == {"block": 15, "slot": oracle.descriptors["slot"]}
    connector._send_worker_stream().wait_event.assert_not_called()


def test_kimi_mla_rows_without_a_ready_event_go_per_token(_no_cuda_stream):
    """The gather could read KV the forward is still writing. A stage without
    index layers (which need the event themselves) keeps the per-token path."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )

    def stage(nic, staged):
        connector = _pp_stage_connector(mc, nic)
        connector._send_worker_stream = MagicMock()
        if staged:
            connector._kv_staging = nic.tensor(_K_KV_STAGING, _K_KV_STAGING + 64)
            connector._kv_staging = connector._kv_staging.view(1, 64)
            connector._kv_staging_slots = mc.SlotPool(1)
            connector._kv_source_views = {0: nic.tensor(_MLA_SRC, _MLA_SRC + 4 * _BPB)}
        return connector

    def run(connector, ready):
        return mc.MooncakeConnector._execute_block_slot_transfer(
            connector,
            _dcp_tp_request(),
            "consumer:1234",
            [0, 1, 2, 3],
            [1, 2],
            {"slot_index": 5},
            "req-1",
            kv_cache_ready_event=ready,
        )

    oracle = _PtrNic()
    assert run(stage(oracle, False), None)
    for ready, label in ((None, "block"), (object(), "staged-kv")):
        nic = _PtrNic()
        assert run(stage(nic, True), ready)
        assert (nic.mem[:16384] == oracle.mem[:16384]).all()
        assert set(nic.descriptors) == {label, "slot"}
    # Eight tokens per descriptor before; destination pages 1-2 in one now.
    assert oracle.descriptors["block"] == 8 and nic.descriptors["staged-kv"] == 1


def test_only_kimi_slot_backends_build_mla_staging_and_landing(monkeypatch):
    """The pools cover the MLA rows alone; any other slot backend keeps none."""
    torch = pytest.importorskip("torch")
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from atom.kv_transfer.disaggregation.staging_config import landing_spec
    from atom.kv_transfer.disaggregation.types import (
        INDEX_CACHE_ROLE,
        MLA_KV_ROLE,
        KVTransferRegion,
    )

    monkeypatch.delenv("ATOM_PD_MLA_LANDING", raising=False)
    monkeypatch.delenv("ATOM_GLM5_KPOOL", raising=False)

    class _DeviceView:
        # What a GPU PAGE view reports, backed by host memory here.
        is_cuda = True
        dtype = torch.uint8
        device = torch.device("cpu")

        def is_contiguous(self):
            return True

    roles = [MLA_KV_ROLE, MLA_KV_ROLE, INDEX_CACHE_ROLE]
    tt = SimpleNamespace(
        block_regions=[
            KVTransferRegion(4096 * i, 8 * 9216, 9216, semantic_role=role)
            for i, role in enumerate(roles)
        ],
        block_tensor_views=[_DeviceView() for _ in roles],
    )
    decode = SimpleNamespace(
        kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_consumer"},
        decode_context_parallel_size=8,
        kv_cache_block_size=16,
        hf_config=SimpleNamespace(model_type="kimi_linear", kv_lora_rank=512),
    )
    for kimi in (True, False):
        conn = object.__new__(mc.MooncakeConnector)
        conn._has_slot_regions = True
        conn._slot_mla_rows = kimi
        conn._page_staging_enabled = False
        conn._num_send_workers = 2
        conn._kv_staging_reserve = 2 * (8 << 20)
        staging = conn._build_kv_staging(tt)
        assert (staging is not None) is kimi
        assert (conn._kv_staging is not None) is kimi
        if kimi:
            # The index layer keeps its own staging pool.
            assert list(conn._kv_source_views) == [0, 1]

        conn._landing_spec = landing_spec(decode)
        conn._block_region_roles = roles
        conn.kv_caches_base_addr = [r.base_addr for r in tt.block_regions]
        conn._per_block_bytes_list = [9216] * 3
        conn._cuda_device = None
        conn.local_ip, conn.rpc_port = "d", 1
        conn._send_on_socket = lambda *_a: None
        landing = conn._build_landing()
        assert (landing is not None) is kimi
        if kimi:
            assert landing._region_enabled.tolist() == [True, True, False]


@pytest.mark.parametrize(
    "producer_tp, consumer_tp, consumer_rank, ssm_segments, match",
    [
        # Consumer rank 2 of 4 holds heads [H/2, 3H/4), none of producer rank
        # 0's [0, H/2): the `tp_rank % remote_tp_size` pairing, which is right
        # for replicated KV and wrong for sharded state.
        (2, 4, 2, True, "holds no part"),
        (2, 3, 1, True, "not a multiple"),
        (1, 2, 1, False, "cannot split"),
    ],
)
def test_a_tp_split_the_state_cannot_take_is_refused(
    producer_tp, consumer_tp, consumer_rank, ssm_segments, match
):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    nic = _ByteNic()
    nic.mem[_MLA_SRC : _MLA_SRC + 4 * _BPB] = 1
    connector = _pp_stage_connector(
        mc, nic, tp_size=producer_tp, ssm_segments=ssm_segments
    )
    request = _dcp_tp_request(consumer_tp, consumer_rank, consumer_tp // producer_tp)

    with pytest.raises(RuntimeError, match=match):
        _run_slot_transfer(mc, connector, request)
    # Refused before Phase 1: no MLA page reached the peer.
    assert not nic.mem[_DST_MLA[0] : _DST_SLOT[0]].any()


@pytest.fixture
def _no_cuda_stream(monkeypatch):
    """The staging stream is a mock: entering it is a no-op here."""
    from contextlib import nullcontext

    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())


def _sparse_pp_stage(mc, index_src=1 << 20):
    """A PP stage of a sparse hybrid (Kimi-K3.1): one MLA and one index region
    at a 16-token block, the peer's second MLA row and second index layer."""
    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    writes = []
    connector = _pp_stage_connector(mc, _ByteNic())
    connector.block_size = 16
    connector._block_regions = [(0, 16 * 576), (index_src, 16 * 144)]
    connector._per_block_bytes_list = [16 * 576, 16 * 144]
    connector._block_region_roles = [MLA_KV_ROLE, INDEX_CACHE_ROLE]
    connector._block_head_major_runs = [None, None]
    connector._block_region_consumer_indices = [1, 3]
    connector._index_staging_chunk_pages = 256
    connector._prepare_sharded_index = MagicMock(return_value="gather")
    connector._gather_sharded_index = MagicMock()
    connector._send_worker_stream = MagicMock()
    connector._execute_staged_index_layer_chunk = MagicMock(return_value=True)
    connector._rdma_write = lambda _t, src, dst, sizes, *_a, **_k: (
        writes.extend(zip(src, dst, sizes)) or True
    )
    request = _dcp_tp_request()
    request["consumer_block_base_addrs"] = [1 << 22, 2 << 22, 3 << 22, 4 << 22]
    request["consumer_block_bpb"] = [16 * 576, 16 * 576, 16 * 144, 16 * 144]
    request["consumer_region_roles"] = [MLA_KV_ROLE] * 2 + [INDEX_CACHE_ROLE] * 2
    return connector, request, writes


def test_slot_transfer_gathers_index_pages_onto_the_dcp_plan(_no_cuda_stream):
    """A stateful model's PP stage sends through the slot path: its MLA tokens
    relayout per token, its preshuffled index pages, which split below one
    16-row tile, go through the staged gather once the prefill is ready."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    index_src = 1 << 20
    connector, request, writes = _sparse_pp_stage(mc, index_src)
    ready = object()
    # The write's deadline bounds the staged gather too.
    request["_deadline"] = deadline = time.monotonic() + 3600

    assert mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        request,
        "consumer:1234",
        [0, 1, 2, 3],
        [1, 2],
        {"slot_index": 5},
        "req-1",
        kv_cache_ready_event=ready,
    )

    connector._send_worker_stream().wait_event.assert_called_once_with(ready)
    connector._execute_staged_index_layer_chunk.assert_called_once_with(
        "consumer:1234",
        1,
        4 << 22,
        16 * 144,
        [1, 2],
        "req-1",
        "gather",
        engine=None,
        deadline=deadline,
    )
    # DCP rank 1 of 2 owns the 32 odd tokens of the 64-token prompt.
    mla = [w for w in writes if 2 << 22 <= w[1] < 3 << 22]
    assert len(mla) == 32 and all(size == 576 for _, _, size in mla)
    assert not any(index_src <= src < 2 * index_src for src, _, _ in writes)


def test_index_regions_without_staging_name_the_backend_opt_out():
    """A pooled index opts out of staging; the refusal says so, not misconfig."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector, _, _ = _sparse_pp_stage(mc)
    connector._prepare_sharded_index = connector._gather_sharded_index = None

    with pytest.raises(RuntimeError, match=r"index regions \[1\].*no index staging"):
        connector._stages_sharded_index(1)


def test_slot_transfer_refuses_index_staging_without_a_ready_event(_no_cuda_stream):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    connector, request, _ = _sparse_pp_stage(mc)

    with pytest.raises(RuntimeError, match="ready event"):
        _run_slot_transfer(mc, connector, request)
    connector._execute_staged_index_layer_chunk.assert_not_called()


def test_dcp_index_staging_waits_for_request_ready_event(monkeypatch):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    from contextlib import contextmanager

    from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

    current = []

    @contextmanager
    def on_stream(stream):
        current.append(stream)
        yield
        current.pop()

    monkeypatch.setattr(mc.torch.cuda, "stream", on_stream)

    connector = object.__new__(mc.MooncakeConnector)
    connector.dcp_size = 1
    connector.block_size = 16
    connector.kv_caches_base_addr = [1_000_000, 2_000_000]
    mla_block_bytes = connector.block_size * 576
    index_block_bytes = connector.block_size * 144
    connector._per_block_bytes_list = [mla_block_bytes, index_block_bytes]
    connector._block_region_roles = [MLA_KV_ROLE, INDEX_CACHE_ROLE]
    connector._block_region_consumer_indices = None
    connector._block_unit_specs = None
    connector._block_head_major_runs = ()
    connector._unit_layout = None
    connector._consumer_region_map = lambda *_args, **_kwargs: [0, 1]
    connector._index_staging_chunk_pages = 256
    gather_indices = object()
    prepared_on = []
    connector._prepare_sharded_index = lambda _plan: (
        prepared_on.append(list(current)) or gather_indices
    )
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
    # The gathers read these indices on the staging stream, so it builds them.
    assert prepared_on == [[staging_stream]]
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
    connector._block_unit_specs = None
    connector._block_head_major_runs = ()
    connector._unit_layout = None
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

    monkeypatch.setattr(
        mooncake_hca, "ib_device_exists", lambda name: name.startswith("ionic_")
    )
    with pytest.raises(ValueError, match=message):
        mc._validate_matched_rails(protocol, devices, rails)


def test_matched_rails_disabled_preserves_tcp_and_multi_hca(monkeypatch):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    mc._validate_matched_rails("tcp", [], [])
    mc._validate_matched_rails("rdma", ["ionic_0", "ionic_1"], [])
    monkeypatch.setattr(mooncake_hca, "ib_device_exists", lambda _: True)
    mc._validate_matched_rails("rdma", ["ionic_2"], ["ionic_2", "ionic_6"])


@pytest.fixture
def matched_rail_sysfs(tmp_path, monkeypatch):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    monkeypatch.setattr(mooncake_hca, "IB_SYSFS_ROOT", tmp_path)

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
    mooncake_hca.IB_SYSFS_ROOT = tmp_path / "not-mounted"
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
    mooncake_hca.IB_SYSFS_ROOT = mooncake_hca.IB_SYSFS_ROOT / "not-mounted"
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

    def transfer(data, target, *_args, engine=None, **_kwargs):
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
