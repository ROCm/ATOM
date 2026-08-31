# SPDX-License-Identifier: MIT
# PD-disaggregation + pipeline-parallel unit tests (GPU-free).

import logging
import os
import sys
import threading
import types
from collections import deque
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
        state_slots=[],
        kv_transfer_params_output=None,
    )
    mc.MooncakeConnectorScheduler.request_finished(sched, seq)
    assert seq.kv_transfer_params_output["remote_pp_size"] == 4
    assert seq.kv_transfer_params_output["hash_block_size"] == 64
    assert seq.kv_transfer_params_output["remote_block_ids"] == [1, 2, 3]


def _mooncake_consumer_scheduler(mc, hash_block_size=64):
    sched = object.__new__(mc.MooncakeConnectorScheduler)
    sched.is_producer = False
    sched.hash_block_size = hash_block_size
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


def test_matching_hash_block_size_enables_incremental_transfer():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    sched = _mooncake_consumer_scheduler(mc)
    seq = _remote_prefill_seq(remote_hash_block_size=64)

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 2


@pytest.mark.parametrize("remote_hash_block_size", [32, None])
def test_mismatched_or_missing_hash_block_size_forces_full_transfer(
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
    assert "falling back to full transfer" in caplog.text


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
    )
    head.scheduler.schedule.side_effect = [(batch, {}), None]
    head.scheduler.take_rejected.return_value = None
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


def test_pp_head_forwards_normal_batch_with_meta():
    """A batch with requests keeps the original path: dispatch, send, forward."""
    meta = _FakeMeta(["load-r1"])
    batch = _fake_batch([7], meta)
    batch.produces_output = lambda: False
    head = _fake_head(batch)

    assert _dispatched_metas(head) == [meta]
    head.pp_transport.send_metadata.assert_called_once_with(batch)
    head.runner_mgr.call_func.assert_any_call("forward", batch, wait_out=True)


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
