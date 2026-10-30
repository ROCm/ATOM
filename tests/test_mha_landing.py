# SPDX-License-Identifier: MIT
"""Byte-exact PP/TP1 -> TP PAGE gathering, landing, and chunk completion."""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import test_m3_kv_transfer_layout as layout
import test_pd_chunked_transfer as protocol
import torch

from atom.kv_transfer.disaggregation.mooncake.landing import (
    LANDING_POOL_SLOTS,
    LANDING_SLOT_BYTES,
    LandingCredits,
    LandingReceiver,
)
from atom.kv_transfer.disaggregation.staging_config import (
    MHA_SCATTER_SEGMENTS,
    LandingSpec,
    mha_landing_reserve_bytes,
    mha_staging_reserve_bytes,
)
from atom.kv_transfer.disaggregation.transfer_plan import PAGE_UNITS

mc = protocol.mc


class Stream:
    def wait_event(self, event):
        event.synchronize()

    def synchronize(self):
        pass


class Harness:
    def __init__(self, held, rank, *, offer=True, pump=True, device="cpu"):
        self.tt, self.source = layout._layout(held, 4, draft=7 in held)
        layout._fill(self.source)
        self.consumer, self.destination = layout._layout(range(8), 1, draft=True)
        layout._fill(self.destination, zeros=True)
        c, self.req = layout._producer(self.tt, 4)
        self.c = c
        layout._request(self.req, self.consumer, rank)
        self.req.update(dst_block_ids=[2, 0, 3], chunked_transfer=False)
        self.rank = rank
        self.device = torch.device(device)
        self.src_views = [p.view.to(device) for p in self.tt.pages]
        self.dst_views = [p.view.to(device) for p in self.consumer.pages]
        c._kv_staging = torch.empty((1, 32768), dtype=torch.uint8, device=device)
        c._kv_staging_slots = mc.SlotPool(1)
        c._kv_source_views = dict(enumerate(self.src_views))
        c._landing_credits = LandingCredits()
        c.kv_caches_base_addr = [v.data_ptr() for v in self.src_views]
        self.req["consumer_base_addrs"] = [v.data_ptr() for v in self.dst_views]
        self.stream = torch.cuda.Stream() if self.device.type == "cuda" else Stream()
        c._send_worker_stream = lambda: self.stream
        self.event = (
            torch.cuda.Event()
            if self.device.type == "cuda"
            else SimpleNamespace(synchronize=lambda: None)
        )
        if self.device.type == "cuda":
            self.event.record()
        self.finished = []
        self.ready = []
        self.writes = []
        self.recv = LandingReceiver(
            device=0 if self.device.type == "cuda" else None,
            pool_slots=2,
            slot_bytes=32768,
            units_per_page=1,
            consumer_key="cpu:1",
            region_bases=self.req["consumer_base_addrs"],
            region_block_bytes=self.req["consumer_block_bpb"],
            region_indices=list(range(len(self.consumer.pages))),
            send=lambda addr, parts: c._landing_credits.release(
                "cpu:1", self.recv.epoch, msgpack.loads(parts[-1])["slots"]
            ),
            finish=lambda req, failed: self.finished.append((req, failed)),
            table_segments=16,
        )
        if self.device.type == "cpu":

            def copy(src, dst, nbytes, events):
                for s, d, n in zip(src, dst, nbytes):
                    ctypes.memmove(
                        self.recv._dst_origin + int(d),
                        self.recv.pool.data_ptr() + int(s),
                        int(n),
                    )

            self.recv._copy_segments = copy
        advertised = self.recv.advertise("s0", 1)
        if offer:
            self.req["mla_landing"] = {**advertised, "layout": PAGE_UNITS}
        self.recv.begin(21, 3, self.req["dst_block_ids"], {0: "s0"})

        def notify(addr, parts):
            data = msgpack.loads(parts[-1])
            self.ready.append(data)
            self.recv.on_ready(data)
            if pump:
                self.pump()

        c._send_on_socket = notify
        destinations = [self.recv.pool, *self.dst_views]

        def write(target, srcs, dsts, sizes):
            self.writes.append((srcs, dsts, sizes))
            for src, dst, n in zip(srcs, dsts, sizes):
                if self.device.type == "cpu":
                    ctypes.memmove(dst, src, n)
                else:
                    source = c._kv_staging.view(-1)[
                        src
                        - c._kv_staging.data_ptr() : src
                        - c._kv_staging.data_ptr()
                        + n
                    ]
                    owner = next(
                        v
                        for v in destinations
                        if v.data_ptr() <= dst and dst + n <= v.data_ptr() + v.numel()
                    )
                    owner.view(-1)[
                        dst - owner.data_ptr() : dst - owner.data_ptr() + n
                    ].copy_(source)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return 0

        c.transfer_engine = SimpleNamespace(batch_transfer_sync_write=write)

    def pump(self):
        tasks = []
        while not self.recv._queue.empty():
            tasks.append(self.recv._queue.get())
        self.recv._process(tasks)

    def send(self, src, dst, offset=0):
        self.req["_transfer_dst_offset"] = offset
        return self.c._execute_block_transfer(
            self.req, "cpu:1", src, dst, 21, self.event
        )

    def check(self, src, dst):
        for page, gpu in zip(self.consumer.pages, self.dst_views):
            if gpu.device.type == "cuda":
                page.view.copy_(gpu.cpu())
        layout._assert_copy(self.source, self.destination, 4, self.rank, src, dst)
        untouched = list(set(range(4)) - set(dst))
        assert all(not v[untouched].any() for v in self.destination.values())


@pytest.mark.parametrize("held", layout.STAGES)
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("offer", [False, True])
def test_full_transfer_matches_physical_nvfp4_fp4_and_draft(held, rank, offer):
    h = Harness(held, rank, offer=offer)
    assert h.send([3, 1, 2], [2, 0, 3])
    h.check([3, 1, 2], [2, 0, 3])
    if offer:
        assert h.ready
        assert all(len(src) == 1 for src, _, _ in h.writes)
        assert [x["seq"] for x in h.ready] == list(range(len(h.ready)))
        h.recv.stage_done(21, 0, 3, True, h.req["_landed_slots"])
        assert h.finished == [(21, False)]


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("cached", [0, 1])
def test_chunks_use_receive_global_page_offsets_and_sequences(rank, cached):
    h = Harness([6, 7], rank)
    ids = [2, 0, 3][cached:]
    h.req["dst_block_ids"] = ids
    h.recv.begin(21, 3, ids, {0: "s0"})
    for offset in range(len(ids)):
        assert h.send([[3, 1, 2][cached + offset]], [ids[offset]], offset)
        h.check([3, 1, 2][cached : cached + offset + 1], ids[: offset + 1])
        assert h.finished == []
    assert [x["seq"] for x in h.ready] == list(range(len(h.ready)))
    assert {row[1] for x in h.ready for row in x["items"]} == set(range(len(ids)))
    h.recv.stage_done(21, 0, 3, True, h.req["_landed_slots"])
    assert h.finished == [(21, False)]


def test_credit_exhaustion_gathers_remaining_pages_to_direct_destinations():
    h = Harness([6, 7], 2, pump=False)
    assert h.send([3, 1, 2], [2, 0, 3])
    assert len(h.ready) == 2
    assert any(len(src) > 1 for src, _, _ in h.writes)
    h.recv.stage_done(21, 0, 3, True, h.req["_landed_slots"])
    assert h.finished == []
    h.pump()
    h.check([3, 1, 2], [2, 0, 3])
    assert h.finished == [(21, False)]


def test_bad_late_region_fails_before_any_rdma():
    h = Harness([6, 7], 0)
    cidx = h.tt.block_region_consumer_indices[-1]
    h.req["consumer_region_roles"][cidx] = "wrong"
    with pytest.raises(ValueError, match="role mismatch"):
        h.send([3, 1, 2], [2, 0, 3])
    assert h.writes == []
    assert h.ready == []


def test_failed_nic_quarantines_landing_and_retains_source_staging(monkeypatch):
    h = Harness([6, 7], 0)
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "60")
    h.c.transfer_engine.batch_transfer_sync_write = lambda *args: -1
    assert not h.send([3, 1, 2], [2, 0, 3])
    assert h.c._kv_staging_failed
    assert h.c._kv_staging_slots.free == []
    with pytest.raises(RuntimeError, match="staging row unavailable"):
        h.c._kv_staging_slots.acquire()
    assert len(h.c._landing_credits._targets["cpu:1"].quarantined) == 1
    assert h.ready == []


def test_gather_failure_synchronizes_before_releasing_slot(monkeypatch):
    h = Harness([6, 7], 0)
    sync = MagicMock()
    h.stream.synchronize = sync

    def fail(*args):
        assert h.c._kv_staging_slots.free == []
        raise RuntimeError("gather failed")

    monkeypatch.setattr(mc, "gather_units", fail)
    with pytest.raises(RuntimeError, match="gather failed"):
        h.send([3, 1, 2], [2, 0, 3])
    assert sync.call_count >= 1
    assert h.c._kv_staging_slots.free == [0]
    assert not h.c._landing_credits._targets["cpu:1"].in_flight
    assert h.writes == []


@pytest.mark.parametrize("bad", [None, {"first_token_id": 44, "draft_token_ids": []}])
def test_chunk_handoff_is_checked_before_landing_completion(bad):
    h = Harness([6, 7], 0)
    c = h.c
    c._landing = h.recv
    c._landing_spec = LandingSpec(1, MHA_SCATTER_SEGMENTS, PAGE_UNITS, True)
    c._pending_handoffs = {
        21: {
            "failed": False,
            "metadata": None,
            "landing_nonce": 3,
            "expected": 2,
            "stages": set(),
        }
    }
    h.recv.advertise("s1", 2)
    h.recv.begin(21, 3, [2, 0, 3], {0: "s0", 1: "s1"})
    good = {"first_token_id": 42, "draft_token_ids": []}
    c._record_write_done(21, 0, write_nonce=3, handoff=good, landed_slots=[])
    assert h.finished == []
    # Wrong nonce and duplicate must not poison metadata or finish another stage.
    c._record_write_done(21, 1, write_nonce=9, handoff=bad, landed_slots=[])
    c._record_write_done(21, 0, write_nonce=3, handoff=bad, landed_slots=[])
    assert not c._pending_handoffs[21]["failed"]
    c._record_write_done(21, 1, write_nonce=3, handoff=bad, landed_slots=[])
    assert h.finished == [(21, True)]


@pytest.mark.parametrize("workers", [1, 8, 64])
def test_reserves_are_gated_and_bounded(monkeypatch, workers):
    monkeypatch.setenv("ATOM_PD_MHA_LANDING", "1")
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="minimax_m3"),
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        kv_transfer_config={"kv_role": "kv_producer", "num_worker_threads": workers},
    )
    assert mha_staging_reserve_bytes(cfg) == min(workers, 32) * LANDING_SLOT_BYTES
    assert mha_landing_reserve_bytes(cfg) == 0
    cfg.tensor_parallel_size = 4
    cfg.kv_transfer_config["kv_role"] = "kv_consumer"
    assert mha_staging_reserve_bytes(cfg) == 0
    assert (
        mha_landing_reserve_bytes(cfg)
        == LANDING_POOL_SLOTS * LANDING_SLOT_BYTES + 3 * 8 * MHA_SCATTER_SEGMENTS
    )
    for key, val in [
        ("decode_context_parallel_size", 4),
        ("hf_config", SimpleNamespace(model_type="glm_moe_dsa")),
    ]:
        old = getattr(cfg, key)
        setattr(cfg, key, val)
        assert mha_staging_reserve_bytes(cfg) == mha_landing_reserve_bytes(cfg) == 0
        setattr(cfg, key, old)
    monkeypatch.setenv("ATOM_PD_MHA_LANDING", "0")
    assert mha_staging_reserve_bytes(cfg) == mha_landing_reserve_bytes(cfg) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU gather/scatter")
@pytest.mark.parametrize("rank", range(4))
def test_gpu_gather_rdma_landing_scatter_real_nvfp4_index_draft(rank):
    h = Harness([6, 7], rank, device="cuda")
    for offset in range(3):
        assert h.send([[3, 1, 2][offset]], [[2, 0, 3][offset]], offset)
    h.recv.stage_done(21, 0, 3, True, h.req["_landed_slots"])
    assert h.finished == [(21, False)]
    h.check([3, 1, 2], [2, 0, 3])
    assert len(h.ready) > h.recv._pool_slots  # NIC rewrites the same slots.


def test_off_path_records_ready_event_with_only_kv_staging(monkeypatch):
    h = Harness([6, 7], 0)
    h.c._index_staging_pool_size = 0
    event = MagicMock()
    stream = object()
    monkeypatch.setattr(mc.torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(mc.torch.cuda, "current_stream", lambda device: stream)
    h.c.record_kv_cache_ready([99])
    assert h.c._kv_cache_ready_events[99] is event
    event.record.assert_called_once_with(stream)


@pytest.mark.parametrize("chunked", [False, True])
def test_real_consumer_offers_page_landing_and_waits_for_scatter(monkeypatch, chunked):
    h = Harness([6, 7], 0, pump=False)
    dummy = (ctypes.c_ubyte * 48)()
    d = protocol.decode_worker(dummy)
    d.tp_size = 4
    d.block_size = 128
    d._num_local_layers = 8
    d._block_region_roles = h.req["consumer_region_roles"]
    d._block_head_major_runs = h.req["consumer_head_major_runs"]
    d._block_regions = list(
        zip(h.req["consumer_base_addrs"], h.req["consumer_block_bpb"])
    )
    d.kv_caches_base_addr = h.req["consumer_base_addrs"]
    d._landing_spec = LandingSpec(1, MHA_SCATTER_SEGMENTS, PAGE_UNITS, True)
    d._landing = h.recv
    h.recv._finish = d._complete_recv
    h.recv._partitions.clear()
    h.recv._unassigned = list(range(h.recv._pool_slots))
    d._send_release = MagicMock()
    meta = protocol.ConnectorMetadata()
    meta.add_new_req_to_recv(
        21,
        [2, 0, 3],
        {
            "chunked_transfer": chunked,
            "remote_tp_size": 1,
            "remote_pp_size": 1,
            "remote_host": "cpu",
            "remote_handshake_port": 6100,
            "remote_block_ids": [3, 1, 2],
            "transfer_id": "xfer-a",
            "num_prompt_tokens": 258,
        },
    )
    d.start_load_kv(meta)
    protocol.post_write_requests(d, monkeypatch)
    requests = [
        msgpack.loads(call.args[1][-1])
        for call in d._send_on_socket.call_args_list
        if call.args[1][0] == mc.MSG_WRITE_REQUEST
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request["mla_landing"]["layout"] == PAGE_UNITS
    h.req.update(request)
    # The fixture's copy engine and credit callbacks name this local target.
    h.req["consumer_host"] = "cpu"
    h.req["consumer_rpc_port"] = 1
    assert h.send([3, 1, 2], [2, 0, 3])
    handoff = {"first_token_id": 42, "draft_token_ids": [43, 44]} if chunked else None
    assert not d._record_write_done(
        21,
        0,
        write_nonce=request["write_nonce"],
        handoff=handoff,
        landed_slots=h.req["_landed_slots"],
    )
    assert 21 not in d.done_recving
    h.pump()
    assert 21 in d.done_recving
    if chunked:
        assert d._received_handoffs[21].first_token_id == 42
    h.check([3, 1, 2], [2, 0, 3])
    d._send_release.assert_called_once_with(21)


def test_mha_budget_is_visible_through_connector_factory(monkeypatch):
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    monkeypatch.setenv("ATOM_PD_MHA_LANDING", "1")
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="minimax_m3"),
        tensor_parallel_size=4,
        decode_context_parallel_size=1,
        kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_consumer"},
    )
    assert KVConnectorFactory.kv_budget_reserve_bytes(cfg) == mha_landing_reserve_bytes(
        cfg
    )
