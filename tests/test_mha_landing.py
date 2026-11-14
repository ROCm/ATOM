# SPDX-License-Identifier: MIT
"""Byte-exact PP/TP1 -> TP PAGE gathering, landing, and chunk completion.

Pages come from the MHA pool's unit layout on main. The NVFP4 M3 catalog
that the original recipes used is not on this branch.
"""

import ctypes
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import test_pd_chunked_transfer as protocol
import torch
from aiter_stub import stubbed_aiter
from test_page_unit_layout import BLOCK, LAYERS, MTP, stage_layers, stage_pages

from atom.kv_transfer.disaggregation.mooncake.landing import (
    LANDING_POOL_SLOTS,
    LANDING_SLOT_BYTES,
    LandingCredits,
    LandingReceiver,
)
from atom.kv_transfer.disaggregation.staging_config import (
    MHA_SCATTER_SEGMENTS,
    LandingSpec,
    landing_reserve_bytes,
    mha_staging_reserve_bytes,
)
from atom.kv_transfer.disaggregation.transfer_plan import PAGE_UNITS

mc = protocol.mc

# (pp_rank, layer partition). The last rank also holds MTP and the draft pool.
STAGES = (
    (0, (LAYERS,)),
    (0, (2, 2, 2, 2)),
    (1, (2, 2, 2, 2)),
    (3, (2, 2, 2, 2)),
    (3, (3, 2, 2, 1)),
)
SINGLE, LAST = STAGES[0], STAGES[3]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU gather/scatter")
    return request.param


def byte_view(buffers, address, size):
    owner = next(
        v
        for v in buffers
        if v.data_ptr() <= address and address + size <= v.data_ptr() + v.numel()
    )
    return owner.view(-1).narrow(0, address - owner.data_ptr(), size)


class Stream:
    def wait_event(self, event):
        event.synchronize()

    def synchronize(self):
        pass


class Harness:
    def __init__(self, stage, rank, *, offer=True, pump=True, device="cpu"):
        pp_rank, partition = stage
        layers, start, _end = stage_layers(partition, pp_rank)
        last = pp_rank == len(partition) - 1
        self.partition = partition
        self.pp_rank = pp_rank
        self.ours = stage_pages(layers, 4, draft=last)
        self.theirs = stage_pages([*range(LAYERS), MTP], 1, draft=True)
        self.rank = rank
        self.device = torch.device(device)
        self.src_views = [p.view.to(self.device) for p in self.ours]
        self.dst_views = [p.view.to(self.device) for p in self.theirs]
        for ordinal, view in enumerate(self.src_views):
            view.copy_(
                torch.arange(view.numel(), device=view.device)
                .add(ordinal * 17)
                .remainder(251)
                .to(torch.uint8)
                .view(view.shape)
            )
        for view in self.dst_views:
            view.zero_()
        c, _, self.req, _, _ = protocol.producer(
            pp_rank=pp_rank, pp_size=len(partition)
        )
        self.c = c
        c.tp_size = 1
        c.dcp_size = 1
        c.block_size = BLOCK
        c.num_hidden_layers = LAYERS
        c._start_layer = start
        c._block_region_roles = [p.region.semantic_role for p in self.ours]
        c._block_head_major_runs = tuple(p.head_major_runs for p in self.ours)
        c._block_unit_specs = tuple(p.unit_spec for p in self.ours)
        c._per_block_bytes_list = [p.region.unit_bytes for p in self.ours]
        c._unit_plans = {}
        c.kv_caches_base_addr = [v.data_ptr() for v in self.src_views]
        c._kv_staging = torch.empty((1, 32768), dtype=torch.uint8, device=self.device)
        c._kv_staging_slots = mc.SlotPool(1)
        c._kv_staging_failed = False
        c._kv_source_views = dict(enumerate(self.src_views))
        c._landing_credits = LandingCredits()
        self.req.update(
            consumer_tp_size=4,
            consumer_tp_rank=rank,
            consumer_dcp_size=1,
            consumer_base_addrs=[v.data_ptr() for v in self.dst_views],
            consumer_region_roles=[p.region.semantic_role for p in self.theirs],
            consumer_block_bpb=[p.region.unit_bytes for p in self.theirs],
            consumer_unit_layout=mc._pack_unit_layout(self.theirs),
            dst_block_ids=[2, 0, 3],
            chunked_transfer=False,
        )
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
            region_indices=list(range(len(self.theirs))),
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
        self.recv.begin(21, 3, self.req["dst_block_ids"], {pp_rank: "s0"})

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
                source = byte_view([c._kv_staging, *self.src_views], src, n)
                byte_view(destinations, dst, n).copy_(source)
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
        old = os.environ.get("VLLM_PP_LAYER_PARTITION")
        os.environ["VLLM_PP_LAYER_PARTITION"] = ",".join(map(str, self.partition))
        try:
            return self.c._execute_block_transfer(
                self.req, "cpu:1", src, dst, 21, self.event
            )
        finally:
            if old is None:
                os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
            else:
                os.environ["VLLM_PP_LAYER_PARTITION"] = old

    def check(self, src, dst):
        by_spec = {
            page.unit_spec: view for page, view in zip(self.theirs, self.dst_views)
        }
        for page, gpu in zip(self.ours, self.src_views):
            runs = page.head_major_runs
            got = by_spec[page.unit_spec]
            for s, d in zip(src, dst):
                want = gpu[s, 0]
                if runs is not None:
                    want = want.reshape(runs[0], runs[1], -1)[:, self.rank]
                assert torch.equal(
                    got[d, 0].reshape(-1), want.reshape(-1)
                ), page.unit_spec
        untouched = [i for i in range(self.dst_views[0].shape[0]) if i not in dst]
        assert all(not view[untouched].any() for view in self.dst_views)

    def owned_page_bytes(self):
        owned = {page.unit_spec for page in self.ours}
        return max(
            page.region.unit_bytes for page in self.theirs if page.unit_spec in owned
        )


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("offer", [False, True])
def test_full_transfer_matches_the_consumers_head(stage, rank, offer):
    h = Harness(stage, rank, offer=offer)
    assert h.send([3, 1, 2], [2, 0, 3])
    h.check([3, 1, 2], [2, 0, 3])
    if offer:
        assert h.ready
        assert all(len(src) == 1 for src, _, _ in h.writes)
        assert [x["seq"] for x in h.ready] == list(range(len(h.ready)))
        h.recv.stage_done(21, h.pp_rank, 3, True, h.req["_landed_slots"])
        assert h.finished == [(21, False)]


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("cached", [0, 1])
def test_chunks_use_receive_global_page_offsets_and_sequences(rank, cached, device):
    h = Harness(LAST, rank, device=device)
    ids = [2, 0, 3][cached:]
    h.req["dst_block_ids"] = ids
    h.recv.begin(21, 3, ids, {h.pp_rank: "s0"})
    for offset in range(len(ids)):
        assert h.send([[3, 1, 2][cached + offset]], [ids[offset]], offset)
        h.check([3, 1, 2][cached : cached + offset + 1], ids[: offset + 1])
        assert h.finished == []
    assert [x["seq"] for x in h.ready] == list(range(len(h.ready)))
    assert {row[1] for x in h.ready for row in x["items"]} == set(range(len(ids)))
    assert len(h.ready) > h.recv._pool_slots  # NIC rewrites the same slots.
    h.recv.stage_done(21, h.pp_rank, 3, True, h.req["_landed_slots"])
    assert h.finished == [(21, False)]


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("chunks", [False, True])
@pytest.mark.parametrize("capacity", ["source_too_small", "landing_too_small", "exact"])
def test_page_capacity_fallback_preserves_bytes_and_completion(
    rank, chunks, capacity, device
):
    h = Harness(LAST, rank, device=device)
    page_bytes = h.owned_page_bytes()
    # A full P PAGE is wider than its destination head slice. The exact-fit
    # case must still stage, even though P's PAGE would not fit in the slot.
    assert max(h.c._per_block_bytes_list) > page_bytes
    slot_bytes = page_bytes - 4 if capacity == "source_too_small" else page_bytes
    h.c._kv_staging = torch.empty((1, slot_bytes), dtype=torch.uint8, device=device)
    if capacity == "landing_too_small":
        h.req["mla_landing"]["slot_bytes"] = page_bytes - 4
    src, dst = [3, 1, 2], [2, 0, 3]
    if chunks:
        for offset in range(3):
            assert h.send(src[offset : offset + 1], dst[offset : offset + 1], offset)
            assert h.finished == []
    else:
        assert h.send(src, dst)
    h.check(src, dst)
    assert bool(h.ready) == (capacity == "exact")
    base = h.c._kv_staging.data_ptr()
    staged_sources = [
        base <= address < base + slot_bytes
        for sources, _, _ in h.writes
        for address in sources
    ]
    assert staged_sources
    assert (
        all(staged_sources)
        if capacity != "source_too_small"
        else not any(staged_sources)
    )
    assert h.c._kv_staging_slots.free == [0]
    h.recv.stage_done(21, h.pp_rank, 3, True, h.req.get("_landed_slots", []))
    assert h.finished == [(21, False)]


def test_bad_unit_layout_fails_before_any_rdma():
    h = Harness(LAST, 0)
    units = msgpack.loads(h.req["consumer_unit_layout"])
    units[-1][4] = "not-a-dtype"
    h.req["consumer_unit_layout"] = msgpack.dumps(units)
    with pytest.raises(RuntimeError, match="unit format mismatch"):
        h.send([3, 1, 2], [2, 0, 3])
    assert h.writes == []
    assert h.ready == []


@pytest.mark.parametrize("bad", [None, {"first_token_id": 44, "draft_token_ids": []}])
def test_chunk_handoff_is_checked_before_landing_completion(bad):
    h = Harness(LAST, 0)
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
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    monkeypatch.setenv("ATOM_PD_MHA_LANDING", "1")
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="minimax_m3"),
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        kv_transfer_config={
            "kv_connector": "mooncake",
            "kv_role": "kv_producer",
            "num_worker_threads": workers,
        },
    )
    assert mha_staging_reserve_bytes(cfg) == min(workers, 32) * LANDING_SLOT_BYTES
    assert landing_reserve_bytes(cfg) == 0
    cfg.tensor_parallel_size = 4
    cfg.kv_transfer_config["kv_role"] = "kv_consumer"
    assert mha_staging_reserve_bytes(cfg) == 0
    # The factory imports the connector module afresh, which binds aiter names.
    with stubbed_aiter():
        factory_reserve = KVConnectorFactory.kv_budget_reserve_bytes(cfg)
    assert (
        landing_reserve_bytes(cfg)
        == factory_reserve
        == LANDING_POOL_SLOTS * LANDING_SLOT_BYTES + 3 * 8 * MHA_SCATTER_SEGMENTS
    )
    for key, val in [
        ("decode_context_parallel_size", 4),
        ("hf_config", SimpleNamespace(model_type="glm_moe_dsa")),
    ]:
        old = getattr(cfg, key)
        setattr(cfg, key, val)
        assert mha_staging_reserve_bytes(cfg) == landing_reserve_bytes(cfg) == 0
        setattr(cfg, key, old)
    monkeypatch.setenv("ATOM_PD_MHA_LANDING", "0")
    assert mha_staging_reserve_bytes(cfg) == landing_reserve_bytes(cfg) == 0


def test_off_path_records_ready_event_with_only_kv_staging(monkeypatch):
    h = Harness(LAST, 0)
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
    h = Harness(SINGLE, 0, pump=False)
    dummy = (ctypes.c_ubyte * 48)()
    d = protocol.decode_worker(dummy)
    d.tp_size = 4
    d.block_size = BLOCK
    d._num_local_layers = LAYERS
    d._block_region_roles = [p.region.semantic_role for p in h.theirs]
    d._block_head_major_runs = tuple(p.head_major_runs for p in h.theirs)
    d._block_unit_specs = tuple(p.unit_spec for p in h.theirs)
    d._unit_layout = mc._pack_unit_layout(h.theirs)
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
    # With scatter paused, exhaust landing credits and write the rest directly.
    assert len(h.ready) == h.recv._pool_slots
    assert any(len(src) > 1 for src, _, _ in h.writes)
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
