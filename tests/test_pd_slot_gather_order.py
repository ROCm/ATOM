# SPDX-License-Identifier: MIT
"""What a P/D producer's staged state-slot gather waits for on the device.

A producer stages a request's state slot (DeepSeek-V4's compressor state) into
a staging row before the RDMA reads it, and waits for that gather on the host.
Streams here are FIFOs on one fake device: a synchronize waits for every item
on its stream and for whatever the events that stream waited on cover. The
connector's real publication, transfer and address planning run on CPU
buffers; only the NIC and CUDA are replaced.
"""

import contextlib
import ctypes
import queue
import threading
from collections import OrderedDict, deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

from atom.kv_transfer.disaggregation.types import ConnectorMetadata

SLOT_BYTES = 8


class FakeStream:
    def __init__(self, device):
        self.device = device
        self.work = []

    def launch(self, item):
        self.work.append(item)

    def wait_event(self, event):
        self.work.extend(w for w in event.covers if w not in self.work)

    def synchronize(self):
        self.device.synchronized.append(list(self.work))


class FakeDevice:
    """One GPU: the default stream every forward runs on, and per-thread
    current streams."""

    def __init__(self):
        self.default = FakeStream(self)
        self.synchronized = []
        self._current = threading.local()

    def current_stream(self, _device=None):
        return getattr(self._current, "stream", self.default)

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.current_stream()
        self._current.stream = stream
        try:
            yield
        finally:
            self._current.stream = previous


@pytest.fixture
def device(monkeypatch):
    dev = FakeDevice()

    class FakeEvent:
        covers = ()

        def record(self, stream):
            self.covers = list(stream.work)

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: FakeStream(dev))
    monkeypatch.setattr(torch.cuda, "current_stream", dev.current_stream)
    monkeypatch.setattr(torch.cuda, "stream", dev.stream)
    return dev


class MemoryEngine:
    """The NIC: copies each write between CPU buffers."""

    def __init__(self):
        self.writes = queue.Queue()

    def batch_transfer_sync_write(self, target, sources, destinations, sizes):
        for src, dst, size in zip(sources, destinations, sizes, strict=True):
            ctypes.memmove(dst, src, size)
        self.writes.put((list(sources), list(destinations), list(sizes)))
        return 0


def staged_slot_producer(device):
    """A TP-1 producer with one PAGE region and one staged state slot."""
    c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    c.is_producer = True
    c.tp_size = c.dcp_size = c.pp_size = 1
    c.tp_rank = c.pp_rank = 0
    c.block_size = 4
    c._num_local_layers = 1
    c._start_layer, c._end_layer = 0, 1
    c._block_region_consumer_indices = None
    c._block_unit_specs = None
    c._block_head_major_runs = ()
    c._block_region_roles = [None]
    c._per_block_bytes_list = [4]
    c._fp4_index_layout = False
    c._swa_block_regions = []
    c._completed_prefills = {}
    c._completed_prefills_lock = threading.Lock()
    c._completed_prefills_cv = threading.Condition(c._completed_prefills_lock)
    c._chunked_prefills = {}
    c._chunked_local_ids = {}
    c._chunked_admissions = {}
    c._admission_results = set()
    c._retired_chunked_transfers = OrderedDict()
    c._pending_chunked_requests = []
    c._kv_cache_ready_events = {}
    c._kv_cache_ready_batches = deque()
    c._kv_ready_expired = {}
    c._kv_ready_done_event = None
    c._completion_lock = threading.Lock()
    c._transfer_refcount_lock = threading.Lock()
    c._transfer_refcount = {}
    c.done_sending = set()
    c._rail_pool = None
    c._cuda_device = 0
    c._send_worker_streams = threading.local()
    c.transfer_engine = MemoryEngine()
    c._notify_transfer_result = MagicMock()

    c.pages = (ctypes.c_ubyte * 16)(*range(1, 17))
    c.state = (ctypes.c_ubyte * (2 * SLOT_BYTES))(*range(101, 101 + 2 * SLOT_BYTES))
    c.staging = (ctypes.c_ubyte * (2 * SLOT_BYTES))()
    c.kv_caches_base_addr = [ctypes.addressof(c.pages)]
    c._block_regions = [(ctypes.addressof(c.pages), 4)]
    c._slot_regions = []
    c._slot_region_consumer_indices = None
    c._slot_region_tp_segments = None
    c._slot_region_tp_replicated = None
    c._staging_base_addr = ctypes.addressof(c.staging)
    c._staging_slot_bytes = SLOT_BYTES
    c._staging_slots = SimpleNamespace(acquire=lambda: 1, release=lambda _row: None)

    def gather_slot(compute_slot, pool_idx):
        device.current_stream().launch("gather")
        ctypes.memmove(
            ctypes.addressof(c.staging) + pool_idx * SLOT_BYTES,
            ctypes.addressof(c.state) + compute_slot * SLOT_BYTES,
            SLOT_BYTES,
        )

    c._gather_slot = gather_slot
    return c


def write_request(c, pages_dst, staging_dst):
    return {
        "request_id": "req-1",
        "transfer_id": "xfer-1",
        "consumer_host": "decode",
        "consumer_rpc_port": 1,
        "notify_host": "decode",
        "notify_port": 2,
        "dst_block_ids": [0, 1],
        "consumer_tp_size": 1,
        "has_slot_regions": True,
        "src_slot_index": 1,
        "dst_slot_index": 0,
        "consumer_block_base_addrs": [ctypes.addressof(pages_dst)],
        "consumer_block_bpb": [4],
        "consumer_slot_base_addrs": [],
        "consumer_slot_bps": [],
        "consumer_staging_addr": ctypes.addressof(staging_dst),
    }


def test_a_staged_slot_gather_waits_only_for_forwards_before_its_publication(
    device,
):
    c = staged_slot_producer(device)
    pages_dst = (ctypes.c_ubyte * 8)()
    staging_dst = (ctypes.c_ubyte * SLOT_BYTES)()
    # The request's last forward: its final prefill chunk and, with deferred
    # output, the decode of its first token.
    device.default.launch("the request's last forward")
    meta = ConnectorMetadata()
    meta.add_new_req_to_save(
        "xfer-1", [2, 3], {"transfer_id": "xfer-1", "local_slot_index": 1}
    )
    c.start_load_kv(meta)
    # The engine goes on: another request's 16k-token chunk is queued on the
    # GPU when the consumer's write request arrives.
    device.default.launch("a later forward")

    c._execute_transfer(write_request(c, pages_dst, staging_dst))

    c._notify_transfer_result.assert_called_once()
    assert c._notify_transfer_result.call_args.kwargs == {"success": True}
    assert device.synchronized == [["the request's last forward", "gather"]]
    assert bytes(staging_dst) == bytes(c.state[SLOT_BYTES:])
    assert bytes(pages_dst) == bytes(c.pages[8:16])


def test_without_a_publication_the_gather_keeps_the_default_stream_order(device):
    c = staged_slot_producer(device)
    staging_dst = (ctypes.c_ubyte * SLOT_BYTES)()
    device.default.launch("a forward")
    request = write_request(c, (ctypes.c_ubyte * 8)(), staging_dst)

    # PP and chunked transfers record no publication.
    assert c._execute_block_slot_transfer(
        request, "decode:1", [], [], {"slot_index": 1}, "req-1"
    )

    assert device.synchronized == [["a forward", "gather"]]
    assert bytes(staging_dst) == bytes(c.state[SLOT_BYTES:])
