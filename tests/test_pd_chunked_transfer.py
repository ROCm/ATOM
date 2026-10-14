# SPDX-License-Identifier: MIT
"""Chunked P/D protocol tests using CPU buffers and the real address planner."""

import ctypes
import queue
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.chunked_prefill import (
    ChunkedPrefill,
    PrefillHandoff,
)
from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    ConnectorMetadata,
    KVConnectorOutput,
    KVTransferRegion,
    KVTransferTensors,
    PageRegion,
)


@pytest.fixture(autouse=True)
def _report_failed_receives_at_once(monkeypatch):
    """Keep protocol tests immediate; failure holds have separate coverage."""
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")


class Event:
    def __init__(self):
        self.ready = threading.Event()
        self.waiting = threading.Event()

    def synchronize(self):
        self.waiting.set()
        assert self.ready.wait(5), "fake GPU event was never completed"


class MemoryEngine:
    """Replace only the NIC; exercise actual region/page address construction."""

    def __init__(self):
        self.writes = queue.Queue()

    def batch_transfer_sync_write(self, target, sources, destinations, sizes):
        for src, dst, size in zip(sources, destinations, sizes, strict=True):
            ctypes.memmove(dst, src, size)
        self.writes.put((list(sources), list(destinations), list(sizes)))
        return 0


def producer(pp_rank=0, pp_size=1, num_tokens=10):
    c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    c.is_producer = True
    c.tp_size = c.dcp_size = 1
    c.tp_rank = 0
    c.pp_size, c.pp_rank = pp_size, pp_rank
    c.block_size = 4
    c._num_local_layers = 1
    c._start_layer = pp_rank
    c._end_layer = pp_rank + 1
    c._block_region_consumer_indices = None
    c._block_region_roles = [None]
    c._per_block_bytes_list = [4]
    c._completed_prefills_lock = threading.Lock()
    c._completed_prefills_cv = threading.Condition(c._completed_prefills_lock)
    c._chunked_prefills = {}
    c._reclaim_results = set()
    c._swa_block_regions = []
    c._send_executor = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
    c._chunked_local_ids = {}
    c._retired_chunked_transfers = OrderedDict()
    c._kv_cache_ready_events = {}
    c._kv_cache_ready_batches = deque()
    c._kv_ready_expired = {}
    c._kv_ready_done_event = None
    c._pending_chunked_requests = []
    c._completion_lock = threading.Lock()
    c._pending_recv_acks = {}
    c._pending_recv_stall_watch = {}
    c._held_failures = {}
    c._pending_recv_failed = set()
    c.done_sending = set()
    c.done_recving = set()
    c.failed_recving = set()
    c._received_handoffs = {}
    c._rail_pool = None
    c.transfer_engine = MemoryEngine()
    c._notify_transfer_result = MagicMock()
    c._cuda_device = 0
    src = (ctypes.c_ubyte * 48)(*range(1, 49))
    dst = [(ctypes.c_ubyte * 48)() for _ in range(pp_size)]
    c.kv_caches_base_addr = [ctypes.addressof(src)]
    meta = ConnectorMetadata()
    meta.add_new_req_to_save(
        7,
        [5, 1, 8],
        {
            "transfer_id": "xfer-a",
            "chunked_transfer": True,
            "num_prompt_tokens": num_tokens,
        },
    )
    c.start_load_kv(meta)
    request = {
        "chunked_transfer": True,
        "request_id": 21,
        "transfer_id": "xfer-a",
        "consumer_host": "cpu",
        "consumer_rpc_port": 1,
        "notify_host": "cpu",
        "notify_port": 2,
        "write_nonce": 3,
        "dst_block_ids": [2, 9, 4],
        "consumer_tp_size": 1,
        "consumer_base_addrs": [ctypes.addressof(buf) for buf in dst],
        "consumer_num_layers": pp_size,
    }
    return c, c._chunked_prefills["xfer-a"], request, src, dst


def dispatch(connector, request):
    """Use the listener's real queue/claim path, with sends run inline."""
    with connector._completed_prefills_lock:
        connector._pending_chunked_requests.append((request, mc.time.monotonic()))
    connector._dispatch_ready_chunked_requests()


def publish(state, end):
    event = Event()
    event.ready.set()
    state.publish(end, event)
    return event


@pytest.mark.parametrize("pp_size", [1, 3])
def test_transfers_only_ready_pages_once_and_waits_for_handoff(pp_size):
    for rank in range(pp_size):
        c, state, req, src, dest = producer(rank, pp_size)
        with ThreadPoolExecutor(1) as pool:
            event = Event()
            state.publish(6, event)  # token 4..5 live in an incomplete page
            future = pool.submit(dispatch, c, req)
            assert event.waiting.wait(5)
            assert c.transfer_engine.writes.empty()
            event.ready.set()
            _, addrs, sizes = c.transfer_engine.writes.get(timeout=5)
            assert sizes == [4]
            assert addrs == [ctypes.addressof(dest[rank]) + 8]
            assert bytes(dest[rank][8:12]) == bytes(src[20:24])
            assert bytes(dest[rank][36:40]) == bytes(4)
            assert c.get_finished().is_empty()
            publish(state, 8)
            assert c.transfer_engine.writes.get(timeout=5)[2] == [4]
            publish(state, 10)  # final partial page is now safe
            assert c.transfer_engine.writes.get(timeout=5)[2] == [4]
            assert not future.done()
            c._notify_transfer_result.assert_not_called()
            state.finish({"first_token_id": 42, "draft_token_ids": [43]})
            future.result(timeout=5)
        assert bytes(dest[rank][36:40]) == bytes(src[4:8])
        assert bytes(dest[rank][16:20]) == bytes(src[32:36])
        for other in range(pp_size):
            if other != rank:
                assert not any(dest[other])
        c._notify_transfer_result.assert_called_once_with(req, success=True)
        assert req["prefill_handoff"]["first_token_id"] == 42
        out = c.get_finished()
        assert out.finished_sending == set()
        assert out.connector_completions == {
            ConnectorCompletion("pd_source_safe", 7, True)
        }
        assert c.get_finished().is_empty()


def test_decode_prefix_is_skipped_even_before_source_blocks_are_available():
    c, state, req, src, dst = producer()
    req.update(num_computed_blocks=1, dst_block_ids=[9, 4])
    publish(state, 10)
    state.finish({"first_token_id": 42})
    dispatch(c, req)
    sources, _, sizes = c.transfer_engine.writes.get_nowait()
    assert sources == [ctypes.addressof(src) + 4, ctypes.addressof(src) + 32]
    assert sizes == [4, 4]
    assert bytes(dst[0][8:12]) == bytes(4)


def test_no_delta_page_until_whole_dcp_destination_is_ready():
    state = ChunkedPrefill(1, [4, 9, 1, 7, 8], 18, 4, timeout=0.01)
    publish(state, 6)
    with pytest.raises(RuntimeError, match="timed out"):
        state.wait_chunk(0, 2)
    publish(state, 10)
    assert state.wait_chunk(0, 2)[::2] == ([4, 9], 2)
    publish(state, 18)
    assert state.wait_chunk(2, 2)[::2] == ([1, 7, 8], 5)


def test_abort_does_not_release_a_source_while_rdma_is_reading():
    c, state, req, _, _ = producer()
    entered, release = threading.Event(), threading.Event()
    engine_write = c.transfer_engine.batch_transfer_sync_write

    def blocked_write(*args):
        entered.set()
        assert release.wait(5)
        return engine_write(*args)

    c.transfer_engine.batch_transfer_sync_write = blocked_write
    publish(state, 4)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(dispatch, c, req)
        assert entered.wait(5)
        state.cancel()
        assert c.get_finished().is_empty()
        release.set()
        future.result(timeout=5)
    c._notify_transfer_result.assert_called_once_with(req, success=False)
    assert c.get_finished().connector_completions == {
        ConnectorCompletion("pd_source_safe", 7, True)
    }
    with pytest.raises(RuntimeError, match="cancelled"):
        state.acquire(("late",), 1)


def test_fanout_and_duplicate_write_request_keep_independent_cursors():
    c, state, req, _, _ = producer()
    req["consumer_tp_size"] = 2
    publish(state, 10)
    state.finish({"first_token_id": 42})
    dispatch(c, req)
    dispatch(c, req.copy())
    assert c.transfer_engine.writes.qsize() == 1
    assert c.get_finished().is_empty()
    dispatch(c, dict(req, request_id=22, notify_port=3))
    assert c.transfer_engine.writes.qsize() == 2
    assert c.get_finished().connector_completions


def test_bad_prompt_or_block_geometry_fails_before_writing():
    for override in ({"prompt_digest": "different"}, {"dst_block_ids": [2]}):
        c, state, req, _, _ = producer()
        req.update(override)
        publish(state, 10)
        state.finish({"first_token_id": 42})
        dispatch(c, req)
        assert c.transfer_engine.writes.empty()
        c._notify_transfer_result.assert_called_once_with(req, success=False)


def consumer(expected=2):
    c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    c._completion_lock = threading.Lock()
    c._pending_recv_acks = {}
    c._pending_recv_stall_watch = {}
    c._held_failures = {}
    c._pending_recv_failed = set()
    c._fence_lock = threading.Lock()
    c._pending_recv_expected = {21: expected}
    c._pending_recv_stages = {}
    c._pending_recv_nonce = {21: 3}
    c._pending_recv = {21}
    c._pending_recv_blocks = {21: [2, 9, 4]}
    c._pending_recv_slots = {}
    c._swa_block_regions = []
    c._pending_handoffs = {21: {"failed": False, "metadata": None}}
    c._release_targets = {}
    c._blocks_pending_fence = []
    c.done_recving = set()
    c.failed_recving = set()
    c._received_handoffs = {}
    c._scatter_slot = None
    return c


@pytest.mark.parametrize("failure", [False, True])
def test_pp_receive_waits_for_every_stage_including_failure(failure):
    c = consumer()
    handoff = {
        "first_token_id": 42,
        "draft_token_ids": [43],
        "prefix_cache_hit_tokens": 4,
    }
    assert not c._record_write_done(21, 0, 0, 999, handoff=handoff)
    assert not c._record_write_done(21, 0, 0, 3, success=not failure, handoff=handoff)
    assert not c._record_write_done(21, 0, 0, 3, success=not failure, handoff=handoff)
    assert not c._record_write_done(21, 0, 1, 3, handoff=handoff)
    assert not c._record_write_done(21, 99, 0, 3, handoff=handoff)
    assert not c.done_recving and not c.failed_recving
    assert c._pending_recv_blocks
    assert c._record_write_done(21, 1, 0, 3, handoff=handoff)
    assert c.failed_recving == ({21} if failure else set())
    assert c.done_recving == (set() if failure else {21})
    assert c._blocks_pending_fence == ([] if failure else [2, 9, 4])
    assert bool(c._received_handoffs) is not failure
    assert not c._record_write_done(21, 1, 0, 3, handoff=handoff)


def test_chunked_failure_preserves_main_receive_hold(monkeypatch):
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "60")
    now = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    c = consumer(2)
    handoff = {"first_token_id": 42}
    assert not c._record_write_done(21, 0, 0, 3, success=False)
    assert c._record_write_done(21, 1, 0, 3, handoff=handoff)
    assert c._pending_recv_blocks == {21: [2, 9, 4]}
    assert not c.failed_recving and not c.done_recving
    assert not c._pending_handoffs and not c._received_handoffs
    now[0] += 59
    c._report_held_failures()
    assert not c.failed_recving and c._pending_recv_blocks
    now[0] += 2
    c._report_held_failures()
    assert c.failed_recving == {21}
    assert not c._pending_recv_blocks and not c._received_handoffs


def test_source_release_requires_every_tp_rank_of_every_pp_stage():
    tp = [KVOutputAggregator(2) for _ in range(3)]
    pp = PPKVAggregator(3)
    event = KVConnectorOutput(
        connector_completions={ConnectorCompletion("pd_source_safe", 7, True)}
    )
    empty = KVConnectorOutput()
    for rank in range(3):
        assert tp[rank].aggregate([event, empty]).is_empty()
        stage_out = tp[rank].aggregate([empty, event])
        result = pp.ingest(rank, stage_out)
        assert bool(result.connector_completions) == (rank == 2)
    sched = mc.MooncakeConnectorScheduler.__new__(mc.MooncakeConnectorScheduler)
    result = sched.process_pd_completions(result)
    assert result.finished_sending == {7}
    assert not result.connector_completions


def test_handoff_reaches_decode_scheduler_only_after_tp_quorum():
    sched = mc.MooncakeConnectorScheduler.__new__(mc.MooncakeConnectorScheduler)
    seq = SimpleNamespace(kv_transfer_params={})
    sched._chunked_receiving = {21: seq}
    agg = KVOutputAggregator(2)
    event = KVConnectorOutput(
        finished_recving={21},
        received_handoffs={21: PrefillHandoff(21, 42, (43,), 4)},
    )
    assert sched.process_pd_completions(
        agg.aggregate([event, KVConnectorOutput()])
    ).is_empty()
    assert not seq.kv_transfer_params
    out = sched.process_pd_completions(agg.aggregate([KVConnectorOutput(), event]))
    assert out.finished_recving == {21}
    assert seq.kv_transfer_params["first_token_id"] == 42
    assert seq.kv_transfer_params["draft_token_ids"] == [43]


def test_worker_publication_uses_batch_snapshot_and_own_gpu_event(monkeypatch):
    c, state, _, _, _ = producer()
    event = MagicMock()
    monkeypatch.setattr(mc.torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(mc.torch.cuda, "current_stream", lambda _: "stage-stream")
    batch = SimpleNamespace(req_ids=[7], total_seqs_num_prefill=1, context_lens=[6])
    c.publish_prefill_chunks(batch)
    event.record.assert_called_once_with("stage-stream")
    assert state.num_ready_blocks == 1
    assert state.event is event


def test_final_only_metadata_and_abort_are_dispatchable():
    c, state, _, _, _ = producer()
    for aborted in (False, True):
        meta = ConnectorMetadata()
        meta.add_new_req_to_save(
            7,
            [5, 1, 8],
            {
                "transfer_id": "xfer-a",
                "chunked_transfer": True,
                "num_prompt_tokens": 10,
                "prefill_aborted": aborted,
                "prefill_handoff": {"first_token_id": 42},
            },
        )
        assert meta.has_work()
        c.start_load_kv(meta)
        assert state.handoff == {"first_token_id": 42}
        assert state.cancelled is aborted


@pytest.mark.parametrize("bad_metadata", [False, True])
def test_tp_failure_or_metadata_disagreement_discards_partial_handoff(bad_metadata):
    agg = KVOutputAggregator(2)
    first = KVConnectorOutput(
        finished_recving={21}, received_handoffs={21: PrefillHandoff(21, 42)}
    )
    assert agg.aggregate([first, KVConnectorOutput()]).is_empty()
    other = (
        KVConnectorOutput(
            finished_recving={21}, received_handoffs={21: PrefillHandoff(21, 99)}
        )
        if bad_metadata
        else KVConnectorOutput(failed_recving={21})
    )
    out = agg.aggregate([KVConnectorOutput(), other])
    assert out.failed_recving == {21}
    assert not out.received_handoffs and not out.finished_recving
    assert not agg._received_handoffs


def scheduler_connector(monkeypatch, is_producer, pp_size=1):
    monkeypatch.setattr(mc, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mc, "get_open_port", lambda: 6400)
    return mc.MooncakeConnectorScheduler(
        SimpleNamespace(
            kv_transfer_config={
                "kv_role": "kv_producer" if is_producer else "kv_consumer",
                "enable_chunked_transfer": True,
            },
            tensor_parallel_size=1,
            pipeline_parallel_size=pp_size,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
        )
    )


@pytest.mark.parametrize("pp_size", [1, 3])
def test_real_scheduler_pins_from_first_chunk_through_final_handoff(
    monkeypatch, seq_factory, pp_size
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
    from atom.sampling_params import SamplingParams

    sched = Scheduler(
        MockConfig(max_num_batched_tokens=4, pipeline_parallel_size=pp_size)
    )
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True, pp_size)
    seq = seq_factory(
        list(range(10)),
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode": True,
            "transfer_id": "xfer-a",
        },
    )
    sched.add(seq)
    batches = []
    for _ in range(3):
        batch, seqs = sched.schedule()
        batches.append(batch)
        assert not sched._is_preemptable(seq)
        result = ScheduledBatchOutput([seq.id], [(42,)], None, None, None)
        sched.postprocess(list(seqs.values()), result, batch=batch)
    assert [b.context_lens.tolist() for b in batches] == [[4], [8], [10]]
    # Earlier snapshots must not follow the live seq past the next PP forward.
    assert batches[0].context_lens.tolist() == [4]
    initial = batches[0].connector_meta_output.reqs_to_save[seq.id]
    assert initial.chunked_transfer and initial.num_prompt_tokens == 10
    assert initial.prefill_handoff is None
    assert initial.transfer_id == "xfer-a"
    final = connector.build_connector_meta().reqs_to_save[seq.id]
    assert final.prefill_handoff["first_token_id"] == 42
    assert connector.should_defer_free(seq)
    assert seq.block_table
    sched._update_from_kv_xfer_finished(
        KVConnectorOutput(
            connector_completions={ConnectorCompletion("pd_source_safe", seq.id, True)}
        )
    )
    assert not connector.should_defer_free(seq)
    assert not seq.block_table


def test_producer_abort_publishes_cancel_instead_of_dropping_claim(
    monkeypatch, seq_factory
):
    connector = scheduler_connector(monkeypatch, True, 3)
    seq = seq_factory(
        list(range(10)),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode": True,
            "transfer_id": "abort",
        },
    )
    seq.block_table = [1, 2, 3]
    connector.update_state_after_alloc(seq)
    connector.build_connector_meta()
    seq.leave_reason = "aborted"
    connector.request_finished(seq)
    assert connector.should_defer_free(seq)
    assert connector.build_connector_meta().reqs_to_save[seq.id].prefill_aborted
    connector.send_finished(seq.id)
    assert not connector.should_defer_free(seq)


def test_mutable_slot_is_sent_once_after_final_metadata():
    c, state, req, _, dst = producer()
    state.slot_index = 0
    slot_src = (ctypes.c_ubyte * 8)(*range(1, 9))
    slot_dst = (ctypes.c_ubyte * 8)()
    c._block_regions = [(c.kv_caches_base_addr[0], 4)]
    c._slot_regions = [(ctypes.addressof(slot_src), 4)]
    c._swa_block_regions = []
    c._fp4_index_layout = False
    c._gather_slot = None
    req.update(
        has_slot_regions=True,
        dst_slot_index=0,
        consumer_block_base_addrs=[ctypes.addressof(dst[0])],
        consumer_block_bpb=[4],
        consumer_slot_base_addrs=[ctypes.addressof(slot_dst)],
        consumer_slot_bps=[4],
    )
    with ThreadPoolExecutor(1) as pool:
        publish(state, 4)
        future = pool.submit(dispatch, c, req)
        c.transfer_engine.writes.get(timeout=5)
        assert not any(slot_dst)
        publish(state, 10)
        c.transfer_engine.writes.get(timeout=5)
        assert not any(slot_dst)
        # Checkpointing may relocate the live state between prefill chunks.
        state.finish({"first_token_id": 42}, slot_index=1)
        future.result(timeout=5)
    assert bytes(slot_dst[:4]) == bytes(slot_src[4:8])
    assert c.transfer_engine.writes.qsize() == 1
    c._notify_transfer_result.assert_called_once_with(req, success=True)


@pytest.mark.parametrize("reverse_indexed", [False, True])
def test_swa_only_pool_transfers_final_relocated_slot(reverse_indexed):
    import msgpack

    p, state, _, page_src, page_destinations = producer()
    d = consumer(1)
    d.is_producer = False
    d.tp_size = d.dcp_size = d.pp_size = 1
    d.tp_rank = d.dp_rank = d.dcp_rank = 0
    d.dcp_interleave_size = 1
    d.local_ip = "cpu"
    d.ib_device = None
    d._notification_port = 2
    d._staging_pool_size = d._staging_slot_bytes = 0
    d._rail_pool = None
    d._send_on_socket = MagicMock()
    d.transfer_engine = MemoryEngine()
    swa_src = (ctypes.c_ubyte * 12)(*range(1, 13))
    swa_dst = (ctypes.c_ubyte * 20)()
    regions = []
    for connector, pages, slots in (
        (p, page_src, swa_src),
        (d, page_destinations[0], swa_dst),
    ):
        connector.engine_id = "cpu-test"
        connector.rpc_port = 1
        connector.block_len = 4
        connector.ib_devices = []
        connector.transfer_engine.batch_register_memory = MagicMock(return_value=0)
        # Exercise real registration with a completed mock listener bind.
        connector._listener_bound = threading.Event()
        connector._listener_error = None
        connector._write_listener = connector._publish_listener_state
        connector._notification_listener = connector._publish_listener_state
        swa_region = KVTransferRegion(
            ctypes.addressof(slots), len(slots), 4, reverse_indexed=reverse_indexed
        )
        regions.append(swa_region)
        transfer = KVTransferTensors(
            pages=[
                PageRegion(KVTransferRegion(ctypes.addressof(pages), len(pages), 4))
            ],
            slot_regions=[],
            swa_block_regions=[swa_region],
            num_slots=len(slots) // 4,
        )
        transfer.set_block_count(12)
        connector.register_kv_caches({"layer": None}, transfer_tensors=transfer)

    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(
        21,
        [2, 9, 4],
        {
            "chunked_transfer": True,
            "transfer_id": "xfer-a",
            "remote_host": "cpu",
            "remote_handshake_port": 6301,
            "remote_tp_size": 1,
            "local_slot_index": 2,
        },
        local_swa_block_ids=[2],
    )
    d._mla_landing = MagicMock()
    d.start_load_kv(meta)
    d._mla_landing.advertise.assert_not_called()
    d._mla_landing.begin.assert_not_called()
    assert meta.reqs_to_recv.keys() <= d._pending_handoffs.keys()
    _, (_, payload) = d._send_on_socket.call_args.args
    request = msgpack.loads(payload)
    assert request["has_slot_regions"]
    assert request["dst_swa_block_ids"] == [2]
    assert request["consumer_staging_addr"] == 0
    assert request["consumer_slot_base_addrs"] == []

    state.slot_index = 0
    state.swa_block_ids = [0]
    with ThreadPoolExecutor(1) as pool:
        publish(state, 4)
        future = pool.submit(dispatch, p, request)
        p.transfer_engine.writes.get(timeout=5)
        assert not any(swa_dst)
        publish(state, 10)
        p.transfer_engine.writes.get(timeout=5)
        assert not any(swa_dst)
        assert not future.done()
        state.finish({"first_token_id": 42}, slot_index=1, swa_block_ids=[1])
        future.result(timeout=5)

    src_region, dst_region = regions
    src_addr = src_region.unit_addr(1)
    dst_addr = dst_region.unit_addr(2)
    assert p.transfer_engine.writes.get_nowait() == ([src_addr], [dst_addr], [4])
    assert p.transfer_engine.writes.empty()
    expected = bytearray(len(swa_dst))
    offset = dst_addr - ctypes.addressof(swa_dst)
    expected[offset : offset + 4] = ctypes.string_at(src_addr, 4)
    assert bytes(swa_dst) == expected
    p._notify_transfer_result.assert_called_once_with(request, success=True)


def test_terminal_without_consumer_expires_but_live_rdma_cannot_expire(monkeypatch):
    state = ChunkedPrefill(1, [0], 4, 4, timeout=1)
    now = state.updated
    monkeypatch.setattr(
        "atom.kv_transfer.disaggregation.chunked_prefill.time.monotonic",
        lambda: now + 2,
    )
    assert not state.source_safe()  # An active prefill has no handoff yet.
    state.finish({"first_token_id": 42})
    state.acquire((1,), 1)
    monkeypatch.setattr(
        "atom.kv_transfer.disaggregation.chunked_prefill.time.monotonic",
        lambda: now + 4,
    )
    assert not state.source_safe()
    state.release()
    assert state.source_safe()


@pytest.mark.parametrize("pp_size", [1, 3])
def test_idle_engine_dispatches_terminal_metadata_to_all_stages(pp_size):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.engine_core import EngineCore
        from atom.model_engine.pp_engine_core import PPEngineCoreProc

    cls = EngineCore if pp_size == 1 else PPEngineCoreProc
    core = cls.__new__(cls)
    meta = ConnectorMetadata()
    meta.reqs_to_send[7] = 1.0
    core.kv_transfer_enabled = True
    core.scheduler = SimpleNamespace(
        kv_connector=SimpleNamespace(
            is_producer=True,
            is_offload=False,
            enable_chunked_transfer=True,
            build_connector_meta=lambda: meta,
        )
    )
    core.runner_mgr = MagicMock()
    core.pp_transport = MagicMock()
    core._dispatch_idle_offload_work()
    core.runner_mgr.call_func.assert_called_once_with(
        "process_kvconnector_output", meta
    )
    if pp_size > 1:
        sent = core.pp_transport.send_metadata.call_args.args[0]
        assert sent.connector_meta_output is meta
        assert sent.req_ids == []


def test_consumer_dispatches_all_pp_stages_without_source_block_ids(
    monkeypatch, seq_factory
):
    import msgpack

    from atom.kv_transfer.disaggregation.port_offset import side_channel_port_offset

    sched = scheduler_connector(monkeypatch, False)
    seq = seq_factory(
        list(range(10)),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_prefill": True,
            "transfer_id": "xfer-a",
            "remote_host": "127.0.0.1",
            "remote_handshake_port": 6301,
            "remote_pp_size": 3,
            "remote_tp_size": 1,
            "block_size": 4,
            "dcp_size": 1,
        },
    )
    seq.block_table = [2, 9, 4]
    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)
    c = consumer(3)
    c.is_producer = False
    c.tp_size = c.dcp_size = 1
    c.tp_rank = c.dp_rank = c.dcp_rank = 0
    c.dcp_interleave_size = 1
    c.local_ip = "127.0.0.1"
    c.rpc_port = 777
    c.ib_device = None
    c._notification_port = 778
    c._num_local_layers = 3
    c._block_regions = [(100, 4), (200, 4), (300, 4)]
    c._block_region_roles = [None] * 3
    c._has_slot_regions = False
    c.kv_caches_base_addr = [100, 200, 300]
    c._send_on_socket = MagicMock()
    c.start_load_kv(sched.build_connector_meta())
    assert c._send_on_socket.call_count == 3
    for stage, call in enumerate(c._send_on_socket.call_args_list):
        address, (_, payload) = call.args
        req = msgpack.loads(payload)
        assert address.endswith(
            str(6301 + side_channel_port_offset(0, 0, 1, stage, 3, 1))
        )
        assert req["chunked_transfer"]
        assert req["transfer_id"] == "xfer-a"
        assert req["dst_block_ids"] == [9, 4]
        assert req["num_computed_blocks"] == 1
        assert req["src_block_ids"] == []
    assert c._pending_recv_expected[seq.id] == 3
    assert not c._release_targets


def test_unadmitted_requests_do_not_starve_computed_requests_in_send_pool():
    import time

    c, state, req, _, _ = producer()
    publish(state, 10)
    state.finish({"first_token_id": 42})
    c._pending_chunked_requests = [
        (dict(req, transfer_id="not-admitted"), time.monotonic()),
        (req, time.monotonic()),
        (req.copy(), time.monotonic()),
    ]
    with ThreadPoolExecutor(1) as executor:
        c._send_executor = executor
        c._dispatch_ready_chunked_requests()
    assert len(c._pending_chunked_requests) == 1
    assert c.transfer_engine.writes.qsize() == 1
    c._notify_transfer_result.assert_called_once_with(req, success=True)


def test_dcp_chunked_descriptors_relayout_each_destination_page_once():
    from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE

    c, state, req, src, dst = producer()
    c._block_region_roles = [MLA_KV_ROLE]
    req.update(
        consumer_dcp_size=2,
        consumer_dcp_rank=1,
        consumer_dcp_interleave=1,
        src_block_skip_factor=2,
        dst_block_ids=[2, 9],
    )
    with ThreadPoolExecutor(1) as pool:
        publish(state, 8)
        future = pool.submit(dispatch, c, req)
        c.transfer_engine.writes.get(timeout=5)
        assert bytes(dst[0][8:12]) == bytes([src[21], src[23], src[5], src[7]])
        assert not any(dst[0][36:40])
        publish(state, 10)
        c.transfer_engine.writes.get(timeout=5)
        state.finish({"first_token_id": 42})
        future.result(timeout=5)
    assert bytes(dst[0][36:40]) == bytes([src[33], src[35], 0, 0])
    c._notify_transfer_result.assert_called_once_with(req, success=True)


@pytest.mark.parametrize(
    "handoff",
    [
        None,
        [],
        "invalid",
        {},
        {"first_token_id": "42"},
        {"first_token_id": -1},
        {"first_token_id": 42, "draft_token_ids": [None]},
    ],
)
def test_malformed_handoff_fails_receive_without_killing_notification_loop(handoff):
    c = consumer(1)
    assert c._record_write_done(21, 0, 0, 3, handoff=handoff)
    assert c.failed_recving == {21}
    assert not c.done_recving and not c._received_handoffs


def test_duplicate_cancelled_request_waits_for_original_rdma():
    p, state, request, _, _ = producer()
    d = consumer(1)
    entered, release = threading.Event(), threading.Event()
    write = p.transfer_engine.batch_transfer_sync_write

    def blocked_write(*args):
        entered.set()
        assert release.wait(5)
        return write(*args)

    def notify(data, *, success):
        d._record_write_done(
            data["request_id"], 0, 0, data["write_nonce"], success=success
        )

    p.transfer_engine.batch_transfer_sync_write = blocked_write
    p._notify_transfer_result.side_effect = notify
    publish(state, 4)
    with ThreadPoolExecutor(1) as pool:
        original = pool.submit(dispatch, p, request)
        try:
            assert entered.wait(5)
            state.cancel()
            dispatch(p, request.copy())
            # A duplicate must not report failure while the original still
            # writes into D's reservation, even after P has been cancelled.
            assert not d.failed_recving
            assert d._pending_recv_blocks
            p._notify_transfer_result.assert_not_called()
        finally:
            release.set()
            original.result(timeout=5)
    assert d.failed_recving == {21}
    p._notify_transfer_result.assert_called_once_with(request, success=False)


@pytest.mark.parametrize("aborted", [False, True])
def test_chunk_source_retirement_releases_legacy_ready_event(aborted):
    p, state, request, _, _ = producer()
    publish(state, 10)
    p._kv_cache_ready_events = {7: object(), "legacy": object()}
    if aborted:
        state.cancel()
    else:
        state.finish({"first_token_id": 42})
        dispatch(p, request)
    assert p.get_finished().connector_completions
    assert set(p._kv_cache_ready_events) == {"legacy"}


def test_completed_chunk_receive_is_not_registered_again_on_reallocation(
    monkeypatch, seq_factory
):
    sched = scheduler_connector(monkeypatch, False)
    seq = seq_factory(
        list(range(10)),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_prefill": True,
            "transfer_id": "xfer-a",
            "block_size": 4,
            "dcp_size": 1,
        },
    )
    seq.block_table = [2, 9, 4]
    sched.update_state_after_alloc(seq)
    sched.build_connector_meta()
    sched.process_pd_completions(
        KVConnectorOutput(
            finished_recving={seq.id},
            received_handoffs={seq.id: PrefillHandoff(seq.id, 42)},
        )
    )
    assert not sched._chunked_receiving
    # A preempted D sequence keeps its transfer parameters when re-prefilling
    # locally. It must not register another remote handoff that will never come.
    sched.update_state_after_alloc(seq)
    assert not sched._chunked_receiving
    assert not sched.build_connector_meta().has_work()


def test_idle_write_listener_expires_unadmitted_chunk_request(monkeypatch):
    from contextlib import nullcontext

    import msgpack

    p, _, request, _, _ = producer()
    request["transfer_id"] = "never-admitted"
    p._side_channel_port = 6301
    p._listener_bound = threading.Event()
    p._listener_error = None
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])

    class StopListener(Exception):
        pass

    class IdleSocket:
        received = False
        idle_polled = False

        def poll(self, timeout):
            if not self.received:
                return mc.zmq.POLLIN
            if self.idle_polled:
                raise StopListener
            self.idle_polled = True
            now[0] += mc.PREFILL_LOOKUP_TIMEOUT + 1
            return 0

        def recv_multipart(self):
            if self.received:
                raise StopListener
            self.received = True
            return [b"consumer", mc.MSG_WRITE_REQUEST, msgpack.dumps(request)]

    sock = IdleSocket()
    monkeypatch.setattr(mc, "zmq_socket_ctx", lambda *a, **kw: nullcontext(sock))
    with ThreadPoolExecutor(1) as pool:
        p._send_executor = pool
        with pytest.raises(StopListener):
            p._write_listener()
    p._notify_transfer_result.assert_called_once_with(
        dict(request, _deadline=mc.PD_TRANSFER_DEADLINE_S), success=False
    )
    assert not p._pending_chunked_requests


@pytest.mark.parametrize("waiting_cleanup", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_scheduler_abort_keeps_source_until_readers_and_tp_pp_drain(
    monkeypatch, seq_factory, waiting_cleanup, composite
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler

    sched = Scheduler(MockConfig(max_num_batched_tokens=4))
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True)
    seq = seq_factory(
        list(range(10)),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode": True,
            "transfer_id": "xfer-a",
        },
    )
    sched.add(seq)
    batch, seqs = sched.schedule()
    worker, _, request, _, _ = producer()
    worker._chunked_prefills.clear()
    worker._chunked_local_ids.clear()
    worker.start_load_kv(batch.connector_meta_output)
    state = worker._chunked_prefills["xfer-a"]
    request["prompt_digest"] = state.prompt_digest
    publish(state, 4)
    entered, release = threading.Event(), threading.Event()
    write = worker.transfer_engine.batch_transfer_sync_write

    def blocked_write(*args):
        entered.set()
        assert release.wait(5)
        return write(*args)

    worker.transfer_engine.batch_transfer_sync_write = blocked_write
    if composite:
        from atom.kv_transfer.disaggregation.multi.multi_connector import (
            MultiConnectorScheduler,
        )

        multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
        multi._connectors = [connector]
        multi._load_winner = {}
        multi.is_producer, multi.is_offload = True, False
        sched.kv_connector = multi
    deallocate = MagicMock(wraps=sched.block_manager.deallocate)
    monkeypatch.setattr(sched.block_manager, "deallocate", deallocate)
    with ThreadPoolExecutor(1) as pool:
        transfer = pool.submit(dispatch, worker, request)
        try:
            assert entered.wait(5)
            if waiting_cleanup:
                # Also cover the cleanup path used by an aborted parked load.
                sched.running.remove(seq)
                sched.waiting.append(seq)
            assert sched.abort_request(seq.id)
            if not waiting_cleanup:
                sched.postprocess(
                    list(seqs.values()),
                    ScheduledBatchOutput([seq.id], [(42,)], None, None, None),
                    batch=batch,
                )
            assert not state.cancelled  # Cancellation is still scheduler metadata.
            assert state.readers == 1 and not state.source_safe()
            assert connector.should_defer_free(seq)
            assert seq.id in sched.deferred_free_blocks
            deallocate.assert_not_called()
            worker.start_load_kv(connector.build_connector_meta())
            assert state.cancelled
            sched._update_from_kv_xfer_finished(worker.get_finished())
            deallocate.assert_not_called()
        finally:
            # Keep assertion failures bounded on an implementation that drops
            # the scheduler claim before it delivers worker cancellation.
            state.cancel()
            release.set()
            transfer.result(timeout=5)

    event = worker.get_finished()
    tp = [KVOutputAggregator(2) for _ in range(2)]
    pp = PPKVAggregator(2)
    for stage in range(2):
        partial = tp[stage].aggregate([event, KVConnectorOutput()])
        sched._update_from_kv_xfer_finished(pp.ingest(stage, partial))
        deallocate.assert_not_called()
        complete = tp[stage].aggregate([KVConnectorOutput(), event])
        sched._update_from_kv_xfer_finished(pp.ingest(stage, complete))
        if stage == 0:
            deallocate.assert_not_called()
    deallocate.assert_called_once_with(seq)
    assert not seq.block_table
    assert not connector.should_defer_free(seq)


@pytest.mark.parametrize("p_hits,d_hits", [(4, 0), (4, 8), (0, 8)])
def test_handoff_cache_hits_reach_final_chat_response(
    monkeypatch, seq_factory, p_hits, d_hits
):
    from conftest import MockConfig

    from atom.entrypoints.openai.serving_chat import build_chat_response
    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
    from atom.sampling_params import SamplingParams

    sched = Scheduler(MockConfig(enable_prefix_caching=True))
    sched.kv_connector = scheduler_connector(monkeypatch, False)
    if d_hits:
        seed = seq_factory(list(range(d_hits)) + [99, 99])
        assert sched.block_manager.allocate(seed)
        sched.block_manager.hash_blocks(seed, d_hits)
        sched.block_manager.deallocate(seed)
    seq = seq_factory(
        list(range(10)),
        sampling_params=SamplingParams(max_tokens=2),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_prefill": True,
            "transfer_id": "xfer-a",
            "block_size": 4,
            "dcp_size": 1,
        },
    )
    sched.add(seq)
    _, seqs = sched.schedule()
    assert not seqs
    assert seq.prefix_cache_hit_tokens == d_hits
    sched._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={seq.id},
            received_handoffs={seq.id: PrefillHandoff(seq.id, 42, (), p_hits)},
        )
    )
    batch, seqs = sched.schedule()
    assert batch.total_seqs_num_decode == 1
    assert seq.prefix_cache_hit_tokens == p_hits
    outputs = queue.Queue()
    assert sched.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput([seq.id], [(43,)], None, None, None),
        stream_output_queue=outputs,
        batch=batch,
    ) == [seq]
    request_id, output = outputs.get_nowait()[0]
    assert output.finished
    assert output.num_cached_tokens == p_hits
    response = build_chat_response(
        str(request_id),
        "test-model",
        "answer",
        {
            "finish_reason": output.finish_reason,
            "num_tokens_input": seq.num_prompt_tokens,
            "num_tokens_output": len(output.output_tokens),
            "num_cached_tokens": output.num_cached_tokens,
        },
    )
    assert (
        response.model_dump()["usage"]["prompt_tokens_details"]["cached_tokens"]
        == p_hits
    )


def test_opposite_admission_order_with_exhausted_kv_pools(monkeypatch, seq_factory):
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
    from atom.sampling_params import SamplingParams

    p = Scheduler(MockConfig(num_kvcache_blocks=3, max_num_batched_tokens=10))
    d = Scheduler(MockConfig(num_kvcache_blocks=3, max_num_batched_tokens=10))
    p.kv_connector = scheduler_connector(monkeypatch, True)
    d.kv_connector = scheduler_connector(monkeypatch, False)

    def request(name, is_producer):
        return seq_factory(
            list(range(10)),
            sampling_params=SamplingParams(max_tokens=1 if is_producer else 2),
            kv_transfer_params={
                "chunked_transfer": True,
                "do_remote_decode" if is_producer else "do_remote_prefill": True,
                "transfer_id": name,
                "block_size": 4,
                "dcp_size": 1,
            },
        )

    pa, pb = request("xfer-a", True), request("xfer-b", True)
    da, db = request("xfer-a", False), request("xfer-b", False)
    for seq in (pa, pb):
        p.add(seq)
    for seq in (db, da):
        d.add(seq)
    pbatch, pseqs = p.schedule()
    _, dseqs = d.schedule()
    assert list(pseqs) == [pa.id] and not dseqs
    assert p.block_manager.kv.num_free == d.block_manager.kv.num_free == 0
    assert not pb.block_table and not da.block_table

    worker, _, wire, _, _ = producer()
    worker._chunked_prefills.clear()
    worker._chunked_local_ids.clear()
    worker._send_executor = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
    worker.start_load_kv(pbatch.connector_meta_output)
    wire_b = dict(wire, request_id=db.id, transfer_id="xfer-b")
    # D's B reservation has already asked P for a transfer, but P only has A.
    worker._pending_chunked_requests.append((wire_b, mc.time.monotonic()))
    worker._dispatch_ready_chunked_requests()
    assert len(worker._pending_chunked_requests) == 1
    publish(worker._chunked_prefills["xfer-a"], 10)
    p.postprocess(
        list(pseqs.values()),
        ScheduledBatchOutput([pa.id], [(42,)], None, None, None),
        batch=pbatch,
    )
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    # P cannot allocate B. Even a snapshot containing ONLY the capacity signal
    # must count as work, or the idle engine will never dispatch it.
    now = [mc.time.monotonic()]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    pressure, seqs = p.schedule()
    assert not seqs and not pressure.connector_meta_output.has_work()
    now[0] += mc.CHUNKED_RECLAIM_STALL_SECONDS
    pressure, seqs = p.schedule()
    assert not seqs and pressure.connector_meta_output.has_work()
    assert pressure.connector_meta_output.reclaim_unclaimed_prefills
    assert not pressure.connector_meta_output.reqs_to_save
    worker.start_load_kv(pressure.connector_meta_output)
    p._update_from_kv_xfer_finished(worker.get_finished())
    assert not pa.block_table and p.block_manager.kv.num_free == 3

    # P can now finish B, well before the 60-second transfer timeout.
    pbatch, pseqs = p.schedule()
    assert list(pseqs) == [pb.id]
    worker.start_load_kv(pbatch.connector_meta_output)
    state_b = worker._chunked_prefills["xfer-b"]
    wire_b["prompt_digest"] = state_b.prompt_digest
    p.postprocess(
        list(pseqs.values()),
        ScheduledBatchOutput([pb.id], [(42,)], None, None, None),
        batch=pbatch,
    )
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    publish(state_b, 10)
    worker._dispatch_ready_chunked_requests()
    worker._notify_transfer_result.assert_called_once_with(wire_b, success=True)
    p._update_from_kv_xfer_finished(worker.get_finished())
    assert not pb.block_table
    d._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={db.id},
            received_handoffs={
                db.id: PrefillHandoff.from_wire(db.id, wire_b["prefill_handoff"])
            },
        )
    )
    dbatch, dseqs = d.schedule()
    assert list(dseqs) == [db.id]
    assert d.postprocess(
        list(dseqs.values()),
        ScheduledBatchOutput([db.id], [(43,)], None, None, None),
        batch=dbatch,
    ) == [db]

    # D's late A admission gets an immediate failure for the retired source,
    # then completes by the existing local-prefill fallback using the same pool.
    _, seqs = d.schedule()
    assert not seqs and da.block_table
    wire_a = dict(wire, request_id=da.id)
    worker._pending_chunked_requests.append((wire_a, mc.time.monotonic()))
    worker._dispatch_ready_chunked_requests()
    worker._notify_transfer_result.assert_called_with(wire_a, success=False)
    assert not worker._pending_chunked_requests
    d._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={da.id}))
    dbatch, dseqs = d.schedule()
    assert list(dseqs) == [da.id] and dbatch.total_seqs_num_prefill == 1
    d.postprocess(
        list(dseqs.values()),
        ScheduledBatchOutput([da.id], [(44,)], None, None, None),
        batch=dbatch,
    )
    dbatch, dseqs = d.schedule()
    assert d.postprocess(
        list(dseqs.values()),
        ScheduledBatchOutput([da.id], [(45,)], None, None, None),
        batch=dbatch,
    ) == [da]
    assert p.block_manager.kv.num_free == d.block_manager.kv.num_free == 3
    assert not p.deferred_free_blocks and not d.deferred_free_blocks


@pytest.mark.parametrize("reader_done", [False, True])
def test_capacity_reclaim_preserves_active_and_partially_served_sources(reader_done):
    worker, state, _, _, _ = producer()
    publish(state, 10)
    state.finish({"first_token_id": 42})
    assert state.acquire(("first-consumer",), 2)
    if reader_done:
        state.release()
    pressure = ConnectorMetadata()
    pressure.reclaim_unclaimed_prefills = {7: "xfer-a"}
    worker.start_load_kv(pressure)
    assert not state.cancelled and not state.source_safe()
    assert worker.get_finished().connector_completions == {
        ConnectorCompletion("pd_reclaim_done", 7, False)
    }


def test_late_metadata_cannot_resurrect_retired_source():
    worker, state, wire, _, _ = producer()
    state.cancel()
    assert worker.get_finished().connector_completions
    meta = ConnectorMetadata()
    meta.add_new_req_to_save(
        state.req_id,
        list(state.block_ids),
        {
            "transfer_id": "xfer-a",
            "chunked_transfer": True,
            "num_prompt_tokens": 10,
            "prefill_handoff": {"first_token_id": 42},
        },
    )
    worker.start_load_kv(meta)
    assert not worker._chunked_prefills and worker.get_finished().is_empty()
    dispatch(worker, wire)
    worker._notify_transfer_result.assert_called_once_with(wire, success=False)
    assert worker.transfer_engine.writes.empty()


def test_capacity_signal_survives_composite_metadata(monkeypatch, seq_factory):
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    connector = scheduler_connector(monkeypatch, True)
    seq = seq_factory(
        list(range(10)),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode": True,
            "transfer_id": "xfer-a",
        },
    )
    seq.block_table = [1, 2, 3]
    connector.update_state_after_alloc(seq)
    connector.build_connector_meta()
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [connector]
    seq.output_tokens = [42]
    connector.request_finished(seq)
    connector.build_connector_meta()
    now = [mc.time.monotonic()]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    multi.admission_blocked(seq)
    now[0] += mc.CHUNKED_RECLAIM_STALL_SECONDS
    multi.admission_blocked(seq)
    meta = multi.build_connector_meta()
    assert meta.has_work() and meta.metas[0].reclaim_unclaimed_prefills
    assert not multi.build_connector_meta().has_work()


def test_capacity_reclaim_waits_for_final_prefill():
    worker, state, _, _, _ = producer()
    publish(state, 4)
    pressure = ConnectorMetadata()
    pressure.reclaim_unclaimed_prefills = {7: "xfer-a"}
    worker.start_load_kv(pressure)
    assert not state.cancelled and not state.source_safe()


def test_terminal_transfer_history_is_bounded_without_retaining_sources(monkeypatch):
    monkeypatch.setattr(mc, "CHUNKED_RETIRED_TRANSFER_LIMIT", 2)
    worker, state, _, _, _ = producer()
    state.cancel()
    worker.get_finished()
    for transfer_id in ("xfer-b", "xfer-c"):
        meta = ConnectorMetadata()
        meta.add_new_req_to_save(
            7,
            [5, 1, 8],
            {
                "transfer_id": transfer_id,
                "chunked_transfer": True,
                "num_prompt_tokens": 10,
                "prefill_aborted": True,
            },
        )
        worker.start_load_kv(meta)
        worker.get_finished()
    assert list(worker._retired_chunked_transfers) == ["xfer-b", "xfer-c"]
    assert not worker._chunked_prefills and not worker._chunked_local_ids


@pytest.mark.parametrize("pp_size", [1, 3])
@pytest.mark.parametrize(
    "is_offload,enabled,is_producer,dispatches",
    [
        (False, False, True, False),
        (False, False, False, False),
        (False, True, False, False),
        (False, True, True, True),
        (True, False, False, True),
    ],
)
def test_idle_dispatch_remains_opt_in(
    pp_size, is_offload, enabled, is_producer, dispatches
):
    with stubbed_aiter():
        from atom.model_engine.engine_core import EngineCore
        from atom.model_engine.pp_engine_core import PPEngineCoreProc

    cls = EngineCore if pp_size == 1 else PPEngineCoreProc
    core = cls.__new__(cls)
    core.kv_transfer_enabled = True
    meta = ConnectorMetadata()
    meta.reqs_to_send[7] = 1.0
    connector = SimpleNamespace(
        is_producer=is_producer,
        is_offload=is_offload,
        enable_chunked_transfer=enabled,
        build_connector_meta=MagicMock(return_value=meta),
    )
    core.scheduler = SimpleNamespace(kv_connector=connector)
    core.runner_mgr = MagicMock()
    core.pp_transport = MagicMock()
    core._dispatch_idle_offload_work()
    assert connector.build_connector_meta.call_count == int(dispatches)
    assert core.runner_mgr.call_func.call_count == int(dispatches)
    assert core.pp_transport.send_metadata.call_count == int(dispatches and pp_size > 1)


def test_reclaim_is_gated_ordered_and_waits_for_tp_pp_ack(monkeypatch, seq_factory):
    now = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    connector = scheduler_connector(monkeypatch, True)
    sources = []
    for transfer_id in ("oldest", "next", "newest"):
        seq = seq_factory(
            list(range(10)),
            kv_transfer_params={
                "chunked_transfer": True,
                "transfer_id": transfer_id,
                "do_remote_decode": True,
            },
        )
        seq.block_table = [1, 2, 3]
        connector.update_state_after_alloc(seq)
        seq.output_tokens = [42]
        connector.request_finished(seq)
        sources.append(seq)
    connector.build_connector_meta()
    head = seq_factory(list(range(10)))
    connector.admission_blocked(head)
    now[0] += mc.CHUNKED_RECLAIM_STALL_SECONDS / 2
    connector.admission_blocked(head)
    assert not connector.build_connector_meta().has_work()
    now[0] += mc.CHUNKED_RECLAIM_STALL_SECONDS / 2
    connector.admission_blocked(head)
    meta = connector.build_connector_meta()
    assert meta.reclaim_unclaimed_prefills == {sources[0].id: "oldest"}
    assert meta.has_work()
    now[0] += 5
    connector.admission_blocked(head)
    assert not connector.build_connector_meta().has_work()

    # A reader raced the oldest candidate. Only the complete TP/PP rejection
    # permits trying the next source; partial reports cannot authorize it.
    tp = [KVOutputAggregator(2), KVOutputAggregator(2)]
    pp = PPKVAggregator(2)
    done = KVConnectorOutput(
        connector_completions={
            ConnectorCompletion("pd_reclaim_done", sources[0].id, False)
        }
    )
    for rank in range(2):
        assert tp[rank].aggregate([done, KVConnectorOutput()]).is_empty()
        out = pp.ingest(rank, tp[rank].aggregate([KVConnectorOutput(), done]))
        connector.process_pd_completions(out)
        connector.admission_blocked(head)
        assert not connector.build_connector_meta().has_work()
    now[0] += mc.CHUNKED_RECLAIM_STALL_SECONDS
    connector.admission_blocked(head)
    assert connector.build_connector_meta().reclaim_unclaimed_prefills == {
        sources[1].id: "next"
    }

    # A successful cancellation acknowledgement alone does not release HBM.
    connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_reclaim_done", sources[1].id, True)
            }
        )
    )
    now[0] += 5
    connector.admission_blocked(head)
    assert not connector.build_connector_meta().has_work()
    terminal = connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_source_safe", sources[1].id, True)
            }
        )
    )
    assert terminal.finished_sending == {sources[1].id}
    connector.send_finished(sources[1].id)
    connector.admission_blocked(head)
    assert not connector.build_connector_meta().has_work()
    # Allocation succeeds after release: it resets pressure, so ordinary
    # progress cannot immediately sacrifice the last completed prefill.
    connector.update_state_after_alloc(head)
    now[0] += 5
    connector.admission_blocked(head)
    assert not connector.build_connector_meta().has_work()


def test_reclaim_targets_same_source_on_pp_ranks_with_different_claims():
    workers = [producer(rank, 2)[0] for rank in range(2)]
    states = [worker._chunked_prefills["xfer-a"] for worker in workers]
    for state in states:
        publish(state, 10)
        state.finish({"first_token_id": 42})
    states[0].acquire(("reader",), 1)
    pressure = ConnectorMetadata()
    pressure.reclaim_unclaimed_prefills = {7: "xfer-a"}
    pp = PPKVAggregator(2)
    for rank, worker in enumerate(workers):
        worker.start_load_kv(pressure)
        result = pp.ingest(rank, worker.get_finished())
    assert states[1].cancelled and not states[0].cancelled
    assert result.connector_completions == {
        ConnectorCompletion("pd_reclaim_done", 7, False)
    }
    states[0].release()
    result = pp.ingest(0, workers[0].get_finished())
    assert result.connector_completions == {
        ConnectorCompletion("pd_source_safe", 7, True)
    }


def test_reclaim_does_not_cancel_other_finished_sources():
    worker, state, _, _, _ = producer()
    other = ChunkedPrefill(8, [1], 4, 4)
    publish(state, 10)
    state.finish({"first_token_id": 42})
    publish(other, 4)
    other.finish({"first_token_id": 43})
    worker._chunked_prefills["xfer-b"] = other
    pressure = ConnectorMetadata()
    pressure.reclaim_unclaimed_prefills = {7: "xfer-a"}
    worker.start_load_kv(pressure)
    assert state.cancelled and not other.cancelled


@pytest.mark.parametrize("chunked", [False, True])
def test_failed_receive_cleanup_is_limited_to_chunked_requests(chunked, seq_factory):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    sched = Scheduler(MockConfig(num_kvcache_blocks=8))
    seq = seq_factory(
        list(range(8)),
        kv_transfer_params={
            "chunked_transfer": chunked,
            "first_token_id": 42,
            "draft_token_ids": [43],
        },
    )
    assert sched.block_manager.allocate(seq)
    blocks = list(seq.block_table)
    sched.failed_recving_kv_req_ids.append(seq.id)
    assert sched._consume_failed_remote_kv(seq)
    if chunked:
        assert not seq.block_table
        assert not seq.kv_transfer_params
    else:
        assert list(seq.block_table) == blocks
        assert seq.kv_transfer_params["first_token_id"] == 42
        assert seq.kv_transfer_params["draft_token_ids"] == [43]
