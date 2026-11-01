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
    c._admission_results = set()
    c._chunked_admissions = {}
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
        "consumer_tp_rank": 0,
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
    c.is_producer = False
    c._cuda_device = 0
    # Drive the real receive queue by hand; CPU tests have no GPU worker.
    c._write_queue = []
    c._write_cv = threading.Condition()
    c._write_closed = False
    c._write_thread = object()
    c._write_request_error = None
    c._verified_producers = set()
    c._has_slot_regions = False
    c._staging_slots = mc.SlotPool()
    c.pp_size = 1
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


def post_write_requests(connector, monkeypatch):
    """Run one real write-queue round with only the GPU wait replaced."""
    synchronize = MagicMock()
    with monkeypatch.context() as patch:
        patch.setattr(mc.torch.cuda, "synchronize", synchronize)
        assert connector._post_queued_write_requests()
    synchronize.assert_called_once_with(connector._cuda_device)
    assert connector._write_request_error is None


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


def scheduler_connector(
    monkeypatch, is_producer, pp_size=1, dcp_size=1, admission_timeout_s=300
):
    monkeypatch.setattr(mc, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mc, "get_open_port", lambda: 6400)
    return mc.MooncakeConnectorScheduler(
        SimpleNamespace(
            kv_transfer_config={
                "kv_role": "kv_producer" if is_producer else "kv_consumer",
                "enable_chunked_transfer": True,
                "chunked_admission_timeout_s": admission_timeout_s,
            },
            tensor_parallel_size=1,
            pipeline_parallel_size=pp_size,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            kv_cache_block_size=4,
            decode_context_parallel_size=dcp_size,
        )
    )


@pytest.mark.parametrize("pp_size", [1, 3])
@pytest.mark.parametrize("prompt_tokens", [10, 11, 12])
@pytest.mark.parametrize("draft_tokens", [0, 3])
def test_real_scheduler_pins_from_first_chunk_through_final_handoff(
    monkeypatch, seq_factory, pp_size, prompt_tokens, draft_tokens
):
    import numpy as np
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
    from atom.sampling_params import SamplingParams

    sched = Scheduler(
        MockConfig(
            max_num_batched_tokens=4,
            pipeline_parallel_size=pp_size,
            speculative_config=(
                SimpleNamespace(
                    num_speculative_tokens=draft_tokens, use_dspark=lambda: False
                )
                if draft_tokens
                else None
            ),
        )
    )
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True, pp_size)
    seq = seq_factory(
        list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        num_draft_tokens=draft_tokens,
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode": True,
            "transfer_id": "xfer-a",
        },
    )
    workers = [idle_producer(stage, pp_size) for stage in range(pp_size)]
    sched.add(seq)
    assert not connector.admission_ready(seq)
    connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_admission_ready", seq.id, True)
            }
        )
    )
    batches = []
    allocated_blocks = (prompt_tokens + max(0, draft_tokens - 1) + 3) // 4
    for _ in range(3):
        batch, seqs = sched.schedule()
        batches.append(batch)
        assert not sched._is_preemptable(seq)
        # The draft round still owns its lookahead, even though D reserves
        # only the prompt and must never receive the extra draft-only page.
        assert len(seq.block_table) == allocated_blocks
        for worker, _, _, _ in workers:
            worker.start_load_kv(batch.connector_meta_output)
            state = worker._chunked_prefills["xfer-a"]
            assert list(state.block_ids) == list(seq.block_table[:3])
            publish(state, int(batch.context_lens[0]))
        result = ScheduledBatchOutput(
            [seq.id],
            [(42, 43, 44, 45) if sched.spec_decode_local else (42,)],
            np.array([0]),
            np.array([0]),
            np.array([[43, 44, 45]]) if draft_tokens else None,
        )
        sched.postprocess(list(seqs.values()), result, batch=batch)
    assert [b.context_lens.tolist() for b in batches] == [[4], [8], [prompt_tokens]]
    # Earlier snapshots must not follow the live seq past the next PP forward.
    assert batches[0].context_lens.tolist() == [4]
    initial = batches[0].connector_meta_output.reqs_to_save[seq.id]
    assert initial.chunked_transfer and initial.num_prompt_tokens == prompt_tokens
    assert initial.prefill_handoff is None
    assert initial.transfer_id == "xfer-a"
    final_meta = connector.build_connector_meta()
    final = final_meta.reqs_to_save[seq.id]
    assert final.prefill_handoff["first_token_id"] == 42
    assert final.prefill_handoff["draft_token_ids"] == (
        [43, 44, 45] if draft_tokens else []
    )
    assert final.local_block_ids == initial.local_block_ids == list(seq.block_table[:3])
    assert connector.should_defer_free(seq)
    assert len(seq.block_table) == allocated_blocks
    pp = PPKVAggregator(pp_size)
    for worker, request, src, destinations in workers:
        worker.start_load_kv(final_meta)
        request["prompt_digest"] = seq.kv_transfer_params["prompt_digest"]
        dispatch(worker, request)
        worker._notify_transfer_result.assert_called_once_with(request, success=True)
        sources, _, sizes = worker.transfer_engine.writes.get_nowait()
        assert sources == [
            ctypes.addressof(src) + block * 4 for block in seq.block_table[:3]
        ]
        assert sum(sizes) == 12
        assert worker.transfer_engine.writes.empty()
        assert bytes(destinations[worker.pp_rank][8:12]) == bytes(src[:4])
        sched._update_from_kv_xfer_finished(
            pp.ingest(worker.pp_rank, worker.get_finished())
        )
        if worker.pp_rank < pp_size - 1:
            assert connector.should_defer_free(seq)
            assert len(seq.block_table) == allocated_blocks
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
def test_swa_only_pool_transfers_final_relocated_slot(monkeypatch, reverse_indexed):
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
    d._send_on_socket.assert_not_called()
    post_write_requests(d, monkeypatch)
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
    c._send_on_socket.assert_not_called()
    post_write_requests(c, monkeypatch)
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
    p._notify_transfer_result.assert_called_once_with(request, success=False)
    assert request["transfer_id"] in p._retired_chunked_transfers
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
    assert not connector.admission_ready(seq)
    connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_admission_ready", seq.id, True)
            }
        )
    )
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


def test_admission_signal_survives_composite_metadata(monkeypatch, seq_factory):
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
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [connector]
    assert not multi.admission_ready(seq)
    meta = multi.build_connector_meta()
    assert meta.has_work() and meta.metas[0].chunked_admissions == {seq.id: "xfer-a"}
    assert not multi.build_connector_meta().has_work()
    multi.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_admission_ready", seq.id, True)
            }
        )
    )
    assert multi.admission_ready(seq)


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


def idle_producer(pp_rank=0, pp_size=1):
    worker, _, request, src, dst = producer(pp_rank, pp_size)
    worker._chunked_prefills.clear()
    worker._chunked_local_ids.clear()
    return worker, request, src, dst


def decode_worker(dst):
    worker, _, _, _, _ = producer()
    worker.__dict__.update(consumer(1).__dict__)
    worker._pending_recv_expected.clear()
    worker._pending_recv_nonce.clear()
    worker._pending_recv.clear()
    worker._pending_recv_blocks.clear()
    worker._pending_handoffs.clear()
    worker._chunked_prefills.clear()
    worker._chunked_local_ids.clear()
    worker.is_producer = False
    worker.tp_rank = worker.dp_rank = worker.dcp_rank = 0
    worker.dcp_interleave_size = 1
    worker.local_ip = "cpu"
    worker.rpc_port = 1
    worker.ib_device = None
    worker._notification_port = 2
    worker._has_slot_regions = False
    worker._block_regions = [(ctypes.addressof(dst), 4)]
    worker.kv_caches_base_addr = [ctypes.addressof(dst)]
    worker._send_on_socket = MagicMock()
    return worker


def pd_sequence(seq_factory, transfer_id="xfer-a", producer=True):
    from atom.sampling_params import SamplingParams

    return seq_factory(
        list(range(10)),
        sampling_params=SamplingParams(max_tokens=1 if producer else 2),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode" if producer else "do_remote_prefill": True,
            "transfer_id": transfer_id,
            "block_size": 4,
            "dcp_size": 1,
            "remote_host": "cpu",
            "remote_handshake_port": 6301,
            "remote_tp_size": 1,
        },
    )


def test_opposite_admission_order_progresses_without_recompute(
    monkeypatch, seq_factory
):
    import msgpack
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler

    p = Scheduler(MockConfig(num_kvcache_blocks=3, max_num_batched_tokens=4))
    d = Scheduler(MockConfig(num_kvcache_blocks=3, max_num_batched_tokens=4))
    p.kv_connector = scheduler_connector(monkeypatch, True)
    d.kv_connector = scheduler_connector(monkeypatch, False)
    pa, pb = [pd_sequence(seq_factory, name) for name in ("xfer-a", "xfer-b")]
    da, db = [pd_sequence(seq_factory, name, False) for name in ("xfer-a", "xfer-b")]
    p.extend([pa, pb])
    d.extend([db, da])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])

    def receive(_addr, message):
        assert message[0] == mc.MSG_WRITE_REQUEST
        dispatch(pw, msgpack.loads(message[1]))

    def complete(data, success):
        assert success, "capacity must not cause transfer failure or D recomputation"
        dw._record_write_done(
            data["request_id"],
            0,
            0,
            data["write_nonce"],
            handoff=data["prefill_handoff"],
        )

    dw._send_on_socket.side_effect = receive
    pw._notify_transfer_result.side_effect = complete
    pw._ack_write_request = MagicMock(
        side_effect=lambda data: dw._record_write_ack(
            {
                "request_id": data["request_id"],
                "write_nonce": data["write_nonce"],
                "pp_rank": 0,
                "tp_rank": 0,
                "write_window_s": mc._write_window_s(),
            }
        )
    )
    batch, seqs = p.schedule()
    assert not seqs and not pa.block_table and not pb.block_table
    assert p.block_manager.kv.num_free == 3
    pw.start_load_kv(batch.connector_meta_output)
    assert pw.get_finished().is_empty()

    for ps, ds in ((pb, db), (pa, da)):
        batch, seqs = d.schedule()
        assert not seqs
        # A 4-token batch budget still reserves all 10 prompt tokens on D.
        assert len(ds.block_table) == 3 and d.block_manager.kv.num_free == 0
        dw.start_load_kv(batch.connector_meta_output)
        # Reserving D's pages alone cannot admit P: the write request still
        # waits in D's queue until prior GPU writes to those pages finish.
        assert dw._write_queue
        assert pw.get_finished().is_empty()
        assert not ps.block_table
        post_write_requests(dw, monkeypatch)
        p._update_from_kv_xfer_finished(pw.get_finished())
        batch, seqs = p.schedule()
        assert list(seqs) == [ps.id]
        assert batch.context_lens.tolist() == [4]
        pw.start_load_kv(batch.connector_meta_output)
        state = pw._chunked_prefills[ps.kv_transfer_params["transfer_id"]]
        # Only real completed forwards publish pages. Source stays pinned.
        with ThreadPoolExecutor(1) as pool:
            pw._send_executor = pool
            for end in (4, 8, 10):
                assert ps.block_table and ds.block_table
                publish(state, end)
                pw._dispatch_ready_chunked_requests()
                assert pw.transfer_engine.writes.get(timeout=5)[2] == [4]
                p.postprocess(
                    list(seqs.values()),
                    ScheduledBatchOutput([ps.id], [(42,)], None, None, None),
                    batch=batch,
                )
                pw.start_load_kv(p.kv_connector.build_connector_meta())
                if end != 10:
                    batch, seqs = p.schedule()
                    assert batch.context_lens.tolist() == [min(end + 4, 10)]
        pw._send_executor = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
        p._update_from_kv_xfer_finished(pw.get_finished())
        assert not ps.block_table and p.block_manager.kv.num_free == 3
        d._update_from_kv_xfer_finished(dw.get_finished())
        batch, seqs = d.schedule()
        assert list(seqs) == [ds.id]
        assert batch.total_seqs_num_decode == 1 and batch.total_seqs_num_prefill == 0
        assert d.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput([ds.id], [(43,)], None, None, None),
            batch=batch,
        ) == [ds]
    assert p.block_manager.kv.num_free == d.block_manager.kv.num_free == 3
    assert not p.deferred_free_blocks and not d.deferred_free_blocks
    assert not pw._pending_chunked_requests and not pw._chunked_local_ids
    assert not p.kv_connector._chunked_waiting and not p.kv_connector._chunked_ready
    assert pw._notify_transfer_result.call_count == 2


@pytest.mark.parametrize("early", [False, True])
def test_admission_requires_all_tp_dcp_and_pp_ranks(monkeypatch, seq_factory, early):
    connector = scheduler_connector(monkeypatch, True, 2)
    seq = pd_sequence(seq_factory)
    assert not connector.admission_ready(seq)
    registration = connector.build_connector_meta()
    tp = [KVOutputAggregator(2) for _ in range(2)]
    pp = PPKVAggregator(2)
    workers = []
    for stage in range(2):
        row = []
        for rank in range(2):
            worker, request, _, _ = idle_producer(stage, 2)
            worker.tp_size, worker.tp_rank = 2, rank
            request.update(consumer_tp_size=4, consumer_tp_rank=rank)
            if not early:
                worker.start_load_kv(registration)
            dispatch(worker, request)
            dispatch(worker, request.copy())  # Duplicates cannot fill another rank.
            if early:
                worker.start_load_kv(registration)
            assert worker.get_finished().is_empty()
            worker.start_load_kv(registration)  # Repeated scheduler snapshot.
            row.append((worker, request))
        workers.append(row)
    for stage in range(2):
        for rank in range(2):
            worker, request = workers[stage][rank]
            dispatch(
                worker,
                dict(
                    request,
                    consumer_tp_rank=rank + 2,
                    notify_port=rank + 10,
                    write_nonce=100 + rank,
                ),
            )
            out = worker.get_finished()
            assert out.connector_completions == {
                ConnectorCompletion("pd_admission_ready", seq.id, True)
            }
            outputs = [KVConnectorOutput(), KVConnectorOutput()]
            outputs[rank] = out
            connector.process_pd_completions(
                pp.ingest(stage, tp[stage].aggregate(outputs))
            )
            assert connector.admission_ready(seq) is (stage == 1 and rank == 1)
            assert worker.get_finished().is_empty()
            assert not worker._chunked_prefills  # Readiness itself allocates nothing.


def test_capacity_wait_does_not_expire_ready_or_queued_requests(
    monkeypatch, seq_factory
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    p = Scheduler(MockConfig(num_kvcache_blocks=3))
    p.kv_connector = scheduler_connector(monkeypatch, True)
    occupied = seq_factory(list(range(10)))
    assert p.block_manager.allocate(occupied)
    seqs = [pd_sequence(seq_factory, f"wait-{i}") for i in range(3)]
    p.extend(seqs)
    assert p._arrived_requests == len(seqs)
    worker, request, _src, _dst = idle_producer()
    worker._ack_write_request = MagicMock()
    # Registration includes requests behind a blocked, ready head.
    batch, scheduled = p.schedule()
    worker.start_load_kv(batch.connector_meta_output)
    for seq in seqs:
        dispatch(
            worker,
            dict(
                request,
                transfer_id=seq.kv_transfer_params["transfer_id"],
                write_ack=True,
            ),
        )
    p._update_from_kv_xfer_finished(worker.get_finished())
    # Beyond the orphan lookup timeout, but within the admission deadline.
    now = mc.time.monotonic()
    monkeypatch.setattr(mc.time, "monotonic", lambda: now + 120)
    for _ in range(2):
        batch, scheduled = p.schedule()
        assert not scheduled and not batch.connector_meta_output.has_work()
        assert worker.get_finished().is_empty()
    worker._ack_write_request.assert_not_called()
    worker._notify_transfer_result.assert_not_called()
    assert len(worker._pending_chunked_requests) == 3
    assert all(not seq.block_table for seq in seqs)
    p.block_manager.deallocate(occupied)
    batch, scheduled = p.schedule()
    assert list(scheduled) == [seqs[0].id]
    worker.start_load_kv(batch.connector_meta_output)
    state = worker._chunked_prefills[seqs[0].kv_transfer_params["transfer_id"]]
    publish(state, 10)
    state.finish({"first_token_id": 42})
    worker._pending_chunked_requests[0][0]["prompt_digest"] = state.prompt_digest
    worker._dispatch_ready_chunked_requests()
    worker._ack_write_request.assert_called_once()
    assert (
        worker._ack_write_request.call_args.args[0]["_deadline"]
        == now + 120 + mc.PD_TRANSFER_DEADLINE_S
    )
    assert len(worker._pending_chunked_requests) == 2


@pytest.mark.parametrize("ready_before_cancel", [False, True])
def test_waiting_cancel_and_late_ready_do_not_resurrect_request(
    monkeypatch, seq_factory, ready_before_cancel
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler
    from atom.model_engine.sequence import SequenceStatus

    p = Scheduler(MockConfig())
    p.kv_connector = scheduler_connector(monkeypatch, True)
    seq = pd_sequence(seq_factory)
    p.add(seq)
    batch, scheduled = p.schedule()
    worker, request, _src, _dst = idle_producer()
    worker.start_load_kv(batch.connector_meta_output)
    if ready_before_cancel:
        dispatch(worker, request)
    report = worker.get_finished()
    assert p.abort_request(seq.id)
    assert not seq.block_table and not p.deferred_free_blocks
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    p._update_from_kv_xfer_finished(report)
    dispatch(worker, request.copy())
    p._update_from_kv_xfer_finished(worker.get_finished())
    assert not p.kv_connector._chunked_waiting and not p.kv_connector._chunked_ready
    assert not worker._chunked_admissions and not worker._chunked_local_ids
    assert not worker._chunked_prefills and not worker._pending_chunked_requests
    assert worker.transfer_engine.writes.empty()
    # A fresh scheduler ID with the same retired transfer ID is rejected.
    reused = pd_sequence(seq_factory)
    p.add(reused)
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    p._update_from_kv_xfer_finished(worker.get_finished())
    assert reused.status == SequenceStatus.ABORTED
    _, scheduled = p.schedule()
    assert not scheduled and not reused.block_table


def test_reusing_live_transfer_id_cannot_cancel_original(monkeypatch, seq_factory):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler
    from atom.model_engine.sequence import SequenceStatus

    p = Scheduler(MockConfig())
    p.kv_connector = scheduler_connector(monkeypatch, True)
    original, reused = pd_sequence(seq_factory), pd_sequence(seq_factory)
    p.extend([original, reused])
    worker, request, _src, _dst = idle_producer()
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    p._update_from_kv_xfer_finished(worker.get_finished())
    assert reused.status == SequenceStatus.ABORTED
    _, scheduled = p.schedule()
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    assert "xfer-a" not in worker._retired_chunked_transfers
    dispatch(worker, request)
    p._update_from_kv_xfer_finished(worker.get_finished())
    _, scheduled = p.schedule()
    assert list(scheduled) == [original.id]


@pytest.mark.parametrize("cancelled", [False, True])
def test_d_transfer_failure_waits_for_write_hold_before_abort(
    monkeypatch, seq_factory, cancelled
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "30")
    now = [mc.time.monotonic()]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    d = Scheduler(MockConfig(num_kvcache_blocks=3))
    d.kv_connector = scheduler_connector(monkeypatch, False)
    seq = pd_sequence(seq_factory, producer=False)
    d.add(seq)
    batch, scheduled = d.schedule()
    assert not scheduled and len(seq.block_table) == 3
    dst = (ctypes.c_ubyte * 48)()
    worker = decode_worker(dst)
    worker.start_load_kv(batch.connector_meta_output)
    post_write_requests(worker, monkeypatch)
    deallocate = MagicMock(wraps=d.block_manager.deallocate)
    monkeypatch.setattr(d.block_manager, "deallocate", deallocate)
    if cancelled:
        assert d.abort_request(seq.id)
    worker._record_write_done(
        seq.id, 0, 0, worker._pending_recv_nonce[seq.id], success=False
    )
    now[0] += 29
    d._update_from_kv_xfer_finished(worker.get_finished())
    _, scheduled = d.schedule()
    assert not scheduled and seq.block_table
    assert d.block_manager.kv.num_free == 0
    deallocate.assert_not_called()

    now[0] += 2
    d._update_from_kv_xfer_finished(worker.get_finished())
    assert seq.leave_reason == "aborted"
    assert not seq.block_table and d.block_manager.kv.num_free == 3
    deallocate.assert_called_once_with(seq)
    assert not d.waiting and not d.running and not d.deferred_free_blocks
    batch, scheduled = d.schedule()
    assert not scheduled and batch.total_seqs_num_prefill == 0
    assert not worker._pending_recv and not worker._held_failures
    assert worker.transfer_engine.writes.empty()

    # A later request can reserve the reclaimed pool and decode a new handoff.
    following = pd_sequence(seq_factory, "after-failure", producer=False)
    d.add(following)
    batch, scheduled = d.schedule()
    assert not scheduled and len(following.block_table) == 3
    following.kv_transfer_params["first_token_id"] = 44
    d._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={following.id}))
    batch, scheduled = d.schedule()
    assert list(scheduled) == [following.id]
    assert batch.total_seqs_num_prefill == 0 and batch.total_seqs_num_decode == 1


@pytest.mark.parametrize("producer_side", [False, True])
def test_legacy_requests_bypass_admission_gate(monkeypatch, seq_factory, producer_side):
    connector = scheduler_connector(monkeypatch, producer_side)
    seq = seq_factory([1, 2, 3], kv_transfer_params={"do_remote_decode": producer_side})
    assert connector.admission_ready(seq)
    assert not connector.build_connector_meta().has_work()


@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_d_reserves_full_prompt_without_max_new_tokens(
    monkeypatch, seq_factory, dcp_size
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    blocks = (10 + 4 * dcp_size - 1) // (4 * dcp_size)
    d = Scheduler(
        MockConfig(
            num_kvcache_blocks=blocks,
            max_num_batched_tokens=4,
            decode_context_parallel_size=dcp_size,
        )
    )
    if dcp_size > 1:
        bm = d.block_manager
        # Isolate this reservation test from GPU-only dcp_ops imports. With
        # rank 0 and token interleaving, this is the equivalent pool count;
        # admission and allocation still run through the real scheduler.
        monkeypatch.setattr(
            bm,
            "num_pool_blocks",
            lambda seq_len: (seq_len + bm.hash_block_size - 1) // bm.hash_block_size,
        )
    d.kv_connector = scheduler_connector(monkeypatch, False, dcp_size=dcp_size)
    seq = pd_sequence(seq_factory, producer=False)
    seq.max_tokens = 32
    d.add(seq)
    batch, scheduled = d.schedule()
    assert not scheduled and len(seq.block_table) == blocks
    assert batch.connector_meta_output.reqs_to_recv[seq.id].local_block_ids == list(
        seq.block_table
    )
    assert d.block_manager.kv.num_free == 0


@pytest.mark.parametrize("remote_tp,d_tp,d_pp", [(2, 1, 1), (3, 4, 1), (1, 1, 2)])
def test_unroutable_chunked_topology_fails_before_any_write_request(
    remote_tp, d_tp, d_pp
):
    dst = (ctypes.c_ubyte * 48)()
    d = decode_worker(dst)
    d.tp_size, d.pp_size = d_tp, d_pp
    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(
        21,
        [1, 2, 3],
        {
            "chunked_transfer": True,
            "remote_tp_size": remote_tp,
        },
    )
    d.start_load_kv(meta)
    d._send_on_socket.assert_not_called()
    assert d.get_finished().failed_recving == {21}


def test_partial_ready_cancel_drains_tp_pp_quorum(monkeypatch, seq_factory):
    connector = scheduler_connector(monkeypatch, True, 2)
    seq = pd_sequence(seq_factory)
    connector.admission_ready(seq)
    registration = connector.build_connector_meta()
    workers = [idle_producer(stage, 2)[0] for stage in range(2)]
    pp = PPKVAggregator(2)
    for worker in workers:
        worker.start_load_kv(registration)
    _, request, _, _ = idle_producer()
    dispatch(workers[0], request)
    assert pp.ingest(0, workers[0].get_finished()).is_empty()
    seq.leave_reason = "aborted"
    connector.request_finished(seq)
    cancel = connector.build_connector_meta()
    for worker in workers:
        worker.start_load_kv(cancel)
    out = pp.ingest(1, workers[1].get_finished())
    assert out.connector_completions == {
        ConnectorCompletion("pd_admission_ready", seq.id, False)
    }
    connector.process_pd_completions(out)
    assert not pp.has_pending() and not connector._chunked_ready
    assert all(not w._chunked_local_ids and not w._chunked_admissions for w in workers)


def test_source_safe_still_waits_for_other_multi_connector_owner(
    monkeypatch, seq_factory
):
    from conftest import MockConfig

    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )
    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler

    p = Scheduler(MockConfig())
    connector = p.kv_connector = scheduler_connector(monkeypatch, True)
    seq = pd_sequence(seq_factory)
    p.add(seq)
    connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_admission_ready", seq.id, True),
            }
        )
    )
    batch, seqs = p.schedule()
    p.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput([seq.id], [(42,)], None, None, None),
        batch=batch,
    )
    # Model LMCache's independent source ownership after prefill has finished.
    owner = SimpleNamespace(
        should_defer_free=lambda _: True, process_pd_completions=lambda output: output
    )
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [connector, owner]
    multi.is_producer, multi.is_offload = True, False
    multi._load_winner = {}
    p.kv_connector = multi
    p._update_from_kv_xfer_finished(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_source_safe", seq.id, True),
            }
        )
    )
    assert not connector.should_defer_free(seq)
    assert seq.block_table and seq.id in p.deferred_free_blocks
    owner.should_defer_free = lambda _: False
    p._maybe_release_deferred(seq)
    assert not seq.block_table and not p.deferred_free_blocks


def test_admitted_source_can_resume_offload_load_without_another_ready(
    monkeypatch, seq_factory
):
    connector = scheduler_connector(monkeypatch, True)
    seq = pd_sequence(seq_factory)
    assert not connector.admission_ready(seq)
    connector.build_connector_meta()
    connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_admission_ready", seq.id, True),
            }
        )
    )
    seq.block_table = [1, 2, 3]
    connector.update_state_after_alloc(seq)
    connector.build_connector_meta()
    assert connector.admission_ready(seq)
    assert not connector.build_connector_meta().has_work()


@pytest.mark.parametrize("received_metadata", [False, True])
def test_last_unadmitted_cancel_reaches_worker_despite_idle_throttle(
    monkeypatch, seq_factory, received_metadata
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    with stubbed_aiter():
        from atom.model_engine.engine_core import EngineCore

    p = Scheduler(MockConfig())
    p.kv_connector = scheduler_connector(monkeypatch, True)
    seq = pd_sequence(seq_factory)
    p.add(seq)
    worker, request, _src, _dst = idle_producer()
    batch, _ = p.schedule()
    worker.start_load_kv(batch.connector_meta_output)
    if received_metadata:
        dispatch(worker, request)
    assert p.abort_request(seq.id)
    core = EngineCore.__new__(EngineCore)
    core.scheduler = p
    core.kv_transfer_enabled = True
    core.label = "pd-cancel"
    core._next_idle_kv_drain = mc.time.monotonic() + 1000
    core.output_queue = queue.Queue()
    core.runner_mgr = MagicMock()
    core.runner_mgr.call_func.side_effect = lambda method, meta: worker.start_load_kv(
        meta
    )
    core.runner_mgr.call_func_with_aggregation.side_effect = (
        lambda _: worker.get_finished()
    )
    core._process_engine_step_inner()
    assert p.is_finished() and not seq.block_table
    assert not p.kv_connector._reqs_need_save
    assert not worker._chunked_admissions
    assert not worker._chunked_local_ids and not worker._pending_chunked_requests


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_chunked_admission_timeout_must_be_finite_and_positive(monkeypatch, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        scheduler_connector(monkeypatch, True, admission_timeout_s=timeout)


@pytest.mark.parametrize("pp_size", [1, 3])
@pytest.mark.parametrize("idle_sweep", [False, True])
def test_admission_timeout_without_decode_retires_all_stages(
    monkeypatch, seq_factory, pp_size, idle_sweep
):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler
    from atom.model_engine.sequence import SequenceStatus

    now = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    p = Scheduler(MockConfig(pipeline_parallel_size=pp_size))
    p.kv_connector = scheduler_connector(
        monkeypatch, True, pp_size, admission_timeout_s=10
    )
    seq = pd_sequence(seq_factory)
    p.add(seq)
    registration = p.kv_connector.build_connector_meta()
    workers = [idle_producer(rank, pp_size)[0] for rank in range(pp_size)]
    for worker in workers:
        worker.start_load_kv(registration)
    now[0] = 109.0
    assert not p.kv_connector.admission_ready(seq)
    assert not p.kv_connector.build_connector_meta().has_work()
    now[0] = 110.0
    if not idle_sweep:
        assert not p.kv_connector.admission_ready(seq)
    cancellation = p.kv_connector.build_connector_meta()
    assert seq.status == SequenceStatus.ABORTED
    assert cancellation.reqs_to_save[seq.id].prefill_aborted
    assert not cancellation.chunked_admissions
    assert not seq.block_table and not p.deferred_free_blocks
    assert not p.kv_connector.admission_ready(seq)
    pp = PPKVAggregator(pp_size)
    for rank, worker in enumerate(workers):
        worker.start_load_kv(cancellation)
        # A delayed ready notification and delayed D write request cannot
        # allocate or write a source after the scheduler's deadline.
        p._update_from_kv_xfer_finished(
            pp.ingest(
                rank,
                KVConnectorOutput(
                    connector_completions={
                        ConnectorCompletion("pd_admission_ready", seq.id, True)
                    }
                ),
            )
        )
        _, request, _, _ = idle_producer(rank, pp_size)
        dispatch(worker, request)
        worker.get_finished()
        assert not worker._chunked_admissions and not worker._chunked_local_ids
        assert not worker._chunked_prefills and not worker._pending_chunked_requests
        assert worker.transfer_engine.writes.empty()
    assert not p.kv_connector._chunked_waiting and not p.kv_connector._chunked_ready
    _, scheduled = p.schedule()
    assert not scheduled and not p.waiting and not seq.block_table
    assert seq.leave_reason == "aborted"


@pytest.mark.parametrize("pp_size", [1, 3])
@pytest.mark.parametrize("cancel_decode", [False, True])
def test_ready_but_capacity_blocked_timeout_preserves_decode_write_safety(
    monkeypatch, seq_factory, pp_size, cancel_decode
):
    import msgpack
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "30")
    now = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    p = Scheduler(MockConfig(num_kvcache_blocks=3, pipeline_parallel_size=pp_size))
    d = Scheduler(MockConfig(num_kvcache_blocks=3))
    p.kv_connector = scheduler_connector(
        monkeypatch, True, pp_size, admission_timeout_s=10
    )
    d.kv_connector = scheduler_connector(monkeypatch, False)
    occupied = seq_factory(list(range(10)))
    assert p.block_manager.allocate(occupied)
    ps = [pd_sequence(seq_factory, f"wait-{i}") for i in range(3)]
    ds = pd_sequence(seq_factory, "wait-0", False)
    ds.kv_transfer_params["remote_pp_size"] = pp_size
    p.extend(ps)
    d.add(ds)
    workers = [idle_producer(rank, pp_size)[0] for rank in range(pp_size)]
    dw = decode_worker((ctypes.c_ubyte * 48)())
    registration = p.kv_connector.build_connector_meta()
    for rank, worker in enumerate(workers):
        worker.start_load_kv(registration)
        worker._notify_transfer_result.side_effect = lambda data, success, rank=rank: (
            dw._record_write_done(
                data["request_id"], rank, 0, data["write_nonce"], success=success
            )
        )
    dw._send_on_socket.side_effect = lambda _addr, message: [
        dispatch(worker, msgpack.loads(message[1])) for worker in workers
    ]
    batch, scheduled = d.schedule()
    assert not scheduled and len(ds.block_table) == 3
    dw.start_load_kv(batch.connector_meta_output)
    post_write_requests(dw, monkeypatch)
    pp = PPKVAggregator(pp_size)
    for rank, worker in enumerate(workers):
        p._update_from_kv_xfer_finished(pp.ingest(rank, worker.get_finished()))
    assert p.kv_connector.admission_ready(ps[0])
    now[0] = 109.0
    # Repeated readiness does not refresh the original admission deadline.
    p._update_from_kv_xfer_finished(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_admission_ready", ps[0].id, True)
            }
        )
    )
    _, scheduled = p.schedule()
    assert not scheduled and all(not seq.block_table for seq in ps)
    now[0] = 110.0
    cancellation = p.kv_connector.build_connector_meta()
    assert set(cancellation.reqs_to_save) == {seq.id for seq in ps}
    if cancel_decode:
        assert d.abort_request(ds.id)
    for rank, worker in enumerate(workers):
        worker.start_load_kv(cancellation)
        worker.get_finished()
        assert worker.transfer_engine.writes.empty()
        d._update_from_kv_xfer_finished(dw.get_finished())
        assert ds.block_table and d.block_manager.kv.num_free == 0
    now[0] = 139.0
    d._update_from_kv_xfer_finished(dw.get_finished())
    assert ds.block_table and d.block_manager.kv.num_free == 0
    now[0] = 140.0
    d._update_from_kv_xfer_finished(dw.get_finished())
    assert not ds.block_table and d.block_manager.kv.num_free == 3
    assert ds.leave_reason == "aborted"
    batch, scheduled = d.schedule()
    assert not scheduled and batch.total_seqs_num_prefill == 0
    assert not d.waiting and not d.running and not d.deferred_free_blocks
    assert not dw._pending_recv and not dw._pending_recv_expected
    assert not dw._held_failures
    assert not p.kv_connector._chunked_waiting and not p.kv_connector._chunked_ready
    _, scheduled = p.schedule()
    assert not scheduled and not p.waiting and not p.deferred_free_blocks
    assert occupied.block_table  # The timeout cannot free another source owner.


def test_admission_deadline_stops_at_source_allocation(monkeypatch, seq_factory):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    now = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    p = Scheduler(MockConfig(max_num_batched_tokens=4))
    p.kv_connector = scheduler_connector(monkeypatch, True, admission_timeout_s=10)
    seq = pd_sequence(seq_factory)
    p.add(seq)
    worker, request, _, _ = idle_producer()
    worker.start_load_kv(p.kv_connector.build_connector_meta())
    dispatch(worker, request)
    p._update_from_kv_xfer_finished(worker.get_finished())
    batch, scheduled = p.schedule()
    assert list(scheduled) == [seq.id]
    worker.start_load_kv(batch.connector_meta_output)
    assert not p.kv_connector._chunked_waiting
    blocks = list(seq.block_table)
    now[0] = 111.0
    assert not p.kv_connector.build_connector_meta().has_work()
    assert p.kv_connector.admission_ready(seq)
    assert list(seq.block_table) == blocks
    assert not seq.kv_transfer_params.get("prefill_aborted")
    assert str(seq.id) in p.kv_connector._awaiting_send
