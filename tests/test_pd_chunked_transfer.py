# SPDX-License-Identifier: MIT
"""Chunked P/D protocol tests using CPU buffers and the real address planner."""

import ctypes
import queue
import threading
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
)


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
    c._chunked_local_ids = {}
    c._pending_chunked_requests = []
    c._completion_lock = threading.Lock()
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
            future = pool.submit(c._execute_transfer, req)
            event = Event()
            state.publish(6, event)  # token 4..5 live in an incomplete page
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
    c._execute_transfer(req)
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
        future = pool.submit(c._execute_transfer, req)
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
    c._execute_transfer(req)
    c._execute_transfer(req.copy())
    assert c.transfer_engine.writes.qsize() == 1
    assert c.get_finished().is_empty()
    c._execute_transfer(dict(req, request_id=22, notify_port=3))
    assert c.transfer_engine.writes.qsize() == 2
    assert c.get_finished().connector_completions


def test_bad_prompt_or_block_geometry_fails_before_writing():
    for override in ({"prompt_digest": "different"}, {"dst_block_ids": [2]}):
        c, state, req, _, _ = producer()
        req.update(override)
        publish(state, 10)
        state.finish({"first_token_id": 42})
        c._execute_transfer(req)
        assert c.transfer_engine.writes.empty()
        c._notify_transfer_result.assert_called_once_with(req, success=False)


def consumer(expected=2):
    c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    c._completion_lock = threading.Lock()
    c._fence_lock = threading.Lock()
    c._pending_recv_expected = {21: expected}
    c._pending_recv_stages = {}
    c._pending_recv_nonce = {21: 3}
    c._pending_recv = {21}
    c._pending_recv_blocks = {21: [2, 9, 4]}
    c._pending_recv_slots = {}
    c._pending_handoffs = {21: {"failed": False, "metadata": None}}
    c._dispatch_in_flight = set()
    c._deferred_failures = {}
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


def test_pp_failure_can_finish_while_dispatch_flag_is_still_set():
    c = consumer(1)
    c._dispatch_in_flight.add(21)
    assert c._record_write_done(21, 0, 0, 3, success=False)
    assert c.failed_recving == {21}


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
    assert state.ready_blocks == 1
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
        future = pool.submit(c._execute_transfer, req)
        publish(state, 4)
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
            is_producer=True, is_offload=False, build_connector_meta=lambda: meta
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
        future = pool.submit(c._execute_transfer, req)
        publish(state, 8)
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
