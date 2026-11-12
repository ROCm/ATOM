# SPDX-License-Identifier: MIT
"""Admission bounds must protect KV ownership without stopping decode progress."""

from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.kv_transfer.disaggregation.chunked_prefill import PrefillHandoff
from atom.kv_transfer.disaggregation.pd_admission import (
    PdAdmissionController,
    ReceivePlan,
)
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceStatus


@pytest.fixture
def mooncake_module(monkeypatch):
    with stubbed_aiter():
        from atom.kv_transfer.disaggregation.mooncake import mooncake_connector
    monkeypatch.setattr(mooncake_connector, "get_open_port", lambda: 30000)
    monkeypatch.setattr(mooncake_connector, "get_ip", lambda: "127.0.0.1")
    return mooncake_connector


@pytest.fixture
def build(monkeypatch, mooncake_module, seq_factory):
    def make(
        *,
        options=None,
        prefix=False,
        connector_wrapper=None,
        chunked=False,
        **overrides,
    ):
        config = MockConfig(
            num_kvcache_blocks=64,
            pool_entry_bytes={"kv": 1024},
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
            ),
            enable_prefix_caching=prefix,
            kv_transfer_config={
                "kv_connector": "mooncake",
                "kv_role": "kv_consumer",
                "enable_chunked_transfer": chunked,
                # By default, exactly one 8-token request (2 blocks) fits.
                "pd_admission": (
                    options
                    if options is not None
                    else {"enabled": True, "max_pending_kv_bytes": 2048}
                ),
            },
        )
        for k, v in overrides.items():
            setattr(config, k, v)
        connector = mooncake_module.MooncakeConnectorScheduler(config)
        if connector_wrapper:
            connector = connector_wrapper(connector)
        monkeypatch.setattr(
            "atom.utils.forward_context.get_kvconnector", lambda *args: connector
        )
        scheduler = Scheduler(config)

        def request(tokens=8, *, chunked_transfer=chunked):
            seq = seq_factory(list(range(tokens)))
            seq.kv_transfer_params = {
                "do_remote_prefill": True,
                "transfer_id": f"p-{seq.id}",
                "remote_block_ids": list(range((tokens + 3) // 4)),
                "block_size": 4,
                "dcp_size": 1,
                "remote_host": "127.0.0.1",
                "remote_port": 30000,
                "remote_handshake_port": 30000,
                "tp_size": 1,
                "remote_pp_size": 4,
                "first_token_id": 42,
            }
            if chunked_transfer:
                seq.kv_transfer_params["chunked_transfer"] = True
                # D reserves before P computes: source IDs and T0 do not exist.
                del seq.kv_transfer_params["remote_block_ids"]
                del seq.kv_transfer_params["first_token_id"]
            return seq

        return scheduler, request

    return make


def receive_success(seq, *, as_str=False):
    """A terminal worker report; chunked handoff follows the real wire contract."""
    return KVConnectorOutput(
        finished_recving={str(seq.id) if as_str else seq.id},
        received_handoffs=(
            {seq.id: PrefillHandoff(seq.id, 42)}
            if seq.kv_transfer_params.get("chunked_transfer")
            else {}
        ),
    )


class SizedSeq(SimpleNamespace):
    """A request with one prompt token per block of its `block_table`."""

    def __len__(self):
        return len(self.block_table)


def standalone(max_bytes, *, reserve_blocks=None, free_blocks=0):
    """A controller over a one-token-per-block pool and a full-transfer connector."""
    connector = SimpleNamespace(pd_receive_block_offset=lambda seq, n, cached: 0)
    block_manager = SimpleNamespace(
        num_pool_blocks=lambda tokens: tokens,
        draft_lookahead_blocks=lambda tokens: 0,
        kv=SimpleNamespace(lookup=lambda h: -1, num_free=free_blocks),
    )
    return PdAdmissionController(
        max_bytes, 1024, connector, block_manager, reserve_blocks=reserve_blocks
    )


def try_admit(policy, seq, *, running=False):
    # No local hit: every block of the request is new.
    probe = SimpleNamespace(
        num_cached_blocks=0, block_hashes=[], num_new_blocks=len(seq)
    )
    admitted, plan = policy.check(seq, probe, running=running)
    if admitted:
        policy.commit(seq, plan)
    return admitted


@pytest.mark.parametrize("chunked", [False, True])
def test_ready_credit_held_until_first_batch(build, chunked):
    sched, request = build(chunked=chunked)
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id}
    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert not b.block_table
    assert len(sched.pd_admission.pending) == 1
    assert sched.pd_admission.pending_bytes == 2048

    sched._update_from_kv_xfer_finished(receive_success(a, as_str=True))
    assert len(sched.pd_admission.pending) == 1
    assert sched.pd_admission.pending[str(a.id)].ready
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids
    assert not b.block_table  # A held its credit during this admission pass.
    assert not sched.pd_admission.pending
    assert a.block_table  # Credit release never frees a running request's KV.
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {b.id}
    assert a.id in batch.req_ids  # Decode progresses while B receives.


@pytest.mark.parametrize("chunked", [False, True])
def test_bytes_limit_and_isolated_large_request(build, chunked):
    sched, request = build(
        chunked=chunked,
        options={"enabled": True, "max_pending_kv_bytes": 3072},
    )
    a, b = request(), request(16)
    sched.extend([a, b])
    sched.schedule()
    assert a.block_table and not b.block_table
    sched.schedule()
    assert not b.block_table
    # B waited two passes; it is one deferred request, not two attempts.
    assert sched.pd_admission.deferred == {"bytes": 1, "free_kv": 0}
    sched._update_from_kv_xfer_finished(receive_success(a))
    sched.schedule()
    sched.schedule()
    # Alone, B is admitted past the 3072-byte limit.
    assert b.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched.pd_admission.pending_bytes == 4096
    assert sched.pd_admission.deferred == {"bytes": 1, "free_kv": 0}


@pytest.mark.parametrize("fraction", [None, 0.25])
def test_fraction_uses_physical_pool_bytes(build, fraction):
    options = {"enabled": True}
    if fraction is not None:
        options["max_pending_kv_fraction"] = fraction
    sched, _ = build(options=options)
    # 64 blocks of 1024 bytes; the default fraction is 0.15.
    assert sched.pd_admission.max_bytes == int(64 * 1024 * (fraction or 0.15))
    assert sched.pd_admission.reserve_blocks is None


@pytest.mark.parametrize("chunked", [False, True])
def test_free_kv_watermark_keeps_room_for_running_requests(build, chunked):
    sched, request = build(
        chunked=chunked,
        num_kvcache_blocks=10,
        options={"enabled": True, "min_free_kv_fraction": 0.5},
    )
    # Setting one limit drops the default byte budget.
    assert sched.pd_admission.max_bytes is None
    assert sched.pd_admission.reserve_blocks == 5
    a, b, c = request(), request(), request()
    sched.extend([a, b, c])
    batch, _ = sched.schedule()
    # After B, 6 blocks stay free; C would leave 4, below the 5 reserved.
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id, b.id}
    assert not c.block_table
    sched.schedule()
    assert not c.block_table
    assert sched.pd_admission.deferred == {"bytes": 0, "free_kv": 1}
    sched._update_from_kv_xfer_finished(receive_success(a))
    sched._update_from_kv_xfer_finished(receive_success(b))
    batch, _ = sched.schedule()
    assert {a.id, b.id} <= set(batch.req_ids)
    assert not c.block_table  # A and B held credits during this pass.
    # Past their first batch A and B hold no credit, but they still decode.
    assert not sched.pd_admission.pending
    sched.schedule()
    assert not c.block_table
    for seq in (a, b):
        assert sched.abort_request(seq.id)
    sched.postprocess([], ScheduledBatchOutput([], [], None, None, None), batch=batch)
    sched.schedule()
    assert c.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert set(sched.pd_admission.pending) == {str(c.id)}


@pytest.mark.parametrize("chunked", [False, True])
def test_free_kv_watermark_holds_while_admitted_requests_decode(build, chunked):
    sched, request = build(
        chunked=chunked,
        num_kvcache_blocks=10,
        options={"enabled": True, "min_free_kv_fraction": 0.5},
    )
    a = request(16)
    sched.add(a)
    sched.schedule()
    sched._update_from_kv_xfer_finished(receive_success(a))
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids
    assert not sched.pd_admission.pending
    # Nothing is pending, but B would leave fewer than 5 free blocks while A
    # decodes.
    b = request()
    sched.add(b)
    sched.schedule()
    assert not b.block_table
    assert sched.pd_admission.deferred == {"bytes": 0, "free_kv": 1}
    assert sched.abort_request(a.id)
    sched.postprocess([], ScheduledBatchOutput([], [], None, None, None), batch=batch)
    sched.schedule()
    assert b.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_free_kv_watermark_admits_below_it_only_on_an_idle_decoder():
    policy = standalone(None, reserve_blocks=8, free_blocks=10)
    big = SizedSeq(id=1, block_table=[1, 2, 3, 4])
    # 10 free, 4 allocated: 6 left, below the 8 reserved.
    assert not try_admit(policy, big, running=True)
    # Nothing pending or running: waiting would free nothing, so it enters.
    assert try_admit(policy, big, running=False)


@pytest.mark.parametrize("chunked", [False, True])
def test_free_kv_watermark_counts_claimed_prefix_hits(build, seq_factory, chunked):
    sched, request = build(
        chunked=chunked,
        prefix=True,
        num_kvcache_blocks=10,
        options={"enabled": True, "min_free_kv_fraction": 0.6},
    )
    bm = sched.block_manager
    seed = seq_factory(list(range(16)))
    bm.allocate(seed, bm.can_allocate(seed))
    bm.hash_blocks(seed, seed.num_tokens)
    bm.deallocate(seed)
    a, b = request(), request(16)
    sched.add(a)
    sched.schedule()
    assert set(sched.pd_admission.pending) == {str(a.id)}
    free = bm.kv.num_free
    assert free == 8
    sched.add(b)
    sched.schedule()
    # B receives one block, but its hit also takes two cached-but-free blocks:
    # 8 - 3 would leave fewer than the 6 reserved.
    assert not b.block_table
    assert bm.kv.num_free == free
    assert sched.pd_admission.deferred == {"bytes": 0, "free_kv": 1}
    sched._update_from_kv_xfer_finished(receive_success(a))
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids
    # A decodes past its first batch; B still waits for room.
    sched.schedule()
    assert not b.block_table
    assert sched.abort_request(a.id)
    sched.postprocess([], ScheduledBatchOutput([], [], None, None, None), batch=batch)
    batch, _ = sched.schedule()
    # With A gone, B enters with the same three-block hit.
    assert batch.connector_meta_output.reqs_to_recv[b.id].num_computed_blocks == 3
    assert set(sched.pd_admission.pending) == {str(b.id)}


@pytest.mark.parametrize("chunked", [False, True])
def test_free_kv_watermark_skips_hits_on_blocks_another_request_holds(
    build, seq_factory, chunked
):
    sched, request = build(
        chunked=chunked,
        prefix=True,
        num_kvcache_blocks=10,
        options={"enabled": True, "min_free_kv_fraction": 0.3},
    )
    bm = sched.block_manager
    assert sched.pd_admission.reserve_blocks == 3
    # A running request holds the four prompt blocks the next one hits.
    holder = seq_factory(list(range(16)))
    bm.allocate(holder, bm.can_allocate(holder))
    bm.hash_blocks(holder, holder.num_tokens)
    a, b = request(4), request(16)
    sched.extend([a, b])
    batch, _ = sched.schedule()
    # After A, 5 blocks are free. B's three-block hit is held by `holder`, so
    # B takes one block and leaves 4; counting its four blocks would leave 1.
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id, b.id}
    assert batch.connector_meta_output.reqs_to_recv[b.id].num_computed_blocks == 3
    assert bm.kv.num_free == 4


def test_probe_reports_free_pool_demand_without_prefix_caching(build, seq_factory):
    from atom.model_engine.block_manager import AdmissionProbe

    sched, _ = build()
    probe = AdmissionProbe()
    seq = seq_factory(list(range(8)))
    assert sched.block_manager.can_allocate(seq, record=False, probe=probe) == 0
    assert probe.num_new_blocks == 2


@pytest.mark.parametrize("geometry_matches", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_prefix_hits_charge_only_receive_targets(
    build, seq_factory, geometry_matches, chunked
):
    sched, request = build(chunked=chunked, prefix=True)
    seed = seq_factory(list(range(8)))
    bm = sched.block_manager
    bm.allocate(seed, bm.can_allocate(seed))
    bm.hash_blocks(seed, seed.num_tokens)
    bm.deallocate(seed)
    a, b = request(), request()
    if not geometry_matches:
        # An incompatible source layout falls back to a full transfer.
        for seq in (a, b):
            seq.kv_transfer_params["block_size"] = 8
    sched.extend([a, b])
    batch, _ = sched.schedule()
    recv = batch.connector_meta_output.reqs_to_recv
    # Each request charges only what it receives: the block past the shared
    # prefix, or both blocks when nothing can be skipped.
    assert set(recv) == ({a.id, b.id} if geometry_matches else {a.id})
    assert sched.pd_admission.pending_bytes == 2048
    assert all(m.num_computed_blocks == int(geometry_matches) for m in recv.values())


@pytest.mark.parametrize("chunked", [False, True])
def test_overlapping_receive_targets_charge_their_physical_union(
    build, seq_factory, chunked
):
    sched, request = build(
        chunked=chunked,
        prefix=True,
        options={"enabled": True, "max_pending_kv_bytes": 3072},
    )
    seed = seq_factory(list(range(8)))
    bm = sched.block_manager
    bm.allocate(seed, bm.can_allocate(seed))
    bm.hash_blocks(seed, seed.num_tokens)
    bm.deallocate(seed)
    a, b = request(), request()
    for seq in (a, b):
        # No usable page geometry: a full transfer, hit block included.
        seq.kv_transfer_params["block_size"] = 8
    sched.extend([a, b])
    batch, _ = sched.schedule()
    # A holds {p, x}; B adds only y, so {p, x, y} fills the three-block W.
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id, b.id}
    assert a.block_table[0] == b.block_table[0]
    assert sched.pd_admission.pending_bytes == 3072


@pytest.mark.parametrize("chunked", [False, True])
def test_cancel_keeps_credit_until_safe_terminal_and_ignores_duplicate(build, chunked):
    sched, request = build(chunked=chunked)
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    free_before = sched.block_manager.kv.num_free
    assert sched.abort_request(a.id)
    assert sched.pd_admission.pending[str(a.id)].cancelled
    sched.schedule()
    assert not b.block_table
    assert sched.block_manager.kv.num_free == free_before
    # The worker has already aggregated all producers/ranks and drained writes.
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={str(a.id)}))
    assert not sched.pd_admission.pending
    assert not a.block_table
    sched.schedule()
    assert b.block_table
    # A late success for A must not retire B's newly acquired credit.
    sched._update_from_kv_xfer_finished(receive_success(a, as_str=True))
    assert set(sched.pd_admission.pending) == {str(b.id)}


@pytest.mark.parametrize("chunked", [False, True])
def test_promoted_request_keeps_credit_until_batch_or_abort(build, chunked):
    sched, request = build(chunked=chunked)
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    sched._update_from_kv_xfer_finished(receive_success(a))
    # A is promoted to RUNNING, but its decode row does not fit this pass.
    sched.mtp_k = 4
    sched.use_spec = True
    sched.max_num_batched_tokens = 4
    batch, _ = sched.schedule()
    assert a.status == SequenceStatus.RUNNING
    assert not batch.req_ids
    # RUNNING alone does not return the credit; an abort before the batch does.
    assert set(sched.pd_admission.pending) == {str(a.id)}
    assert not b.block_table
    assert sched.abort_request(a.id)
    finished = sched.postprocess(
        [], ScheduledBatchOutput([], [], None, None, None), batch=batch
    )
    assert finished == [a]
    assert not a.block_table
    assert not sched.pd_admission.pending
    # B no longer waits behind A's credit.
    sched.mtp_k = 0
    sched.use_spec = False
    sched.max_num_batched_tokens = 64
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {b.id}
    assert set(sched.pd_admission.pending) == {str(b.id)}


@pytest.mark.parametrize("chunked", [False, True])
def test_disabled_preserves_admission(build, chunked):
    sched, request = build(options={"enabled": False}, chunked=chunked)
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id, b.id}
    assert sched.pd_admission is None


@pytest.mark.parametrize("chunked", [False, True])
def test_speculative_first_decode_uses_existing_lookahead(build, chunked):
    sched, request = build(
        chunked=chunked,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, use_dspark=lambda: True
        ),
    )
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    assert len(a.block_table) == 2  # Only the prompt is a receive target.
    assert sched.pd_admission.pending_bytes == 2048
    sched._update_from_kv_xfer_finished(receive_success(a))
    batch, _ = sched.schedule()
    assert batch.req_ids == [a.id]
    assert batch.num_spec_step == 3
    assert sched.block_manager.draft_writes_allocated(a)
    assert not sched.pd_admission.pending
    assert not b.block_table


@pytest.mark.parametrize("chunked", [False, True])
def test_failed_receive_releases_credits_without_local_recompute(build, chunked):
    sched, request = build(chunked=chunked)
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={a.id}))
    assert sched.take_rejected() == [a]
    assert not a.block_table
    assert not sched.pd_admission.pending
    batch, _ = sched.schedule()
    assert a.id not in batch.req_ids
    assert set(batch.connector_meta_output.reqs_to_recv) == {b.id}


@pytest.mark.parametrize("chunked", [False, True])
def test_multi_pd_ready_resume_releases_only_on_batch(build, monkeypatch, chunked):
    from test_multi_connector import FakeSchedSub

    from atom.kv_transfer.disaggregation.multi import multi_connector

    store = FakeSchedSub(is_offload=True, offload_methods=True, has_state_tier=False)
    # Complete the base P/D no-op face on the offload test double.
    store.admission_ready = lambda seq: True
    store.process_pd_completions = lambda output: output

    def wrap(pd):
        monkeypatch.setattr(
            multi_connector, "_build_subconnectors", lambda *args, **kwargs: [pd, store]
        )
        return multi_connector.MultiConnectorScheduler(SimpleNamespace())

    sched, request = build(connector_wrapper=wrap, chunked=chunked)
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.metas[0].reqs_to_recv) == {a.id}
    assert not b.block_table
    sched._update_from_kv_xfer_finished(receive_success(a))
    assert sched.pd_admission.pending[str(a.id)].ready
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids  # The existing Multi suffix-resume path runs.
    assert a.block_table
    assert not sched.pd_admission.pending


@pytest.mark.parametrize("chunked", [False, True])
def test_ready_preempted_before_first_forward_releases_credit(build, chunked):
    sched, request = build(num_kvcache_blocks=2, chunked=chunked)
    a = request()
    sched.add(a)
    sched.schedule()
    sched._update_from_kv_xfer_finished(receive_success(a))
    # The inherited decode capacity check needs another block for T0. No
    # decode-slot reservation exists, so the ready row may be preempted.
    batch, _ = sched.schedule()
    assert not batch.req_ids
    assert not a.block_table
    assert sched.total_preemptions == 1
    assert not sched.pd_admission.pending


@pytest.mark.parametrize("first_chunked", [False, True])
def test_chunked_and_nonchunked_requests_share_one_budget(build, first_chunked):
    sched, request = build(chunked=True)
    a = request(chunked_transfer=first_chunked)
    b = request(chunked_transfer=not first_chunked)
    sched.extend([a, b])
    sched.schedule()
    assert a.block_table and not b.block_table
    sched._update_from_kv_xfer_finished(receive_success(a))
    sched.schedule()
    sched.schedule()
    assert set(sched.pd_admission.pending) == {str(b.id)}


@pytest.mark.parametrize("geometry_matches", [False, True])
def test_chunked_charge_matches_dispatched_targets_without_source_ids(
    build, seq_factory, monkeypatch, geometry_matches
):
    import ctypes

    import msgpack
    from test_pd_chunked_transfer import decode_worker, post_write_requests

    # The consumer flag is intentionally off: the router's per-request mode
    # controls the transfer, including on a D service receiving both modes.
    sched, request = build(prefix=True, pool_entry_bytes={"kv": 4})
    seed = seq_factory(list(range(8)))
    bm = sched.block_manager
    bm.allocate(seed, bm.can_allocate(seed))
    bm.hash_blocks(seed, seed.num_tokens)
    bm.deallocate(seed)
    a = request(chunked_transfer=True)
    if not geometry_matches:
        a.kv_transfer_params["block_size"] = 8
    sched.add(a)
    batch, _ = sched.schedule()
    buf = (ctypes.c_ubyte * (64 * 4))()
    worker = decode_worker(buf)
    worker.tp_size = 4
    worker.start_load_kv(batch.connector_meta_output)
    post_write_requests(worker, monkeypatch)
    assert worker._send_on_socket.call_count == 4  # P PP4, one D rank.
    offset = 1 if geometry_matches else 0
    for call in worker._send_on_socket.call_args_list:
        payload = msgpack.loads(call.args[1][1])
        assert payload["src_block_ids"] == []
        assert payload["num_computed_blocks"] == offset
        assert payload["dst_block_ids"] == list(a.block_table[offset:])
        assert sched.pd_admission.pending_bytes == len(payload["dst_block_ids"]) * 4


@pytest.mark.parametrize("outcome", ["success", "failed_stage", "bad_handoff"])
def test_chunked_pp4_tp4_credits_wait_for_terminal_handoff(build, monkeypatch, outcome):
    import ctypes

    from test_pd_chunked_transfer import decode_worker, post_write_requests

    from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator

    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")
    sched, request = build(chunked=True)
    a, b = request(12), request(12)
    sched.extend([a, b])
    batch, _ = sched.schedule()
    workers = []
    # Keep the CPU destination buffers alive through receive completion.
    buffers = [(ctypes.c_ubyte * (64 * 4))() for _ in range(4)]
    for rank, buf in enumerate(buffers):
        worker = decode_worker(buf)
        worker.tp_size, worker.tp_rank = 4, rank
        worker.start_load_kv(batch.connector_meta_output)
        post_write_requests(worker, monkeypatch)
        assert worker._pending_recv_expected[a.id] == 4
        workers.append(worker)
    aggregate = KVOutputAggregator(4)
    for stage in range(4):
        for rank, worker in enumerate(workers):
            handoff = {"first_token_id": 42, "draft_token_ids": [43]}
            last = stage == rank == 3
            if last and outcome == "bad_handoff":
                handoff = {"first_token_id": -1}
            worker._record_write_done(
                a.id,
                stage,
                0,
                worker._pending_recv_nonce[a.id],
                success=not (stage == rank == 0 and outcome == "failed_stage"),
                handoff=handoff,
            )
            sched._update_from_kv_xfer_finished(
                aggregate.aggregate([w.get_finished() for w in workers])
            )
            if not last:
                assert sched.pd_admission.pending_bytes == 3072
                assert not sched.pd_admission.pending[str(a.id)].ready
                assert "first_token_id" not in a.kv_transfer_params
                _, scheduled = sched.schedule()
                assert not scheduled and not b.block_table

    if outcome == "success":
        assert a.kv_transfer_params["first_token_id"] == 42
        assert a.kv_transfer_params["draft_token_ids"] == [43]
        assert sched.pd_admission.pending[str(a.id)].ready
        assert sched.pd_admission.pending_bytes == 3072
        _, scheduled = sched.schedule()
        assert list(scheduled) == [a.id]
        assert not sched.pd_admission.pending and a.block_table
    else:
        assert not sched.pd_admission.pending and not a.block_table
        assert sched.take_rejected() == [a]
        batch, scheduled = sched.schedule()
        assert not scheduled
        assert set(batch.connector_meta_output.reqs_to_recv) == {b.id}


def test_multiconnector_uses_load_winner(build):
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    store = SimpleNamespace(
        is_offload=True,
        is_producer=False,
        # The `KVConnectorSchedulerBase` defaults every offload scheduler has.
        supports_pd_admission=False,
        pd_receive_block_offset=lambda *args: None,
        get_num_new_matched_tokens=lambda seq: (8, True),
        cancel_pending_load=lambda seq: None,
    )
    # Exercise the real first-hit selection without invoking an offload backend.
    sched, request = build()
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [sched.kv_connector, store]
    multi._load_winner = {}
    assert multi.supports_pd_admission
    a = request()
    multi.get_num_new_matched_tokens(a)
    assert multi.pd_receive_block_offset(a, 2, 0) == 0
    a.kv_transfer_params = {}
    multi.get_num_new_matched_tokens(a)
    assert multi.pd_receive_block_offset(a, 2, 0) is None
    multi._connectors.reverse()
    assert not multi.supports_pd_admission


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "false"},
        # There is no pending-request count; the option is unknown.
        {"enabled": True, "max_pending_requests": 4},
        {"enabled": True, "max_pending_kv_bytes": -1},
        {"enabled": True, "max_pending_kv_bytes": True},
        {"enabled": True, "max_pending_kv_fraction": float("nan")},
        {"enabled": True, "max_pending_kv_fraction": 1.1},
        {"enabled": True, "max_pending_kv_fraction": 0.1, "max_pending_kv_bytes": 10},
        {"enabled": True, "max_pending_kv_fraction": None},
        # A limit is either set or omitted; null does not disable it.
        {"enabled": True, "max_pending_kv_fraction": None, "min_free_kv_fraction": 0.5},
        {"enabled": True, "max_pending_kv_bytes": None},
        {"enabled": True, "extra_growth_blocks_per_request": 1},
        {"enabled": True, "min_free_kv_fraction": 0},
        {"enabled": True, "min_free_kv_fraction": 1},
        {"enabled": True, "min_free_kv_fraction": True},
        {"enabled": True, "min_free_kv_fraction": float("inf")},
    ],
)
def test_invalid_config_rejected(build, options):
    with pytest.raises(ValueError):
        build(options=options)


@pytest.mark.parametrize(
    "override",
    [
        {"pool_entry_bytes": {}},
        {"pool_entry_bytes": {"kv": 1024, "state": 64}},
        {"pipeline_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        # Ulysses SP runs over the PCP group even with PCP=1.
        {"sequence_parallel_size": 2},
    ],
)
def test_unsupported_layout_rejected(build, override):
    with pytest.raises(ValueError):
        build(**override)


def test_physical_targets_are_deduplicated_and_release_is_idempotent():
    policy = PdAdmissionController(4096, 1024, connector=None, block_manager=None)
    a = SimpleNamespace(id=1, block_table=[1, 2])
    b = SimpleNamespace(id=2, block_table=[2, 3])
    policy.commit(a, ReceivePlan(0, 2))
    policy.commit(b, ReceivePlan(0, 2))
    assert policy.pending_bytes == 3072
    policy.release(a)
    policy.release(a)
    assert policy.pending_bytes == 2048
    # Same numeric ID on a different sequence cannot release an existing entry.
    policy.release(SimpleNamespace(id=2))
    assert policy.pending_bytes == 2048
    policy.release(b)
    assert policy.pending_bytes == 0


def test_a_request_over_both_limits_counts_under_both(build):
    sched, _ = build(
        options={
            "enabled": True,
            "max_pending_kv_bytes": 1024,
            "min_free_kv_fraction": 0.5,
        }
    )
    assert (sched.pd_admission.max_bytes, sched.pd_admission.reserve_blocks) == (
        1024,
        32,
    )
    policy = standalone(1024, reserve_blocks=32, free_blocks=32)
    a = SizedSeq(id=1, block_table=[1])
    b = SizedSeq(id=2, block_table=[2])
    assert try_admit(policy, a)
    # B would exceed the byte budget and leave 31 free blocks, below the 32 kept.
    for _ in range(2):
        assert not try_admit(policy, b)
    assert policy.deferred == {"bytes": 1, "free_kv": 1}


def test_rejected_deferred_request_leaves_no_marker():
    policy = standalone(1024)
    a = SizedSeq(id=1, block_table=[1])
    b = SizedSeq(id=2, block_table=[2])
    assert try_admit(policy, a)
    assert not try_admit(policy, b)
    # B is rejected from waiting without ever being admitted.
    policy.cancel(b)
    assert not policy._deferred_by


def test_admission_state_is_exported_from_the_metrics_snapshot(build):
    from queue import Queue

    from atom.metrics.exporter import _AtomMetricsCollector
    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.llm_engine import LLMEngine

    sched, request = build()
    handler = EngineUtilityHandler(None, Queue(), scheduler=sched)
    exporter = SimpleNamespace(stream_silence=lambda: 0.0, other_processes_gc=list)

    def exported():
        """The admission series a scrape serves, through the engine's DP aggregation."""
        engine = SimpleNamespace(
            core_mgr=SimpleNamespace(
                latest_metrics={0: handler.collect_metrics()},
                get_dp_router_statistics=dict,
            )
        )
        snapshot = LLMEngine.get_metrics_statistics(engine)
        return {
            (sample.name, tuple(sorted(sample.labels.items()))): sample.value
            for family in _AtomMetricsCollector(exporter)._collect((snapshot, 0, 0.0))
            for sample in family.samples
            if sample.name.startswith("atom:pd_admission_")
        }

    # A plain `Scheduler` reports no role, which aggregation exports as "default".
    labels = (("dp_rank", "0"), ("engine_role", "default"))
    pending_bytes = ("atom:pd_admission_pending_kv_bytes", labels)
    deferred = "atom:pd_admission_deferred_total"
    # Cumulative: B deferred once by bytes, never by the free-KV reserve.
    counters = {
        (deferred, (*labels, ("reason", "bytes"))): 1,
        (deferred, (*labels, ("reason", "free_kv"))): 0,
    }

    def pending(**counts):
        states = {"receiving": 0, "ready": 0, "cancel_draining": 0} | counts
        return {
            ("atom:pd_admission_pending_requests", (*labels, ("state", s))): n
            for s, n in states.items()
        }

    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    sched.schedule()
    assert exported() == pending(receiving=1) | {pending_bytes: 2048, **counters}
    sched._update_from_kv_xfer_finished(receive_success(a, as_str=True))
    assert exported() == pending(ready=1) | {pending_bytes: 2048, **counters}
    # A's first forward returns its credit; B is admitted on the next pass.
    sched.schedule()
    assert exported() == pending() | {pending_bytes: 0, **counters}
    sched.schedule()
    assert sched.abort_request(b.id)
    # A cancelled receive keeps its credit while its writes drain.
    assert exported() == pending(cancel_draining=1) | {pending_bytes: 2048, **counters}
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={str(b.id)}))
    assert exported() == pending() | {pending_bytes: 0, **counters}
