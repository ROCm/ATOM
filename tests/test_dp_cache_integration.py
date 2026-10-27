# SPDX-License-Identifier: MIT
"""Real manager wire format and scheduler admission, with CPU cache pools."""

import pickle
import queue
from threading import Event, Lock
from types import SimpleNamespace as NS

import pytest
from conftest import MockConfig
from test_dp_load_balance import _make_mgr, _route
from test_lmcache_mp import _config

from atom.kv_transfer.offload.mp import deployment
from atom.model_engine.collective_rpc import RpcResponseRouter
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.model_engine.engine_core_mgr import _add_frame
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.prefix_hash import attach_shipped_chains, ship_prompt_hash_chain
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.model_engine.state_runtime import StateRuntime, StateTransfer

BLOCK = 128
PREFIX = 100 * 1024
TOKENS = list(range(PREFIX + 1024))


def request(*, rank=None):
    seq = Sequence(TOKENS, BLOCK, dp_session_id="session", data_parallel_rank=rank)
    ship_prompt_hash_chain(seq, BLOCK)
    return seq


def manager(monkeypatch, *, legacy_session_affinity=False):
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "1")
    monkeypatch.setenv("ATOM_DP_CACHE_BALANCE_CONFIG", "{}")
    mgr = _make_mgr(
        2, strategy="cache_affinity", session_affinity=legacy_session_affinity
    )
    mgr._init_dp_cache_balance(
        NS(
            dp_load_balance="cache_affinity",
            enable_prefix_caching=True,
            kv_transfer_config={"kv_connector": "lmcache_mp"},
        )
    )
    seq = request(rank=0)
    assert _route(mgr, [seq]) == [0]
    mgr._cache_admitted(0, (seq.id, 0, 0))
    mgr._release_seq_load(seq.id, succeeded=True)
    mgr._rank_tokens[0] = 1000000
    return mgr


def test_v41_requires_native_checkpoints_before_model_startup():
    mgr = _make_mgr(2, strategy="cache_affinity")
    with pytest.raises(ValueError, match="no-encoder-swa-bounded-replay"):
        mgr._init_dp_cache_balance(
            NS(
                dp_load_balance="cache_affinity",
                enable_prefix_caching=True,
                kv_transfer_config={"kv_connector": "lmcache_mp"},
                hf_config=NS(model_type="deepseek_v41"),
                enable_encoder_swa_bounded_replay=True,
            )
        )


@pytest.mark.parametrize("legacy_session_affinity", [False, True])
def test_retry_preserves_add_hash_chain_and_does_not_finish_request(
    monkeypatch, legacy_session_affinity
):
    mgr = manager(monkeypatch, legacy_session_affinity=legacy_session_affinity)
    seq = request()
    assert _route(mgr, [seq]) == [1]
    assert mgr._dp_session_owners == {}
    assert mgr._dp_session_prompt_tokens == {}
    frame = pickle.loads(_add_frame([seq]))
    [target] = frame[1]
    target.dp_cache_retry_reason = "recoverable_prefix_too_short"
    assert seq.prompt_hash_chain is None
    sent = []
    mgr._send_control = lambda rank, payload: sent.append((rank, payload))
    assert mgr._retry_cache_migration(1, target)
    assert len(sent) == 1 and sent[0][0] == 0
    _, [retry], *chains = pickle.loads(sent[0][1])
    attach_shipped_chains([retry], *chains)
    assert retry.id == seq.id
    assert list(retry.token_ids) == TOKENS
    assert retry.prompt_hash_chain is not None
    assert retry.dp_cache_migration == retry.dp_cache_minimum_hit == 0
    assert mgr._seq_load[seq.id][0] == 0
    assert mgr._cache_balancer.sessions["session"].owner == 0
    assert (
        mgr._cache_balancer.counters["migration_fallback_recoverable_prefix_too_short"]
        == 1
    )
    assert mgr._retry_cache_migration(1, target)  # stale target completion
    assert len(sent) == 1
    assert (
        mgr._cache_balancer.counters["migration_fallback_recoverable_prefix_too_short"]
        == 1
    )


def test_target_ack_corrects_load_then_changes_owner(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    assert _route(mgr, [seq]) == [1]
    before = mgr._seq_load[seq.id][2]
    event = (seq.id, seq.dp_cache_migration, seq.dp_cache_minimum_hit)
    mgr._cache_admitted(0, event)  # wrong engine cannot acknowledge target bytes
    assert mgr._cache_balancer.sessions["session"].owner == 0
    mgr._cache_admitted(1, event)
    assert mgr._cache_balancer.sessions["session"].owner == 1
    assert mgr._seq_load[seq.id][2] == seq.num_prompt_tokens - event[2] < before
    assert seq.id not in mgr._cache_retry_payloads
    mgr._mark_seq_prefill_complete(seq.id)
    assert mgr._rank_tokens[1] == 0
    mgr._release_seq_load(seq.id, succeeded=True)
    assert mgr._rank_reqs == [0, 0]


def test_cancelled_migration_cannot_retry(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    target = pickle.loads(_add_frame([seq]))[1][0]
    target.dp_cache_retry_reason = "preparation_timeout"
    mgr._release_seq_load(seq.id)
    mgr._send_control = lambda *args: pytest.fail("cancelled request was retried")
    assert mgr._retry_cache_migration(1, target)
    assert not mgr._cache_retry_payloads


def short_rejection(mgr):
    seq = request()
    assert _route(mgr, [seq]) == [1]
    target = pickle.loads(_add_frame([seq]))[1][0]
    target.dp_cache_retry_reason = "recoverable_prefix_too_short"
    target.dp_cache_recoverable_hit = 64 * 1024
    return seq, target


@pytest.mark.parametrize(
    "source_hit,source_load,chosen",
    [
        (0, 0, 1),
        (None, 0, 1),
        (PREFIX, 0, 0),
        (PREFIX, 1000000, 1),
    ],
)
def test_short_restore_requeues_using_fresh_costs(
    monkeypatch, source_hit, source_load, chosen
):
    mgr = manager(monkeypatch)
    seq, target = short_rejection(mgr)
    mgr._rank_tokens[0] = source_load
    sent, ready = [], Event()
    mgr._probe_cache_source = lambda *args: (
        source_hit,
        "ok" if source_hit is not None else "timeout",
    )

    def send(rank, frame):
        sent.append((rank, frame))
        ready.set()

    mgr._send_control = send
    assert mgr._retry_cache_migration(1, target)
    assert ready.wait(2)
    assert sent[0][0] == chosen
    _, [retry], *chains = pickle.loads(sent[0][1])
    attach_shipped_chains([retry], *chains)
    assert retry.prompt_hash_chain is not None
    assert retry.dp_cache_minimum_hit == 0
    assert mgr._cache_balancer.sessions["session"].owner == 0
    hit = PREFIX if chosen == 0 else 64 * 1024
    assert mgr._seq_load[seq.id] == (chosen, 1, seq.num_prompt_tokens - hit)
    assert mgr._retry_cache_migration(1, target)  # old generation cannot requeue again
    assert len(sent) == 1
    if chosen == 1:
        assert retry.dp_cache_migration > target.dp_cache_migration
        # The target lost more cache while the source probe was pending.
        mgr._cache_admitted(1, (seq.id, retry.dp_cache_migration, 0))
        assert mgr._seq_load[seq.id][2] == seq.num_prompt_tokens
        assert mgr._cache_balancer.sessions["session"].owner == 1
    mgr._release_seq_load(seq.id)
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [source_load, 0]


def test_cancel_during_source_probe_does_not_resurrect_request(monkeypatch):
    mgr = manager(monkeypatch)
    seq, target = short_rejection(mgr)
    entered, release, done = Event(), Event(), Event()

    def probe(*args):
        entered.set()
        assert release.wait(2)
        return 0, "ok"

    mgr._probe_cache_source = probe
    mgr._send_control = lambda *args: pytest.fail("cancelled retry")
    slots = mgr._cache_probe_slots
    mgr._cache_probe_slots = NS(
        acquire=slots.acquire, release=lambda: (slots.release(), done.set())
    )
    assert mgr._retry_cache_migration(1, target)
    assert entered.wait(2)
    mgr._release_seq_load(seq.id)
    release.set()
    assert done.wait(2)
    assert not mgr._cache_balancer.requests
    assert not mgr._cache_retry_payloads


def test_cache_clear_discards_both_observations_and_fences_owner(monkeypatch):
    mgr = manager(monkeypatch)
    seq, target = short_rejection(mgr)
    entered, release, ready = Event(), Event(), Event()
    sent = []

    def probe(*args):
        entered.set()
        assert release.wait(2)
        return 0, "ok"

    mgr._probe_cache_source = probe
    mgr._send_control = lambda rank, frame: (sent.append((rank, frame)), ready.set())
    assert mgr._retry_cache_migration(1, target)
    assert entered.wait(2)
    mgr._invalidate_cache_predictions()
    release.set()
    assert ready.wait(2)
    retry = pickle.loads(sent[0][1])[1][0]
    assert mgr._seq_load[seq.id][2] == seq.num_prompt_tokens
    mgr._cache_admitted(1, (seq.id, retry.dp_cache_migration, 0))
    assert mgr._cache_balancer.sessions["session"].owner == 0


def test_async_failed_send_delivers_terminal_and_releases_load(monkeypatch):
    mgr = manager(monkeypatch)
    seq, target = short_rejection(mgr)
    mgr._probe_cache_source = lambda *args: (0, "ok")
    mgr.outputs_queue = queue.Queue()
    outputs, flushes = [], []
    mgr._seq_id_to_callback = {seq.id: outputs.append}
    mgr._flush_stream_batch_fn = lambda: flushes.append(True)

    def send(*args):
        raise RuntimeError("closed target socket")

    mgr._send_control = send
    assert mgr._retry_cache_migration(1, target)
    [finished] = mgr.outputs_queue.get(timeout=2)
    assert finished.leave_reason == "cache_migration_retry_failed"
    assert len(outputs) == len(flushes) == 1
    assert not mgr._cache_retry_payloads
    assert not mgr._cache_balancer.requests
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [1000000, 0]


@pytest.mark.parametrize("reply", [True, False])
def test_source_rpc_preserves_chain_and_unregisters_on_timeout(monkeypatch, reply):
    from dataclasses import replace

    mgr = manager(monkeypatch)
    seq, target = short_rejection(mgr)
    balancer = mgr._cache_balancer
    balancer.config = replace(balancer.config, source_probe_timeout_seconds=0.02)
    assert balancer.begin_reconsider(seq.id, 1, target.dp_cache_migration)
    mgr._rpc_router = RpcResponseRouter()
    mgr._control_send_lock = Lock()
    mgr.control_identities = [b"a", b"b"]
    ids = []

    def send(parts, **kwargs):
        _, body = pickle.loads(parts[1])
        assert parts[0] == b"a"
        assert body["cmd"] == "probe_local_prefix"
        assert body["chain"] is not None
        assert list(body["seq"].token_ids) == TOKENS
        ids.append(body["request_id"])
        if reply:
            mgr._route_utility_response(
                0,
                {
                    "cmd": "probe_local_prefix",
                    "request_id": ids[-1],
                    "result": 0,
                },
            )

    mgr.control_sockets = [NS(send_multipart=send), None]
    result = mgr._probe_cache_source(
        balancer, target, mgr._cache_retry_payloads[seq.id]
    )
    assert result == ((0, "ok") if reply else (None, "timeout"))
    assert mgr._rpc_router.in_flight() == 0
    assert not mgr._rpc_router.route(ids[0], (0, {"result": PREFIX}))


def test_source_probe_does_not_count_tier_only_state_or_allocate():
    spec = PagedStateCheckpointSpec(10, 50, "probe-test", image_bytes=25)
    scheduler = Scheduler(
        MockConfig(
            enable_prefix_caching=True, num_kvcache_blocks=40, pool_entries={"state": 4}
        ),
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id), checkpoint_spec=spec
        ),
    )
    bm = scheduler.block_manager
    warm = Sequence(list(range(20)), 4)
    bm.allocate(warm, bm.can_allocate(warm))
    bm.hash_blocks(warm, 20)
    bm.deallocate(warm)
    probe = Sequence(list(range(24)), 4, has_per_req_cache=True)
    checkpoints = bm.paged_state_checkpoints
    # Normal admission can optimistically consider the tier. The source probe
    # must use only HBM PAGEs and a ready STATE image at a common boundary.
    checkpoints.offload = NS(could_serve=lambda h: True)
    hashes = bm._walk_prefix(probe).hashes
    assert bm._gated_hit(probe, len(hashes), hashes) == len(hashes)
    assert bm.probe_local_prefix(probe) == 0
    op = checkpoints.store.begin_store(hashes[2], src_slot=0)
    assert op is not None
    assert bm.probe_local_prefix(probe) == 0  # still copying
    checkpoints.store.complete_inflight()
    before = bm.pool_occupancy()
    assert bm.probe_local_prefix(probe) == 12
    assert bm.pool_occupancy() == before
    assert probe not in bm._admission_walks
    assert not probe.block_table


def test_probe_utility_reports_unknown_without_killing_engine():
    output = queue.Queue()

    def fail(seq):
        raise RuntimeError("cache metadata unavailable")

    handler = EngineUtilityHandler(
        None, output, scheduler=NS(block_manager=NS(probe_local_prefix=fail))
    )
    handler._execute_utility_command(
        "probe_local_prefix", {"request_id": "probe", "seq": request(), "chain": None}
    )
    _, reply = output.get_nowait()
    assert reply == {"cmd": "probe_local_prefix", "result": None, "request_id": "probe"}


def test_failed_retry_send_uses_normal_terminal_cleanup(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    target = pickle.loads(_add_frame([seq]))[1][0]
    target.dp_cache_retry_reason = "preparation_timeout"

    def failed_send(*args):
        raise RuntimeError("source socket closed")

    mgr._send_control = failed_send
    assert not mgr._retry_cache_migration(1, target)
    assert target.leave_reason == "cache_migration_retry_failed"
    mgr._release_seq_load(seq.id)
    assert not mgr._cache_retry_payloads
    assert not mgr._cache_balancer.requests
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [1000000, 0]


@pytest.mark.parametrize("resident,accepted", [(64 * 1024, False), (100 * 1024, True)])
def test_real_scheduler_gates_before_first_forward(resident, accepted):
    scheduler = Scheduler(
        MockConfig(
            kv_cache_block_size=BLOCK,
            num_kvcache_blocks=4096,
            enable_prefix_caching=True,
            max_model_len=128 * 1024,
            max_num_batched_tokens=2048,
        )
    )
    bm = scheduler.block_manager
    warm = Sequence(TOKENS[:resident], BLOCK)
    assert bm.allocate(warm, bm.can_allocate(warm))
    bm.hash_blocks(warm, resident)
    bm.deallocate(warm)
    baseline_used = bm.kv.num_used
    seq = Sequence(TOKENS, BLOCK)
    seq.dp_cache_balance = True
    seq.dp_cache_migration = 7
    seq.dp_cache_minimum_hit = PREFIX
    seq.dp_cache_prepare_timeout = 5
    events = []
    scheduler.dp_cache_admission_callback = events.append
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    rejected = scheduler.take_rejected()
    if accepted:
        assert seq.id in batch.req_ids
        assert events == [(seq.id, 7, PREFIX)]
        assert not rejected
    else:
        assert batch is None or seq.id not in batch.req_ids
        assert rejected == [seq]
        assert not events
        assert bm.kv.num_used == baseline_used


def test_multinode_mp_routes_registration_to_host_local_server():
    urls = {
        "0": "host-a:5555",
        "1": "host-a:5555",
        "2": "host-b:5555",
        "3": "host-b:5555",
    }
    cfg = _config(
        dp=2,
        dp_local=1,
        tp=2,
        dp_rank=3,
        enable_dp_attention=True,
        extra={"lmcache.mp.server_urls_by_rank": urls},
    )
    assert deployment._validate_mp_config(cfg) == (2, 1)
    assert deployment._server_urls(cfg) == ["tcp://host-b:5555"]
    cfg.parallel_config.data_parallel_rank = 0
    assert deployment._server_urls(cfg) == ["tcp://host-a:5555"]
    cfg.kv_transfer_config["kv_connector_extra_config"][
        "lmcache.mp.server_urls_by_rank"
    ].pop("2")
    with pytest.raises(ValueError, match="all 4"):
        deployment._validate_mp_config(cfg)


def test_multinode_mp_requires_explicit_local_endpoints():
    with pytest.raises(NotImplementedError, match="server_urls_by_rank"):
        deployment._validate_mp_config(_config(dp=2, dp_local=1))


def test_dp_dummy_forward_reaps_cancelled_alignment_without_more_prefill():
    seq = request()
    seq.status = SequenceStatus.ABORTED
    seq.is_partial_prefill = True
    calls = []

    def forward(*args, **kwargs):
        calls.append("forward_finished")
        return "dummy-output"

    core = NS(
        scheduler=NS(dummy_prefill=False, running=[seq]),
        runner_mgr=NS(call_func=forward),
        _postprocess=lambda seqs, output, batch: calls.append((seqs, output, batch)),
    )
    DPEngineCoreProc._execute_dummy_batch(core)
    assert calls == ["forward_finished", ((), "dummy-output", None)]
