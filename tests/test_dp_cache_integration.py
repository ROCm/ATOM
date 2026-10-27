# SPDX-License-Identifier: MIT
"""Real manager wire format and scheduler admission, with CPU cache pools."""

import pickle
from types import SimpleNamespace as NS

import pytest
from conftest import MockConfig
from test_dp_load_balance import _make_mgr, _route
from test_lmcache_mp import _config

from atom.kv_transfer.offload.mp import deployment
from atom.model_engine.engine_core_mgr import _add_frame
from atom.model_engine.prefix_hash import attach_shipped_chains, ship_prompt_hash_chain
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus

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


def test_offline_depth_cannot_bypass_cache_placement(monkeypatch):
    mgr = manager(monkeypatch)
    mgr._bp_depth = 1
    mgr._bp_dispatch_or_refill = lambda **kwargs: pytest.fail(
        "offline request bypassed cache placement"
    )
    seq = request()
    with mgr.offline_backpressure_batch():
        assert _route(mgr, [seq]) == [1]
    assert seq.id in mgr._cache_balancer.requests
    assert seq.dp_cache_migration > 0


@pytest.mark.parametrize("legacy_session_affinity", [False, True])
@pytest.mark.parametrize("partial_snapshot", [None, 64 * 1024])
def test_retry_preserves_add_hash_chain_and_does_not_finish_request(
    monkeypatch, legacy_session_affinity, partial_snapshot
):
    mgr = manager(monkeypatch, legacy_session_affinity=legacy_session_affinity)
    seq = request()
    assert _route(mgr, [seq]) == [1]
    assert mgr._dp_session_owners == {}
    assert mgr._dp_session_prompt_tokens == {}
    frame = pickle.loads(_add_frame([seq]))
    [target] = frame[1]
    target.dp_cache_retry_reason = "recoverable_prefix_too_short"
    if partial_snapshot is not None:
        # A snapshot taken before rejection cleanup does not reserve reusable
        # PAGE/STATE. It must not override fallback to the previous owner.
        target.dp_cache_recoverable_hit = partial_snapshot
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
    pytest.importorskip("aiter", reason="DPEngineCoreProc imports AITER shared memory")
    from atom.model_engine.engine_core import DPEngineCoreProc

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
