# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""P/D early handoff when P's prefill queue is long.

Once P answers early, D reserves pages and sends its write request at once,
and P fails the request if it then waits too long on its DP rank (the chunked
admission deadline, the worker's lookup timeout). So the dispatch answers
early only while the chosen rank has at most
EARLY_HANDOFF_MAX_QUEUED_PREFILL_TOKENS queued for prefill; otherwise the
request takes exactly the parameters it would carry with the flag off (the
legacy relay, which runs neither clock).

CPU only.
"""

import asyncio
import pickle
from types import SimpleNamespace

import pytest
from test_pd_early_handoff import (
    ROUTER_ASKS,
    _core_mgr,
    _engine,
    _finished,
    _generate,
    make_scheduler_connector,
    p_params,
)

LEGACY = {"do_remote_decode": True, "do_remote_prefill": False}


@pytest.fixture
def api(monkeypatch):
    from atom.entrypoints.openai import api_server

    monkeypatch.setattr(api_server, "_early_handoff_endpoint_cache", [])
    monkeypatch.setattr("atom.utils.network.get_ip", lambda: "10.0.0.1")
    return api_server


@pytest.fixture
def ecm():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core_mgr

    return engine_core_mgr


def _early(seq):
    seq.kv_transfer_params = {
        **LEGACY,
        "chunked_transfer": True,
        "transfer_id": f"early-{seq.id}",
        "pd_early_handoff": True,
    }
    seq.pd_legacy_kv_transfer_params = dict(LEGACY)
    return seq


def _is_early(seq):
    return bool(seq.kv_transfer_params.get("pd_early_handoff"))


class Socket:
    def __init__(self):
        self.messages = []

    def send_multipart(self, parts, copy=False):
        self.messages.append(parts)


def _dp_mgr(n_ranks):
    from test_dp_load_balance import _make_mgr

    mgr = _make_mgr(n_ranks, strategy="least_requests")
    mgr.engine_core_identities = [b"e%d" % r for r in range(n_ranks)]
    mgr.input_sockets = [Socket() for _ in range(n_ranks)]
    return mgr


def _shipped(mgr, rank):
    """The seqs the rank received, in its pickled add frames."""
    out = []
    for msg in mgr.input_sockets[rank].messages:
        _, seqs = pickle.loads(msg[1])
        out.extend(seqs)
    return out


def test_limit_sits_between_measured_windows_and_a_cold_flood(ecm):
    # Measured per rank: windows <= 3.5M, cold-start floods 26-51M.
    assert 3_500_000 < ecm.EARLY_HANDOFF_MAX_QUEUED_PREFILL_TOKENS < 26_000_000


def test_short_queue_answers_early_long_queue_ships_the_legacy_parameters(ecm):
    from test_dp_load_balance import _FakeSeq

    limit = ecm.EARLY_HANDOFF_MAX_QUEUED_PREFILL_TOKENS
    mgr = _dp_mgr(2)
    mgr._rank_tokens = [limit, limit + 1]
    at_limit = _early(_FakeSeq(1, 10, data_parallel_rank=0))
    over = _early(_FakeSeq(2, 10, data_parallel_rank=1))
    mgr._dispatch_to_dp_ranks([at_limit, over])
    assert _is_early(at_limit)
    assert over.kv_transfer_params == LEGACY
    (r0,) = _shipped(mgr, 0)
    (r1,) = _shipped(mgr, 1)
    assert _is_early(r0) and r1.kv_transfer_params == LEGACY
    for s in (at_limit, over, r0, r1):
        assert "pd_legacy_kv_transfer_params" not in vars(s)
    assert mgr._early_declining_ranks == {1}


def test_cold_start_flood_falls_back_once_the_rank_is_full_and_recovers(ecm):
    """A burst over four DP ranks before any prefill completed: each rank
    answers early until more than the limit is queued ahead, then every
    further request goes out with the legacy parameters; once the queue
    drains, early again."""
    from test_dp_load_balance import _FakeSeq

    limit = ecm.EARLY_HANDOFF_MAX_QUEUED_PREFILL_TOKENS
    per_req = limit // 2  # the third request on a rank sees 1.0x, the 4th 1.5x
    mgr = _dp_mgr(4)
    seqs = [_early(_FakeSeq(i, per_req)) for i in range(4 * 6)]
    mgr._dispatch_to_dp_ranks(seqs)
    for rank in range(4):
        shipped = _shipped(mgr, rank)
        assert [_is_early(s) for s in shipped] == [True] * 3 + [False] * 3
        assert all(s.kv_transfer_params == LEGACY for s in shipped[3:])
    assert mgr._early_declining_ranks == {0, 1, 2, 3}

    for s in seqs:  # prefill done everywhere
        mgr._mark_seq_prefill_complete(s.id)
    later = [_early(_FakeSeq(100 + r, 10)) for r in range(4)]
    mgr._dispatch_to_dp_ranks(later)
    assert all(_is_early(s) for s in later)
    assert mgr._early_declining_ranks == frozenset()


def test_a_request_without_a_fallback_is_untouched(ecm):
    from test_dp_load_balance import _FakeSeq

    mgr = _dp_mgr(2)
    mgr._rank_tokens = [10**9, 10**9]
    plain = _FakeSeq(1, 10)
    plain.kv_transfer_params = LEGACY
    mgr._dispatch_to_dp_ranks([plain])
    assert plain.kv_transfer_params is LEGACY
    assert mgr._early_declining_ranks == frozenset()


def test_untracked_load_keeps_the_early_answer(monkeypatch):
    """A single engine or a PP head tracks no per-rank load: the early answer
    is kept, and the fallback is still never shipped."""
    mgr, sent = _core_mgr(monkeypatch, engines=1)
    seq = _early(SimpleNamespace(id=1, stream_callback=None))
    mgr._add_prepared_requests([seq])
    assert _is_early(seq) and "pd_legacy_kv_transfer_params" not in vars(seq)
    assert sent == [0]


def test_declined_request_is_answered_after_its_prefill_as_with_the_flag_off(
    monkeypatch, api, ecm
):
    """The dispatch declined: P waits for the prefill and answers plain JSON
    with the legacy parameters, which carry no chunked metadata and no
    `early_handoff` (exactly the flag-off leg)."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    seq, calls = _engine(api, monkeypatch)
    monkeypatch.setattr(api, "delivered_text", lambda _ids: "")
    preprocess = api.engine.io_processor.preprocess

    def preprocess_keeping_params(*a, **kw):
        out = preprocess(*a, **kw)
        out.kv_transfer_params = kw["kv_transfer_params"]
        return out

    api.engine.io_processor.preprocess = preprocess_keeping_params
    mgr = ecm.CoreManager.__new__(ecm.CoreManager)
    mgr.label = "t"

    def add_request(seqs):
        mgr._settle_early_handoff(
            seqs[0], 2, ecm.EARLY_HANDOFF_MAX_QUEUED_PREFILL_TOKENS + 1
        )
        seqs[0].pd_routed_dp_rank = 2

    api.engine.core_mgr.add_request = add_request

    async def go():
        task = asyncio.ensure_future(_collect(_generate(api, dict(ROUTER_ASKS))))
        await asyncio.sleep(0.02)
        assert not task.done(), "legacy: no answer before the prefill ends"
        assert seq.kv_transfer_params == LEGACY
        assert calls.params["pd_early_handoff"] is True  # what preprocess saw
        _finished(calls)
        (out,) = await task
        assert "_early_prefill" not in out
        response = api._nonstream_response({"x": 1}, out)
        assert response.media_type == "application/json"
        assert api.EARLY_HANDOFF_HEADER not in response.headers

    asyncio.run(go())
    assert calls.aborted == []
    assert 11 not in api.engine.io_processor.requests
    assert "pd_legacy_kv_transfer_params" not in vars(seq)


def test_accepted_request_still_answers_at_dispatch(monkeypatch, api, ecm):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    seq, calls = _engine(api, monkeypatch)
    mgr = ecm.CoreManager.__new__(ecm.CoreManager)
    mgr.label = "t"

    def add_request(seqs):
        mgr._settle_early_handoff(seqs[0], 2, 0)
        seqs[0].pd_routed_dp_rank = 2

    api.engine.core_mgr.add_request = add_request

    async def go():
        (out,) = [o async for o in _generate(api, dict(ROUTER_ASKS))]
        assert out["kv_transfer_output_meta_info"]["chunked_transfer"] is True
        assert "pd_legacy_kv_transfer_params" not in vars(seq)
        _finished(calls)
        await out["_early_prefill"].task

    asyncio.run(go())


async def _collect(gen):
    return [o async for o in gen]


def test_legacy_leg_registers_no_chunked_admission(monkeypatch, seq_factory):
    """On P's scheduler the fallback leg is admitted at once and never enters
    the chunked admission set, so neither the admission deadline nor the
    worker's orphan lookup (fed by `chunked_admissions`) can fail it; the
    early leg does enter it."""
    p = make_scheduler_connector(monkeypatch, True, early=True)
    legacy = seq_factory(list(range(8)), kv_transfer_params=dict(LEGACY))
    assert p.admission_ready(legacy) is True
    assert legacy.id not in p._chunked_waiting
    assert legacy.id not in p._pending_chunked_admissions

    early = seq_factory(list(range(8)), kv_transfer_params=p_params())
    p.admission_ready(early)
    assert early.id in p._chunked_waiting
    assert early.id in p._pending_chunked_admissions
