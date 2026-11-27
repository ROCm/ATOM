# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The runtime switch that stops an lmcache_mp tier's lookups for new requests
(`POST /offload/lookups`): no lookup, no load plan and no admission wait once
off, while an answer held from before still loads and a load planned before
the switch still re-takes its pin."""

import queue
import time
from types import SimpleNamespace

import pytest
from test_collective_rpc_dispatch import _Engine, _responses, _RunnerMgr
from test_lmcache_mp import _config
from test_lmcache_mp_async_lookup import CHUNK, _Adapter, _Client
from test_lmcache_mp_async_threads import (
    _driven_client,
    _PerRequestClient,
    _wait_until,
)

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import transfer
from atom.kv_transfer.offload.mp.connector import (
    LMCacheMPConnectorScheduler as MPSchedulerShell,
)
from atom.kv_transfer.offload.mp.scheduler import (
    LMCacheMPConnectorScheduler as MPScheduler,
)
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.sequence import SequenceStatus
from tests.lmcache_mp_support import bare_mp_scheduler

PROMPT = 64
HIT = 8 * CHUNK  # what the fake tier answers


def _scheduler(monkeypatch, role="kv_consumer"):
    monkeypatch.setattr(transfer.time, "sleep", lambda _s: None)
    adapter = _Adapter(_Client(chunks=HIT // CHUNK))
    client = mp_lookup._MPLookupClient(
        adapter, config=_config(role=role), timeout=1.0, poll_interval=0.01
    )
    sched = bare_mp_scheduler()
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(role=role), chunk_size=CHUNK, lookup_client=client
    )
    sched._min_load_tokens = 2 * CHUNK
    return sched, client, adapter


def _seq():
    return SimpleNamespace(
        id=7,
        num_tokens=PROMPT,
        num_prompt_tokens=PROMPT,
        num_cached_tokens=0,
        token_ids=list(range(PROMPT)),
        block_table=[],
        offload_loaded=False,
        offload_load_failed=False,
        status=SequenceStatus.WAITING,
    )


def test_lookups_are_on_until_switched_off(monkeypatch):
    sched, _, adapter = _scheduler(monkeypatch)
    sched.prefetch_lookups([_seq()])
    assert len(adapter._client.lookups) == 1


def test_the_switch_adds_nothing_to_a_scheduler_until_it_is_switched_off(
    monkeypatch,
):
    # The off checks live on the instance, so serving with lookups on runs
    # the base methods as it did before the switch existed.
    names = {"_lookup_wanted", "lookup_pending", "lookup_would_defer"}
    assert not names & vars(MPScheduler).keys()
    sched, _, _ = _scheduler(monkeypatch)
    assert not names & vars(sched).keys()
    sched.set_offload_lookups(False)
    assert names <= vars(sched).keys()
    sched.set_offload_lookups(True)
    assert not names & vars(sched).keys()


def test_switched_off_a_new_request_is_not_looked_up_and_leaves_no_state(
    monkeypatch,
):
    sched, client, adapter = _scheduler(monkeypatch)
    seq = _seq()
    assert sched.set_offload_lookups(False) is True

    sched.prefetch_lookups([seq])
    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched.can_detach_lookup(seq, PROMPT - 1) is False
    assert client.submit(seq.token_ids, "7") is False

    # Neither an async nor a synchronous lookup was started.
    assert adapter._client.lookups == []
    assert sched._lookup_in_step == []
    assert "7" not in client._lookups
    assert "7" not in sched._lookup_results
    assert "7" not in sched._load_specs


def test_the_switch_releases_an_answered_lookup_nobody_consumed(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch)
    seq = _seq()
    sched.prefetch_lookups([seq])
    client.pump()

    sched.set_offload_lookups(False)
    assert [(c["start"], c["end"]) for c in adapter.freed] == [(0, HIT)]
    assert sched.get_num_new_matched_tokens(seq) == (0, False)


def test_an_in_flight_lookup_stops_holding_its_request_and_is_released_on_answer(
    monkeypatch,
):
    sched, client, adapter = _scheduler(monkeypatch)
    adapter._client.lookup_future.done = False
    seq = _seq()
    sched.prefetch_lookups([seq])
    assert sched.lookup_pending(seq) is True

    sched.set_offload_lookups(False)
    assert sched.lookup_pending(seq) is False
    assert sched.lookup_would_defer(seq) is False
    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    adapter._client.lookup_future.done = True
    client.pump()
    assert [(c["start"], c["end"]) for c in adapter.freed] == [(0, HIT)]
    assert len(adapter._client.lookups) == 1


def test_the_driver_sends_nothing_once_lookups_are_off():
    fake = _PerRequestClient(chunks=1)
    adapter = _Adapter(fake)
    client = _driven_client(adapter, window=1)
    try:
        client.offer("answered", lambda: list(range(8)))
        assert _wait_until(lambda: "answered" in client._async_hits)
        client.offer("queued", lambda: list(range(8)))  # the window is full

        client.set_enabled(False)
        # The answer is released, not refreshed; the queued offer is dropped.
        assert adapter.freed[0]["end"] == CHUNK
        assert not client._offered and not client._answered_at
        assert client.offer("late", lambda: list(range(8))) is False
        assert client.submit(list(range(8)), "late") is False
        client._driver.notify()
        time.sleep(0.05)
        assert len(fake.lookups) == 1
        assert client.driver_stats["stale_released"] == 0
    finally:
        client.close()


def test_switched_back_on_lookups_resume(monkeypatch):
    sched, _, adapter = _scheduler(monkeypatch)
    sched.set_offload_lookups(False)
    assert sched.set_offload_lookups(True) is True
    sched.prefetch_lookups([_seq()])
    assert len(adapter._client.lookups) == 1


def test_an_answer_held_from_before_the_switch_still_loads(monkeypatch):
    sched, client, _ = _scheduler(monkeypatch)
    seq = _seq()
    sched.prefetch_lookups([seq])
    client.pump()
    before = sched.get_num_new_matched_tokens(seq)
    assert before == (HIT, True)

    sched.set_offload_lookups(False)
    assert sched.get_num_new_matched_tokens(seq) == before
    assert sched._load_specs["7"].lmcache_cached_tokens == HIT


def test_a_plan_without_a_held_answer_is_dropped_by_the_switch(monkeypatch):
    # A remembered hit plans a load with no answer held; once off it must not
    # load after allocation.
    sched, _, adapter = _scheduler(monkeypatch)
    seq = _seq()
    sched._load_specs["7"] = LoadSpec(
        hbm_cached_tokens=0, lmcache_cached_tokens=HIT, can_load=False
    )
    sched.set_offload_lookups(False)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert "7" not in sched._load_specs
    sched.update_state_after_alloc(seq)
    assert "7" not in sched._reqs_need_recv
    assert adapter._client.lookups == []


def test_a_load_planned_before_the_switch_retakes_its_pin_with_one_lookup(
    monkeypatch,
):
    # The one lookup left once off. A load planned before the switch (here
    # from a remembered hit, so with no answer held) asks the tier again,
    # synchronously, when it is dispatched.
    sched, _, adapter = _scheduler(monkeypatch)
    asked = []

    def lookup(request_id, _tokens):
        asked.append(request_id)
        adapter._lookup_results[request_id] = HIT

    adapter.maybe_submit_lookup_request = lookup
    spec = LoadSpec(hbm_cached_tokens=0, lmcache_cached_tokens=HIT, can_load=True)
    sched._load_specs["7"] = spec
    sched.set_offload_lookups(False)

    assert sched._ensure_lookup_pin(_seq(), "7", spec) is True
    assert len(asked) == 1


def test_the_mp_shell_forwards_the_switch_to_its_scheduler(monkeypatch):
    sched, _, _ = _scheduler(monkeypatch)
    shell = MPSchedulerShell.__new__(MPSchedulerShell)
    shell.__dict__["_impl"] = sched
    assert shell.set_offload_lookups(False) is True
    assert sched._lookups_enabled is False


def test_a_saving_scheduler_refuses_the_switch(monkeypatch):
    # Without a lookup its saves would store every prompt from token 0 again.
    sched, _, _ = _scheduler(monkeypatch, role="offload")
    assert sched.set_offload_lookups(False) is False
    assert sched._lookups_enabled is True


@pytest.mark.parametrize(
    "connector,applied",
    [
        (None, False),
        (SimpleNamespace(), False),
        (SimpleNamespace(set_offload_lookups=lambda enabled: False), False),
        (SimpleNamespace(set_offload_lookups=lambda enabled: True), True),
    ],
)
def test_the_utility_command_reports_whether_the_connector_took_it(connector, applied):
    assert _utility_replies(connector, enabled=False) == [
        {
            "cmd": "set_offload_lookups",
            "result": {"enabled": False, "applied": applied},
            "request_id": "u1",
        }
    ]


@pytest.mark.parametrize("args", [{}, {"enabled": "false"}, {"enabled": 0}])
def test_the_utility_command_rejects_an_enabled_that_is_not_a_bool(args):
    # Coerced, a missing value or "false" would switch lookups on.
    seen = []
    connector = SimpleNamespace(
        set_offload_lookups=lambda enabled: seen.append(enabled) or True
    )
    (reply,) = _utility_replies(connector, **args)
    assert "enabled must be a bool" in reply["error"]
    assert "result" not in reply
    assert seen == []


def _utility_replies(connector, **args):
    out = queue.Queue()
    handler = EngineUtilityHandler(
        _RunnerMgr(),
        out,
        label="test",
        scheduler=SimpleNamespace(kv_connector=connector),
    )
    handler.engine = _Engine()
    handler._execute_utility_command(
        "set_offload_lookups", {**args, "request_id": "u1"}
    )
    return _responses(out)


@pytest.mark.parametrize(
    "answers,expected", [((True,), True), ((True, False), False), ((), False)]
)
def test_multi_connector_switches_every_sub_that_has_a_tier(answers, expected):
    seen = []

    def sub(answer):
        return SimpleNamespace(
            set_offload_lookups=lambda enabled: seen.append(enabled) or answer
        )

    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [sub(a) for a in answers] + [SimpleNamespace()]
    assert multi.set_offload_lookups(False) is expected
    assert seen == [False] * len(answers)
