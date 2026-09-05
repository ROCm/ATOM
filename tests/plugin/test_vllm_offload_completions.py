"""The scheduler half's only news of what the worker finished.

vLLM runs a connector as two objects in two processes: the worker sees ATOM's
completion objects, the scheduler sees `update_connector_output` and a set of
plain request-id strings. Miss that hook and nothing on the scheduler side ever
clears -- including the SeqView of every deferred request, each pinning that
request's prompt token ids.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

connector_mod = pytest.importorskip(
    "atom.plugin.vllm.kv_transfer.connector",
    reason="the adapter imports vLLM's connector base",
)


class _FakeScheduler:
    """Records the resolver calls; those are the contract under test."""

    def __init__(self) -> None:
        self.saves: list[str] = []
        self.loads: list[str] = []

    def save_finished_by_request(self, req_id) -> None:
        self.saves.append(str(req_id))

    def load_finished_by_request(self, req_id) -> bool:
        self.loads.append(str(req_id))
        return True


def _adapter() -> tuple[object, _FakeScheduler]:
    # Built without __init__: constructing it for real needs a VllmConfig and
    # would pull the whole offload stack in, which is not what this covers.
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    scheduler = _FakeScheduler()
    adapter._scheduler = scheduler
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    return adapter, scheduler


def _output(sending=(), recving=()):
    return SimpleNamespace(finished_sending=set(sending), finished_recving=set(recving))


def test_completions_reach_atoms_scheduler():
    adapter, scheduler = _adapter()

    adapter.update_connector_output(_output(sending=["a"], recving=["b"]))

    assert scheduler.saves == ["a"]
    assert scheduler.loads == ["b"]


def test_a_finished_save_releases_the_seq_view():
    """The leak that matters: a deferred view holds the prompt token ids."""
    adapter, _ = _adapter()
    request = SimpleNamespace(request_id="a", prompt_token_ids=[1, 2, 3])
    adapter._seqs.get_or_create(request)
    assert len(adapter._seqs) == 1

    adapter.update_connector_output(_output(sending=["a"]))

    assert len(adapter._seqs) == 0


def test_empty_output_is_harmless():
    adapter, scheduler = _adapter()

    adapter.update_connector_output(
        SimpleNamespace(finished_sending=None, finished_recving=None)
    )

    assert scheduler.saves == [] and scheduler.loads == []


class _ParkScheduler:
    """A scheduler that reports a hit and then declines to load it."""

    def __init__(self, hit: int, park: bool) -> None:
        self._hit = hit
        self._park = park
        self.asked_park = 0

    def get_num_new_matched_tokens(self, seq):
        return self._hit, True

    def should_park_for_load_after_alloc(self, seq) -> bool:
        self.asked_park += 1
        return self._park


def _lookup_adapter(scheduler):
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    adapter._scheduler = scheduler
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    return adapter


def _req(rid="r1", prompt_len=4096):
    return SimpleNamespace(request_id=rid, prompt_token_ids=list(range(prompt_len)))


def test_a_hit_atom_will_not_load_is_not_promised():
    """The deadlock: vLLM parks on async=True and only the worker can release.

    ATOM drops a hit that is below its transfer floor or not chunk aligned. If
    the promise has already been made, nothing ever reports the load, the
    request sits in WAITING_FOR_REMOTE_KVS forever and the engine spins with
    every GPU idle.
    """
    scheduler = _ParkScheduler(hit=2560, park=False)
    adapter = _lookup_adapter(scheduler)

    assert adapter.get_num_new_matched_tokens(_req(), 0) == (0, False)
    assert scheduler.asked_park == 1


def test_a_hit_atom_will_load_is_promised_async():
    scheduler = _ParkScheduler(hit=10496, park=True)
    adapter = _lookup_adapter(scheduler)

    assert adapter.get_num_new_matched_tokens(_req(), 0) == (10496, True)


def test_no_hit_does_not_ask_about_parking():
    scheduler = _ParkScheduler(hit=0, park=True)
    adapter = _lookup_adapter(scheduler)

    assert adapter.get_num_new_matched_tokens(_req(), 0) == (0, False)
    assert scheduler.asked_park == 0


def test_a_promise_that_never_dispatches_is_named(caplog):
    """The hang leaves no trace of its own; this is the only breadcrumb."""
    scheduler = _ParkScheduler(hit=10496, park=True)
    adapter = _lookup_adapter(scheduler)
    adapter.get_num_new_matched_tokens(_req("stuck"), 0)

    empty = SimpleNamespace(requests=[])
    with caplog.at_level("ERROR", logger="atom"):
        for _ in range(adapter._PROMISE_GRACE_STEPS + 1):
            adapter._check_promised_loads(empty)

    assert "stuck" in caplog.text
    # Reported once, not every step afterwards.
    caplog.clear()
    adapter._check_promised_loads(empty)
    assert caplog.text == ""


def test_a_dispatched_load_is_not_reported():
    scheduler = _ParkScheduler(hit=10496, park=True)
    adapter = _lookup_adapter(scheduler)
    adapter.get_num_new_matched_tokens(_req("ok"), 0)

    adapter._check_promised_loads(SimpleNamespace(requests=[SimpleNamespace(req_id="ok")]))

    assert adapter._promised_loads == {}
