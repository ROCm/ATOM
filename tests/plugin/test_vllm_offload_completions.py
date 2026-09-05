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
