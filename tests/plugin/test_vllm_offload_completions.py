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

    def __init__(self, defer=()) -> None:
        self.saves: list[str] = []
        self.loads: list[str] = []
        self.failed_loads: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []
        self.defer = set(defer)

    def save_finished_by_request(self, req_id) -> None:
        self.saves.append(str(req_id))

    def load_finished_by_request(self, req_id) -> bool:
        self.loads.append(str(req_id))
        return True

    def load_failed_by_request(self, req_id) -> bool:
        self.failed_loads.append(str(req_id))
        return True

    def should_defer_free(self, seq) -> bool:
        return str(seq.id) in self.defer

    def request_finished(self, seq) -> None:
        self.finished.append(str(seq.id))

    def cancel_pending_load(self, seq) -> None:
        self.cancelled.append(str(seq.id))


def _adapter(world_size: int = 1, defer=()) -> tuple[object, _FakeScheduler]:
    # Built without __init__: constructing it for real needs a VllmConfig and
    # would pull the whole offload stack in, which is not what this covers.
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    scheduler = _FakeScheduler(defer)
    adapter._scheduler = scheduler
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    adapter._deferred_frees = set()
    adapter._releases_in_flight = set()
    adapter._save_reports = {}
    adapter._load_failure_reports = {}
    adapter._world_size = world_size
    return adapter, scheduler


def _output(sending=(), recving=(), worker_meta=None):
    return SimpleNamespace(
        finished_sending=set(sending),
        finished_recving=set(recving),
        kv_connector_worker_meta=worker_meta,
    )


def _worker_meta(saved=None, load_failed=None):
    return connector_mod.AtomOffloadWorkerMetadata(saved, load_failed)


def test_completions_reach_atoms_scheduler():
    adapter, scheduler = _adapter()

    adapter.update_connector_output(
        _output(recving=["b"], worker_meta=_worker_meta(saved={"a": 1}))
    )

    assert scheduler.saves == ["a"]
    assert scheduler.loads == ["b"]


def test_a_save_completes_only_once_every_rank_has_written():
    """Firing on the first rank drops `should_defer_free` under a live reader.

    The blocks are freed the moment the deferral lifts, so a slower rank would
    go on gathering from blocks that already belong to another request.
    """
    adapter, scheduler = _adapter(world_size=4)

    for _ in range(3):
        adapter.update_connector_output(_output(worker_meta=_worker_meta({"a": 1})))
    assert scheduler.saves == []

    adapter.update_connector_output(_output(worker_meta=_worker_meta({"a": 1})))

    assert scheduler.saves == ["a"]


def test_a_mid_decode_save_completes_without_the_request_finishing():
    """The bug this replaces: ATOM keeps one save per request in flight.

    Gating the completion on the request finishing left `_save_inflight`
    occupied for the whole request, so a chunked long prompt offloaded its first
    chunk and silently skipped every chunk after it.
    """
    adapter, scheduler = _adapter()

    adapter.update_connector_output(_output(worker_meta=_worker_meta({"long": 1})))

    assert scheduler.saves == ["long"]
    assert adapter._deferred_frees == set()


def test_a_failed_load_is_not_reported_as_a_successful_one():
    """`load_finished` pops the floor that says the range is NOT persisted."""
    adapter, scheduler = _adapter()

    adapter.update_connector_output(
        _output(recving=["a"], worker_meta=_worker_meta(load_failed={"a": 1}))
    )

    assert scheduler.failed_loads == ["a"]
    assert scheduler.loads == []


def test_one_rank_failing_fails_the_whole_load():
    adapter, scheduler = _adapter(world_size=4)

    adapter.update_connector_output(
        _output(recving=["a"], worker_meta=_worker_meta(load_failed={"a": 1}))
    )

    assert scheduler.failed_loads == ["a"] and scheduler.loads == []


def test_worker_metadata_aggregates_across_ranks():
    first = _worker_meta(saved={"a": 1}, load_failed={"b": 1})

    merged = first.aggregate(_worker_meta(saved={"a": 1, "c": 1}))

    assert merged is first
    assert merged.saved == {"a": 2, "c": 1}
    assert merged.load_failed == {"b": 1}


def test_empty_output_is_harmless():
    adapter, scheduler = _adapter()

    adapter.update_connector_output(
        SimpleNamespace(
            finished_sending=None, finished_recving=None, kv_connector_worker_meta=None
        )
    )

    assert scheduler.saves == [] and scheduler.loads == []


# -- deferred frees ------------------------------------------------------


def _finish(adapter, rid="a"):
    request = SimpleNamespace(request_id=rid, prompt_token_ids=[1, 2, 3])
    adapter._seqs.get_or_create(request)
    return adapter.request_finished(request, [])


def test_a_deferred_free_is_released_once_the_save_lands():
    """The leak that matters: a deferred view holds the prompt token ids.

    vLLM never revisits `request_finished`; it holds the blocks until the
    connector names the id in `finished_sending`. So the check is re-run every
    step, and the release rides out with the next metadata.
    """
    adapter, scheduler = _adapter(defer={"a"})
    assert _finish(adapter) == (True, None)
    assert adapter._deferred_frees == {"a"}
    assert adapter.has_pending_push_work()

    # Still saving: nothing is released, and the engine must keep stepping.
    assert adapter._collect_releases() == []
    assert len(adapter._seqs) == 1

    scheduler.defer.clear()
    assert adapter._collect_releases() == ["a"]

    # The final `request_finished` is what pops ATOM's save tracker.
    assert scheduler.finished == ["a", "a"]
    assert len(adapter._seqs) == 0
    assert adapter._deferred_frees == set()
    # Released, but not yet freed: the worker still has to echo it back.
    assert adapter.has_pending_push_work()

    adapter.update_connector_output(_output(sending=["a"]))

    assert not adapter.has_pending_push_work()


def test_a_release_is_emitted_exactly_once():
    """A second `finished_sending` for the same id trips vLLM's free assertion."""
    adapter, scheduler = _adapter(defer={"a"})
    _finish(adapter)
    scheduler.defer.clear()

    assert adapter._collect_releases() == ["a"]
    assert adapter._collect_releases() == []


def test_a_request_with_nothing_pending_frees_immediately():
    adapter, _ = _adapter()

    assert _finish(adapter) == (False, None)
    assert adapter._deferred_frees == set()
    assert len(adapter._seqs) == 0
    assert not adapter.has_pending_push_work()


# -- preemption ----------------------------------------------------------


def test_preemption_forgets_the_block_table_and_cancels_a_queued_load():
    """vLLM frees the blocks inside `schedule()` and reports nothing.

    Left alone, the SeqView still names blocks that now belong to another
    request, and the save loop stores that request's KV under these token ids.
    """
    adapter, scheduler = _adapter()
    request = SimpleNamespace(request_id="a", prompt_token_ids=[1, 2, 3])
    view = adapter._seqs.get_or_create(request)
    view.set_block_table([4, 5, 6])
    view.set_num_cached_tokens(256)

    preempted = adapter._handle_preempted(SimpleNamespace(preempted_req_ids=["a", "x"]))

    assert preempted == ["a", "x"]
    assert scheduler.cancelled == ["a"]
    assert view.block_table == [] and view.num_cached_tokens == 0


def test_no_preemptions_is_the_common_path():
    adapter, scheduler = _adapter()

    assert adapter._handle_preempted(SimpleNamespace()) == []
    assert scheduler.cancelled == []


# -- worker half ---------------------------------------------------------


class _FakeWorker:
    def __init__(self, **out) -> None:
        self.out = SimpleNamespace(
            finished_loading=out.get("finished_loading", set()),
            failed_loading=out.get("failed_loading", set()),
            finished_saving=out.get("finished_saving", set()),
        )
        self.error_blocks = out.get("error_blocks", set())
        self.fenced: list = []

    def get_finished(self):
        return self.out

    def take_load_error_blocks(self):
        return self.error_blocks

    def wait_for_requests(self, req_ids) -> None:
        self.fenced.append(list(req_ids))


def _worker_adapter(worker):
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    adapter._worker = worker
    adapter._pending_release_ids = []
    adapter._worker_saved = {}
    adapter._worker_load_failed = {}
    return adapter


def test_a_save_travels_back_as_worker_metadata_not_finished_sending():
    """`finished_sending` means "free the blocks", and vLLM asserts on it.

    ATOM's saves land mid-decode, so reporting one there crashed the engine on
    `assert request.is_finished()`.
    """
    adapter = _worker_adapter(_FakeWorker(finished_saving={"a"}))

    finished_sending, finished_recving = adapter.get_finished({"a"})

    assert finished_sending == set() and finished_recving == set()
    assert adapter.build_connector_worker_meta().saved == {"a": 1}
    # Drained, so the next step does not report it a second time.
    assert adapter.build_connector_worker_meta() is None


def test_a_failed_load_wakes_the_request_and_names_its_blocks():
    worker = _FakeWorker(failed_loading={"a"}, error_blocks={3, 4})
    adapter = _worker_adapter(worker)

    _, finished_recving = adapter.get_finished(set())

    assert finished_recving == {"a"}, "the request is parked; not waking it hangs"
    assert adapter.get_block_ids_with_load_errors() == {3, 4}
    assert adapter.build_connector_worker_meta().load_failed == {"a": 1}


def test_the_scheduler_halfs_release_list_is_echoed_once():
    adapter = _worker_adapter(_FakeWorker())
    adapter._pending_release_ids.extend(["a", "b"])

    assert adapter.get_finished(set())[0] == {"a", "b"}
    assert adapter.get_finished(set())[0] == set()


def test_preemption_fences_the_worker_before_the_forward():
    worker = _FakeWorker()
    adapter = _worker_adapter(worker)

    adapter.handle_preemptions(SimpleNamespace(preempted_req_ids=["a"]))
    adapter.handle_preemptions(SimpleNamespace(preempted_req_ids=[]))

    assert worker.fenced == [["a"]], "an empty list must not cost a fence"


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

    adapter._check_promised_loads(
        SimpleNamespace(requests=[SimpleNamespace(req_id="ok")])
    )

    assert adapter._promised_loads == {}
