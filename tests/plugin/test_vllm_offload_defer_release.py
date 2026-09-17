"""Deferred frees against the REAL offload scheduler, not a stand-in.

`test_vllm_offload_completions.py` drives the adapter against a fake scheduler,
so it pins the adapter's own bookkeeping and nothing else. That is exactly the
gap this file covers: the fake's `should_defer_free` is a set lookup, while the
real one also waits on the save's block lease -- and the lease is released by a
channel the adapter used to drop on the floor.

The symptom was not subtle. Every finished request deferred forever, each
keeping its blocks and its SeqView; after ~27 long requests the pool had no free
block left and the server sat at zero running, one waiting on capacity, with
every metric otherwise healthy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

connector_mod = pytest.importorskip(
    "atom.plugin.vllm.kv_transfer.connector",
    reason="the adapter imports vLLM's connector base",
)
dense_mod = pytest.importorskip(
    "atom.kv_transfer.offload.dense.connector",
    reason="the dense scheduler pulls the offload stack",
)

from atom.kv_transfer.disaggregation.types import ConnectorCompletion
from atom.kv_transfer.offload import config as offcfg

BLOCK, CHUNK, WORLD = 64, 64, 4
PROMPT = 256  # four blocks, so two chunks with room for a second save


def _adapter(monkeypatch):
    """The adapter wired to a real `DenseOffloadScheduler`.

    Built without `__init__` for the same reason the sibling file does it: a
    real construction needs a VllmConfig and a live LMCache engine. The
    scheduler half, which is what is under test here, is genuine.
    """
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _c=None: SimpleNamespace(chunk_size=CHUNK),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_a: object())
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_both"},
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
        tensor_parallel_size=WORLD,
    )
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    adapter._scheduler = dense_mod.DenseOffloadScheduler(config)
    adapter._config = config
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    adapter._deferred_frees = set()
    adapter._deferred_free_at = {}
    adapter._next_save_reconcile_at = 0.0
    adapter._releases_in_flight = set()
    adapter._save_reports = {}
    adapter._load_failure_reports = {}
    adapter._completion_reports = {}
    adapter._world_size = WORLD
    return adapter, adapter._scheduler


def _admit(adapter, req_id="r0", prompt=PROMPT):
    request = SimpleNamespace(request_id=req_id, prompt_token_ids=list(range(prompt)))
    seq = adapter._seqs.get_or_create(request)
    seq.set_block_table(list(range(prompt // BLOCK)))
    adapter._scheduler.update_state_after_alloc(seq)
    return request, seq


def _emit_save(adapter, seq, frontier):
    """Advance the computed frontier and let the scheduler dispatch a save."""
    seq.set_num_cached_tokens(frontier)
    meta = adapter._scheduler.build_connector_meta()
    return list(meta.requests)


def _report(adapter, operation, *, ranks=WORLD, succeeded=True, store=True):
    """Replay what the dense worker sends back for one landed save.

    Both channels, because the dense connector emits both: the legacy terminal
    that clears `_save_inflight`, and the connector-owned store event that pops
    the block lease.
    """
    for _ in range(ranks):
        completions = (
            [
                ConnectorCompletion(
                    dense_mod.DENSE_PAGE_STORE_CHANNEL, operation, succeeded
                )
            ]
            if store
            else []
        )
        adapter.update_connector_output(
            SimpleNamespace(
                finished_sending=set(),
                finished_recving=set(),
                kv_connector_worker_meta=connector_mod.AtomOffloadWorkerMetadata(
                    {str(operation.req_id): 1}, {}, completions
                ),
            )
        )


def test_a_landed_save_releases_the_request(monkeypatch):
    """The deadlock, end to end: finish mid-save, then let the save land."""
    adapter, scheduler = _adapter(monkeypatch)
    request, seq = _admit(adapter)
    (save,) = _emit_save(adapter, seq, PROMPT)

    deferred, _ = adapter.request_finished(request, [])
    assert deferred is True, "a save is still reading the blocks"
    assert adapter._collect_releases() == []

    _report(adapter, save.save_operation)

    assert scheduler.should_defer_free(seq) is False
    assert adapter._collect_releases() == ["r0"]
    assert not adapter.has_pending_push_work() or adapter._releases_in_flight


def test_the_operation_lease_does_not_outlive_the_save(monkeypatch):
    """What actually leaked: the per-operation maps only ever grew.

    Each entry pins the SeqView, so this is a block leak AND a memory leak --
    one prompt's token ids per save operation, for the life of the process.
    """
    adapter, scheduler = _adapter(monkeypatch)
    _request, seq = _admit(adapter)

    (first,) = _emit_save(adapter, seq, 128)
    _report(adapter, first.save_operation)
    (second,) = _emit_save(adapter, seq, PROMPT)
    _report(adapter, second.save_operation)

    assert scheduler._save_operation_owner == {}
    assert scheduler._save_operation_safe == {}
    assert scheduler._save_operation_blocks == {}


def test_a_partial_quorum_does_not_release_the_lease(monkeypatch):
    """Releasing on the first rank frees blocks a slower rank is still reading."""
    adapter, scheduler = _adapter(monkeypatch)
    request, seq = _admit(adapter)
    (save,) = _emit_save(adapter, seq, PROMPT)
    adapter.request_finished(request, [])

    _report(adapter, save.save_operation, ranks=WORLD - 1)

    assert scheduler.should_defer_free(seq) is True
    assert adapter._collect_releases() == []

    _report(adapter, save.save_operation, ranks=1)

    assert adapter._collect_releases() == ["r0"]


def test_a_quorum_that_never_completes_is_abandoned(monkeypatch):
    """A rank that never reports must not pin the blocks for the whole run.

    Every gate on the release path waits for all ranks. Before the reconcile,
    that wait was unbounded: one worker dying mid-store, or one rank whose
    `store()` parked inside LMCache and neither returned nor raised, left
    `should_defer_free` true forever. vLLM holds that request's blocks until the
    connector names it in `finished_sending`, so the pool lost that capacity for
    the life of the server while `has_pending_push_work` kept the engine
    stepping over a request that could never finish.
    """
    adapter, scheduler = _adapter(monkeypatch)
    request, seq = _admit(adapter)
    (save,) = _emit_save(adapter, seq, PROMPT)
    adapter.request_finished(request, [])

    # Every rank but one. The quorum can never be reached from here.
    _report(adapter, save.save_operation, ranks=WORLD - 1)
    assert scheduler.should_defer_free(seq) is True

    # Inside the window the deferral stands: reclaiming early would race a copy
    # that is still reading those blocks.
    adapter._reconcile_stale_saves()
    assert adapter._collect_releases() == []
    assert scheduler.should_defer_free(seq) is True

    timeout = connector_mod.offload_save_abandon_timeout_s()
    assert timeout > 0, "the default pin timeout must arm the reconcile"
    adapter._deferred_free_at["r0"] -= timeout + 1
    adapter._next_save_reconcile_at = 0.0

    adapter._reconcile_stale_saves()

    assert scheduler.should_defer_free(seq) is False
    assert adapter._collect_releases() == ["r0"]
    # Nothing left holding the request: not the partial quorum, not the partial
    # save tally, not the SeqView.
    assert adapter._completion_reports == {}
    assert adapter._save_reports == {}
    assert len(adapter._seqs) == 0


def test_the_reconcile_is_off_when_lmcache_pinning_is_disabled(monkeypatch):
    """A non-positive pin timeout disables reclamation, as on the native path.

    The window is safe only because it is derived from LMCache's own pin
    timeout. With pinning off there is no such derivation, so there is no honest
    window, and abandoning would be guesswork against a possibly-live copy.
    """
    adapter, scheduler = _adapter(monkeypatch)
    request, seq = _admit(adapter)
    (save,) = _emit_save(adapter, seq, PROMPT)
    adapter.request_finished(request, [])
    _report(adapter, save.save_operation, ranks=WORLD - 1)

    monkeypatch.setattr(connector_mod, "offload_save_abandon_timeout_s", lambda: 0.0)
    adapter._deferred_free_at["r0"] -= 10_000.0

    adapter._reconcile_stale_saves()

    assert scheduler.should_defer_free(seq) is True
    assert adapter._collect_releases() == []


def test_build_connector_meta_runs_the_reconcile_before_releasing(monkeypatch):
    """Order matters: a save abandoned this step is released this step.

    Reconciling after `_collect_releases` would hold each abandoned request one
    extra step -- harmless on a busy engine, but the failure this bounds is an
    engine with nothing else to run.
    """
    adapter, _scheduler = _adapter(monkeypatch)
    order = []
    monkeypatch.setattr(
        adapter, "_reconcile_stale_saves", lambda: order.append("reconcile")
    )
    monkeypatch.setattr(
        adapter, "_collect_releases", lambda: order.append("collect") or []
    )

    adapter.build_connector_meta(
        SimpleNamespace(
            preempted_req_ids=set(),
            scheduled_new_reqs=(),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=()
            ),
        )
    )

    assert order == ["reconcile", "collect"]


def test_one_rank_failing_fails_the_whole_store(monkeypatch):
    """A range that did not persist everywhere must not be called source-safe."""
    adapter, scheduler = _adapter(monkeypatch)
    _request, seq = _admit(adapter)
    (save,) = _emit_save(adapter, seq, PROMPT)
    operation = save.save_operation

    _report(adapter, operation, ranks=WORLD - 1)
    _report(adapter, operation, ranks=1, succeeded=False)

    # `_store_finished` keeps a failed store's ranges leased rather than
    # advertising them, so the operation's blocks stay recorded as unsafe.
    assert operation in scheduler._save_operation_blocks


def test_the_store_completion_does_not_double_complete_the_save(monkeypatch):
    """Both channels report one save; completing it twice retires a newer one.

    ATOM's native worker treats a handled store event as ALSO a terminal save.
    Here the legacy `saved` tally already did that, so the adapter must ignore
    the return value -- otherwise the second chunk's save generation is retired
    by the first chunk's report.
    """
    adapter, scheduler = _adapter(monkeypatch)
    _request, seq = _admit(adapter)

    (first,) = _emit_save(adapter, seq, 128)
    _report(adapter, first.save_operation)
    (second,) = _emit_save(adapter, seq, PROMPT)

    assert second.save_operation != first.save_operation
    assert scheduler._save_inflight == {"r0": second.save_operation}
