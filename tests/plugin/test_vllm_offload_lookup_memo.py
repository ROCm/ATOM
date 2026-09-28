"""One external-tier lookup per waiting request, not one per scheduler step.

vLLM asks the connector how many tokens the tier can supply before it knows
whether the request can be admitted, and a request that fails allocation is
asked the identical question next step. Why that livelocks a full KV cache, and
what bounds the memo that stops it, is in
`OffloadSchedulerMixin._init_tier_hit_memo`.

The memo lives in ATOM's own scheduler, which has the same retry shape on the
native path; these tests drive the real one through the plugin so the two
halves are exercised together. `tests/test_dense_offload_connector.py` covers
the native caller and the memo's lifetime rules.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

connector_mod = pytest.importorskip(
    "atom.plugin.vllm.kv_transfer.connector",
    reason="the adapter imports vLLM's connector base",
)

CHUNK = 256
HIT = 16 * CHUNK


def _adapter(monkeypatch, calls, *, min_load):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=CHUNK),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    scheduler = DenseOffloadScheduler(
        SimpleNamespace(
            kv_transfer_config={"kv_role": "kv_consumer"},
            kv_cache_block_size=16,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        )
    )
    scheduler._min_load_tokens = min_load

    def lookup(_tokens, lookup_id):
        calls.append(lookup_id)
        return HIT

    scheduler._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    # Built without __init__: constructing it for real needs a VllmConfig and
    # would pull the whole offload stack in.
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    adapter._scheduler = scheduler
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    adapter._requests = {}
    adapter._kda_planner = None
    adapter._attn_group_id = 0
    return adapter


def _request(req_id: str = "r0", ntok: int = 150_000):
    return SimpleNamespace(request_id=req_id, prompt_token_ids=list(range(ntok)))


def _end_of_step(adapter):
    """What every scheduler step does after the connector has answered.

    This is the boundary the bug lived on: building the metadata dispatches the
    cleanup for any lookup no longer backed by a pending load, and a dispatched
    lookup is dropped. Without it a test cannot tell a memo from the in-step
    result that was always there.

    The adapter's own `build_connector_meta` wraps the inner one and feeds the
    result to `_check_promised_loads`; that watchdog is the only thing that ages
    `_promised_loads`, so a step that skips it cannot tell a promise held for
    one step from one held forever. Calling the inner build plus the watchdog
    reproduces that pair without standing up a whole `scheduler_output`.
    """

    inner = adapter._scheduler.build_connector_meta()
    adapter._check_promised_loads(inner)


def test_a_declined_request_stuck_in_waiting_is_looked_up_twice_in_64_steps():
    """The decline is what releases the lookup, and so what used to lose it.

    A hit below the transfer floor is dropped, which clears the pending load,
    which makes the lookup dispatchable -- and a dispatched lookup is gone. The
    request is still at the head of the waiting queue at the same frontier, and
    with the default 8192-token floor this is the common case, not the corner.
    Before the memo that was 64 tier round trips for 64 steps.

    Two, not one: the memo is a bounded optimisation, not a cache. Nothing tells
    this connector that the tier's answer changed, so a remembered hit is
    replayed at most `OFFLOAD_LOOKUP_MEMO_STEPS` (32) times before the question
    is put to the tier again.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)
        request = _request()

        answers = []
        for _ in range(64):
            answers.append(adapter.get_num_new_matched_tokens(request, 0))
            _end_of_step(adapter)

    assert calls == ["r0", "r0"]
    assert answers == [(0, False)] * 64
    assert adapter._promised_loads == {}


def test_a_promised_load_stuck_in_waiting_keeps_its_in_step_result():
    """Parking is no protection: allocation can fail for a parked request too.

    vLLM still has to find blocks for the load to land in, and when it cannot
    the request goes back to waiting at the same frontier like any other.

    This one case was already safe on `main`, and the memo is not what saves it:
    while the `LoadSpec` stays armed, `build_connector_meta` neither dispatches
    nor clears the in-step lookup result, so the next call reads the pin's own
    carrier -- hence the memo counter below is zero. It is here because that
    carrier is also what `_ensure_lookup_pin` consults, so a change to either
    could silently turn this into a per-step lookup. The declined case above is
    the one the memo actually rescues.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=0)
        request = _request()

        answers = []
        for _ in range(64):
            answers.append(adapter.get_num_new_matched_tokens(request, 0))
            _end_of_step(adapter)

    assert calls == ["r0"]
    assert answers == [(HIT, True)] * 64
    assert adapter._scheduler.total_lookups_skipped_by_memo == 0


def test_a_repeatedly_answered_promise_does_not_cry_wolf(caplog):
    """A queued request is not a hung one, however often it is re-answered.

    vLLM parks a request in WAITING_FOR_REMOTE_KVS only after `allocate_slots`
    succeeds; when it fails the request is simply left in `waiting`. The
    watchdog exists for the first case -- a park that no load was ever
    dispatched for -- and it is the per-step reset of the counter that tells the
    two apart, because a request that is merely queued re-answers (and so
    re-promises) on every step.

    That matters more now, not less: the memo makes re-answering cheap, so this
    is exactly the workload the PR targets. Had the promise been recorded with
    `setdefault`, the counter would climb past the grace window and log a
    never-dispatched load for a healthy request -- and then drop it, so the real
    hang it is there to catch would no longer be tracked for that id.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=0)
        request = _request()

        with caplog.at_level(logging.ERROR):
            for _ in range(2 * adapter._PROMISE_GRACE_STEPS):
                adapter.get_num_new_matched_tokens(request, 0)
                _end_of_step(adapter)

    assert "cannot be released" not in caplog.text
    # One, not zero: the last step re-promised and the watchdog then aged that
    # promise once. It never gets further than that, which is the point.
    assert adapter._promised_loads == {"r0": 1}


def test_admission_spends_the_memo():
    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)
        request = _request()

        adapter.get_num_new_matched_tokens(request, 0)
        adapter.update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda: [[]]), 0
        )
        _end_of_step(adapter)
        adapter.get_num_new_matched_tokens(request, 0)

    assert calls == ["r0", "r0"]


def test_a_moved_frontier_is_re_derived_from_the_same_hit():
    """The frontier moves; the tier's answer about the prompt does not.

    How much of the prompt the tier holds is a property of the prompt, so the
    frontier is not part of the question -- it only decides how much of that hit
    is still worth transferring. The answer is derived on every call; only the
    one expensive input is remembered.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)
        request = _request()

        first = adapter.get_num_new_matched_tokens(request, 0)
        _end_of_step(adapter)
        second = adapter.get_num_new_matched_tokens(request, CHUNK)

    assert calls == ["r0"]
    assert first == second == (0, False)


def test_a_reused_request_id_is_a_new_question():
    """Identity, not the id: a new vLLM Request under an old id is new work."""

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)

        adapter.get_num_new_matched_tokens(_request(), 0)
        _end_of_step(adapter)
        # Same id, different request: the memo is keyed on the sequence, so it
        # cannot answer for a lifecycle it was not filled by.
        adapter.get_num_new_matched_tokens(_request(), 0)

    assert calls == ["r0", "r0"]
