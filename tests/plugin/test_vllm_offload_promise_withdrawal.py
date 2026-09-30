"""A parked request must always get a report, even when its load is declined.

On the vLLM plugin path a request is parked in WAITING_FOR_REMOTE_KVS by
``get_num_new_matched_tokens``, which runs *before* ``allocate_slots``.  The
decision to actually issue the load is taken later, in
``build_connector_meta``, and it evaluates conditions the park gate never saw:
a lookup pin the tier has since lost, or -- on hybrid models -- a recurrent
boundary state evicted between the lookup and the dispatch.  Those paths
decline the load and emit nothing.  Nothing then releases the park: the engine
spins in ``schedule()`` with every GPU idle.

ATOM's own engine is safe here because it parks a sequence only when it
registers the load operation, so its declines really are before the park.  The
comment on the ``no_recurrent_state`` path says exactly that, and it is true --
for the other engine.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

BLOCK = 16


def _scheduler_half(block_table):
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._promised_loads = {}
    c._config = SimpleNamespace(kv_cache_block_size=BLOCK)
    c._seqs = {"7": SimpleNamespace(block_table=list(block_table))}
    c._scheduler = SimpleNamespace(
        last_load_skip_reason=lambda _rid: "no_recurrent_state"
    )
    return c


def _meta(requests):
    return SimpleNamespace(requests=requests)


def _promise(c, start, need):
    c._promised_loads["7"] = [0, start, need]


def _run_steps(c, meta, n):
    out = []
    for _ in range(n):
        out.extend(c._check_promised_loads(meta))
    return out


def test_save_only_meta_does_not_settle_a_load_promise():
    """The blinding that hid this: a save rides the same ``requests`` list."""
    c = _scheduler_half(range(100, 116))
    _promise(c, start=32, need=64)
    save_only = _meta(
        [SimpleNamespace(req_id="7", load_operation=None, load_spec=None)]
    )

    withdrawn = _run_steps(c, save_only, c._PROMISE_GRACE_STEPS)

    assert withdrawn == [], "withdrawn before the grace window elapsed"
    assert "7" in c._promised_loads, "a save settled a load promise"

    withdrawn = c._check_promised_loads(save_only)

    assert [r for r, _ in withdrawn] == ["7"]
    assert c._promised_loads == {}


def test_withdrawal_names_the_blocks_the_promise_reserved():
    """An empty block set would have vLLM serve never-written blocks as KV."""
    c = _scheduler_half(range(100, 116))
    _promise(c, start=32, need=64)

    withdrawn = _run_steps(c, _meta([]), c._PROMISE_GRACE_STEPS + 1)

    assert len(withdrawn) == 1
    req_id, blocks = withdrawn[0]
    assert req_id == "7"
    # [32, 96) at block 16 -> table indices [2, 6).
    assert blocks == [102, 103, 104, 105]


def test_a_dispatched_load_settles_the_promise():
    c = _scheduler_half(range(100, 116))
    _promise(c, start=32, need=64)
    loaded = _meta(
        [SimpleNamespace(req_id="7", load_operation=object(), load_spec=None)]
    )

    assert _run_steps(c, loaded, c._PROMISE_GRACE_STEPS + 5) == []
    assert c._promised_loads == {}


@pytest.mark.parametrize("missing_seq", [True, False])
def test_withdrawal_survives_a_missing_seqview(missing_seq):
    """No blocks is wrong, but a permanent park is worse."""
    c = _scheduler_half(range(100, 116))
    if missing_seq:
        c._seqs = {}
    _promise(c, start=32, need=64)

    withdrawn = _run_steps(c, _meta([]), c._PROMISE_GRACE_STEPS + 1)

    assert [r for r, _ in withdrawn] == ["7"]
    assert (withdrawn[0][1] == []) is missing_seq


def _worker_half():
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._abandoned_loads = set()
    c._abandoned_error_blocks = set()
    c._kda_error_blocks = set()
    c._kda_tier = None
    c._kda_planner = None
    c._pending_release_ids = []
    c._worker_saved = {}
    c._worker_load_failed = {}
    c._worker_completions = []
    c._worker = SimpleNamespace(
        get_finished=lambda: SimpleNamespace(
            finished_loading=set(),
            failed_loading=set(),
            finished_saving=set(),
            connector_completions=set(),
        ),
        take_load_error_blocks=lambda: set(),
    )
    return c


def test_worker_half_releases_the_park_and_reports_the_blocks():
    c = _worker_half()
    c._abandoned_loads.add("7")
    c._abandoned_error_blocks.update({102, 103})

    _, finished_recving = c.get_finished(set())

    assert finished_recving == {"7"}, "the park is released by nothing else"
    assert c.get_block_ids_with_load_errors() == {102, 103}
    # Reported once: a second step must not re-truncate a live request.
    assert c.get_finished(set())[1] == set()
    assert c.get_block_ids_with_load_errors() == set()
