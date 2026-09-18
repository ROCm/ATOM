"""SeqView identity contract.

ATOM's offload scheduler stores the seq object and compares identity to detect
that a request id was reused. Get this wrong and either every step looks like a
new request (load lifecycle reset forever) or a genuinely new request inherits
the previous one's pending load.
"""

from __future__ import annotations

from types import SimpleNamespace

from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry


def _request(rid: str = "r1", prompt=(1, 2, 3)):
    return SimpleNamespace(request_id=rid, prompt_token_ids=list(prompt))


def test_same_request_yields_the_same_view():
    reg = SeqViewRegistry()
    req = _request()

    assert reg.get_or_create(req) is reg.get_or_create(req)


def test_reused_id_with_a_new_request_yields_a_new_view():
    reg = SeqViewRegistry()
    first = reg.get_or_create(_request("r1"))

    second = reg.get_or_create(_request("r1"))  # same id, different request

    assert second is not first, (
        "a recycled request id must present as a new seq, or the new request "
        "inherits the old one's pending load"
    )


def test_mutable_offload_state_survives_across_lookups():
    reg = SeqViewRegistry()
    req = _request()
    view = reg.get_or_create(req)
    view.offload_loaded_tokens = 256
    view.set_block_table([7, 8, 9])

    again = reg.get_or_create(req)

    assert again.offload_loaded_tokens == 256
    assert again.block_table == [7, 8, 9]


def test_prompt_tokens_are_the_key_source_not_decode_output():
    reg = SeqViewRegistry()
    req = _request(prompt=(1, 2, 3))
    req.all_token_ids = [1, 2, 3, 99, 100]  # decode has appended output
    view = reg.get_or_create(req)

    # LMCache keys come from these; letting decode output in would change a
    # prefix's key mid-request and orphan everything already stored.
    assert view.token_ids == [1, 2, 3]
    assert view.num_prompt_tokens == 3


def test_frontier_is_pushed_in_from_vllm():
    reg = SeqViewRegistry()
    view = reg.get_or_create(_request())
    assert view.num_cached_tokens == 0

    view.set_num_cached_tokens(128)

    assert view.num_cached_tokens == 128


def test_drop_forgets_the_request():
    reg = SeqViewRegistry()
    reg.get_or_create(_request("r1"))
    reg.drop("r1")
    assert reg.get("r1") is None and len(reg) == 0


def test_preemption_forgets_placement_but_not_what_was_stored():
    """vLLM reuses the same Request object, so the view has to be reset in place.

    Preemption hands the blocks to somebody else without telling the connector,
    and the save loop sizes its next store from exactly these two fields -- a
    stale block table plus a stale frontier is another request's KV stored under
    this request's token ids.
    """
    reg = SeqViewRegistry()
    req = _request()
    view = reg.get_or_create(req)
    view.set_block_table([7, 8, 9])
    view.set_num_cached_tokens(384)
    view.offload_loaded_tokens = 256
    view.offload_handoff_boundary_tokens = 256
    view.prefix_hashes_published = True

    view.reset_for_preemption()

    assert view.block_table == []
    assert view.num_cached_tokens == 0
    assert view.offload_loaded_tokens == 0
    assert view.offload_handoff_boundary_tokens == 0
    assert view.prefix_hashes_published is False
    # Same view: the request keeps its identity, so ATOM's scheduler must not
    # see this as a recycled request id.
    assert reg.get_or_create(req) is view


def test_view_accepts_the_frozen_placement_the_chunked_scheduler_writes():
    """`__slots__` makes "the scheduler sets an attribute on the seq" a contract.

    ATOM's `ChunkedOffloadSchedulerBase` freezes a finishing request's placement
    onto the seq object so a final save can still be dispatched after vLLM has
    taken the blocks back. ATOM's own `Sequence` has a `__dict__` and absorbs
    that silently; a slotted view raises `AttributeError` instead -- out of
    `request_finished`, which runs on every completed request.

    Calling the real unbound method is the point: a fake scheduler would still
    pass if the base class grew another such attribute tomorrow.
    """
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    reg = SeqViewRegistry()
    view = reg.get_or_create(_request("r1", prompt=(1, 2, 3, 4)))
    view.set_block_table([5, 6])
    view.set_num_cached_tokens(4)

    scheduler = SimpleNamespace(
        _load_lifecycles={},
        _active_load_operations={},
        _load_failed_seqs={},
        _save_tracker={"r1": [view, 0]},
        _early_release=True,
        should_defer_free=lambda seq: False,
    )
    # Bind the real helper rather than stubbing it: it takes the seq, so a stub
    # would hide exactly the kind of attribute access this test exists to catch.
    scheduler._release_failed_load_attempt = (
        ChunkedOffloadSchedulerBase._release_failed_load_attempt.__get__(scheduler)
    )

    ChunkedOffloadSchedulerBase.request_finished(scheduler, view)

    assert view._offload_finished_cached_tokens == 4
    # Popped, because nothing was still deferring the free.
    assert "r1" not in scheduler._save_tracker


def test_preemption_forgets_the_frozen_placement_too():
    """Frozen placement is placement; a preempted request's is equally stale."""
    reg = SeqViewRegistry()
    view = reg.get_or_create(_request())
    view._offload_finished_block_ids = [7, 8, 9]
    view._offload_finished_cached_tokens = 384

    view.reset_for_preemption()

    assert not hasattr(view, "_offload_finished_block_ids")
    assert not hasattr(view, "_offload_finished_cached_tokens")
