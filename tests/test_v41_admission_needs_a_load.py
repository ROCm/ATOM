"""An admitted V4.1 prefix must have a state load behind it, or be refused.

`resolve_load` returns False for two reasons -- the index no longer claims the
boundary, and the request already has a load in flight -- and its caller used
to discard that. The admission then went ahead with PAGE restored at the
boundary and nothing queued to put STATE there, which the forward met as a
stale cursor: measured as `needs state at 7936, found 9657` (the request's own
state from before it was preempted) and `needs state at 7936, found 5`
(another tenant's). The engine does not survive either.

Both reasons became reachable at once when local prefix hits began being
served -- the second one's source comment still said "No in-tree path reaches
this today" -- so these pin the invariant rather than the symptom: queued, or
refused; never neither.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.kda_state import NULL_BLOCK_ID
from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner

INTERVAL = 4096
BLOCK = 256


def make_planner(**kw):
    fields = {
        "hash_block_size": BLOCK,
        "chunk_size": BLOCK,
        "state_interval": INTERVAL,
        "max_num_batched_tokens": INTERVAL * 2,
        "world_size": 1,
    }
    fields.update(kw)
    return V41BoundaryPlanner(**fields)


def make_request(req_id="r0", nblocks=64):
    return SimpleNamespace(
        request_id=req_id, block_hashes=tuple(range(1, nblocks + 1))
    )


def test_a_refusal_is_a_load_that_cannot_resolve():
    """It rides the existing miss path rather than inventing a second one."""
    planner = make_planner()
    request = make_request()
    planner.refuse_admission(request, (11, 12, 13))
    (load,) = planner.take_loads()
    assert load.req_id == "r0"
    assert all(b <= NULL_BLOCK_ID for b in load.block_ids)
    assert load.error_block_ids == (11, 12, 13)
    assert planner.stats()["admission_refused"] == 1


def test_a_refusal_carries_the_blocks_that_must_be_invalidated():
    """Without them vLLM caches a prefix whose window ring never existed."""
    planner = make_planner()
    planner.refuse_admission(make_request(), ())
    (load,) = planner.take_loads()
    assert load.error_block_ids == ()


def test_an_unclaimed_boundary_resolves_to_nothing():
    """The precondition for the caller's refusal: it has to be able to tell."""
    planner = make_planner()
    assert planner.resolve_load(make_request(), INTERVAL, (1, 2)) is False
    assert planner.take_loads() == []


def test_a_second_load_for_one_request_is_declined():
    """Reports are keyed by request id, so two in flight cannot be told apart.

    Reachable since local prefix hits: a request can be admitted, parked, and
    admitted again. The first-ever run that served them died on this.
    """
    planner = make_planner()
    request = make_request()
    planner._index.note_stored(planner.boundary_hash(request.block_hashes, INTERVAL))
    assert planner.resolve_load(request, INTERVAL, (1,)) is True
    assert planner.resolve_load(request, INTERVAL, (1,)) is False


def test_a_claimed_boundary_queues_a_real_load():
    """The positive control: the refusal path must not swallow the good case."""
    planner = make_planner()
    request = make_request()
    planner._index.note_stored(planner.boundary_hash(request.block_hashes, INTERVAL))
    assert planner.resolve_load(request, INTERVAL, (7,)) is True
    (load,) = planner.take_loads()
    assert load.block_ids == ()
    assert planner.stats()["admission_refused"] == 0


@pytest.mark.parametrize("boundary", [INTERVAL, INTERVAL + BLOCK, BLOCK])
def test_every_admitted_boundary_either_loads_or_refuses(boundary):
    """The invariant itself, over the boundaries the cap can hand back.

    A boundary off the interval ladder is one `cap_hit` accepts -- the store
    side offers block-aligned frontiers plus the prompt-end anchor -- so it
    has to be covered here too.
    """
    planner = make_planner()
    request = make_request()
    if not planner.resolve_load(request, boundary, (3,)):
        planner.refuse_admission(request, (3,))
    (load,) = planner.take_loads()
    assert load.req_id == "r0"


def test_speculation_and_this_connector_are_refused_together():
    """Both are supported alone; the pair has never been run.

    The plugin once refused every KV connector for V4.1, and separately every
    speculative method. Both refusals were lifted independently -- DSpark is
    admitted, and this connector is how V4.1 gets prefix caching at all -- so
    the combination became reachable with nobody having measured it. Drafts
    resize every registered plane and let a frontier move backwards, which is
    precisely what the boundary machinery assumes cannot happen.
    """
    import atom.plugin.vllm.kv_transfer.connector as conn

    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3)
    )
    with pytest.raises(ValueError, match="has not been run with"):
        conn._refuse_v41_speculation_with_offload(cfg)


def test_no_speculation_is_not_refused():
    """The positive control: the guard must not refuse the measured path."""
    import atom.plugin.vllm.kv_transfer.connector as conn

    conn._refuse_v41_speculation_with_offload(SimpleNamespace(speculative_config=None))


def test_the_refusal_can_be_overridden_on_purpose():
    """Whoever measures it needs a way in that is not editing this file."""
    import os

    import atom.plugin.vllm.kv_transfer.connector as conn

    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3)
    )
    os.environ["ATOM_V41_ALLOW_SPECULATIVE_OFFLOAD"] = "1"
    try:
        conn._refuse_v41_speculation_with_offload(cfg)
    finally:
        del os.environ["ATOM_V41_ALLOW_SPECULATIVE_OFFLOAD"]
