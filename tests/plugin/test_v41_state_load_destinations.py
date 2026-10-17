# SPDX-License-Identifier: MIT
"""Where a V4.1 checkpoint riding the PAGE object is loaded to, or saved from."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector


def _ride_connector():
    """A ride-mode connector whose planner records what it is asked."""
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = True
    c._offload_backend = "mp"
    c._v41_checkpoint_group_ids = (1,)
    c._mamba_groups = ()
    c._attn_group_id = 0
    c._requests = {}
    c._abandoned_loads = set()
    c._promised_loads = {}
    c._kda_planner = None
    c._scheduler = SimpleNamespace(update_state_after_alloc=lambda seq: None)
    seq = SimpleNamespace(num_cached_tokens=0, set_block_table=lambda ids: None)
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._v41_planner = SimpleNamespace(resolve_load=Mock(return_value=False))
    return c


def _blocks():
    return SimpleNamespace(get_block_ids=lambda: ([3, 4], [7, 8]))


def test_an_external_hit_names_its_checkpoint_destination():
    c = _ride_connector()
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), _blocks(), 512)
    c._v41_planner.resolve_load.assert_called_once()
    _, kwargs = c._v41_planner.resolve_load.call_args
    assert kwargs["group_blocks"] == ([3, 4], [7, 8])


def test_an_unresolved_ride_withdraws_nothing_itself():
    """`_abandoned_loads` is reported as `finished_recving`, once per id.

    A refused ride used to withdraw its promise here as well. The scheduler
    already declines the load when `take_ride_state` has no destination, and
    `_check_promised_loads` withdraws the promise it leaves behind -- so the
    request was released twice: once into RUNNING, and the second report hit
    vLLM's `assert RequestStatus.is_finished(req.status)` in
    `_update_from_kv_xfer_finished`. Measured as a bare AssertionError on the
    second gsm8k pass of a DSpark run.
    """
    c = _ride_connector()
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), _blocks(), 256)
    assert c._abandoned_loads == set()


def test_no_external_tokens_is_no_load():
    """A purely local admission has nothing for the tier to restore."""
    c = _ride_connector()
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), _blocks(), 0)
    c._v41_planner.resolve_load.assert_not_called()


def _ride_planner():
    from atom.plugin.vllm.kv_transfer.page_ride import PageRideMixin

    class _P(PageRideMixin):
        hash_block_size = 256
        ride_boundary_grid = 256
        ride_group_ids = (1,)

        def __init__(self):
            self._counters = {}
            self._init_page_ride()

    p = _P()
    block = SimpleNamespace(block_id=42)
    p._pool = SimpleNamespace(
        get_cached_block=lambda h, gids: [block],
        touch=lambda blocks: None,
        blocks={42: block},
    )
    return p


def test_a_load_rides_only_on_its_recorded_destination():
    """A load with nothing recorded has no correct destination.

    The committed source blocks a save would carry belong to whoever stored
    that prefix; loading into them is a write into somebody else's state, and
    it also put a load in flight for a request whose admission was refused.
    """
    p = _ride_planner()
    request = SimpleNamespace(block_hashes=tuple(range(1, 9)))
    assert p.take_ride_state(request, "r0", 512, operation=None) is None
    assert p._counters["ride_missing_load_unrecorded"] == 1
    p._record_ride_boundary("r0", 512, (7,))
    got = p.take_ride_state(request, "r0", 512, operation=None)
    assert tuple(got.block_ids) == (7,)


def test_a_save_still_rides_on_the_committed_source():
    p = _ride_planner()
    request = SimpleNamespace(block_hashes=tuple(range(1, 9)))
    got = p.take_ride_state(request, "r0", 512, operation="save-op")
    assert tuple(got.block_ids) == (42,)
    assert p._counters["ride_pinned"] == 1


@pytest.mark.parametrize("boundary", [500, 0])
def test_a_save_off_the_grid_carries_nothing(boundary):
    p = _ride_planner()
    request = SimpleNamespace(block_hashes=tuple(range(1, 9)))
    assert p.take_ride_state(request, "r0", boundary, operation="op") is None
