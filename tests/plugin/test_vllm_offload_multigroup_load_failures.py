"""Which channel a load failure travels on, and why the choice is forced.

vLLM 0.31 raises when a connector returns a non-empty `invalid_block_ids` for
a model with more than one KV cache group -- block ids are unique only within
a group, so a flat set cannot be mapped back to requests. K3 is multi-group
today; V4.1 becomes multi-group the moment its state images get groups of
their own. The failure mode is an engine death on the first failed load, which
is exactly the path this connector exists to make survivable.

Both directions matter and both are tested: dropping the failures (engine
serves never-written KV as real) and reporting them twice (a prefix that the
block channel would only have trimmed gets discarded whole).
"""

import pytest


class _Worker:
    def __init__(self, blocks=(), reqs=()):
        self._blocks, self._reqs = set(blocks), set(reqs)

    def take_load_error_blocks(self):
        out, self._blocks = self._blocks, set()
        return out

    def take_load_error_reqs(self):
        out, self._reqs = self._reqs, set()
        return out


class _Cfg:
    def __init__(self, n):
        self.kv_cache_groups = tuple(object() for _ in range(n))


def _connector(groups, blocks=(), reqs=(), finished=((), ())):
    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = object.__new__(AtomLMCacheOffloadConnector)
    c._worker = _Worker(blocks, reqs)
    c._kv_cache_config = _Cfg(groups)
    c._kda_error_blocks = set()
    c._kda_error_reqs = set()
    c._abandoned_error_blocks = set()
    c._abandoned_loads = set()
    c.get_finished = lambda ids: finished
    return c


class TestSingleGroupKeepsTheBlockChannel:
    def test_blocks_are_reported(self):
        c = _connector(1, blocks={5, 6})
        assert c.get_block_ids_with_load_errors() == {5, 6}

    def test_and_failed_recving_stays_empty(self):
        # Naming them again here would discard the whole external prefix,
        # where the block channel only truncates to the first bad block.
        c = _connector(1, reqs={"r0"}, finished=({"s"}, {"r0"}))
        res = c.get_transfer_results(set())
        assert res.failed_recving == set()
        assert res.finished_recving == {"r0"}


class TestMultiGroupMustUseTheRequestChannel:
    def test_the_block_channel_goes_silent(self):
        # Not "returns whatever it had": 0.31 raises on a non-empty set.
        c = _connector(2, blocks={5, 6})
        assert c.get_block_ids_with_load_errors() == set()

    def test_the_failures_are_not_lost_they_move(self):
        c = _connector(2, reqs={"r0"})
        assert c.get_transfer_results(set()).failed_recving == {"r0"}

    def test_a_failed_request_is_not_also_reported_finished(self):
        # vLLM would cache the external prefix of a request it sees in both.
        c = _connector(2, reqs={"r0"}, finished=((), {"r0", "r1"}))
        res = c.get_transfer_results(set())
        assert res.failed_recving == {"r0"}
        assert res.finished_recving == {"r1"}

    def test_the_block_sets_are_drained_not_stashed(self):
        """Draining matters even though the answer is empty.

        If the sets accumulated while silent, they would all surface the first
        time the group count read as one -- blocks belonging to requests that
        finished many steps earlier.
        """
        c = _connector(2, blocks={5})
        c._kda_error_blocks = {7}
        c._abandoned_error_blocks = {9}
        assert c.get_block_ids_with_load_errors() == set()
        c._kv_cache_config = _Cfg(1)
        assert c.get_block_ids_with_load_errors() == set()


def test_vllm_still_refuses_block_ids_for_multiple_groups():
    """The upstream condition this whole file is shaped around.

    If 0.31 ever lifts the restriction, this test fails and the split can be
    removed rather than carried on faith.
    """
    import inspect

    from vllm.v1.core.sched.scheduler import Scheduler

    src = inspect.getsource(Scheduler._handle_invalid_blocks)
    assert "kv_cache_groups) > 1" in src
    assert "failed_recving" in src


@pytest.mark.parametrize("groups", [1, 2, 12])
def test_the_group_count_is_read_not_assumed(groups):
    c = _connector(groups)
    assert c._multi_group_failures() is (groups > 1)


def test_a_connector_without_a_config_reads_as_single_group():
    """Hand-built connectors keep the old channel rather than raising.

    The base class sets `_kv_cache_config` in `__init__`, so this only ever
    describes a fixture. Asserting it keeps the robustness from drifting into
    "the group count is optional in production", which it is not.
    """
    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = object.__new__(AtomLMCacheOffloadConnector)
    assert c._multi_group_failures() is False
