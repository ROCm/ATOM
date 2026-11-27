# SPDX-License-Identifier: MIT
"""Session affinity with ATOM_DP_SESSION_PREFIX_OWNER_TOKENS.

Session affinity places a new session on the lightest rank. With the switch
set, a new session goes to the rank whose cache holds the longest finished
prefix of its prompt instead, when that is enough tokens more than the
lightest rank holds and the rank is not too far ahead in requests in flight.
Later turns stay on the session's rank, as before.

Bare ``CoreManager`` instances, as in ``test_dp_load_balance.py``: no engine
cores, sockets or GPU.
"""

import array
import logging
from threading import Lock
from types import SimpleNamespace

import pytest
from prometheus_client.parser import text_string_to_metric_families

from atom.kv_transfer.disaggregation.factory import OffloadTierProfile
from atom.metrics.exporter import AtomMetricsExporter
from atom.model_engine.dp_routing.prefix_spread import PrefixSpreadRouter
from atom.model_engine.engine_core_mgr import (
    PREFIX_OWNER_COUNTERS,
    PREFIX_OWNER_LOG_SESSIONS,
    CoreManager,
)
from atom.model_engine.prefix_hash import ShippedChain

HBS = 128


class _Seq:
    def __init__(self, seq_id, tokens, session=None, first_block=0):
        self.id = seq_id
        self.num_prompt_tokens = tokens
        self.dp_session_id = session
        self.dp_parent_session_id = None
        # Prompts with the same `first_block` are prefixes of one token stream.
        blocks = range(first_block, first_block + tokens // HBS)
        self.prompt_hash_chain = ShippedChain(
            HBS, 0, tokens, 0, array.array("Q", blocks)
        )


def _mgr(n_ranks=2, owner_tokens=16384, extra_reqs=8):
    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "Engine Core Mgr"
    mgr.local_engine_count = n_ranks
    mgr._dp_lb_strategy = "least_tokens"
    mgr._dp_lb_req_equiv = 512
    mgr._dp_session_affinity_enabled = True
    mgr._dp_session_owners = {}
    mgr._dp_session_prompt_tokens = {}
    mgr._dp_route_counters = {
        "affinity_new_total": 0,
        "affinity_owner_hit_total": 0,
        "affinity_spill_total": 0,
        "affinity_parent_ignored_total": 0,
        "explicit_total": 0,
        "load_balanced_total": 0,
    }
    mgr._rank_routed_total = [0] * n_ranks
    mgr._rank_rotation_cursor = 0
    mgr._rank_reqs = [0] * n_ranks
    mgr._rank_tokens = [0] * n_ranks
    mgr._seq_load = {}
    mgr._prefix_router = None
    mgr._lb_lock = Lock()
    if owner_tokens:
        mgr._session_prefix_index = PrefixSpreadRouter(
            32768, OffloadTierProfile(loads=False, min_load_tokens=0), max_seqs=256
        )
        mgr._session_prefix_owner_tokens = owner_tokens
        mgr._session_prefix_owner_extra_reqs = extra_reqs
        mgr._dp_route_counters.update(dict.fromkeys(PREFIX_OWNER_COUNTERS, 0))
    return mgr


def _route(mgr, seq, explicit_rank=None):
    with mgr._lb_lock:
        route = mgr._select_dp_rank_for_seq_locked(seq, explicit_rank)
        mgr._charge_seq_load_locked(seq, route.rank, credit=route.credit)
    return route.rank


def _finish(mgr, seq):
    """First token out, then the request ends."""
    mgr._mark_seq_prefill_complete(seq.id)
    mgr._release_seq(seq.id)


def _count(mgr, name):
    return mgr._dp_route_counters[name]


def _first_session(mgr, tokens=65536):
    """A finished session; returns its request and rank."""
    first = _Seq(1, tokens, session="parent")
    rank = _route(mgr, first)
    _finish(mgr, first)
    return first, rank


def test_new_session_follows_the_rank_that_holds_its_prefix():
    mgr = _mgr()
    _, owner = _first_session(mgr)
    # The owner is the heavier rank now: by load alone the child goes elsewhere.
    mgr._rank_tokens[owner] = 20_000
    child = _Seq(2, 70_000, session="child")
    assert _route(mgr, child) == owner
    assert _count(mgr, "affinity_new_prefix_owner_total") == 1
    assert mgr._dp_session_owners["child"] == owner
    assert _count(mgr, "affinity_prefix_owner_gained_tokens_total") == 65_536
    # Charged its prompt less the prefix it moved for.
    assert mgr._seq_load[2][2] == 70_000 - 65_536


def test_off_places_by_load_alone():
    mgr = _mgr(owner_tokens=0)
    assert mgr._session_prefix_index is None
    _, owner = _first_session(mgr)
    mgr._rank_tokens[owner] = 20_000
    child = _Seq(2, 70_000, session="child")
    assert _route(mgr, child) == 1 - owner
    assert mgr._seq_load[2][2] == 70_000
    assert not set(PREFIX_OWNER_COUNTERS) & set(mgr._dp_route_counters)
    assert _count(mgr, "affinity_new_total") == 2


def test_a_prefix_below_the_threshold_does_not_move_a_session():
    mgr = _mgr()
    _, owner = _first_session(mgr, tokens=8192)
    mgr._rank_tokens[owner] = 20_000
    child = _Seq(2, 70_000, session="child")
    assert _route(mgr, child) == 1 - owner
    assert _count(mgr, "affinity_new_prefix_owner_total") == 0
    assert mgr._seq_load[2][2] == 70_000
    assert _count(mgr, "affinity_prefix_owner_refused_threshold_total") == 1
    assert _count(mgr, "affinity_prefix_owner_refused_guard_total") == 0
    assert _count(mgr, "affinity_prefix_owner_gained_tokens_total") == 0


@pytest.mark.parametrize(
    "owner_tokens, rank, charged", [(16384, 1, 70_000), (8192, 0, 70_000 - 65_536)]
)
def test_the_threshold_counts_what_the_lightest_rank_lacks(owner_tokens, rank, charged):
    """Both ranks hold a prefix: only the difference is a reason to move."""
    mgr = _mgr(owner_tokens=owner_tokens)
    for seq_id, at, tokens in ((1, 0, 65536), (2, 1, 57344)):
        seq = _Seq(seq_id, tokens, session=f"s{seq_id}")
        assert _route(mgr, seq, explicit_rank=at) == at
        _finish(mgr, seq)
    mgr._rank_tokens[:] = [20_000, 0]
    # Rank 0 holds 65536 of the prompt, the lightest rank 57344: 8192 more.
    assert _route(mgr, _Seq(3, 70_000, session="child")) == rank
    assert mgr._seq_load[3][2] == charged


def test_a_prefix_still_being_prefilled_is_not_followed():
    mgr = _mgr()
    first = _Seq(1, 65536, session="parent")
    owner = _route(mgr, first)  # no first token yet
    child = _Seq(2, 70_000, session="child")
    assert _route(mgr, child) == 1 - owner
    assert _count(mgr, "affinity_new_prefix_owner_total") == 0
    mgr._mark_seq_prefill_complete(1)
    sibling = _Seq(3, 70_000, session="sibling")
    mgr._rank_tokens[owner] += 10**6
    assert _route(mgr, sibling) == owner


def test_load_guard_keeps_a_session_off_a_crowded_owner():
    mgr = _mgr(extra_reqs=4)
    _, owner = _first_session(mgr)
    mgr._rank_reqs[owner] = 5  # five more in flight than the other rank
    assert _route(mgr, _Seq(2, 70_000, session="child")) == 1 - owner
    assert _count(mgr, "affinity_new_prefix_owner_total") == 0
    assert _count(mgr, "affinity_prefix_owner_refused_guard_total") == 1
    assert _count(mgr, "affinity_prefix_owner_refused_threshold_total") == 0
    # Four more is inside the guard; the other rank is still the lightest.
    mgr._rank_reqs[:] = [0, 0]
    mgr._rank_reqs[owner] = 4
    mgr._rank_tokens[:] = [0, 0]
    mgr._rank_tokens[owner] = 20_000
    assert _route(mgr, _Seq(3, 70_000, session="sibling")) == owner
    assert _count(mgr, "affinity_new_prefix_owner_total") == 1
    assert _count(mgr, "affinity_prefix_owner_refused_guard_total") == 1
    assert _count(mgr, "affinity_prefix_owner_gained_tokens_total") == 65_536


def test_later_turns_stay_on_the_session_rank():
    mgr = _mgr()
    _, owner = _first_session(mgr)
    child = _Seq(2, 70_000, session="child")
    mgr._rank_tokens[owner] = 20_000
    assert _route(mgr, child) == owner
    _finish(mgr, child)
    mgr._rank_tokens[owner] = 10**6
    turn = _Seq(3, 75_000, session="child")
    assert _route(mgr, turn) == owner
    assert _count(mgr, "affinity_owner_hit_total") == 1
    # As without the switch: only the prompt's growth is charged.
    assert mgr._seq_load[3][2] == 5_000


def test_the_longest_prefix_wins_and_a_tie_goes_to_the_lighter_rank():
    mgr = _mgr(n_ranks=3)
    for seq_id, rank, tokens in ((1, 0, 32768), (2, 1, 65536), (3, 2, 65536)):
        seq = _Seq(seq_id, tokens, session=f"s{seq_id}")
        assert _route(mgr, seq, explicit_rank=rank) == rank
        _finish(mgr, seq)
    mgr._rank_tokens[:] = [0, 30_000, 30_000]
    mgr._rank_reqs[:] = [0, 2, 1]
    assert _route(mgr, _Seq(4, 70_000, session="child")) == 2


def test_an_aborted_prefill_leaves_no_prefix_to_follow():
    mgr = _mgr()
    first = _Seq(1, 65536, session="parent")
    owner = _route(mgr, first)
    mgr._release_seq(1)  # gone before its first token
    mgr._rank_tokens[owner] = 20_000
    assert _route(mgr, _Seq(2, 70_000, session="child")) == 1 - owner
    assert mgr._session_prefix_index.confirmed_hits(first, 2) == {}


def test_reset_keeps_the_index_and_drops_the_sessions():
    mgr = _mgr()
    _, owner = _first_session(mgr)
    mgr.reset_dp_router()
    assert mgr._dp_session_owners == {}
    mgr._rank_tokens[owner] = 20_000
    assert _route(mgr, _Seq(2, 70_000, session="child")) == owner


def test_statistics_carry_the_decision_and_the_index():
    mgr = _mgr()
    _, owner = _first_session(mgr)
    mgr._rank_tokens[owner] = 20_000
    _route(mgr, _Seq(2, 70_000, session="child"))
    stats = mgr.get_dp_router_statistics()
    assert stats["affinity_new_prefix_owner_total"] == 1
    assert stats["affinity_new_total"] == 1
    assert stats["prefix_index"]["indexed_tokens"] > 0


def _config(**overrides):
    values = {
        "enable_prefix_caching": True,
        "pipeline_parallel_size": 1,
        "enable_rapidserve": False,
        "max_num_seqs": 256,
        "kv_transfer_config": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_index_is_built_only_with_the_switch_and_session_affinity(monkeypatch):
    mgr = _mgr(owner_tokens=0)
    monkeypatch.delenv("ATOM_DP_SESSION_PREFIX_OWNER_TOKENS", raising=False)
    assert mgr._build_session_prefix_index(_config()) is None
    monkeypatch.setenv("ATOM_DP_SESSION_PREFIX_OWNER_TOKENS", "16384")
    assert isinstance(mgr._build_session_prefix_index(_config()), PrefixSpreadRouter)
    mgr._dp_session_affinity_enabled = False
    assert mgr._build_session_prefix_index(_config()) is None
    mgr._dp_session_affinity_enabled = True
    assert mgr._build_session_prefix_index(_config(pipeline_parallel_size=2)) is None
    with pytest.raises(ValueError, match="ATOM_DP_SESSION_PREFIX_OWNER_TOKENS"):
        mgr._build_session_prefix_index(_config(enable_prefix_caching=False))


def test_a_session_with_nothing_to_gain_is_not_counted():
    """No prefix anywhere, or the lightest rank already holds the longest."""
    mgr = _mgr()
    _, owner = _first_session(mgr)
    mgr._rank_tokens[1 - owner] = 20_000  # the prefix's rank is the lightest
    assert _route(mgr, _Seq(2, 70_000, session="child")) == owner
    assert _count(mgr, "affinity_new_total") == 2
    assert [_count(mgr, name) for name in PREFIX_OWNER_COUNTERS] == [0, 0, 0, 0]


def _owner_lines(caplog):
    lines = [record.getMessage() for record in caplog.records]
    return [line for line in lines if "dp prefix owner:" in line]


def test_log_line_carries_the_counters_and_is_rate_limited(caplog):
    mgr = _mgr()
    with caplog.at_level(logging.INFO, logger="atom"):
        _, owner = _first_session(mgr)  # the first new session: a line at once
        assert len(_owner_lines(caplog)) == 1
        mgr._rank_tokens[owner] = 20_000
        _route(mgr, _Seq(2, 70_000, session="child"))  # follows; within the minute
        _route(mgr, _Seq(3, 75_000, session="child"))  # a later turn
        assert len(_owner_lines(caplog)) == 1
        mgr._prefix_owner_logged_at -= 61.0
        _route(mgr, _Seq(4, 1024, session="other", first_block=10**6))
        assert len(_owner_lines(caplog)) == 2
        # Many new sessions inside one minute: a line every so many of them.
        mgr._prefix_owner_unlogged_sessions = PREFIX_OWNER_LOG_SESSIONS - 1
        _route(mgr, _Seq(5, 1024, session="another", first_block=2 * 10**6))
    lines = _owner_lines(caplog)
    assert len(lines) == 3
    assert lines[0].endswith(
        "dp prefix owner: new_sessions=1 follows=0 refused_threshold=0 "
        "refused_guard=0 gained_tokens=0 index_tokens=65536"
    )
    assert "new_sessions=3 follows=1 refused_threshold=0 refused_guard=0" in lines[1]
    assert "gained_tokens=65536" in lines[1]
    assert "new_sessions=4 follows=1" in lines[2]


def test_off_logs_nothing(caplog):
    mgr = _mgr(owner_tokens=0)
    with caplog.at_level(logging.INFO, logger="atom"):
        _first_session(mgr)
        _route(mgr, _Seq(2, 70_000, session="child"))
    assert _owner_lines(caplog) == []


def _exported(mgr):
    exporter = AtomMetricsExporter()
    exporter.update({"dp_router": mgr.get_dp_router_statistics()})
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
    }


def test_exporter_publishes_the_counters_only_with_the_switch():
    mgr = _mgr(extra_reqs=4)
    _, owner = _first_session(mgr)
    mgr._rank_reqs[owner] = 5
    _route(mgr, _Seq(2, 70_000, session="refused"))
    mgr._rank_reqs[:] = [0, 0]
    mgr._rank_tokens[:] = [0, 0]
    mgr._rank_tokens[owner] = 20_000
    _route(mgr, _Seq(3, 70_000, session="child"))
    samples = _exported(mgr)
    assert samples[("atom:dp_affinity_new_prefix_owner_total", ())] == 1
    assert samples[("atom:dp_affinity_prefix_owner_gained_tokens_total", ())] == 65_536
    refused = "atom:dp_affinity_prefix_owner_refused_total"
    assert samples[(refused, (("reason", "guard"),))] == 1
    assert samples[(refused, (("reason", "threshold"),))] == 0
    assert samples[("atom:dp_affinity_new_total", ())] == 2
    assert samples[("atom:dp_prefix_index_tokens", ())] > 0

    off = _mgr(owner_tokens=0)
    _first_session(off)
    names = {name for name, _labels in _exported(off)}
    assert "atom:dp_affinity_new_total" in names
    assert not [n for n in names if "prefix_owner" in n or "dp_prefix_index" in n]


def test_confirmed_hits_require_every_ancestor():
    # A completed descendant cannot supply credit across a missing ancestor.
    from types import SimpleNamespace

    router = PrefixSpreadRouter(
        32768, OffloadTierProfile(loads=False, min_load_tokens=0), max_seqs=256
    )
    root, child = object(), object()
    router.index = {root: {0: None}, child: {0: None, 1: None}}
    router._path = lambda chain: [(root, 128), (child, 256)]
    seq = SimpleNamespace(prompt_hash_chain=object())
    assert router.confirmed_hits(seq, 2) == {0: 256}
    router.pending[(root, 0)] = 123
    assert router.confirmed_hits(seq, 2) == {}


@pytest.mark.parametrize("migrate", [False, True])
def test_prefix_owner_load_reader_does_not_enable_session_spill(migrate):
    mgr = _mgr()
    if migrate:
        mgr._session_migrate_margin = 0.25
    _, owner = _first_session(mgr)
    policy = mgr._get_dp_policy()
    assert policy.load_reqs() is mgr._rank_reqs
    assert (policy.rank_reqs is not None) == migrate
    mgr._rank_reqs[:] = [40 if rank == owner else 0 for rank in range(2)]
    turn = _Seq(2, 70_000, session="parent")
    assert _route(mgr, turn) == (1 - owner if migrate else owner)
    assert _count(mgr, "affinity_spill_total") == int(migrate)


def test_session_spill_positional_arguments_still_match_main():
    from atom.model_engine.dp_routing.policies import SessionPolicy

    requests = [10, 0]
    policy = SessionPolicy(
        lambda: 1,
        lambda session: 1,
        {"s": 0},
        {},
        2,
        lambda: None,
        lambda: requests,
        0.25,
    )
    route = policy.select(_Seq(1, 1024, session="s"), None)
    assert route.rank == 1
    assert route.decision == "affinity_spill_total"
    assert policy.load_reqs is None
