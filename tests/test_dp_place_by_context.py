# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_DP_PLACE_BY_CONTEXT: new sticky sessions go to the DP rank with the least
attention load (queued prefill cost + CTX_WEIGHT x owned live context)."""

import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import pytest

ecm = pytest.importorskip("atom.model_engine.engine_core_mgr")


def _router(n=4, idle_s=120.0):
    r = object.__new__(ecm.CoreManager)
    r.label = "test"
    r._lb_lock = threading.Lock()
    r.global_engine_count = r.local_engine_count = n
    r._rank_rotation_cursor = 0
    r._dp_lb_strategy = "least_requests"
    r._dp_lb_req_equiv = 512
    r._dp_session_affinity_enabled = True
    r._dp_session_owners = {}
    r._dp_session_prompt_tokens = {}
    r._dp_route_counters = {
        k: 0
        for k in (
            "affinity_new_total",
            "affinity_owner_hit_total",
            "affinity_spill_total",
            "affinity_parent_ignored_total",
            "explicit_total",
            "load_balanced_total",
        )
    }
    r._rank_routed_total = [0] * n
    r._rank_reqs = [0] * n
    r._rank_tokens = [0] * n
    r._seq_load = {}
    r._dp_place_by_context = True
    r._dp_place_token_weight = 53000.0
    r._dp_place_ctx_weight = 564.0
    r._dp_place_idle_s = idle_s
    r._rank_prefill_cost = [0.0] * n
    r._rank_owned_ctx = [0] * n
    r._seq_prefill_cost = {}
    r._live_sessions = OrderedDict()
    r._dp_place_count = 0
    return r


def _route(r, seq_id, session, prompt):
    seq = SimpleNamespace(
        id=seq_id,
        dp_session_id=session,
        dp_parent_session_id=None,
        num_prompt_tokens=prompt,
    )
    with r._lb_lock:
        rank = r._select_dp_rank_for_seq_locked(seq, None)
        r._charge_seq_load_locked(seq, rank)
    return rank


def test_new_session_avoids_the_rank_owning_long_context():
    r = _router()
    # four sessions, one a rank; the one on rank 0 is long
    owners = [_route(r, i, f"s{i}", 150_000 if i == 0 else 20_000) for i in range(4)]
    assert sorted(owners) == [0, 1, 2, 3]
    long_rank = owners[0]
    for i in range(4):
        r._mark_seq_prefill_complete(i)  # prefill done: only owned context remains
    # rank long_rank now has fewer in-flight requests than the others ...
    r._release_seq_load(0)
    # ... but owns the most context, so a new session must not land there
    assert _route(r, 10, "new", 5_000) != long_rank


def test_existing_session_stays_on_its_owner():
    r = _router()
    first = _route(r, 1, "conv", 30_000)
    r._mark_seq_prefill_complete(1)
    r._release_seq_load(1)
    for k in range(5):  # other sessions pile onto the owner's competitors or the owner
        _route(r, 100 + k, f"other{k}", 200_000)
    assert _route(r, 2, "conv", 31_000) == first


def test_prefill_cost_is_released_and_context_kept():
    r = _router()
    rank = _route(r, 1, "conv", 10_000)
    assert r._rank_prefill_cost[rank] > 0 and r._rank_owned_ctx[rank] == 10_000
    r._mark_seq_prefill_complete(1)
    assert r._rank_prefill_cost[rank] == 0
    r._release_seq_load(1)
    assert r._rank_owned_ctx[rank] == 10_000  # live until idle
    # a later turn charges only its growth, against the cached prefix
    _route(r, 2, "conv", 12_000)
    assert r._rank_owned_ctx[rank] == 12_000
    assert r._rank_prefill_cost[rank] == pytest.approx(2_000 * (53000 + 10_000 + 1_000))


def test_idle_sessions_expire_from_owned_context():
    r = _router(idle_s=0.05)
    rank = _route(r, 1, "old", 80_000)
    r._mark_seq_prefill_complete(1)
    r._release_seq_load(1)
    time.sleep(0.1)
    _route(r, 2, "fresh", 1_000)  # placement expires idle sessions first
    assert r._rank_owned_ctx[rank] in (0, 1_000)
    assert "old" not in r._live_sessions
