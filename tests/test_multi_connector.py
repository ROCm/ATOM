# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the composite ``multi`` KV connector.

Pure-Python: sub-connectors are mocked, so no GPU / lmcache / moriio runtime is
needed. Covers the merge strategy (first-hit-wins, fan-out, metadata routing,
completion union) and the send/save pairing that protects a producer node's
blocks from being freed while a transfer is still reading them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.multi import multi_connector as mc_module
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnector,
    MultiConnectorMetadata,
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    ConnectorMetadata,
    KVConnectorOutput,
    SaveOperationId,
)

# ---------------------------------------------------------------------------
# Mock sub-connectors
# ---------------------------------------------------------------------------


class FakeSchedSub:
    """Scheduler-side sub-connector mock."""

    def __init__(
        self,
        *,
        match=(0, False),
        is_producer=False,
        is_offload=False,
        offload_methods=False,
        has_state_tier=None,
    ):
        self._match = match
        self.is_producer = is_producer
        if is_offload:
            self.is_offload = True
        self.alloc_calls = []
        self.finished_calls = []
        self.meta = ConnectorMetadata()
        self._offload = offload_methods
        # The real `LMCacheOffloadConnectorScheduler` shell defines the whole
        # state face unconditionally and exposes `has_state_tier` to say whether
        # its `_impl` actually carries the tier. An offload shell always has the
        # flag; a producer like moriio has no such attribute at all. Default a
        # tier-carrying shell to True so existing offload subs keep routing.
        if offload_methods:
            self.has_state_tier = (
                offload_methods if has_state_tier is None else has_state_tier
            )

        if offload_methods:
            self.park = False
            self.partial_park = False
            self.defer = False
            self.chunk_ret = None
            self.saved = []
            self.load_failed_ids = []
            self.pending = False
            self.state_loads = []
            self.state_stores = []
            self.state_reports = (set(), set())
            self.state_source_releases = set()
            self.pending_load = False
            self.cancelled = []

    def get_num_new_matched_tokens(self, seq):
        # Model the offload lookup's side effect: a matching prefix arms a load
        # (LMCache pin + _load_specs) that update_state_after_alloc would later
        # turn into a real recv. A miss arms nothing.
        if self._offload and self._match[0] > 0:
            self.pending_load = True
        return self._match

    def build_connector_meta(self):
        return self.meta

    def update_state_after_alloc(self, seq):
        self.alloc_calls.append(seq)

    def request_finished(self, seq):
        self.finished_calls.append(seq)

    # offload-specific (only present when offload_methods=True)
    def should_park_for_load_after_alloc(self, seq):
        return self.park

    def adjust_prefill_chunk_after_alloc(self, seq, chunk):
        return self.chunk_ret if self.chunk_ret is not None else chunk

    def should_park_partial_prefill_for_load(self, seq):
        return self.partial_park

    def should_defer_free(self, seq):
        return self.defer

    def save_finished(self, req_id):
        self.saved.append(req_id)

    def load_failed(self, req_id):
        self.load_failed_ids.append(req_id)

    def cancel_pending_load(self, seq):
        # Idempotent, like the real connector's `_load_lifecycles` guard.
        self.cancelled.append(seq)
        self.pending_load = False

    def process_completions(self, output):
        # Mirrors OffloadSchedulerMixin: apply the completions, then hand the
        # scheduler bare request ids.
        for value in output.finished_saving:
            self.save_finished(value)
        output.finished_saving = {
            getattr(value, "req_id", value) for value in output.finished_saving
        }
        return output

    def has_pending_work(self):
        return self.pending

    def enqueue_state_loads(self, loads):
        self.state_loads.extend(loads)
        return True

    def enqueue_state_stores(self, stores):
        self.state_stores.extend(stores)
        return True

    def take_state_reports(self):
        return self.state_reports

    def take_state_source_releases(self):
        return set(self.state_source_releases)

    def __getattribute__(self, name):
        # Hide offload-specific methods unless this mock opts in, so
        # MultiConnector's hasattr() guards are exercised realistically.
        offload_api = {
            "should_park_for_load_after_alloc",
            "adjust_prefill_chunk_after_alloc",
            "should_park_partial_prefill_for_load",
            "should_defer_free",
            "save_finished",
            "load_failed",
            "cancel_pending_load",
            "process_completions",
            "has_pending_work",
            "enqueue_state_loads",
            "enqueue_state_stores",
            "take_state_reports",
            "take_state_source_releases",
        }
        if name in offload_api and not object.__getattribute__(self, "_offload"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class DestructiveSub:
    """Faithful model of ``OffloadSchedulerMixin.process_completions``.

    Unlike ``FakeSchedSub`` (which only appends to ``finished_saving``), this
    reproduces the *destructive* behaviour the real mixin has: it filters
    ``finished_loading`` down to the one load it owns and ``.clear()``s
    ``connector_completions`` wholesale after skimming its own channel. Two of
    these in one composite is the ``[dense, kimi_k3]`` shape from the review —
    each sub rewrites the shared output and eats what the other owns.
    """

    is_offload = True

    def __init__(self, *, owned_load, owned_channel):
        self._owned_load = owned_load
        self._owned_channel = owned_channel
        self.settled_channels = []

    def process_completions(self, output):
        # Keep only the load this sub owns; the mixin drops the rest as "not
        # mine" (dense/connector.py load_finished returns False on foreign ids).
        output.finished_loading = {
            v for v in output.finished_loading if v == self._owned_load
        }
        # Consume only this sub's completion channel...
        for completion in output.connector_completions:
            if completion.channel == self._owned_channel:
                self.settled_channels.append(completion.channel)
        # ...but clear ALL of them, exactly as the mixin does.
        output.connector_completions.clear()
        return output


class FakeWorkerSub:
    """Worker-side sub-connector mock."""

    def __init__(self, *, is_producer=False, finished=None, recv_blocks=None):
        self.is_producer = is_producer
        self._finished = finished if finished is not None else KVConnectorOutput()
        self._recv_blocks = recv_blocks or []
        self.registered = None
        self.loaded_meta = None

    def register_kv_caches(self, kv_caches, transfer_tensors=None, num_blocks=None):
        self.registered = (kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata):
        self.loaded_meta = metadata

    def get_finished(self):
        return self._finished

    def get_finished_recv_blocks(self):
        return self._recv_blocks


def _sched(connectors):
    obj = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    obj._connectors = connectors
    obj.is_producer = any(getattr(c, "is_producer", False) for c in connectors)
    obj.is_offload = any(getattr(c, "is_offload", False) for c in connectors)
    obj._load_winner = {}
    return obj


def _worker(connectors, pp_is_head=True):
    obj = MultiConnector.__new__(MultiConnector)
    obj._connectors = connectors
    obj.is_producer = any(getattr(c, "is_producer", False) for c in connectors)
    obj._pp_is_head = pp_is_head
    obj._pending_save_ops = {}
    obj._sent = {}
    obj._saved = {}
    obj._state_tier = None
    return obj


def _save_meta(*req_ids):
    """An offload-style metadata: .requests with save_spec set."""
    meta = ConnectorMetadata()
    meta.requests = [
        SimpleNamespace(req_id=r, save_spec=object(), load_spec=None) for r in req_ids
    ]
    return meta


def _save_operation_meta(*operations):
    """Offload metadata carrying exact save operation identities."""
    meta = ConnectorMetadata()
    meta.requests = [
        SimpleNamespace(
            req_id=operation.req_id,
            save_spec=object(),
            load_spec=None,
            save_operation=operation,
        )
        for operation in operations
    ]
    return meta


# ---------------------------------------------------------------------------
# Scheduler-side
# ---------------------------------------------------------------------------


def test_matched_tokens_first_hit_wins():
    a = FakeSchedSub(match=(0, False))
    b = FakeSchedSub(match=(5, True))
    sched = _sched([a, b])
    assert sched.get_num_new_matched_tokens(object()) == (5, True)


def test_matched_tokens_earlier_connector_wins_over_later():
    a = FakeSchedSub(match=(3, True))
    b = FakeSchedSub(match=(5, True))
    sched = _sched([a, b])
    assert sched.get_num_new_matched_tokens(object()) == (3, True)


def test_no_match_returns_zero():
    sched = _sched([FakeSchedSub(), FakeSchedSub()])
    assert sched.get_num_new_matched_tokens(object()) == (0, False)


def test_losing_offload_sub_load_is_cancelled_when_another_sub_wins():
    """Review #6b. The lookup arms a load; only the winner may keep it.

    In `[moriio, lmcache_offload]`, moriio answers the match first, but the
    offload sub's own lookup still armed a KV load for the same prefix. Because
    `update_state_after_alloc` fans to every sub, that armed load would fire
    into the same block table moriio is already filling -- a second writer plus
    an unaccounted `finished_loading`. Once moriio wins, the composite must
    cancel the offload sub's pending load.
    """
    moriio = FakeSchedSub(is_producer=True, match=(5, True))
    off = FakeSchedSub(is_offload=True, offload_methods=True, match=(4, True))
    seq = object()
    sched = _sched([moriio, off])

    assert sched.get_num_new_matched_tokens(seq) == (5, True)  # moriio wins
    assert off.cancelled == [seq]  # its armed load was undone
    assert off.pending_load is False


def test_winning_offload_sub_keeps_its_load():
    """The mirror: when the offload sub is the first match, its load is the one
    that owns the block table, so it must NOT be cancelled."""
    moriio = FakeSchedSub(is_producer=True, match=(0, False))
    off = FakeSchedSub(is_offload=True, offload_methods=True, match=(4, True))
    seq = object()
    sched = _sched([moriio, off])

    assert sched.get_num_new_matched_tokens(seq) == (4, True)  # offload wins
    assert off.cancelled == []
    assert off.pending_load is True


def test_cancel_pending_load_forwards_to_offload_subs():
    """The scheduler abandons a parked load by calling `cancel_pending_load` on
    the connector it holds -- the composite under `multi`. With no forwarder the
    offload sub never heard it and leaked the load; fan to every sub that arms
    loads (idempotent, and moriio has no such method)."""
    moriio = FakeSchedSub(is_producer=True)
    off = FakeSchedSub(is_offload=True, offload_methods=True)
    seq = object()
    _sched([moriio, off]).cancel_pending_load(seq)

    assert off.cancelled == [seq]
    assert not hasattr(moriio, "cancel_pending_load")


def test_update_and_finished_fan_out_to_all():
    a, b = FakeSchedSub(), FakeSchedSub()
    sched = _sched([a, b])
    seq = object()
    sched.update_state_after_alloc(seq)
    sched.request_finished(seq)
    assert a.alloc_calls == [seq] and b.alloc_calls == [seq]
    assert a.finished_calls == [seq] and b.finished_calls == [seq]


def test_build_connector_meta_wraps_subs_in_order():
    a, b = FakeSchedSub(), FakeSchedSub()
    sched = _sched([a, b])
    meta = sched.build_connector_meta()
    assert isinstance(meta, MultiConnectorMetadata)
    assert meta.metas == [a.meta, b.meta]


def test_role_attrs_aggregate():
    sched = _sched(
        [
            FakeSchedSub(is_producer=True),
            FakeSchedSub(is_offload=True, offload_methods=True),
        ]
    )
    assert sched.is_producer is True
    assert sched.is_offload is True


def test_offload_methods_forwarded_to_owning_sub():
    # The load-side decisions route to the sub that armed the load, recorded at
    # `get_num_new_matched_tokens` time. Here the offload sub is the sole match,
    # so it owns the load and answers all three.
    moriio = FakeSchedSub(is_producer=True)  # no offload methods, no match
    off = FakeSchedSub(is_offload=True, offload_methods=True, match=(4, True))
    off.park = True
    off.partial_park = True
    off.defer = True
    off.chunk_ret = 7
    sched = _sched([moriio, off])
    seq = SimpleNamespace(id="s1")
    assert sched.get_num_new_matched_tokens(seq) == (4, True)  # off owns the load
    assert sched.should_park_for_load_after_alloc(seq) is True
    assert sched.should_park_partial_prefill_for_load(seq) is True
    assert sched.should_defer_free(seq) is True
    assert sched.adjust_prefill_chunk_after_alloc(seq, 10) == 7
    sched.save_finished("r1")
    sched.load_failed("r2")
    assert off.saved == ["r1"]
    assert off.load_failed_ids == ["r2"]


def test_load_side_methods_follow_the_load_owner_not_the_tier_sub():
    """Finding 1. The load-side questions belong to whichever sub armed the
    load, which need not be the tier sub.

    `[moriio, kimi_k3]`: moriio matches first and owns the KV load; kimi_k3
    hosts the state tier but armed nothing (its load was cancelled). Routing
    `should_park_for_load_after_alloc` through the tier sub asked kimi_k3, whose
    `_load_specs.get(sid)` is empty, so it answered "don't park" and the prefill
    ran over moriio's in-flight KV. It must route to moriio -- the load owner --
    which does not refine parking, so the composite parks (the scheduler's own
    absent-hook default) and the forward waits.
    """
    moriio = FakeSchedSub(is_producer=True, match=(5, True))  # wins, no offload API
    k3 = FakeSchedSub(
        is_offload=True, offload_methods=True, has_state_tier=True, match=(4, True)
    )
    k3.park = False  # the tier sub would say "don't park"
    sched = _sched([moriio, k3])
    seq = SimpleNamespace(id="s2")

    assert sched.get_num_new_matched_tokens(seq) == (5, True)  # moriio owns it
    assert k3.cancelled == [seq]  # kimi_k3's armed load was undone
    assert k3.pending_load is False
    # Routes to moriio (owner), which lacks the hook -> park, NOT to k3's False.
    assert sched.should_park_for_load_after_alloc(seq) is True
    # And the tier sub still owns the state face regardless of load ownership.
    assert sched._state_tier_sub() is k3


def test_process_completions_reaches_the_offload_sub():
    # The scheduler calls process_completions and nothing else — it is the only
    # caller of the offload sub's save_finished. Without the fan-out the sub's
    # inflight saves never clear (has_pending_work stays true forever) and the
    # exact SaveOperationId reaches a scheduler that looks requests up by id.
    moriio = FakeSchedSub(is_producer=True)
    off = FakeSchedSub(offload_methods=True)
    sched = _sched([moriio, off])

    op = SaveOperationId(9, 1)
    out = sched.process_completions(KVConnectorOutput(finished_saving={op}))

    assert off.saved == [op]
    assert out.finished_saving == {9}


def test_process_completions_refuses_two_offload_subs():
    # [dense, kimi_k3]: BOTH subs define process_completions, and the mixin is
    # destructive over the FULL sets it is handed -- it cannot partition. There
    # is no shared key by which the composite could split the completions per
    # sub, so one sub would retire the other's saves and clear its channels.
    # That composite is refused at startup (`_offload_subconfig`); reaching this
    # method with two offload handlers is a should-never-happen guarded loudly,
    # because silently corrupting saves is the worse failure.
    dense = DestructiveSub(owned_load="dense_load", owned_channel="dense_ch")
    k3 = DestructiveSub(owned_load="k3_load", owned_channel="k3_state_index")
    sched = _sched([dense, k3])

    with pytest.raises(RuntimeError, match="cannot be partitioned"):
        sched.process_completions(
            KVConnectorOutput(finished_loading={"dense_load", "k3_load"})
        )


def test_process_completions_single_offload_sub_is_called_directly():
    # The common [producer, offload] shape has exactly one process_completions
    # handler. That must stay the direct, copy-free path so its in-place
    # rewrites reach the caller unchanged.
    only = DestructiveSub(owned_load="x", owned_channel="ch")
    sched = _sched([FakeSchedSub(is_producer=True), only])

    out = sched.process_completions(
        KVConnectorOutput(
            finished_loading={"x", "y"},
            connector_completions={
                ConnectorCompletion("ch", SaveOperationId(3, 0), True)
            },
        )
    )

    assert out.finished_loading == {"x"}
    assert only.settled_channels == ["ch"]
    assert out.connector_completions == set()


def test_offload_methods_default_when_no_sub_implements():
    sched = _sched([FakeSchedSub(is_producer=True), FakeSchedSub()])
    seq = object()
    assert sched.should_park_for_load_after_alloc(seq) is False
    assert sched.should_park_partial_prefill_for_load(seq) is False
    assert sched.should_defer_free(seq) is False
    assert sched.adjust_prefill_chunk_after_alloc(seq, 10) == 10  # unchanged
    assert sched.has_pending_work() is False


def test_pending_work_is_the_union_over_subs():
    # The engine's idle drain keeps running while this holds, so one sub with
    # an unfinished save has to outvote every drained sibling.
    a = FakeSchedSub(offload_methods=True)
    b = FakeSchedSub(offload_methods=True)
    sched = _sched([a, b])
    assert sched.has_pending_work() is False

    b.pending = True
    assert sched.has_pending_work() is True


# ---------------------------------------------------------------------------
# Worker-side
# ---------------------------------------------------------------------------


def test_register_kv_caches_fans_out():
    a, b = FakeWorkerSub(), FakeWorkerSub()
    w = _worker([a, b])
    kv = {"layer_0": object()}
    w.register_kv_caches(kv, transfer_tensors="tt", num_blocks=42)
    assert a.registered == (kv, "tt", 42)
    assert b.registered == (kv, "tt", 42)


def test_start_load_kv_routes_by_index_and_records_saves():
    a, b = FakeWorkerSub(is_producer=True), FakeWorkerSub()
    w = _worker([a, b])
    m0 = ConnectorMetadata()  # moriio sub-meta (no .requests)
    m1 = _save_meta(101, 102)  # offload sub-meta with two saves
    w.start_load_kv(MultiConnectorMetadata([m0, m1]))
    assert a.loaded_meta is m0
    assert b.loaded_meta is m1
    assert w._pending_save_ops == {"101": {101}, "102": {102}}


def test_get_finished_unions_and_normalizes_tuple():
    # moriio returns a legacy tuple; offload returns KVConnectorOutput.
    moriio = FakeWorkerSub(finished=(set(), {"d1"}))  # recving d1
    off = FakeWorkerSub(
        finished=KVConnectorOutput(finished_recving={"d2"}, failed_recving={"f1"})
    )
    w = _worker([moriio, off])  # not producer
    out = w.get_finished()
    assert out.finished_recving == {"d1", "d2"}
    assert out.failed_recving == {"f1"}


def test_get_finished_carries_connector_completions():
    # DSV4 reports its SLOT sidecar outcome only on a connector-owned channel.
    # Dropping it here leaves the scheduler's _sidecar_save_inflight uncleared
    # and has_pending_work() true forever.
    done = ConnectorCompletion("dsv4.checkpoint.save", SaveOperationId(7, 1), True)
    moriio = FakeWorkerSub(finished=(set(), set()))
    off = FakeWorkerSub(finished=KVConnectorOutput(connector_completions={done}))
    w = _worker([moriio, off])  # not producer: pass-through path
    assert w.get_finished().connector_completions == {done}


def test_paired_get_finished_carries_connector_completions():
    # The pairing path builds its own output, so it needs the same union.
    done = ConnectorCompletion("dsv4.checkpoint.save", SaveOperationId(7, 1), True)
    off = FakeWorkerSub(
        is_producer=True, finished=KVConnectorOutput(connector_completions={done})
    )
    w = _worker([off], pp_is_head=True)
    assert w.get_finished().connector_completions == {done}


def test_producer_offload_load_completion_uses_loading_state():
    moriio = FakeWorkerSub(is_producer=True, finished=(set(), set()))
    off = FakeWorkerSub(
        finished=KVConnectorOutput(finished_loading={"l1"}, failed_loading={"f1"})
    )
    w = _worker([moriio, off])

    out = w.get_finished()

    assert out.finished_recving == set()
    assert out.failed_recving == set()
    assert out.finished_loading == {"l1"}
    assert out.failed_loading == {"f1"}


def test_state_loads_go_to_the_sub_that_can_carry_them():
    """The scheduler calls `enqueue_state_loads` on whatever connector it
    holds. Under `multi` that is this object, and a load it swallowed would
    leave its request parked against a transfer nobody was asked to make.
    """
    plain = FakeSchedSub()
    off = FakeSchedSub(is_offload=True, offload_methods=True)
    loads = [(1, 111, 0), (2, 222, 3)]

    assert _sched([plain, off]).enqueue_state_loads(loads) is True

    assert off.state_loads == loads
    assert not hasattr(plain, "enqueue_state_loads")


def test_no_sub_to_carry_a_state_load_is_reported_not_swallowed():
    """Every one of these belongs to a parked request that only a report can
    wake, and the scheduler's `hasattr` guard cannot catch this case because
    this method always exists on the composite. So it has to say no."""
    plain = FakeSchedSub()
    accepted = _sched([plain, FakeSchedSub()]).enqueue_state_loads([(1, 111, 0)])
    assert accepted is False


def test_state_stores_and_reports_forward_to_the_owning_sub():
    """The store leg is symmetric to the load leg, but only the load forwarder
    existed. The engine calls `enqueue_state_stores`, `take_state_reports` and
    `take_state_source_releases` on whatever connector it holds; under `multi`
    that is the composite. Missing these, `enqueue_state_stores` misses on the
    shell, the engine takes its "did not carry" branch and releases each store's
    PAGE units before the D2H, and reports/source-releases are never drained --
    so the CPU tier cannot fill or be found under `kv_connector: multi`.
    """
    moriio = FakeSchedSub(is_producer=True)  # no offload methods
    off = FakeSchedSub(is_offload=True, offload_methods=True)
    off.state_reports = ({7}, {9})
    off.state_source_releases = {7, 9}
    sched = _sched([moriio, off])

    assert sched.enqueue_state_stores([(111, (1, 2, 3))]) is True
    assert off.state_stores == [(111, (1, 2, 3))]
    assert sched.take_state_reports() == ({7}, {9})
    assert sched.take_state_source_releases() == {7, 9}
    assert not hasattr(moriio, "enqueue_state_stores")


def test_state_calls_route_by_tier_ownership_not_method_presence():
    """The reason `_first_with` was the wrong selector for the state face.

    Every `LMCacheOffloadConnectorScheduler` shell -- dense OR kimi_k3 --
    defines the whole state face unconditionally, delegating through `getattr`
    to its `_impl` and returning the no-tier default when the impl has none. So
    with `[dense_offload, kimi_k3_offload]` both subs answer `hasattr` for
    `enqueue_state_stores`, and `_first_with` picks the dense shell -- whose
    `_impl` silently refuses every store, discards every load, and drains no
    reports. The tier actually lives on the second sub.

    Selection must follow `has_state_tier` (does the `_impl` really host the
    tier), not method presence. Here the dense-like shell is first and carries
    no tier; the kimi_k3-like shell is second and does. Every state call must
    land on the second.

    A two-offload composite is refused at startup (`_offload_subconfig`, finding
    0), so this shape does not reach a running engine; the test exercises the
    selector directly to keep it robust if that refusal is ever bypassed.
    """
    dense = FakeSchedSub(is_offload=True, offload_methods=True, has_state_tier=False)
    k3 = FakeSchedSub(is_offload=True, offload_methods=True, has_state_tier=True)
    k3.state_reports = ({7}, {9})
    k3.state_source_releases = {7, 9}
    sched = _sched([dense, k3])

    assert sched.enqueue_state_stores([(111, (1, 2, 3))]) is True
    assert sched.enqueue_state_loads([(1, 111, 0)]) is True
    assert sched.take_state_reports() == ({7}, {9})
    assert sched.take_state_source_releases() == {7, 9}

    # The tier-carrying sub got everything; the dense shell got nothing.
    assert k3.state_stores == [(111, (1, 2, 3))]
    assert k3.state_loads == [(1, 111, 0)]
    assert dense.state_stores == []
    assert dense.state_loads == []


def test_state_stores_default_when_no_sub_implements():
    """No tier under the composite: refuse the stores (not silently accept) and
    drain nothing, matching the offload connector's own no-tier fallbacks. A
    swallowed store would strand its pinned PAGE units."""
    sched = _sched([FakeSchedSub(is_producer=True), FakeSchedSub()])
    assert sched.enqueue_state_stores([(1, (0,))]) is False
    assert sched.take_state_reports() == (set(), set())
    assert sched.take_state_source_releases() == set()


def test_the_composite_exposes_the_sub_connectors_state_tier():
    """The K3 store/load path reads `_state_tier` off whatever
    connector the forward context holds. Under `multi` that is this object, so
    without the re-export nothing is ever submitted and every slot leaks."""
    tier = object()
    off = FakeWorkerSub()
    off._state_tier = tier
    w = _worker([FakeWorkerSub(), off])

    w.register_kv_caches({}, transfer_tensors=None, num_blocks=1)

    assert w._state_tier is tier


def test_no_sub_with_a_tier_leaves_the_composite_tier_none():
    w = _worker([FakeWorkerSub(), FakeWorkerSub()])
    w.register_kv_caches({}, transfer_tensors=None, num_blocks=1)
    assert w._state_tier is None


def test_a_state_only_step_under_multi_is_not_dropped():
    """A step whose only work is a state load must reach the worker.

    `state_loads` carries no `LMCacheReqMeta`, so a wrapper that answers from
    its own (always empty) fields reports no work, the engine drops the
    snapshot, and the request parked on that load is woken by nothing.
    """
    from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

    sub = LMCacheOffloadMetadata()
    sub.state_loads = [("req-1", 12345, 7)]

    assert connector_metadata_has_work(sub)
    assert connector_metadata_has_work(MultiConnectorMetadata(metas=[sub]))


def test_a_multi_wrapper_over_idle_subs_still_reports_no_work():
    from types import SimpleNamespace

    from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

    assert not connector_metadata_has_work(
        MultiConnectorMetadata(metas=[LMCacheOffloadMetadata(), SimpleNamespace()])
    )


def test_every_metadata_field_a_subclass_adds_is_declared_work_or_not():
    """`WORK_FIELDS` must name real attributes, or a typo silences a field.

    A misspelled entry is invisible: `getattr` returns None, the field never
    counts, and the only symptom is a parked request much later.
    """
    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

    meta = LMCacheOffloadMetadata()
    for name in LMCacheOffloadMetadata.WORK_FIELDS:
        assert hasattr(meta, name), f"WORK_FIELDS names a missing attribute: {name}"


def test_two_sub_connectors_with_a_tier_is_refused():
    """First-one-wins would be wrong, not merely arbitrary: the spill goes to
    one tier and the load may ask the other, so a hash could be reported
    indexed by a tier that never stored it."""
    a, b = FakeWorkerSub(), FakeWorkerSub()
    a._state_tier, b._state_tier = object(), object()
    w = _worker([a, b])

    with pytest.raises(ValueError, match="exactly one may"):
        w.register_kv_caches({}, transfer_tensors=None, num_blocks=1)


def test_recv_blocks_concat():
    w = _worker([FakeWorkerSub(recv_blocks=[1, 2]), FakeWorkerSub(recv_blocks=[3])])
    assert w.get_finished_recv_blocks() == [1, 2, 3]


def test_non_producer_passes_saving_through():
    off = FakeWorkerSub(finished=KVConnectorOutput(finished_saving={"s1"}))
    w = _worker([off])  # is_producer False
    out = w.get_finished()
    assert out.finished_saving == {"s1"}


def test_send_without_pending_save_is_released_immediately():
    moriio = FakeWorkerSub(is_producer=True, finished=({"r1"}, set()))
    w = _worker([moriio])
    out = w.get_finished()
    assert out.finished_sending == {"r1"}


def test_send_is_withheld_until_save_completes():
    # One producer (moriio) + one offload sub, sharing req "r9".
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub()
    w = _worker([moriio, off])

    # offload will save r9
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))
    assert w._pending_save_ops == {"9": {9}}

    # Step 1: moriio reports send done, offload's save still in flight.
    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput()
    out1 = w.get_finished()
    assert out1.finished_sending == set()  # withheld
    assert out1.finished_saving == set()

    # Step 2: offload reports save done -> both released together.
    moriio._finished = (set(), set())
    off._finished = KVConnectorOutput(finished_saving={9})
    out2 = w.get_finished()
    assert out2.finished_sending == {9}
    assert out2.finished_saving == {9}
    assert w._pending_save_ops == {}  # cleared after release


def test_save_then_send_also_pairs():
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub()
    w = _worker([moriio, off])
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))

    # Step 1: save completes first, send not yet -> nothing released.
    off._finished = KVConnectorOutput(finished_saving={9})
    out1 = w.get_finished()
    assert out1.finished_sending == set()
    assert out1.finished_saving == set()

    # Step 2: send completes -> both released.
    off._finished = KVConnectorOutput()
    moriio._finished = ({9}, set())
    out2 = w.get_finished()
    assert out2.finished_sending == {9}
    assert out2.finished_saving == {9}


def test_pairing_matches_save_operation_id():
    # The offload connector reports a SaveOperationId(req_id, generation), not
    # a bare request id, whenever it tracks save generations. Pairing keys the
    # send side by request, so the completion has to collapse onto req_id or
    # every send is withheld forever and the producer never frees its blocks.
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub()
    w = _worker([moriio, off])
    op = SaveOperationId(9, 3)
    w.start_load_kv(
        MultiConnectorMetadata([ConnectorMetadata(), _save_operation_meta(op)])
    )
    assert w._pending_save_ops == {"9": {op}}

    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput(finished_saving={op})
    out = w.get_finished()
    assert out.finished_sending == {9}
    assert out.finished_saving == {op}
    assert w._pending_save_ops == {}
    assert w._sent == {}
    assert w._saved == {}


def test_pairing_waits_for_all_save_operation_ids():
    # Hybrid offload can have multiple save generations for one request in
    # flight. A request-level single-value _saved entry would overwrite the
    # first completion and release the send after only one save.
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub()
    w = _worker([moriio, off])
    op0 = SaveOperationId(9, 2)
    op1 = SaveOperationId(9, 3)
    w.start_load_kv(
        MultiConnectorMetadata([ConnectorMetadata(), _save_operation_meta(op0, op1)])
    )

    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput(finished_saving={op0})
    out1 = w.get_finished()
    assert out1.finished_sending == set()
    assert out1.finished_saving == set()

    moriio._finished = (set(), set())
    off._finished = KVConnectorOutput(finished_saving={op1})
    out2 = w.get_finished()
    assert out2.finished_sending == {9}
    assert out2.finished_saving == {op0, op1}
    assert w._pending_save_ops == {}
    assert w._sent == {}
    assert w._saved == {}


def test_non_head_pp_stage_does_not_pair():
    # Downstream stages never see mooncake's done_sending (it is recorded on
    # stage 0 only), so pairing there would strand every save. get_finished
    # returns before the release loop, so registering state there leaks it.
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub(finished=KVConnectorOutput(finished_saving={9}))
    w = _worker([moriio, off], pp_is_head=False)
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))

    out = w.get_finished()
    assert out.finished_saving == {9}
    assert out.finished_sending == set()
    assert w._pending_save_ops == {}


@pytest.mark.parametrize("pp_rank, holds_send", [(0, True), (1, False)])
def test_real_constructor_populates_the_pairing_state(monkeypatch, pp_rank, holds_send):
    # _worker() builds the instance with __new__ and hand-sets its fields, so
    # it drifts silently whenever __init__ grows one. Drive the real
    # constructor instead: a field it forgets fails here, not on a GPU node.
    moriio = FakeWorkerSub(
        is_producer=True, finished=KVConnectorOutput(finished_sending={9})
    )
    off = FakeWorkerSub()
    monkeypatch.setattr(
        mc_module, "_build_subconnectors", lambda config, role: [moriio, off]
    )

    w = MultiConnector(
        SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_rank=pp_rank))
    )
    assert w._pp_is_head is (pp_rank == 0)

    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))
    out = w.get_finished()
    assert out.finished_sending == (set() if holds_send else {9})
