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
    ):
        self._match = match
        self.is_producer = is_producer
        if is_offload:
            self.is_offload = True
        self.alloc_calls = []
        self.finished_calls = []
        self.meta = ConnectorMetadata()
        self._offload = offload_methods

        if offload_methods:
            self.park = False
            self.partial_park = False
            self.defer = False
            self.chunk_ret = None
            self.saved = []
            self.load_failed_ids = []
            self.pending = False

    def get_num_new_matched_tokens(self, seq):
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

    def has_pending_work(self):
        return self.pending

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
            "has_pending_work",
        }
        if name in offload_api and not object.__getattribute__(self, "_offload"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)


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
    return obj


def _worker(connectors, pp_is_head=True):
    obj = MultiConnector.__new__(MultiConnector)
    obj._connectors = connectors
    obj.is_producer = any(getattr(c, "is_producer", False) for c in connectors)
    obj._pp_is_head = pp_is_head
    obj._pending_save = set()
    obj._sent = {}
    obj._saved = {}
    return obj


def _save_meta(*req_ids):
    """An offload-style metadata: .requests with save_spec set."""
    meta = ConnectorMetadata()
    meta.requests = [
        SimpleNamespace(req_id=r, save_spec=object(), load_spec=None) for r in req_ids
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
    moriio = FakeSchedSub(is_producer=True)  # no offload methods
    off = FakeSchedSub(is_offload=True, offload_methods=True)
    off.park = True
    off.partial_park = True
    off.defer = True
    off.chunk_ret = 7
    sched = _sched([moriio, off])
    seq = object()
    assert sched.should_park_for_load_after_alloc(seq) is True
    assert sched.should_park_partial_prefill_for_load(seq) is True
    assert sched.should_defer_free(seq) is True
    assert sched.adjust_prefill_chunk_after_alloc(seq, 10) == 7
    sched.save_finished("r1")
    sched.load_failed("r2")
    assert off.saved == ["r1"]
    assert off.load_failed_ids == ["r2"]


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
    assert w._pending_save == {"101", "102"}


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
    assert w._pending_save == {"9"}

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
    assert w._pending_save == set()  # cleared after release


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
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))

    op = SaveOperationId(9, 3)
    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput(finished_saving={op})
    out = w.get_finished()
    assert out.finished_sending == {9}
    assert out.finished_saving == {op}
    assert w._pending_save == set()
    assert w._sent == {}
    assert w._saved == {}


def test_non_head_pp_stage_does_not_pair():
    # Downstream stages never see mooncake's done_sending (it is recorded on
    # stage 0 only), so pairing there would strand every save.
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub(finished=KVConnectorOutput(finished_saving={9}))
    w = _worker([moriio, off], pp_is_head=False)
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))

    out = w.get_finished()
    assert out.finished_saving == {9}
    assert out.finished_sending == set()


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
