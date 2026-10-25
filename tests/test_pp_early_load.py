# SPDX-License-Identifier: MIT
# PP downstream stages starting offload loads ahead of their batch (GPU-free).

from collections import deque
from types import SimpleNamespace

from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorMetadata,
    )
    from atom.kv_transfer.disaggregation.types import (
        ConnectorMetadata,
        LoadOperationId,
        SaveOperationId,
        split_early_loads,
    )
    from atom.kv_transfer.offload.metadata import (
        LMCacheOffloadMetadata,
        LMCacheReqMeta,
        LoadSpec,
        SaveSpec,
    )
    from atom.model_engine.pp_engine_core import PPEngineCoreProc


def _req(req_id, *, load=False, save=False, **kw):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[1, 2, 3],
        block_ids=[7],
        load_spec=LoadSpec(0, 256, can_load=True) if load else None,
        save_spec=SaveSpec(0) if save else None,
        load_operation=LoadOperationId(req_id, 1) if load else None,
        save_operation=SaveOperationId(req_id, 1) if save else None,
        **kw,
    )


def _offload(*reqs, lookups=(), state_loads=(), state_stores=()):
    meta = LMCacheOffloadMetadata()
    meta.requests = list(reqs)
    meta.lookup_requests_in_step = list(lookups)
    meta.state_loads = list(state_loads)
    meta.state_stores = list(state_stores)
    return meta


def test_loads_split_from_saves():
    meta = _offload(
        _req("load", load=True),
        _req("save", save=True),
        _req("both", load=True, save=True),
        lookups=["load", "miss"],
        state_loads=[("load", "s")],
        state_stores=[("save", "s")],
    )

    loads, rest = split_early_loads(meta)

    assert [r.req_id for r in loads.requests] == ["load", "both"]
    assert all(r.save_spec is None and r.save_operation is None for r in loads.requests)
    assert all(r.load_operation is not None for r in loads.requests)
    assert loads.lookup_requests_in_step == ["load", "miss"]
    assert loads.state_loads == [("load", "s")]
    assert loads.state_stores == []
    assert [r.req_id for r in rest.requests] == ["save", "both"]
    assert all(r.load_spec is None and r.load_operation is None for r in rest.requests)
    assert all(r.save_operation is not None for r in rest.requests)
    assert rest.lookup_requests_in_step == []
    assert rest.state_loads == []
    assert rest.state_stores == [("save", "s")]
    assert [r.req_id for r in meta.requests] == ["load", "save", "both"]
    assert meta.requests[2].save_spec is not None


def test_load_only_metadata_leaves_no_rest():
    loads, rest = split_early_loads(_offload(_req("a", load=True)))

    assert [r.req_id for r in loads.requests] == ["a"]
    assert rest is None


def test_lookup_unpins_alone_go_early():
    loads, rest = split_early_loads(_offload(lookups=["a"]))

    assert loads.lookup_requests_in_step == ["a"]
    assert rest is None


def test_save_only_metadata_is_not_split():
    meta = _offload(_req("a", save=True))

    assert split_early_loads(meta) == (None, meta)


def test_slot_and_native_state_transfers_are_not_split():
    for extra in ({"slot_load_spec": object()}, {"native_state": object()}):
        meta = _offload(_req("a", load=True, **extra), _req("b", save=True))

        assert split_early_loads(meta) == (None, meta)


def test_metadata_without_a_split_is_all_rest():
    meta = ConnectorMetadata()
    meta.reqs_to_recv["r"] = object()

    assert split_early_loads(meta) == (None, meta)
    assert split_early_loads(None) == (None, None)


def test_multi_keeps_one_slot_per_connector():
    pd = ConnectorMetadata()
    pd.reqs_to_send["s"] = object()
    meta = MultiConnectorMetadata(
        [pd, _offload(_req("load", load=True), _req("save", save=True))]
    )

    loads, rest = split_early_loads(meta)

    assert loads.metas[0] is None
    assert [r.req_id for r in loads.metas[1].requests] == ["load"]
    assert rest.metas[0] is pd
    assert [r.req_id for r in rest.metas[1].requests] == ["save"]
    assert len(meta.metas) == 2 and meta.metas[1].requests[0].req_id == "load"


def test_multi_without_loads_is_not_split():
    pd = ConnectorMetadata()
    pd.reqs_to_send["s"] = object()
    meta = MultiConnectorMetadata([pd, _offload(_req("save", save=True))])

    assert split_early_loads(meta) == (None, meta)


class FakeTransport:
    def __init__(self, batches):
        self.batches = list(batches)

    def recv_metadata(self, timeout_ms=0):
        return self.batches.pop(0) if self.batches else None


class FakeRunnerMgr:
    def __init__(self):
        self.dispatched = []

    def call_func(self, name, *args, **kwargs):
        assert name == "process_kvconnector_output"
        self.dispatched.append(args[0])


def _batch(name, meta):
    return SimpleNamespace(name=name, connector_meta_output=meta)


def _stage(batches, early):
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc._early_loads = early
    proc._stage_backlog = deque()
    proc.pp_transport = FakeTransport(batches)
    proc.runner_mgr = FakeRunnerMgr()
    return proc


def test_stage_starts_every_queued_load_before_the_first_batch_runs():
    a = _batch("a", _offload(_req("a", save=True)))
    b = _batch("b", _offload(_req("b", load=True), _req("x", save=True)))
    c = _batch("c", _offload(_req("c", load=True)))
    proc = _stage([a, b, c], early=True)

    first = proc._recv_stage_batch(timeout_ms=100)

    assert first is a
    assert [m.requests[0].req_id for m in proc.runner_mgr.dispatched] == ["b", "c"]
    assert [r.req_id for r in a.connector_meta_output.requests] == ["a"]
    assert [r.req_id for r in b.connector_meta_output.requests] == ["x"]
    assert c.connector_meta_output is None
    assert proc._recv_stage_batch(timeout_ms=100) is b
    assert proc._recv_stage_batch(timeout_ms=100) is c
    assert proc._recv_stage_batch(timeout_ms=100) is None
    assert len(proc.runner_mgr.dispatched) == 2


def test_stage_without_early_loads_leaves_metadata_alone():
    meta = _offload(_req("b", load=True), _req("x", save=True))
    b = _batch("b", meta)
    proc = _stage([b], early=False)

    assert proc._recv_stage_batch(timeout_ms=100) is b
    assert b.connector_meta_output is meta
    assert proc.runner_mgr.dispatched == []
