# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The P->D send process (``ATOM_PD_SEND_PROCESS``), on CPU.

`create` builds its connector from a stage's state, and each transfer case runs
it beside the in-process producer and compares bytes, READYs and write-dones.
"""

import ctypes
import multiprocessing
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import torch
from test_mha_landing import (  # noqa: F401 - host_gather, host_streams: autouse
    LAST,
    SINGLE,
    STAGES,
    Harness,
    host_gather,
    host_streams,
    mc,
)

from atom.kv_transfer.disaggregation.mooncake import send_process as sp
from atom.mono.runtime import hip_ipc

SRC, DST = [3, 1, 2], [2, 0, 3]


@pytest.fixture(autouse=True)
def harness_connector_module(monkeypatch):
    # The harness imported the connector under a dropped aiter stub; reuse it.
    import atom.kv_transfer.disaggregation.mooncake as package

    monkeypatch.setitem(sys.modules, mc.__name__, mc)
    monkeypatch.setattr(package, "mooncake_connector", mc, raising=False)


def minimax_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type="minimax_m3", num_hidden_layers=2),
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(pipeline_parallel_rank=0),
        kv_cache_block_size=16,
        dcp_config=SimpleNamespace(interleave_size=1),
        kv_transfer_config={
            "kv_connector": "mooncake",
            "kv_role": "kv_producer",
            "protocol": "rdma",
            "ib_device": "ionic_0",
        },
    )


def stage_connector(monkeypatch):
    """A MiniMax page-staging producer stage, as `__init__` builds it."""
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(mc, "get_tp_group", lambda: group)
    monkeypatch.setattr(mc, "get_dp_group", lambda: group)
    monkeypatch.setattr(mc, "get_ip", lambda: "192.0.2.2")
    monkeypatch.setattr(mc, "get_open_port", lambda: 17000)
    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(mc, "_ib_device_exists", lambda device: True)
    monkeypatch.setattr(mc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mc, "ThreadPoolExecutor", MagicMock())
    monkeypatch.setattr(mc.zmq, "Context", MagicMock())
    engine = MagicMock()
    engine.initialize.return_value = 0
    engine.get_rpc_port.return_value = 17001
    monkeypatch.setattr(mc, "TransferEngine", lambda: engine, raising=False)
    monkeypatch.setenv("ATOM_PD_MINIMAX_LANDING", "1")
    monkeypatch.setenv("ATOM_MOONCAKE_MATCHED_RAILS", "")
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE", raising=False)
    return mc.MooncakeConnector(minimax_config())


# What register_kv_caches sets on a stage; taken from the harness's pages.
REGISTERED = (
    "pp_rank", "pp_size", "num_hidden_layers", "block_size", "_start_layer",
    "_num_local_layers", "_per_block_bytes_list", "_block_region_roles",
    "_block_head_major_runs", "_block_unit_specs", "_block_region_consumer_indices",
)  # fmt: skip


def created_state(h, monkeypatch):
    """The state the send process takes from a stage `__init__` built over the
    harness's pages; a key the stage lacks raises."""
    executor, context = mc.ThreadPoolExecutor, mc.zmq.Context
    stage = stage_connector(monkeypatch)
    monkeypatch.setattr(mc, "ThreadPoolExecutor", executor)
    monkeypatch.setattr(mc.zmq, "Context", context)
    for key in REGISTERED:
        setattr(stage, key, getattr(h.c, key))
    return {key: getattr(stage, key) for key in sp.STATE_KEYS}


# What `serve` sets, and the harness's sockets and streams, taken from its stage.
SERVED = (
    "transfer_engine", "_kv_staging", "_kv_staging_slots", "_send_on_socket",
    "_send_worker_stream",
)  # fmt: skip


def as_child(h, monkeypatch, ready=("xfer-a",), keep=16):
    """The send process's connector, built by `create` from the harness stage's
    state; it becomes ``h.c``."""
    monkeypatch.setattr(mc, "KV_CACHE_READY_EXPIRED_MAX", keep)
    done = []
    c = sp._connector_class().create(
        created_state(h, monkeypatch), dict(enumerate(h.src_views)), done.append
    )
    c._done_ids = done
    c._ready.mark(ready)
    for key in SERVED:
        setattr(c, key, getattr(h.c, key))
    c._ready_event = h.event
    if "mla_landing" in h.req:  # what its write listener does with the request
        c._landing_credits.sync("cpu:1", h.req["mla_landing"])
    h.c._landing_credits = c._landing_credits  # where the decode returns credits
    h.c = c
    return c


def prepare(
    h, monkeypatch, *, child, fault=None, ready=("xfer-a",), keep=16, deadline_s=60
):
    """A write request for ``_execute_transfer``, in process or in the send
    process's connector; ``go()`` runs it and returns what it did."""
    h.recv.pool.zero_()  # only what lands is compared
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, h.partition)))
    h.req.update(src_block_ids=list(SRC), _deadline=time.monotonic() + deadline_s)
    prefill = {"block_ids": list(SRC), "slot_index": -1, "swa_block_ids": []}
    if child:
        c = as_child(h, monkeypatch, ready, keep)
        c.cache_saved_prefills({"xfer-a": prefill})
    else:
        c = h.c
        del c._notify_transfer_result  # the protocol fixture's mock
        c._transfer_refcount, c._transfer_refcount_lock = {}, threading.Lock()
        c._completed_prefills = {"xfer-a": prefill}
        c._kv_cache_ready_events["xfer-a"] = h.event
    done, ready_sink = [], c._send_on_socket

    def send(addr, parts, repeat=1):
        if parts[0] == mc.MSG_WRITE_DONE:
            done.append(msgpack.loads(parts[-1]))
        else:
            ready_sink(addr, parts)

    c._send_on_socket = send
    write = c.transfer_engine.batch_transfer_sync_write
    calls = []

    def faulty(target, srcs, dsts, sizes):
        calls.append(len(srcs))
        if fault == "rdma" and len(calls) == 2:
            return -1
        ret = write(target, srcs, dsts, sizes)
        if fault == "deadline_mid":
            h.req["_deadline"] = time.monotonic() - 1
        return ret

    c.transfer_engine = SimpleNamespace(batch_transfer_sync_write=faulty)
    if fault == "deadline_first":
        h.req["_deadline"] = time.monotonic() - 1

    def go(request=None):
        c._execute_transfer(h.req if request is None else request)
        return trace(h, done)

    return go


def full_transfer(h, monkeypatch, **kwargs):
    return prepare(h, monkeypatch, **kwargs)()


def trace(h, done):
    buffers = [h.recv.pool, *h.dst_views]

    def where(address):
        for i, b in enumerate(buffers):
            if b.data_ptr() <= address < b.data_ptr() + b.numel():
                return i, address - b.data_ptr()
        raise AssertionError(hex(address))

    credits = {
        target: (sorted(t.free), sorted(t.in_flight), sorted(t.quarantined))
        for target, t in h.c._landing_credits._targets.items()
    }
    return {
        "bytes": [b.clone() for b in buffers],
        "writes": [([where(d) for d in dsts], sizes) for _, dsts, sizes in h.writes],
        "ready": list(h.ready),
        "done": list(done),
        "credits": credits,
        "staging_free": list(h.c._kv_staging_slots.free),
    }


def assert_same(child, inproc):
    for got, want in zip(child.pop("bytes"), inproc.pop("bytes"), strict=True):
        assert torch.equal(got, want)
    assert child == inproc


def write_done(pp_rank, success, landed_slots):
    return {
        "request_id": 21,
        "pp_rank": pp_rank,
        "tp_rank": 0,
        "write_nonce": 3,
        "success": success,
        "landed_slots": landed_slots,
        "prefill_handoff": None,
    }


# ---- T1: the send process's connector sends what the stage sends today ----


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("offer", [False, True])
def test_child_sends_the_same_bytes_readys_and_write_done(
    stage, rank, offer, monkeypatch
):
    inproc = full_transfer(Harness(stage, rank, offer=offer), monkeypatch, child=False)
    h = Harness(stage, rank, offer=offer)
    child = full_transfer(h, monkeypatch, child=True)
    h.check(SRC, DST)
    landed = h.req.get("_landed_slots", []) if offer else None
    assert child["done"] == [write_done(stage[0], True, landed)]
    assert bool(child["ready"]) == offer
    assert_same(child, inproc)


@pytest.mark.parametrize("stage", [SINGLE, LAST])
@pytest.mark.parametrize(
    "fault, wrote",
    [
        ("credit_timeout", True),
        ("deadline_first", False),
        ("deadline_mid", True),
        ("rdma", True),
    ],
)
def test_child_fails_as_the_stage_does(stage, fault, wrote, monkeypatch):
    pump = fault != "credit_timeout"
    arms = {}
    for child in (False, True):
        h = Harness(stage, 1, pump=pump)
        arms[child] = full_transfer(h, monkeypatch, child=child, fault=fault)
    inproc, child = arms[False], arms[True]
    assert bool(child["writes"]) == wrote
    if fault == "credit_timeout":
        # Two landing slots never come back; the rest goes straight to the pages.
        assert len(child["ready"]) == 2 and child["done"][0]["success"]
        assert any(len(dsts) > 1 for dsts, _ in child["writes"])
    else:
        assert [d["success"] for d in inproc["done"]] == [False]
    if fault == "rdma":
        assert len(child["writes"]) == 1  # the failed second write moved nothing
        (target,) = child["credits"]
        assert child["credits"][target][2], "the failed slot is quarantined"
    assert_same(child, inproc)


def test_a_transfer_without_a_head_cut_fails_closed_with_no_write(monkeypatch, caplog):
    # Without a cut the stage writes straight from its KV, which the send
    # process registers with no NIC.
    h = Harness(LAST, 1)
    go = prepare(h, monkeypatch, child=True)
    plan = h.c._block_plan(h.req, len(h.req["consumer_base_addrs"]), "uncut")
    monkeypatch.setattr(h.c, "_block_plan", lambda *a: mc._UnitPlan(plan.cmap, None))
    got = go()
    assert "is not from staging" in caplog.text
    assert not got["writes"] and not got["ready"]
    assert got["done"] == [write_done(LAST[0], False, [])]


# ---- T2: KV readiness through the stage's READY ----


def test_a_request_before_ready_waits_then_sends(monkeypatch):
    h = Harness(LAST, 2)
    go = prepare(h, monkeypatch, child=True, ready=())
    result = {}
    worker = threading.Thread(target=lambda: result.update(go()))
    worker.start()
    time.sleep(0.3)
    assert worker.is_alive() and not h.writes
    h.c._ready.mark(["xfer-a"])
    worker.join(10)
    assert not worker.is_alive()
    assert result["done"] == [write_done(LAST[0], True, h.req["_landed_slots"])]
    h.check(SRC, DST)


@pytest.mark.parametrize("why", ["never", "evicted", "deadline"])
def test_a_request_whose_ready_never_comes_fails_closed_with_no_write(why, monkeypatch):
    monkeypatch.setattr(mc, "PREFILL_LOOKUP_TIMEOUT", 60 if why == "deadline" else 0.2)
    kwargs = {
        "never": {"ready": ()},
        # The newest two READYs stay: this request's was the oldest of three.
        "evicted": {"ready": ("xfer-a", "b", "c"), "keep": 2},
        "deadline": {"ready": (), "deadline_s": 0.2},
    }[why]
    h = Harness(LAST, 0)
    started = time.monotonic()
    got = full_transfer(h, monkeypatch, child=True, **kwargs)
    assert time.monotonic() - started < 10
    assert not got["writes"] and not got["ready"]
    assert got["done"] == [write_done(LAST[0], False, [])]
    assert got["credits"]["cpu:1"][1] == [], "no landing slot taken"


def test_a_duplicate_ready_is_one_entry_and_a_dropped_one_can_return():
    table = sp.ReadyTable(2)
    table.mark(["a"])
    table.mark(["a"])
    assert list(table._ready) == ["a"] and table.wait("a", 0)
    table.mark(["b", "c"])  # the newest two stay
    assert not table.wait("a", 0) and table.wait("b", 0) and table.wait("c", 0)
    table.drop("c")
    assert not table.wait("c", 0)
    table.mark(["c"])
    assert table.wait("c", 0)
    threading.Timer(0.05, table.mark, (["late"],)).start()
    assert table.wait("late", 5)


def test_the_stage_messages_mark_ready_and_cache_saved_until_stop(monkeypatch):
    c = as_child(Harness(LAST, 0), monkeypatch, ready=())
    reader, writer = multiprocessing.Pipe(duplex=False)
    entry = {"block_ids": SRC, "swa_block_ids": [], "slot_index": -1}
    for message in ((sp.SAVED, {"r": entry}), (sp.READY, ["r"]), (sp.STOP, None)):
        writer.send(message)
    writer.send((sp.READY, ["after-stop"]))
    sp._follow_stage(c, reader)
    assert c._completed_prefills["r"] == entry
    assert c._ready.wait("r", 0) and not c._ready.wait("after-stop", 0)
    writer.close()
    sp._follow_stage(c, reader)  # returns at EOF too


# ---- T3: releases ----


def test_all_decode_ranks_releasing_sends_one_done_and_drops_the_ready(monkeypatch):
    c = as_child(Harness(SINGLE, 0), monkeypatch)
    for _ in range(5):  # four ranks and a duplicate
        c._record_release("xfer-a", 4)
    assert c._done_ids == ["xfer-a"] and not c.done_sending
    assert not c._ready.wait("xfer-a", 0)


@pytest.mark.parametrize("child", [False, True])
def test_a_single_stage_marks_done_once_every_consumer_rank_is_served(
    child, monkeypatch
):
    h = Harness(SINGLE, 0, offer=False)
    go = prepare(h, monkeypatch, child=child)
    for rank in range(4):
        go(dict(h.req, consumer_tp_rank=rank))
    if child:
        assert h.c._done_ids == ["xfer-a"] and not h.c.done_sending
    else:
        assert h.c.done_sending == {"xfer-a"}


# ---- T4: the pool's views across the process boundary ----


@pytest.fixture
def pool(monkeypatch):
    """4096 bytes, a fake allocation of ``size["bytes"]`` of them, and the HIP
    calls as seen from it."""
    memory = torch.arange(4096, dtype=torch.int64).remainder(251).to(torch.uint8)
    base, size = memory.data_ptr(), {"bytes": 4096}

    def allocation_base(pointer):
        return base if base <= pointer < base + size["bytes"] else pointer

    def handle(allocation):
        assert allocation == base
        return b"h" * 64

    def view_at(address, shape, device):
        raw = (ctypes.c_ubyte * int(torch.Size(shape).numel())).from_address(address)
        return torch.frombuffer(raw, dtype=torch.uint8).view(shape)

    monkeypatch.setattr(hip_ipc, "get_allocation_base", allocation_base)
    monkeypatch.setattr(hip_ipc, "get_ipc_handle", handle)
    monkeypatch.setattr(hip_ipc, "open_ipc_handle", {b"h" * 64: base}.__getitem__)
    monkeypatch.setattr(sp, "_device_view", view_at)
    return memory, size


def test_views_cross_as_offsets_into_one_mapping(pool):
    memory, _ = pool
    views = {
        0: memory[0:1024].view(4, 1, 256),
        1: memory[1024:2048].view(8, 1, 128),
        2: memory[3072:4096].view(2, 1, 512),
    }
    handle, placed = sp.export_views(views)
    assert placed == [
        (0, 0, (4, 1, 256)),
        (1, 1024, (8, 1, 128)),
        (2, 3072, (2, 1, 512)),
    ]
    rebuilt = sp._open_views(
        sp.SendProcessSpec({"_cuda_device": 0}, handle, placed, "pd-send-pp0")
    )
    for region, view in views.items():
        assert rebuilt[region].data_ptr() == view.data_ptr()
        assert rebuilt[region].shape == view.shape
        assert torch.equal(rebuilt[region], view)


@pytest.mark.parametrize("where", ["another allocation", "past its end"])
def test_a_view_outside_the_pool_allocation_is_refused(where, pool):
    memory, size = pool
    if where == "past its end":
        size["bytes"] = 3072  # the last view's tail lies beyond it
        stray = memory[2048:4096].view(4, 1, 512)
    else:
        stray = torch.zeros(512, dtype=torch.uint8).view(1, 1, 512)
    views = {0: memory[0:1024].view(4, 1, 256), 1: stray}
    with pytest.raises(RuntimeError, match="inside one allocation"):
        sp.export_views(views)


@pytest.mark.parametrize("views", ["one short", "one extra", "renumbered"])
def test_create_refuses_views_but_one_per_block_region(views, monkeypatch):
    h = Harness(LAST, 0)
    given = dict(enumerate(h.src_views))
    n = len(given)
    given = {
        "one short": {i: given[i] for i in range(n - 1)},
        "one extra": {**given, n: given[0]},
        "renumbered": {i + 1: view for i, view in given.items()},
    }[views]
    with pytest.raises(RuntimeError, match="one view per block region"):
        sp._connector_class().create(created_state(h, monkeypatch), given, print)


def test_serve_registers_staging_only_and_warms_each_cut_before_listening(
    monkeypatch,
):
    h = Harness(LAST, 0)
    state = created_state(h, monkeypatch)
    state.update(_kv_staging_reserve=2 * mc.STAGING_SLOT_BYTES, _num_send_workers=2)
    steps, gathers = [], []
    engine = MagicMock()
    engine.initialize.side_effect = lambda *a: steps.append("engine") or 0
    engine.get_rpc_port.return_value = 17002
    engine.batch_register_memory.side_effect = lambda ptrs, sizes: (
        steps.append(("register", ptrs, sizes)) or 0
    )
    monkeypatch.setattr(mc, "TransferEngine", lambda: engine, raising=False)

    def gather(source, index, out, plan):
        steps.append("gather")
        gathers.append((index.data_ptr() % 16 == 0, out.data_ptr() % 16 == 0))

    monkeypatch.setattr(mc, "gather_units", gather)
    monkeypatch.setattr(sp, "_completed_event", lambda device: steps.append("event"))

    def listen(c):
        steps.append("listen")
        c._publish_listener_state()

    monkeypatch.setattr(mc.MooncakeConnector, "_write_listener", listen)
    c = sp._connector_class().create(state, dict(enumerate(h.src_views)), print)
    c.serve()
    assert steps[0] == "engine" and steps[-2:] == ["event", "listen"]
    (_, ptrs, sizes), warm = steps[1], steps[2:-2]
    low = c._kv_staging.data_ptr()
    assert sum(sizes) == c._kv_staging.numel() == 2 * mc.STAGING_SLOT_BYTES
    assert ptrs[0] == low and ptrs[-1] + sizes[-1] == low + sum(sizes)
    cut = [runs for runs in state["_block_head_major_runs"] if runs is not None]
    assert cut and set(warm) == {"gather"}
    tps = [t for t in (2, 4, 8) if 4 % t == 0 or t % 4 == 0]
    assert len(gathers) == len(cut) * len(tps) * 4
    assert set(gathers) == {(True, True), (False, True), (True, False), (False, False)}
