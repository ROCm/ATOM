# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""Who sizes the state transfer slots, and who fills them.

`gdn_attn` and `model_runner` need aiter to import, so their methods under test
are lifted from the source and run on their own. The MoRIIO connector loads
against the AITER and Triton stand-ins of `tests/aiter_stub.py` where those
cannot run.
"""

import ast
import queue
import threading
import time
import types
from collections import deque
from pathlib import Path
from unittest import mock

import pytest
from aiter_stub import atom_without_aiter

from atom.kv_transfer.disaggregation.factory import (
    requested_state_transfer_slots,
    warn_unsized_state_transfer_slots,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata

with atom_without_aiter() as load:
    moriio = load("atom.kv_transfer.disaggregation.moriio.moriio_connector")

ROOT = Path(__file__).resolve().parents[1]
GDN_ATTN = ROOT / "atom/model_ops/attentions/gdn_attn.py"
MODEL_RUNNER = ROOT / "atom/model_engine/model_runner.py"


def lift(path, cls, name, scope):
    """Method `name` of class `cls` in `path`, undecorated, run in `scope`."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    item.decorator_list = []
                    code = compile(ast.Module([item], []), str(path), "exec")
                    exec(code, scope)  # noqa: S102
                    return scope[name]
    raise AssertionError(f"{cls}.{name} not found in {path}")


def builder(kv_transfer_config):
    config = types.SimpleNamespace(kv_transfer_config=kv_transfer_config)
    return types.SimpleNamespace(model_runner=types.SimpleNamespace(config=config))


MULTI = {
    "kv_connector": "multi",
    "connectors": [
        {"kv_connector": "moriio", "kv_role": "kv_producer", "state_transfer_slots": 8},
        {"kv_connector": "lmcache_offload", "kv_role": "offload"},
    ],
}


@pytest.mark.parametrize(
    "cfg, slots",
    [
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_producer",
                "state_transfer_slots": 8,
            },
            8,
        ),
        ({"kv_connector": "moriio", "state_transfer_slots": 8}, 8),
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_consumer",
                "state_transfer_slots": 8,
            },
            0,
        ),
        (
            {
                "kv_connector": "mooncake",
                "kv_role": "kv_producer",
                "state_transfer_slots": 8,
            },
            0,
        ),
        ({"kv_connector": "moriio", "kv_role": "kv_producer"}, 0),
        ({}, 0),
        # Spelled as the factory accepts it.
        ({"kv_connector": "MoRIIO", "state_transfer_slots": 8}, 8),
        ({"kv_connector": " moriio ", "state_transfer_slots": 8}, 8),
        # Under multi an offload sub may still read the live slot.
        (MULTI, 0),
        ({"kv_connector": "multi", "state_transfer_slots": 8}, 0),
    ],
)
def test_only_a_moriio_producer_sizes_transfer_slots(cfg, slots):
    sizing = lift(GDN_ATTN, "GDNStateMixin", "state_transfer_slots", {})
    assert sizing(builder(cfg)) == slots


@pytest.mark.parametrize(
    "cfg, sized, warned",
    [
        (MULTI, 0, True),
        ({"kv_connector": "multi", "state_transfer_slots": 8}, 0, True),
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_consumer",
                "state_transfer_slots": 8,
            },
            0,
            True,
        ),
        # A model other than KDA/GDN (its builder's base answer) sizes none either.
        ({"kv_connector": "moriio", "state_transfer_slots": 8}, 0, True),
        ({"kv_connector": "moriio", "state_transfer_slots": 8}, 8, False),
        ({"kv_connector": "moriio"}, 0, False),
        ({}, 0, False),
    ],
)
def test_a_set_key_that_sizes_nothing_warns(caplog, cfg, sized, warned):
    with caplog.at_level("WARNING", logger="atom"):
        warn_unsized_state_transfer_slots(cfg, sized)
    assert ("state_transfer_slots=8 is ignored" in caplog.text) == warned


@pytest.mark.parametrize("value", [-1, "8", 1.5, True])
def test_a_bad_count_is_refused_wherever_it_is_set(value):
    sizing = lift(GDN_ATTN, "GDNStateMixin", "state_transfer_slots", {})
    cfg = {"kv_connector": "moriio", "state_transfer_slots": value}
    with pytest.raises(ValueError, match="non-negative integer"):
        sizing(builder(cfg))
    # Where it would size nothing, too, rather than warn with a wrong count.
    with pytest.raises(ValueError, match="non-negative integer"):
        warn_unsized_state_transfer_slots(dict(cfg, kv_role="kv_consumer"), 0)
    assert requested_state_transfer_slots({}) == 0


def test_the_runner_copies_staged_state_before_the_connector_sees_it():
    calls = []
    connector = types.SimpleNamespace(
        note_state_stages=lambda stages: calls.append(("staged", stages)),
        start_load_kv=lambda meta: calls.append(("connector", meta)),
    )
    scope = {"get_kvconnector": lambda: connector}
    relocating = lift(
        MODEL_RUNNER, "ModelRunner", "_process_kvconnector_output_relocating", scope
    )
    plain = lift(MODEL_RUNNER, "ModelRunner", "process_kvconnector_output", scope)
    runner = types.SimpleNamespace(
        attn_metadata_builder=types.SimpleNamespace(
            relocate_state_slots=lambda pairs: calls.append(("copy", pairs))
        ),
    )
    stages = {9: (1, 4), 10: (2, 5)}
    meta = types.SimpleNamespace(state_stages=stages)
    relocating(runner, meta)
    assert calls == [
        ("copy", [(1, 4), (2, 5)]),
        ("staged", stages),
        ("connector", meta),
    ]
    calls.clear()
    meta = types.SimpleNamespace(state_stages={})
    relocating(runner, meta)
    assert calls == [("connector", meta)]
    # Without transfer slots main's dispatch never reads the field.
    calls.clear()
    meta = types.SimpleNamespace()
    plain(runner, meta)
    assert calls == [("connector", meta)]


def test_the_state_spec_adds_the_transfer_slots():
    """The GDN/KDA state pool is planned with the transfer slots on top."""
    import math

    import torch

    from atom.model_engine.kv_block import STATE_SLOT_CLASS
    from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
        SubPoolSpec,
        state_pool,
    )

    spec = lift(
        GDN_ATTN,
        "GDNStateMixin",
        "state_spec",
        {
            "math": math,
            "SubPoolSpec": SubPoolSpec,
            "state_pool": state_pool,
            "STATE_SLOT_CLASS": STATE_SLOT_CLASS,
            "KEEP_SNAPSHOT": object(),
            "DECODE_SNAPSHOT_SCRATCH_SLOTS": 4,
        },
    )
    config = types.SimpleNamespace(
        enable_prefix_caching=False, state_checkpoint_interval_tokens=0
    )
    stub = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(config=config),
        _state_shape_for_runner=lambda: ((2, 3), (4, 5)),
        _state_dtypes=lambda: (torch.bfloat16, torch.float32),
        state_transfer=lambda: types.SimpleNamespace(decode_keep=None),
        num_state_layers=lambda: 2,
        _replayssm_bytes_per_slot=lambda: 0,
        slots_per_req=lambda: 1,
        state_transfer_slots=lambda: 8,
    )
    assert spec(stub).extra_entries == 8


def runner_sizing():
    """`ModelRunner.get_num_blocks`, parsed: the runner needs aiter to import."""
    return next(
        node
        for cls in ast.parse(MODEL_RUNNER.read_text()).body
        if isinstance(cls, ast.ClassDef) and cls.name == "ModelRunner"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_num_blocks"
    )


def test_the_runner_sizes_from_the_builder_and_warns_when_unsized():
    """The builder's count as is (`state_spec` already planned the slots),
    then the key checked against what was sized."""
    sizing = runner_sizing()
    assigned = [
        ast.unparse(node.value)
        for node in ast.walk(sizing)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "config.state_transfer_slots"
    ]
    assert assigned == ["self.attn_metadata_builder.state_transfer_slots()"]
    warned = [
        [ast.unparse(arg) for arg in node.args]
        for node in ast.walk(sizing)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "warn_unsized_state_transfer_slots"
    ]
    assert warned == [["config.kv_transfer_config", "config.state_transfer_slots"]]


def test_the_sized_count_reaches_the_engine_config():
    """The runner ships `state_transfer_slots` in its block info, and the engine
    copies it into the config BlockManager and the scheduler connector read."""
    sizing = runner_sizing()
    returned = [n.value for n in ast.walk(sizing) if isinstance(n, ast.Return)][-1]
    keys = [k.value for k in returned.keys]
    value = returned.values[keys.index("state_transfer_slots")]
    shipped = eval(
        compile(ast.Expression(value), str(MODEL_RUNNER), "eval"),
        {"config": types.SimpleNamespace(state_transfer_slots=8)},
    )
    engine = ROOT / "atom/model_engine/engine_core.py"
    assign = next(
        node
        for node in ast.walk(ast.parse(engine.read_text()))
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "config.state_transfer_slots"
    )
    scope = {
        "config": types.SimpleNamespace(),
        "block_info": {"state_transfer_slots": shipped},
    }
    exec(compile(ast.Module([assign], []), str(engine), "exec"), scope)  # noqa: S102
    assert scope["config"].state_transfer_slots == 8


def test_the_runner_binds_relocation_once_sized():
    """With transfer slots, sizing swaps in the relocating dispatch, which the
    worker's RPC then resolves by name; without them main's stays."""
    sizing = runner_sizing()
    guarded = [
        node
        for node in ast.walk(sizing)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "config.state_transfer_slots"
    ]
    bound = {
        (ast.unparse(stmt.targets[0]), ast.unparse(stmt.value))
        for node in guarded
        for stmt in node.body
        if isinstance(stmt, ast.Assign)
    }
    assert (
        "self.process_kvconnector_output",
        "self._process_kvconnector_output_relocating",
    ) in bound


def test_only_moriio_metadata_carries_state_stages():
    assert not hasattr(ConnectorMetadata(), "state_stages")
    meta, other = moriio.MoRIIOConnectorMetadata(), moriio.MoRIIOConnectorMetadata()
    assert meta.state_stages == {} and not meta.has_work()
    meta.state_stages[7] = (1, 4)
    assert meta.has_work() and other.state_stages == {}


def scheduler(*, slots: int = 8, role: str = "kv_producer"):
    """The real scheduler-side connector, from the config fields it reads."""
    config = types.SimpleNamespace(
        kv_transfer_config={"kv_role": role},
        tensor_parallel_size=8,
        parallel_config=types.SimpleNamespace(
            data_parallel_size=1, data_parallel_rank=0
        ),
        state_transfer_slots=slots,
    )
    return moriio.MoRIIOConnectorScheduler(config)


@pytest.mark.parametrize("slots", [0, 8])
def test_the_scheduler_decides_staging_once(slots):
    """`__init__` reads the runner's count and binds the staging snapshot only
    when there are slots; without them the metadata is the plain base class."""
    sched = scheduler(slots=slots)
    assert sched._has_transfer_slots is (slots > 0)
    bound = vars(sched).get("build_connector_meta")
    assert (bound == sched._build_staging_connector_meta) is (slots > 0)
    staging = moriio.MoRIIOConnectorMetadata
    assert type(sched.build_connector_meta()) is (
        staging if slots else ConnectorMetadata
    )


def test_a_staged_copy_rides_one_metadata_unless_its_send_is_released():
    sched = scheduler()
    for seq_id, src, dst in ((7, 1, 4), (8, 2, 5)):
        sched.stage_state_for_send(types.SimpleNamespace(id=seq_id), src, dst)
        # `request_finished` claims the send right after the staging.
        sched._awaiting_send.add(str(seq_id))
    sched.send_finished("7")
    meta = sched.build_connector_meta()
    assert meta.state_stages == {8: (2, 5)} and meta.has_work()
    meta = sched.build_connector_meta()
    assert not meta.state_stages and not meta.has_work()


class FakeEvent:
    """Stands in for `torch.cuda.Event`; the test says when the copy landed."""

    def __init__(self):
        self.done = False
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.done


def producer(*, live: int = 4, transfer: int = 2):
    """A worker-side producer with `live` then `transfer` state slots, its
    consumers handshaken and its RDMA writes settled as soon as issued."""
    conn = object.__new__(moriio.MoRIIOConnector)
    conn.is_producer = True
    conn.tp_rank, conn.tp_size = 0, 1
    conn._config = types.SimpleNamespace(state_transfer_slots=transfer)
    conn._num_state_slots = live + transfer
    conn._locate_transfer_slots()
    conn._staged_sources = {}
    # Each transfer's KV-ready event (`_source_ready`); `queue_push` marks its
    # prefill finished unless told otherwise.
    conn._kv_ready_lock = threading.Lock()
    conn._kv_ready_events, conn._kv_ready_expired = {}, {}
    conn._write_requests = queue.Queue()
    conn._pending_writes = []
    conn._sending_writes = {}
    conn._write_ids = {}
    conn._finished_writes = deque()
    conn._write_timeout_s = 600.0
    conn._handshake_timeout_s = 300.0
    conn._done_sending_lock = threading.Lock()
    conn.done_sending = set()
    # The rest of what `get_finished` reads: no receives.
    conn._awaiting_write, conn._write_reports, conn._held_failures = {}, {}, {}
    conn._failed_before_read = set()
    conn._verify_kv = False
    conn._pop_done_transfers = lambda: (set(), set())
    conn._write_target_ready = lambda request: True
    conn.pushed, conn.reports = [], []

    def write_blocks(request, engine_id):
        conn.pushed.append(request["transfer_id"])
        return []

    conn._write_blocks = write_blocks
    conn.moriio_wrapper = types.SimpleNamespace(
        send_message=lambda frames, host, port: conn.reports.append(
            moriio.msgpack.loads(frames[1])
        )
    )
    return conn


def queue_push(
    conn, transfer_id: int, src_slot: int, *, queued_at=None, prefilled=True
):
    """A consumer's WRITE_REQ for `transfer_id`, accepted and waiting."""
    if prefilled:
        conn._kv_ready_expired[transfer_id] = None
    request = {
        "transfer_id": transfer_id,
        "src_slot_index": src_slot,
        "decode_req_id": f"d{transfer_id}",
        "decode_host": "10.0.0.2",
        "decode_handshake_port": 6301,
        "deadline": time.time() + 60,
    }
    conn._write_ids[transfer_id] = None
    queued = time.monotonic() if queued_at is None else queued_at
    conn._pending_writes.append((request, queued))


def stage(conn, stages):
    """`note_state_stages` right after the runner's copies."""
    with mock.patch("torch.cuda.Event", FakeEvent):
        conn.note_state_stages(stages)


def registered_worker(transfer_slots: int, *, live: int = 6):
    """The worker connector from its own `__init__` (RDMA engine, process groups
    and address faked), after `register_kv_caches` of one KDA layer holding
    `live + transfer_slots` state slots and one MLA layer."""
    import torch

    group = types.SimpleNamespace(rank_in_group=0, world_size=1)
    wrapper = mock.MagicMock()
    wrapper.register_local_buffer.return_value = b"desc"
    wrapper.get_agent_metadata.return_value = b"agent"
    rdma = ("IOEngine", "IOEngineConfig", "RdmaBackendConfig", "PollCqMode")
    config = types.SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_producer", "moriio_mode": "write"},
        kv_cache_block_size=2,
        state_transfer_slots=transfer_slots,
    )
    with mock.patch.multiple(
        moriio,
        create=True,
        get_tp_group=lambda: group,
        get_dp_group=lambda: group,
        get_ip=lambda: "10.0.0.2",
        MoRIIOWrapper=lambda moriio_engine: wrapper,
        BackendType=mock.MagicMock(),
        **{name: mock.MagicMock() for name in rdma},
    ):
        conn = moriio.MoRIIOConnector(config)
    conn._handshake_listener = lambda *args: None
    slots = live + transfer_slots
    state = types.SimpleNamespace(
        k_cache=torch.zeros((slots, 3, 8), dtype=torch.bfloat16),
        v_cache=torch.zeros((slots, 2, 4, 4), dtype=torch.float32),
        replay_buf_k=None,
        replay_buf_u=None,
        replay_buf_g=None,
        per_request_state=True,
    )
    mla = types.SimpleNamespace(
        k_cache=torch.zeros((16, 1, 4), dtype=torch.uint8), v_cache=None
    )
    conn.register_kv_caches({"layer_0": state, "layer_1": mla})
    conn._handshake_listener_thread.join(timeout=5)
    conn._handshake_executor.shutdown(wait=False)
    conn.zmq_context.term()
    return conn


@pytest.mark.parametrize("transfer_slots", [0, 8])
def test_registering_the_caches_locates_the_transfer_slots(transfer_slots):
    conn = registered_worker(transfer_slots, live=6)
    assert conn._first_transfer_slot == 6
    assert conn._has_transfer_slots is (transfer_slots > 0)
    bound = vars(conn).get("get_finished")
    assert (bound == conn._get_finished_forgetting_stages) is (transfer_slots > 0)


def test_transfer_slots_start_past_the_live_ones():
    conn = producer(live=4, transfer=2)
    assert (conn._first_transfer_slot, conn._has_transfer_slots) == (4, True)
    conn = producer(live=6, transfer=0)
    assert (conn._first_transfer_slot, conn._has_transfer_slots) == (6, False)


def test_one_event_marks_every_copy_of_a_step():
    conn = producer()
    stage(conn, {9: (1, 4), 10: (2, 5)})
    landed = conn._staged_sources[9]
    assert conn._staged_sources == {9: landed, 10: landed} and landed.recorded


def test_a_live_slot_is_pushed_without_staging():
    conn = producer()
    queue_push(conn, 9, src_slot=2)
    conn._service_write_requests()
    assert conn.pushed == [9]
    assert conn.reports == [{"decode_req_id": "d9", "ok": True}]


def test_a_transfer_slot_is_pushed_only_once_its_copy_landed():
    conn = producer()
    queue_push(conn, 9, src_slot=5)
    conn._service_write_requests()
    # The runner has not copied the state there yet.
    assert (conn.pushed, len(conn._pending_writes)) == ([], 1)
    stage(conn, {9: (1, 5)})
    conn._service_write_requests()
    assert (conn.pushed, len(conn._pending_writes)) == ([], 1)
    conn._staged_sources[9].done = True
    conn._service_write_requests()
    assert (conn.pushed, conn._pending_writes) == ([9], [])
    assert conn.reports == [{"decode_req_id": "d9", "ok": True}]


def test_a_landed_copy_still_waits_for_the_prefill_forward():
    conn = producer()
    queue_push(conn, 9, src_slot=5, prefilled=False)
    stage(conn, {9: (1, 5)})
    conn._staged_sources[9].done = True
    forward = FakeEvent()
    conn._kv_ready_events[9] = forward
    conn._service_write_requests()
    assert (conn.pushed, len(conn._pending_writes)) == ([], 1)
    forward.done = True
    conn._service_write_requests()
    assert (conn.pushed, conn._pending_writes) == ([9], [])


def test_a_copy_that_never_lands_fails_after_the_handshake_timeout(caplog):
    conn = producer()
    stale = time.monotonic() - conn._handshake_timeout_s - 1
    queue_push(conn, 9, src_slot=5, queued_at=stale)
    with caplog.at_level("ERROR", logger="atom"):
        conn._service_write_requests()
    assert "its state never reached transfer slot 5" in caplog.text
    assert (conn.pushed, conn.done_sending) == ([], {9})
    assert conn.reports == [{"decode_req_id": "d9", "ok": False}]


def test_a_released_transfer_forgets_its_copy():
    """A push that finishes, and a consumer that failed before asking (its
    notify lands in `done_sending`), both release the staged copy."""
    conn = producer()
    stage(conn, {9: (1, 4), 10: (2, 5)})
    conn._staged_sources[9].done = True
    queue_push(conn, 9, src_slot=4)
    conn.done_sending.add(10)
    assert conn.get_finished().finished_sending == {9, 10}
    assert conn._staged_sources == {}
    # Unsized, the worker keeps main's `get_finished`.
    assert "get_finished" not in vars(producer(transfer=0))
