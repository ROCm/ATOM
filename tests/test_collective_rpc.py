# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import pickle
import queue
import sys
import threading
import time
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from atom.model_engine.collective_rpc import (
    COLLECTIVE_RPC_COMMAND,
    CollectiveRPCError,
    CollectiveRPCRequest,
    CollectiveRPCResponseRouter,
    EngineCoreRPCResponse,
    RPCErrorInfo,
    WorkerRPCResponse,
    validate_worker_responses,
)
from atom.model_engine.engine_utility import EngineUtilityHandler

ATOM_ROOT = Path(__file__).resolve().parents[1]


def _load_module_with_stubs(module_name, relative_path, stubs):
    originals = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = spec_from_file_location(module_name, ATOM_ROOT / relative_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _MessageQueue:
    pass


aiter_module = types.ModuleType("aiter")
aiter_dist_module = types.ModuleType("aiter.dist")
shm_broadcast_module = types.ModuleType("aiter.dist.shm_broadcast")
shm_broadcast_module.MessageQueue = _MessageQueue

atom_utils_module = types.ModuleType("atom.utils")
atom_utils_module.get_mp_context = lambda: None
atom_utils_module.get_open_zmq_ipc_path = lambda: ""
atom_utils_module.init_exit_handler = lambda *args, **kwargs: None
atom_utils_module.make_zmq_socket = lambda *args, **kwargs: None
atom_utils_module.resolve_obj_by_qualname = lambda *args, **kwargs: None
atom_utils_module.shutdown_all_processes = lambda *args, **kwargs: None

numa_utils_module = types.ModuleType("atom.utils.numa_utils")
numa_utils_module.numa_bind_to_node = lambda *args, **kwargs: None

kv_disaggregation_module = types.ModuleType("atom.kv_transfer.disaggregation")
kv_disaggregation_module.KVOutputAggregator = object

async_proc_module = _load_module_with_stubs(
    "_collective_rpc_async_proc",
    "atom/model_engine/async_proc.py",
    {
        "aiter": aiter_module,
        "aiter.dist": aiter_dist_module,
        "aiter.dist.shm_broadcast": shm_broadcast_module,
        "atom.utils": atom_utils_module,
        "atom.utils.numa_utils": numa_utils_module,
        "atom.kv_transfer.disaggregation": kv_disaggregation_module,
    },
)
AsyncIOProc = async_proc_module.AsyncIOProc
AsyncIOProcManager = async_proc_module.AsyncIOProcManager


core_manager_utils_module = types.ModuleType("atom.utils")
core_manager_utils_module.get_open_zmq_inproc_path = lambda: ""
core_manager_utils_module.get_open_zmq_ipc_path = lambda: ""
core_manager_utils_module.make_zmq_socket = lambda *args, **kwargs: None
core_manager_utils_module.set_device_control_env_var = lambda *args, **kwargs: None

engine_core_module = types.ModuleType("atom.model_engine.engine_core")


class _EngineCoreRequestType:
    UTILITY = b"utility"
    UTILITY_RESPONSE = b"utility_response"
    SHUTDOWN = b"shutdown"
    STREAM = b"stream"
    ADD = b"add"
    READY = b"ready"


engine_core_module.EngineCore = object
engine_core_module.EngineCoreRequestType = _EngineCoreRequestType

engine_core_mgr_module = _load_module_with_stubs(
    "_collective_rpc_engine_core_mgr",
    "atom/model_engine/engine_core_mgr.py",
    {
        "atom.utils": core_manager_utils_module,
        "atom.model_engine.engine_core": engine_core_module,
    },
)
CoreManager = engine_core_mgr_module.CoreManager


class _Runner:
    def echo(self, value, *, suffix=""):
        return f"{value}{suffix}"

    def returns_none(self):
        return None

    def fails(self):
        raise ValueError("boom")


class _AliveProc:
    def is_alive(self):
        return True


class _DeadProc:
    def is_alive(self):
        return False


class _BroadcastQueue:
    def __init__(self):
        self.messages = []

    def enqueue(self, message):
        self.messages.append(message)


def _make_worker(rank=0):
    worker = AsyncIOProc.__new__(AsyncIOProc)
    worker.rank = rank
    worker.runners = [_Runner()]
    worker.rpc_queue = queue.Queue()
    return worker


def _success(request_id, tp_rank, result):
    return WorkerRPCResponse.success(request_id, tp_rank, result)


def _request(request_id="req", method="echo", timeout=1.0, args=(), kwargs=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    return CollectiveRPCRequest(request_id, method, args, kwargs or {}, deadline)


def _make_manager(responses, procs=None):
    manager = AsyncIOProcManager.__new__(AsyncIOProcManager)
    manager.label = "test-manager"
    manager.rpc_broadcast_mq = _BroadcastQueue()
    manager.rpc_outputs_queues = []
    for rank_responses in responses:
        output_queue = queue.Queue()
        for response in rank_responses:
            output_queue.put_nowait(response)
        manager.rpc_outputs_queues.append(output_queue)
    manager.procs = procs or [_AliveProc() for _ in responses]
    return manager


def test_protocol_request_factory_validates_inputs_and_tracks_deadline(monkeypatch):
    monkeypatch.setattr("atom.model_engine.collective_rpc.time.monotonic", lambda: 10.0)

    request = CollectiveRPCRequest.create(
        "echo", timeout=2.5, args=("value",), kwargs={"suffix": "!"}
    )
    assert request.method == "echo"
    assert request.args == ("value",)
    assert request.kwargs == {"suffix": "!"}
    assert request.deadline == 12.5
    assert request.remaining_timeout() == 2.5

    no_deadline = CollectiveRPCRequest.create("echo")
    assert no_deadline.deadline is None
    assert no_deadline.remaining_timeout() is None

    with pytest.raises(TypeError, match="non-empty string"):
        CollectiveRPCRequest.create("")
    with pytest.raises(TypeError, match="args must be a tuple"):
        CollectiveRPCRequest.create("echo", args=[])
    with pytest.raises(TypeError, match="kwargs must be a dict"):
        CollectiveRPCRequest.create("echo", kwargs=[])
    with pytest.raises(ValueError, match="non-negative"):
        CollectiveRPCRequest.create("echo", timeout=-1)


def test_async_io_proc_keeps_existing_positional_constructor_contract():
    parameters = inspect.signature(AsyncIOProc.__init__).parameters

    assert (
        parameters["all_ranks_barrier"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert parameters["rpc_output_addr"].kind is inspect.Parameter.KEYWORD_ONLY


def test_protocol_validates_complete_tp_rank_set_independent_of_arrival_order():
    request = _request()
    out_of_order = EngineCoreRPCResponse.success(
        request,
        2,
        [
            _success(request.request_id, 1, "tp1"),
            _success(request.request_id, 0, "tp0"),
        ],
    )
    assert validate_worker_responses(out_of_order) is None

    missing = EngineCoreRPCResponse.success(
        request, 2, [_success(request.request_id, 0, "tp0")]
    )
    assert validate_worker_responses(missing).type == "CollectiveRPCProtocolError"

    duplicate = EngineCoreRPCResponse.success(
        request,
        2,
        [
            _success(request.request_id, 0, "first"),
            _success(request.request_id, 0, "second"),
        ],
    )
    assert validate_worker_responses(duplicate).type == "CollectiveRPCProtocolError"


def test_protocol_types_are_pickle_safe():
    request = _request(args=("value",), kwargs={"suffix": "!"})
    worker_response = WorkerRPCResponse.failure(
        request.request_id, 0, RPCErrorInfo.transport("disconnected")
    )
    engine_response = EngineCoreRPCResponse.success(request, 1, [worker_response])

    for value in (request, worker_response, engine_response):
        assert pickle.loads(pickle.dumps(value)) == value


def test_response_router_rejects_duplicates_and_cleans_up_registration():
    router = CollectiveRPCResponseRouter()

    with router.register("req") as response_queue:
        with (
            pytest.raises(RuntimeError, match="already registered"),
            router.register("req"),
        ):
            pass
        assert router.route("req", "response") is True
        assert response_queue.get_nowait() == "response"

    assert router.route("req", "late") is False


def test_worker_collective_rpc_forwards_args_kwargs_and_none_result():
    worker = _make_worker(rank=2)

    worker._execute_collective_rpc(
        _request("req-1", args=("value",), kwargs={"suffix": "!"})
    )
    assert worker.rpc_queue.get_nowait() == _success("req-1", 2, "value!")

    worker._execute_collective_rpc(_request("req-2", method="returns_none"))
    assert worker.rpc_queue.get_nowait() == _success("req-2", 2, None)


def test_worker_collective_rpc_reports_missing_method_and_exception():
    worker = _make_worker(rank=1)

    worker._execute_collective_rpc(_request("missing", method="does_not_exist"))
    missing = worker.rpc_queue.get_nowait()
    assert missing.ok is False
    assert missing.tp_rank == 1
    assert missing.error.type == "AttributeError"

    worker._execute_collective_rpc(_request("failure", method="fails"))
    failure = worker.rpc_queue.get_nowait()
    assert failure.ok is False
    assert failure.error.type == "ValueError"
    assert failure.error.message == "boom"


def test_manager_collective_rpc_returns_tp_rank_order_and_drops_stale_response():
    request_id = "current"
    manager = _make_manager(
        [
            [_success("stale", 0, "old"), _success(request_id, 0, "tp0")],
            [_success(request_id, 1, "tp1")],
        ]
    )

    request = _request(request_id, args=("value",), kwargs={"suffix": "!"}, timeout=1.0)
    responses = manager.collective_rpc(request)

    assert [response.result for response in responses] == ["tp0", "tp1"]
    assert manager.rpc_broadcast_mq.messages == [(COLLECTIVE_RPC_COMMAND, request)]


def test_manager_collective_rpc_reports_dead_and_timed_out_workers():
    manager = _make_manager([[], []], procs=[_DeadProc(), _AliveProc()])

    responses = manager.collective_rpc(_request(timeout=0.01))

    assert [response.ok for response in responses] == [False, False]
    assert "not alive" in responses[0].error.message
    assert "timed out" in responses[1].error.message


class _RunnerManager:
    def __init__(self, responses=None, error=None):
        self.responses = responses
        self.error = error
        self.calls = []
        self.proc_num = len(responses) if responses else 1

    def collective_rpc(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.responses


def test_utility_handler_preserves_request_id_and_reports_tp_results():
    tp_responses = [_success("req", 0, None), _success("req", 1, "ok")]
    runner_manager = _RunnerManager(responses=tp_responses)
    output_queue = queue.Queue()
    handler = EngineUtilityHandler(runner_manager, output_queue)

    request = _request(args=("value",), kwargs={"suffix": "!"}, timeout=5)
    handler._handle_collective_rpc({"request": request})

    kind, response = output_queue.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    assert response.request_id == "req"
    assert response.tp_responses == tuple(tp_responses)
    assert response.error is None
    assert runner_manager.calls == [request]


def test_utility_handler_turns_local_dispatch_exception_into_response():
    output_queue = queue.Queue()
    handler = EngineUtilityHandler(
        _RunnerManager(error=TypeError("bad arguments")), output_queue
    )

    handler._handle_collective_rpc({"request": _request()})

    _, response = output_queue.get_nowait()
    assert response.request_id == "req"
    assert response.tp_responses == ()
    assert response.error.type == "TypeError"
    assert response.error.message == "bad arguments"


def _make_core_manager(dp_count=2):
    manager = CoreManager.__new__(CoreManager)
    manager.label = "test-core-manager"
    manager.local_engine_count = dp_count
    manager.utility_response_queue = queue.Queue()
    manager._collective_rpc_router = CollectiveRPCResponseRouter()
    manager._utility_send_lock = threading.Lock()
    manager.engine_core_processes = [_AliveProc() for _ in range(dp_count)]
    return manager


def test_core_manager_collective_rpc_flattens_dp_major_order():
    manager = _make_core_manager()

    def broadcast(_cmd, **payload):
        request = payload["request"]
        manager._route_utility_response(
            1,
            EngineCoreRPCResponse.success(
                request,
                2,
                [
                    _success(request.request_id, 0, "dp1-tp0"),
                    _success(request.request_id, 1, "dp1-tp1"),
                ],
            ),
        )
        manager._route_utility_response(
            0,
            EngineCoreRPCResponse.success(
                request,
                2,
                [
                    _success(request.request_id, 1, "dp0-tp1"),
                    _success(request.request_id, 0, "dp0-tp0"),
                ],
            ),
        )

    manager.broadcast_utility_command = broadcast

    assert manager.collective_rpc("echo", timeout=1.0) == [
        "dp0-tp0",
        "dp0-tp1",
        "dp1-tp0",
        "dp1-tp1",
    ]


def test_core_manager_collective_rpc_raises_ranked_worker_failure():
    manager = _make_core_manager(dp_count=1)

    def broadcast(_cmd, **payload):
        request = payload["request"]
        failure = WorkerRPCResponse.failure(
            request.request_id,
            0,
            RPCErrorInfo("ValueError", "boom", "trace"),
        )
        manager._route_utility_response(
            0,
            EngineCoreRPCResponse.success(request, 1, [failure]),
        )

    manager.broadcast_utility_command = broadcast

    try:
        manager.collective_rpc("fails", timeout=1.0)
    except CollectiveRPCError as exc:
        assert "DP0/TP0" in str(exc)
        assert "ValueError: boom" in str(exc)
    else:
        raise AssertionError("CollectiveRPCError was not raised")


def test_core_manager_routes_legacy_and_drops_late_correlated_responses():
    manager = _make_core_manager(dp_count=1)
    manager._route_utility_response(0, {"cmd": "legacy", "result": True})
    assert manager.utility_response_queue.get_nowait()["cmd"] == "legacy"

    request = _request("already-finished")
    manager._route_utility_response(
        0,
        EngineCoreRPCResponse.success(
            request, 1, [_success(request.request_id, 0, None)]
        ),
    )
    assert manager.utility_response_queue.empty()


def test_core_manager_routes_concurrent_request_ids_to_separate_queues():
    manager = _make_core_manager(dp_count=1)
    first = EngineCoreRPCResponse.failure(
        "first", "echo", RPCErrorInfo.transport("first")
    )
    second = EngineCoreRPCResponse.failure(
        "second", "echo", RPCErrorInfo.transport("second")
    )

    with (
        manager._collective_rpc_router.register("first") as first_queue,
        manager._collective_rpc_router.register("second") as second_queue,
    ):
        manager._route_utility_response(0, second)
        manager._route_utility_response(0, first)

        assert first_queue.get_nowait() == (0, first)
        assert second_queue.get_nowait() == (0, second)


def test_async_engine_collective_rpc_delegates_public_contract(monkeypatch):
    llm_engine_module = types.ModuleType("atom.model_engine.llm_engine")
    llm_engine_module.LLMEngine = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.llm_engine", llm_engine_module)

    weight_sync_module = types.ModuleType("atom.rollout.weight_sync")
    weight_sync_module.load_weights_via_ipc = lambda *args, **kwargs: None
    weight_sync_module.load_weights_via_shm = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "atom.rollout.weight_sync", weight_sync_module)

    module_path = (
        Path(__file__).resolve().parents[1] / "atom" / "rollout" / "async_engine.py"
    )
    spec = spec_from_file_location("_collective_rpc_async_engine", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    AsyncLLMEngine = module.AsyncLLMEngine

    class _CoreManager:
        def __init__(self):
            self.calls = []

        def collective_rpc(self, **kwargs):
            self.calls.append(kwargs)
            return ["ok"]

    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = _CoreManager()

    result = engine.collective_rpc(
        "echo",
        timeout=3.0,
        args=("value",),
        kwargs={"suffix": "!"},
    )

    assert result == ["ok"]
    assert engine.core_mgr.calls == [
        {
            "method": "echo",
            "timeout": 3.0,
            "args": ("value",),
            "kwargs": {"suffix": "!"},
        }
    ]
