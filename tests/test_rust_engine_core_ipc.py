import inspect
from types import SimpleNamespace
from threading import Event, Lock

import pytest

from atom.config import PER_REQ_CACHE_MODEL_TYPES, Config
from atom.model_engine import engine_core_mgr
from atom.model_engine.ipc_utils import EngineCoreIpcCodec


def _config(**overrides):
    values = {
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "tp_world_size": 1,
        "enable_dp_attention": False,
        "enable_rapidserve": False,
        "fake_eplb": False,
        "parallel_config": SimpleNamespace(
            data_parallel_size=2,
            data_parallel_size_local=2,
            data_parallel_rank=0,
            is_multinode_dp=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rust_owned_endpoint_plan_reuses_python_address_allocator(monkeypatch):
    allocated = iter(f"ipc:///tmp/atom-{index}" for index in range(6))
    monkeypatch.setattr(
        engine_core_mgr,
        "get_open_zmq_ipc_path",
        lambda: next(allocated),
    )

    endpoints = engine_core_mgr.plan_rust_owned_engine_core_endpoints(_config())

    assert endpoints == [
        {
            "engine_rank": 0,
            "dp_rank": 0,
            "pp_rank": 0,
            "input_address": "ipc:///tmp/atom-0",
            "control_address": "ipc:///tmp/atom-1",
            "output_address": "ipc:///tmp/atom-2",
        },
        {
            "engine_rank": 1,
            "dp_rank": 1,
            "pp_rank": 0,
            "input_address": "ipc:///tmp/atom-3",
            "control_address": "ipc:///tmp/atom-4",
            "output_address": "ipc:///tmp/atom-5",
        },
    ]


@pytest.mark.parametrize("pp_size", [2, 4])
def test_rust_owned_endpoint_plan_allocates_every_pp_stage(monkeypatch, pp_size):
    allocated = iter(f"ipc:///tmp/pp-{index}" for index in range(pp_size * 3))
    monkeypatch.setattr(
        engine_core_mgr,
        "get_open_zmq_ipc_path",
        lambda: next(allocated),
    )
    config = _config(
        pipeline_parallel_size=pp_size,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            is_multinode_dp=False,
        ),
    )

    endpoints = engine_core_mgr.plan_rust_owned_engine_core_endpoints(config)

    assert [
        (endpoint["engine_rank"], endpoint["dp_rank"], endpoint["pp_rank"])
        for endpoint in endpoints
    ] == [(rank, 0, rank) for rank in range(pp_size)]
    assert len(
        {
            endpoint[address_name]
            for endpoint in endpoints
            for address_name in (
                "input_address",
                "control_address",
                "output_address",
            )
        }
    ) == pp_size * 3


def test_engine_core_process_assignments_expand_pp_stages():
    config = _config(
        pipeline_parallel_size=4,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            is_multinode_dp=False,
        ),
    )

    assignments = engine_core_mgr.iter_engine_core_process_assignments(config)

    assert [
        (
            assignment.engine_rank,
            assignment.dp_rank,
            assignment.local_dp_rank,
            assignment.pp_rank,
        )
        for assignment in assignments
    ] == [(rank, 0, 0, rank) for rank in range(4)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"pipeline_parallel_size": 2},
        {
            "pipeline_parallel_size": 2,
            "enable_dp_attention": True,
            "parallel_config": SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                data_parallel_rank=0,
                is_multinode_dp=False,
            ),
        },
        {
            "pipeline_parallel_size": 2,
            "tensor_parallel_size": 4,
            "tp_world_size": 2,
            "parallel_config": SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                data_parallel_rank=0,
                is_multinode_dp=False,
            ),
        },
        {"enable_rapidserve": True},
        {"enable_dp_attention": True, "fake_eplb": True},
        {
            "parallel_config": SimpleNamespace(
                data_parallel_size=2,
                is_multinode_dp=True,
            )
        },
    ],
)
def test_rust_owned_endpoint_plan_rejects_unsupported_topologies(overrides):
    with pytest.raises(ValueError, match="Rust-owned EngineCore transport"):
        engine_core_mgr.plan_rust_owned_engine_core_endpoints(
            _config(**overrides)
        )


def test_rust_owned_endpoint_plan_expands_dp_attention_ranks(monkeypatch):
    allocated = iter(f"ipc:///tmp/dpa-{index}" for index in range(24))
    monkeypatch.setattr(
        engine_core_mgr,
        "get_open_zmq_ipc_path",
        lambda: next(allocated),
    )
    config = _config(
        tensor_parallel_size=4,
        enable_dp_attention=True,
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_size_local=2,
            data_parallel_rank=0,
            is_multinode_dp=False,
        ),
    )

    endpoints = engine_core_mgr.plan_rust_owned_engine_core_endpoints(config)

    assert [endpoint["dp_rank"] for endpoint in endpoints] == list(range(8))
    assert [endpoint["engine_rank"] for endpoint in endpoints] == list(range(8))
    assert {endpoint["pp_rank"] for endpoint in endpoints} == {0}
    addresses = {
        address
        for endpoint in endpoints
        for address in (
            endpoint["input_address"],
            endpoint["control_address"],
            endpoint["output_address"],
        )
    }
    assert len(addresses) == 24


def test_core_manager_binds_rust_transport_before_starting_engine_cores():
    source = inspect.getsource(engine_core_mgr.CoreManager.__init__)

    dp_attention_rewrite = source.index(
        "config.parallel_config.data_parallel_size *= config.tensor_parallel_size"
    )
    endpoint_plan = source.index("plan_rust_owned_engine_core_endpoints(")
    bind = source.index("self.external_transport_owner = external_transport_factory(")
    start = source.index('info["process"].start()')
    connected = source.index(
        "self.external_transport_owner.wait_until_all_connected()"
    )
    ready = source.index("self.external_transport_owner.wait_until_all_ready()")

    assert dp_attention_rewrite < endpoint_plan < bind < start < connected < ready
    assert "dp_attention_enabled=config.enable_dp_attention" in source


def test_external_process_monitor_reports_idle_engine_exit():
    reported = []
    reported_event = Event()
    manager = engine_core_mgr.CoreManager.__new__(engine_core_mgr.CoreManager)
    manager.external_transport_mode = True
    manager._external_process_monitor_thread = None
    manager._external_process_monitor_stop = Event()
    manager._engine_core_engine_ranks = [3]
    manager.engine_core_processes = [SimpleNamespace(exitcode=-9)]
    manager.label = "test"
    manager.external_transport_owner = SimpleNamespace(
        mark_engine_failed=lambda rank, message: (
            reported.append((rank, message)),
            reported_event.set(),
        )
    )

    manager._start_external_process_monitor()
    assert reported_event.wait(timeout=1)
    manager._external_process_monitor_stop.set()
    manager._external_process_monitor_thread.join(timeout=1)

    assert reported == [(3, "EngineCore process exited with code -9")]


@pytest.mark.parametrize("model_type", sorted(PER_REQ_CACHE_MODEL_TYPES))
def test_config_reports_per_request_cache_capability(model_type):
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    assert Config.has_per_req_cache.fget(config)


def test_config_rejects_per_request_cache_for_stateless_model():
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    assert not Config.has_per_req_cache.fget(config)


class _FakeExternalTransport:
    def __init__(self):
        self.command = None

    def execute_control_frame_all(self, frame, expected_count, timeout_ms=300_000):
        envelope = EngineCoreIpcCodec.decode_engine_core_envelope(frame)
        self.command = envelope.utility_command.command
        assert expected_count == 2
        assert timeout_ms > 0
        return [
            (
                rank,
                EngineCoreIpcCodec.encode_utility_response(
                    self.command,
                    {"cmd": self.command, "result": {"rank": rank}},
                ),
            )
            for rank in range(expected_count)
        ]


def test_external_transport_utility_bridge_returns_every_rank():
    manager = engine_core_mgr.CoreManager.__new__(engine_core_mgr.CoreManager)
    manager.external_transport_mode = True
    manager.external_transport_owner = _FakeExternalTransport()
    manager.global_engine_count = 2
    manager._external_utility_lock = Lock()

    responses = manager.broadcast_utility_command_sync(
        "get_cache_statistics", timeout=1
    )

    assert [response["result"]["rank"] for response in responses] == [0, 1]


def test_external_transport_rejects_python_request_path():
    manager = engine_core_mgr.CoreManager.__new__(engine_core_mgr.CoreManager)
    manager.external_transport_mode = True

    with pytest.raises(RuntimeError, match="Rust owns"):
        manager.add_request([])
    with pytest.raises(RuntimeError, match="Rust owns"):
        manager.get_output()
    with pytest.raises(RuntimeError, match="Rust owns"):
        manager.is_rest()
