# SPDX-License-Identifier: MIT
"""CPU tests for per-rail registration ownership and concurrent first use."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock, call

import pytest

from atom.kv_transfer.disaggregation.mooncake.rail_engine_pool import RailEnginePool


class FakeEngine:
    def __init__(self):
        self.device = None
        self.ip = None
        self.regions = []
        self.released = []
        self.fail_ptr = None
        self.init_result = 0

    def initialize(self, ip, metadata, protocol, device):
        assert metadata == "P2PHANDSHAKE"
        assert protocol == "rdma"
        self.device = device
        self.ip = ip
        return self.init_result

    def register_memory(self, ptr, size):
        if ptr == self.fail_ptr:
            return -202
        self.regions.append((ptr, size))
        return 0

    def unregister_memory(self, ptr):
        self.released.append(ptr)
        return 0

    def get_rpc_port(self):
        return 17000


@pytest.fixture
def pool_fixture():
    primary = FakeEngine()
    created = []

    def factory():
        engine = FakeEngine()
        created.append(engine)
        return engine

    pool = RailEnginePool(
        factory,
        primary,
        "ionic_2",
        ["ionic_2", "ionic_4", "ionic_6"],
        lambda device: "127.0.0.1",
    )
    pool.set_regions([1024, 2048, 3072], [64, 128, 256])
    return pool, primary, created


def test_concurrent_first_use_registers_one_engine_per_rail(pool_fixture):
    pool, primary, created = pool_fixture
    barrier = Barrier(16)

    def get_engine(index):
        barrier.wait(timeout=10)
        device = "ionic_4" if index % 2 else "ionic_6"
        return device, pool.get(device)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(get_engine, range(16)))

    assert len(created) == 2
    for device in ("ionic_4", "ionic_6"):
        selected = [engine for rail, engine in results if rail == device]
        assert all(engine is selected[0] for engine in selected)
        assert selected[0] is not primary
        assert selected[0].device == device
        assert selected[0].regions == [(1024, 64), (2048, 128), (3072, 256)]
    assert pool.get("ionic_2") is primary
    assert not primary.regions  # The connector already registered this engine.


@pytest.mark.parametrize(
    "device", [None, "", "ionic_0", "ionic_2,ionic_6", ["ionic_6"]]
)
def test_invalid_peer_rail_never_falls_back(pool_fixture, device):
    pool, primary, created = pool_fixture
    with pytest.raises(ValueError, match="configured matching rail"):
        pool.get(device)
    assert not created
    assert pool.get("ionic_2") is primary


def test_registration_failure_rolls_back_in_reverse_and_is_not_retried():
    primary, failed = FakeEngine(), FakeEngine()
    failed.fail_ptr = 3072
    factory = Mock(return_value=failed)
    pool = RailEnginePool(
        factory, primary, "ionic_2", ["ionic_2", "ionic_6"], lambda device: "127.0.0.1"
    )
    pool.set_regions([1024, 2048, 3072], [64, 128, 256])

    with pytest.raises(RuntimeError, match="register_memory"):
        pool.get("ionic_6")
    assert failed.released == [2048, 1024]

    failed.fail_ptr = None
    with pytest.raises(RuntimeError, match="previously failed"):
        pool.get("ionic_6")
    factory.assert_called_once()
    assert pool.get("ionic_2") is primary


def test_initialize_failure_is_remembered_without_registering():
    primary, failed = FakeEngine(), FakeEngine()
    failed.init_result = -1
    factory = Mock(return_value=failed)
    pool = RailEnginePool(
        factory, primary, "ionic_2", ["ionic_2", "ionic_6"], lambda device: "127.0.0.1"
    )
    pool.set_regions([1024], [64])
    with pytest.raises(RuntimeError, match="initialize"):
        pool.get("ionic_6")
    assert not failed.regions
    with pytest.raises(RuntimeError, match="previously failed"):
        pool.get("ionic_6")
    factory.assert_called_once()


def test_rollback_continues_after_unregister_error():
    failed = FakeEngine()
    failed.fail_ptr = 3072
    failed.unregister_memory = Mock(side_effect=[RuntimeError("driver error"), 0])
    pool = RailEnginePool(
        lambda: failed,
        FakeEngine(),
        "ionic_2",
        ["ionic_2", "ionic_6"],
        lambda device: "127.0.0.1",
    )
    pool.set_regions([1024, 2048, 3072], [64, 128, 256])
    with pytest.raises(RuntimeError, match="register_memory"):
        pool.get("ionic_6")
    assert [call.args[0] for call in failed.unregister_memory.call_args_list] == [
        2048,
        1024,
    ]
    with pytest.raises(RuntimeError, match="previously failed"):
        pool.get("ionic_6")


def test_regions_must_be_ready_and_cannot_be_replaced():
    pool = RailEnginePool(
        FakeEngine, FakeEngine(), "ionic_2", ["ionic_2"], lambda device: "127.0.0.1"
    )
    with pytest.raises(RuntimeError, match="not ready"):
        pool.get("ionic_2")
    for ptrs, sizes in [([], []), ([1024], [])]:
        with pytest.raises(ValueError):
            pool.set_regions(ptrs, sizes)
    pool.set_regions([1024], [64])
    with pytest.raises(RuntimeError, match="Replacing live"):
        pool.set_regions([2048], [128])


def test_primary_must_be_a_configured_rail():
    with pytest.raises(ValueError, match="primary HCA"):
        RailEnginePool(
            FakeEngine, FakeEngine(), "ionic_2", ["ionic_6"], lambda device: "127.0.0.1"
        )


def test_each_rail_resolves_its_own_address_once():
    primary = FakeEngine()
    created = []

    def factory():
        engine = FakeEngine()
        created.append(engine)
        return engine

    addresses = {"ionic_4": "192.0.2.4", "ionic_6": "192.0.2.6"}
    resolver = Mock(side_effect=addresses.__getitem__)
    pool = RailEnginePool(
        factory,
        primary,
        "ionic_2",
        ["ionic_2", "ionic_4", "ionic_6"],
        local_ip_for_device=resolver,
    )
    pool.set_regions([1024], [64])

    assert pool.get("ionic_2") is primary
    resolver.assert_not_called()
    for device, ip in addresses.items():
        engine = pool.get(device)
        assert engine.device == device
        assert engine.ip == ip
        assert pool.get(device) is engine
    assert resolver.call_args_list == [call("ionic_4"), call("ionic_6")]
    assert len(created) == 2


def test_address_resolution_failure_is_cached_before_engine_creation():
    primary = FakeEngine()
    factory = Mock()
    resolver = Mock(side_effect=OSError("RDMA address unavailable"))
    pool = RailEnginePool(
        factory,
        primary,
        "ionic_2",
        ["ionic_2", "ionic_6"],
        local_ip_for_device=resolver,
    )
    pool.set_regions([1024], [64])

    with pytest.raises(OSError, match="RDMA address unavailable"):
        pool.get("ionic_6")
    with pytest.raises(RuntimeError, match="previously failed"):
        pool.get("ionic_6")
    factory.assert_not_called()
    resolver.assert_called_once_with("ionic_6")
    assert pool.get("ionic_2") is primary


@pytest.mark.parametrize("address", [None, ""])
def test_empty_rail_address_does_not_initialize_an_engine(address):
    factory = Mock()
    pool = RailEnginePool(
        factory,
        FakeEngine(),
        "ionic_2",
        ["ionic_2", "ionic_6"],
        local_ip_for_device=lambda device: address,
    )
    pool.set_regions([1024], [64])
    with pytest.raises(ValueError, match="No local address"):
        pool.get("ionic_6")
    factory.assert_not_called()
