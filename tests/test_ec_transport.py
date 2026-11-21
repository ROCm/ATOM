# SPDX-License-Identifier: MIT
"""E's image embedding pool and P's read, over a fake Mooncake TransferEngine."""

import ctypes
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# The connector imports aiter's parallel state, which a CPU runner lacks: stub
# what is missing, as tests/test_pd_pp.py does, and take the top-level name
# back down so later modules' `importorskip("aiter")` still skips.
_stubbed = []
for _name in ("aiter", "aiter.dist", "aiter.dist.parallel_state"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
        sys.modules[_name].__path__ = []
        _stubbed.append(_name)
for _fn in ("get_dp_group", "get_tp_group"):
    if not hasattr(sys.modules["aiter.dist.parallel_state"], _fn):
        setattr(sys.modules["aiter.dist.parallel_state"], _fn, MagicMock())

from atom.kv_transfer.disaggregation.mooncake import ec_transport as ec
from atom.kv_transfer.disaggregation.mooncake import (
    mooncake_connector as mc,
)

if "aiter" in _stubbed:
    del sys.modules["aiter"]

LEASE_S = 10.0


class FakeEngine:
    """Moves bytes between registered regions of engines in this process."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.regions = []
        self.init_args = None
        self.fail_reads = False

    def initialize(self, ip, metadata, protocol, ib_device):
        self.init_args = (ip, metadata, protocol, ib_device)
        self.port = 17000 + len(self.sessions)
        self.sessions[f"{ip}:{self.port}"] = self
        return 0

    def get_rpc_port(self):
        return self.port

    def register_memory(self, ptr, length):
        self.regions.append((ptr, length))
        return 0

    def _registered(self, ptr, length):
        return any(p <= ptr and ptr + length <= p + n for p, n in self.regions)

    def transfer_sync_read(self, session, dst, src, length):
        peer = self.sessions.get(session)
        if self.fail_reads or peer is None:
            return -1
        assert self._registered(dst, length) and peer._registered(src, length)
        ctypes.memmove(dst, src, length)
        return 0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def engines(monkeypatch):
    sessions = {}
    made = []

    def make_engine():
        made.append(FakeEngine(sessions))
        return made[-1]

    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(mc, "TransferEngine", make_engine, raising=False)
    monkeypatch.setattr(ec, "get_ip", lambda: "192.0.2.7")
    monkeypatch.setenv("ATOM_MOONCAKE_IB_DEVICE", "rocep139s0")
    monkeypatch.setenv("ATOM_EC_LEASE_S", str(LEASE_S))
    # Set first so the TCP case's own write is undone too.
    monkeypatch.setenv("MC_FORCE_TCP", "false")
    monkeypatch.delenv("MC_FORCE_TCP")
    return made


def rows(seed, n=4, hidden=32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, hidden, generator=g).to(torch.bfloat16)


def test_p_reads_stored_rows_until_lease_expires_and_slot_is_reused(engines):
    clock = Clock()
    # Room for one 4 x 32 bf16 entry only: the next put must reuse its slot.
    pool = ec.EcPool(320, "cpu", clock=clock)
    reader = ec.EcReader(1024, "cpu")
    stored = rows(1)
    params = pool.put(11, stored)

    got = reader.read(params, 11, torch.bfloat16)
    assert torch.equal(got, stored)

    clock.now += LEASE_S
    params2 = pool.put(12, rows(2))
    assert params2["address"] == params["address"]
    assert reader.read(params, 11, torch.bfloat16) is None
    assert torch.equal(reader.read(params2, 12, torch.bfloat16), rows(2))


def test_read_racing_a_slot_reuse_sees_a_changed_word(engines, monkeypatch):
    clock = Clock()
    pool = ec.EcPool(320, "cpu", clock=clock)
    reader = ec.EcReader(1024, "cpu")
    params = pool.put(11, rows(1))

    # P reads the old entry at E's first sync of the reuse: the victim is
    # given up, its old rows not yet overwritten.
    mid_overwrite = []
    monkeypatch.setattr(
        ec,
        "_sync",
        lambda buf: mid_overwrite.append(reader.read(params, 11, torch.bfloat16)),
    )
    clock.now += LEASE_S
    pool.put(12, rows(2))
    assert mid_overwrite[0] is None


def test_full_pool_raises_and_keeps_entries_inside_their_lease(engines):
    clock = Clock()
    pool = ec.EcPool(320, "cpu", clock=clock)
    reader = ec.EcReader(1024, "cpu")
    params = pool.put(11, rows(1))

    clock.now += LEASE_S - 0.5
    with pytest.raises(ec.EcPoolFull):
        pool.put(12, rows(2))
    with pytest.raises(ec.EcPoolFull):
        pool.put(13, rows(3, n=100))
    assert torch.equal(reader.read(params, 11, torch.bfloat16), rows(1))


def test_space_is_reused_oldest_first(engines):
    clock = Clock()
    # Three 320-byte entries fill the pool.
    pool = ec.EcPool(960, "cpu", clock=clock)
    reader = ec.EcReader(1024, "cpu")
    first = pool.put(1, rows(1))
    clock.now += 1
    second = pool.put(2, rows(2))
    clock.now += 1
    third = pool.put(3, rows(3))

    # Only the first entry is past its lease: it alone gives way.
    clock.now = 1000.0 + LEASE_S + 0.5
    fourth = pool.put(4, rows(4))
    assert fourth["address"] == first["address"]
    assert reader.read(first, 1, torch.bfloat16) is None
    for params, seed in ((second, 2), (third, 3), (fourth, 4)):
        assert torch.equal(reader.read(params, seed, torch.bfloat16), rows(seed))
    with pytest.raises(ec.EcPoolFull):
        pool.put(5, rows(5))


def test_read_returns_none_on_foreign_seed_or_failed_read(engines):
    pool = ec.EcPool(1024, "cpu", clock=Clock())
    reader = ec.EcReader(1024, "cpu")
    params = pool.put(11, rows(1))

    assert reader.read(params, 99, torch.bfloat16) is None
    assert reader.read({**params, "session": "192.0.2.9:1"}, 11, torch.bfloat16) is None
    engines[-1].fail_reads = True
    assert reader.read(params, 11, torch.bfloat16) is None


def test_reader_refuses_rows_larger_than_its_buffer(engines):
    pool = ec.EcPool(4096, "cpu", clock=Clock())
    reader = ec.EcReader(64, "cpu")
    params = pool.put(11, rows(1))
    assert reader.read(params, 11, torch.bfloat16) is None


def test_pool_registers_on_the_configured_device(engines):
    pool = ec.EcPool(1024, "cpu", clock=Clock())
    engine = engines[0]
    assert engine.init_args == ("192.0.2.7", "P2PHANDSHAKE", "rdma", "rocep139s0")
    assert engine.regions == [(pool._buf.data_ptr(), 1024)]
    assert pool.session == "192.0.2.7:17000"


def test_pool_registers_on_every_configured_device(engines, monkeypatch):
    monkeypatch.setenv("ATOM_MOONCAKE_IB_DEVICE", "rocep139s0, rocep140s0,rocep139s0")
    ec.EcPool(1024, "cpu", clock=Clock())
    assert engines[0].init_args[3] == "rocep139s0,rocep140s0"


def test_engine_without_configured_device_uses_the_gpu_local_hca(engines, monkeypatch):
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(mc, "_auto_select_ib_device", lambda idx: f"rdma{idx}")
    engine, _ = ec._open_engine(torch.device("cuda", 1), "rdma")
    assert engine.init_args[3] == "rdma5"


def test_pool_on_tcp_registers_no_hca(engines, monkeypatch):
    ec.EcPool(1024, "cpu", protocol="tcp", clock=Clock())
    assert engines[0].init_args[2:] == ("tcp", "")
    assert mc.os.environ["MC_FORCE_TCP"] == "true"
