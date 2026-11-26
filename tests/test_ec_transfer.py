# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The image embedding transfer interface and its Mooncake backend: E's pool
and P's read, over a fake Mooncake TransferEngine."""

import ctypes
import os

import pytest
import torch

from atom.ec_transfer import EcStoreFull, create_reader, create_writer
from atom.ec_transfer import mooncake as ec
from atom.utils import mooncake_hca

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

    monkeypatch.setattr(ec, "MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(ec, "TransferEngine", make_engine, raising=False)
    monkeypatch.setattr(ec, "get_ip", lambda: "192.0.2.7")
    monkeypatch.setenv("ATOM_MOONCAKE_IB_DEVICE", "rocep139s0")
    monkeypatch.setenv("ATOM_EC_LEASE_S", str(LEASE_S))
    # Set first so the TCP case's own write is undone too.
    monkeypatch.setenv("MC_FORCE_TCP", "false")
    monkeypatch.delenv("MC_FORCE_TCP")
    return made


def with_mooncake(params, **fields):
    return {**params, "mooncake": {**params["mooncake"], **fields}}


def rows(seed, n=4, hidden=32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, hidden, generator=g).to(torch.bfloat16)


def test_p_reads_stored_rows_until_lease_expires_and_slot_is_reused(engines):
    clock = Clock()
    # Room for one 4 x 32 bf16 entry only: the next put must reuse its slot.
    pool = ec.MooncakeEcWriter(320, "cpu", clock=clock)
    reader = ec.MooncakeEcReader(1024, "cpu")
    stored = rows(1)
    params = pool.put(11, stored)

    got = reader.get(params, 11, torch.bfloat16)
    assert torch.equal(got, stored)

    clock.now += LEASE_S
    params2 = pool.put(12, rows(2))
    assert params2["mooncake"]["address"] == params["mooncake"]["address"]
    assert reader.get(params, 11, torch.bfloat16) is None
    assert torch.equal(reader.get(params2, 12, torch.bfloat16), rows(2))


def test_read_racing_a_slot_reuse_sees_a_changed_word(engines, monkeypatch):
    clock = Clock()
    pool = ec.MooncakeEcWriter(320, "cpu", clock=clock)
    reader = ec.MooncakeEcReader(1024, "cpu")
    params = pool.put(11, rows(1))

    # P reads the old entry at E's first sync of the reuse: the victim is
    # given up, its old rows not yet overwritten.
    mid_overwrite = []
    monkeypatch.setattr(
        ec,
        "_sync",
        lambda buf: mid_overwrite.append(reader.get(params, 11, torch.bfloat16)),
    )
    clock.now += LEASE_S
    pool.put(12, rows(2))
    assert mid_overwrite[0] is None


def test_full_pool_raises_and_keeps_entries_inside_their_lease(engines):
    clock = Clock()
    pool = ec.MooncakeEcWriter(320, "cpu", clock=clock)
    reader = ec.MooncakeEcReader(1024, "cpu")
    params = pool.put(11, rows(1))

    clock.now += LEASE_S - 0.5
    with pytest.raises(EcStoreFull):
        pool.put(12, rows(2))
    with pytest.raises(EcStoreFull):
        pool.put(13, rows(3, n=100))
    assert torch.equal(reader.get(params, 11, torch.bfloat16), rows(1))


def test_space_is_reused_oldest_first(engines):
    clock = Clock()
    # Three 320-byte entries fill the pool.
    pool = ec.MooncakeEcWriter(960, "cpu", clock=clock)
    reader = ec.MooncakeEcReader(1024, "cpu")
    first = pool.put(1, rows(1))
    clock.now += 1
    second = pool.put(2, rows(2))
    clock.now += 1
    third = pool.put(3, rows(3))

    # Only the first entry is past its lease: it alone gives way.
    clock.now = 1000.0 + LEASE_S + 0.5
    fourth = pool.put(4, rows(4))
    assert fourth["mooncake"]["address"] == first["mooncake"]["address"]
    assert reader.get(first, 1, torch.bfloat16) is None
    for params, seed in ((second, 2), (third, 3), (fourth, 4)):
        assert torch.equal(reader.get(params, seed, torch.bfloat16), rows(seed))
    with pytest.raises(EcStoreFull):
        pool.put(5, rows(5))


def test_read_returns_none_on_foreign_seed_or_failed_read(engines):
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    reader = ec.MooncakeEcReader(1024, "cpu")
    params = pool.put(11, rows(1))

    assert reader.get(params, 99, torch.bfloat16) is None
    assert (
        reader.get(with_mooncake(params, session="192.0.2.9:1"), 11, torch.bfloat16)
        is None
    )
    engines[-1].fail_reads = True
    assert reader.get(params, 11, torch.bfloat16) is None


def test_reader_refuses_rows_larger_than_its_buffer(engines):
    pool = ec.MooncakeEcWriter(4096, "cpu", clock=Clock())
    reader = ec.MooncakeEcReader(64, "cpu")
    params = pool.put(11, rows(1))
    assert reader.get(params, 11, torch.bfloat16) is None


def test_pool_registers_on_the_configured_device(engines):
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    engine = engines[0]
    assert engine.init_args == ("192.0.2.7", "P2PHANDSHAKE", "rdma", "rocep139s0")
    assert engine.regions == [(pool._buf.data_ptr(), 1024)]
    assert pool.session == "192.0.2.7:17000"


def test_pool_registers_on_every_configured_device(engines, monkeypatch):
    monkeypatch.setenv("ATOM_MOONCAKE_IB_DEVICE", "rocep139s0, rocep140s0,rocep139s0")
    ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    assert engines[0].init_args[3] == "rocep139s0,rocep140s0"


def test_engine_without_configured_device_uses_the_gpu_local_hca(engines, monkeypatch):
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(mooncake_hca, "auto_select_ib_device", lambda idx: f"rdma{idx}")
    engine, _ = ec._open_engine(torch.device("cuda", 1), "rdma")
    assert engine.init_args[3] == "rdma5"


def test_pool_on_tcp_registers_no_hca(engines, monkeypatch):
    ec.MooncakeEcWriter(1024, "cpu", protocol="tcp", clock=Clock())
    assert engines[0].init_args[2:] == ("tcp", "")
    assert os.environ["MC_FORCE_TCP"] == "true"


def test_read_returns_none_when_p_reads_another_dtype(engines):
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    reader = ec.MooncakeEcReader(1024, "cpu")
    params = pool.put(11, rows(1))
    assert params["dtype"] == "bfloat16"
    assert reader.get(params, 11, torch.float16) is None


def test_put_into_free_space_syncs_once(engines, monkeypatch):
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    syncs = []
    monkeypatch.setattr(ec, "_sync", syncs.append)
    pool.put(11, rows(1))
    assert len(syncs) == 1


def test_matched_rails_warn_that_reads_take_no_rail(engines, monkeypatch, caplog):
    monkeypatch.setenv("ATOM_MOONCAKE_MATCHED_RAILS", "rocep139s0,rocep140s0")
    with caplog.at_level("WARNING", logger="atom"):
        ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    assert "ATOM_MOONCAKE_MATCHED_RAILS" in caplog.text


def test_handle_names_its_transport_and_keeps_mooncake_fields_apart(engines):
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    handle = pool.put(11, rows(1))
    assert {k: v for k, v in handle.items() if k != "mooncake"} == {
        "transport": "mooncake",
        "key": 11,
        "rows": 4,
        "dtype": "bfloat16",
    }
    assert set(handle["mooncake"]) == {"session", "address", "length", "generation"}


def test_read_refuses_generation_zero(engines):
    # A zeroed or given-up entry's word is 0: a handle claiming 0 must not
    # read it as a match.
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    reader = ec.MooncakeEcReader(1024, "cpu")
    handle = pool.put(11, rows(1))
    pool._word(handle["mooncake"]["address"] - pool._buf.data_ptr()).fill_(0)
    assert reader.get(with_mooncake(handle, generation=0), 11, torch.bfloat16) is None


@pytest.mark.parametrize(
    "broken",
    [
        None,
        "not a dict",
        {},
        {"transport": "other"},
        {"rows": "4"},
        {"rows": 0},
        {"mooncake": None},
        {"mooncake": {}},
        {"mooncake": {"length": -256}},
        {"mooncake": {"length": 250}},
        {"mooncake": {"address": "0x10"}},
        {"mooncake": {"session": 7}},
    ],
)
def test_malformed_handle_reads_none_without_raising(engines, broken):
    pool = ec.MooncakeEcWriter(1024, "cpu", clock=Clock())
    reader = ec.MooncakeEcReader(1024, "cpu")
    handle = pool.put(11, rows(1))
    if isinstance(broken, dict) and "mooncake" in broken and broken["mooncake"]:
        bad = with_mooncake(handle, **broken["mooncake"])
    elif isinstance(broken, dict):
        bad = {k: v for k, v in {**handle, **broken}.items() if broken or k != "key"}
    else:
        bad = broken
    assert reader.get(bad, 11, torch.bfloat16) is None
    assert torch.equal(reader.get(handle, 11, torch.bfloat16), rows(1))


def test_factory_builds_mooncake_and_refuses_unknown_transports(engines):
    writer = create_writer("mooncake", 1024, "cpu")
    reader = create_reader("mooncake", 1024, "cpu")
    handle = writer.put(11, rows(1))
    assert torch.equal(reader.get(handle, 11, torch.bfloat16), rows(1))
    with pytest.raises(ValueError, match="unknown image embedding transport"):
        create_reader("nixl", 1024, "cpu")
    with pytest.raises(ValueError, match="unknown image embedding transport"):
        create_writer("nixl", 1024, "cpu")


def test_reader_without_mooncake_raises_on_construction(engines, monkeypatch):
    monkeypatch.setattr(ec, "MOONCAKE_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Mooncake is not installed"):
        create_reader("mooncake", 1024, "cpu")


def test_reader_takes_the_kv_connectors_hca_options(engines, monkeypatch):
    ec.MooncakeEcReader(1024, "cpu", kv_transfer_config={"ib_device": "rocep140s0"})
    assert engines[-1].init_args[3] == "rocep140s0"

    opened = []
    monkeypatch.setattr(
        ec, "_open_engine", lambda *a, **kw: opened.append((a, kw)) or (engines[-1], "")
    )
    kv = {"ib_device": "", "ib_enable_alternate_hca": True, "ib_hca_count": 3}
    ec.MooncakeEcReader(1024, "cpu", kv_transfer_config=kv)
    assert opened[0][0][2] == ""
    assert opened[0][1] == {"enable_alternate_hca": True, "hca_count": 3}


def test_engine_registers_on_alternate_hcas(engines, monkeypatch):
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE")
    monkeypatch.setattr(mooncake_hca, "auto_select_ib_device", lambda i: f"ionic_{i}")
    monkeypatch.setattr(mooncake_hca, "ib_device_exists", lambda _: True)
    monkeypatch.setattr(ec, "_physical_index", lambda i: i)
    ec._open_engine(
        torch.device("cuda", 1), "rdma", enable_alternate_hca=True, hca_count=3
    )
    assert engines[-1].init_args[3] == "ionic_1,ionic_0,ionic_2"
