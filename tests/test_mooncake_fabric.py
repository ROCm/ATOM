# SPDX-License-Identifier: MIT
"""CPU tests for Mooncake over the scale-up fabric (EXTERNAL metadata)."""

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.mooncake import fabric
from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc


class FakeEngine:
    def __init__(self):
        self.imports = []
        self.binds = []
        self.import_result = 7
        self.bind_result = 0

    def import_segment(self, name, blob):
        self.imports.append((name, blob))
        return self.import_result

    def bind_segment(self, sid, device):
        self.binds.append((sid, device))
        return self.bind_result


def request(blob=b"seg-v1"):
    return {"consumer_segment": blob, "consumer_segment_tag": fabric.segment_tag(blob)}


def test_fabric_protocols():
    assert fabric.is_fabric_protocol("ualoe")
    assert fabric.is_fabric_protocol(" HIP ")
    assert not fabric.is_fabric_protocol("rdma")
    assert not fabric.is_fabric_protocol(None)


def test_fabric_selects_no_hca():
    assert mc._select_ib_devices("ualoe", "rdma0", 0) == []
    assert mc._select_ib_devices("hip", "", None) == []


def test_registration_ranges_merge_without_pool(monkeypatch):
    monkeypatch.setattr(fabric, "_pool_range", None)
    regions = [(300, 100), (0, 100), (100, 50), (500, 0)]
    assert fabric.registration_ranges(regions) == [(0, 150), (300, 100)]


def test_registration_ranges_collapse_into_pool(monkeypatch):
    monkeypatch.setattr(fabric, "_pool_range", (1000, 4000))
    regions = [(1000, 1000), (3000, 500), (4500, 500), (9000, 64)]
    assert fabric.registration_ranges(regions) == [(1000, 4000), (9000, 64)]


def test_importer_maps_once_per_tag():
    engine = FakeEngine()
    importer = fabric.SegmentImporter(engine, local_device=3)
    importer.ensure("10.0.0.2:17000", request())
    importer.ensure("10.0.0.2:17000", request())
    assert engine.imports == [("10.0.0.2:17000", b"seg-v1")]
    assert engine.binds == [(7, 3)]

    importer.ensure("10.0.0.2:17000", request(b"seg-v2"))
    importer.ensure("10.0.0.3:17000", request(b"seg-v2"))
    assert [name for name, _ in engine.imports] == [
        "10.0.0.2:17000",
        "10.0.0.2:17000",
        "10.0.0.3:17000",
    ]


def test_importer_rejects_requests_without_segment():
    importer = fabric.SegmentImporter(FakeEngine(), local_device=0)
    with pytest.raises(RuntimeError, match="no Mooncake segment"):
        importer.ensure("10.0.0.2:17000", {"request_id": "r"})


@pytest.mark.parametrize("field", ["import_result", "bind_result"])
def test_importer_failure_is_retried(field):
    engine = FakeEngine()
    setattr(engine, field, -1)
    importer = fabric.SegmentImporter(engine, local_device=0)
    with pytest.raises(RuntimeError):
        importer.ensure("10.0.0.2:17000", request())
    setattr(engine, field, 7 if field == "import_result" else 0)
    importer.ensure("10.0.0.2:17000", request())
    assert len(engine.imports) == 2


def test_allocate_kv_pool_only_for_fabric(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mc, "allocate_fabric_pool", lambda cfg, n, dev: calls.append(n) or "pool"
    )

    def cfg(**kv):
        return SimpleNamespace(kv_transfer_config=kv)

    assert mc.MooncakeConnector.allocate_kv_pool(cfg(protocol="rdma"), 64, 0) is None
    assert (
        mc.MooncakeConnector.allocate_kv_pool(
            cfg(protocol="ualoe", fabric_kv_pool=False), 64, 0
        )
        is None
    )
    assert mc.MooncakeConnector.allocate_kv_pool(cfg(protocol="ualoe"), 64, 0) == (
        "pool"
    )
    assert calls == [64]


def test_contiguous_block_runs():
    assert mc._contiguous_block_runs([4, 5, 6, 9, 10, 2], [0, 1, 2, 3, 7, 8]) == [
        (4, 0, 3),
        (9, 3, 1),
        (10, 7, 1),
        (2, 8, 1),
    ]
    assert mc._contiguous_block_runs([], []) == []


def test_connector_without_init_is_not_fabric():
    connector = object.__new__(mc.MooncakeConnector)
    assert connector._segment_importer is None
    assert connector._segment_blob is None
