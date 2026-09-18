# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capability discovery: the worker report and the engine-wide intersection.

The load-bearing decision is that engine-wide features are the **intersection**
of the per-rank sets, never the union. A feature present on only some ranks
cannot be driven by a collective -- the ranks that have it would sit in one
collective while the rest went elsewhere, deadlocking the group with no error.
So advertising it is worse than not knowing about it.
"""

import pytest

from atom.model_engine.capabilities import (
    COLLECTIVE_RPC_PROTOCOL_VERSION,
    EngineCapabilities,
    WorkerCapabilities,
)
from atom.rollout.capabilities import CapabilityProviderMixin

# ── harness ────────────────────────────────────────────────────────────────


class _Parallel:
    data_parallel_size = 2
    data_parallel_rank_local = 1
    pipeline_parallel_size = 1


class _Config:
    tp_world_size = 4
    kv_cache_dtype = "fp8"
    parallel_config = _Parallel()


class _Runner(CapabilityProviderMixin):
    """A runner exposing a chosen subset of the advertised methods."""

    def __init__(self, rank=0, methods=(), fp8=False, vocab=0, rdma=False):
        self.rank = rank
        self.config = _Config()
        self._true_vocab_size = vocab
        for name in methods:
            setattr(self, name, lambda *a, **k: None)
        if fp8:
            self._is_fp8_param = lambda *a: True
        if rdma:
            self.receive_weights_rdma = lambda *a, **k: {}


def _worker(rank, methods=(), features=(), version=None):
    return WorkerCapabilities(
        protocol_version=(
            COLLECTIVE_RPC_PROTOCOL_VERSION if version is None else version
        ),
        tp_rank=rank,
        dp_rank_local=0,
        methods=frozenset(methods),
        features=frozenset(features),
    )


# ── the worker report ──────────────────────────────────────────────────────


def test_a_worker_reports_its_version_and_position():
    report = _Runner(rank=3).get_worker_capabilities()
    assert report["protocol_version"] == COLLECTIVE_RPC_PROTOCOL_VERSION
    assert report["tp_rank"] == 3
    assert report["dp_rank_local"] == 1  # from parallel_config, not guessed


def test_it_reports_only_methods_that_actually_exist():
    report = _Runner(
        methods=("clear_kv_cache", "release_memory")
    ).get_worker_capabilities()
    # get_worker_capabilities is the mixin's own method, so it is always there --
    # a consumer can test for discovery support the same way as anything else.
    assert set(report["methods"]) == {
        "clear_kv_cache",
        "release_memory",
        "get_worker_capabilities",
    }
    assert "update_weights" not in report["methods"]


def test_it_does_not_advertise_private_helpers():
    """A dir() sweep would turn every refactor into a capability change, and
    would offer methods that are not safe to drive from outside."""
    runner = _Runner(methods=("clear_kv_cache",))
    runner._secret_helper = lambda: None
    report = runner.get_worker_capabilities()
    assert all(not m.startswith("_") for m in report["methods"])
    assert "_secret_helper" not in report["methods"]


def test_features_are_reported_separately_from_methods():
    """`_is_fp8_param` existing says the code path is there; it does not say
    this particular runner is quantised. Features carry that."""
    plain = _Runner().get_worker_capabilities()
    assert "fp8_weight_update" not in plain["features"]
    assert "vocab_masking" not in plain["features"]

    rich = _Runner(fp8=True, vocab=151936, rdma=True).get_worker_capabilities()
    assert "fp8_weight_update" in rich["features"]
    assert "vocab_masking" in rich["features"]
    assert "rdma_weight_receive" in rich["features"]


def test_rdma_is_absent_until_the_receiver_exists():
    """A7 has not landed, so no runner should claim it yet."""
    assert "rdma_weight_receive" not in _Runner().get_worker_capabilities()["features"]


def test_the_report_is_plain_data():
    """It crosses the worker boundary, so it must not require both sides to
    share a class definition."""
    import pickle

    report = _Runner(fp8=True).get_worker_capabilities()
    assert isinstance(report, dict)
    assert pickle.loads(pickle.dumps(report)) == report


def test_the_provider_is_mixed_into_the_rlhf_runner():
    """Otherwise get_capabilities() reaches a runner that cannot answer."""
    import ast
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "atom"
        / "rollout"
        / "model_runner_ext.py"
    ).read_text()
    cls = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "RLHFModelRunner"
    )
    bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    assert "CapabilityProviderMixin" in bases, f"bases are {bases}"


# ── parsing a report ───────────────────────────────────────────────────────


def test_a_missing_field_degrades_rather_than_fails():
    """An older worker should report less, not break the negotiation."""
    caps = WorkerCapabilities.from_payload({"tp_rank": 2})
    assert caps.tp_rank == 2
    assert caps.protocol_version == 0
    assert caps.methods == frozenset()


def test_a_non_dict_report_is_rejected():
    with pytest.raises(TypeError, match="must be a dict"):
        WorkerCapabilities.from_payload(["not", "a", "dict"])


# ── the engine-wide intersection ───────────────────────────────────────────


def test_features_are_intersected_not_unioned():
    workers = [
        _worker(0, methods=("a", "b"), features=("fp8_weight_update",)),
        _worker(1, methods=("a",), features=()),
    ]
    caps = EngineCapabilities.from_workers(config=_Config(), workers=workers)

    assert caps.methods == frozenset({"a"})
    assert caps.features == frozenset()
    assert not caps.supports("b")
    assert not caps.supports("fp8_weight_update")


def test_what_was_dropped_is_still_reportable():
    """So someone debugging a 'missing' feature can tell present-but-partial
    from absent-everywhere."""
    workers = [
        _worker(0, methods=("a", "b"), features=("x",)),
        _worker(1, methods=("a",), features=()),
    ]
    caps = EngineCapabilities.from_workers(config=_Config(), workers=workers)
    assert caps.partial() == ("b", "x")


def test_a_capability_on_every_rank_is_advertised():
    workers = [_worker(r, methods=("a",), features=("x",)) for r in range(4)]
    caps = EngineCapabilities.from_workers(config=_Config(), workers=workers)
    assert caps.supports("a")
    assert caps.supports("x")
    assert caps.partial() == ()
    assert caps.worker_count == 4


def test_topology_comes_from_config_not_from_the_workers():
    caps = EngineCapabilities.from_workers(
        config=_Config(), workers=[_worker(0), _worker(1)]
    )
    assert caps.tp_world_size == 4  # config says 4 even though 2 workers replied
    assert caps.data_parallel_size == 2
    assert caps.pipeline_parallel_size == 1
    assert caps.kv_cache_dtype == "fp8"


def test_disagreeing_protocol_versions_are_fatal():
    """Negotiating across an incompatible wire contract is worse than refusing."""
    workers = [_worker(0), _worker(1, version=COLLECTIVE_RPC_PROTOCOL_VERSION + 1)]
    with pytest.raises(RuntimeError, match="disagree on the RPC protocol version"):
        EngineCapabilities.from_workers(config=_Config(), workers=workers)


def test_no_workers_is_an_error_not_an_empty_answer():
    with pytest.raises(ValueError, match="no worker capabilities"):
        EngineCapabilities.from_workers(config=_Config(), workers=[])


def test_negotiation_types_import_without_aiter():
    """Both layers must be importable on a machine with no GPU build; this
    module imported them at the top without stubbing AITER."""
    import sys

    for name in ("atom.model_engine.capabilities", "atom.rollout.capabilities"):
        assert name in sys.modules
        assert not hasattr(sys.modules[name], "MessageQueue")
