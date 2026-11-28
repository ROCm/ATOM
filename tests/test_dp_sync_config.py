# SPDX-License-Identifier: MIT
"""DP ranks reject invalid or divergent wire settings before serving steps."""

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.utils import envs
from atom.utils.tbo import ubatching as ub


@pytest.fixture(autouse=True)
def fresh_config(monkeypatch):
    monkeypatch.setattr(ub, "_DP_SYNC_CONFIG", None)
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", "gloo")
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", "0")


def _peer(monkeypatch, votes):
    group = SimpleNamespace(cpu_group=object(), world_size=2)
    calls = []

    def all_reduce(local, op, group):
        assert local.device.type == "cpu" and local.dtype == torch.int64
        assert op == torch.distributed.ReduceOp.MAX
        calls.append(local.tolist())
        local.copy_(torch.maximum(local, torch.tensor(votes, dtype=torch.int64)))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return group, calls


@pytest.mark.parametrize("raw", ["nccl", "rccl ", "", "other"])
def test_unknown_backend_is_rejected(monkeypatch, raw):
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", raw)
    with pytest.raises(ValueError, match="ATOM_DP_SYNC_BACKEND"):
        _ = envs.ATOM_DP_SYNC_BACKEND


@pytest.mark.parametrize("raw", ["-1", "nan", "1.5", "", str(1 << 63)])
def test_verify_requires_nonnegative_int64(monkeypatch, raw):
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", raw)
    with pytest.raises(ValueError, match="ATOM_DP_SYNC_VERIFY"):
        _ = envs.ATOM_DP_SYNC_VERIFY


@pytest.mark.parametrize("raw", ["0", "100", str((1 << 63) - 1)])
def test_verify_accepts_full_valid_range(monkeypatch, raw):
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", raw)
    assert envs.ATOM_DP_SYNC_VERIFY == int(raw)


def test_identical_settings_agree_in_one_cpu_collective_and_freeze(monkeypatch):
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", "RCCL")
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", "100")
    group, calls = _peer(monkeypatch, [0, 0, 1, -1, 100, -100])
    assert ub.initialize_dp_metadata_sync(group) == ("rccl", 100)
    assert len(calls) == 1
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", "gloo")
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", "not-an-integer")
    assert ub._dp_sync_config() == ("rccl", 100)


@pytest.mark.parametrize(
    "backend,verify,peer",
    [
        ("gloo", "0", [0, 0, 1, -1, 0, 0]),
        ("rccl", "0", [0, 0, 0, 0, 0, 0]),
        ("gloo", "0", [0, 0, 0, 0, 100, -100]),
        ("gloo", "100", [0, 0, 0, 0, 0, 0]),
    ],
)
def test_both_ranks_reject_backend_or_verify_divergence(
    monkeypatch, backend, verify, peer
):
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", backend)
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", verify)
    group, calls = _peer(monkeypatch, peer)
    with pytest.raises(ValueError, match="must agree across DP ranks"):
        ub.initialize_dp_metadata_sync(group)
    assert len(calls) == 1 and ub._DP_SYNC_CONFIG is None


@pytest.mark.parametrize(
    "name,raw,peer",
    [
        ("ATOM_DP_SYNC_BACKEND", "nccl", [0, 0, 0, 0, 0, 0]),
        ("ATOM_DP_SYNC_VERIFY", "invalid", [0, 0, 0, 0, 0, 0]),
        ("ATOM_DP_SYNC_VERIFY", str(1 << 63), [0, 0, 0, 0, 0, 0]),
        ("ATOM_DP_SYNC_BACKEND", "gloo", [1, 0, 0, 0, 0, 0]),
        ("ATOM_DP_SYNC_VERIFY", "0", [0, 1, 0, 0, 0, 0]),
    ],
)
def test_invalid_local_or_peer_value_fails_after_collective(
    monkeypatch, name, raw, peer
):
    monkeypatch.setenv(name, raw)
    group, calls = _peer(monkeypatch, peer)
    with pytest.raises(ValueError, match=name):
        ub.initialize_dp_metadata_sync(group)
    assert len(calls) == 1 and ub._DP_SYNC_CONFIG is None


def test_single_rank_initializes_without_any_collective(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("a singleton needs no agreement collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", refused)
    assert ub.initialize_dp_metadata_sync(SimpleNamespace(world_size=1)) == ("gloo", 0)


def _agreement_worker(rank, store_uri, output):
    import os

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=store_uri,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=2)
    cases = [
        ("gloo", "0", False),
        ("rccl", "100", False),
        ("gloo" if rank == 0 else "rccl", "0", True),
        ("gloo", "0" if rank == 0 else "100", True),
        ("nccl" if rank == 0 else "gloo", "0", True),
        ("gloo", "invalid" if rank == 0 else "0", True),
        ("gloo", str(1 << 63) if rank == 0 else "0", True),
    ]
    outcomes = []
    try:
        for backend, verify, reject in cases:
            os.environ["ATOM_DP_SYNC_BACKEND"] = backend
            os.environ["ATOM_DP_SYNC_VERIFY"] = verify
            try:
                ub.initialize_dp_metadata_sync(group)
            except ValueError:
                assert reject
                outcomes.append("rejected")
            else:
                assert not reject
                outcomes.append("accepted")
            dist.barrier()
        (Path(output) / f"rank{rank}.json").write_text(json.dumps(outcomes))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32" or not torch.distributed.is_gloo_available(),
    reason="needs a Linux CPU Gloo device; this Windows wheel cannot create one",
)
def test_startup_agreement_with_two_real_cpu_gloo_ranks(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _agreement_worker, args=((tmp_path / "store").as_uri(), str(tmp_path)), nprocs=2
    )
    expected = ["accepted", "accepted"] + ["rejected"] * 5
    for rank in (0, 1):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == expected
