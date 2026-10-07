"""CPU contracts for an explicit, bounded NIXL READ-only transport control."""

import argparse
import builtins
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/pd_transport_probe.py"
)
# Synthetic inventory fixture only; never an operational NIC/GID selection.
SELECTION = "fixture_rdma:1@3"


def setup_probe(tmp_path, monkeypatch, mode="1", rank=0):
    spec = importlib.util.spec_from_file_location("read_only_probe", SCRIPT)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    env = {
        "NODE_RANK": str(rank),
        "IPADDRS": "192.0.2.1,192.0.2.2",
        "RUN_DIR": str(tmp_path),
        "SLURM_JOB_ID": "fixture",
        "ATOMESH_VLLM_SOURCE_SHA": probe.FIXED_SHA,
        "DOCKER_IMAGE": "image@" + probe.IMAGE_DIGEST,
        "ATOMESH_TRANSPORT_GPU": "1",
        "ATOMESH_TRANSPORT_UCX_SELECTION": SELECTION,
        "ATOMESH_TRANSPORT_MORI_BYTES": "4096,2493186048",
    }
    if mode is not None:
        env["ATOMESH_TRANSPORT_NIXL_READ_ONLY"] = mode
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr(probe.signal, "signal", lambda *_: None)
    monkeypatch.setattr(probe.signal, "alarm", lambda *_: None)
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "local-fixture")
    inventory = {
        "ports": [
            {
                "device": "fixture_rdma",
                "port": "1",
                "state": "4: ACTIVE",
                "link_layer": "Ethernet",
                "gids": [
                    {
                        "index": "3",
                        "gid": "::ffff:192.0.2.1",
                        "type": "RoCE v2",
                        "ndev": "fixture_eth",
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(probe, "inventory", lambda: inventory)
    peer = {
        "gpu": "1",
        "eligible": True,
        "hostname": "peer-fixture",
        "selection": SELECTION,
        "nixl_read_only": "1",
    }
    monkeypatch.setattr(probe, "wait_json", lambda *_: peer)
    calls = []

    def supervise(argv, log, timeout, child_env):
        kind = argv[argv.index("--child") + 1]
        calls.append((kind, argv, child_env))
        assert timeout == 60
        probe.write_json(argv[argv.index("--result") + 1], {"status": "PASS"})
        return 0, False

    monkeypatch.setattr(probe, "supervise", supervise)
    return probe, env, peer, inventory, calls


def report(tmp_path, rank=0):
    return json.loads(
        (
            tmp_path / f"transport-diagnostic/benchmark/rank-{rank}/summary.json"
        ).read_text()
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_read_only_skips_mori_parsing_and_all_registration(tmp_path, monkeypatch, rank):
    probe, env, _, _, calls = setup_probe(tmp_path, monkeypatch, rank=rank)
    env["ATOMESH_TRANSPORT_MORI_BYTES"] = "must-not-be-parsed"
    assert probe.main() == 0
    assert [call[0] for call in calls] == [
        "nixl-create",
        "nixl-create",
        "nixl-transfer",
    ]
    for _, argv, _ in calls:
        assert argv[argv.index("--size") + 1] == "4096"
    assert "UCX_NET_DEVICES" not in calls[0][2]
    assert calls[1][2]["UCX_NET_DEVICES"] == "fixture_rdma:1"
    assert calls[1][2]["UCX_IB_GID_INDEX"] == "3"
    assert "UCX_TLS" not in calls[1][2]
    assert calls[2][2]["UCX_TLS"] == "rc,rocm_copy,rocm_ipc"
    summary = report(tmp_path, rank)
    assert summary["collection"] == "COMPLETED"
    assert summary["model_pd"] == "NOT_TESTED"
    assert summary["stages"]["mori-register"] == {
        "status": "UNKNOWN",
        "classification": "NOT_TESTED",
        "reason": "NIXL READ-only control",
    }
    ready = json.loads(
        (tmp_path / f"transport-diagnostic/benchmark/ready-{rank}.json").read_text()
    )
    assert ready["nixl_read_only"] == "1"


@pytest.mark.parametrize("mode", [None, "0"])
def test_default_gpu_stages_and_mori_sizes_remain_unchanged(
    tmp_path, monkeypatch, mode
):
    probe, _, peer, _, calls = setup_probe(tmp_path, monkeypatch, mode=mode)
    peer.pop("nixl_read_only")
    assert probe.main() == 0
    assert [call[0] for call in calls] == [
        "nixl-create",
        "nixl-create",
        "nixl-transfer",
        "mori-register",
        "mori-register",
    ]
    assert [argv[argv.index("--size") + 1] for _, argv, _ in calls[-2:]] == [
        "4096",
        "2493186048",
    ]
    assert "nixl_read_only" not in json.loads(
        (tmp_path / "transport-diagnostic/benchmark/ready-0.json").read_text()
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("ATOMESH_TRANSPORT_NIXL_READ_ONLY", "true"),
        ("ATOMESH_TRANSPORT_GPU", "0"),
        ("ATOMESH_TRANSPORT_UCX_SELECTION", ""),
    ],
)
def test_read_only_requires_explicit_valid_opt_ins(tmp_path, monkeypatch, key, value):
    probe, env, _, _, calls = setup_probe(tmp_path, monkeypatch)
    env[key] = value
    with pytest.raises(ValueError):
        probe.main()
    assert calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "bad_selection",
        "inactive",
        "zero_gid",
        "peer_default",
        "peer_boolean",
        "peer_selection",
        "same_host",
        "peer_gpu",
        "peer_ineligible",
        "missing_peer",
        "selected_failure",
    ],
)
def test_read_only_gates_transfer_without_mori_fallback(tmp_path, monkeypatch, fault):
    probe, env, peer, inventory, calls = setup_probe(tmp_path, monkeypatch)
    if fault == "bad_selection":
        env["ATOMESH_TRANSPORT_UCX_SELECTION"] = "unobserved:1@3"
    elif fault == "inactive":
        inventory["ports"][0]["state"] = "1: DOWN"
    elif fault == "zero_gid":
        inventory["ports"][0]["gids"][0]["gid"] = "::"
    elif fault == "peer_default":
        peer.pop("nixl_read_only")
    elif fault == "peer_boolean":
        peer["nixl_read_only"] = True
    elif fault == "peer_selection":
        peer["selection"] = "unobserved:1@3"
    elif fault == "same_host":
        peer["hostname"] = "local-fixture"
    elif fault == "peer_gpu":
        peer["gpu"] = "0"
    elif fault == "peer_ineligible":
        peer["eligible"] = False
    elif fault == "missing_peer":

        def missing(*_):
            raise probe.Unknown("peer missing")

        monkeypatch.setattr(probe, "wait_json", missing)
    elif fault == "selected_failure":
        original = probe.supervise

        def failed(argv, log, timeout, env):
            result = original(argv, log, timeout, env)
            if len(calls) == 2:
                probe.write_json(argv[argv.index("--result") + 1], {"status": "FAIL"})
            return result

        monkeypatch.setattr(probe, "supervise", failed)
    assert probe.main() == 0
    assert all(kind == "nixl-create" for kind, _, _ in calls)
    summary = report(tmp_path)
    assert summary["stages"]["nixl-rdma-gpu-read"]["classification"] == "NOT_TESTED"
    assert summary["stages"]["mori-register"]["reason"] == "NIXL READ-only control"


def test_default_peer_cannot_transfer_with_read_only_peer(tmp_path, monkeypatch):
    probe, _, _, _, calls = setup_probe(tmp_path, monkeypatch, mode=None)
    assert probe.main() == 0
    assert [kind for kind, _, _ in calls] == [
        "nixl-create",
        "nixl-create",
        "mori-register",
        "mori-register",
    ]


@pytest.mark.parametrize("outcome", ["DONE", "FAILED", "PROC", "corrupt"])
def test_existing_tiny_read_payload_and_cleanup_contract(
    tmp_path, monkeypatch, outcome
):
    probe, _, _, _, _ = setup_probe(tmp_path, monkeypatch, rank=1)
    events = []

    class Tensor:
        def to(self, _):
            return self

        def zero_(self):
            pass

        def data_ptr(self):
            return 100

        def get_device(self):
            return 0

        def cpu(self):
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return bytes(range(256)) * 16

    class Agent:
        backends = ("UCX",)

        def __init__(self, *args):
            pass

        def get_plugin_list(self):
            return ["UCX"]

        def get_backend_params(self, _):
            return {}

        def register_memory(self, tensor, **kwargs):
            events.append("register")
            return "registration"

        def get_agent_metadata(self):
            return b"metadata"

        def add_remote_agent(self, metadata):
            return "remote"

        def get_xfer_descs(self, *args):
            return "descriptor"

        def initialize_xfer(self, operation, *args, **kwargs):
            assert operation == "READ"
            return "handle"

        def transfer(self, handle):
            return "DONE" if outcome == "corrupt" else outcome

        def check_xfer_state(self, handle):
            return "PROC"

        def release_xfer_handle(self, handle):
            events.append("release")

        def remove_remote_agent(self, remote):
            events.append("remove")

        def deregister_memory(self, registration, **kwargs):
            events.append("deregister")

    from types import SimpleNamespace

    torch = SimpleNamespace(
        version=SimpleNamespace(hip="fixture"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda _: None,
            synchronize=lambda: None,
        ),
        int32="int32",
        uint8="uint8",
        equal=lambda *_: outcome != "corrupt",
    )

    def arange(size, **kwargs):
        assert size == 4096
        return Tensor()

    torch.arange = arange
    monkeypatch.setitem(sys.modules, "torch", torch)
    real_import = builtins.__import__

    def no_mori(name, *args, **kwargs):
        assert name.split(".")[0] != "mori"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mori)
    monkeypatch.setattr(probe, "load_nixl", lambda: (Agent, lambda **kwargs: kwargs))
    exchange = tmp_path / "exchange"

    def peer(path):
        if path.name.startswith("metadata-"):
            return {
                "hostname": "peer-fixture",
                "rank": 0,
                "bytes": 4096,
                "metadata": "bWV0YWRhdGE=",
                "pointer": 200,
                "gpu": 0,
            }
        return json.loads(path.read_text())

    monkeypatch.setattr(probe, "wait_json", peer)
    ticks = iter((0, 21))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    args = argparse.Namespace(
        child="nixl-transfer",
        rank=1,
        result=tmp_path / "result.json",
        exchange=exchange,
    )
    if outcome == "DONE":
        result = probe.nixl_probe(args)
        assert result["receipt"]["payload_verified"] is True
        assert result["receipt"]["bytes"] == 4096
        assert (
            result["receipt"]["direct_gpu_zero_copy"]
            == "NOT_ESTABLISHED_HOST_STAGING_POSSIBLE"
        )
    else:
        with pytest.raises((probe.Unknown, RuntimeError)):
            probe.nixl_probe(args)
        assert not (exchange / "verified.json").exists()
    assert events == ["register", "release", "remove", "deregister"]
