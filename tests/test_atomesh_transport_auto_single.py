"""CPU contracts for one fresh-allocation, sysfs-selected NIXL READ attempt."""

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_atomesh_transport_nixl_read_only import report, setup_probe


def auto_probe(tmp_path, monkeypatch, fault=None):
    probe, env, _, inv, calls = setup_probe(tmp_path, monkeypatch)
    env.pop("ATOMESH_TRANSPORT_UCX_SELECTION")
    env["ATOMESH_TRANSPORT_UCX_AUTO_SINGLE"] = "1"
    env["ATOMESH_RUN_TOKEN"] = "fixture-run-token"
    env["ATOMESH_TRANSPORT_MORI_BYTES"] = "not-parsed"
    second = copy.deepcopy(inv["ports"][0])
    second["device"] = "fixture_z"
    inv["ports"].append(second)
    root = tmp_path / "transport-diagnostic/benchmark"
    publish = probe.publish_once
    startup_peer = None

    def publish_with_peer(path, doc):
        nonlocal startup_peer
        publish(path, doc)
        if path.name == "auto-startup-entry-0.json":
            startup_peer = {
                **doc,
                "rank": 1,
                "hostname": "peer-fixture",
                "node_ip": "192.0.2.2",
                "nonce": "a" * 32,
            }
            if fault == "startup_timeout":
                return
            if fault == "startup_stale":
                startup_peer["created_at"] -= 1000
            startup_mismatches = {
                "startup_cross_job": ("job_id", "old-job"),
                "startup_token": ("run_token", "old-token"),
                "startup_phase": ("phase", "eval"),
                "startup_image": ("image", "other-image"),
                "startup_script": ("script_sha256", "other-script"),
                "startup_rank": ("rank", 0),
                "startup_host": ("hostname", "local-fixture"),
                "startup_ip": ("node_ip", "192.0.2.99"),
            }
            if fault in startup_mismatches:
                key, value = startup_mismatches[fault]
                startup_peer[key] = value
            kind = "error" if fault == "startup_peer_error" else "entry"
            publish(root / f"auto-startup-{kind}-1.json", startup_peer)
        elif path.name == "auto-startup-ack-0.json" and startup_peer:
            agreement = {
                "0": probe.evidence_digest(
                    json.loads((root / "auto-startup-entry-0.json").read_text())
                ),
                "1": probe.evidence_digest(startup_peer),
            }
            if fault == "startup_old_ack":
                agreement["0"] = "old-entry"
            publish(
                root / "auto-startup-ack-1.json",
                {**startup_peer, "agreement": agreement},
            )

    monkeypatch.setattr(probe, "publish_once", publish_with_peer)
    if fault == "startup_timeout":
        monkeypatch.setattr(probe, "STARTUP_SECONDS", 0.01)

    def peer(path, timeout=45):
        assert timeout == 45
        local = json.loads(Path(str(path).replace("-1.json", "-0.json")).read_text())
        remote = copy.deepcopy(local)
        remote.update(
            rank=1, hostname="peer-fixture", node_ip="192.0.2.2", nonce="b" * 32
        )
        if path.name == "auto-inventory-1.json":
            remote["inventory"]["ports"][0]["gids"][0]["gid"] = "::ffff:192.0.2.2"
            if fault == "no_candidate":
                remote["inventory"]["ports"] = []
            elif fault in (
                "inactive",
                "zero_gid",
                "roce_v1",
                "no_netdev",
                "not_ethernet",
            ):
                for port in remote["inventory"]["ports"]:
                    if fault == "inactive":
                        port["state"] = "1: DOWN"
                    elif fault == "not_ethernet":
                        port["link_layer"] = "InfiniBand"
                    else:
                        key, value = {
                            "zero_gid": ("gid", "::"),
                            "roce_v1": ("type", "RoCE v1"),
                            "no_netdev": ("ndev", ""),
                        }[fault]
                        port["gids"][0][key] = value
            elif fault == "stale":
                remote["created_at"] -= 100
            elif fault == "cross_job":
                remote["job_id"] = "other-job"
            elif fault == "mode":
                remote["auto_single"] = "0"
            elif fault == "run_token":
                remote["run_token"] = "old-run"
            elif fault == "phase":
                remote["phase"] = "eval"
            elif fault == "node_ip":
                remote["node_ip"] = "192.0.2.99"
            elif fault == "image":
                remote["image"] = "other-image"
            elif fault == "script":
                remote["script_sha256"] = "different"
            elif fault == "missing_peer":
                raise probe.Unknown("peer missing within45s")
        elif path.name == "auto-selection-1.json" and fault == "old_nonce":
            remote["nonce"] = "c" * 32
        elif path.name == "auto-ready-1.json" and fault == "peer_failure":
            remote["eligible"] = False
        return remote

    monkeypatch.setattr(probe, "wait_json", peer)
    entry = {
        **{k: v for k, v in inv["ports"][0].items() if k != "gids"},
        "gid": inv["ports"][0]["gids"][0],
    }
    monkeypatch.setattr(
        probe,
        "current_selection",
        lambda _: {**entry, "state": "1: DOWN"} if fault == "changed" else entry,
    )
    original = probe.supervise

    def supervise(argv, log, timeout, child_env):
        result = original(argv, log, timeout, child_env)
        if (len(calls) == 2 and fault == "selected_failure") or (
            len(calls) == 1 and fault == "original_failure"
        ):
            probe.write_json(argv[argv.index("--result") + 1], {"status": "FAIL"})
        return result

    monkeypatch.setattr(probe, "supervise", supervise)
    return probe, env, inv, calls, root


@pytest.mark.parametrize(
    "fault,classification",
    [
        ("startup_timeout", "STARTUP_TIMEOUT"),
        ("startup_stale", "STARTUP_INVALID"),
        ("startup_cross_job", "STARTUP_INVALID"),
        ("startup_token", "STARTUP_INVALID"),
        ("startup_phase", "STARTUP_INVALID"),
        ("startup_image", "STARTUP_INVALID"),
        ("startup_script", "STARTUP_INVALID"),
        ("startup_rank", "STARTUP_INVALID"),
        ("startup_host", "STARTUP_INVALID"),
        ("startup_ip", "STARTUP_INVALID"),
        ("startup_peer_error", "STARTUP_PEER_FAILED"),
        ("startup_old_ack", "STARTUP_INVALID"),
        ("startup_conflict", "STARTUP_INVALID"),
        ("startup_workload_error", "STARTUP_PEER_FAILED"),
    ],
)
def test_startup_failure_never_collects_inventory_or_launches_transport(
    tmp_path, monkeypatch, fault, classification
):
    probe, env, _, calls, root = auto_probe(tmp_path, monkeypatch, fault)
    if fault == "startup_conflict":
        root.mkdir(parents=True)
        (root / "auto-startup-entry-0.json").write_text('{"old":true}')
    if fault == "startup_workload_error":
        (tmp_path / "workload-failure.json").write_text(
            json.dumps(
                {
                    "job_id": env["SLURM_JOB_ID"],
                    "run_token": env["ATOMESH_RUN_TOKEN"],
                    "num_ranks": 2,
                    "return_code": 2,
                }
            )
        )
    inventories = []
    monkeypatch.setattr(probe, "inventory", lambda: inventories.append(True))
    assert probe.main() == 2
    assert inventories == []
    assert calls == []
    assert not list(root.glob("auto-inventory-*.json"))
    summary = report(tmp_path)
    assert summary["stages"]["startup-rendezvous"]["classification"] == classification
    assert summary["stages"]["nixl-rdma-gpu-read"]["classification"] == "NOT_TESTED"
    assert summary["stages"]["mori-register"]["classification"] == "NOT_TESTED"
    if fault == "startup_conflict":
        assert (root / "auto-startup-entry-0.json").read_text() == '{"old":true}'


@pytest.mark.parametrize("mode", [None, "1"])
def test_default_and_explicit_modes_do_not_enter_startup_rendezvous(
    tmp_path, monkeypatch, mode
):
    probe, _, _, _, calls = setup_probe(tmp_path, monkeypatch, mode=mode)
    assert probe.main() == 0
    assert calls
    assert not list(tmp_path.rglob("auto-startup-*.json"))
    assert "startup-rendezvous" not in report(tmp_path)["stages"]


def test_startup_keeps_one_total_alarm_and_unchanged_fresh_inventory_wait(
    tmp_path, monkeypatch
):
    probe, _, _, _, _ = auto_probe(tmp_path, monkeypatch)
    alarms = []
    monkeypatch.setattr(probe.signal, "alarm", alarms.append)
    assert probe.main() == 0
    assert alarms == [600, 0]
    assert report(tmp_path)["stages"]["startup-rendezvous"]["timeout_seconds"] == 180
    # auto_probe's peer exchange asserts the existing 45-second wait.


@pytest.mark.parametrize("fault", [None, "original_failure"])
def test_auto_single_selects_once_and_performs_only_one_read(
    tmp_path, monkeypatch, fault
):
    probe, _, _, calls, root = auto_probe(tmp_path, monkeypatch, fault)
    assert probe.main() == 0
    assert [kind for kind, _, _ in calls] == [
        "nixl-create",
        "nixl-create",
        "nixl-transfer",
    ]
    assert "UCX_NET_DEVICES" not in calls[0][2]
    assert calls[1][2]["UCX_NET_DEVICES"] == "fixture_rdma:1"
    assert calls[1][2]["UCX_IB_GID_INDEX"] == "3"
    assert calls[2][2]["UCX_TLS"] == "rc,rocm_copy,rocm_ipc"
    assert all(argv[argv.index("--size") + 1] == "4096" for _, argv, _ in calls)
    evidence = json.loads((root / "rank-0/auto-selection.json").read_text())
    assert evidence["selection"] == "fixture_rdma:1@3"
    assert evidence["peer_entry"]["gid"]["gid"] == "::ffff:192.0.2.2"
    assert report(tmp_path)["stages"]["mori-register"]["classification"] == "NOT_TESTED"


@pytest.mark.parametrize(
    "fault,creates",
    [
        ("no_candidate", 1),
        ("inactive", 1),
        ("zero_gid", 1),
        ("roce_v1", 1),
        ("no_netdev", 1),
        ("not_ethernet", 1),
        ("stale", 0),
        ("cross_job", 0),
        ("mode", 0),
        ("run_token", 0),
        ("phase", 0),
        ("node_ip", 0),
        ("image", 0),
        ("script", 0),
        ("missing_peer", 0),
        ("old_nonce", 1),
        ("changed", 1),
        ("selected_failure", 2),
        ("peer_failure", 2),
    ],
)
def test_auto_single_failure_never_retries_another_tuple_or_reads(
    tmp_path, monkeypatch, fault, creates
):
    probe, _, _, calls, _ = auto_probe(tmp_path, monkeypatch, fault)
    assert probe.main() == 2
    assert [kind for kind, _, _ in calls] == ["nixl-create"] * creates
    summary = report(tmp_path)
    assert summary["collection"] == "UNKNOWN"
    assert summary["stages"]["nixl-rdma-gpu-read"]["classification"] == "NOT_TESTED"
    assert summary["stages"]["mori-register"]["classification"] == "NOT_TESTED"


@pytest.mark.parametrize(
    "changes",
    [
        {"ATOMESH_TRANSPORT_UCX_AUTO_SINGLE": "true"},
        {"ATOMESH_TRANSPORT_GPU": "0"},
        {"ATOMESH_RUN_TOKEN": ""},
        {"ATOMESH_RUN_TOKEN": " "},
        {"ATOMESH_TRANSPORT_NIXL_READ_ONLY": "0"},
        {"ATOMESH_TRANSPORT_UCX_SELECTION": "fixture_rdma:1@3"},
    ],
)
def test_auto_single_rejects_conflicting_or_missing_opt_ins(
    tmp_path, monkeypatch, changes
):
    probe, env, _, _, calls = setup_probe(tmp_path, monkeypatch)
    env.pop("ATOMESH_TRANSPORT_UCX_SELECTION")
    env["ATOMESH_TRANSPORT_UCX_AUTO_SINGLE"] = "1"
    env["ATOMESH_RUN_TOKEN"] = "fixture-run-token"
    env.update(changes)
    with pytest.raises(ValueError):
        probe.main()
    assert calls == []


def test_conflicting_inventory_is_not_overwritten(tmp_path, monkeypatch):
    probe, _, _, calls, root = auto_probe(tmp_path, monkeypatch)
    root.mkdir(parents=True)
    path = root / "auto-inventory-0.json"
    path.write_text('{"old_run": true}\n')
    assert probe.main() == 2
    assert path.read_text() == '{"old_run": true}\n'
    assert calls == []


def test_optional_absolute_ucx_path_absence_is_recorded_not_required(
    tmp_path, monkeypatch
):
    probe, _, _, _, _ = auto_probe(tmp_path, monkeypatch)
    # Restore actual public inventory entry point, with external commands mocked.
    fresh_spec = importlib.util.spec_from_file_location(
        "inventory_auto", probe.__file__
    )
    fresh = importlib.util.module_from_spec(fresh_spec)
    fresh_spec.loader.exec_module(fresh)

    def missing(argv, **kwargs):
        assert kwargs["timeout"] == 6
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(fresh.subprocess, "run", missing)
    monkeypatch.setattr(Path, "glob", lambda *_: [])
    result = fresh.inventory()
    absolute = [
        c for c in result["commands"] if c["argv"][0] == "/usr/local/ucx/bin/ucx_info"
    ]
    assert [c["argv"][1] for c in absolute] == ["-v", "-d"]
    assert all(c["status"] == "UNKNOWN" for c in absolute)


@pytest.mark.parametrize("fault", [None, "metadata", "receipt"])
def test_read_session_rejects_stale_metadata_and_receipt(tmp_path, monkeypatch, fault):
    probe, env, _, _, _ = setup_probe(tmp_path, monkeypatch, rank=0)
    session = {
        "run_token": "fresh",
        "hosts": {"0": "local-fixture", "1": "peer-fixture"},
        "allocation_ips": ["192.0.2.1", "192.0.2.2"],
    }
    env["ATOMESH_TRANSPORT_READ_SESSION"] = json.dumps(session)
    events = []

    class Tensor:
        def to(self, _):
            return self

        def data_ptr(self):
            return 100

        def get_device(self):
            return 0

    class Agent:
        backends = ("UCX",)

        def __init__(self, *_):
            pass

        def get_plugin_list(self):
            return ["UCX"]

        def get_backend_params(self, _):
            return {}

        def register_memory(self, *args, **kwargs):
            return "registration"

        def get_agent_metadata(self):
            return b"metadata"

        def add_remote_agent(self, _):
            events.append("remote")
            return "peer"

        def remove_remote_agent(self, _):
            pass

        def deregister_memory(self, *args, **kwargs):
            events.append("cleanup")

    monkeypatch.setattr(probe, "load_nixl", lambda: (Agent, lambda **kwargs: kwargs))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(hip="fixture"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                set_device=lambda _: None,
                synchronize=lambda: None,
            ),
            arange=lambda *a, **kw: Tensor(),
            int32="int32",
            uint8="uint8",
        ),
    )

    def peer(path):
        if path.name.startswith("metadata"):
            return {
                "session": (
                    {**session, "run_token": "old"} if fault == "metadata" else session
                ),
                "hostname": "peer-fixture",
                "rank": 1,
                "node_ip": "192.0.2.2",
                "bytes": 4096,
                "metadata": "bWV0YWRhdGE=",
            }
        return {
            "session": (
                {**session, "run_token": "old"} if fault == "receipt" else session
            ),
            "status": "PASS",
            "source": "local-fixture",
            "destination": "peer-fixture",
            "bytes": 4096,
            "operation": "READ",
            "payload_verified": True,
        }

    monkeypatch.setattr(probe, "wait_json", peer)
    args = argparse.Namespace(
        child="nixl-transfer",
        rank=0,
        result=tmp_path / "result.json",
        exchange=tmp_path / "exchange",
    )
    if fault:
        with pytest.raises(probe.Unknown, match="identity"):
            probe.nixl_probe(args)
    else:
        assert probe.nixl_probe(args)["status"] == "PASS"
    assert events[-1] == "cleanup"
    assert ("remote" in events) == (fault != "metadata")


@pytest.mark.parametrize("mismatch,delay", [(False, 0), (True, 0), (False, 0.6)])
def test_independent_rank_processes_agree_or_fail_closed(tmp_path, mismatch, delay):
    script = (
        Path(__file__).resolve().parents[1]
        / ".github/scripts/atomesh/pd_transport_probe.py"
    )
    driver = r"""
import sys, os, json, importlib.util, time
from pathlib import Path
spec=importlib.util.spec_from_file_location("probe", sys.argv[1])
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
rank=int(os.environ["NODE_RANK"])
p.socket.gethostname=lambda: f"fixture-{rank}"
def port(name):
 return {"device":name,"port":"1","state":"4: ACTIVE","link_layer":"Ethernet","gids":[{"index":"3","gid":f"::ffff:192.0.2.{rank+1}","type":"RoCE v2","ndev":"fixture_eth"}]}
ports=[port("fixture_a"),port("fixture_z")]
if rank: ports.reverse()
def inventory():
 root=Path(os.environ["RUN_DIR"])/"transport-diagnostic/benchmark"
 assert (root/"auto-startup-entry-0.json").exists()
 assert (root/"auto-startup-entry-1.json").exists()
 return {"ports":ports}
p.inventory=inventory
p.current_selection=lambda selection:{**{k:v for k,v in port(selection.split(":")[0]).items() if k!="gids"},"gid":port("fixture_a")["gids"][0]}
def supervised(argv, log, timeout, env):
 kind=argv[argv.index("--child")+1]
 with (Path(os.environ["RUN_DIR"])/f"calls-{rank}.jsonl").open("a") as f: f.write(json.dumps({"kind":kind,"env":{k:v for k,v in env.items() if k.startswith("UCX_") or k=="ATOMESH_TRANSPORT_READ_SESSION"}})+"\n")
 p.write_json(argv[argv.index("--result")+1],{"status":"PASS"})
 return 0,False
p.supervise=supervised
real_wait=p.wait_json
p.wait_json=lambda path,timeout=45:real_wait(path,timeout=min(timeout,0.3))
p.STARTUP_SECONDS=3
if rank: time.sleep(float(os.environ["PEER_DELAY"]))
sys.argv=[sys.argv[1]]
sys.exit(p.main())
"""
    processes = []
    for rank in (0, 1):
        env = {
            **os.environ,
            "NODE_RANK": str(rank),
            "PEER_DELAY": str(delay),
            "IPADDRS": "192.0.2.1,192.0.2.2",
            "RUN_DIR": str(tmp_path),
            "SLURM_JOB_ID": "fixture-job",
            "ATOMESH_RUN_TOKEN": "other" if mismatch and rank else "shared-token",
            "ATOMESH_VLLM_SOURCE_SHA": "b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
            "DOCKER_IMAGE": "image@sha256:659b28319fef4ea0e3d8f33e25b4c35d6f663f5d818a5b2fa06dceaf859234e4",
            "ATOMESH_TRANSPORT_UCX_AUTO_SINGLE": "1",
            "ATOMESH_TRANSPORT_NIXL_READ_ONLY": "1",
            "ATOMESH_TRANSPORT_GPU": "1",
        }
        env.pop("ATOMESH_TRANSPORT_UCX_SELECTION", None)
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", driver, str(script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == (2 if mismatch else 0), stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
    if mismatch:
        assert not list(tmp_path.glob("calls-*.jsonl"))
    else:
        sessions = []
        for rank in (0, 1):
            calls = [
                json.loads(line)
                for line in (tmp_path / f"calls-{rank}.jsonl").read_text().splitlines()
            ]
            assert [call["kind"] for call in calls] == [
                "nixl-create",
                "nixl-create",
                "nixl-transfer",
            ]
            assert calls[1]["env"]["UCX_NET_DEVICES"] == "fixture_a:1"
            sessions.append(
                json.loads(calls[-1]["env"]["ATOMESH_TRANSPORT_READ_SESSION"])
            )
        assert sessions[0] == sessions[1]
        assert len(set(sessions[0]["agreement"]["nonces"].values())) == 2
        root = tmp_path / "transport-diagnostic/benchmark"
        entries = [
            json.loads((root / f"auto-startup-entry-{rank}.json").read_text())
            for rank in (0, 1)
        ]
        for rank in (0, 1):
            inv = json.loads((root / f"auto-inventory-{rank}.json").read_text())
            assert inv["created_at"] >= max(entry["created_at"] for entry in entries)
            assert inv["nonce"] != entries[rank]["nonce"]
            summary = json.loads((root / f"rank-{rank}/summary.json").read_text())
            assert (
                summary["stages"]["startup-rendezvous"]["classification"]
                == "STARTUP_READY"
            )
        if delay:
            assert entries[1]["created_at"] - entries[0]["created_at"] >= 0.3
