# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU/port leases: atomicity across processes, staleness, and the launch
path refusing GPUs another agent holds."""

import multiprocessing as mp
import os
import stat
import subprocess
import sys
import time

import pytest

from atom.agent import deploy, hardware, leases
from atom.agent.config import AgentConfig
from atom.agent.leases import LeaseConflict, LeaseManager


def _race(lease_dir, i, start_at, results, done):
    while time.time() < start_at:
        pass
    try:
        LeaseManager(lease_dir).acquire(f"agent-{i}", [0, 1], None, "docker")
        results.put(i)
    except LeaseConflict:
        results.put(None)
    # A pending lease belongs to a live agent; an agent that exits abandons
    # it. Stay alive until everyone has tried.
    done.wait(30)


def test_only_one_of_many_processes_wins(tmp_path):
    ctx = mp.get_context("fork")
    results = ctx.Queue()
    done = ctx.Event()
    start_at = time.time() + 0.5
    procs = [
        ctx.Process(target=_race, args=(tmp_path, i, start_at, results, done))
        for i in range(8)
    ]
    for p in procs:
        p.start()
    winners = [results.get(timeout=30) for _ in procs]
    done.set()
    for p in procs:
        p.join()
    assert len([w for w in winners if w is not None]) == 1


def test_conflicts_name_the_holder_and_release_frees(tmp_path):
    lm = LeaseManager(tmp_path)
    lm.acquire("a", [0, 1], 8100, "local", label="model-a (d1)", check_port=False)
    with pytest.raises(LeaseConflict) as e:
        lm.acquire("b", [1, 2], None, "local")
    assert e.value.holders[0]["gpus"] == [1]
    assert e.value.holders[0]["label"] == "model-a (d1)"
    with pytest.raises(LeaseConflict):
        lm.acquire("c", [5], 8100, "local", check_port=False)  # same port
    assert lm.reserved() == ({0, 1}, {8100})
    assert lm.release("a")
    lm.acquire("b", [1, 2], None, "local")


def test_reacquiring_own_lease_is_idempotent(tmp_path):
    lm = LeaseManager(tmp_path)
    lm.acquire("a", [0], None, "local")
    lm.acquire("a", [0], None, "local")
    assert len(lm.held()) == 1


def test_stale_leases_are_reclaimed(tmp_path, monkeypatch):
    lm = LeaseManager(tmp_path)
    # Bound to a server that has exited.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lm.acquire("bound", [0], None, "local")
    lm.bind("bound", pid=dead.pid)
    # Pending (no server yet) past its TTL.
    lm.acquire("pending", [1], None, "local")
    monkeypatch.setattr(leases, "PENDING_TTL_S", -1)
    lm.acquire("new", [0, 1], None, "local")
    lm.bind("new", pid=os.getpid())  # bound to a live process: not stale
    assert [x.id for x in lm.held()] == ["new"]


def test_live_bound_lease_survives(tmp_path):
    lm = LeaseManager(tmp_path)
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lm.acquire("srv", [3], None, "local")
        lm.bind("srv", pid=server.pid)
        with pytest.raises(LeaseConflict):
            lm.acquire("other", [3], None, "local")
    finally:
        server.kill()
        server.wait()
    lm.acquire("other", [3], None, "local")  # server gone -> lease stale


def test_files_are_shareable_between_users(tmp_path):
    d = tmp_path / "shared"
    LeaseManager(d).acquire("a", [0], None, "local")
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o1777
    assert stat.S_IMODE(os.stat(d / "leases.json").st_mode) == 0o666
    assert stat.S_IMODE(os.stat(d / ".lock").st_mode) == 0o666


def test_gpu_busy_outside_any_lease_blocks(tmp_path):
    lm = LeaseManager(tmp_path)
    with pytest.raises(LeaseConflict) as e:
        lm.acquire(
            "a", [0, 1], None, "local", gpu_busy=lambda i: "bob pid 7" if i else None
        )
    assert e.value.holders == [{"gpus": [1], "held_by": "bob pid 7"}]


def test_two_agents_cannot_launch_on_the_same_gpu(tmp_path, monkeypatch):
    """Two users' agents (separate state dirs, one lease dir) race for GPU 0."""
    monkeypatch.setattr(hardware, "list_gpus", lambda with_names=True: [])
    # A long-lived stand-in for the server (the real module needs torch).
    monkeypatch.setattr(deploy, "SERVER_MODULE", "http.server")
    shared = tmp_path / "leases"
    agents = [
        AgentConfig(state_dir=tmp_path / f"agent{i}", docs_root=None,
                    runtime="local", lease_dir=shared)
        for i in range(2)
    ]  # fmt: skip
    first = deploy.launch(
        agents[0],
        deploy.DeploymentStore(agents[0].state_dir),
        model="m",
        gpus=[0],
        port=18990,
        args=["18990", "--bind", "127.0.0.1"],
    )
    try:
        store2 = deploy.DeploymentStore(agents[1].state_dir)
        with pytest.raises(LeaseConflict):
            deploy.launch(agents[1], store2, model="m", gpus=[0], port=18991, args=[])
        assert store2.all() == []  # no ghost record for the refused launch
    finally:
        deploy.stop(
            deploy.DeploymentStore(agents[0].state_dir),
            first,
            grace_s=5,
            leases=LeaseManager(shared),
        )
    assert LeaseManager(shared).held() == []


def test_inspect_system_reports_leases_and_excludes_leased_gpus(tmp_path, monkeypatch):
    from atom.agent.tools import ToolBox, ToolContext

    gpus = [
        hardware.Gpu(i, "gfx950", "MI355X", "", "", 288 * 1024, 300, 0)
        for i in range(4)
    ]
    monkeypatch.setattr(hardware, "list_gpus", lambda with_names=True: gpus)
    monkeypatch.setattr(hardware, "_docker_info", lambda: {"available": False})
    cfg = AgentConfig(
        state_dir=tmp_path / "s",
        docs_root=None,
        runtime="local",
        lease_dir=tmp_path / "l",
    )
    other = LeaseManager(cfg.lease_dir)
    other.acquire("bob:job", [2, 3], 9000, "local", label="bob's job", check_port=False)
    other.bind("bob:job", pid=os.getpid())
    report = ToolBox(ToolContext(cfg)).call("inspect_system", {})
    assert report["free_gpus"] == [0, 1]
    assert report["leases"][0]["label"] == "bob's job"
    assert report["leases"][0]["ours"] is False
    assert report["gpus"][2]["leased_by"].endswith("bob's job")
