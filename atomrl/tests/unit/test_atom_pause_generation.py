# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Pausing the ATOM rollout around a weight sync, for partial rollout."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from atomrl.engine.inference.atom_ray_server import ATOMRayServer, ATOMReplicaManager


class CoreMgr:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._fail = fail

    def broadcast_utility_command_sync(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self._fail is not None:
            raise self._fail
        return [{"cmd": cmd}]


class Engine:
    def __init__(self, core_mgr: CoreMgr, features=()) -> None:
        self.core_mgr = core_mgr
        self._features = set(features)

    def pause_generation(self, mode: str = "keep") -> None:
        self.core_mgr.broadcast_utility_command_sync("pause_generation", mode=mode)

    def resume_generation(self) -> None:
        self.core_mgr.broadcast_utility_command_sync("resume_generation")

    def get_capabilities(self):
        return Caps(self._features)

    def weight_sync_session(self):
        return self.core_mgr.weight_sync_session()


def _server(core_mgr: CoreMgr) -> ATOMRayServer:
    server = object.__new__(ATOMRayServer)
    server.engine = Engine(core_mgr)
    return server


def test_pause_and_resume_reach_every_engine_with_the_mode() -> None:
    core_mgr = CoreMgr()
    server = _server(core_mgr)

    asyncio.run(server.pause_generation("keep", True))
    asyncio.run(server.resume_generation())

    assert core_mgr.calls == [
        ("pause_generation", {"mode": "keep"}),
        ("resume_generation", {}),
    ]


def test_a_refused_pause_fails_the_sync() -> None:
    server = _server(CoreMgr(fail=RuntimeError("connector holds 2")))

    with pytest.raises(RuntimeError, match="connector holds 2"):
        asyncio.run(server.pause_generation())


class Caps:
    def __init__(self, features) -> None:
        self.features = set(features)

    def supports(self, feature: str) -> bool:
        return feature in self.features


@pytest.mark.parametrize(
    ("features", "expected"),
    [({"pause_generation"}, ["keep"]), (set(), [])],
    ids=["capability", "no-capability"],
)
def test_pause_modes_come_from_the_atom_build(features, expected):
    server = _server(CoreMgr())
    server.engine = Engine(server.engine.core_mgr, features)

    assert asyncio.run(server.generation_pause_modes()) == expected


def test_missing_capability_api_fails_loudly() -> None:
    server = _server(CoreMgr())
    server.engine = SimpleNamespace()

    with pytest.raises(AttributeError, match="get_capabilities"):
        asyncio.run(server.generation_pause_modes())


class Remote:
    def __init__(self, value=None) -> None:
        self.value = value
        self.args: list[tuple] = []

    def remote(self, *args):
        self.args.append(args)
        return self.value


def test_the_pool_pauses_in_the_modes_every_replica_takes(monkeypatch) -> None:
    import ray

    monkeypatch.setattr(ray, "get", lambda refs: refs)
    servers = [
        SimpleNamespace(
            generation_pause_modes=Remote(modes),
            pause_generation=Remote(True),
            resume_generation=Remote(True),
        )
        for modes in (["keep"], ["keep", "wait"])
    ]
    mgr = object.__new__(ATOMReplicaManager)
    mgr.servers = servers

    assert mgr.generation_pause_modes() == {"keep"}
    mgr.pause_all(mode="keep", clear_cache=True)
    mgr.resume_all()

    assert [s.pause_generation.args for s in servers] == [[("keep", True)]] * 2
    assert [s.resume_generation.args for s in servers] == [[()]] * 2


class SessionCoreMgr(CoreMgr):
    """A CoreManager whose session the buckets have to be sent inside."""

    def __init__(self) -> None:
        super().__init__()
        self.open = False

    def weight_sync_session(self):
        import contextlib

        @contextlib.contextmanager
        def session():
            self.open = True
            try:
                yield
            finally:
                self.open = False

        return session()


@pytest.mark.parametrize("use_shm", [False, True])
def test_a_bucketed_sync_is_sent_inside_the_engines_session(use_shm) -> None:
    """An ATOM since ROCm/ATOM#2297 refuses a non-final bucket outside it."""
    core_mgr = SessionCoreMgr()
    server = _server(core_mgr)
    seen = []
    server._update_weights_from_ipc_sync = lambda version: seen.append(core_mgr.open)
    server._update_weights_from_shm_sync = lambda version: seen.append(core_mgr.open)

    asyncio.run(server.update_weights_from_ipc(use_shm, 3))

    assert seen == [True] and not core_mgr.open


def test_missing_weight_sync_session_fails_loudly() -> None:
    server = _server(CoreMgr())
    seen = []
    server._update_weights_from_ipc_sync = lambda version: seen.append(version)

    with pytest.raises(AttributeError, match="weight_sync_session"):
        asyncio.run(server.update_weights_from_ipc(False, 3))
    assert seen == []
