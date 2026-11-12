# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon's per-GPU Unix socket, lock and state file, and its server."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import pickle
import socket
import stat
import struct
import threading
import time
from typing import Any

import torch

from atom.model_loader.weight_cache.values import _TAG, _digest, logger
from atom.utils import envs

PROTOCOL_VERSION = 3
# Set by a daemon's API server for its runners (preload.record_launch).
_DAEMON_PID_ENV = "ATOM_WEIGHT_CACHE_DAEMON_PID"  # its main process
_DAEMON_LOG_ENV = "ATOM_WEIGHT_CACHE_DAEMON_LOG"  # a daemon ATOM_PRELOAD started


def socket_dir() -> str:
    d = envs.ATOM_WEIGHT_CACHE_DIR or f"/dev/shm/atom_weight_cache_{os.getuid()}"
    os.makedirs(d, mode=0o700, exist_ok=True)
    st = os.lstat(d)
    if stat.S_ISLNK(st.st_mode) or st.st_uid != os.getuid():
        raise PermissionError(f"{d} is a symlink or not owned by uid {os.getuid()}")
    os.chmod(d, 0o700)
    return d


def gpu_identity(device: torch.device) -> str:
    p = torch.cuda.get_device_properties(device)
    pci = f"{getattr(p, 'pci_domain_id', 0):04x}:{getattr(p, 'pci_bus_id', 0):02x}:{getattr(p, 'pci_device_id', 0):02x}"
    return f"{getattr(p, 'uuid', '')}|{pci}"


def socket_path(device: torch.device) -> str:
    gid = hashlib.sha256(gpu_identity(device).encode()).hexdigest()[:16]
    return os.path.join(socket_dir(), f"gpu_{gid}.sock")


def _send(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("<Q", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("weight cache peer closed the connection")
        buf += chunk
    return bytes(buf)


def _recv(sock: socket.socket) -> Any:
    (n,) = struct.unpack("<Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, n))


def _check_peer(sock: socket.socket) -> None:
    cred = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _, uid, _ = struct.unpack("3i", cred)
    if uid != os.getuid():
        raise PermissionError(f"weight cache peer uid {uid} != {os.getuid()}")


def _ping(path: str, timeout: float = 10) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        _check_peer(sock)
        _send(sock, {"cmd": "ping"})
        return _recv(sock)


def daemon_alive(path: str) -> bool:
    """Whether a daemon (loading or serving) holds this GPU's lock."""
    try:
        fd = os.open(path + ".lock", os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES):
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def daemon_state(path: str) -> dict | None:
    try:
        with open(path + ".state") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class _DaemonLock:
    """Held for the daemon process's lifetime, from before loading starts."""

    def __init__(self, path: str, **info):
        self.path = path
        self.fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() > deadline:  # clients probe with brief shared locks
                    os.close(self.fd)
                    raise RuntimeError(
                        f"another weight cache daemon holds this GPU ({path}.lock)"
                    ) from None
                time.sleep(0.2)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)  # stale socket from a dead daemon
        # Read by `status` and by servers that stop this daemon.
        self.info = {
            "pid": os.getpid(),
            # The launch's main process: SIGTERM there stops the whole daemon.
            "daemon_pid": int(os.environ.get(_DAEMON_PID_ENV) or 0) or None,
            "log": os.environ.get(_DAEMON_LOG_ENV),
            "started": time.time(),
            **info,
        }
        self.set_state("loading")

    def set_state(self, state: str, **info) -> None:
        self.info.update(info, state=state, since=time.time())
        tmp = f"{self.path}.state.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.info, f)
        os.replace(tmp, self.path + ".state")

    def release(self) -> None:
        for p in (self.path, self.path + ".state"):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(p)
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)


class _Server:
    def __init__(
        self,
        path: str,
        payload: dict,
        views: dict,
        label: str,
        idle_timeout_s: int = 0,
        on_idle=None,
    ):
        self.path, self.payload, self.label = path, payload, label
        self.views = views
        t0 = time.perf_counter()
        self.digests = {k: _digest(v) for k, v in views.items()}
        logger.info(
            f"{_TAG} {label}: digested {len(views)} shared storages in {time.perf_counter() - t0:.2f}s"
        )
        self.poisoned: list[int] = []
        self._verify_lock = threading.Lock()
        # Connected clients: a server keeps its connection for its lifetime.
        self.clients = 0
        self.idle_since = time.monotonic()
        self._clients_lock = threading.Lock()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        os.chmod(path, 0o600)
        self.sock.listen(64)
        threading.Thread(
            target=self._accept, name="weight-cache-server", daemon=True
        ).start()
        if idle_timeout_s > 0 and on_idle is not None:
            threading.Thread(
                target=self._idle_watch,
                args=(idle_timeout_s, on_idle),
                name="weight-cache-idle",
                daemon=True,
            ).start()

    def _idle_watch(self, timeout_s: int, on_idle) -> None:
        while True:
            time.sleep(min(30.0, timeout_s / 4))
            with self._clients_lock:
                idle_for = (
                    time.monotonic() - self.idle_since if self.clients == 0 else 0.0
                )
            if idle_for >= timeout_s:
                logger.info(
                    f"{_TAG} {self.label}: no server connected for {idle_for:.0f}s "
                    f"(ATOM_PRELOAD_IDLE_TIMEOUT_S={timeout_s}); stopping the daemon"
                )
                on_idle()
                return

    def _client(self, delta: int) -> None:
        with self._clients_lock:
            self.clients += delta
            if self.clients == 0:
                self.idle_since = time.monotonic()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()

    def _verify(self) -> list[int]:
        """Storages whose bytes changed since export (a client wrote into them)."""
        with self._verify_lock:
            if not self.poisoned:
                self.poisoned = [
                    k for k, v in self.views.items() if _digest(v) != self.digests[k]
                ]
            return self.poisoned

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        peer = "?"
        try:
            _check_peer(conn)
            req = _recv(conn)
            peer = req.get("who", "?")
            if req.get("cmd") == "ping":
                with self._clients_lock:
                    clients = self.clients
                    idle_s = time.monotonic() - self.idle_since if clients == 0 else 0.0
                _send(
                    conn,
                    {
                        "summary": self.payload["summary"],
                        "poisoned": len(self.poisoned),
                        "clients": clients,
                        "idle_s": idle_s,
                    },
                )
                return
            if req.get("cmd") != "get":
                _send(conn, {"error": f"unknown command {req.get('cmd')!r}"})
                return
            changed = self._verify()
            if changed:
                logger.error(
                    f"{_TAG} {self.label}: {len(changed)} shared storages changed since export "
                    "(something wrote into the cached weights); refusing to serve them"
                )
                _send(
                    conn,
                    {
                        "error": f"{len(changed)} of the daemon's shared storages changed since export; restart the daemon"
                    },
                )
                return
            self._client(+1)
            try:
                _send(conn, self.payload)
                logger.info(f"{_TAG} {self.label}: served {peer}")
                # The connection stays open while the client lives, so each side
                # sees the other go away.
                with contextlib.suppress(OSError):
                    while conn.recv(1 << 12):
                        pass
            finally:
                self._client(-1)
            logger.info(f"{_TAG} {self.label}: {peer} disconnected")
        except Exception as e:  # noqa: BLE001 - one bad client must not stop serving
            logger.warning(f"{_TAG} {self.label}: request from {peer} failed: {e}")
        finally:
            conn.close()
