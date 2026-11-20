# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon's per-GPU Unix socket, lock and state file, and its server."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import glob
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

from atom.model_loader.weight_cache.values import (
    _TAG,
    _checksums,
    logger,
    release_checksum_buffers,
)

_MAX_MESSAGE = 1 << 30
# Sockets, locks and daemon logs. One place whatever shell a server or `stop`
# runs in, and one that outlives a login: not $XDG_RUNTIME_DIR, which some
# shells lack and which goes at logout while the daemon stays. Not /dev/shm:
# containers started with --ipc=host share it (and are all root), and IPC
# handles do not cross containers anyway.
_DIR = f"/tmp/atom_weight_cache_{os.getuid()}"


def socket_dir() -> str:
    d = _DIR
    os.makedirs(d, mode=0o700, exist_ok=True)
    st = os.lstat(d)
    if stat.S_ISLNK(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise PermissionError(
            f"{d} must be a directory of uid {os.getuid()} that only it can access (mode 700)"
        )
    return d


def gpu_identity(device: torch.device) -> str:
    p = torch.cuda.get_device_properties(device)
    return f"{p.uuid}|{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}"


def socket_path(device: torch.device) -> str:
    gid = hashlib.sha256(gpu_identity(device).encode()).hexdigest()[:16]
    return os.path.join(socket_dir(), f"gpu_{gid}.sock")


def _send(sock: socket.socket, obj: Any) -> None:
    _send_bytes(sock, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _send_bytes(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack("<Q", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("weight cache peer closed the connection")
        buf += chunk
    return bytes(buf)


def _recv(sock: socket.socket, limit: int = _MAX_MESSAGE) -> Any:
    (n,) = struct.unpack("<Q", _recv_exact(sock, 8))
    if n > limit:
        raise ConnectionError(f"weight cache message of {n} bytes refused")
    return pickle.loads(_recv_exact(sock, n))


def _ping(path: str, timeout: float = 10) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        _send(sock, {"cmd": "ping"})
        return _recv(sock)


def running_daemons() -> list[str]:
    """The socket paths of the GPUs a daemon here holds."""
    locks = sorted(glob.glob(os.path.join(socket_dir(), "gpu_*.sock.lock")))
    return [p for p in (lock[: -len(".lock")] for lock in locks) if daemon_alive(p)]


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
        self.info = {"pid": os.getpid(), "started": time.time(), **info}
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
    """Serves the payload on the GPU's socket.

    A connected client could write into the shared storages, so they are
    checked against their export checksums before a hand-out: in the
    background once the last client has gone, on the spot if another one is
    still connected."""

    def __init__(self, path: str, payload: dict, views: dict, label: str):
        self.label = label
        self._payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.views = views
        t0 = time.perf_counter()
        self.checksums = dict(zip(views, _checksums(views.values())))
        self._release_checksum_memory()
        logger.info(
            f"{_TAG} {label}: checksummed {len(views)} shared storages in {time.perf_counter() - t0:.2f}s"
        )
        self.poisoned: list[int] = []
        self._check_lock = threading.Lock()
        self._lock = threading.Lock()  # guards the fields below
        self.clients = 0  # connected servers (each keeps its connection)
        self.pending = 0  # requests being answered
        self.connects = 0  # connections so far
        self.checked = 0  # connections the last clean check covered
        self.stopping = False
        self.held = False  # for whoever is about to stop this daemon (_hold)
        self.idle_since = time.monotonic()
        # Listening before it appears under its name: a client that sees the
        # path can connect.
        tmp = f"{path}.{os.getpid()}.tmp"
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(tmp)
        os.chmod(tmp, 0o600)
        self.sock.listen(64)
        os.replace(tmp, path)
        threading.Thread(
            target=self._accept, name="weight-cache-server", daemon=True
        ).start()

    def _idle_for(self) -> float:
        busy = self.pending or self.held  # a connected server's request is pending
        return 0.0 if busy else time.monotonic() - self.idle_since

    def watch_idle(self, timeout_s: int, on_idle, all_serving) -> None:
        """Call ``on_idle`` after ``timeout_s`` without a server, counted from
        when every rank of the daemon serves (``all_serving``, an Event):
        servers connect only then."""

        def watch():
            all_serving.wait()
            with self._lock:
                self.idle_since = time.monotonic()
            while True:
                time.sleep(min(30.0, timeout_s / 4))
                with self._lock:
                    if self.stopping:
                        return  # being stopped already
                    idle_for = self._idle_for()
                    if idle_for >= timeout_s:
                        self.stopping = True  # no request is answered from here on
                        break
            logger.info(
                f"{_TAG} {self.label}: no server connected for {idle_for:.0f}s; "
                "stopping the daemon"
            )
            on_idle()

        threading.Thread(target=watch, name="weight-cache-idle", daemon=True).start()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()

    def _release_checksum_memory(self) -> None:
        release_checksum_buffers()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    def _check(self) -> list[int]:
        """Storages whose bytes changed since export (a client wrote into them)."""
        with self._check_lock:
            if self.poisoned:
                return self.poisoned
            with self._lock:
                if self.checked == self.connects:
                    return []  # nobody connected since the last clean check
                connects, connected = self.connects, self.clients
            device = next(iter(self.views.values())).device if self.views else None
            on_gpu = device is not None and device.type == "cuda"
            try:
                with torch.cuda.device(device) if on_gpu else contextlib.nullcontext():
                    now = _checksums(self.views.values())
            finally:
                self._release_checksum_memory()
            changed = [k for k, d in zip(self.views, now) if d != self.checksums[k]]
            if changed:
                self.poisoned = changed
            elif connected == 0:
                # Clean as of now for the clients that had gone.
                self.checked = connects
            return changed

    def _connected(self, delta: int) -> None:
        with self._lock:
            self.clients += delta
            if delta > 0:
                self.connects += 1
            last_gone = delta < 0 and self.clients == 0
        if last_gone:
            threading.Thread(
                target=self._check, name="weight-cache-check", daemon=True
            ).start()

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
            conn.settimeout(30)  # a request is small and sent at once
            req = _recv(conn, limit=1 << 20)
            conn.settimeout(None)
            peer = req.get("who", "?")
            if req.get("cmd") == "ping":
                with self._lock:
                    # Connected servers and those still being served.
                    clients, idle_s = self.pending, self._idle_for()
                _send(
                    conn,
                    {
                        "poisoned": len(self.poisoned),
                        "clients": clients,
                        "idle_s": idle_s,
                    },
                )
                return
            if req.get("cmd") == "hold":
                self._hold(conn)
                return
            if req.get("cmd") != "get":
                _send(conn, {"error": f"unknown command {req.get('cmd')!r}"})
                return
            with self._lock:
                stopping = self.stopping or self.held
                if not stopping:
                    self.pending += 1
            if stopping:
                _send(conn, {"error": "the daemon is shutting down"})
                return
            try:
                self._serve(conn, peer)
            finally:
                with self._lock:
                    self.pending -= 1
                    if not self.pending:
                        self.idle_since = time.monotonic()
        except Exception as e:  # noqa: BLE001 - one bad client must not stop serving
            logger.warning(f"{_TAG} {self.label}: request from {peer} failed: {e}")
        finally:
            conn.close()

    def _hold(self, conn: socket.socket) -> None:
        """For whoever is about to stop this daemon: if no server has its
        weights now, none gets them while this connection stays open. Found
        and refused under one lock, so no server slips in between."""
        with self._lock:
            held = not (self.pending or self.stopping or self.held)
            self.held = self.held or held
            clients = self.pending
        if not held:
            _send(conn, {"held": False, "clients": clients})
            return
        try:
            _send(conn, {"held": True, "clients": clients})  # the holder may be gone
            with contextlib.suppress(OSError):
                while conn.recv(1 << 12):
                    pass
        finally:  # gone without stopping the daemon: servers may come again
            with self._lock:
                self.held = False
                self.idle_since = time.monotonic()

    def _serve(self, conn: socket.socket, peer: str) -> None:
        changed = self._check()
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
        self._connected(+1)
        try:
            _send_bytes(conn, self._payload_bytes)
            logger.info(f"{_TAG} {self.label}: served {peer}")
            # The connection stays open while the client lives, so each side
            # sees the other go away.
            with contextlib.suppress(OSError):
                while conn.recv(1 << 12):
                    pass
        finally:
            self._connected(-1)
        logger.info(f"{_TAG} {self.label}: {peer} disconnected")
