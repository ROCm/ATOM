# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon's Unix socket (one, in its main process) and the server on it,
each GPU's lock and state file, and what holds a GPU's weights behind the
server."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import glob
import hashlib
import json
import os
import pickle
import queue
import socket
import stat
import struct
import threading
import time
from typing import Any

import torch

from atom.model_loader.weight_cache.key import _diff_key
from atom.model_loader.weight_cache.values import (
    _TAG,
    _checksums,
    logger,
    release_checksum_buffers,
)

_MAX_MESSAGE = 1 << 30
# The messages' version, in every request: a server and a daemon of different
# versions do not use each other.
PROTOCOL = 3
# A server answers `stop` once nothing of it runs on the weights: a forward
# step at most, and what its engines take to hear of it.
_NOTICE_TIMEOUT_S = 600
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


def gpu_path(identity: str) -> str:
    """A GPU's name here (``identity``: gpu_identity's): the daemon rank that
    holds it has its lock (.lock) and state (.state) under it. The state
    names the daemon's socket."""
    gid = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return os.path.join(socket_dir(), f"gpu_{gid}.sock")


def socket_path(device: torch.device) -> str:
    return gpu_path(gpu_identity(device))


def daemon_socket(pid: int) -> str:
    """The socket of the daemon whose main process is ``pid``."""
    return os.path.join(socket_dir(), f"daemon_{pid}.sock")


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


def _connect(path: str, timeout: float = 30) -> socket.socket:
    """A connection to the daemon whose rank holds the GPU at ``path``."""
    main = (daemon_state(path) or {}).get("socket")
    if main is None:
        raise FileNotFoundError(f"no daemon socket is named for {path}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(main)
    except BaseException:
        sock.close()
        raise
    return sock


def _ping(path: str, timeout: float = 10) -> dict:
    """What the daemon says of the GPU at ``path``."""
    with _connect(path, timeout) as sock:
        _send(sock, {"cmd": "ping", "v": PROTOCOL})
        reply = _recv(sock)
    gpu = os.path.basename(path)
    return {
        **reply,
        "clients": reply.get("clients", {}).get(gpu, 0),
        "poisoned": reply.get("poisoned", {}).get(gpu, 0),
    }


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


class _Holder:
    """One GPU's weights, in the rank process that loaded them: gives the
    payload to the daemon's main process, which serves it (_Server).

    A connected server could write into the shared storages, so they are
    checked against their export checksums before a hand-out: in the
    background once the last server has gone, on the spot if another one is
    still connected."""

    def __init__(self, conn, payload: dict, views: dict, label: str, lock=None):
        self.conn, self.label = conn, label
        self.lock = lock  # this GPU's (_DaemonLock): its state names the model
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
        self.clients = 0  # servers that have these weights now
        self.connects = 0  # hand-outs so far
        self.checked = 0  # hand-outs the last clean check covered

    def serve(self, stopped) -> None:
        """Answer the main process until ``stopped()``, or until it has gone."""
        while not stopped():
            try:
                if not self.conn.poll(1.0):
                    continue
                request = self.conn.recv()
            except (EOFError, OSError):
                return
            if request == "get":
                self.conn.send(self._hand_out())
            elif request == "gone":
                self._connected(-1)
            elif request == "poisoned":
                self.conn.send(len(self.poisoned))
            elif request == "payload":
                self.conn.send(self._payload_bytes)
            elif request == "writing":
                # A load writes into the storages from here on: nobody gets
                # them unless it finishes (_filled).
                with self._check_lock:
                    self.poisoned = list(self.views)
            elif request[0] == "filled":
                self.conn.send(self._filled(*request[1:]))

    def _filled(self, key: dict, model: str) -> list[str]:
        """A load wrote ``model``'s weights into the storages: serve them as
        they are now, under ``key``. Returns why not."""
        with self._check_lock:
            try:
                payload = pickle.loads(self._payload_bytes)
                payload["key"] = key
                self._payload_bytes = pickle.dumps(
                    payload, protocol=pickle.HIGHEST_PROTOCOL
                )
                self.checksums = dict(zip(self.views, _checksums(self.views.values())))
            except Exception as e:  # noqa: BLE001 - told to whoever asked
                return [f"{self.label}: {e!r}"]
            finally:
                self._release_checksum_memory()
            self.poisoned = []
            with self._lock:
                self.checked = self.connects
        if self.lock is not None:
            self.lock.set_state("serving", model=model)
        logger.info(f"{_TAG} {self.label}: now holds {model}")
        return []

    def _hand_out(self) -> tuple[bool, bytes]:
        """(handed out, the payload or why not, as a client reads it)."""
        changed = self._check()
        if changed:
            logger.error(
                f"{_TAG} {self.label}: {len(changed)} shared storages changed since export "
                "(something wrote into the cached weights); refusing to serve them"
            )
            error = f"{len(changed)} of the daemon's shared storages changed since export; restart the daemon"
            return False, pickle.dumps({"error": error})
        self._connected(+1)
        return True, self._payload_bytes

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


class _Server:
    """The daemon's socket, in its main process: every server connects here,
    whichever GPU's weights it takes. Passes a GPU's requests to the rank
    process that holds it (_Holder), and knows whether the daemon is in use."""

    def __init__(self, path: str, label: str, ranks: int = 0):
        self.path, self.label = path, label
        self.ranks = ranks  # the daemon's rank processes, once all have GPUs
        self._lock = threading.Lock()  # guards the fields below
        # GPU name -> (the pipe to its rank process, one request at a time).
        self.holders: dict[str, tuple] = {}
        self.keys: dict[str, dict] = {}  # GPU name -> its rank's launch key
        self.clients: dict[str, int] = {}  # connected servers, by GPU
        # Connections that use the daemon (a launch's main process, its
        # ranks), by launch.
        self.users: dict = {}
        # The connected main processes of launches, by connection: the launch,
        # whom to name, its answers to notices, whether its servers serve and
        # why they could not go on with weights rewritten under them.
        self.launches: dict = {}
        self.stopped = False  # `stop`: the weights are not to be used
        self.load = None  # `load`'s doing: (checkpoint) -> why not (the daemon's)
        self._changing = threading.Lock()  # one stop or load at a time
        self.stopping = False
        self.held_by = None  # whoever is about to stop this daemon (_hold)
        self.holds = 0  # its connections
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

    def add(self, gpu: str, conn, key: dict | None = None) -> None:
        """A rank process holds ``gpu`` and answers on ``conn``; ``key``: what
        a launch's main process has for that rank if this daemon serves it."""
        with self._lock:
            self.holders[gpu] = (conn, threading.Lock())
            self.keys[gpu] = key
            self.clients.setdefault(gpu, 0)

    def _users(self, but=None) -> int:
        """Connections that use the daemon; ``but``: not that launch's own."""
        return sum(
            n for launch, n in self.users.items() if but is None or launch != but
        )

    def _use(self, launch, delta: int) -> None:
        self.users[launch] = self.users.get(launch, 0) + delta
        if not self._users():
            self.idle_since = time.monotonic()

    def _idle_for(self) -> float:
        busy = self._users() or self.held_by is not None
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
                    # Stopped weights nobody has are of no use to keep.
                    unused = idle_for and self.stopped and not self._changing.locked()
                    if idle_for >= timeout_s or unused:
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
        with contextlib.suppress(OSError):
            os.unlink(self.path)

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _ask(self, gpu: str, request: str, wait_s: float = -1):
        """A rank process's answer; None: it is busy past ``wait_s``."""
        conn, lock = self.holders[gpu]
        if not lock.acquire(timeout=wait_s):
            return None
        try:
            conn.send(request)
            return conn.recv()
        finally:
            lock.release()

    def _tell(self, gpu: str, message: str) -> None:
        conn, lock = self.holders[gpu]
        with lock:
            conn.send(message)

    def _handle(self, conn: socket.socket):
        peer = "?"
        try:
            conn.settimeout(30)  # a request is small and sent at once
            req = _recv(conn, limit=1 << 20)
            conn.settimeout(None)
            peer = req.get("who", "?")
            if req.get("v") != PROTOCOL:
                _send(
                    conn,
                    {
                        "error": f"the daemon speaks version {PROTOCOL} of the weight "
                        f"cache messages, not {req.get('v')}: restart it"
                    },
                )
                return
            if req.get("cmd") == "ping":
                with self._lock:
                    clients, idle_s = dict(self.clients), self._idle_for()
                    gpus, stopped = list(self.holders), self.stopped
                poisoned = {}
                for gpu in gpus:
                    with contextlib.suppress(Exception):
                        poisoned[gpu] = self._ask(gpu, "poisoned", wait_s=2) or 0
                _send(
                    conn,
                    {
                        "clients": clients,
                        "idle_s": idle_s,
                        "poisoned": poisoned,
                        "stopped": stopped,
                    },
                )
                return
            if req.get("cmd") == "hold":
                self._hold(conn, req.get("by"))
                return
            if req.get("cmd") in ("stop", "load"):
                self._change(conn, req)
                return
            gpu, launch = req.get("gpu"), req.get("launch")
            if req.get("cmd") not in ("get", "attach"):
                _send(conn, {"error": f"unknown command {req.get('cmd')!r}"})
                return
            attach = req.get("cmd") == "attach"
            with self._lock:
                unfit = self._unfit(req.get("keys") or {}) if attach else None
                known = attach or gpu in self.holders
                closed = self._closed()
                if known and not closed and not unfit:
                    self._use(launch, +1)
                    if attach:
                        self.launches[conn] = {
                            "launch": launch,
                            "peer": peer,
                            "replies": queue.Queue(),
                            "serves": False,
                            "rewrite": None,
                        }
            if not known:
                _send(conn, {"error": f"no rank of this daemon holds {gpu}"})
                return
            if closed:
                _send(conn, {"error": closed})
                return
            if unfit:
                _send(conn, {"fits": False, **unfit})
                return
            try:
                if attach:
                    self._attached(conn, peer)
                else:
                    self._serve(conn, gpu, peer)
            finally:
                with self._lock:
                    self._use(launch, -1)
        except Exception as e:  # noqa: BLE001 - one bad client must not stop serving
            logger.warning(f"{_TAG} {self.label}: request from {peer} failed: {e}")
        finally:
            conn.close()

    def _unfit(self, keys: dict) -> dict | None:
        """Why this daemon does not serve the launch whose ranks have ``keys``
        (by GPU); None: it does."""
        if len(self.keys) < self.ranks:
            return {"starting": True}  # its ranks have not all taken their GPUs
        why = []
        if set(keys) != set(self.keys):
            shared = len(set(keys) & set(self.keys))
            why.append(
                f"it holds {len(self.keys)} GPU(s) and the launch runs on "
                f"{len(keys)}, {shared} of them the same"
            )
        for gpu in sorted(set(keys) & set(self.keys)):
            why += _diff_key(self.keys[gpu] or {}, keys[gpu])
            if len(why) > 25:
                break
        return {"why": why[:25]} if why else None

    def _closed(self) -> str | None:
        """Why no server gets the weights now; None: one does."""
        if self.stopping or self.held_by is not None:
            return "the daemon is shutting down"
        if self.stopped:
            return "the daemon's weights are stopped, until a load"
        if self._changing.locked():
            return "the daemon's weights are being stopped or loaded"
        return None

    def _attached(self, conn: socket.socket, peer: str) -> None:
        """A launch's main process stays connected: the launch uses the
        daemon from before its ranks take their weights, and hears `stop` and
        `load` here once its servers serve."""
        _send(conn, {"fits": True})
        logger.info(f"{_TAG} {self.label}: serves the launch of {peer}")
        launch = self.launches[conn]
        try:
            with contextlib.suppress(Exception):  # until it has gone, however
                while True:
                    said = _recv(conn, limit=1 << 20)
                    if "serves" in said:
                        launch.update(serves=True, rewrite=said.get("rewrite"))
                    else:
                        launch["replies"].put(said)
        finally:
            with self._lock:
                del self.launches[conn]
            launch["replies"].put({})  # nothing of it runs on the weights
        logger.info(f"{_TAG} {self.label}: the launch of {peer} has gone")

    def _change(self, conn: socket.socket, req: dict) -> None:
        """`stop` or `load`: answers why it was not done ([]: it was), and
        how many launches' servers have the weights."""
        if not self._changing.acquire(blocking=False):
            _send(conn, {"why": ["another stop or load is under way"]})
            return
        try:
            try:
                why = self.stop() if req["cmd"] == "stop" else self.load(req["model"])
            except Exception as e:  # noqa: BLE001 - told to whoever asked
                why = [f"{req['cmd']} failed: {e!r}"]
            _send(conn, {"why": why, "servers": len(self.launches)})
        finally:
            self._changing.release()

    def _tell_launches(self, event: dict) -> list[str]:
        """Give ``event`` to every launch that has the weights and wait for
        each one's answer. Returns why one could not do as told."""
        with self._lock:
            launches = [(c, x) for c, x in self.launches.items() if x["serves"]]
        why = []
        for conn, launch in launches:
            while not launch["replies"].empty():  # to a notice given up on
                launch["replies"].get_nowait()
            with contextlib.suppress(OSError):
                _send(conn, event)
        for _, launch in launches:
            try:
                said = launch["replies"].get(timeout=_NOTICE_TIMEOUT_S)
            except queue.Empty:
                said = {"why": f"no answer in {_NOTICE_TIMEOUT_S}s"}
            if said.get("why"):
                why.append(f"the server of {launch['peer']}: {said['why']}")
        return why

    def cannot_stop(self, rewrite: bool = False) -> list[str]:
        """Why the servers that have the weights could not stop using them
        ([]: they could); ``rewrite``: nor go on once they hold other values."""
        with self._lock:
            launches = list(self.launches.values())
            heard = {x["launch"] for x in launches if x["serves"]}
            deaf = [x for x, n in self.users.items() if n and x not in heard]
        why = [
            f"the server of {x['peer']}: {x['rewrite']}"
            for x in launches
            if rewrite and x["rewrite"]
        ]
        if deaf:
            why.append(
                f"{len(deaf)} launch(es) that have them cannot be told: their "
                "servers are still starting"
            )
        return why

    def stop(self) -> list[str]:
        """`stop`: the weights are not to be used. Returns once no server
        runs anything on them, and none does until `resume`. Returns why not
        ([]: done), and then the servers use them as before."""
        why = self.cannot_stop()
        if why:
            return why
        with self._lock:
            self.stopped = True
        why = self._tell_launches({"event": "stop"})
        if why:
            self.resume(changed=False)
        return why

    def resume(self, changed: bool, keys: dict | None = None) -> list[str]:
        """The weights are to be used (again); ``changed``: they hold other
        values than at the stop, under ``keys`` (by GPU). Returns why a server
        that has them could not go on."""
        with self._lock:
            self.keys.update(keys or {})
            self.stopped = False
        return self._tell_launches({"event": "use", "changed": changed})

    def _hold(self, conn: socket.socket, by) -> None:
        """For whoever is about to stop this daemon (``by``: the ranks of one
        launch ask each for their GPU): if no server has its weights now, none
        gets them while such a connection stays open. Found and refused under
        one lock, so no server slips in between."""
        with self._lock:
            free = self.held_by is None or (by is not None and self.held_by == by)
            # Its own launch's connections do not keep it: that launch asks.
            # The others' are counted by launch: a server has several.
            others = [x for x, n in self.users.items() if n and x != by]
            clients = len(others)
            held = free and not (clients or self.stopping)
            if held:
                self.held_by = by if by is not None else conn
                self.holds += 1
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
                self.holds -= 1
                if not self.holds:
                    self.held_by = None
                    self.idle_since = time.monotonic()

    def _serve(self, conn: socket.socket, gpu: str, peer: str) -> None:
        handed_out, data = self._ask(gpu, "get")
        if not handed_out:
            _send_bytes(conn, data)
            return
        with self._lock:
            self.clients[gpu] += 1
        try:
            _send_bytes(conn, data)
            logger.info(f"{_TAG} {self.label}: served {gpu} to {peer}")
            # The connection stays open while the client lives, so each side
            # sees the other go away.
            with contextlib.suppress(OSError):
                while conn.recv(1 << 12):
                    pass
        finally:
            with self._lock:
                self.clients[gpu] -= 1
            self._tell(gpu, "gone")
        logger.info(f"{_TAG} {self.label}: {peer} disconnected from {gpu}")
