# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# The per-rank operations port TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``libtilert_dsv41_rocm.so``'s
# ``dsv41_e2e_set_sampling`` / ``_begin`` / ``_admit`` / ``_enqueue`` / ``_wait`` and ``DsV41Graph.import_kv``, which
# TileRT's single process runs for 8 GPUs; here one process a GPU.
"""The decode host's rank fan-out: the server process drives one worker process a GPU (``spawn``), each holding its
rank's device state (``tkworker.TkRankWorker``: a ``tk.TkRank``; ``FakeRank``: CPU), and ``RankEngine`` is the
``session.DecodeEngine`` over them.

Control channel: a shared-memory mailbox (``multiprocessing.shared_memory``, in ``/dev/shm``), one 4 KiB slot a worker.
The server writes a command's op and int64 arguments, then its sequence number; the worker, polling that number, runs
the op and writes its status, int64 results and text (an error's traceback, or a JSON report), then the number back.
Why not pipes: a command is on the per-chunk critical path twice (``step``, then ``wait_read`` whose answer carries the
tokens to stream), and a blocked reader's wake-up through a pipe costs tens of microseconds a rank plus pickling, where a
polling worker sees a mailbox write in about a microsecond; TileRT's own worker threads poll a mailbox the same way
(``fn_87a9c0``). A worker spins ``spin_s`` (50 ms) after its last command, which covers the gaps between a request's
chunks, then sleeps 0.2 ms between polls while idle; the server's wait yields the GIL between polls. Ordering relies on
x86-64 (TSO): the payload's stores precede the sequence number's, and the reader re-reads the number until stable.

Every op goes to every rank and waits for all answers (a rank's step polls the others, so ranks must not drift);
``wait_read`` returns rank 0's logs. Failures:

- an op raising in a worker: ``RankError`` (every failing rank's text); the group stays usable (the next request
  re-admits and reloads every rank);
- an answer not in ``timeout`` (per op: ``TIMEOUTS``) or a worker process gone: ``RankTimeout`` / ``RankDied``, and the
  group is broken (``RankGroupBroken`` on every later call): the host must restart;
- a worker whose parent died exits (polls ``os.getppid``).

Ops (``OPS``): ``hello`` (the backend's report: rank 0's ``max_seq_len``, ``num_mtp``, ``stop_token_ids``, ``greedy``,
``base_ptr`` / ``transport`` / ``session_id`` of the receive buffer it owns), ``set_sampling(on, p0..p3)``,
``admit(first, pos, mtp, has_params, p0..p3)``, ``load_kv(seq_len, generation)``, ``step(k, ticket)``,
``wait_read(ticket, acc_from, k, tok_from)``, ``sync``, ``state`` (a backend's JSON, for tests), ``stop``.

The sampling counter (``samp_params[3]``): TileRT's .so keeps one int32 counter a process, bumped by every
``dsv41_e2e_set_sampling`` and ``dsv41_e2e_begin`` (host ``fn_86bf60`` / ``fn_86b390``, global ``0x8b5d5c``) and written
as the parameters' fourth word; ``RankEngine`` keeps it and sends every rank the same parameters.
"""

from __future__ import annotations

import importlib
import json
import logging
import multiprocessing as mp
import os
import struct
import time
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

SLOT = 4096
NARGS, NRES, TEXT = 24, 64, 3072
_CMD, _OP, _ARGS = 0, 4, 16  # u32 seq; u32 op, u32 nargs; i64 args[NARGS]
_DONE, _STATUS, _RES, _TEXT = (
    256,
    260,
    272,
    1024,
)  # u32 seq; i32 status, u32 nres, u32 ntext; i64 res[NRES]; text

OPS = {
    "hello": 1,
    "set_sampling": 2,
    "admit": 3,
    "load_kv": 4,
    "step": 5,
    "wait_read": 6,
    "sync": 7,
    "state": 8,
    "stop": 9,
}
_OP_NAMES = {v: k for k, v in OPS.items()}
TIMEOUTS = {  # seconds an op may take on every rank (wait_read: a chunk of up to 8 steps, or a device fault's timeout)
    "hello": 7200.0,
    "set_sampling": 30.0,
    "admit": 30.0,
    "load_kv": 300.0,
    "step": 30.0,
    "wait_read": 150.0,
    "sync": 150.0,
    "state": 30.0,
    "stop": 30.0,
}


class RankError(RuntimeError):
    """An op raised on one or more ranks (``errors``: ``{rank: text}``)."""

    def __init__(self, op: str, errors: dict):
        self.op, self.errors = op, errors
        first = min(errors)
        super().__init__(
            f"{op} failed on rank(s) {sorted(errors)}; rank {first}: {errors[first].strip()[-800:]}"
        )


class RankTimeout(RuntimeError):
    pass


class RankDied(RuntimeError):
    pass


class RankGroupBroken(RuntimeError):
    pass


@dataclass
class Answer:
    status: int
    res: list[int]
    text: str

    def json(self):
        return json.loads(self.text) if self.text else None


class _Slot:
    def __init__(self, buf, rank: int):
        self.b, self.o = buf, rank * SLOT

    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.b, self.o + off)[0]

    def _stable(self, off: int) -> int:
        v = self._u32(off)
        while True:
            w = self._u32(off)
            if w == v:
                return v
            v = w

    def cmd(self) -> int:
        return self._u32(_CMD)

    def done(self) -> int:
        return self._u32(_DONE)

    def post(self, seq: int, op: int, args) -> None:
        if len(args) > NARGS:
            raise ValueError(f"{len(args)} arguments > {NARGS}")
        struct.pack_into(
            f"<III{len(args)}q", self.b, self.o + _OP, op, len(args), 0, *args
        )
        struct.pack_into("<I", self.b, self.o + _CMD, seq)

    def request(self) -> tuple[int, int, list[int]]:
        seq = self._stable(_CMD)
        op, n = struct.unpack_from("<II", self.b, self.o + _OP)
        return seq, op, list(struct.unpack_from(f"<{n}q", self.b, self.o + _ARGS))

    def answer(self, seq: int, status: int, res, text: str) -> None:
        res = [int(x) for x in res]
        if len(res) > NRES:
            raise ValueError(f"{len(res)} results > {NRES}")
        tb = text.encode()[-TEXT:]
        struct.pack_into(
            f"<iII{len(res)}q",
            self.b,
            self.o + _STATUS,
            status,
            len(res),
            len(tb),
            *res,
        )
        self.b[self.o + _TEXT : self.o + _TEXT + len(tb)] = tb
        struct.pack_into("<I", self.b, self.o + _DONE, seq)

    def result(self) -> Answer:
        status, n, nt = struct.unpack_from("<iII", self.b, self.o + _STATUS)
        res = list(struct.unpack_from(f"<{n}q", self.b, self.o + _RES))
        text = bytes(self.b[self.o + _TEXT : self.o + _TEXT + nt]).decode(
            errors="replace"
        )
        return Answer(status, res, text)


# ---------------------------------------------------------------------------------------------------------------
# the worker process


def _load(factory: str):
    mod, _, attr = factory.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _run_op(backend, op: str, a: list[int]) -> tuple[list[int], str]:
    """One op on a backend: (int results, text)."""
    if op == "hello":
        return [], json.dumps(backend.hello())
    if op == "set_sampling":
        backend.set_sampling(a[1:5], bool(a[0]))
    elif op == "admit":
        backend.admit(a[0], a[1], bool(a[2]), a[4:8] if a[3] else None)
    elif op == "load_kv":
        backend.load_kv(a[0], a[1])
    elif op == "step":
        backend.step(a[0], a[1])
    elif op == "wait_read":
        out = backend.wait_read(*a[:4])
        if out is not None:
            acc, toks = out
            return [len(acc), *acc, *toks], ""
    elif op == "sync":
        backend.sync()
    elif op == "state":
        return [], json.dumps(backend.state())
    elif op != "stop":
        raise ValueError(f"unknown op {op}")
    return [], ""


def _worker_main(
    rank: int,
    nranks: int,
    shm_name: str,
    factory: str,
    kwargs: dict,
    spin_s: float,
    env: dict,
):
    """A rank's loop: build the backend, then answer commands until ``stop`` or the parent's death."""
    os.environ.update(env)
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        _serve(rank, nranks, _Slot(shm.buf, rank), factory, kwargs, spin_s)
    finally:
        shm.close()


def _serve(rank, nranks, slot: _Slot, factory, kwargs, spin_s) -> None:
    ppid = os.getppid()
    backend, err = None, ""
    try:
        backend = _load(factory)(rank, nranks, **kwargs)
    # any failure goes back to the server, as the hello's answer
    except BaseException:  # noqa: BLE001
        err = traceback.format_exc()
    last = 0  # the server's first sequence number is 1 (hello), which may be posted before this process starts
    idle_since, polls = time.monotonic(), 0
    while True:
        if slot.cmd() == last:
            polls += 1
            if polls % 4096 == 0 and os.getppid() != ppid:
                return  # the server is gone
            if time.monotonic() - idle_since > spin_s:
                time.sleep(2e-4)
            continue
        seq, op_id, args = slot.request()
        op = _OP_NAMES.get(op_id, f"op{op_id}")
        if err:
            slot.answer(seq, 1, [], err)
            return
        try:
            res, text = _run_op(backend, op, args)
            slot.answer(seq, 0, res, text)
        # any failure goes back to the server, as the op's answer
        except BaseException:  # noqa: BLE001
            slot.answer(seq, 1, [], traceback.format_exc())
        last, idle_since, polls = seq, time.monotonic(), 0
        if op == "stop":
            close = getattr(backend, "close", None)
            if close is not None:
                close()
            return


# ---------------------------------------------------------------------------------------------------------------
# the server side


class RankGroup:
    """``nranks`` worker processes running ``factory(rank, nranks, **kwargs)`` (``"module:callable"``), started with
    ``spawn``; ``reports``: each rank's ``hello``. ``env``: set in each worker before the backend is built.
    """

    def __init__(
        self,
        nranks: int,
        factory: str,
        kwargs: dict | None = None,
        timeouts: dict | None = None,
        spin_s: float = 0.05,
        env: dict | None = None,
    ):
        self.nranks = nranks
        self.timeouts = {**TIMEOUTS, **(timeouts or {})}
        self.shm = shared_memory.SharedMemory(create=True, size=nranks * SLOT)
        self.shm.buf[: nranks * SLOT] = bytes(nranks * SLOT)
        self.slots = [_Slot(self.shm.buf, r) for r in range(nranks)]
        self.seq = 0
        self.broken: str | None = None
        ctx = mp.get_context("spawn")
        self.procs = [
            ctx.Process(
                target=_worker_main,
                args=(
                    r,
                    nranks,
                    self.shm.name,
                    factory,
                    kwargs or {},
                    spin_s,
                    env or {},
                ),
                name=f"pd-rank{r}",
                daemon=True,
            )
            for r in range(nranks)
        ]
        for p in self.procs:
            p.start()
        try:
            self.reports = [a.json() for a in self.call("hello")]
        except BaseException:
            self.close(graceful=False)
            raise

    def call(self, op: str, args=(), timeout: float | None = None) -> list[Answer]:
        """``op(*args)`` on every rank; the answers in rank order."""
        if self.broken:
            raise RankGroupBroken(self.broken)
        self.seq = (self.seq + 1) & 0xFFFFFFFF or 1
        seq, op_id = self.seq, OPS[op]
        for s in self.slots:
            s.post(seq, op_id, args)
        limit = self.timeouts[op] if timeout is None else timeout
        deadline = time.monotonic() + limit
        pending = set(range(self.nranks))
        next_check, polls = 0.0, 0
        while pending:
            pending = {r for r in pending if self.slots[r].done() != seq}
            if not pending:
                break
            now = time.monotonic() if polls % 16 == 0 else 0.0
            if now >= next_check:
                dead = [r for r in pending if not self.procs[r].is_alive()]
                if dead:
                    self.broken = f"{op}: rank(s) {dead} exited (codes {[self.procs[r].exitcode for r in dead]})"
                    raise RankDied(self.broken)
                if now > deadline:
                    self.broken = f"{op}: rank(s) {sorted(pending)} did not answer in {limit:.1f}s"
                    raise RankTimeout(self.broken)
                next_check = now + 0.01
            polls += 1
            if polls % 64 == 0:
                time.sleep(0)  # let the server's other threads (HTTP) have the GIL
        answers = [s.result() for s in self.slots]
        errors = {r: a.text for r, a in enumerate(answers) if a.status}
        if errors:
            raise RankError(op, errors)
        return answers

    def close(self, graceful: bool = True) -> None:
        if graceful and not self.broken and any(p.is_alive() for p in self.procs):
            try:
                self.call("stop")
            except Exception:
                logger.exception("rank group stop failed")
        for p in self.procs:
            p.join(timeout=5 if graceful else 0.1)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self.slots = []
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            pass


class RankEngine:
    """``session.DecodeEngine`` over a ``RankGroup``: every op on every rank, the sampling counter kept here, ``wait`` +
    ``read_outputs`` one ``wait_read`` round trip (rank 0's logs). ``load_kv`` ignores the server's buffer: the workers
    read the landed image themselves (rank 0 owns the buffer, the others map it)."""

    name = "ranks"

    def __init__(self, group: RankGroup):
        from . import session as S

        self._S = S
        self.group = group
        info = group.reports[0]
        self.max_seq_len = int(info["max_seq_len"])
        self.num_mtp = int(info["num_mtp"])
        self.stop_token_ids = frozenset(
            int(t) for t in info.get("stop_token_ids", [S.EOS_ID])
        )
        self.greedy = bool(info.get("greedy", False))
        self.counter = 0  # TileRT's sampling counter (0x8b5d5c), 0 at load
        self._ticket = 0
        self._waiting = None

    def check_sampling(self, sp) -> None:
        if sp is None and not self.greedy:
            raise self._S.CapabilityUnavailable(
                "greedy decoding (temperature 0) is not served: it runs TileRT's mode-2 graphs, and this host's plan "
                "(plan/decode_launches.json) holds the mode-5 (sampling) step only"
            )

    def _counted(self, params: list[int]) -> list[int]:
        self.counter = (self.counter + 1) & 0x7FFFFFFF
        return [int(p) for p in params[:3]] + [self.counter]

    def set_sampling(self, params: list[int], on: bool) -> None:
        self.group.call("set_sampling", [int(on), *self._counted(params)])

    def admit(
        self, first_token: int, pos: int, params: list[int] | None = None
    ) -> None:
        p = self._counted(params) if params is not None else [0, 0, 0, 0]
        self.group.call(
            "admit", [int(first_token), int(pos), 1, int(params is not None), *p]
        )

    def load_kv(self, received, buffer) -> None:
        self.group.call("load_kv", [int(received.seq_len), int(received.generation)])

    def step(self, k: int) -> int:
        self._ticket += 1
        self.group.call("step", [int(k), self._ticket])
        return self._ticket

    def wait(self, ticket) -> None:
        self._waiting = ticket

    def read_outputs(
        self, acc_from: int, k: int, tok_from: int
    ) -> tuple[list[int], list[int]]:
        ticket, self._waiting = self._waiting, None
        res = self.group.call("wait_read", [ticket, acc_from, k, tok_from])[0].res
        n = res[0]
        acc = res[1 : 1 + n]
        return acc, res[1 + n : 1 + n + sum(acc)]


# ---------------------------------------------------------------------------------------------------------------
# the CPU rank


class FakeRank:
    """A CPU worker backend: every rank runs ``session.FakeEngine`` (the logs) on the same commands and builds its
    ``AdmitArgs`` (``tk.host.admit_args``) over fake addresses; rank 0 answers ``wait_read``. Test hooks: ``fail`` =
    ``{"op", "rank"}`` raises there, ``hang`` = ``{"op", "rank", "s"}`` sleeps there; ``step_s``: a step's time.
    """

    def __init__(
        self,
        rank: int,
        nranks: int,
        max_seq_len: int = 1 << 20,
        fail=None,
        hang=None,
        step_s: float = 0.0,
        **fake,
    ):
        from ..tk import host
        from . import session as S

        self.rank, self.nranks = rank, nranks
        self.engine = S.FakeEngine(max_seq_len=max_seq_len, **fake)
        self.fail, self.hang, self.step_s = fail, hang, step_s
        self._admit_args = host.admit_args
        names = (
            "temp:plain_acc",
            "temp:ar_acc",
            "temp:ar_num",
            "temp:ctl",
            "temp:blk_tok",
            "temp:blk_pos",
        )
        names += ("temp:vtok", "temp:vpos", "param:embed", "temp:X", "temp:samp_params")
        self.addr = {
            n: 0x7F0000000000 + (rank << 32) + (i << 20) for i, n in enumerate(names)
        }
        self.ready: dict[int, float] = {}
        self.log: list = []
        self.admit_blob = b""

    def _hooks(self, op: str) -> None:
        if (
            self.fail
            and self.fail.get("op") == op
            and self.fail.get("rank") == self.rank
        ):
            raise RuntimeError(f"injected failure in {op} on rank {self.rank}")
        if (
            self.hang
            and self.hang.get("op") == op
            and self.hang.get("rank") == self.rank
        ):
            time.sleep(float(self.hang.get("s", 3600)))

    def hello(self) -> dict:
        self._hooks("hello")
        e = self.engine
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "max_seq_len": e.max_seq_len,
            "num_mtp": e.num_mtp,
            "stop_token_ids": sorted(e.stop_token_ids),
            "greedy": True,
            "transport": "mooncake",
            "session_id": "fake:0",
            "base_ptr": 0x7E0000000000,
        }

    def set_sampling(self, params, on) -> None:
        self._hooks("set_sampling")
        self.log.append(["set_sampling", list(params), on])
        self.engine.set_sampling(list(params), on)

    def admit(self, first, pos, mtp, params) -> None:
        self._hooks("admit")
        self.log.append(["admit", first, pos, mtp, params])
        self.admit_blob = self._admit_args(
            self.addr, first, pos, mtp, params, self.engine.max_seq_len
        )
        self.engine.admit(first, pos, params)

    def load_kv(self, seq_len, generation) -> None:
        self._hooks("load_kv")
        self.log.append(["load_kv", seq_len, generation])

    def step(self, k, ticket) -> None:
        self._hooks("step")
        if len(self.ready) >= 4:
            raise RuntimeError("more than 4 chunks in flight")
        self.engine.step(k)
        self.ready[ticket] = time.monotonic() + self.step_s * k

    def wait_read(self, ticket, acc_from, k, tok_from):
        self._hooks("wait_read")
        delay = self.ready.pop(ticket) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return (
            self.engine.read_outputs(acc_from, k, tok_from) if self.rank == 0 else None
        )

    def sync(self) -> None:
        self.ready.clear()

    def state(self) -> dict:
        return {
            "rank": self.rank,
            "log": self.log,
            "admit_blob": self.admit_blob.hex(),
            "addr": self.addr,
        }
