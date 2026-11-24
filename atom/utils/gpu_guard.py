# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Other processes on this server's GPUs.

A Mono decode kernel holds every CU of its GPU (one CTA per CU, each taking the
CU's whole LDS) and its CTAs spin on each other and on the peer GPUs' pushes
(``atom.mono.plan.execution``). So one kernel of another process that holds LDS
on one CU of one TP GPU keeps a Mono CTA from launching, and the whole TP group
spins until that process leaves the CU alone: base0_c8 lost 212 s that way. The
original decode path only slows down under the same load.

KFD lists every process that opened a GPU under ``/sys/class/kfd/kfd/proc/<host
pid>/`` with its VRAM per GPU id, readable from inside a container. The guard
takes the processes on the server's GPUs once the engine is up as its own and
reports any other that shows up there (and when it leaves); the engine keeps
decode off Mono while one is present.
"""

import logging
import os
import subprocess
import sys
import threading
import time

logger = logging.getLogger("atom")

KFD = "/sys/class/kfd/kfd"
_MIB = 1 << 20
# any VRAM: a process shows its context's 2 MiB on a GPU as soon as it opened
# that GPU, before its own allocations and kernels -- the guard's head start
_MIN_VRAM = 1


def _read_int(path: str) -> int:
    try:
        with open(path) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def kfd_gpu_ids() -> list[int]:
    """KFD gpu ids in topology order (the HIP / ROCr device order); CPU nodes skipped."""
    nodes = f"{KFD}/topology/nodes"
    try:
        names = sorted(os.listdir(nodes), key=int)
    except (OSError, ValueError):
        return []
    ids = [_read_int(f"{nodes}/{n}/gpu_id") for n in names]
    return [i for i in ids if i]


def _kfd_unique_ids() -> list[int]:
    """Each GPU's ``unique_id`` (``ROCR_VISIBLE_DEVICES=GPU-<hex>``), in
    ``kfd_gpu_ids`` order; 0 where a node does not report one."""
    nodes = f"{KFD}/topology/nodes"
    try:
        names = sorted(os.listdir(nodes), key=int)
    except (OSError, ValueError):
        return []
    out = []
    for n in names:
        if not _read_int(f"{nodes}/{n}/gpu_id"):
            continue
        uid = 0
        try:
            with open(f"{nodes}/{n}/properties") as f:
                for line in f:
                    key, _, value = line.partition(" ")
                    if key == "unique_id":
                        uid = int(value.strip() or 0)
        except (OSError, ValueError):
            pass
        out.append(uid)
    return out


def _rocr_visible() -> list[int] | None:
    """Physical indices ROCR_VISIBLE_DEVICES leaves visible (indices or
    ``GPU-<unique id hex>``), in its order; None when it is unset."""
    raw = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    uids = None
    out = []
    for tok in (t.strip() for t in raw.split(",")):
        if tok.isdigit():
            out.append(int(tok))
        elif tok[:4].upper() == "GPU-":
            if uids is None:
                uids = _kfd_unique_ids()
            try:
                want = int(tok[4:], 16)
            except ValueError:
                continue
            if want in uids:
                out.append(uids.index(want))
    return out


def visible_gpus() -> list[int]:
    """Physical indices of this process's GPUs: ROCR_VISIBLE_DEVICES (else every
    GPU), narrowed by HIP_VISIBLE_DEVICES, else CUDA_VISIBLE_DEVICES, which
    index into what ROCr leaves visible."""
    base = _rocr_visible()
    if base is None:
        base = list(range(len(kfd_gpu_ids())))
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(var, "").strip()
        if raw:
            return [
                base[int(x)]
                for x in raw.split(",")
                if x.strip().isdigit() and int(x) < len(base)
            ]
    return base


def scan(gpus: dict[int, int], min_vram: int = _MIN_VRAM) -> dict[int, dict[int, int]]:
    """{host pid: {physical GPU: VRAM bytes}} for every process holding at least
    ``min_vram`` on one of ``gpus`` ({physical index: KFD gpu id})."""
    out = {}
    try:
        pids = os.listdir(f"{KFD}/proc")
    except OSError:
        return out
    for p in pids:
        if not p.isdigit():
            continue
        use = {}
        for g, gid in gpus.items():
            v = _read_int(f"{KFD}/proc/{p}/vram_{gid}")
            if v >= min_vram:
                use[g] = v
        if use:
            out[int(p)] = use
    return out


# A process that opens the GPU, creates a queue and dispatches one tiny kernel.
# KFD remaps every queue of a GPU when a process adds or drops one, which ends
# the Mono spin stall (tools/mono_mailbox_copy_repro.py at
# ATOM_MONO_POLL_SLEEP_UNITS=1: 5 of 9 stalled rounds finished within 0.5 s of
# such a process, unkicked ones after 15-71 s). Raw HIP: importing torch would
# take seconds.
KICK_SCRIPT = """
import ctypes
h = ctypes.CDLL("libamdhip64.so")
h.hipSetDevice(0)
s, p = ctypes.c_void_p(), ctypes.c_void_p()
h.hipStreamCreate(ctypes.byref(s))
h.hipMalloc(ctypes.byref(p), ctypes.c_size_t(4096))
h.hipMemsetAsync(p, 0, ctypes.c_size_t(4096), s)
h.hipStreamSynchronize(s)
"""
# a kicker whose kernel cannot run is killed (its exit remaps the queues too)
KICK_TIMEOUT_S = 20.0


def kick(gpus) -> None:
    """Start a kicker on each of ``gpus`` (physical indices); returns at once."""

    def one(g):
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(g))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        try:
            p = subprocess.Popen(
                [sys.executable, "-c", KICK_SCRIPT],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            p.wait(timeout=KICK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        except OSError as e:
            logger.warning("GPU kick on GPU %d failed to start: %s", g, e)

    for g in gpus:
        threading.Thread(target=one, args=(g,), daemon=True).start()


def describe(procs: dict[int, dict[int, int]]) -> str:
    return "; ".join(
        f"pid {p}: "
        + ", ".join(f"GPU {g} {v / _MIB:.0f} MiB" for g, v in sorted(use.items()))
        for p, use in sorted(procs.items())
    )


class GpuGuard:
    """Polls KFD for processes on this server's GPUs that were not there at start."""

    def __init__(self, label: str, interval_s: float):
        self.label = label
        self.interval_s = interval_s
        ids = kfd_gpu_ids()
        self.gpus = {g: ids[g] for g in visible_gpus() if g < len(ids)}
        start = scan(self.gpus)
        self.own = set(start)
        self.foreign: dict[int, dict[int, int]] = {}
        # until then newcomers are this server's own kickers (`kick`)
        self.transient_until = 0.0
        self._stop = threading.Event()
        crowded = {}
        for p, use in start.items():
            for g, v in use.items():
                crowded.setdefault(g, []).append((p, v))
        logger.info(
            "%s: GPU guard on GPUs %s (KFD ids %s), %d processes of this server",
            label,
            sorted(self.gpus),
            [self.gpus[g] for g in sorted(self.gpus)],
            len(self.own),
        )
        for g, users in sorted(crowded.items()):
            big = [(p, v) for p, v in users if v >= (1 << 30)]
            if len(big) > 1:
                logger.error(
                    "%s: GPU %d already has %d processes holding >= 1 GiB at start (%s): "
                    "the GPUs are shared; Mono decode stalls while another process runs kernels here",
                    label,
                    g,
                    len(big),
                    ", ".join(f"pid {p} {v / _MIB:.0f} MiB" for p, v in big),
                )
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-guard")
        self._thread.start()

    def present(self) -> bool:
        return bool(self.foreign)

    def kick(self) -> None:
        """`kick` this server's GPUs, its kickers not reported as foreign."""
        self.transient_until = time.monotonic() + KICK_TIMEOUT_S + 10.0
        kick(sorted(self.gpus))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            now = {p: u for p, u in scan(self.gpus).items() if p not in self.own}
            if time.monotonic() < self.transient_until:
                now = {p: u for p, u in now.items() if p in self.foreign}
            came = {p: u for p, u in now.items() if p not in self.foreign}
            gone = [p for p in self.foreign if p not in now]
            # one assignment: the engine thread reads it without a lock
            self.foreign = now
            if came:
                logger.error(
                    "%s: another process is on this server's GPUs (%s); Mono decode is off "
                    "until it leaves -- a Mono step cannot finish while another process's "
                    "kernels hold one of its CUs",
                    self.label,
                    describe(came),
                )
            if gone:
                logger.warning(
                    "%s: pid(s) %s left this server's GPUs%s",
                    self.label,
                    sorted(gone),
                    "; Mono decode back on" if not now else "",
                )

    def stop(self) -> None:
        self._stop.set()
