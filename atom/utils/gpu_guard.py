# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Other processes on this server's GPUs.

A Mono decode kernel holds every CU of its GPU (one CTA per CU, each taking the
CU's whole LDS) and its CTAs spin on each other and on the peer GPUs' pushes
(``atom.mono.plan.execution``). So one kernel of another process that holds LDS
on one CU of one TP GPU keeps a Mono CTA from launching, and the whole TP group
spins until that process leaves the CU alone (a hang of minutes was seen with a
second process on one TP GPU). The original decode path only slows down under
the same load.

KFD lists every process that opened a GPU under ``/sys/class/kfd/kfd/proc/<host
pid>/`` with its VRAM per GPU id, readable from inside a container. The guard
takes the processes on the server's GPUs once the engine is up as its own and
reports any other that shows up there (and when it leaves); the engine keeps
decode off Mono while one is present.
"""

import logging
import os
import threading
import time

logger = logging.getLogger("atom")

# How often the guard polls KFD for other processes on the server's GPUs.
GUARD_INTERVAL_S = 0.2
# A forward running longer than this is reported by the engine's step watchdog
# (the engine thread is blocked in it and logs nothing of its own meanwhile).
STEP_WATCHDOG_S = 5.0
# A foreign process must be gone this long before Mono decode comes back on, so
# a periodic neighbour does not flip steps between Mono and eager.
HOLD_OFF_S = 5.0

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


def visible_gpus() -> list[int]:
    """Topology positions of this process's GPUs. ROCR_VISIBLE_DEVICES, when
    set, picks the GPUs the HIP runtime numbers from 0, and
    HIP_VISIBLE_DEVICES (else CUDA_VISIBLE_DEVICES) then indexes into those;
    with neither set every GPU is visible. Entries that are not indices (UUIDs)
    are dropped with an error, never silently."""
    avail = list(range(len(kfd_gpu_ids())))

    def pick(var: str, pool: list[int]) -> list[int]:
        raw = os.environ.get(var, "").strip()
        if not raw:
            return pool
        items = [x.strip() for x in raw.split(",") if x.strip()]
        idx = [int(x) for x in items if x.isdigit() and int(x) < len(pool)]
        if len(idx) < len(items):
            logger.error(
                "GPU guard: %s=%r has entries that are not GPU indices (UUIDs?) "
                "or are out of range; only %s are watched",
                var,
                raw,
                idx,
            )
        return [pool[i] for i in idx]

    avail = pick("ROCR_VISIBLE_DEVICES", avail)
    hip = (
        "HIP_VISIBLE_DEVICES"
        if os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
        else "CUDA_VISIBLE_DEVICES"
    )
    return pick(hip, avail)


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


def describe(procs: dict[int, dict[int, int]]) -> str:
    return "; ".join(
        f"pid {p}: "
        + ", ".join(f"GPU {g} {v / _MIB:.0f} MiB" for g, v in sorted(use.items()))
        for p, use in sorted(procs.items())
    )


class GpuGuard:
    """Polls KFD for processes on this server's GPUs that were not there at start."""

    def __init__(self, label: str, interval_s: float, hold_off_s: float = HOLD_OFF_S):
        self.label = label
        self.hold_off_s = hold_off_s
        self.interval_s = interval_s
        ids = kfd_gpu_ids()
        self.gpus = {g: ids[g] for g in visible_gpus() if g < len(ids)}
        if not self.gpus:
            logger.error(
                "%s: GPU guard found no GPU to watch (no KFD sysfs, or "
                "HIP/ROCR_VISIBLE_DEVICES names none); it will not see other processes",
                label,
            )
        elif not (
            os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
            or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            or os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
        ):
            logger.warning(
                "%s: no *_VISIBLE_DEVICES set: the GPU guard watches all %d GPUs of "
                "this host, so another server on them counts as foreign",
                label,
                len(self.gpus),
            )
        start = scan(self.gpus)
        self.own = set(start)
        self.foreign: dict[int, dict[int, int]] = {}
        self._gone_since: float | None = None
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

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._poll()
            except Exception:
                logger.exception("%s: GPU guard poll failed", self.label)

    def _poll(self) -> None:
        now = {p: u for p, u in scan(self.gpus).items() if p not in self.own}
        t = time.monotonic()
        if now:
            self._gone_since = None
        elif self.foreign:
            # all gone: hold Mono off until they have stayed away
            if self._gone_since is None:
                self._gone_since = t
            if t - self._gone_since < self.hold_off_s:
                return
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
