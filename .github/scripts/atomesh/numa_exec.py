#!/usr/bin/env python3
"""Run a command bound to one NUMA node: ``numactl -N <node> -m <node>``.

The ATOM images ship no numactl. CPU affinity and an MPOL_BIND memory policy
both survive execve, so this sets them on itself and execs the command.

Usage: numa_exec.py <node> <command> [args...]

Docker's default seccomp profile refuses set_mempolicy unless the container has
CAP_SYS_NICE (``--cap-add=SYS_NICE``, which pd_slurm_job.sh passes).
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import sys

MPOL_BIND = 2
# set_mempolicy has no glibc wrapper and libnuma is not guaranteed.
SET_MEMPOLICY_SYSCALL = {"x86_64": 238, "aarch64": 237}
ULONG_BITS = ctypes.sizeof(ctypes.c_ulong) * 8


def parse_cpulist(text: str) -> set[int]:
    """Parse a sysfs cpulist such as ``0-63,128-191`` into CPU ids."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def node_cpus(node: int) -> set[int]:
    path = f"/sys/devices/system/node/node{node}/cpulist"
    try:
        with open(path) as f:
            return parse_cpulist(f.read())
    except FileNotFoundError:
        raise SystemExit(f"numa_exec: NUMA node {node} does not exist ({path})")


def bind_cpus(node: int) -> None:
    # A container cpuset can hide part of the node; keep only allowed CPUs.
    cpus = node_cpus(node) & os.sched_getaffinity(0)
    if not cpus:
        raise SystemExit(
            f"numa_exec: no CPU of NUMA node {node} is allowed in this cpuset"
        )
    os.sched_setaffinity(0, cpus)


def bind_memory(node: int) -> None:
    syscall_nr = SET_MEMPOLICY_SYSCALL.get(platform.machine())
    if syscall_nr is None:
        raise SystemExit(f"numa_exec: unsupported architecture {platform.machine()}")
    words = node // ULONG_BITS + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // ULONG_BITS] = 1 << (node % ULONG_BITS)
    libc = ctypes.CDLL(None, use_errno=True)
    # The kernel reads maxnode - 1 bits.
    maxnode = ctypes.c_ulong(words * ULONG_BITS + 1)
    if libc.syscall(syscall_nr, MPOL_BIND, mask, maxnode) == 0:
        return
    err = ctypes.get_errno()
    hint = ""
    if err == errno.EPERM:
        hint = " (the container needs --cap-add=SYS_NICE to call set_mempolicy)"
    elif err == errno.EINVAL:
        hint = " (is the node outside this container's cpuset.mems?)"
    raise SystemExit(
        f"numa_exec: set_mempolicy(MPOL_BIND, node {node}) failed: "
        f"{os.strerror(err)}{hint}"
    )


def main(argv: list[str]) -> None:
    if len(argv) < 2 or not argv[0].isdigit():
        raise SystemExit(f"usage: {sys.argv[0]} <numa-node> <command> [args...]")
    node, command = int(argv[0]), argv[1:]
    bind_cpus(node)
    bind_memory(node)
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        raise SystemExit(f"numa_exec: cannot exec {command[0]}: {exc.strerror}")


if __name__ == "__main__":
    main(sys.argv[1:])
