#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Populate temporary base pages to reclaim cache before THP preparation.

The parent chooses a size from current NUMA/cgroup headroom, binds this process
to that NUMA node, and enforces its command deadline. No pages are pinned and
no system settings are changed. All scratch memory is unmapped before exit.
"""

import argparse
import ctypes
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from compact_huge_pages import (
    GIB,
    MAP_PRIVATE_ANONYMOUS,
    PROT_READ_WRITE,
    mapping_memory,
)
from numa_memory_budget import non_negative_gib

MADV_NOHUGEPAGE = 15
MADV_POPULATE_WRITE = 23  # Linux 5.14+: fault writable base pages in kernel space.


def populate_and_release(gib):
    page = os.sysconf("SC_PAGE_SIZE")
    size = int(gib * GIB) // page * page
    if size <= 0:
        raise ValueError("scratch allocation must contain at least one page")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    start = libc.mmap(None, size, PROT_READ_WRITE, MAP_PRIVATE_ANONYMOUS, -1, 0)
    if start in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "base-page scratch mmap failed")
    began = time.monotonic()
    completed = 0

    def report(event, memory=None):
        row = {
            "event": event,
            "target_gib": size / GIB,
            "populated_gib": completed / GIB,
            "seconds": round(time.monotonic() - began, 3),
        }
        # Repeated smaps walks over a TiB of base-page PTEs would perturb
        # preparation. Only the final verification walks the mapping.
        if memory is not None:
            row.update(
                rss_gib=memory["Rss"] / GIB, anon_huge_gib=memory["AnonHugePages"] / GIB
            )
        else:
            with open("/proc/self/statm") as stream:
                row["process_rss_gib"] = int(stream.read().split()[1]) * page / GIB
        print("[base-page-reclaim] " + json.dumps(row), flush=True)

    def populate(offset):
        length = min(8 * GIB, size - offset)
        if libc.madvise(start + offset, length, MADV_POPULATE_WRITE) != 0:
            raise OSError(ctypes.get_errno(), "MADV_POPULATE_WRITE failed")
        return length

    try:
        if libc.madvise(start, size, MADV_NOHUGEPAGE) != 0:
            raise OSError(ctypes.get_errno(), "MADV_NOHUGEPAGE failed")
        report("start")
        # Each thread faults disjoint ranges; ctypes releases the GIL. Avoid
        # serial Python loops over hundreds of millions of 4 KiB pages.
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = {
                pool.submit(populate, offset) for offset in range(0, size, 8 * GIB)
            }
            last_time, last_bytes = began, 0
            try:
                while pending:
                    ready, pending = wait(
                        pending, timeout=10, return_when=FIRST_COMPLETED
                    )
                    for future in ready:
                        completed += future.result()
                    if (
                        completed - last_bytes >= 32 * GIB
                        or time.monotonic() - last_time >= 10
                    ):
                        report("progress")
                        last_time, last_bytes = time.monotonic(), completed
            finally:
                for future in pending:
                    future.cancel()
        memory = mapping_memory(start, start + size)
        report("populated", memory)
        if memory["Rss"] < size or memory["AnonHugePages"]:
            raise RuntimeError("scratch pages were not fully resident base pages")
    finally:
        if libc.munmap(start, size) != 0:
            raise OSError(ctypes.get_errno(), "base-page scratch munmap failed")
    print(
        "[base-page-reclaim] "
        + json.dumps(
            {
                "event": "released",
                "released_gib": size / GIB,
                "seconds": round(time.monotonic() - began, 3),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gib", type=non_negative_gib)
    args = parser.parse_args()
    populate_and_release(args.gib)


if __name__ == "__main__":
    main()
