#!/usr/bin/env python3
"""Fault GiB of transparent huge pages on this process's NUMA node, then free them.

A Mooncake Store owner registers its segment with the NICs from dozens of
threads at once, and each huge-page fault there runs its own direct
compaction. On a node whose free memory is fragmented -- a page cache just
dropped leaves it so -- those faults race each other and fall back to 4 KiB
pages, and an ionic NIC refuses the MR (pit2-p03-g35: four 192 GiB owners
failed to mount with "Cannot allocate memory", while one process then faulted
768 GiB, all huge pages, in 34 s). Faulting the same amount from one thread
under MADV_HUGEPAGE compacts the node first; freed, that memory stays in
huge-page blocks for the owners and the L1s that pin it next.

Usage, bound to the node: numa_exec.py <node> compact_huge_pages.py <GiB>
Prints how much came as huge pages; exits 0 either way (the owners and ATOM's
L1 check their own pages).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time

GIB = 1 << 30
HUGE_PAGE = 2 << 20
PROT_READ_WRITE = 0x1 | 0x2
MAP_PRIVATE_ANONYMOUS = 0x02 | 0x20
MADV_HUGEPAGE = 14
SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")


def mapping_memory(start: int, end: int, smaps: str = "/proc/self/smaps") -> dict:
    """Read RSS/THP bytes of VMAs overlapping the preparation mapping."""
    totals = {"Rss": 0, "AnonHugePages": 0}
    inside = False
    with open(smaps, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace")
            header = SMAPS_HEADER.match(line)
            if header:
                low, high = int(header.group(1), 16), int(header.group(2), 16)
                inside = low < end and high > start
            elif inside:
                key, _, value = line.partition(":")
                if key in totals:
                    totals[key] += int(value.split()[0]) * 1024
    return totals


def anon_huge_page_bytes(start: int, end: int, smaps: str = "/proc/self/smaps") -> int:
    return mapping_memory(start, end, smaps)["AnonHugePages"]


def report_progress(start, size, touched, began):
    row = {
        "target_gib": size / GIB,
        # One byte is touched per 2 MiB interval. This range is not the RSS:
        # a THP fallback may have allocated only one 4 KiB page in an interval.
        "touched_range_gib": round(touched / GIB, 3),
        "elapsed_seconds": round(time.monotonic() - began, 3),
    }
    try:
        memory = mapping_memory(start, start + size)
        row.update(
            rss_gib=round(memory["Rss"] / GIB, 3),
            anon_huge_gib=round(memory["AnonHugePages"] / GIB, 3),
            base_page_rss_gib=round(
                max(0, memory["Rss"] - memory["AnonHugePages"]) / GIB, 3
            ),
        )
    except OSError as exc:
        row["smaps_error"] = str(exc)
    # Keep progress distinct from the full-THP completion marker.
    print("[thp-progress] " + json.dumps(row), flush=True)


def fault_huge_pages(gib: float, *, progress: bool = False) -> tuple[int, int, float]:
    """Fault and free ``gib`` GiB under MADV_HUGEPAGE.

    Returns:
        (bytes faulted, bytes that were huge pages, seconds).
    """
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
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    size = int(gib * GIB) // HUGE_PAGE * HUGE_PAGE
    mapped = size + HUGE_PAGE
    base = libc.mmap(None, mapped, PROT_READ_WRITE, MAP_PRIVATE_ANONYMOUS, -1, 0)
    if base in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), f"mmap of {gib:g} GiB failed")
    try:
        start = (base + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1)
        if libc.madvise(start, size, MADV_HUGEPAGE) != 0:
            raise OSError(ctypes.get_errno(), "madvise(MADV_HUGEPAGE) failed")
        began = time.monotonic()
        if progress:
            report_progress(start, size, 0, began)
        last_time, last_range = began, 0
        pages = (ctypes.c_char * size).from_address(start)
        for offset in range(0, size, HUGE_PAGE):
            pages[offset] = b"\1"
            touched = offset + HUGE_PAGE
            if progress and (
                touched - last_range >= 32 * GIB or time.monotonic() - last_time >= 10
            ):
                report_progress(start, size, touched, began)
                last_time, last_range = time.monotonic(), touched
        if progress:
            report_progress(start, size, size, began)
        seconds = time.monotonic() - began
        return size, anon_huge_page_bytes(start, start + size), seconds
    finally:
        libc.munmap(base, mapped)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="log every 32 GiB or 10 seconds of first-touch",
    )
    parser.add_argument("gib", type=float)
    args = parser.parse_args(argv)
    size, huge, seconds = fault_huge_pages(args.gib, progress=args.progress)
    share = 100 * huge / size if size else 100.0
    tag = "[thp-compact]" if huge >= size else "[thp-compact][WARN]"
    print(
        f"{tag} {huge / GIB:.0f} of {size / GIB:.0f} GiB came as huge pages "
        f"({share:.1f}%) in {seconds:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
