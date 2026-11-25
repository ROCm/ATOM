#!/usr/bin/env python3
"""evict_cache.py (runs ON a node, host python3; our lease covers the whole node):

  evict_cache.py fadvise [--budget-s S] [--keep DIR] ROOT...
      POSIX_FADV_DONTNEED over every readable regular file >= 1 MiB under ROOT... (clean page cache only; nothing is
      written, deleted or unmapped), skipping --keep (our checkpoint) and .git / __pycache__ / node_modules dirs,
      within a time budget. Prints files / bytes advised and the unreadable count.
  evict_cache.py pressure --gib G [--floor-gib F]
      Run under numa_exec.py <node>: fault G GiB of anonymous 4 KiB pages (MADV_NOHUGEPAGE) bound to that NUMA node,
      1 GiB at a time, so the kernel reclaims that node's clean inactive page cache, then free them. Stops early when
      the node's MemFree + Inactive(file) would fall below F GiB (default 96), so it never pushes into anon reclaim.
Both print the per-NUMA FilePages / MemFree / Inactive(file) before and after."""

import argparse
import ctypes
import mmap
import os
import re
import time

GIB = 1 << 30
MADV_NOHUGEPAGE = 15


def numa_mem():
    out = {}
    for d in sorted(os.listdir("/sys/devices/system/node")):
        m = re.fullmatch(r"node(\d+)", d)
        if not m:
            continue
        vals = {}
        with open(f"/sys/devices/system/node/{d}/meminfo") as f:
            for line in f:
                x = re.match(r"Node \d+ ([\w()]+):\s+(\d+) kB", line)
                if x:
                    vals[x.group(1)] = int(x.group(2)) * 1024
        out[int(m.group(1))] = {k: round(vals.get(k, 0) / GIB, 1) for k in ("MemFree", "FilePages", "Inactive(file)")}
    return out


def fadvise(args):
    keep = [os.path.realpath(k) for k in args.keep]
    t0, files, nbytes, unreadable = time.time(), 0, 0, 0
    skip = {".git", "__pycache__", "node_modules"}
    for root in args.roots:
        stack = [root]
        while stack:
            if time.time() - t0 > args.budget_s:
                print(f"budget of {args.budget_s} s reached")
                stack.clear()
                break
            d = stack.pop()
            if any(os.path.realpath(d).startswith(k) for k in keep):
                continue
            try:
                it = os.scandir(d)
            except OSError:
                unreadable += 1
                continue
            with it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name not in skip:
                                stack.append(e.path)
                        elif e.is_file(follow_symlinks=False) and e.stat(follow_symlinks=False).st_size >= (1 << 20):
                            fd = os.open(e.path, os.O_RDONLY | os.O_NONBLOCK)
                            try:
                                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                                files += 1
                                nbytes += e.stat(follow_symlinks=False).st_size
                            finally:
                                os.close(fd)
                    except OSError:
                        unreadable += 1
    print(f"fadvise: {files} files, {nbytes / GIB:.0f} GiB of file size advised, {unreadable} unreadable entries, "
          f"{time.time() - t0:.0f} s")


def pressure(args):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    node = int(os.environ.get("EVICT_NODE", "0"))
    with open("/proc/self/status") as f:
        m = re.search(r"Mems_allowed_list:\s*(\S+)", f.read())
    maps, t0 = [], time.time()
    for i in range(int(args.gib)):
        mi = numa_mem()[node]
        if mi["MemFree"] + mi["Inactive(file)"] - 1 < args.floor_gib:
            print(f"stop at {i} GiB: node{node} MemFree {mi['MemFree']} + Inactive(file) {mi['Inactive(file)']} GiB "
                  f"near the {args.floor_gib} GiB floor")
            break
        mm = mmap.mmap(-1, GIB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        addr = ctypes.c_void_p.from_buffer(mm)
        libc.madvise(ctypes.c_void_p(ctypes.addressof(addr)), ctypes.c_size_t(GIB), MADV_NOHUGEPAGE)
        del addr
        for off in range(0, GIB, 4096):
            mm[off] = 1
        maps.append(mm)
    n = len(maps)
    for mm in maps:
        mm.close()
    print(f"pressure: faulted and freed {n} GiB on node{node} in {time.time() - t0:.0f} s "
          f"(mems_allowed {m.group(1) if m else '?'})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fadvise")
    f.add_argument("--budget-s", type=float, default=300)
    f.add_argument("--keep", action="append", default=[])
    f.add_argument("roots", nargs="+")
    p = sub.add_parser("pressure")
    p.add_argument("--gib", type=float, required=True)
    p.add_argument("--floor-gib", type=float, default=96)
    args = ap.parse_args()
    print("before:", numa_mem(), flush=True)
    fadvise(args) if args.mode == "fadvise" else pressure(args)
    print("after: ", numa_mem(), flush=True)


if __name__ == "__main__":
    main()
