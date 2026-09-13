#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Minimal-dependency safetensors load-speed benchmark across every local disk.

Pure Python **stdlib only** — no torch, no safetensors, no ATOM install. The
safetensors container is just `<u64 header_len><json header><raw tensor bytes>`,
and a load is dominated by *reading those bytes*, so we generate a dummy
safetensors file and time reading it back with the two strategies ATOM's loader
uses: `mmap` (+`posix_fadvise` read-ahead, like
`model_loader/loader.py`) and a plain `read()` (the `ATOM_DISABLE_MMAP` path).

It auto-discovers all real local filesystems (from /proc/mounts), labels each
HDD vs SSD/NVMe (/sys/block/.../queue/rotational), writes a dummy on each, and
reports GB/s cold (page cache evicted -> real disk) and warm (cached -> RAM
ceiling), single- and multi-threaded, plus major page-fault counts (the
ServerlessLLM/OSDI'24 mmap effect). GB/s is the metric fastsafetensors / NVIDIA
Run:ai Model Streamer report.

Run (no deps):  python3 benchmark_disk_loading.py
                python3 benchmark_disk_loading.py --list-disks
                python3 benchmark_disk_loading.py --size-gb 8 --disks / /mnt/sda1
"""

import argparse
import json
import mmap
import os
import re
import shutil
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

GiB = 1024**3
CHUNK = 16 << 20
_PSEUDO = {
    "tmpfs", "devtmpfs", "proc", "sysfs", "cgroup", "cgroup2", "overlay",
    "squashfs", "autofs", "mqueue", "hugetlbfs", "debugfs", "tracefs",
    "securityfs", "pstore", "bpf", "configfs", "fusectl", "nsfs", "ramfs",
    "devpts", "efivarfs", "binfmt_misc", "rpc_pipefs", "fuse.gvfsd-fuse",
}
_NET = {"nfs", "nfs4", "cifs", "smbfs", "ceph", "glusterfs", "9p", "fuse.sshfs"}
_ELT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}


# ------------------------------ disk discovery ------------------------------
def _unescape(s: str) -> str:
    return (s.replace("\\040", " ").replace("\\011", "\t")
             .replace("\\012", "\n").replace("\\134", "\\"))


def _base_block(dev: str) -> str:
    base = os.path.basename(os.path.realpath(dev))
    if base.startswith("nvme"):
        m = re.match(r"(nvme\d+n\d+)", base)
        return m.group(1) if m else base
    return re.sub(r"\d+$", "", base)  # sda1 -> sda


def disk_info(blk: str):
    def _read(p):
        try:
            return open(p).read().strip()
        except OSError:
            return None
    rot = _read(f"/sys/block/{blk}/queue/rotational")
    model = _read(f"/sys/block/{blk}/device/model") or _read(f"/sys/block/{blk}/device/name")
    kind = "HDD" if rot == "1" else ("SSD/NVMe" if rot == "0" else "?")
    return kind, (model or "?")


def writable_dir_on(mnt, dev_id):
    """A writable dir on the *same* filesystem as `mnt` (mountpoints like / and
    /mnt/sda1 are usually root-owned, so fall back to a user subdir / $HOME)."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())
    for c in (mnt, os.path.join(mnt, user), os.path.expanduser("~"), os.getcwd()):
        try:
            if not c or not os.path.isdir(c) or os.stat(c).st_dev != dev_id:
                continue
            t = os.path.join(c, f".lbtest_{os.getpid()}")
            os.mkdir(t)
            os.rmdir(t)
            return c
        except OSError:
            continue
    return None


def discover_disks():
    """One physical block device per row, with a resolved writable subdir."""
    out, seen = [], set()
    with open("/proc/mounts") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            dev, mnt, fstype = _unescape(parts[0]), _unescape(parts[1]), parts[2]
            if fstype in _PSEUDO or fstype in _NET or not dev.startswith("/dev/"):
                continue
            blk = _base_block(dev)
            if blk in seen:
                continue
            try:
                dev_id = os.stat(mnt).st_dev
            except OSError:
                continue
            seen.add(blk)
            kind, model = disk_info(blk)
            out.append({"dev": dev, "mnt": mnt, "fstype": fstype, "blk": blk,
                        "kind": kind, "model": model, "dev_id": dev_id,
                        "wdir": writable_dir_on(mnt, dev_id)})
    return out


# ------------------------- dummy safetensors writer -------------------------
def write_shard(path: str, total_bytes: int, ntensors: int, dtype: str) -> int:
    elt = _ELT[dtype]
    per = max(elt, (total_bytes // ntensors) // elt * elt)
    header, off = {}, 0
    for i in range(ntensors):
        n = per // elt
        cols = 4096 if n >= 4096 else n
        rows = max(1, n // cols)
        nbytes = rows * cols * elt
        header[f"t{i}"] = {"dtype": dtype, "shape": [rows, cols],
                           "data_offsets": [off, off + nbytes]}
        off += nbytes
    hjson = json.dumps(header).encode()
    hjson += b" " * ((-len(hjson)) % 8)          # safetensors 8-byte align
    buf = os.urandom(min(CHUNK, max(1, off)))    # non-sparse, non-zero payload
    with open(path, "wb", buffering=0) as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        w = 0
        while w < off:
            n = min(len(buf), off - w)
            f.write(buf[:n]); w += n
        f.flush(); os.fsync(f.fileno())
    return off  # data bytes


def gen_dummy(d: str, size_gb: float, shards: int, dtype: str):
    os.makedirs(d, exist_ok=True)
    files, total = [], 0
    per = int(size_gb * GiB) // shards
    for s in range(shards):
        p = os.path.join(d, f"model-{s + 1:05d}-of-{shards:05d}.safetensors")
        total += write_shard(p, per, 8, dtype)
        files.append(p)
    return files, total


# --------------------------------- readers ---------------------------------
def _fadvise(fd, sz, advice):
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, sz, advice)
        except OSError:
            pass


def read_mmap(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        sz = os.fstat(fd).st_size
        _fadvise(fd, sz, os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED)
        mm = mmap.mmap(fd, sz, prot=mmap.PROT_READ)
        hlen = struct.unpack("<Q", mm[:8])[0]
        p, end_all, crc = 8 + hlen, sz, 0
        while p < end_all:
            e = min(p + CHUNK, end_all)
            crc = zlib.crc32(mm[p:e], crc)       # slice forces page-in
            p = e
        mm.close()
        return end_all - (8 + hlen)
    finally:
        os.close(fd)


def read_plain(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        sz = os.fstat(fd).st_size
        _fadvise(fd, sz, os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED)
        hlen = struct.unpack("<Q", os.read(fd, 8))[0]
        os.lseek(fd, 8 + hlen, os.SEEK_SET)
        moved, crc = 0, 0
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                break
            crc = zlib.crc32(b, crc); moved += len(b)
        return moved
    finally:
        os.close(fd)


def evict(files):
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        _fadvise(fd, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def _majflt():
    try:
        s = open("/proc/self/stat").read()
        return int(s[s.rindex(")") + 2:].split()[9])
    except Exception:
        return 0


def run_cfg(files, total, reader, threads, cold, runs):
    best_t = None
    mj = 0
    for _ in range(runs):
        if cold:
            evict(files)
        else:
            for p in files:
                read_plain(p)               # prime
        m0 = _majflt()
        t0 = time.perf_counter()
        if threads <= 1:
            for p in files:
                reader(p)
        else:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                list(ex.map(reader, files))
        dt = time.perf_counter() - t0
        mj = _majflt() - m0
        best_t = dt if best_t is None else min(best_t, dt)
    return total / best_t / 1e9, mj


# ---------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-gb", type=float, default=4.0, help="dummy size per disk")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="multi-thread read width")
    ap.add_argument("--runs", type=int, default=1, help="timed runs (min taken)")
    ap.add_argument("--dtype", default="BF16", choices=list(_ELT))
    ap.add_argument("--disks", nargs="*", help="mountpoints to test (default: all)")
    ap.add_argument("--list-disks", action="store_true")
    ap.add_argument("--keep", action="store_true", help="don't delete the dummy")
    args = ap.parse_args()

    disks = discover_disks()
    if args.disks:
        want = {os.path.realpath(d) for d in args.disks}
        disks = [d for d in disks if os.path.realpath(d["mnt"]) in want]

    print("Discovered disks:")
    for d in disks:
        free = shutil.disk_usage(d["wdir"] or d["mnt"]).free / GiB
        wd = d["wdir"] or "(no writable dir)"
        print(f"  {d['mnt']:<14} {d['dev']:<15} {d['kind']:<9} {d['fstype']:<6} "
              f"free={free:6.0f}G  write={wd}  {d['model']}")
    if args.list_disks:
        return
    if not disks:
        raise SystemExit("no writable local disks discovered (try --disks)")
    print(f"\nDummy: {args.size_gb} GiB, {args.shards} shards, {args.dtype}, "
          f"threads={args.threads}, runs={args.runs}\n")

    cols = ("disk", "kind", "mmap c x1", "mmap c xN", "read c xN", "mmap w xN", "majflt")
    print(f"{cols[0]:<16}{cols[1]:<10}" + "".join(f"{c:>11}" for c in cols[2:]))
    print("-" * 92)
    for d in disks:
        if not d["wdir"]:
            print(f"{d['mnt']:<16}{d['kind']:<10}  (skip: no writable dir)")
            continue
        free = shutil.disk_usage(d["wdir"]).free
        if free < args.size_gb * GiB * 1.1:
            print(f"{d['mnt']:<16}{d['kind']:<10}  (skip: <{args.size_gb} GiB free)")
            continue
        tmp = os.path.join(d["wdir"], f".loadbench_{os.getpid()}")
        try:
            files, total = gen_dummy(tmp, args.size_gb, args.shards, args.dtype)
            mm1, _ = run_cfg(files, total, read_mmap, 1, True, args.runs)
            mmN, mj = run_cfg(files, total, read_mmap, args.threads, True, args.runs)
            rdN, _ = run_cfg(files, total, read_plain, args.threads, True, args.runs)
            wmN, _ = run_cfg(files, total, read_mmap, args.threads, False, args.runs)
            print(f"{d['mnt']:<16}{d['kind']:<10}{mm1:>11.2f}{mmN:>11.2f}"
                  f"{rdN:>11.2f}{wmN:>11.2f}{mj:>11,}")
        finally:
            if not args.keep:
                shutil.rmtree(tmp, ignore_errors=True)
    print("\nGB/s.  c=cold (evicted -> disk), w=warm (page cache -> RAM); "
          "x1/xN = 1 vs N threads. mmap=safe_open-style, read=ATOM_DISABLE_MMAP-style.")


if __name__ == "__main__":
    main()
