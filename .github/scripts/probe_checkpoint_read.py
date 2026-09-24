#!/usr/bin/env python3
"""How fast a CI runner pulls a checkpoint off its storage, cold, per read pattern.

Two parts:

* ``sweep``: raw read patterns over disjoint shard subsets -- every pattern
  reads shards no earlier pattern touched, so a storage server's own cache
  cannot hand a later pattern the bytes an earlier one fetched.
* ``load``: the real loader, `openai_server` killed as soon as every rank has
  logged its weight-load phases, once per loader configuration.

Every measurement starts from an evicted page cache, and the residency is
checked with mincore(2) rather than assumed: a DONTNEED the kernel ignored
would otherwise make the pattern look like the page cache's speed.
"""

import argparse
import concurrent.futures
import ctypes
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

GiB = 1024**3
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_PAGE = os.sysconf("SC_PAGE_SIZE")


def log(msg):
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mount_of(path):
    """The /proc/self/mountinfo entry serving `path`: longest mount-point prefix."""
    real = os.path.realpath(path)
    best = None
    with open("/proc/self/mountinfo") as f:
        for line in f:
            left, right = line.rstrip("\n").split(" - ", 1)
            fields = left.split()
            point = fields[4]
            if (real == point or real.startswith(point.rstrip("/") + "/")) and (
                best is None or len(point) > len(best["mount_point"])
            ):
                fstype, source, super_opts = (right.split() + ["", ""])[:3]
                best = {
                    "mount_point": point,
                    "dev": fields[2],
                    "fstype": fstype,
                    "source": source,
                    "mount_opts": fields[5],
                    "super_opts": super_opts,
                }
    return best


def resident_fraction(files):
    """Fraction of the files' pages in the page cache, via mincore(2)."""
    total = resident = 0
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if not size:
                continue
            addr = _libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), "mmap failed")
            try:
                pages = (size + _PAGE - 1) // _PAGE
                vec = (ctypes.c_ubyte * pages)()
                if _libc.mincore(addr, size, vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore failed")
                resident += sum(b & 1 for b in vec)
                total += pages
            finally:
                _libc.munmap(addr, size)
        finally:
            os.close(fd)
    return resident / total if total else 0.0


def evict(files):
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return resident_fraction(files)


def _read_blocks(path, block, start=0, length=None):
    with open(path, "rb", buffering=0) as f:
        f.seek(start)
        left = os.fstat(f.fileno()).st_size - start if length is None else length
        buf = bytearray(block)
        view = memoryview(buf)
        while left > 0:
            n = f.readinto(view[: min(block, left)])
            if not n:
                break
            left -= n


def pattern_whole_per_rank(files, ranks, **_):
    """What CI does today: every rank `f.read()`s every shard, in the same order."""

    def rank_loop():
        for path in files:
            with open(path, "rb") as f:
                f.read()

    threads = [threading.Thread(target=rank_loop) for _ in range(ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def pattern_files(files, threads, block, **_):
    """`threads` readers, each streaming whole shards -- ATOM's prefetcher today."""
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        list(pool.map(_read_blocks, files, [block] * len(files)))


_EXTENT = 256 << 20


def pattern_extents(files, threads, block, **_):
    """`threads` readers pulling 256 MiB extents off one queue, in file order.

    Unlike `files`, the concurrency is not capped by the shard count.
    """
    work = [
        (path, start, min(_EXTENT, size - start))
        for path in files
        for size in [os.path.getsize(path)]
        for start in range(0, size, _EXTENT)
    ]
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        list(
            pool.map(
                _read_blocks,
                [w[0] for w in work],
                [block] * len(work),
                [w[1] for w in work],
                [w[2] for w in work],
            )
        )


PATTERNS = {
    "whole_per_rank": pattern_whole_per_rank,
    "files": pattern_files,
    "extents": pattern_extents,
}

# (pattern, kwargs). Threads are node totals; ATOM's prefetcher runs
# ATOM_LOADER_PREFETCH_THREADS per rank over whole shards, so 8 ranks x 4 is
# today's default. The last row repeats the first single stream: if it comes
# out faster, something between the runs (a server-side cache) is helping.
SWEEP = [
    ("whole_per_rank", {}),
    ("files", {"threads": 4, "block": 16 << 20}),
    ("files", {"threads": 13, "block": 16 << 20}),
    ("extents", {"threads": 1, "block": 16 << 20}),
    ("extents", {"threads": 4, "block": 16 << 20}),
    ("extents", {"threads": 8, "block": 16 << 20}),
    ("extents", {"threads": 16, "block": 16 << 20}),
    ("extents", {"threads": 32, "block": 16 << 20}),
    ("extents", {"threads": 64, "block": 16 << 20}),
    ("extents", {"threads": 32, "block": 1 << 20}),
    ("extents", {"threads": 32, "block": 64 << 20}),
    ("extents", {"threads": 1, "block": 16 << 20}),
]


def run_sweep(files, per_run, ranks):
    results = []
    for i, (name, kwargs) in enumerate(SWEEP):
        subset = files[i * per_run : (i + 1) * per_run]
        if len(subset) < per_run:
            log(f"out of unread shards at {name} {kwargs}; stopping the sweep")
            break
        size = sum(os.path.getsize(p) for p in subset)
        before = evict(subset)
        t0 = time.perf_counter()
        PATTERNS[name](subset, ranks=ranks, **kwargs)
        secs = time.perf_counter() - t0
        row = {
            "pattern": name,
            **{k: (v >> 20 if k == "block" else v) for k, v in kwargs.items()},
            "shards": len(subset),
            "gib": round(size / GiB, 1),
            "seconds": round(secs, 1),
            "gb_per_s": round(size / secs / 1e9, 3),
            "resident_before": round(before, 4),
            "resident_after": round(resident_fraction(subset), 4),
        }
        log(json.dumps(row))
        results.append(row)
        evict(subset)
    return results


_PHASES = re.compile(r"weight load phases.*?read\+queue ([0-9.]+)s.*?drain ([0-9.]+)s")
_LOADED = re.compile(r"weights loaded in ([0-9.]+)s")


def run_load(model, files, label, env_over, server_args, log_dir, max_minutes, tp):
    before = evict(files)
    log_path = os.path.join(log_dir, f"load_{label}.log")
    env = {**os.environ, "AITER_LOG_LEVEL": "WARNING", **env_over}
    cmd = [
        sys.executable,
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        model,
        "--server-port",
        "8123",
        *server_args,
    ]
    log(f"load {label}: env={env_over} resident_before={before:.4f}")
    t0 = time.perf_counter()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(
            cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    phases, loaded, status = [], [], "timeout"
    try:
        while time.perf_counter() - t0 < max_minutes * 60:
            with open(log_path, errors="replace") as f:
                text = f.read()
            phases = [tuple(map(float, m)) for m in _PHASES.findall(text)]
            loaded = [float(m) for m in _LOADED.findall(text)]
            if len(loaded) >= tp:
                status = "loaded"
                break
            if proc.poll() is not None:
                status = f"exited {proc.returncode}"
                break
            time.sleep(5)
    finally:
        wall = time.perf_counter() - t0
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    row = {
        "label": label,
        "env": env_over,
        "status": status,
        "wall_to_loaded_s": round(wall, 1),
        "max_weights_loaded_s": max(loaded, default=None),
        "max_read_queue_s": max((p[0] for p in phases), default=None),
        "max_drain_s": max((p[1] for p in phases), default=None),
        "resident_before": round(before, 4),
    }
    log(json.dumps(row))
    # The next launch needs the GPUs back; the workers are gone with the group.
    time.sleep(20)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--per-run", type=int, default=13)
    ap.add_argument("--out", default="probe_results")
    ap.add_argument("--load-minutes", type=int, default=45)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.model, "*.safetensors")))
    total = sum(os.path.getsize(p) for p in files)
    info = {
        "model": args.model,
        "shards": len(files),
        "gib": round(total / GiB, 1),
        "mount": mount_of(args.model),
        "nproc": os.cpu_count(),
        "meminfo": subprocess.run(
            ["free", "-g"], capture_output=True, text=True, check=False
        ).stdout,
    }
    log(json.dumps(info, indent=1))
    probe_evict = evict(files[:2])
    log(f"eviction self-check: resident after DONTNEED = {probe_evict:.4f}")

    results = {"info": info, "sweep": [], "load": []}
    if not args.skip_sweep:
        results["sweep"] = run_sweep(files, args.per_run, args.tp)

    if not args.skip_load:
        server_args = ["--kv_cache_dtype", "fp8", "-tp", str(args.tp)]
        configs = [
            ("mmap_prefetch4", {"ATOM_DISABLE_MMAP": "false"}),
            ("disable_mmap", {"ATOM_DISABLE_MMAP": "true"}),
            ("mmap_prefetch4_again", {"ATOM_DISABLE_MMAP": "false"}),
        ]
        streams = [r for r in results["sweep"] if r["pattern"] == "extents"]
        if streams:
            best = max(streams, key=lambda r: r["gb_per_s"])
            per_rank = max(1, best["threads"] // args.tp)
            if per_rank != 4:
                configs.append(
                    (
                        f"mmap_prefetch{per_rank}",
                        {
                            "ATOM_DISABLE_MMAP": "false",
                            "ATOM_LOADER_PREFETCH_THREADS": str(per_rank),
                            "ATOM_LOADER_PREFETCH_BLOCK_MB": str(best["block"]),
                        },
                    )
                )
        for label, env_over in configs:
            results["load"].append(
                run_load(
                    args.model,
                    files,
                    label,
                    env_over,
                    server_args,
                    args.out,
                    args.load_minutes,
                    args.tp,
                )
            )

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
