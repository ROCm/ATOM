# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone model-loading (safetensors) speed benchmark for ATOM.

Runs WITHOUT installing ATOM: it only needs ``torch`` + ``safetensors`` (already
present in any ATOM env) and mirrors ATOM's actual read path in
``atom/model_loader/loader.py::safetensors_weights_iterator`` — the default
``safe_open`` mmap path (with ``posix_fadvise`` read-ahead) and the
``ATOM_DISABLE_MMAP`` "read whole file then ``safetensors.torch.load``" path —
plus an optional thread pool (``ATOM_LOADER_USE_THREADPOOL``) for concurrent
shard reads.

Metrics & methodology, following the published safetensors-load benchmarks:
  * **GB/s throughput** is the headline number (fastsafetensors, NVIDIA Run:ai
    Model Streamer) — load time alone is meaningless without model size.
  * **cold vs warm page cache** and **major page-fault counts**
    (ServerlessLLM, OSDI'24): mmap cold-start cost is dominated by page faults
    and the OS cache, so we evict (cold) or prime (warm) deliberately and
    report majflt/minflt deltas.
  * **median over N runs**, and a **CPU vs GPU (host vs host+H2D) split**
    (HuggingFace official speed comparison).

Examples
--------
  # default: sweep mmap×threads on CPU, cold cache, 3 runs, against a local dir
  python atom/benchmarks/benchmark_model_loading.py --model /path/to/weights

  # measure the full storage->GPU path (safe_open device=cuda), warm cache
  python atom/benchmarks/benchmark_model_loading.py --model /weights \
      --device cuda --cache warm

  # true global cold cache (needs sudo) + benchmark ATOM's real iterator
  python atom/benchmarks/benchmark_model_loading.py --model /weights \
      --drop-caches --use-atom-loader

  # generate a dummy model to test load speed without the real checkpoint
  python atom/benchmarks/benchmark_model_loading.py --make-dummy --out /nvme/dummy \
      --size-gb 24 --shards 12 --dtype bf16          # generic
  python atom/benchmarks/benchmark_model_loading.py --make-dummy --like /real/model \
      --out /nvme/dummy_like                          # clone exact layout (incl. MXFP8)

Reference result (24 GiB bf16 dummy, Samsung 9100 PRO Gen5 NVMe, CPU host read,
median of 2; rated seq read ~14 GB/s):
                       GB/s (cold)   GB/s (warm)
    mmap      x1            5.4          11.2
    mmap      x8            6.8          15.9     <- concurrent reads help
    no-mmap   x1            1.9           2.3     <- deserialize-bound
    no-mmap   x8            2.8           3.5
  i.e. the default mmap path reaches ~half the drive's rated BW cold (the OSDI'24
  page-fault effect; majflt ~30-49K), warm hits the RAM ceiling, and the
  ATOM_DISABLE_MMAP path is ~3x slower (read+deserialize). A concurrent/direct-IO
  loader (fastsafetensors / Run:ai-style) is where the remaining ~2x lives.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import safetensors
import safetensors.torch
import torch

# Match loader.py: register the MX scale dtype older safetensors miss, so the
# non-mmap (`safetensors.torch.load(bytes)`) path doesn't KeyError on MXFP8.
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu

GiB = 1024**3


# ----------------------------- shard discovery -----------------------------
def list_shards(path: str, limit: int | None = None) -> list[str]:
    """Local *.safetensors shards, de-duplicated via the index (mirrors ATOM)."""
    if not os.path.isdir(path):
        raise SystemExit(
            f"--model must be a local directory of safetensors shards; got {path!r}. "
            "(This standalone bench does not download; point it at a snapshot dir.)"
        )
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        keep = {os.path.join(path, v) for v in weight_map.values()}
        files = [f for f in files if f in keep]
    if not files:
        raise SystemExit(f"no *.safetensors found in {path!r}")
    return files[:limit] if limit else files


def shard_nbytes(fpath: str) -> int:
    """Total tensor bytes in a shard, read from its safetensors header only."""
    with open(fpath, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    total = 0
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        off = meta["data_offsets"]
        total += off[1] - off[0]
    return total


# ----------------------------- cache control -------------------------------
def _fadvise(fpath: str, advice: int) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(fpath, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, advice)
        finally:
            os.close(fd)
    except OSError:
        pass


def evict(shards: list[str]) -> None:
    """Best-effort per-file cache eviction (no root needed)."""
    for f in shards:
        _fadvise(f, getattr(os, "POSIX_FADV_DONTNEED", 4))


def prime(shards: list[str]) -> None:
    """Pull every shard into the page cache (warm best-case)."""
    for f in shards:
        with open(f, "rb", buffering=0) as fh:
            while fh.read(1 << 24):
                pass


def drop_caches() -> bool:
    """Global cold cache (needs root/sudo). Returns True on success."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True, timeout=60,
        )
        return True
    except Exception:
        return False


def page_faults() -> tuple[int, int]:
    """(minflt, majflt) for this process from /proc/self/stat."""
    try:
        s = open("/proc/self/stat").read()
        after = s[s.rindex(")") + 2:].split()  # fields after comm
        return int(after[7]), int(after[9])    # minflt, majflt
    except Exception:
        return (0, 0)


# ----------------------------- the read paths ------------------------------
def read_shard(fpath: str, device: str, use_mmap: bool, fadvise: bool) -> int:
    """Read+deserialize one shard exactly like ATOM. Returns bytes moved."""
    if use_mmap and fadvise:
        _fadvise(fpath, getattr(os, "POSIX_FADV_SEQUENTIAL", 2)
                 | getattr(os, "POSIX_FADV_WILLNEED", 3))
    moved = 0
    if not use_mmap:
        # ATOM_DISABLE_MMAP path: read whole file, then deserialize.
        with open(fpath, "rb") as fh:
            data = fh.read()
        result = safetensors.torch.load(data)
        for _, t in result.items():
            moved += t.numel() * t.element_size()
            if device != "cpu":
                t.to(device, non_blocking=True)
    else:
        # Default path: safe_open mmap. NOTE: on CPU, get_tensor() can return a
        # zero-copy *view* into the mmap — counting numel alone never touches the
        # data, so we must materialize (clone) to force the actual storage read,
        # mirroring what ATOM's weight_loader does via param.data.copy_(). On a
        # GPU device the host->device copy already forces the read.
        with safetensors.safe_open(fpath, framework="pt", device=device) as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.device.type == "cpu":
                    t = t.clone()
                moved += t.numel() * t.element_size()
    return moved


def run_once(shards: list[str], device: str, use_mmap: bool, fadvise: bool,
             threads: int) -> tuple[float, int, int, int]:
    """One full load. Returns (seconds, bytes, d_minflt, d_majflt)."""
    mn0, mj0 = page_faults()
    t0 = time.perf_counter()
    if threads <= 1:
        total = sum(read_shard(f, device, use_mmap, fadvise) for f in shards)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            total = sum(ex.map(
                lambda f: read_shard(f, device, use_mmap, fadvise), shards))
    if device != "cpu":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mn1, mj1 = page_faults()
    return dt, total, mn1 - mn0, mj1 - mj0


def _torch_dtype(st_str: str):
    """safetensors header dtype string -> torch dtype (covers MXFP8 on this build)."""
    t = safetensors.torch._TYPES.get(st_str)
    if t is not None:
        return t
    fallback = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }
    return fallback.get(st_str, torch.uint8)


def make_dummy(args) -> None:
    """Write a dummy safetensors model (real bytes) so load speed can be measured
    without the real checkpoint.

      --like REAL_DIR : clone the exact name/shape/dtype/shard layout (faithful,
                        incl. MXFP8 F8_E4M3 weights + F8_E8M0 scales).
      otherwise       : N shards totalling ~--size-gb of --dtype.
    """
    from safetensors.torch import save_file

    if not args.out:
        raise SystemExit("--make-dummy requires --out DIR")
    os.makedirs(args.out, exist_ok=True)
    weight_map: dict[str, str] = {}

    if args.like:
        src = list_shards(args.like)
        print(f"cloning layout of {len(src)} shards from {args.like}")

        def _spec(meta):
            """(dtype, shape) that is byte-faithful even if the host torch lacks
            the exact dtype (e.g. F8_E8M0). Keeps the dtype+shape when supported;
            otherwise emits a flat uint8 tensor of identical byte count."""
            shape = meta["shape"]
            numel = 1
            for d in shape:
                numel *= d
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            dt = _torch_dtype(meta["dtype"])
            if numel and torch.empty(0, dtype=dt).element_size() * numel != nbytes:
                return torch.uint8, [nbytes]  # same total bytes, fine for I/O timing
            return dt, shape

        for sf in src:
            with open(sf, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                hdr = json.loads(fh.read(n))
            tensors = {}
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                dt, shape = _spec(meta)
                tensors[name] = torch.empty(shape, dtype=dt)
            fn = os.path.basename(sf)
            save_file(tensors, os.path.join(args.out, fn), metadata={"format": "pt"})
            for name in tensors:
                weight_map[name] = fn
            print(f"  wrote {fn} ({len(tensors)} tensors)")
    else:
        dmap = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if hasattr(torch, "float8_e4m3fn"):
            dmap["fp8_e4m3"] = torch.float8_e4m3fn
        dt = dmap.get(args.dtype, torch.bfloat16)
        elt = torch.empty(0, dtype=dt).element_size()
        total_elts = int(args.size_gb * GiB / elt)
        tps = max(1, args.tensors_per_shard)
        per_tensor = max(1, total_elts // (args.shards * tps))
        cols = min(4096, per_tensor)
        rows = max(1, per_tensor // cols)
        print(f"generating {args.shards} shards x {tps} tensors [{rows},{cols}] "
              f"{args.dtype} ~= {args.size_gb} GiB")
        for s in range(args.shards):
            tensors = {f"dummy.{s}.w{i}": torch.empty((rows, cols), dtype=dt)
                       for i in range(tps)}
            fn = f"model-{s + 1:05d}-of-{args.shards:05d}.safetensors"
            save_file(tensors, os.path.join(args.out, fn), metadata={"format": "pt"})
            for name in tensors:
                weight_map[name] = fn

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"note": "dummy"}, "weight_map": weight_map}, f)
    sz = sum(os.path.getsize(os.path.join(args.out, f))
             for f in os.listdir(args.out) if f.endswith(".safetensors"))
    print(f"dummy model -> {args.out}: {sz / GiB:.2f} GiB, "
          f"{sum(f.endswith('.safetensors') for f in os.listdir(args.out))} shards. "
          f"Now run:  python {os.path.basename(__file__)} --model {args.out}")


def atom_iterator_load(model: str, disable_mmap: bool) -> tuple[float, int]:
    """Benchmark ATOM's *real* iterator (needs the atom package importable)."""
    from atom.model_loader.loader import safetensors_weights_iterator
    t0 = time.perf_counter()
    total = 0
    for _, t in safetensors_weights_iterator(model, disable_mmap=disable_mmap):
        total += t.numel() * t.element_size()
    return time.perf_counter() - t0, total


# --------------------------------- main ------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="local dir of *.safetensors shards to benchmark")
    # --- dummy-weight generation (no real checkpoint needed) ---
    ap.add_argument("--make-dummy", action="store_true",
                    help="generate a dummy safetensors model into --out, then exit")
    ap.add_argument("--out", help="[--make-dummy] output dir for the dummy model")
    ap.add_argument("--like", help="[--make-dummy] clone exact layout from this real model dir")
    ap.add_argument("--size-gb", type=float, default=8.0,
                    help="[--make-dummy] total size when not using --like")
    ap.add_argument("--shards", type=int, default=8, help="[--make-dummy] # shards")
    ap.add_argument("--tensors-per-shard", type=int, default=16,
                    help="[--make-dummy] tensors per shard (per-tensor dispatch overhead)")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32", "fp8_e4m3"],
                    help="[--make-dummy] element dtype when not using --like")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per config (median)")
    ap.add_argument("--device", default="cpu",
                    help="cpu (host read) or cuda[:i] (storage->GPU). default cpu")
    ap.add_argument("--threads", type=int, default=0,
                    help="concurrent shard reads for the multi-thread config "
                         "(0 = min(8, #shards))")
    ap.add_argument("--cache", choices=["cold", "warm"], default="cold",
                    help="cold = evict each shard before each run; warm = prime first")
    ap.add_argument("--drop-caches", action="store_true",
                    help="global cold cache via sudo drop_caches (true cold)")
    ap.add_argument("--no-fadvise", action="store_true",
                    help="disable posix_fadvise read-ahead (ATOM uses it)")
    ap.add_argument("--limit-shards", type=int, default=0, help="smoke test: first N shards")
    ap.add_argument("--use-atom-loader", action="store_true",
                    help="also time ATOM's real safetensors_weights_iterator")
    args = ap.parse_args()

    if args.make_dummy:
        make_dummy(args)
        return
    if not args.model:
        raise SystemExit("--model is required (or use --make-dummy --out DIR)")

    shards = list_shards(args.model, args.limit_shards or None)
    total_bytes = sum(shard_nbytes(f) for f in shards)
    nthreads = args.threads or min(8, len(shards))
    fadvise = not args.no_fadvise

    print(f"model      : {args.model}")
    print(f"shards     : {len(shards)}   size: {total_bytes / GiB:.2f} GiB")
    print(f"device     : {args.device}   cache: {args.cache}"
          f"{' +drop_caches' if args.drop_caches else ''}   fadvise: {fadvise}")
    print(f"runs       : {args.runs} (median reported)   threads(multi): {nthreads}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    print()

    # config matrix: (label, use_mmap, threads)
    configs = [
        ("mmap            x1 ", True, 1),
        (f"mmap            x{nthreads:<2}", True, nthreads),
        ("no-mmap         x1 ", False, 1),
        (f"no-mmap         x{nthreads:<2}", False, nthreads),
    ]

    hdr = f"{'config':<19} {'GB/s':>8} {'time(s)':>9} {'majflt':>10} {'minflt':>11}"
    print(hdr); print("-" * len(hdr))
    for label, use_mmap, threads in configs:
        times, faults = [], []
        for _ in range(args.runs):
            if args.drop_caches:
                drop_caches()
            if args.cache == "cold":
                evict(shards)
            else:
                prime(shards)
            dt, moved, dmn, dmj = run_once(shards, args.device, use_mmap, fadvise, threads)
            times.append(dt); faults.append((dmn, dmj))
        med = statistics.median(times)
        gbps = total_bytes / med / 1e9
        dmn = int(statistics.median(f[0] for f in faults))
        dmj = int(statistics.median(f[1] for f in faults))
        print(f"{label:<19} {gbps:>8.2f} {med:>9.3f} {dmj:>10,} {dmn:>11,}")

    if args.use_atom_loader:
        print("\n-- ATOM real iterator (atom.model_loader.loader) --")
        try:
            for dm in (False, True):
                if args.cache == "cold":
                    evict(shards)
                dt, moved = atom_iterator_load(args.model, disable_mmap=dm)
                tag = "disable_mmap" if dm else "mmap        "
                print(f"atom iterator {tag} : {total_bytes/dt/1e9:6.2f} GB/s  {dt:7.3f}s")
        except Exception as e:
            print(f"  (skipped: could not import/run ATOM loader: {e})")

    print("\nGB/s = model_bytes / median_time. Cold = storage-bound (the real "
          "first-load cost); high majflt on the mmap rows is the OSDI'24 "
          "page-fault effect. Warm shows the OS-cache best case (HF's number).")


if __name__ == "__main__":
    main()
