# SPDX-License-Identifier: MIT
"""Copy into `hipHostMalloc` memory: one `copy_` against N `memmove` threads,
the measurement behind `host_memory.copy_threads`. Warm anonymous source.

    OMP_NUM_THREADS=4 python tools/engram/bench_host_copy.py
"""

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

lib = ctypes.CDLL("libamdhip64.so")
lib.hipHostMalloc.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_uint,
]
lib.hipHostFree.argtypes = [ctypes.c_void_p]


def memmove_threads(dst, src, n, k):
    step = -(-n // (k * 4))
    with ThreadPoolExecutor(k) as pool:
        list(
            pool.map(
                lambda s: ctypes.memmove(dst + s, src + s, min(step, n - s)),
                range(0, n, step),
            )
        )


print("torch threads", torch.get_num_threads(), "cpus", len(os.sched_getaffinity(0)))
for mib in (64, 128, 256, 512, 1024, 4096, 16384):
    n = mib << 20
    src = torch.ones(n, dtype=torch.uint8)
    p = ctypes.c_void_p()
    assert lib.hipHostMalloc(ctypes.byref(p), n, 0) == 0
    dst = torch.frombuffer((ctypes.c_char * n).from_address(p.value), dtype=torch.uint8)
    dst.zero_()
    r = {}
    t = time.perf_counter()
    dst.copy_(src)
    r["copy_"] = time.perf_counter() - t
    for k in (2, 4, 8, 16, 32):
        t = time.perf_counter()
        memmove_threads(p.value, src.data_ptr(), n, k)
        r[f"t{k}"] = time.perf_counter() - t
    print(f"{mib:>6} MiB " + " ".join(f"{k}={1e3 * v:6.1f}ms" for k, v in r.items()))
    del dst
    lib.hipHostFree(p.value)
